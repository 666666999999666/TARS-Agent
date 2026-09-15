from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from tars_agent.core.bus.envelope import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    HandlerError,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcSuccess,
    make_error,
)
from tars_agent.core.trace.record import TraceRecord
from tars_agent.core.trace.writer import TraceWriter

logger = logging.getLogger(__name__)

type CommandHandler = Callable[[dict[str, Any]], Awaitable[Any]]
type AfterResponseHandler = Callable[[], None]

_MAX_LINE_BYTES = 64 * 1024 * 1024
_OUTGOING_QUEUE_CAPACITY = 1_024
_STOP = object()


@dataclass(slots=True)
class _Outgoing:
    message: BaseModel
    sent: asyncio.Future[None] | None


class ConnectionSender:
    """单连接唯一写协程，串行化 RPC 响应与事件推送。"""

    def __init__(
        self,
        writer: asyncio.StreamWriter,
        *,
        queue_capacity: int = _OUTGOING_QUEUE_CAPACITY,
    ) -> None:
        self._writer = writer
        self._queue: asyncio.Queue[_Outgoing | object] = asyncio.Queue(
            maxsize=queue_capacity
        )
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def client_id(self) -> str:
        return str(self._writer.get_extra_info("peername", "<unknown>"))

    @property
    def writer(self) -> asyncio.StreamWriter:
        return self._writer

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._write_loop(), name="ipc-single-writer")

    async def send(self, message: BaseModel, *, wait: bool = False) -> None:
        if self._closed:
            raise ConnectionResetError("connection sender is closed")
        sent = asyncio.get_running_loop().create_future() if wait else None
        await self._queue.put(_Outgoing(message, sent))
        if sent is not None:
            await sent

    async def close(self) -> None:
        if self._closed:
            task = self._task
            if task is not None and task is not asyncio.current_task():
                await asyncio.gather(task, return_exceptions=True)
            return
        self._closed = True
        task = self._task
        if task is not None:
            try:
                await asyncio.wait_for(self._queue.put(_STOP), timeout=1.0)
                if task is not asyncio.current_task():
                    await asyncio.wait_for(
                        asyncio.gather(task, return_exceptions=True),
                        timeout=1.0,
                    )
            except TimeoutError:
                task.cancel()
                if task is not asyncio.current_task():
                    await asyncio.gather(task, return_exceptions=True)
        try:
            self._writer.close()
            await asyncio.wait_for(self._writer.wait_closed(), timeout=1.0)
        except (TimeoutError, ConnectionError, OSError):
            pass

    async def _write_loop(self) -> None:
        failure: BaseException | None = None
        try:
            while True:
                item = await self._queue.get()
                if item is _STOP:
                    self._queue.task_done()
                    return
                assert isinstance(item, _Outgoing)
                try:
                    self._writer.write(item.message.model_dump_json().encode() + b"\n")
                    await self._writer.drain()
                except BaseException as exc:
                    failure = exc
                    if item.sent is not None and not item.sent.done():
                        item.sent.set_exception(exc)
                    raise
                else:
                    if item.sent is not None and not item.sent.done():
                        item.sent.set_result(None)
                finally:
                    self._queue.task_done()
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("single writer stopped after connection failure")
        finally:
            error = failure or ConnectionResetError("connection writer stopped")
            while True:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(item, _Outgoing) and item.sent is not None and not item.sent.done():
                    item.sent.set_exception(error)
                self._queue.task_done()


class ConnectionBroadcaster(Protocol):
    async def unsubscribe(self, sender: ConnectionSender) -> None: ...


_sender_var: ContextVar[ConnectionSender] = ContextVar("_sender_var")


def get_connection_sender() -> ConnectionSender:
    return _sender_var.get()


def get_connection_writer() -> asyncio.StreamWriter:
    """兼容旧扩展；新代码必须通过 ConnectionSender 的单写队列发送。"""

    return get_connection_sender().writer


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validation_details(error: ValidationError) -> list[dict[str, Any]]:
    # Pydantic's rendered exception includes raw inputs, including control tokens.
    return [{"loc": item["loc"], "type": item["type"]}
            for item in error.errors(include_input=False, include_context=False)]


class SocketServer:
    def __init__(
        self,
        host: str,
        port: int,
        broadcaster: ConnectionBroadcaster | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._handlers: dict[str, CommandHandler] = {}
        self._after_response: dict[str, AfterResponseHandler] = {}
        self._server: asyncio.AbstractServer | None = None
        self._broadcaster = broadcaster
        self._trace = trace
        self._active_senders: set[ConnectionSender] = set()

    def register(
        self,
        method: str,
        handler: CommandHandler,
        *,
        after_response: AfterResponseHandler | None = None,
    ) -> None:
        self._handlers[method] = handler
        if after_response is not None:
            self._after_response[method] = after_response

    async def start(self) -> str:
        try:
            _reader, writer = await asyncio.open_connection(self._host, self._port)
            writer.close()
            await writer.wait_closed()
            raise SystemExit(f"core already running at {self._host}:{self._port}")
        except (ConnectionRefusedError, OSError):
            pass

        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self._host,
            port=self._port,
            limit=_MAX_LINE_BYTES,
        )
        return f"{self._host}:{self._port}"

    async def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.close()
        try:
            await asyncio.wait_for(server.wait_closed(), timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            pass
        await asyncio.gather(
            *(sender.close() for sender in tuple(self._active_senders)),
            return_exceptions=True,
        )
        self._server = None

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", "<unknown>")
        logger.debug("client connected: %s", peer)
        sender = ConnectionSender(writer)
        sender.start()
        self._active_senders.add(sender)
        request_tasks: set[asyncio.Task[None]] = set()
        try:
            await self._read_loop(reader, sender, request_tasks)
        finally:
            self._active_senders.discard(sender)
            if self._broadcaster is not None:
                await self._broadcaster.unsubscribe(sender)
            for task in request_tasks:
                task.cancel()
            if request_tasks:
                await asyncio.gather(*request_tasks, return_exceptions=True)
            await sender.close()
            logger.debug("client disconnected: %s", peer)

    async def _read_loop(
        self,
        reader: asyncio.StreamReader,
        sender: ConnectionSender,
        request_tasks: set[asyncio.Task[None]],
    ) -> None:
        while True:
            try:
                line = await reader.readline()
            except asyncio.LimitOverrunError:
                await self._send(sender, make_error(None, INVALID_REQUEST, "Request too large"))
                return
            except (ConnectionResetError, OSError):
                return
            if not line:
                return

            task = asyncio.create_task(self._handle_line(line, sender), name="ipc-request")
            request_tasks.add(task)
            task.add_done_callback(request_tasks.discard)

    async def _handle_line(self, line: bytes, sender: ConnectionSender) -> None:
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            await self._send(sender, make_error(None, PARSE_ERROR, f"Parse error: {exc}"))
            return

        try:
            request = JsonRpcRequest.model_validate(raw)
        except ValidationError as exc:
            await self._send(
                sender,
                make_error(None, INVALID_REQUEST, "Invalid Request", _validation_details(exc)),
            )
            return

        if self._trace is not None:
            self._trace.emit(
                TraceRecord(
                    ts=_now(),
                    direction="CLIENT→CORE",
                    layer="ipc",
                    kind="command",
                    client_id=sender.client_id,
                    data={
                        "method": request.method,
                        "id": request.id,
                        "params": request.params,
                    },
                )
            )

        handler = self._handlers.get(request.method)
        if handler is None:
            await self._send(
                sender,
                make_error(
                    request.id,
                    METHOD_NOT_FOUND,
                    f"Method not found: {request.method}",
                ),
            )
            return

        token = _sender_var.set(sender)
        try:
            result = await handler(request.params)
        except HandlerError as exc:
            await self._send(sender, make_error(request.id, exc.code, str(exc), exc.data))
            return
        except ValidationError as exc:
            await self._send(
                sender,
                make_error(request.id, INVALID_REQUEST, "Invalid params", _validation_details(exc)),
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("handler %s raised: %s", request.method, exc)
            await self._send(sender, make_error(request.id, INTERNAL_ERROR, "Internal error"))
            return
        finally:
            _sender_var.reset(token)

        result_data: Any = result.model_dump() if isinstance(result, BaseModel) else result
        try:
            await self._send(
                sender,
                JsonRpcSuccess(id=request.id, result=result_data),
            )
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("client disconnected before response for %s", request.method)
        else:
            after_response = self._after_response.get(request.method)
            if after_response is not None:
                after_response()

    async def _send(self, sender: ConnectionSender, message: BaseModel) -> None:
        await sender.send(message, wait=True)
        if self._trace is not None:
            kind = "error" if isinstance(message, JsonRpcError) else "response"
            self._trace.emit(
                TraceRecord(
                    ts=_now(),
                    direction="CORE→CLIENT",
                    layer="ipc",
                    kind=kind,
                    client_id=sender.client_id,
                    data=message.model_dump(),
                )
            )
