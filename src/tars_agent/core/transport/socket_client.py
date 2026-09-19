from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from tars_agent.core.bus.envelope import JsonRpcRequest

type EventHandler = Callable[[dict[str, Any]], Awaitable[None]]
type EventEnvelopeHandler = Callable[[dict[str, Any]], Awaitable[None]]

_MAX_LINE_BYTES = 64 * 1024 * 1024  # 64 MB per frame，兼容 MCP 大文件工具结果


class IpcError(RuntimeError):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.data = data


class IpcDisconnectedError(IpcError):
    def __init__(self, message: str = "core connection closed") -> None:
        super().__init__(-32098, message)


class SocketClient:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._event_handlers: list[EventHandler] = []
        self._envelope_handlers: list[EventEnvelopeHandler] = []
        self._write_lock = asyncio.Lock()
        self._closed = False
        self._last_cursor = 0

    # 建立到 core 守护进程的 TCP 连接
    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_connection(
            self._host, self._port, limit=_MAX_LINE_BYTES
        )
        self._closed = False
        self._last_cursor = 0

    @property
    def last_cursor(self) -> int:
        return self._last_cursor

    # 关闭 TCP 连接并等待底层 socket 释放
    async def close(self) -> None:
        self._closed = True
        self._fail_pending(IpcDisconnectedError("client connection closed"))
        writer = self._writer
        if writer is not None:
            writer.close()
            # Closing has begun; repeat close must not await a cancelled close waiter.
            self._writer = None
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
            except (ConnectionResetError, BrokenPipeError):
                # The read loop may already have handled this peer disconnect.
                pass
            except TimeoutError:
                writer.transport.abort()
            except asyncio.CancelledError:
                writer.transport.abort()
                raise

    # 注册服务器推送事件的回调，可多次调用以添加多个 handler
    def on_event(self, handler: EventHandler) -> None:
        self._event_handlers.append(handler)

    # 注册完整事件 envelope 回调，供 cursor/Schema 感知客户端使用
    def on_event_envelope(self, handler: EventEnvelopeHandler) -> None:
        self._envelope_handlers.append(handler)

    # 发送 JSON-RPC 命令并等待响应，成功返回 result dict，失败抛出 IpcError
    async def send_command(
        self, method: str, params: dict[str, Any], *, timeout_s: float | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise IpcDisconnectedError()
        if self._writer is None:
            raise RuntimeError("not connected — call connect() first")
        req_id = str(uuid.uuid4())
        request = JsonRpcRequest(id=req_id, method=method, params=params)
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            async with self._write_lock:
                if self._closed or self._writer is None:
                    raise IpcDisconnectedError()
                self._writer.write(request.model_dump_json().encode() + b"\n")
                await self._writer.drain()
            if timeout_s is None:
                return await fut
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except BaseException:
            self._pending.pop(req_id, None)
            if not fut.done():
                fut.cancel()
            elif not fut.cancelled():
                # close() may fail this waiter while sending still waits on the lock/drain.
                fut.exception()
            raise

    # 持续读取服务器消息，分发 RPC 响应到 pending future 或事件到 event handler
    async def run_event_loop(self) -> None:
        if self._reader is None:
            raise RuntimeError("not connected — call connect() first")
        try:
            while True:
                try:
                    line = await self._reader.readline()
                except (ConnectionResetError, OSError):
                    break
                except (ValueError, asyncio.LimitOverrunError):
                    # 单行超出 limit；丢弃本行，继续读取后续消息
                    continue
                if not line:
                    break
                await self._dispatch(line)
        finally:
            self._closed = True
            self._fail_pending(IpcDisconnectedError())

    # 解析单行消息并路由到 pending future（RPC 响应）或 event handler（服务器推送）
    async def _dispatch(self, line: bytes) -> None:
        try:
            msg: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            return

        if "jsonrpc" in msg:
            req_id: str | None = msg.get("id")
            if req_id and req_id in self._pending:
                fut = self._pending.pop(req_id)
                if not fut.done():
                    if "error" in msg:
                        err = msg["error"]
                        fut.set_exception(
                            IpcError(
                                err.get("code", -1), err.get("message", "unknown"), err.get("data"),
                            )
                        )
                    else:
                        fut.set_result(msg.get("result") or {})
        elif msg.get("kind") == "event":
            if msg.get("protocol_version") != 2 or msg.get("event_schema_version") != 1:
                cursor = msg.get("cursor")
                if isinstance(cursor, int) and cursor > self._last_cursor:
                    self._last_cursor = cursor
                compatibility_error = {
                    "kind": "compatibility.error",
                    "protocol_version": 2,
                    "cursor": cursor,
                    "received_protocol_version": msg.get("protocol_version"),
                    "received_event_schema_version": msg.get("event_schema_version"),
                    "reason": "unsupported event envelope version",
                }
                for handler in self._envelope_handlers:
                    await handler(compatibility_error)
                return
            cursor = msg.get("cursor")
            if not isinstance(cursor, int) or cursor <= self._last_cursor:
                return
            self._last_cursor = cursor
            for handler in self._envelope_handlers:
                await handler(msg)
            event_data: dict[str, Any] = msg.get("event", {})
            for handler in self._event_handlers:
                await handler(event_data)
        elif msg.get("kind") in {"overflow", "compatibility.error"}:
            for handler in self._envelope_handlers:
                await handler(msg)

    def _fail_pending(self, error: IpcDisconnectedError) -> None:
        pending = tuple(self._pending.values())
        self._pending.clear()
        for future in pending:
            if not future.done():
                future.set_exception(error)
