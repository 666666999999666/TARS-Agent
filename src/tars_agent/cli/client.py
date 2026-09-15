from __future__ import annotations

import asyncio
import json
import signal
import sys
import time
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from tars_agent.cli.input import TerminalInput
from tars_agent.core.config import TarsConfig
from tars_agent.core.transport.message_submission import (
    new_client_message_id,
    submit_message_with_retry,
)
from tars_agent.core.transport.socket_client import IpcError, SocketClient

_TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
_DECISIONS = {"y": "allow_once", "a": "allow_session", "n": "deny_once", "d": "deny_session"}


class TerminalClient:
    """One CLI connection and input loop; all Agent execution remains in Core."""

    def __init__(
        self, config: TarsConfig, *, interactive: bool, reader: TerminalInput | None = None,
    ) -> None:
        self._config = config
        self._interactive = interactive
        self._reader = reader or TerminalInput()
        self._client: SocketClient | None = None
        self._network: asyncio.Task[None] | None = None
        self._input: asyncio.Task[None] | None = None
        self._actions: set[asyncio.Future[Any]] = set()
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=4096)
        self._session_id = ""
        self._active_run_id: str | None = None
        self._goal: str | None = None
        self._submitting = False
        self._cancel_requested = False
        self._snapshot_pending = False
        self._next_poll = 0.0
        self._candidate: dict[str, Any] | None = None
        self._permissions: dict[str, dict[str, Any]] = {}
        self._shown_permission: str | None = None
        self._prompt_shown = False
        self._parents: dict[str, str] = {}
        self._children: dict[str, str] = {}
        self._tool_errors: set[str] = set()
        self._streamed: dict[str, str] = {}
        self._last_cursor = 0
        self._replaying = True
        self._replay_remaining = 0
        self._replay_truncated = False
        self._reconnecting = False
        self._pending_envelopes: list[tuple[SocketClient, dict[str, Any]]] = []
        self._exit_code: int | None = None
        self._unattended_denial = False
        self._closing = False

    def interrupt(self) -> None:
        self._queue.put_nowait(("interrupt", None))

    @staticmethod
    def _notice(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    async def _rpc(
        self, method: str, params: dict[str, Any], *, timeout_s: float = 20.0,
    ) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Core connection is unavailable")
        return await self._client.send_command(method, params, timeout_s=timeout_s)

    async def _open(self) -> None:
        client = SocketClient(self._config.host, self._config.port)
        try:
            await asyncio.wait_for(client.connect(), timeout=10.0)
        except (OSError, TimeoutError) as exc:
            raise RuntimeError(
                f"无法连接 Core ({self._config.host}:{self._config.port})；"
                "请先运行 tars core start"
            ) from exc
        self._client = client

        async def receive(envelope: dict[str, Any]) -> None:
            self._queue.put_nowait(("envelope", (client, envelope)))

        client.on_event_envelope(receive)

        async def read_events() -> None:
            try:
                await client.run_event_loop()
            finally:
                if not self._closing:
                    self._queue.put_nowait(("disconnected", client))

        self._network = asyncio.create_task(read_events(), name="cli-events")

    async def _subscribe(self) -> None:
        result = await self._rpc("event.subscribe", {
            "topics": ["*"], "session_id": self._session_id,
            "after_cursor": self._last_cursor,
        })
        self._replay_remaining = int(result.get("replayed_count", 0))
        self._replay_truncated = bool(result.get("replay_truncated", False))
        self._replaying = self._replay_remaining > 0 or self._replay_truncated

    def _start(self, kind: str, operation: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(operation)
        self._actions.add(task)

        def finished(completed: asyncio.Future[Any]) -> None:
            self._actions.discard(completed)
            if not completed.cancelled():
                value = completed.exception() or completed.result()
                if not self._closing:
                    self._queue.put_nowait((kind, value))

        task.add_done_callback(finished)

    async def _read_input(self) -> None:
        while not self._closing:
            value = await self._reader.read()
            self._queue.put_nowait(("input", (value, self._shown_permission)))
            if value is None:
                return

    def _submit(self, content: str) -> None:
        self._submitting = True
        self._cancel_requested = False
        self._candidate = None
        self._prompt_shown = False
        self._start("submitted", submit_message_with_retry(
            self._rpc, session_id=self._session_id, content=content,
            client_message_id=new_client_message_id(),
        ))

    def _request_cancel(self) -> None:
        if self._active_run_id is None:
            self._cancel_requested = True  # Resolve an in-flight submission first.
            return
        self._cancel_requested = True
        self._notice(f"[cancel: {self._active_run_id}] 已请求取消，等待清理")
        self._start("cancelled", self._rpc(
            "run.cancel", {"run_id": self._active_run_id}, timeout_s=16.0,
        ))

    async def _snapshot(self, run_id: str) -> dict[str, Any]:
        run = await self._rpc("run.get", {"run_id": run_id})
        result: dict[str, Any] = {"run": run}
        if run["status"] in _TERMINAL:
            result["metrics"] = await self._rpc("run.metrics", {"run_id": run_id})
            # This also flushes persisted events. Drain to this cursor before
            # judging tool failures or listing detached children.
            session = await self._rpc("session.get", {"session_id": self._session_id})
            result["cursor"] = int(session.get("latest_cursor", 0))
        return result

    async def _reconnect(self) -> None:
        old = self._client
        self._client = None
        if old is not None:
            await old.close()
        if self._network is not None:
            await asyncio.gather(self._network, return_exceptions=True)
        await self._open()
        await self._subscribe()  # Never repeat a user submission here.

    def _receive(self, source: SocketClient, envelope: dict[str, Any]) -> None:
        if source is not self._client:
            return  # The replacement subscription replays from the saved cursor.
        if self._reconnecting:
            # Core may send replay frames before the subscribe RPC reply. Keep
            # their order until that reply initializes the replay counter.
            self._pending_envelopes.append((source, envelope))
        else:
            self._event(envelope)

    def _event(self, envelope: dict[str, Any]) -> None:
        kind = envelope.get("kind")
        if kind == "overflow":
            self._reconnecting = True
            self._replaying = True
            self._notice(f"[events] 从已接收游标 {self._last_cursor} 继续回放")
            self._start("reconnected", self._reconnect())
            return
        if kind == "compatibility.error":
            raise RuntimeError("event stream protocol/schema is incompatible")
        if kind != "event":
            return
        cursor = int(envelope["cursor"])
        if cursor <= self._last_cursor:
            return
        self._last_cursor = cursor
        event = envelope.get("event", {})
        event_type = event.get("type")
        run_id = str(event.get("run_id", ""))
        if event_type == "step.started":
            self._streamed[run_id] = ""
        elif event_type == "llm.token" and self._goal is None and run_id == self._active_run_id:
            token = str(event.get("token", ""))
            self._streamed[run_id] = self._streamed.get(run_id, "") + token
            print(token, end="", flush=True)
        elif event_type == "tool.call_started":
            self._notice(f"[tool: {run_id}] {event.get('tool_name', '')}")
        elif event_type == "tool.call_failed":
            self._tool_errors.add(run_id)
            self._notice(f"[error: {run_id}] {event.get('tool_name', '')}: "
                         f"{event.get('error_message', 'tool failed')}")
        elif event_type == "permission.requested":
            if event.get("session_id") == self._session_id:
                self._permissions[str(event["request_id"])] = event
        elif event_type in {"permission.granted", "permission.denied"}:
            self._permissions.pop(str(event.get("request_id", "")), None)
        elif event_type == "subagent.started":
            self._parents[run_id] = str(event.get("parent_run_id", ""))
            self._children[run_id] = "running"
        if event_type in {"run.finished", "subagent.finished"}:
            if event_type == "subagent.finished":
                self._children[run_id] = str(event.get("status", "failed"))
            for request_id, request in list(self._permissions.items()):
                if request.get("run_id") == run_id:
                    self._permissions.pop(request_id, None)
            if run_id == self._active_run_id:
                self._next_poll = 0.0
        if self._replay_remaining:
            self._replay_remaining -= 1
            if not self._replay_remaining and not self._replay_truncated:
                self._replaying = False

    def _show_input(self) -> None:
        if self._replaying or self._reconnecting:
            return
        request_id = next(iter(self._permissions), None)
        if request_id is not None:
            if self._shown_permission == request_id:
                return
            request = self._permissions[request_id]
            self._shown_permission = request_id
            self._notice(f"[approval: {request_id}] {request['tool_name']} "
                         f"session={self._session_id}")
            self._notice(json.dumps(request.get("params", {}), ensure_ascii=False, sort_keys=True))
            if not self._interactive:
                self._unattended_denial = True
                self._notice("[error] 当前没有交互终端，拒绝需要人工审批的操作")
                self._approve("deny_once", request)
                if not self._cancel_requested:
                    self._request_cancel()
                return
            if request.get("request_kind") == "host_fallback":
                self._notice(str(request.get("warning", "此操作将在宿主机执行")))
                self._notice("y=allow host once, n=deny > ")
            else:
                self._notice("y=allow once, a=allow same parameters this session, "
                             "n=deny, d=deny session > ")
        else:
            self._shown_permission = None
            if (self._goal is None and not self._active_run_id
                    and not self._submitting and not self._prompt_shown):
                print("> ", end="", file=sys.stderr, flush=True)
                self._prompt_shown = True

    def _approve(self, decision: str, request: dict[str, Any]) -> None:
        request_id = str(request["request_id"])
        self._permissions.pop(request_id, None)
        self._shown_permission = None
        self._start("approved", self._rpc("permission.respond", {
            "request_id": request_id, "session_id": self._session_id, "decision": decision,
        }))

    def _line(self, value: str | None, shown_permission: str | None) -> None:
        if value is None:
            if self._goal is not None and self._cancel_requested:
                return  # Input closure must not replace an explicit Ctrl+C cancellation.
            self._notice(f"[session: {self._session_id}] 输入结束，断开客户端")
            if self._goal is not None:
                self._unattended_denial = True
                if not self._cancel_requested:
                    self._request_cancel()
            else:
                self._exit_code = 0
            return
        content = value.strip()
        if not content:
            self._prompt_shown = False
            return
        if shown_permission != self._shown_permission:
            self._notice("[input] 审批状态已改变，请针对当前提示重新输入")
            return
        if self._shown_permission:
            request = self._permissions.get(self._shown_permission)
            if request is None:
                return
            decisions = ({"y": "allow_host_once", "n": "deny_once"}
                         if request.get("request_kind") == "host_fallback" else _DECISIONS)
            decision = decisions.get(content.lower())
            if (decision is None
                    or decision not in request.get("allowed_decisions", decisions.values())):
                self._notice("[input] 请输入当前列出的审批选项")
                return
            self._approve(decision, request)
        elif self._active_run_id or self._submitting or self._replaying:
            self._notice("[input] 当前任务仍在执行；这条输入未提交，Ctrl+C 可取消")
        elif self._goal is None:
            self._submit(content)

    def _finish(self) -> None:
        snapshot = self._candidate
        if snapshot is None or self._replaying or self._last_cursor < snapshot["cursor"]:
            return
        run = snapshot["run"]
        run_id, status = str(run["run_id"]), str(run["status"])
        descendants = {run_id}
        while True:
            found = {child for child, parent in self._parents.items() if parent in descendants}
            if found <= descendants:
                break
            descendants.update(found)
        tools = snapshot["metrics"]["tools"]
        tool_error = int(tools["total"]) != int(tools["succeeded"])
        tool_error |= bool(descendants & self._tool_errors)
        tool_error |= bool(snapshot["metrics"].get("subagents", {}).get("failed", 0))
        result = (run.get("result") or {}).get("text", "")
        missing_result = status == "succeeded" and not (isinstance(result, str) and result.strip())
        if result and (self._goal is not None
                       or self._streamed.get(run_id, "").strip() != result.strip()):
            print(result, flush=True)
        elif self._streamed.get(run_id):
            print(flush=True)
        reason = f"; {run['reason']}" if run.get("reason") else ""
        self._notice(f"[result: {run_id}] {status}{reason}")
        if tool_error:
            self._notice("[error] 本次有工具失败或无最终返回；"
                         "即使模型进行了补救，一次性命令仍返回非零")
        if missing_result:
            self._notice("[error] Core 没有返回可确认的最终文本")
        for child_id in sorted(descendants - {run_id}):
            if self._children.get(child_id) in {"queued", "running"}:
                self._notice(f"[background: {child_id}] 仍在执行；tars run status {child_id}; "
                             f"tars run cancel {child_id}")
        self._active_run_id = None
        self._candidate = None
        self._prompt_shown = False
        if self._goal is not None:
            if self._unattended_denial:
                self._exit_code = 1
            elif self._cancel_requested or status == "cancelled":
                self._exit_code = 130
            else:
                successful = status == "succeeded" and not tool_error and not missing_result
                self._exit_code = 0 if successful else 1
        self._cancel_requested = False

    async def run(self, goal: str | None = None, resume_session_id: str | None = None) -> int:
        self._goal = goal
        try:
            await self._open()
            if resume_session_id is not None:
                session = await self._rpc("session.resume", {"session_id": resume_session_id})
                self._session_id = str(session["session"]["session_id"])
                active = session.get("active_run")
                self._active_run_id = str(active["run_id"]) if active else None
            else:
                session = await self._rpc("session.create", {
                    "mode": "one_shot" if goal is not None else "chat",
                    "workspace_root": str(Path.cwd().resolve()),
                })
                self._session_id = str(session["session_id"])
            self._notice(f"[session: {self._session_id}]")
            await self._subscribe()
            if self._interactive or goal is None:
                self._input = asyncio.create_task(self._read_input(), name="cli-input")
            if goal is not None:
                self._submit(goal)
            while self._exit_code is None:
                if (self._active_run_id and not self._snapshot_pending and self._candidate is None
                        and not self._reconnecting and time.monotonic() >= self._next_poll):
                    self._snapshot_pending = True
                    self._start("snapshot", self._snapshot(self._active_run_id))
                self._finish()
                if self._exit_code is not None:
                    break
                self._show_input()
                try:
                    kind, value = await asyncio.wait_for(self._queue.get(), timeout=0.25)
                except TimeoutError:
                    continue
                if kind == "envelope":
                    self._receive(*value)
                elif kind == "input":
                    self._line(*value)
                elif kind == "interrupt":
                    if not self._active_run_id and not self._submitting:
                        self._exit_code = 130
                    elif not self._cancel_requested:
                        self._request_cancel()
                elif kind == "submitted":
                    self._submitting = False
                    if isinstance(value, Exception):
                        raise value
                    self._active_run_id = str(value["run_id"])
                    self._notice(f"[run: {self._active_run_id}]")
                    if self._cancel_requested:
                        self._request_cancel()
                elif kind == "snapshot":
                    self._snapshot_pending = False
                    self._next_poll = time.monotonic() + 1.0
                    if isinstance(value, Exception):
                        if not self._reconnecting:
                            raise value
                    elif value["run"]["status"] in _TERMINAL:
                        self._candidate = value
                elif kind == "cancelled":
                    if isinstance(value, (TimeoutError, IpcError)):
                        self._notice(f"[cancel] 取消尚未确认：{value}")
                    elif isinstance(value, Exception):
                        raise value
                    else:
                        self._notice(f"[cancel] 当前状态：{value.get('status', 'unknown')}")
                    self._next_poll = 0.0
                    if goal is not None:
                        self._exit_code = 1 if self._unattended_denial else 130
                elif kind == "approved":
                    if isinstance(value, Exception):
                        raise value
                    if not value.get("ok", False):
                        self._notice("[approval] 请求已失效或已被处理，本次没有授予权限")
                elif kind == "reconnected":
                    if isinstance(value, Exception):
                        raise value
                    self._reconnecting = False
                    pending, self._pending_envelopes = self._pending_envelopes, []
                    for source, envelope in pending:
                        self._receive(source, envelope)
                    if (not self._reconnecting and self._network is not None
                            and self._network.done()):
                        raise RuntimeError("Core connection closed during event replay")
                elif kind == "disconnected" and value is self._client and not self._reconnecting:
                    raise RuntimeError("Core connection closed; task completion is not confirmed")
            return self._exit_code
        except asyncio.CancelledError:
            return 130
        except (Exception, SystemExit) as exc:
            self._notice(f"[error] {exc}")
            if self._session_id:
                self._notice(f"[session: {self._session_id}] 未自动重新提交任务")
            if self._active_run_id:
                self._notice(f"tars run status {self._active_run_id}")
            return 1
        finally:
            self._closing = True
            self._reader.close()
            if self._input is not None:
                self._input.cancel()
            for task in tuple(self._actions):
                task.cancel()
            if self._client is not None:
                await self._client.close()
            await asyncio.gather(
                *tuple(self._actions), *([self._input] if self._input else []),
                *([self._network] if self._network else []), return_exceptions=True,
            )


def run_client(
    config: TarsConfig, *, goal: str | None = None, resume_session_id: str | None = None,
) -> int:
    async def execute() -> int:
        client = TerminalClient(config, interactive=sys.stdin.isatty())
        loop = asyncio.get_running_loop()
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(client.interrupt))
        try:
            return await client.run(goal=goal, resume_session_id=resume_session_id)
        finally:
            signal.signal(signal.SIGINT, previous)

    try:
        return asyncio.run(execute())
    except KeyboardInterrupt:
        return 130
