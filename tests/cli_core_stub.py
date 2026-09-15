"""Small local Wire Protocol peer for CLI tests; never starts Core or a model."""
from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tars_agent.core.config import TarsConfig

SubmitHandler = Callable[["CliCoreStub", str], Awaitable[None]]
PermissionHandler = Callable[["CliCoreStub", dict[str, Any]], Awaitable[None]]


class CliCoreStub:
    def __init__(self, on_submit: SubmitHandler | None = None) -> None:
        self.on_submit = on_submit
        self.on_cancel: SubmitHandler | None = None
        self.on_permission: PermissionHandler | None = None
        self.on_subscribe: Callable[[CliCoreStub, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.runs: dict[str, dict[str, Any]] = {}
        self.tool_failures: dict[str, int] = {}
        self.tool_successes: dict[str, int] = {}
        self.permissions: dict[str, dict[str, Any]] = {}
        self.approval_answers: list[dict[str, Any]] = []
        self.replay: list[dict[str, Any]] = []
        self.session_id = "sess-cli"
        self.mode = "chat"
        self.workspace = str(Path.cwd())
        self.cursor = 0
        self.main_run_id: str | None = None
        self.changed = asyncio.Event()
        self.writers: list[asyncio.StreamWriter] = []
        self.handlers: set[asyncio.Task[Any]] = set()
        self.disconnected = asyncio.Event()

    async def __aenter__(self) -> CliCoreStub:
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *args: object) -> None:
        self.server.close()
        await self.server.wait_closed()
        for writer in self.writers:
            writer.close()
        handlers = tuple(self.handlers)
        for handler in handlers:
            handler.cancel()
        if handlers:
            await asyncio.gather(*handlers, return_exceptions=True)
        for writer in self.writers:
            with contextlib.suppress(OSError, TimeoutError):
                await asyncio.wait_for(writer.wait_closed(), timeout=1)

    def config(self) -> TarsConfig:
        config = TarsConfig()
        config.host = "127.0.0.1"
        config.port = self.port
        return config

    async def wait_command(self, method: str, count: int = 1) -> None:
        async with asyncio.timeout(5):
            while True:
                self.changed.clear()
                if sum(name == method for name, _ in self.calls) >= count:
                    return
                await self.changed.wait()

    async def emit(self, event: dict[str, Any], *, cursor: int | None = None) -> None:
        cursor = self.cursor + 1 if cursor is None else cursor
        self.cursor = max(self.cursor, cursor)
        event = {"ts": datetime.now(UTC).isoformat(), **event}
        await self._write(self.writers[-1], {
            "kind": "event", "protocol_version": 2, "event_schema_version": 1,
            "cursor": cursor, "session_id": self.session_id,
            "run_id": event.get("run_id"), "occurred_at": event["ts"], "event": event,
        })

    async def overflow(self, *, reason: str, last_cursor: int) -> None:
        writer = self.writers[-1]
        await self._write(writer, {"kind": "overflow", "protocol_version": 2,
                                   "reason": reason, "last_cursor": last_cursor})
        writer.close()

    async def permission(self, run_id: str, request_id: str, *, host: bool = False) -> None:
        event = {
            "type": "permission.requested", "run_id": run_id,
            "request_id": request_id, "tool_use_id": "tool-" + request_id,
            "tool_name": "write_file", "params": {"path": request_id + ".txt", "content": "test"},
            "param_preview": request_id + ".txt", "session_id": self.session_id,
            "request_kind": "host_fallback" if host else "tool",
            "backend": "host" if host else "workspace_sandbox", "risk": "high",
            "reason": "approval_required", "allowed_decisions":
                ["allow_host_once", "deny_once"] if host else
                ["allow_once", "allow_session", "deny_once", "deny_session"],
        }
        self.permissions[request_id] = event
        await self.emit(event)

    async def finish(self, run_id: str, *, status: str = "succeeded", text: str = "wire-final") -> None:
        self.runs[run_id].update(status=status, result={"text": text, "steps": 1})
        if self.runs[run_id].get("parent_run_id"):
            await self.emit({"type": "subagent.finished", "run_id": run_id,
                             "parent_run_id": self.runs[run_id]["parent_run_id"], "status": "success" if status == "succeeded" else status})
        else:
            await self.emit({"type": "run.finished", "run_id": run_id,
                             "status": "success" if status == "succeeded" else status,
                             "reason": None if status == "succeeded" else status, "steps": 1})
            await self.emit({"type": "session.closed" if self.mode == "one_shot" else "session.waiting_for_input",
                             "session_id": self.session_id, "last_run_id": run_id})

    async def child(self, parent_run_id: str, child_run_id: str) -> None:
        self.runs[child_run_id] = self._run_info(child_run_id, parent_run_id=parent_run_id)
        await self.emit({"type": "subagent.started", "run_id": child_run_id,
                         "parent_run_id": parent_run_id, "description": "test child"})

    def _run_info(self, run_id: str, *, parent_run_id: str | None = None) -> dict[str, Any]:
        return {"run_id": run_id, "session_id": self.session_id, "turn_id": "turn-" + run_id,
                "parent_run_id": parent_run_id, "retry_of_run_id": None,
                "kind": "subagent" if parent_run_id else "chat", "attempt": 1,
                "status": "running", "reason": None, "side_effects_started": False,
                "result": None, "created_at": "2026-09-15T00:00:00+00:00",
                "started_at": "2026-09-15T00:00:00+00:00", "finished_at": None}

    def _session(self) -> dict[str, Any]:
        run = self.runs.get(self.main_run_id or "")
        active = run is not None and run["status"] in {"queued", "running"}
        return {"session_id": self.session_id, "mode": self.mode,
                "status": "running" if active else "closed" if self.mode == "one_shot" and run else "ready", "workspace_root": self.workspace,
                "active_run_id": self.main_run_id if active else None, "title": "CLI test",
                "created_at": "2026-09-15T00:00:00+00:00", "updated_at": "2026-09-15T00:00:00+00:00",
                "closed_at": None}

    async def _write(self, writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
        writer.write(json.dumps(message).encode() + b"\n")
        await writer.drain()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.handlers.add(task)
        self.writers.append(writer)
        try:
            while line := await reader.readline():
                request = json.loads(line)
                method, params = request["method"], request.get("params", {})
                self.calls.append((method, params))
                self.changed.set()
                result = await self._respond(method, params)
                if not writer.is_closing():
                    await self._write(writer, {"jsonrpc": "2.0", "id": request["id"], "result": result})
        except (ConnectionError, OSError):
            pass
        finally:
            self.disconnected.set()
            writer.close()
            self.handlers.discard(task)

    async def _respond(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "session.create":
            self.mode = params["mode"]
            self.workspace = params["workspace_root"]
            return {"session_id": self.session_id, "status": "ready", "workspace_root": self.workspace}
        if method in {"session.resume", "session.get"}:
            last = self.runs.get(self.main_run_id or "")
            return {"session": self._session(), "latest_run": last,
                    "active_run": last if last and last["status"] == "running" else None,
                    "latest_cursor": self.cursor}
        if method == "session.get_history":
            return {"messages": []}
        if method == "event.subscribe":
            if self.on_subscribe is not None:
                return await self.on_subscribe(self, params)
            for event in self.replay:
                await self.emit(event)
            return {"subscription_id": "sub-cli", "replayed_count": len(self.replay),
                    "high_water_cursor": self.cursor, "replay_truncated": False}
        if method == "session.send_message":
            run_id = f"run-{1 + sum(row['parent_run_id'] is None for row in self.runs.values())}"
            self.main_run_id = run_id
            self.runs[run_id] = self._run_info(run_id)
            if self.on_submit is None:
                await self.finish(run_id)
            else:
                await self.on_submit(self, run_id)
            return {"run_id": run_id, "status": self.runs[run_id]["status"], "deduplicated": False}
        if method == "run.get":
            return dict(self.runs[params["run_id"]])
        if method == "run.metrics":
            run_id = params["run_id"]
            failed = self.tool_failures.get(run_id, 0)
            succeeded = self.tool_successes.get(run_id, 0)
            children = [row for row in self.runs.values() if row["parent_run_id"] == run_id]
            return {"run_id": run_id, "status": self.runs[run_id]["status"],
                    "tools": {"total": failed + succeeded, "succeeded": succeeded, "failed": failed, "active": 0},
                    "subagents": {"active": sum(row['status'] == 'running' for row in children),
                                  "failed": sum(row['status'] == 'failed' for row in children)}}
        if method == "permission.respond":
            pending = self.permissions.pop(params["request_id"], None)
            if pending is None:
                return {"ok": False}
            self.approval_answers.append(dict(params))
            await self.emit({"type": "permission.granted" if params["decision"].startswith("allow") else "permission.denied",
                             "run_id": pending["run_id"], "request_id": pending["request_id"],
                             "tool_use_id": pending["tool_use_id"], "decision": params["decision"]})
            if self.on_permission is not None:
                await self.on_permission(self, params)
            return {"ok": True}
        if method == "run.cancel":
            if self.on_cancel is not None:
                await self.on_cancel(self, params["run_id"])
            await self.finish(params["run_id"], status="cancelled", text="")
            return dict(self.runs[params["run_id"]])
        raise AssertionError(f"unexpected CLI RPC: {method}")
