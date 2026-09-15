from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tars_agent.core.bus.envelope import HandlerError
from tars_agent.core.bus.events import RunFinishedEvent
from tars_agent.core.context import ExecutionContext
from tars_agent.core.events.bus import EventBus
from tars_agent.core.persistence import Database
from tars_agent.core.runner import RunOutcome
from tars_agent.core.runtime import service
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.subagent.registry import BackgroundTaskRegistry


class SlowCancellationRunner:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_count = 0

    async def run_and_capture(self, goal: str, **kwargs: Any) -> RunOutcome:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancel_count += 1
            self.cancelled.set()
            await self.release.wait()
        return RunOutcome(status="cancelled", result="", reason="cancelled")


async def make_runtime(tmp_path: Path, runner: Any) -> tuple[Database, RuntimeService]:
    db = Database(tmp_path / "state.db")
    await db.create_schema()
    runtime = RuntimeService(db, lambda: runner, EventBus(), artifacts_root=tmp_path / "artifacts")
    return db, runtime


async def test_timeout_keeps_one_cleanup_alive_after_rpc_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 2.0)
    runner = SlowCancellationRunner()
    db, runtime = await make_runtime(tmp_path, runner)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "wait")
        await runner.started.wait()
        waiter = asyncio.create_task(runtime.cancel_run(submitted.run_id))
        await runner.cancelled.wait()
        waiter.cancel()  # Models the RPC handler disappearing on disconnect.
        with pytest.raises(asyncio.CancelledError):
            await waiter
        monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 0.03)
        with pytest.raises(HandlerError) as caught:
            await runtime.cancel_run(submitted.run_id)
        assert caught.value.code == -32033
        assert caught.value.data == {
            "run_id": submitted.run_id, "cancellation_requested": True,
            "pending_run_ids": [submitted.run_id],
        }
        assert runner.cancel_count == 1
        assert (await runtime.get_run(submitted.run_id)).status == "running"
        runner.release.set()
        monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 2.0)
        assert (await runtime.cancel_run(submitted.run_id)).status == "cancelled"
        assert runner.cancel_count == 1
        assert (await runtime.get_session(session.id)).status == "ready"
    finally:
        runner.release.set()
        await runtime.shutdown()
        await db.dispose()


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "cancelled", "interrupted"])
async def test_late_completion_never_rewrites_terminal_or_publishes_twice(
    tmp_path: Path, terminal: str,
) -> None:
    runner = SlowCancellationRunner()
    db, runtime = await make_runtime(tmp_path, runner)
    seen: list[RunFinishedEvent] = []
    async def observe(event: Any) -> None:
        if isinstance(event, RunFinishedEvent):
            seen.append(event)
    runtime._bus.subscribe(observe)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        # Create queued state without starting an executor.
        run = await runtime._create_message_run(session.id, "test", client_message_id=None, run_id=None)
        await runtime._finalize_without_outcome(run.run_id, terminal, "first")
        await runtime._finish_run(run.run_id, RunOutcome(
            status="success", result="must not commit", reason=None,
            messages=[{"role": "assistant", "content": "must not commit"}],
        ))
        assert (await runtime.get_run(run.run_id)).status == terminal
        assert (await runtime.get_run(run.run_id)).reason == "first"
        assert await runtime.get_history(session.id) == []
        assert len(seen) == 1
    finally:
        await runtime.shutdown()
        await db.dispose()


async def test_cancel_completed_parent_cleans_grandchild_preserves_unrelated_task(tmp_path: Path) -> None:
    runner = SlowCancellationRunner()
    db, runtime = await make_runtime(tmp_path, runner)
    registry = BackgroundTaskRegistry(db, runtime._bus)
    runtime._subagent_registry = registry
    child_started = asyncio.Event()
    grandchild_started = asyncio.Event()
    unrelated_started = asyncio.Event()
    release_unrelated = asyncio.Event()

    async def child_task(run_id: str, started: asyncio.Event, release: asyncio.Event) -> None:
        await registry.mark_running(run_id)
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await registry.finish(run_id, status="cancelled", reason="cancelled")
            raise
        await registry.finish(run_id, status="success", result="ok")

    async def add_child(run_id: str, parent: str, started: asyncio.Event, release: asyncio.Event) -> None:
        await registry.create_run(run_id=run_id, session_id=session.id, parent_run_id=parent,
                                  description=run_id, prompt="wait", depth=1, background=True)
        task = asyncio.create_task(child_task(run_id, started, release))
        registry.register(run_id, task, ExecutionContext(run_id, "wait", 1),
                          session_id=session.id, parent_run_id=parent)
        await started.wait()

    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        root = await runtime._create_message_run(session.id, "root", client_message_id=None, run_id=None)
        await runtime._finalize_without_outcome(root.run_id, "succeeded", "done")
        unrelated = await runtime._create_message_run(session.id, "other", client_message_id=None, run_id=None)
        await runtime._finalize_without_outcome(unrelated.run_id, "succeeded", "done")
        await add_child("child", root.run_id, child_started, asyncio.Event())
        await add_child("grandchild", "child", grandchild_started, asyncio.Event())
        await add_child("unrelated", unrelated.run_id, unrelated_started, release_unrelated)
        result = await runtime.cancel_run(root.run_id)
        assert result.status == "succeeded"
        assert (await runtime.get_run("child")).status == "cancelled"
        assert (await runtime.get_run("grandchild")).status == "cancelled"
        assert (await runtime.get_run("unrelated")).status == "running"
        with pytest.raises(RuntimeError):
            registry.assert_can_spawn("child")
    finally:
        release_unrelated.set()
        await runtime.shutdown()
        await db.dispose()


async def test_real_rpc_disconnect_does_not_revoke_cancel(
    tmp_path: Path, free_port: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.app import CoreApp
    from tars_agent.core.transport.socket_client import IpcDisconnectedError, IpcError, SocketClient
    from tars_agent.core.transport.socket_server import SocketServer
    monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 2.0)
    runner = SlowCancellationRunner()
    db, runtime = await make_runtime(tmp_path, runner)
    app = CoreApp()
    app._runtime = runtime
    server = SocketServer("127.0.0.1", free_port)
    server.register("run.cancel", app._run_cancel_handler)
    first = SocketClient("127.0.0.1", free_port)
    second = SocketClient("127.0.0.1", free_port)
    loops: list[asyncio.Task[None]] = []
    try:
        await server.start()
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        run = await runtime.submit_message(session.id, "wait")
        await runner.started.wait()
        await first.connect()
        loops.append(asyncio.create_task(first.run_event_loop()))
        request = asyncio.create_task(first.send_command("run.cancel", {"run_id": run.run_id}))
        await runner.cancelled.wait()
        await first.close()
        with pytest.raises(IpcDisconnectedError):
            await request
        monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 0.03)
        await second.connect()
        loops.append(asyncio.create_task(second.run_event_loop()))
        with pytest.raises(IpcError) as caught:
            await second.send_command("run.cancel", {"run_id": run.run_id})
        assert caught.value.code == -32033
        assert caught.value.data["cancellation_requested"] is True
        assert runner.cancel_count == 1
        monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 2.0)
        runner.release.set()
        stopped = await second.send_command("run.cancel", {"run_id": run.run_id})
        assert stopped["status"] == "cancelled"
        assert runner.cancel_count == 1
    finally:
        runner.release.set()
        await first.close()
        await second.close()
        await asyncio.gather(*loops, return_exceptions=True)
        await server.stop()
        await runtime.shutdown()
        await db.dispose()


async def test_queued_one_shot_cancel_closes_session_without_starting_runner(tmp_path: Path) -> None:
    runner = SlowCancellationRunner()
    db, runtime = await make_runtime(tmp_path, runner)
    try:
        session = await runtime.create_session("one_shot", workspace_root=tmp_path)
        run = await runtime._create_message_run(
            session.id, "queued", client_message_id=None, run_id=None,
        )
        result = await runtime.cancel_run(run.run_id)
        assert result.status == "cancelled"
        assert not runner.started.is_set()
        assert (await runtime.get_session(session.id)).status == "closed"
    finally:
        await runtime.shutdown()
        await db.dispose()
