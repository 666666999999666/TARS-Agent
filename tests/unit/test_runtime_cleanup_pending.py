from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tars_agent.core.bus.envelope import HandlerError
from tars_agent.core.events.bus import EventBus
from tars_agent.core.persistence import Database, StateRepository, ToolInvocationRecord
from tars_agent.core.runtime import service
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeCleanupPending, RuntimeRouter


class UnconfirmedSandbox(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.pending: set[str] = set()
        self.confirmed = False
        self.attempts = 0
    async def cleanup_run(self, run_id: str) -> None:
        if run_id not in self.pending:
            return
        self.attempts += 1
        if not self.confirmed:
            raise RuntimeCleanupPending(run_id)
        self.pending.discard(run_id)
    def pending_cleanup_run_ids(self) -> tuple[str, ...]:
        return tuple(self.pending)


class PendingRunner:
    def __init__(self, sandbox: UnconfirmedSandbox, normal_finish: bool) -> None:
        self.sandbox = sandbox
        self.normal_finish = normal_finish
        self.started = asyncio.Event()
    async def run_and_capture(self, goal: str, **kwargs: Any) -> Any:
        run_id = str(kwargs["run_id"])
        self.sandbox.pending.add(run_id)
        self.started.set()
        if not self.normal_finish:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
        # Models the real Runner finally receiving a failed Docker rm/inspect.
        raise RuntimeCleanupPending(run_id)


async def add_running_tool(database: Database, run_id: str) -> None:
    async with database.transaction() as sql:
        await StateRepository(sql).add_tool_invocation(ToolInvocationRecord(
            id=f"{run_id}:pending-tool", run_id=run_id, tool_name="bash",
            parameters={"command": "fixture"}, backend="workspace_sandbox", status="running",
            started_at=datetime.now(UTC),
        ))


async def tool_status(database: Database, run_id: str) -> str:
    async with database.session() as sql:
        tool = await StateRepository(sql).get_tool_invocation(f"{run_id}:pending-tool")
        assert tool is not None
        if tool.status == "running":
            assert tool.finished_at is None
        else:
            assert tool.finished_at is not None
        return tool.status


@pytest.mark.parametrize("normal_finish", [False, True])
async def test_cleanup_must_be_confirmed_before_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, normal_finish: bool,
) -> None:
    monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 0.08)
    monkeypatch.setattr(service, "CLEANUP_RETRY_SECONDS", 0.01)
    db = Database(tmp_path / "state.db")
    await db.create_schema()
    sandbox = UnconfirmedSandbox()
    runner = PendingRunner(sandbox, normal_finish)
    runtime = RuntimeService(db, lambda: runner, EventBus(),  # type: ignore[arg-type,return-value]
                             artifacts_root=tmp_path / "artifacts", tool_runtime=RuntimeRouter(sandbox))
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "wait")
        await runner.started.wait()
        await add_running_tool(db, submitted.run_id)
        if normal_finish:
            async with asyncio.timeout(2):
                while submitted.run_id not in runtime._cancellations:
                    await asyncio.sleep(0.01)
        with pytest.raises(HandlerError) as caught:
            await runtime.cancel_run(submitted.run_id)
        assert caught.value.code == -32033
        assert submitted.run_id in caught.value.data["pending_run_ids"]
        pending = await runtime.get_run(submitted.run_id)
        assert pending.status == "running"
        assert pending.reason == "cleanup_pending"
        assert await tool_status(db, submitted.run_id) == "running"
        assert sandbox.attempts > 0
        sandbox.confirmed = True
        monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 2.0)
        final = await runtime.cancel_run(submitted.run_id)
        assert final.status == ("failed" if normal_finish else "cancelled")
        assert await tool_status(db, submitted.run_id) == (
            "interrupted" if normal_finish else "cancelled"
        )
        assert sandbox.pending == set()
    finally:
        sandbox.confirmed = True
        await runtime.shutdown()
        await db.dispose()


async def test_shutdown_stops_cleanup_retries_without_claiming_container_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 0.08)
    monkeypatch.setattr(service, "CLEANUP_RETRY_SECONDS", 0.01)
    db = Database(tmp_path / "state.db")
    await db.create_schema()
    sandbox = UnconfirmedSandbox()
    runner = PendingRunner(sandbox, False)
    runtime = RuntimeService(db, lambda: runner, EventBus(),  # type: ignore[arg-type,return-value]
                             artifacts_root=tmp_path / "artifacts", tool_runtime=RuntimeRouter(sandbox))
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "wait")
        await runner.started.wait()
        with pytest.raises(HandlerError):
            await runtime.cancel_run(submitted.run_id)
        await asyncio.wait_for(runtime.shutdown(), timeout=2.0)
        assert (await runtime.get_run(submitted.run_id)).status == "running"
        assert sandbox.pending == {submitted.run_id}
        attempts = sandbox.attempts
        await asyncio.sleep(0.03)
        assert sandbox.attempts == attempts
    finally:
        sandbox.confirmed = True
        await db.dispose()


async def test_background_child_pending_cleanup_emits_completion_only_after_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.bus.events import SubagentFinishedEvent
    from tars_agent.core.subagent.registry import BackgroundTaskRegistry
    monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 0.08)
    monkeypatch.setattr(service, "CLEANUP_RETRY_SECONDS", 0.01)
    db = Database(tmp_path / "state.db")
    await db.create_schema()
    sandbox = UnconfirmedSandbox()
    bus = EventBus()
    registry = BackgroundTaskRegistry(db, bus)
    finished: list[Any] = []
    async def observe(event: Any) -> None:
        if isinstance(event, SubagentFinishedEvent):
            finished.append(event)
    bus.subscribe(observe)
    runtime = RuntimeService(db, lambda: None, bus,  # type: ignore[arg-type,return-value]
                             artifacts_root=tmp_path / "artifacts", tool_runtime=RuntimeRouter(sandbox),
                             subagent_registry=registry)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        parent = await runtime._create_message_run(session.id, "parent", client_message_id=None, run_id=None)
        await runtime._finalize_without_outcome(parent.run_id, "succeeded", "done")
        await registry.create_run(run_id="child", session_id=session.id, parent_run_id=parent.run_id,
                                  description="child", prompt="done", depth=1, background=True)
        await registry.mark_running("child")
        await add_running_tool(db, "child")
        sandbox.pending.add("child")
        registry.mark_cleanup_pending("child")
        with pytest.raises(HandlerError):
            await runtime.cancel_run("child")
        assert finished == []
        assert (await runtime.get_run("child")).status == "running"
        assert await tool_status(db, "child") == "running"
        sandbox.confirmed = True
        monkeypatch.setattr(service, "CANCEL_WAIT_SECONDS", 2.0)
        assert (await runtime.cancel_run("child")).status == "failed"
        assert await tool_status(db, "child") == "interrupted"
        assert (await runtime.get_run(parent.run_id)).status == "succeeded"
        assert len(finished) == 1
        assert finished[0].run_id == "child"
    finally:
        sandbox.confirmed = True
        await runtime.shutdown()
        await db.dispose()
