from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tars_agent.core.bus.events import StepFinishedEvent, SubagentFinishedEvent
from tars_agent.core.config import TarsConfig
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from tars_agent.core.persistence import Database, RunRecord, SessionRecord, StateRepository
from tars_agent.core.runner import AgentRunner, RunOutcome
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.subagent.registry import BackgroundTaskRegistry
from tars_agent.core.subagent.tool import AgentCancelTool, AgentResultTool, SpawnAgentTool
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeCleanupPending, RuntimeRouter


def _response(text: str) -> LlmResponse:
    return LlmResponse(
        stop_reason="end_turn",
        tool_calls=[],
        text=text,
        usage=UsageStats(1, 1, 0, 0, 0.01),
    )


async def _database_with_parent(tmp_path: Path) -> Database:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    async with database.transaction() as session:
        repository = StateRepository(session)
        await repository.add_session(
            SessionRecord(
                id="sess-owner",
                status="running",
                workspace_root=str(tmp_path),
                active_run_id="parent-run",
            )
        )
        await repository.add_run(
            RunRecord(
                id="parent-run",
                session_id="sess-owner",
                kind="chat",
                status="running",
            )
        )
    return database


def _spawn_tool(
    tmp_path: Path,
    registry: BackgroundTaskRegistry,
    provider: Any,
) -> SpawnAgentTool:
    return SpawnAgentTool(
        provider=provider,
        parent_bus=registry._bus,
        parent_run_id="parent-run",
        permission_manager=None,
        max_steps=5,
        task_registry=registry,
        runs_dir=tmp_path / "runs",
        session_id="sess-owner",
        workspace_root=tmp_path,
        tool_runtime=RuntimeRouter(FakeRuntime(), allow_host_fallback=False),
    )


def _run_id(content: str) -> str:
    return content.split("run_id=", 1)[1].split(".", 1)[0]


async def _wait_terminal(
    registry: BackgroundTaskRegistry,
    run_id: str,
) -> None:
    for _ in range(200):
        snapshot = await registry.get_snapshot(run_id, session_id="sess-owner")
        if snapshot is not None and snapshot.status not in {"queued", "running"}:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"subagent did not finish: {run_id}")


async def test_competing_child_finishes_return_the_committed_result(tmp_path: Path) -> None:
    database = await _database_with_parent(tmp_path)
    bus = EventBus()
    registry = BackgroundTaskRegistry(database, bus)
    events: list[SubagentFinishedEvent] = []

    async def record(event: Any) -> None:
        if isinstance(event, SubagentFinishedEvent):
            events.append(event)

    bus.subscribe(record)
    try:
        await registry.create_run(
            run_id="child", session_id="sess-owner", parent_run_id="parent-run",
            description="race", prompt="race", depth=1, background=True,
        )
        finishes = await asyncio.gather(
            registry.finish("child", status="success", result="answer", reason="done"),
            registry.finish("child", status="cancelled", reason="cancelled"),
        )
        assert sum(changed for _, changed in finishes) == 1
        committed = await registry.get_snapshot("child")
        assert committed is not None
        assert all(snapshot == committed for snapshot, _ in finishes)
        late, changed = await registry.finish("child", status="failed", result="late")
        assert not changed
        assert late == committed
        assert len(events) == 1
        assert events[0].status == (
            "success" if committed.status == "succeeded" else committed.status
        )
    finally:
        await registry.shutdown()
        await database.dispose()


async def test_success_racing_cancel_and_shutdown_emits_one_persisted_child_terminal(
    tmp_path: Path,
) -> None:
    database = await _database_with_parent(tmp_path)
    bus = EventBus()
    registry = BackgroundTaskRegistry(database, bus)
    release = asyncio.Event()
    started_model = asyncio.Event()
    notifications: list[SubagentFinishedEvent] = []

    async def chat(**_kwargs: Any) -> LlmResponse:
        started_model.set()
        await release.wait()
        return _response("completed answer")

    async def observe(event: Any) -> None:
        if isinstance(event, SubagentFinishedEvent):
            snapshot = await registry.get_snapshot(event.run_id)
            assert snapshot is not None and snapshot.status in {"succeeded", "cancelled"}
            assert event.status == ("success" if snapshot.status == "succeeded" else snapshot.status)
            notifications.append(event)

    bus.subscribe(observe)
    provider = MagicMock()
    provider.chat = chat
    try:
        result = await _spawn_tool(tmp_path, registry, provider).invoke({
            "description": "race", "prompt": "finish or cancel", "run_in_background": True,
        })
        run_id = _run_id(result.content)
        await started_model.wait()
        release.set()
        await registry.cancel(run_id)
        await registry.shutdown()
        snapshot = await registry.get_snapshot(run_id)
        assert snapshot is not None and snapshot.status in {"succeeded", "cancelled"}
        assert len(notifications) == 1
        rows = [json.loads(line) for line in (
            tmp_path / "runs" / run_id / "events.jsonl"
        ).read_text(encoding="utf-8").splitlines()]
        assert rows[0]["type"] == "subagent.started"
        assert rows[-1]["type"] == "subagent.finished"
        assert len([row for row in rows if row["type"] == "subagent.finished"]) == 1
        assert {row["run_id"] for row in rows} == {run_id}
        assert registry._writers == {}
    finally:
        release.set()
        await registry.shutdown()
        await database.dispose()


async def test_child_log_stays_open_until_cleanup_is_confirmed(tmp_path: Path) -> None:
    class PendingCleanup(FakeRuntime):
        confirmed = False

        async def cleanup_run(self, run_id: str) -> None:
            if not self.confirmed:
                raise RuntimeCleanupPending(run_id)
            await super().cleanup_run(run_id)

    database = await _database_with_parent(tmp_path)
    bus = EventBus()
    registry = BackgroundTaskRegistry(database, bus)
    sandbox = PendingCleanup()
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=_response("answer before cleanup"))
    tool = _spawn_tool(tmp_path, registry, provider)
    tool._tool_runtime = RuntimeRouter(sandbox, allow_host_fallback=False)
    try:
        started = await tool.invoke({
            "description": "pending", "prompt": "finish", "run_in_background": True,
        })
        run_id = _run_id(started.content)
        entry = registry.get(run_id)
        assert entry is not None
        with pytest.raises(RuntimeCleanupPending):
            await entry[0]
        assert registry.cleanup_is_pending(run_id)
        snapshot = await registry.get_snapshot(run_id)
        assert snapshot is not None and snapshot.status == "running"
        writer = registry._writers[run_id]
        path = tmp_path / "runs" / run_id / "events.jsonl"
        await registry.start_recording(run_id, path)
        assert registry._writers[run_id] is writer
        assert writer._file is not None and not writer._file.closed
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert not any(row["type"] == "subagent.finished" for row in rows)

        # A late event is still recorded while the owner is waiting for cleanup.
        await bus.publish(StepFinishedEvent(run_id=run_id, step=99, ts="cleanup pending"))
        sandbox.confirmed = True
        await tool._tool_runtime.cleanup_run(run_id)
        registry.complete_cleanup(run_id)
        await registry.finish(run_id, status="failed", reason="cleanup_failed")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows[-2]["step"] == 99
        assert rows[-1]["type"] == "subagent.finished"
        assert len([row for row in rows if row["type"] == "subagent.finished"]) == 1
        assert writer._file is None
        assert registry._writers == {}
    finally:
        sandbox.confirmed = True
        await registry.shutdown()
        await database.dispose()


async def test_repeated_cancellation_cannot_interrupt_committed_child_notification(
    tmp_path: Path,
) -> None:
    database = await _database_with_parent(tmp_path)
    bus = EventBus()
    registry = BackgroundTaskRegistry(database, bus)
    publishing = asyncio.Event()
    release = asyncio.Event()

    async def slow_subscriber(event: Any) -> None:
        if isinstance(event, SubagentFinishedEvent):
            publishing.set()
            await release.wait()

    bus.subscribe(slow_subscriber)
    try:
        await registry.create_run(
            run_id="child", session_id="sess-owner", parent_run_id="parent-run",
            description="notify", prompt="done", depth=1, background=True,
        )
        path = tmp_path / "child.jsonl"
        await registry.start_recording("child", path)
        operation = asyncio.create_task(registry.finish("child", status="success", result="done"))
        await publishing.wait()
        for _ in range(2):
            operation.cancel()
            await asyncio.sleep(0)
        assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        snapshot, changed = await registry.finish("child", status="cancelled")
        assert not changed
        assert snapshot.status == "succeeded"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1 and rows[0]["type"] == "subagent.finished"
        assert registry._writers == {}
    finally:
        release.set()
        await registry.shutdown()
        await database.dispose()


async def test_late_recording_is_closed_when_finish_finds_an_existing_terminal(
    tmp_path: Path,
) -> None:
    database = await _database_with_parent(tmp_path)
    registry = BackgroundTaskRegistry(database, EventBus())
    try:
        await registry.create_run(
            run_id="child", session_id="sess-owner", parent_run_id="parent-run",
            description="late cleanup", prompt="done", depth=1, background=True,
        )
        path = tmp_path / "child.jsonl"
        await registry.start_recording("child", path)
        original, changed = await registry.finish("child", status="success", result="done")
        assert changed
        # Runtime cleanup can have observed running just before natural completion.
        await registry.start_recording("child", path)
        assert "child" in registry._writers
        repeated, changed = await registry.finish("child", status="cancelled")
        assert not changed and repeated == original
        assert registry._writers == {}
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1 and rows[0]["status"] == "success"
    finally:
        await registry.shutdown()
        await database.dispose()


async def test_background_result_survives_active_task_pruning_and_new_parent_run(
    tmp_path: Path,
) -> None:
    database = await _database_with_parent(tmp_path)
    registry = BackgroundTaskRegistry(database, EventBus())
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=_response("durable answer"))
    try:
        started = await _spawn_tool(tmp_path, registry, provider).invoke(
            {
                "description": "durable child",
                "prompt": "produce an answer",
                "run_in_background": True,
            }
        )
        run_id = _run_id(started.content)
        await _wait_terminal(registry, run_id)
        assert registry.get(run_id) is None

        async with database.transaction() as session:
            repository = StateRepository(session)
            original_parent = await repository.get_run("parent-run")
            owner_session = await repository.get_session("sess-owner")
            assert original_parent is not None
            assert owner_session is not None
            original_parent.status = "succeeded"
            owner_session.status = "ready"
            owner_session.active_run_id = None

        child_run_id = run_id

        class QueryResultProvider:
            def __init__(self) -> None:
                self.calls = 0
                self.observed_tool_result = False

            async def chat(
                self,
                messages: list[dict[str, object]],
                tool_schemas: list[dict[str, object]],
                bus: EventBus,
                run_id: str,
                *,
                step: int = 0,
                system: str | None = None,
            ) -> LlmResponse:
                del bus, run_id, step, system
                self.calls += 1
                if self.calls == 1:
                    assert any(schema.get("name") == "agent_result" for schema in tool_schemas)
                    return LlmResponse(
                        stop_reason="tool_use",
                        tool_calls=[
                            ToolCallBlock(
                                id="query-child-result",
                                name="agent_result",
                                input={"run_id": child_run_id},
                            )
                        ],
                        usage=UsageStats(1, 1, 0, 0, 0.01),
                    )
                self.observed_tool_result = "durable answer" in str(messages)
                return _response("queried from a later parent run")

        query_provider = QueryResultProvider()
        bus = registry._bus
        runtime = RuntimeService(
            database,
            lambda: AgentRunner(
                TarsConfig(),
                provider=query_provider,  # type: ignore[arg-type]
                bus=bus,
                task_registry=registry,
                tool_runtime=RuntimeRouter(FakeRuntime(), allow_host_fallback=False),
            ),
            bus,
            artifacts_root=tmp_path / "artifacts",
            subagent_registry=registry,
        )
        submitted = await runtime.submit_message(
            "sess-owner",
            "retrieve the earlier child result",
            client_message_id="later-parent-message",
        )
        for _ in range(200):
            later_parent = await runtime.get_run(submitted.run_id)
            if later_parent.status not in {"queued", "running"}:
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("later parent run did not finish")

        assert submitted.run_id != "parent-run"
        assert later_parent.kind == "chat"
        assert later_parent.status == "succeeded"
        assert query_provider.observed_tool_result

        async with database.session() as session:
            record = await StateRepository(session).get_run(run_id)
        assert record is not None
        assert record.parent_run_id == "parent-run"
        assert record.kind == "subagent"
        assert record.status == "succeeded"
        await runtime.shutdown()
    finally:
        await registry.shutdown()
        await database.dispose()


async def test_subagent_query_and_cancel_are_session_scoped(tmp_path: Path) -> None:
    database = await _database_with_parent(tmp_path)
    registry = BackgroundTaskRegistry(database, EventBus())
    release = asyncio.Event()

    async def slow_chat(**_kwargs: Any) -> LlmResponse:
        await release.wait()
        return _response("too late")

    provider = MagicMock()
    provider.chat = slow_chat
    try:
        started = await _spawn_tool(tmp_path, registry, provider).invoke(
            {
                "description": "cancel child",
                "prompt": "wait",
                "run_in_background": True,
            }
        )
        run_id = _run_id(started.content)

        foreign_result = await AgentResultTool(
            registry,
            session_id="sess-other",
        ).invoke({"run_id": run_id})
        foreign_cancel = await AgentCancelTool(
            registry,
            session_id="sess-other",
        ).invoke({"run_id": run_id})
        assert foreign_result.is_error
        assert foreign_cancel.is_error

        cancelled = await AgentCancelTool(
            registry,
            session_id="sess-owner",
        ).invoke({"run_id": run_id})
        assert not cancelled.is_error
        snapshot = await registry.get_snapshot(run_id, session_id="sess-owner")
        assert snapshot is not None
        assert snapshot.status == "cancelled"
        assert registry.active_count == 0
    finally:
        release.set()
        await registry.shutdown()
        await database.dispose()


async def test_registry_shutdown_cancels_and_persists_all_background_tasks(
    tmp_path: Path,
) -> None:
    database = await _database_with_parent(tmp_path)
    registry = BackgroundTaskRegistry(database, EventBus())
    blocker = asyncio.Event()

    async def slow_chat(**_kwargs: Any) -> LlmResponse:
        await blocker.wait()
        return _response("done")

    provider = MagicMock()
    provider.chat = slow_chat
    run_ids: list[str] = []
    try:
        tool = _spawn_tool(tmp_path, registry, provider)
        for index in range(2):
            started = await tool.invoke(
                {
                    "description": f"child {index}",
                    "prompt": "wait",
                    "run_in_background": True,
                }
            )
            run_ids.append(_run_id(started.content))
        assert registry.active_count == 2

        await registry.shutdown()
        assert registry.active_count == 0
        for run_id in run_ids:
            snapshot = await registry.get_snapshot(run_id, session_id="sess-owner")
            assert snapshot is not None
            assert snapshot.status == "cancelled"
            assert snapshot.reason in {"cancelled", "core_shutdown"}
    finally:
        blocker.set()
        await registry.shutdown()
        await database.dispose()


async def test_runtime_run_cancel_routes_subagent_to_daemon_registry(
    tmp_path: Path,
) -> None:
    database = await _database_with_parent(tmp_path)
    registry = BackgroundTaskRegistry(database, EventBus())
    release = asyncio.Event()

    async def slow_chat(**_kwargs: Any) -> LlmResponse:
        await release.wait()
        return _response("done")

    provider = MagicMock()
    provider.chat = slow_chat

    class _UnusedRunner:
        async def run_and_capture(self, *_args: Any, **_kwargs: Any) -> RunOutcome:
            raise AssertionError("main runner must not execute")

    runtime = RuntimeService(
        database,
        lambda: _UnusedRunner(),  # type: ignore[return-value]
        EventBus(),
        artifacts_root=tmp_path / "artifacts",
        subagent_registry=registry,
    )
    try:
        started = await _spawn_tool(tmp_path, registry, provider).invoke(
            {
                "description": "runtime cancel",
                "prompt": "wait",
                "run_in_background": True,
            }
        )
        run_id = _run_id(started.content)
        cancelled = await runtime.cancel_run(run_id)
        assert cancelled.status == "cancelled"

        parent = await runtime.get_run("parent-run")
        assert parent.status == "running"
    finally:
        release.set()
        await runtime.shutdown()
        await database.dispose()


async def test_background_subagents_complete_in_parallel_and_persist_failure(
    tmp_path: Path,
) -> None:
    database = await _database_with_parent(tmp_path)
    registry = BackgroundTaskRegistry(database, EventBus())
    both_started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    max_active = 0

    async def chat(**kwargs: Any) -> LlmResponse:
        nonlocal active, max_active
        prompt = str(kwargs.get("messages", ""))
        if "explode" in prompt:
            raise RuntimeError("provider exploded")
        active += 1
        max_active = max(max_active, active)
        if active == 2:
            both_started.set()
        await release.wait()
        active -= 1
        return _response("parallel answer")

    provider = MagicMock()
    provider.chat = chat
    try:
        tool = _spawn_tool(tmp_path, registry, provider)
        parallel_ids: list[str] = []
        for index in range(2):
            started = await tool.invoke(
                {
                    "description": f"parallel {index}",
                    "prompt": f"parallel work {index}",
                    "run_in_background": True,
                }
            )
            parallel_ids.append(_run_id(started.content))
        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert max_active == 2
        release.set()
        for run_id in parallel_ids:
            await _wait_terminal(registry, run_id)
            snapshot = await registry.get_snapshot(run_id, session_id="sess-owner")
            assert snapshot is not None
            assert snapshot.status == "succeeded"

        failed = await tool.invoke(
            {
                "description": "failure",
                "prompt": "explode",
                "run_in_background": True,
            }
        )
        failed_id = _run_id(failed.content)
        await _wait_terminal(registry, failed_id)
        failed_snapshot = await registry.get_snapshot(failed_id, session_id="sess-owner")
        assert failed_snapshot is not None
        assert failed_snapshot.status == "failed"
        assert failed_snapshot.reason in {"llm_error", "runtime_error"}
    finally:
        release.set()
        await registry.shutdown()
        await database.dispose()


async def test_recovery_marks_subagent_kind_interrupted(tmp_path: Path) -> None:
    database = await _database_with_parent(tmp_path)
    bus = EventBus()
    registry = BackgroundTaskRegistry(database, bus)
    async with database.transaction() as session:
        await StateRepository(session).add_run(
            RunRecord(
                id="child-before-restart",
                session_id="sess-owner",
                parent_run_id="parent-run",
                kind="subagent",
                status="running",
            )
        )

    class _UnusedRunner:
        async def run_and_capture(self, *_args: Any, **_kwargs: Any) -> RunOutcome:
            raise AssertionError("recovery must not replay a subagent coroutine")

    runtime = RuntimeService(
        database,
        lambda: _UnusedRunner(),  # type: ignore[return-value]
        bus,
        artifacts_root=tmp_path / "artifacts",
        subagent_registry=registry,
    )
    try:
        recovered = await runtime.recover_interrupted()
        assert recovered == 2  # the fixture's parent Run and its child were both active
        child = await runtime.get_run("child-before-restart")
        assert child.kind == "subagent"
        assert child.status == "interrupted"
        assert child.reason == "daemon_restarted"
    finally:
        await runtime.shutdown()
        await database.dispose()
