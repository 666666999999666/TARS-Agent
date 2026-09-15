from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tars_agent.core.bus.events import RunFinishedEvent, SubagentFinishedEvent
from tars_agent.core.events.bus import EventBus
from tars_agent.core.persistence import Database, StateRepository, ToolInvocationRecord
from tars_agent.core.runner import RunOutcome
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.subagent.registry import BackgroundTaskRegistry


@pytest.mark.parametrize("owner,status", [
    ("outcome", "cancelled"), ("outcome", "failed"), ("outcome", "success"),
    ("fallback", "cancelled"), ("recovery", "interrupted"),
    ("child", "cancelled"), ("child", "failed"), ("child", "success"),
    ("child_recovery", "interrupted"),
])
async def test_run_terminal_closes_missing_tool_results_without_rewriting_finished_tools(
    tmp_path: Path, owner: str, status: str,
) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    bus = EventBus()
    children = BackgroundTaskRegistry(database, bus)

    def unused_runner() -> Any:
        raise AssertionError("this test finalizes existing execution records")

    runtime = RuntimeService(
        database, unused_runner, bus, artifacts_root=tmp_path / "artifacts",
        subagent_registry=children,
    )
    original_finished_at = datetime(2026, 1, 1, tzinfo=UTC)
    expected = "cancelled" if status == "cancelled" else "interrupted"
    notifications: list[Any] = []
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        parent = await runtime._create_message_run(
            session.id, "original", client_message_id="original", run_id=None,
        )
        run_id = parent.run_id
        if owner in {"child", "child_recovery"}:
            run_id = "child"
            await children.create_run(
                run_id=run_id, session_id=session.id, parent_run_id=parent.run_id,
                description="child", prompt="original", depth=1, background=True,
            )
        async with database.transaction() as sql:
            repository = StateRepository(sql)
            for tool_status in ("queued", "running", "succeeded", "failed"):
                done = tool_status in {"succeeded", "failed"}
                await repository.add_tool_invocation(ToolInvocationRecord(
                    id=f"{run_id}:{tool_status}", run_id=run_id, tool_name="bash",
                    parameters={"command": "fixture"}, backend="workspace_sandbox",
                    status=tool_status, retryable=True,
                    started_at=None if tool_status == "queued" else original_finished_at,
                    finished_at=original_finished_at if done else None,
                    result={"content": "original output"} if done else None,
                    error_class="original_error" if tool_status == "failed" else None,
                    error_message="original detail" if tool_status == "failed" else None,
                ))

        async def observe(event: Any) -> None:
            if isinstance(event, (RunFinishedEvent, SubagentFinishedEvent)) and event.run_id == run_id:
                async with database.session() as sql:
                    records = await StateRepository(sql).list_tool_invocations(run_id)
                    assert not any(record.status in {"queued", "running"} for record in records)
                notifications.append(event)

        bus.subscribe(observe)
        if owner == "outcome":
            await runtime._finish_run(run_id, RunOutcome(status=status, result="", reason="test_end"))
        elif owner == "fallback":
            await runtime._finalize_without_outcome(run_id, status, "test_end")
        elif owner in {"recovery", "child_recovery"}:
            await runtime.recover_interrupted()
        else:
            await children.finish(run_id, status=status, reason="test_end")

        async with database.session() as sql:
            records = {r.id.rsplit(":", 1)[1]: r for r in await StateRepository(sql).list_tool_invocations(run_id)}
            run = await StateRepository(sql).get_run(run_id)
            assert run is not None
        for original_status in ("queued", "running"):
            record = records[original_status]
            assert record.status == expected
            assert record.finished_at == run.finished_at
            assert record.retryable is False
            assert record.result is None
            assert record.error_class == f"{expected}_without_result"
            assert "no final tool result" in (record.error_message or "")
        for original_status in ("succeeded", "failed"):
            record = records[original_status]
            assert record.status == original_status
            assert record.finished_at is not None
            assert record.finished_at.replace(tzinfo=UTC) == original_finished_at
            assert record.retryable is True
            assert record.result == {"content": "original output"}
        assert records["failed"].error_class == "original_error"
        assert records["failed"].error_message == "original detail"
        assert len(notifications) == 1
    finally:
        await runtime.shutdown()
        await database.dispose()
