from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from tars_agent.core.bus.envelope import HandlerError
from tars_agent.core.bus.events import RunFinishedEvent, ToolCallFailedEvent, ToolCallStartedEvent
from tars_agent.core.compact.compactor import CompactionResult
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import LlmResponse, UsageStats
from tars_agent.core.persistence import (
    CompactionRecord,
    Database,
    RunRecord,
    SessionRecord,
    StateRepository,
    TurnRecord,
)
from tars_agent.core.runner import RunOutcome
from tars_agent.core.runtime.service import (
    COMPACTION_FAILED,
    RUN_SIDE_EFFECT_CONFIRMATION_REQUIRED,
    RuntimeService,
)


class _ControlledRunner:
    def __init__(
        self,
        started: asyncio.Event,
        release: asyncio.Event,
        outcome: RunOutcome,
    ) -> None:
        self.started = started
        self.release = release
        self.outcome = outcome
        self.goals: list[str] = []
        self.histories: list[list[dict[str, Any]]] = []

    async def run_and_capture(
        self,
        goal: str,
        **kwargs: Any,
    ) -> RunOutcome:
        self.goals.append(goal)
        self.histories.append(list(kwargs.get("history") or []))
        self.started.set()
        await self.release.wait()
        return self.outcome


async def _runtime(
    tmp_path: Path,
    runner: _ControlledRunner,
    *,
    bus: EventBus | None = None,
) -> tuple[Database, RuntimeService]:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    runtime = RuntimeService(
        database,
        lambda: runner,  # type: ignore[arg-type,return-value]
        bus or EventBus(),
        artifacts_root=tmp_path / "artifacts",
    )
    return database, runtime


async def _wait_terminal(runtime: RuntimeService, run_id: str) -> str:
    for _ in range(200):
        snapshot = await runtime.get_run(run_id)
        if snapshot.status in {"succeeded", "failed", "cancelled", "interrupted"}:
            return snapshot.status
        await asyncio.sleep(0.005)
    raise AssertionError(f"run did not finish: {run_id}")


# 功能：验证消息提交立即返回 queued Run，且同一 client_message_id 重发只返回原 Run
# 设计：受控 Runner 在 Event 上阻塞，使测试能在执行未完成时观察返回值和数据库幂等结果
async def test_submit_returns_before_execution_and_deduplicates_message_id(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(
            status="success",
            result="done",
            reason=None,
            messages=[{"role": "assistant", "content": "done"}],
            steps=1,
        ),
    )
    database, runtime = await _runtime(tmp_path, runner)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        first = await runtime.submit_message(
            session.id,
            "hello",
            client_message_id="client-1",
        )
        duplicate = await runtime.submit_message(
            session.id,
            "hello again",
            client_message_id="client-1",
        )

        assert first.status == "queued"
        assert duplicate.run_id == first.run_id
        assert duplicate.deduplicated is True
        await asyncio.wait_for(started.wait(), timeout=1)
        assert runtime.supervisor.is_active(first.run_id)

        release.set()
        assert await _wait_terminal(runtime, first.run_id) == "succeeded"
    finally:
        await runtime.shutdown()
        await database.dispose()


# 功能：验证成功 Run 原子提交用户输入与模型输出，并把 Session 恢复为 ready
# 设计：Runner 完成前历史为空，释放后从只读 Runtime API 检查正式上下文和状态快照
async def test_success_commits_complete_turn_to_history(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(
            status="success",
            result="answer",
            reason=None,
            messages=[{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}],
            steps=2,
        ),
    )
    database, runtime = await _runtime(tmp_path, runner)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "question")
        await asyncio.wait_for(started.wait(), timeout=1)
        assert await runtime.get_history(session.id) == []

        release.set()
        assert await _wait_terminal(runtime, submitted.run_id) == "succeeded"

        history = await runtime.get_history(session.id)
        assert [(item["role"], item["content"]) for item in history] == [
            ("user", "question"),
            ("assistant", [{"type": "text", "text": "answer"}]),
        ]
        resumed = await runtime.get_session(session.id)
        assert resumed.status == "ready"
        assert resumed.active_run_id is None
    finally:
        await runtime.shutdown()
        await database.dispose()


# 功能：验证失败 Run 的中间消息保留在数据库审计区，但不会进入正式历史
# 设计：让 Runner 返回失败和一条部分输出，分别通过 Runtime 历史与 Repository 审计查询断言
async def test_failed_run_messages_remain_uncommitted_for_audit(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(
            status="failed",
            result="",
            reason="llm_stream_interrupted",
            messages=[{"role": "assistant", "content": "partial"}],
            steps=1,
        ),
    )
    database, runtime = await _runtime(tmp_path, runner)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "question")
        await asyncio.wait_for(started.wait(), timeout=1)
        release.set()
        assert await _wait_terminal(runtime, submitted.run_id) == "failed"

        assert await runtime.get_history(session.id) == []
        async with database.session() as db_session:
            messages = await StateRepository(db_session).list_run_messages(submitted.run_id)
        assert [message.committed for message in messages] == [False, False]
        assert [message.content for message in messages] == ["question", "partial"]
    finally:
        await runtime.shutdown()
        await database.dispose()


async def test_tool_retryability_is_persisted_for_audit(tmp_path: Path) -> None:
    started = asyncio.Event()
    runner = _ControlledRunner(
        started,
        asyncio.Event(),
        RunOutcome(status="success", result="", reason=None),
    )
    database, runtime = await _runtime(tmp_path, runner)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "audit tool")
        await asyncio.wait_for(started.wait(), timeout=1)
        await runtime._observe_tool_event(  # noqa: SLF001 - verifies durable observer contract
            ToolCallStartedEvent(
                run_id=submitted.run_id,
                tool_use_id="tool-1",
                tool_name="remote__counter",
                params={"value": 1},
                ts="2026-01-01T00:00:00Z",
            )
        )
        await runtime._observe_tool_event(  # noqa: SLF001 - verifies durable observer contract
            ToolCallFailedEvent(
                run_id=submitted.run_id,
                tool_use_id="tool-1",
                tool_name="remote__counter",
                error_class="rate_limited",
                error_message="retry later",
                elapsed_ms=1,
                retryable=True,
                ts="2026-01-01T00:00:01Z",
            )
        )

        async with database.session() as db_session:
            invocations = await StateRepository(db_session).list_tool_invocations(
                submitted.run_id
            )
        assert len(invocations) == 1
        assert invocations[0].status == "failed"
        assert invocations[0].retryable is True
        assert invocations[0].error_class == "rate_limited"
    finally:
        await runtime.shutdown()
        await database.dispose()


# 功能：验证取消受监管 Run 会落库为 cancelled，并清除 Session 活动 Run
# 设计：Runner 永久阻塞，调用公开 cancel 后轮询终态并检查 Supervisor 与 Session 双重清理
async def test_cancel_run_is_durable_and_clears_supervisor(tmp_path: Path) -> None:
    started = asyncio.Event()
    runner = _ControlledRunner(
        started,
        asyncio.Event(),
        RunOutcome(status="success", result="unexpected", reason=None),
    )
    database, runtime = await _runtime(tmp_path, runner)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "wait")
        await asyncio.wait_for(started.wait(), timeout=1)

        cancelled = await runtime.cancel_run(submitted.run_id)

        assert cancelled.status == "cancelled"
        assert cancelled.reason == "cancelled"
        assert not runtime.supervisor.is_active(submitted.run_id)
        resumed = await runtime.get_session(session.id)
        assert resumed.status == "ready"
        assert resumed.active_run_id is None
    finally:
        await runtime.shutdown()
        await database.dispose()


# 功能：验证 daemon 启动恢复会把遗留 queued/running Run 原子标记为 interrupted
# 设计：直接构造两个不同状态的持久化 Run，再调用恢复入口检查 Run、Turn、Session 一致性
async def test_recover_interrupted_runs_after_restart(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    now = datetime.now(UTC)
    async with database.transaction() as db_session:
        repository = StateRepository(db_session)
        for index, status in enumerate(("queued", "running"), start=1):
            session_id = f"sess-{index}"
            turn_id = f"turn-{index}"
            run_id = f"run-{index}"
            await repository.add_session(
                SessionRecord(
                    id=session_id,
                    mode="chat",
                    status="running",
                    title="",
                    workspace_root=str(tmp_path),
                    active_run_id=run_id,
                    created_at=now,
                    updated_at=now,
                )
            )
            await repository.add_turn(
                TurnRecord(
                    id=turn_id,
                    session_id=session_id,
                    raw_content="x",
                    effective_content="x",
                    status=status,
                    created_at=now,
                    updated_at=now,
                )
            )
            await repository.add_run(
                RunRecord(
                    id=run_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    status=status,
                    execution_options={},
                    created_at=now,
                    updated_at=now,
                )
            )
    runner = _ControlledRunner(asyncio.Event(), asyncio.Event(), RunOutcome("success", "", None))
    runtime = RuntimeService(
        database,
        lambda: runner,  # type: ignore[arg-type,return-value]
        EventBus(),
        artifacts_root=tmp_path / "artifacts",
    )
    try:
        assert await runtime.recover_interrupted() == 2
        for index in (1, 2):
            run = await runtime.get_run(f"run-{index}")
            session = await runtime.get_session(f"sess-{index}")
            assert run.status == "interrupted"
            assert run.reason == "daemon_restarted"
            assert session.status == "ready"
            assert session.active_run_id is None
    finally:
        await runtime.shutdown()
        await database.dispose()


# 功能：验证存在已开始副作用的失败 Run 必须确认后才能创建新 attempt
# 设计：直接构造 side_effects_started 失败记录，先断言结构化拒绝，再确认并完成第二 attempt
async def test_retry_requires_confirmation_after_side_effects(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(
            status="success",
            result="retried",
            reason=None,
            messages=[{"role": "assistant", "content": "retried"}],
        ),
    )
    database, runtime = await _runtime(tmp_path, runner)
    now = datetime.now(UTC)
    async with database.transaction() as db_session:
        repository = StateRepository(db_session)
        await repository.add_session(
            SessionRecord(
                id="sess-retry",
                mode="chat",
                status="ready",
                title="",
                workspace_root=str(tmp_path),
                created_at=now,
                updated_at=now,
            )
        )
        await repository.add_turn(
            TurnRecord(
                id="turn-retry",
                session_id="sess-retry",
                raw_content="change file",
                effective_content="change file",
                status="failed",
                created_at=now,
                updated_at=now,
            )
        )
        await repository.add_run(
            RunRecord(
                id="run-original",
                session_id="sess-retry",
                turn_id="turn-retry",
                attempt=1,
                status="failed",
                reason="runtime_error",
                side_effects_started=True,
                execution_options={},
                created_at=now,
                updated_at=now,
                finished_at=now,
            )
        )
    try:
        with pytest.raises(HandlerError) as captured:
            await runtime.retry_run("run-original")
        assert captured.value.code == RUN_SIDE_EFFECT_CONFIRMATION_REQUIRED

        retried = await runtime.retry_run(
            "run-original",
            confirm_side_effects=True,
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        retry_snapshot = await runtime.get_run(retried.run_id)
        assert retry_snapshot.attempt == 2
        assert retry_snapshot.retry_of_run_id == "run-original"
        release.set()
        assert await _wait_terminal(runtime, retried.run_id) == "succeeded"
    finally:
        await runtime.shutdown()
        await database.dispose()


# 功能：验证 Session list/resume 返回持久化摘要，并发布 resumed 事件及最新 cursor
# 设计：创建 Session 后由 EventBus 模拟 DurableEventHub 追加事件，再检查列表顺序和恢复状态
async def test_list_resume_and_latest_event_cursor(tmp_path: Path) -> None:
    started = asyncio.Event()
    runner = _ControlledRunner(
        started,
        asyncio.Event(),
        RunOutcome(status="success", result="", reason=None),
    )
    database, runtime = await _runtime(tmp_path, runner)
    try:
        first = await runtime.create_session("chat", title="first", workspace_root=tmp_path)
        second = await runtime.create_session("chat", title="second", workspace_root=tmp_path)
        async with database.transaction() as db_session:
            repository = StateRepository(db_session)
            event = await repository.append_event(
                session_id=first.id,
                event_type="session.created",
                payload={"type": "session.created", "session_id": first.id},
            )

        sessions = await runtime.list_sessions(limit=10)
        resumed = await runtime.resume_session(first.id)

        assert {item.id for item in sessions} == {first.id, second.id}
        assert resumed.id == first.id
        assert resumed.status == "ready"
        assert await runtime.latest_event_cursor(first.id) == event.cursor
    finally:
        await runtime.shutdown()
        await database.dispose()


# 功能：验证 Skill 参数展开后只作为 effective user input 进入模型，且原始命令留在 Turn
# 设计：受控 Runner 记录 goal/history/system override，Repository 校验 raw/effective 双份持久化
async def test_skill_arguments_reach_model_without_duplicate_system_prompt(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(status="success", result="done", reason=None),
    )
    database, runtime = await _runtime(tmp_path, runner)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "/review src/example.py")
        await asyncio.wait_for(started.wait(), timeout=1)

        assert runner.goals and "src/example.py" in runner.goals[0]
        assert "$ARGUMENTS" not in runner.goals[0]
        assert runner.histories[-1][-1]["content"] == runner.goals[0]
        async with database.session() as db_session:
            run = await StateRepository(db_session).get_run(submitted.run_id)
            assert run is not None and run.turn_id is not None
            turn = await StateRepository(db_session).get_turn(run.turn_id)
        assert turn is not None
        assert turn.raw_content == "/review src/example.py"
        assert turn.effective_content == runner.goals[0]
        assert "system_prompt_override" not in run.execution_options

        release.set()
        assert await _wait_terminal(runtime, submitted.run_id) == "succeeded"
    finally:
        await runtime.shutdown()
        await database.dispose()


# 功能：验证成功 Run 的压缩结果事务性替换活动上下文并写入 compactions 审计记录
# 设计：Runner 返回 audit messages、摘要 active_context 与指标，完成后同时检查历史和数据库 active 标志
async def test_successful_run_commits_compacted_active_context(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    summary = "## 1. Original Goal\nContinue safely"
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(
            status="success",
            result="done",
            reason=None,
            messages=[{"role": "assistant", "content": "full audit output"}],
            active_context=[
                {"role": "user", "content": summary},
                {"role": "assistant", "content": "Understood"},
            ],
            compactions=[
                CompactionResult(
                    summary_text=summary,
                    original_token_estimate=1200,
                    summary_tokens=80,
                )
            ],
        ),
    )
    database, runtime = await _runtime(tmp_path, runner)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "long task")
        await asyncio.wait_for(started.wait(), timeout=1)
        release.set()
        assert await _wait_terminal(runtime, submitted.run_id) == "succeeded"

        history = await runtime.get_history(session.id)
        assert [(item["role"], item["content"]) for item in history] == [
            ("user", summary),
            ("assistant", "Understood"),
        ]
        async with database.session() as db_session:
            repository = StateRepository(db_session)
            audit = await repository.list_run_messages(submitted.run_id)
            compactions = await db_session.scalars(select(CompactionRecord))
        assert any(message.content == "full audit output" and not message.active for message in audit)
        records = list(compactions)
        assert len(records) == 1
        assert records[0].summary == summary
        assert records[0].context_version == 1

        followup = await runtime.submit_message(session.id, "next question")
        assert await _wait_terminal(runtime, followup.run_id) == "succeeded"
        assert runner.histories[1] == [
            {"role": "user", "content": summary},
            {"role": "assistant", "content": "Understood"},
            {"role": "user", "content": "next question"},
        ]
    finally:
        await runtime.shutdown()
        await database.dispose()


async def test_deferred_compaction_side_effects_publish_only_after_successful_commit(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    summary = "durable deferred summary"
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(
            status="success",
            result="done",
            reason=None,
            active_context=[
                {"role": "user", "content": summary},
                {"role": "assistant", "content": "Understood"},
            ],
            compactions=[CompactionResult(summary, 100, 10)],
        ),
    )
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    bus = EventBus()
    observed: list[object] = []

    async def collect(event: object) -> None:
        observed.append(event)

    bus.subscribe(collect)  # type: ignore[arg-type]
    runtime = RuntimeService(
        database,
        lambda: runner,  # type: ignore[arg-type,return-value]
        bus,
        artifacts_root=tmp_path / "artifacts",
    )
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "compact")
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not list((tmp_path / "artifacts").rglob("summary_*.md"))
        assert not any(
            getattr(event, "type", "") == "context.compacted" for event in observed
        )

        release.set()
        assert await _wait_terminal(runtime, submitted.run_id) == "succeeded"
        assert len(list((tmp_path / "artifacts").rglob("summary_*.md"))) == 1
        event_types = [getattr(event, "type", "") for event in observed]
        assert event_types.index("context.compacted") < event_types.index("run.finished")
    finally:
        await runtime.shutdown()
        await database.dispose()


async def test_failed_run_discards_deferred_compaction_file_and_event(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(
            status="failed",
            result="",
            reason="llm_error",
            compactions=[CompactionResult("uncommitted summary", 100, 10)],
        ),
    )
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    bus = EventBus()
    observed: list[object] = []

    async def collect(event: object) -> None:
        observed.append(event)

    bus.subscribe(collect)  # type: ignore[arg-type]
    runtime = RuntimeService(
        database,
        lambda: runner,  # type: ignore[arg-type,return-value]
        bus,
        artifacts_root=tmp_path / "artifacts",
    )
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "compact then fail")
        await asyncio.wait_for(started.wait(), timeout=1)
        release.set()
        assert await _wait_terminal(runtime, submitted.run_id) == "failed"
        assert not list((tmp_path / "artifacts").rglob("summary_*.md"))
        assert not any(
            getattr(event, "type", "") == "context.compacted" for event in observed
        )
    finally:
        await runtime.shutdown()
        await database.dispose()


async def test_close_waits_for_inflight_submit_then_cancels_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    runner = _ControlledRunner(
        started,
        asyncio.Event(),
        RunOutcome(status="success", result="unexpected", reason=None),
    )
    database, runtime = await _runtime(tmp_path, runner)
    entered_create = asyncio.Event()
    release_create = asyncio.Event()
    original_create = runtime._create_message_run

    async def blocked_create(*args: Any, **kwargs: Any) -> Any:
        entered_create.set()
        await release_create.wait()
        return await original_create(*args, **kwargs)

    monkeypatch.setattr(runtime, "_create_message_run", blocked_create)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submit_task = asyncio.create_task(runtime.submit_message(session.id, "race"))
        await asyncio.wait_for(entered_create.wait(), timeout=1)
        close_task = asyncio.create_task(runtime.close_session(session.id))
        await asyncio.sleep(0)
        assert not close_task.done()

        release_create.set()
        submitted = await asyncio.wait_for(submit_task, timeout=1)
        closed = await asyncio.wait_for(close_task, timeout=1)

        assert closed.status == "closed"
        assert closed.active_run_id is None
        assert (await runtime.get_run(submitted.run_id)).status == "cancelled"
    finally:
        release_create.set()
        await runtime.shutdown()
        await database.dispose()


async def test_runtime_publishes_run_finished_only_after_state_commit(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(status="success", result="done", reason=None, steps=2),
    )
    bus = EventBus()
    database, runtime = await _runtime(tmp_path, runner, bus=bus)
    observed_statuses: list[str] = []
    event_observed = asyncio.Event()

    async def observe(event: Any) -> None:
        if isinstance(event, RunFinishedEvent):
            observed_statuses.append((await runtime.get_run(event.run_id)).status)
            event_observed.set()

    bus.subscribe(observe)
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "finish safely")
        await asyncio.wait_for(started.wait(), timeout=1)
        release.set()

        assert await _wait_terminal(runtime, submitted.run_id) == "succeeded"
        await asyncio.wait_for(event_observed.wait(), timeout=1)
        assert observed_statuses == ["succeeded"]
    finally:
        release.set()
        await runtime.shutdown()
        await database.dispose()


class _CompactionProvider:
    def __init__(self, summary: str = "durable summary") -> None:
        self.summary = summary

    async def chat(self, **kwargs: Any) -> LlmResponse:
        del kwargs
        return LlmResponse(
            stop_reason="end_turn",
            text=self.summary,
            usage=UsageStats(input_tokens=200, output_tokens=20),
        )


# 功能：验证手动压缩成功后仅保留摘要消息为活动上下文，旧消息仍在审计区
# 设计：先完成一轮真实 Runtime 提交，再注入 Fake Provider 调用 compact_session 并检查 active/all 两视图
async def test_manual_compaction_is_transactional_and_auditable(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(
            status="success",
            result="done",
            reason=None,
            messages=[{"role": "assistant", "content": "original answer"}],
        ),
    )
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    runtime = RuntimeService(
        database,
        lambda: runner,  # type: ignore[arg-type,return-value]
        EventBus(),
        artifacts_root=tmp_path / "artifacts",
        compaction_provider_factory=lambda: _CompactionProvider(),
    )
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "original question")
        await asyncio.wait_for(started.wait(), timeout=1)
        release.set()
        assert await _wait_terminal(runtime, submitted.run_id) == "succeeded"

        result = await runtime.compact_session(session.id, focus="keep decisions")

        assert result.summary_text == "durable summary"
        assert [item["content"] for item in await runtime.get_history(session.id)] == [
            "durable summary",
            "Understood, I'll continue from this summary.",
        ]
        async with database.session() as db_session:
            all_messages = list(
                await StateRepository(db_session).list_messages(
                    session.id,
                    committed_only=True,
                    active_only=False,
                )
            )
            records = list(await db_session.scalars(select(CompactionRecord)))
        assert [message.active for message in all_messages] == [False, False, True, True]
        assert records[0].start_sequence == 0
        assert records[0].end_sequence == 1
        assert records[0].summary_message_id == all_messages[2].id

        followup = await runtime.submit_message(session.id, "next question")
        assert await _wait_terminal(runtime, followup.run_id) == "succeeded"
        assert runner.histories[1] == [
            {"role": "user", "content": "durable summary"},
            {
                "role": "assistant",
                "content": "Understood, I'll continue from this summary.",
            },
            {"role": "user", "content": "next question"},
        ]
    finally:
        await runtime.shutdown()
        await database.dispose()


# 功能：验证压缩 Provider 失败时保持原活动上下文与 compactions 表不变
# 设计：Fake Provider 返回空文本触发结构化失败，前后读取历史并断言没有半提交记录
async def test_manual_compaction_failure_preserves_context(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runner = _ControlledRunner(
        started,
        release,
        RunOutcome(
            status="success",
            result="done",
            reason=None,
            messages=[{"role": "assistant", "content": "answer"}],
        ),
    )
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    runtime = RuntimeService(
        database,
        lambda: runner,  # type: ignore[arg-type,return-value]
        EventBus(),
        artifacts_root=tmp_path / "artifacts",
        compaction_provider_factory=lambda: _CompactionProvider(""),
    )
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "question")
        await asyncio.wait_for(started.wait(), timeout=1)
        release.set()
        assert await _wait_terminal(runtime, submitted.run_id) == "succeeded"
        before = await runtime.get_history(session.id)

        with pytest.raises(HandlerError) as captured:
            await runtime.compact_session(session.id)

        assert captured.value.code == COMPACTION_FAILED
        assert await runtime.get_history(session.id) == before
        async with database.session() as db_session:
            assert list(await db_session.scalars(select(CompactionRecord))) == []
    finally:
        await runtime.shutdown()
        await database.dispose()
