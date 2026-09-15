from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from tars_agent.core.observability import RunMetricsProjection, TokenCostRates
from tars_agent.core.persistence import (
    Database,
    RunRecord,
    SessionRecord,
    StateRepository,
    ToolInvocationRecord,
)


async def _database_with_metrics_fixture(tmp_path: Path) -> Database:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    started = datetime(2026, 8, 24, 1, 2, 3, tzinfo=UTC)
    finished = started + timedelta(seconds=4, milliseconds=250)
    async with database.transaction() as db_session:
        repository = StateRepository(db_session)
        await repository.add_session(
            SessionRecord(
                id="sess-metrics",
                mode="chat",
                status="ready",
                title="",
                workspace_root=str(tmp_path),
            )
        )
        await repository.add_run(
            RunRecord(
                id="run-parent",
                session_id="sess-metrics",
                kind="chat",
                attempt=1,
                status="succeeded",
                result={"text": "SECRET-OUTPUT", "steps": 2},
                started_at=started,
                finished_at=finished,
            )
        )
        await repository.add_run(
            RunRecord(
                id="run-child",
                session_id="sess-metrics",
                parent_run_id="run-parent",
                kind="subagent",
                attempt=1,
                status="succeeded",
                started_at=started + timedelta(milliseconds=500),
                finished_at=started + timedelta(seconds=1, milliseconds=500),
            )
        )
        await repository.add_run(
            RunRecord(
                id="run-grandchild",
                session_id="sess-metrics",
                parent_run_id="run-child",
                kind="subagent",
                attempt=1,
                status="succeeded",
                started_at=started + timedelta(milliseconds=750),
                finished_at=started + timedelta(seconds=1, milliseconds=250),
            )
        )
        await repository.add_tool_invocation(
            ToolInvocationRecord(
                id="run-parent:tool-1",
                run_id="run-parent",
                tool_name="read_file",
                parameters={"path": "SECRET-PATH"},
                backend="workspace_sandbox",
                status="succeeded",
                result={"content": "SECRET-CONTENT"},
                started_at=started,
                finished_at=started + timedelta(milliseconds=250),
            )
        )
        await repository.add_tool_invocation(
            ToolInvocationRecord(
                id="run-parent:tool-2",
                run_id="run-parent",
                tool_name="bash",
                parameters={"command": "SECRET-COMMAND"},
                backend="workspace_sandbox",
                status="failed",
                error_class="sandbox_lost",
                started_at=started + timedelta(seconds=1),
                finished_at=started + timedelta(seconds=1, milliseconds=100),
            )
        )
        await repository.add_tool_invocation(
            ToolInvocationRecord(
                id="run-parent:tool-3",
                run_id="run-parent",
                tool_name="write_file",
                parameters={"path": "fallback.txt"},
                backend="host",
                status="succeeded",
            )
        )
        await repository.add_tool_invocation(
            ToolInvocationRecord(
                id="run-parent:tool-4",
                run_id="run-parent",
                tool_name="write_file",
                parameters={"path": "denied.txt"},
                backend="workspace_sandbox",
                status="failed",
                error_class="permission_denied",
            )
        )
        await repository.add_tool_invocation(
            ToolInvocationRecord(
                id="run-parent:tool-5",
                run_id="run-parent",
                tool_name="bash",
                parameters={"command": "allocate"},
                backend="workspace_sandbox",
                status="failed",
                error_class="sandbox_oom",
                started_at=started + timedelta(seconds=2),
                finished_at=started + timedelta(seconds=2, milliseconds=50),
            )
        )
        await repository.add_tool_invocation(
            ToolInvocationRecord(
                id="run-parent:tool-6",
                run_id="run-parent",
                tool_name="bash",
                parameters={"command": "wait"},
                backend="workspace_sandbox",
                status="failed",
                error_class="timeout",
                started_at=started + timedelta(seconds=3),
                finished_at=started + timedelta(seconds=3, milliseconds=75),
            )
        )
        await repository.add_tool_invocation(
            ToolInvocationRecord(
                id="run-parent:mcp-1",
                run_id="run-parent",
                tool_name="SECRET-SERVER__SECRET-TOOL",
                parameters={"secret": "SECRET-MCP-PARAM"},
                backend="external",
                status="succeeded",
                result={"content": "SECRET-MCP-OUTPUT"},
                started_at=started + timedelta(seconds=3, milliseconds=500),
                finished_at=started + timedelta(seconds=3, milliseconds=625),
            )
        )
        events = [
            (
                "run.started",
                {
                    "type": "run.started",
                    "run_id": "run-parent",
                    "ts": "2026-08-24T01:02:03Z",
                },
            ),
            (
                "step.started",
                {
                    "type": "step.started",
                    "run_id": "run-parent",
                    "step": 1,
                    "ts": "2026-08-24T01:02:03.100Z",
                },
            ),
            (
                "llm.token",
                {
                    "type": "llm.token",
                    "run_id": "run-parent",
                    "token": "SECRET-TOKEN",
                    "ts": "2026-08-24T01:02:03.250Z",
                },
            ),
            (
                "llm.usage",
                {
                    "type": "llm.usage",
                    "run_id": "run-parent",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "cache_creation_input_tokens": 4,
                    "ts": "2026-08-24T01:02:03.600Z",
                },
            ),
            (
                "tool.call_started",
                {
                    "type": "tool.call_started",
                    "run_id": "run-parent",
                    "tool_use_id": "tool-span-1",
                    "tool_name": "SECRET-TOOL",
                    "params": {"path": "SECRET-PATH"},
                    "ts": "2026-08-24T01:02:03.800Z",
                },
            ),
            (
                "tool.call_finished",
                {
                    "type": "tool.call_finished",
                    "run_id": "run-parent",
                    "tool_use_id": "tool-span-1",
                    "tool_name": "SECRET-TOOL",
                    "output": "SECRET-OUTPUT",
                    "elapsed_ms": 100,
                    "ts": "2026-08-24T01:02:03.900Z",
                },
            ),
            (
                "step.finished",
                {
                    "type": "step.finished",
                    "run_id": "run-parent",
                    "step": 1,
                    "ts": "2026-08-24T01:02:03.700Z",
                },
            ),
            (
                "step.started",
                {
                    "type": "step.started",
                    "run_id": "run-parent",
                    "step": 2,
                    "ts": "2026-08-24T01:02:04Z",
                },
            ),
            (
                "llm.token",
                {
                    "type": "llm.token",
                    "run_id": "run-parent",
                    "token": "SECOND-SECRET-TOKEN",
                    "ts": "2026-08-24T01:02:04.200Z",
                },
            ),
            (
                "llm.usage",
                {
                    "type": "llm.usage",
                    "run_id": "run-parent",
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 1,
                    "ts": "2026-08-24T01:02:04.700Z",
                },
            ),
            (
                "permission.requested",
                {
                    "type": "permission.requested",
                    "run_id": "run-parent",
                    "request_id": "perm-tool",
                    "request_kind": "tool",
                    "params": {"secret": "SECRET-PARAM"},
                    "ts": "2026-08-24T01:02:05Z",
                },
            ),
            (
                "tool.call_started",
                {
                    "type": "tool.call_started",
                    "run_id": "run-parent",
                    "tool_use_id": "tool-span-2",
                    "tool_name": "SECRET-FAILED-TOOL",
                    "params": {"command": "SECRET-COMMAND"},
                    "ts": "2026-08-24T01:02:04.800Z",
                },
            ),
            (
                "tool.call_failed",
                {
                    "type": "tool.call_failed",
                    "run_id": "run-parent",
                    "tool_use_id": "tool-span-2",
                    "tool_name": "SECRET-FAILED-TOOL",
                    "error_message": "SECRET-ERROR",
                    "error_class": "timeout",
                    "elapsed_ms": 100,
                    "ts": "2026-08-24T01:02:04.900Z",
                },
            ),
            (
                "permission.granted",
                {
                    "type": "permission.granted",
                    "run_id": "run-parent",
                    "request_id": "perm-tool",
                    "ts": "2026-08-24T01:02:05.400Z",
                },
            ),
            (
                "permission.requested",
                {
                    "type": "permission.requested",
                    "run_id": "run-parent",
                    "request_id": "perm-host",
                    "request_kind": "host_fallback",
                    "ts": "2026-08-24T01:02:06Z",
                },
            ),
            (
                "permission.granted",
                {
                    "type": "permission.granted",
                    "run_id": "run-parent",
                    "request_id": "perm-host",
                    "ts": "2026-08-24T01:02:06.900Z",
                },
            ),
            (
                "permission.requested",
                {
                    "type": "permission.requested",
                    "run_id": "run-parent",
                    "request_id": "perm-denied",
                    "request_kind": "tool",
                    "ts": "2026-08-24T01:02:07Z",
                },
            ),
            (
                "permission.denied",
                {
                    "type": "permission.denied",
                    "run_id": "run-parent",
                    "request_id": "perm-denied",
                    "ts": "2026-08-24T01:02:07.200Z",
                },
            ),
            (
                "subagent.started",
                {
                    "type": "subagent.started",
                    "run_id": "run-child",
                    "parent_run_id": "run-parent",
                    "ts": "2026-08-24T01:02:03.500Z",
                },
            ),
            (
                "subagent.finished",
                {
                    "type": "subagent.finished",
                    "run_id": "run-child",
                    "parent_run_id": "run-parent",
                    "status": "success",
                    "ts": "2026-08-24T01:02:04.500Z",
                },
            ),
            (
                "subagent.started",
                {
                    "type": "subagent.started",
                    "run_id": "run-grandchild",
                    "parent_run_id": "run-child",
                    "description": "SECRET-DESCRIPTION",
                    "ts": "2026-08-24T01:02:03.750Z",
                },
            ),
            (
                "subagent.finished",
                {
                    "type": "subagent.finished",
                    "run_id": "run-grandchild",
                    "parent_run_id": "run-child",
                    "status": "success",
                    "ts": "2026-08-24T01:02:04.250Z",
                },
            ),
            (
                "context.compacted",
                {
                    "type": "context.compacted",
                    "run_id": "run-parent",
                    "session_id": "sess-metrics",
                    "original_tokens": 1000,
                    "summary_tokens": 100,
                    "ts": "2026-08-24T01:02:07.100Z",
                },
            ),
            (
                "run.finished",
                {
                    "type": "run.finished",
                    "run_id": "run-parent",
                    "status": "success",
                    "ts": "2026-08-24T01:02:07.250Z",
                },
            ),
        ]
        for event_type, payload in events:
            event_run_id = payload.get("run_id")
            await repository.append_event(
                event_type=event_type,
                payload=payload,
                session_id="sess-metrics",
                run_id=str(event_run_id) if event_run_id is not None else None,
            )
    return database


# 功能：验证 RunMetricsProjection 聚合 Run、Tool 与 Durable Event 的完整只读快照
# 设计：在临时 SQLite 中覆盖 token、权限、sandbox、host fallback、subagent 和 cleanup 边界
async def test_run_metrics_projection_aggregates_runtime_evidence(tmp_path: Path) -> None:
    database = await _database_with_metrics_fixture(tmp_path)
    try:
        metrics = await RunMetricsProjection(
            database,
            cost_rates=TokenCostRates(
                input_usd_per_million=3.0,
                output_usd_per_million=15.0,
                cache_read_usd_per_million=0.3,
                cache_creation_usd_per_million=3.75,
            ),
        ).project("run-parent")

        assert metrics is not None
        assert metrics.status == "succeeded"
        assert metrics.event_terminal_status == "success"
        assert metrics.duration_ms == 4_250
        assert metrics.steps == 2
        assert metrics.tokens.input_tokens == 150
        assert metrics.tokens.output_tokens == 30
        assert metrics.tokens.total_tokens == 180
        assert metrics.model.calls == 2
        assert metrics.model.first_token_latency_ms_average == 175
        assert metrics.model.first_token_latency_ms_max == 200
        assert metrics.model.completion_latency_ms_average == 600
        assert metrics.model.completion_latency_ms_max == 700
        assert metrics.cost.estimated_usd is not None
        assert metrics.cost.source == "configured_token_rates"
        assert metrics.tools.total == 7
        assert metrics.tools.succeeded == 3
        assert metrics.tools.failed == 4
        assert metrics.tools.rejected == 1
        assert metrics.tools.elapsed_ms == 600
        assert metrics.permissions.requested == 3
        assert metrics.permissions.granted == 2
        assert metrics.permissions.denied == 1
        assert metrics.permissions.tool_denied == 1
        assert metrics.permissions.host_fallback_requested == 1
        assert metrics.permissions.wait_ms_total == 1_500
        assert metrics.permissions.wait_ms_average == 500
        assert metrics.permissions.wait_ms_max == 900
        assert metrics.sandbox.invocations == 5
        assert metrics.sandbox.executions == 4
        assert metrics.sandbox.host_fallback_executions == 1
        assert metrics.sandbox.host_fallback_requests == 1
        assert metrics.sandbox.oom_failures == 1
        assert metrics.sandbox.timeout_failures == 1
        assert metrics.sandbox.cleanup_failures is None
        assert metrics.sandbox.failure_reasons == {
            "permission_denied": 1,
            "sandbox_lost": 1,
            "sandbox_oom": 1,
            "timeout": 1,
        }
        assert metrics.subagents.direct_children == 1
        assert metrics.subagents.descendants == 2
        assert metrics.subagents.started == 2
        assert metrics.subagents.succeeded == 2
        assert metrics.subagents.active == 0
        assert metrics.subagents.max_depth == 2
        assert metrics.subagents.completed_duration_ms_total == 1_500
        assert metrics.subagents.completed_duration_ms_max == 1_000
        assert metrics.cleanup.state == "unconfirmed"
        assert metrics.cleanup.sandbox_cleanup_expected is True
        assert metrics.cleanup.failure_count is None
        assert metrics.cleanup.orphan_free_confirmed is None
        span_names = [span.name for span in metrics.spans]
        assert span_names.count("run") == 1
        assert span_names.count("llm.step") == 2
        assert span_names.count("tool.invoke") == 2
        assert span_names.count("permission.wait") == 3
        assert span_names.count("subagent.run") == 2
        assert span_names.count("compact") == 1
        assert span_names.count("mcp.call") == 1
        root_span = metrics.spans[0]
        assert root_span.incomplete is False
        assert root_span.started_at == datetime(2026, 8, 24, 1, 2, 3, tzinfo=UTC)
        compact_span = next(span for span in metrics.spans if span.name == "compact")
        assert compact_span.started_at is None
        assert compact_span.incomplete is True
        subagent_spans = [span for span in metrics.spans if span.name == "subagent.run"]
        assert any(span.parent_span_id == "run" for span in subagent_spans)
        assert any(
            span.parent_span_id is not None
            and span.parent_span_id.startswith("subagent:")
            for span in subagent_spans
        )
        rendered = repr(metrics)
        assert "SECRET" not in rendered
    finally:
        await database.dispose()


# 功能：验证投影读取不存在的 Run 时不创建记录或伪造零值指标
# 设计：空临时数据库前后都只执行只读查询，并以 None 表达不存在
async def test_run_metrics_projection_returns_none_for_unknown_run(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    try:
        assert await RunMetricsProjection(database).project("missing") is None
    finally:
        await database.dispose()


# 功能：验证恢复得到的 interrupted Run 不会被误报为已完成清理
# 设计：只写终态数据库记录而不写 run.finished 事件，模拟 daemon 重启恢复后的证据缺口
async def test_cleanup_is_unconfirmed_without_finished_event(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    async with database.transaction() as db_session:
        repository = StateRepository(db_session)
        await repository.add_session(
            SessionRecord(
                id="sess-interrupted",
                mode="chat",
                status="ready",
                title="",
                workspace_root=str(tmp_path),
            )
        )
        await repository.add_run(
            RunRecord(
                id="run-interrupted",
                session_id="sess-interrupted",
                kind="chat",
                attempt=1,
                status="interrupted",
                reason="daemon_restarted",
            )
        )
    try:
        metrics = await RunMetricsProjection(database).project("run-interrupted")

        assert metrics is not None
        assert metrics.cleanup.state == "unconfirmed"
        assert metrics.duration_ms is None
        assert metrics.event_terminal_status is None
        assert metrics.cost.estimated_usd is None
        assert metrics.permissions.wait_ms_total is None
    finally:
        await database.dispose()
