from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from tars_agent.core.persistence import (
    Database,
    EventRecord,
    RunRecord,
    StateRepository,
    ToolInvocationRecord,
)

_EVENT_PAGE_SIZE = 2_000
_TERMINAL_DB_RUN_STATUSES = frozenset(
    {"succeeded", "failed", "cancelled", "interrupted"}
)
type RuntimeSpanName = Literal[
    "run",
    "llm.step",
    "tool.invoke",
    "permission.wait",
    "subagent.run",
    "compact",
    "mcp.call",
]


@dataclass(frozen=True, slots=True)
class RuntimeSpan:
    span_id: str
    parent_span_id: str | None
    name: RuntimeSpanName
    started_at: datetime | None
    finished_at: datetime | None
    incomplete: bool


@dataclass(frozen=True, slots=True)
class TokenMetrics:
    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    calls: int
    first_token_latency_ms_average: int | None
    first_token_latency_ms_max: int | None
    completion_latency_ms_average: int | None
    completion_latency_ms_max: int | None


@dataclass(frozen=True, slots=True)
class TokenCostRates:
    input_usd_per_million: float
    output_usd_per_million: float
    cache_read_usd_per_million: float = 0.0
    cache_creation_usd_per_million: float = 0.0


@dataclass(frozen=True, slots=True)
class CostMetrics:
    estimated_usd: float | None
    source: Literal["configured_token_rates", "unconfigured"]


@dataclass(frozen=True, slots=True)
class ToolMetrics:
    total: int
    succeeded: int
    failed: int
    active: int
    rejected: int
    elapsed_ms: int
    by_backend: dict[str, int]


@dataclass(frozen=True, slots=True)
class PermissionMetrics:
    requested: int
    granted: int
    denied: int
    host_fallback_requested: int
    tool_denied: int
    host_fallback_denied: int
    wait_ms_total: int | None
    wait_ms_average: int | None
    wait_ms_max: int | None


@dataclass(frozen=True, slots=True)
class SandboxMetrics:
    invocations: int
    executions: int
    succeeded: int
    failed: int
    host_fallback_executions: int
    host_fallback_requests: int
    oom_failures: int
    timeout_failures: int
    cleanup_failures: int | None
    failure_reasons: dict[str, int]


@dataclass(frozen=True, slots=True)
class SubagentMetrics:
    direct_children: int
    descendants: int
    started: int
    succeeded: int
    failed: int
    active: int
    max_depth: int | None
    completed_duration_ms_total: int | None
    completed_duration_ms_max: int | None


@dataclass(frozen=True, slots=True)
class CleanupMetrics:
    state: Literal["pending", "unconfirmed"]
    run_finished_event: bool
    sandbox_cleanup_expected: bool
    failure_count: int | None = None
    orphan_free_confirmed: bool | None = None


@dataclass(frozen=True, slots=True)
class RunMetrics:
    run_id: str
    session_id: str
    kind: str
    status: str
    event_terminal_status: str | None
    reason: str | None
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: int | None
    steps: int | None
    tokens: TokenMetrics
    model: ModelMetrics
    cost: CostMetrics
    tools: ToolMetrics
    permissions: PermissionMetrics
    sandbox: SandboxMetrics
    subagents: SubagentMetrics
    cleanup: CleanupMetrics
    spans: tuple[RuntimeSpan, ...]


class RunMetricsProjection:
    """只读聚合已有 Runtime 表与 Durable Event，不创建第二套状态真相源。"""

    def __init__(
        self,
        database: Database,
        *,
        cost_rates: TokenCostRates | None = None,
    ) -> None:
        self._database = database
        self._cost_rates = cost_rates

    async def project(self, run_id: str) -> RunMetrics | None:
        async with self._database.session() as db_session:
            repository = StateRepository(db_session)
            run = await repository.get_run(run_id)
            if run is None:
                return None
            invocations = list(await repository.list_tool_invocations(run_id))
            session_runs = list(await repository.list_runs(run.session_id, limit=10_000))
            events = await self._session_events(repository, run.session_id)

        direct_events = [
            record
            for record in events
            if _payload_run_id(record.payload) == run_id or record.run_id == run_id
        ]
        event_terminal_status = _event_terminal_status(direct_events)
        tokens = _token_metrics(direct_events)
        permissions = _permission_metrics(direct_events)
        return RunMetrics(
            run_id=run.id,
            session_id=run.session_id,
            kind=run.kind,
            status=run.status,
            event_terminal_status=event_terminal_status,
            reason=run.reason,
            started_at=run.started_at,
            finished_at=run.finished_at,
            duration_ms=_duration_ms(run.started_at, run.finished_at),
            steps=_step_count(run.result, direct_events),
            tokens=tokens,
            model=_model_metrics(direct_events),
            cost=_cost_metrics(tokens, self._cost_rates),
            tools=_tool_metrics(invocations),
            permissions=permissions,
            sandbox=_sandbox_metrics(invocations, permissions),
            subagents=_subagent_metrics(run_id, session_runs),
            cleanup=_cleanup_metrics(
                run.status,
                run_finished_event=event_terminal_status is not None,
                sandbox_cleanup_expected=any(
                    invocation.backend == "workspace_sandbox"
                    and invocation.started_at is not None
                    for invocation in invocations
                ),
            ),
            spans=_runtime_spans(
                run_id,
                direct_events,
                events,
                session_runs,
                invocations,
            ),
        )

    @staticmethod
    async def _session_events(
        repository: StateRepository,
        session_id: str,
    ) -> list[EventRecord]:
        records: list[EventRecord] = []
        after_cursor = 0
        while True:
            page = list(
                await repository.list_events(
                    after_cursor=after_cursor,
                    session_id=session_id,
                    limit=_EVENT_PAGE_SIZE,
                )
            )
            records.extend(page)
            if len(page) < _EVENT_PAGE_SIZE:
                return records
            after_cursor = page[-1].cursor


def _duration_ms(started_at: datetime | None, finished_at: datetime | None) -> int | None:
    if started_at is None or finished_at is None:
        return None
    return max(0, int((finished_at - started_at).total_seconds() * 1_000))


def _step_count(result: dict[str, Any] | None, events: list[EventRecord]) -> int | None:
    if result is not None:
        steps = result.get("steps")
        if isinstance(steps, int) and not isinstance(steps, bool) and steps >= 0:
            return steps
    seen = [
        value
        for record in events
        if record.event_type in {"step.started", "step.finished"}
        if isinstance((value := record.payload.get("step")), int)
        and not isinstance(value, bool)
        and value >= 0
    ]
    return max(seen, default=None)


def _token_metrics(events: list[EventRecord]) -> TokenMetrics:
    totals: Counter[str] = Counter()
    for record in events:
        if record.event_type != "llm.usage":
            continue
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = record.payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] += value
    return TokenMetrics(
        input_tokens=totals.get("input_tokens"),
        output_tokens=totals.get("output_tokens"),
        cache_read_input_tokens=totals.get("cache_read_input_tokens"),
        cache_creation_input_tokens=totals.get("cache_creation_input_tokens"),
    )


def _model_metrics(events: list[EventRecord]) -> ModelMetrics:
    first_token_latencies: list[int] = []
    completion_latencies: list[int] = []
    ordered = sorted(events, key=lambda record: record.cursor)
    step_indexes = [
        index for index, record in enumerate(ordered) if record.event_type == "step.started"
    ]
    for position, start_index in enumerate(step_indexes):
        end_index = (
            step_indexes[position + 1] if position + 1 < len(step_indexes) else len(ordered)
        )
        started_at = _event_time(ordered[start_index])
        if started_at is None:
            continue
        window = ordered[start_index + 1 : end_index]
        first_token = next(
            (_event_time(record) for record in window if record.event_type == "llm.token"),
            None,
        )
        usage = next(
            (_event_time(record) for record in window if record.event_type == "llm.usage"),
            None,
        )
        first_token_ms = _duration_ms(started_at, first_token)
        completion_ms = _duration_ms(started_at, usage)
        if first_token_ms is not None:
            first_token_latencies.append(first_token_ms)
        if completion_ms is not None:
            completion_latencies.append(completion_ms)
    calls = sum(record.event_type == "llm.usage" for record in events)
    return ModelMetrics(
        calls=calls,
        first_token_latency_ms_average=_average(first_token_latencies),
        first_token_latency_ms_max=max(first_token_latencies, default=None),
        completion_latency_ms_average=_average(completion_latencies),
        completion_latency_ms_max=max(completion_latencies, default=None),
    )


def _cost_metrics(tokens: TokenMetrics, rates: TokenCostRates | None) -> CostMetrics:
    if rates is None:
        return CostMetrics(estimated_usd=None, source="unconfigured")
    if (tokens.input_tokens is None or tokens.output_tokens is None
            or tokens.cache_read_input_tokens is None
            or tokens.cache_creation_input_tokens is None):
        return CostMetrics(estimated_usd=None, source="configured_token_rates")
    total = (
        tokens.input_tokens * rates.input_usd_per_million
        + tokens.output_tokens * rates.output_usd_per_million
        + tokens.cache_read_input_tokens * rates.cache_read_usd_per_million
        + tokens.cache_creation_input_tokens * rates.cache_creation_usd_per_million
    ) / 1_000_000
    return CostMetrics(estimated_usd=round(total, 8), source="configured_token_rates")


def _tool_metrics(invocations: list[ToolInvocationRecord]) -> ToolMetrics:
    status_counts = Counter(str(invocation.status) for invocation in invocations)
    elapsed_ms = sum(
        _duration_ms(invocation.started_at, invocation.finished_at) or 0
        for invocation in invocations
    )
    return ToolMetrics(
        total=len(invocations),
        succeeded=status_counts["succeeded"],
        failed=status_counts["failed"],
        active=status_counts["queued"] + status_counts["running"],
        rejected=sum(
            invocation.status == "failed"
            and invocation.started_at is None
            and invocation.error_class
            in {
                "schema_error",
                "permission_denied",
                "sandbox_policy_denied",
                "host_fallback_denied",
            }
            for invocation in invocations
        ),
        elapsed_ms=elapsed_ms,
        by_backend=dict(Counter(str(item.backend) for item in invocations)),
    )


def _permission_metrics(events: list[EventRecord]) -> PermissionMetrics:
    requested = [record for record in events if record.event_type == "permission.requested"]
    requests_by_id = {
        request_id: record
        for record in requested
        if (request_id := _optional_string(record.payload.get("request_id"))) is not None
    }
    decisions = [
        record
        for record in events
        if record.event_type in {"permission.granted", "permission.denied"}
    ]
    waits: list[int] = []
    for decision in decisions:
        request_id = _optional_string(decision.payload.get("request_id"))
        request = requests_by_id.get(request_id or "")
        if request is None:
            continue
        wait_ms = _duration_ms(_event_time(request), _event_time(decision))
        if wait_ms is not None:
            waits.append(wait_ms)
    denied = [record for record in decisions if record.event_type == "permission.denied"]
    return PermissionMetrics(
        requested=len(requested),
        granted=sum(record.event_type == "permission.granted" for record in decisions),
        denied=len(denied),
        host_fallback_requested=sum(
            record.payload.get("request_kind") == "host_fallback" for record in requested
        ),
        tool_denied=_permission_decisions_of_kind(denied, requests_by_id, "tool"),
        host_fallback_denied=_permission_decisions_of_kind(
            denied,
            requests_by_id,
            "host_fallback",
        ),
        wait_ms_total=sum(waits) if waits else None,
        wait_ms_average=_average(waits),
        wait_ms_max=max(waits, default=None),
    )


def _sandbox_metrics(
    invocations: list[ToolInvocationRecord],
    permissions: PermissionMetrics,
) -> SandboxMetrics:
    sandbox = [item for item in invocations if item.backend == "workspace_sandbox"]
    failure_reasons = Counter(
        str(item.error_class)
        for item in sandbox
        if item.status == "failed" and item.error_class
    )
    return SandboxMetrics(
        invocations=len(sandbox),
        executions=sum(item.started_at is not None for item in sandbox),
        succeeded=sum(item.status == "succeeded" for item in sandbox),
        failed=sum(item.status == "failed" for item in sandbox),
        host_fallback_executions=sum(item.backend == "host" for item in invocations),
        host_fallback_requests=permissions.host_fallback_requested,
        oom_failures=sum(item.error_class == "sandbox_oom" for item in sandbox),
        timeout_failures=sum(item.error_class == "timeout" for item in sandbox),
        cleanup_failures=None,
        failure_reasons=dict(failure_reasons),
    )


def _subagent_metrics(run_id: str, session_runs: list[RunRecord]) -> SubagentMetrics:
    children_by_parent: dict[str, list[RunRecord]] = {}
    for run in session_runs:
        if run.parent_run_id is not None:
            children_by_parent.setdefault(run.parent_run_id, []).append(run)
    descendants: list[tuple[RunRecord, int]] = []
    frontier: list[tuple[str, int]] = [(run_id, 0)]
    seen = {run_id}
    while frontier:
        parent_id, parent_depth = frontier.pop()
        for child in children_by_parent.get(parent_id, []):
            if child.id in seen:
                continue
            seen.add(child.id)
            depth = parent_depth + 1
            descendants.append((child, depth))
            frontier.append((child.id, depth))
    durations = [
        duration
        for child, _depth in descendants
        if (duration := _duration_ms(child.started_at, child.finished_at)) is not None
    ]
    return SubagentMetrics(
        direct_children=len(children_by_parent.get(run_id, [])),
        descendants=len(descendants),
        started=sum(
            child.started_at is not None or child.status != "queued"
            for child, _depth in descendants
        ),
        succeeded=sum(child.status == "succeeded" for child, _depth in descendants),
        failed=sum(child.status == "failed" for child, _depth in descendants),
        active=sum(
            child.status in {"queued", "running"} for child, _depth in descendants
        ),
        max_depth=max((depth for _child, depth in descendants), default=None),
        completed_duration_ms_total=sum(durations) if durations else None,
        completed_duration_ms_max=max(durations, default=None),
    )


def _cleanup_metrics(
    status: str,
    *,
    run_finished_event: bool,
    sandbox_cleanup_expected: bool,
) -> CleanupMetrics:
    if status not in _TERMINAL_DB_RUN_STATUSES:
        state: Literal["pending", "unconfirmed"] = "pending"
    else:
        state = "unconfirmed"
    return CleanupMetrics(
        state=state,
        run_finished_event=run_finished_event,
        sandbox_cleanup_expected=sandbox_cleanup_expected,
    )


def _runtime_spans(
    run_id: str,
    direct_events: list[EventRecord],
    session_events: list[EventRecord],
    session_runs: list[RunRecord],
    invocations: list[ToolInvocationRecord],
) -> tuple[RuntimeSpan, ...]:
    root_started = _first_event(direct_events, "run.started")
    root_finished = _last_event(direct_events, {"run.finished"})
    spans: list[RuntimeSpan] = [
        _runtime_span(
            span_id="run",
            parent_span_id=None,
            name="run",
            started=root_started,
            finished=root_finished,
        )
    ]
    spans.extend(
        _paired_runtime_spans(
            direct_events,
            start_type="step.started",
            finish_types={"step.finished"},
            key="step",
            span_prefix="llm-step",
            name="llm.step",
        )
    )
    spans.extend(
        _paired_runtime_spans(
            direct_events,
            start_type="tool.call_started",
            finish_types={"tool.call_finished", "tool.call_failed"},
            key="tool_use_id",
            span_prefix="tool",
            name="tool.invoke",
        )
    )
    spans.extend(
        _paired_runtime_spans(
            direct_events,
            start_type="permission.requested",
            finish_types={"permission.granted", "permission.denied"},
            key="request_id",
            span_prefix="permission",
            name="permission.wait",
        )
    )
    spans.extend(_subagent_runtime_spans(run_id, session_events, session_runs))
    for record in direct_events:
        if record.event_type == "context.compacted":
            spans.append(
                _runtime_span(
                    span_id=f"compact:{record.cursor}",
                    parent_span_id="run",
                    name="compact",
                    started=None,
                    finished=record,
                )
            )
    mcp_invocations = sorted(
        (
            invocation
            for invocation in invocations
            if invocation.backend == "external" and invocation.started_at is not None
        ),
        key=lambda invocation: (invocation.created_at, invocation.id),
    )
    for index, invocation in enumerate(mcp_invocations, start=1):
        spans.append(
            RuntimeSpan(
                span_id=f"mcp:{index}",
                parent_span_id="run",
                name="mcp.call",
                started_at=_as_utc(invocation.started_at),
                finished_at=_as_utc(invocation.finished_at),
                incomplete=invocation.finished_at is None,
            )
        )
    # external 目前只由 McpTool 使用；不把普通 tool event 或名称前缀猜成 MCP 边界。
    return tuple(spans)


def _paired_runtime_spans(
    events: list[EventRecord],
    *,
    start_type: str,
    finish_types: set[str],
    key: str,
    span_prefix: str,
    name: RuntimeSpanName,
) -> list[RuntimeSpan]:
    pending: dict[str, list[EventRecord]] = {}
    pairs: list[tuple[EventRecord | None, EventRecord | None]] = []
    for record in sorted(events, key=lambda item: item.cursor):
        pairing_key = _optional_string(record.payload.get(key))
        if pairing_key is None:
            continue
        if record.event_type == start_type:
            pending.setdefault(pairing_key, []).append(record)
        elif record.event_type in finish_types:
            starts = pending.get(pairing_key, [])
            started = starts.pop(0) if starts else None
            pairs.append((started, record))
    for starts in pending.values():
        pairs.extend((started, None) for started in starts)
    return [
        _runtime_span(
            span_id=f"{span_prefix}:{_pair_cursor(started, finished)}",
            parent_span_id="run",
            name=name,
            started=started,
            finished=finished,
        )
        for started, finished in pairs
        if started is not None or finished is not None
    ]


def _subagent_runtime_spans(
    run_id: str,
    events: list[EventRecord],
    session_runs: list[RunRecord],
) -> list[RuntimeSpan]:
    descendants = _descendant_run_ids(run_id, session_runs)
    relevant = [
        record
        for record in events
        if record.event_type in {"subagent.started", "subagent.finished"}
        and _payload_run_id(record.payload) in descendants
    ]
    pending: dict[str, list[EventRecord]] = {}
    pairs: list[tuple[str, EventRecord | None, EventRecord | None]] = []
    for record in sorted(relevant, key=lambda item: item.cursor):
        child_id = _payload_run_id(record.payload)
        if child_id is None:
            continue
        if record.event_type == "subagent.started":
            pending.setdefault(child_id, []).append(record)
        else:
            starts = pending.get(child_id, [])
            started = starts.pop(0) if starts else None
            pairs.append((child_id, started, record))
    for child_id, starts in pending.items():
        pairs.extend((child_id, started, None) for started in starts)

    span_ids = {
        child_id: f"subagent:{_pair_cursor(started, finished)}"
        for child_id, started, finished in pairs
        if started is not None or finished is not None
    }
    result: list[RuntimeSpan] = []
    for child_id, started, finished in pairs:
        evidence = started or finished
        if evidence is None:
            continue
        parent_run_id = _optional_string(evidence.payload.get("parent_run_id"))
        parent_span_id = span_ids.get(parent_run_id or "", "run")
        result.append(
            _runtime_span(
                span_id=span_ids[child_id],
                parent_span_id=parent_span_id,
                name="subagent.run",
                started=started,
                finished=finished,
            )
        )
    return result


def _descendant_run_ids(run_id: str, session_runs: list[RunRecord]) -> set[str]:
    children: dict[str, list[str]] = {}
    for run in session_runs:
        if run.parent_run_id is not None:
            children.setdefault(run.parent_run_id, []).append(run.id)
    descendants: set[str] = set()
    frontier = [run_id]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, []):
            if child not in descendants:
                descendants.add(child)
                frontier.append(child)
    return descendants


def _runtime_span(
    *,
    span_id: str,
    parent_span_id: str | None,
    name: RuntimeSpanName,
    started: EventRecord | None,
    finished: EventRecord | None,
) -> RuntimeSpan:
    return RuntimeSpan(
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name,
        started_at=_event_time(started) if started is not None else None,
        finished_at=_event_time(finished) if finished is not None else None,
        incomplete=started is None or finished is None,
    )


def _pair_cursor(
    started: EventRecord | None,
    finished: EventRecord | None,
) -> int:
    evidence = started if started is not None else finished
    if evidence is None:
        raise ValueError("span pair has no durable event evidence")
    return evidence.cursor


def _first_event(events: list[EventRecord], event_type: str) -> EventRecord | None:
    matching = [record for record in events if record.event_type == event_type]
    return min(matching, key=lambda record: record.cursor, default=None)


def _last_event(
    events: list[EventRecord], event_types: set[str]
) -> EventRecord | None:
    matching = [record for record in events if record.event_type in event_types]
    return max(matching, key=lambda record: record.cursor, default=None)


def _event_terminal_status(events: list[EventRecord]) -> str | None:
    terminal = [record for record in events if record.event_type == "run.finished"]
    if not terminal:
        return None
    return _optional_string(max(terminal, key=lambda record: record.cursor).payload.get("status"))


def _permission_decisions_of_kind(
    decisions: list[EventRecord],
    requests_by_id: dict[str, EventRecord],
    kind: str,
) -> int:
    count = 0
    for decision in decisions:
        request_id = _optional_string(decision.payload.get("request_id"))
        request = requests_by_id.get(request_id or "")
        if request is not None and request.payload.get("request_kind") == kind:
            count += 1
    return count


def _event_time(record: EventRecord) -> datetime | None:
    value = record.payload.get("ts")
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _average(values: list[int]) -> int | None:
    return round(sum(values) / len(values)) if values else None


def _payload_run_id(payload: dict[str, Any]) -> str | None:
    return _optional_string(payload.get("run_id"))


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
