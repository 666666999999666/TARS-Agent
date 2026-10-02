from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel, JsonValue

from tars_agent.core.bus.events import (
    ContextCompactedEvent,
    LlmModelSelectedEvent,
    LlmUsageEvent,
    PermissionDeniedEvent,
    PermissionRequestedEvent,
    RunFinishedEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    ToolExecutionStartedEvent,
)
from tars_agent.core.config import TarsConfig
from tars_agent.core.eval.models import (
    CleanupMetrics,
    EvalStatus,
    EvalSuiteManifest,
    EvalTaskSpec,
    ModelPrice,
    PricingSnapshot,
    SandboxMetrics,
    TaskAttemptResult,
    ToolMetrics,
    UsageMetrics,
)
from tars_agent.core.eval.runner import utc_now
from tars_agent.core.events.bus import EventBus
from tars_agent.core.events.durable import DurableEventHub
from tars_agent.core.permissions.manager import PermissionManager
from tars_agent.core.permissions.policy import PermissionDecision, ToolPolicy
from tars_agent.core.persistence import Database
from tars_agent.core.persistence.request_budget import RequestLedger
from tars_agent.core.runner import AgentRunner, RunOutcome
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.skills.loader import SkillLoader
from tars_agent.core.subagent.registry import BackgroundTaskRegistry
from tars_agent.core.tools.runtime import DockerRuntime, RuntimeRouter, build_runtime_router

_EVAL_SYSTEM_PROMPT = (
    "You are running a reproducible local evaluation. Work only inside the provided "
    "workspace, use only the exposed tools, and follow the task literally. Do not access "
    "credentials, the network, parent directories, or unrelated files."
)
_COORDINATION_TOOLS = {
    "spawn_agent", "agent_result", "agent_cancel", "task_create", "task_update", "task_list",
}
_TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


@dataclass
class _SpawnStage:
    parent_run_id: str
    tool_use_id: str
    role: str
    requested: int
    child_run_id: str | None = None
    started: int | None = None
    finished: int | None = None
    call_succeeded: bool | None = None
    child_status: str | None = None


class _EventAccumulator:
    # 初始化仅包含本次 attempt 的可验证计数器
    def __init__(self, *, track_runs: bool = True) -> None:
        self._track_runs = track_runs
        self.by_run: dict[str, _EventAccumulator] = {}
        self.models: set[str] = set()
        self.spawn_roles: list[str] = []
        self.usage_events = 0
        self.model_calls_started = 0
        self.compaction_seen = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0
        self.calls_started = 0
        self.calls_succeeded = 0
        self.calls_failed = 0
        self.elapsed_ms = 0
        self.refusals = 0
        self.tool_timeouts = 0
        self.tool_names: set[str] = set()
        self.backends: set[str] = set()
        self.spawn_stages: list[_SpawnStage] = []
        self._stage_calls: dict[tuple[str, str], _SpawnStage] = {}
        self._stage_children: dict[str, _SpawnStage] = {}
        self._sequence = 0
        self._run_finished: dict[str, int] = {}

    # 从 EventBus 事件累积 token、工具耗时和实际执行后端
    async def record(self, event: BaseModel) -> None:
        if self._track_runs:
            self._record_stage(event)
        run_id = getattr(event, "run_id", None)
        if self._track_runs and isinstance(run_id, str):
            if run_id not in self.by_run:
                self.by_run[run_id] = _EventAccumulator(track_runs=False)
            child = self.by_run[run_id]
            await child.record(event)
        if isinstance(event, LlmModelSelectedEvent):
            self.model_calls_started += 1
            self.models.add(event.model)
        elif isinstance(event, ContextCompactedEvent):
            self.compaction_seen = True
        elif isinstance(event, LlmUsageEvent):
            self.usage_events += 1
            self.input_tokens += event.input_tokens
            self.output_tokens += event.output_tokens
            self.cache_read_input_tokens += event.cache_read_input_tokens
            self.cache_creation_input_tokens += event.cache_creation_input_tokens
        elif isinstance(event, ToolCallStartedEvent):
            self.calls_started += 1
            self.tool_names.add(event.tool_name)
            if event.tool_name == "spawn_agent":
                self.spawn_roles.append(str(event.params.get("subagent_type", "")))
        elif isinstance(event, ToolCallFinishedEvent):
            self.calls_succeeded += 1
            self.elapsed_ms += event.elapsed_ms
            self.tool_names.add(event.tool_name)
        elif isinstance(event, ToolCallFailedEvent):
            self.calls_failed += 1
            self.elapsed_ms += event.elapsed_ms
            self.tool_names.add(event.tool_name)
            if event.error_class == "timeout":
                self.tool_timeouts += 1
        elif isinstance(event, ToolExecutionStartedEvent):
            self.backends.add(event.backend)
        elif isinstance(event, PermissionDeniedEvent):
            self.refusals += 1

    def _record_stage(self, event: BaseModel) -> None:
        self._sequence += 1
        stage: _SpawnStage | None
        if isinstance(event, ToolCallStartedEvent) and event.tool_name == "spawn_agent":
            stage = _SpawnStage(
                event.run_id, event.tool_use_id, str(event.params.get("subagent_type", "")),
                self._sequence,
            )
            self.spawn_stages.append(stage)
            self._stage_calls[(event.run_id, event.tool_use_id)] = stage
        elif isinstance(event, (ToolCallFinishedEvent, ToolCallFailedEvent)):
            stage = self._stage_calls.get((event.run_id, event.tool_use_id))
            if stage is not None:
                stage.call_succeeded = isinstance(event, ToolCallFinishedEvent)
        elif isinstance(event, SubagentStartedEvent):
            candidates = [stage for stage in self.spawn_stages
                          if stage.parent_run_id == event.parent_run_id
                          and stage.child_run_id is None and stage.call_succeeded is None]
            if len(candidates) == 1:
                stage = candidates[0]
                stage.child_run_id = event.run_id
                stage.started = self._sequence
                self._stage_children[event.run_id] = stage
        elif isinstance(event, SubagentFinishedEvent):
            stage = self._stage_children.get(event.run_id)
            if stage is not None and stage.parent_run_id == event.parent_run_id:
                stage.finished = self._sequence
                stage.child_status = event.status
        elif isinstance(event, RunFinishedEvent):
            self._run_finished[event.run_id] = self._sequence

    def workflow_followed(self, root_run_id: str) -> bool:
        stages = [stage for stage in self.spawn_stages if stage.parent_run_id == root_run_id]
        if [stage.role for stage in stages] != ["planner", "executor", "reviewer"]:
            return False
        if any(stage.call_succeeded is not True or stage.child_status != "success"
               or stage.child_run_id is None or stage.started is None or stage.finished is None
               for stage in stages):
            return False
        if any(cast(int, prior.finished) >= following.requested
               for prior, following in zip(stages, stages[1:])):
            return False
        return cast(int, stages[-1].finished) < self._run_finished.get(root_run_id, 0)

    # 缺失或不完整的 usage 不能用已观测子集冒充完整消耗。
    def usage(self) -> UsageMetrics:
        if (not self.usage_events or self.usage_events < self.model_calls_started
                or self.compaction_seen):
            return UsageMetrics()
        return UsageMetrics(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
        )

    # 生成本次运行的工具调用统计
    def tools(self) -> ToolMetrics:
        return ToolMetrics(
            calls_started=self.calls_started,
            calls_succeeded=self.calls_succeeded,
            calls_failed=self.calls_failed,
            elapsed_ms=self.elapsed_ms,
            errors=self.calls_failed,
            refusals=self.refusals,
            tool_names=sorted(self.tool_names),
        )


# 暂时切换到隔离目录并隔离 TARS_HOME，结束时恢复进程状态
@contextmanager
def _isolated_process_context(
    workspace: Path, *, isolated_home: Path | None = None,
) -> Iterator[None]:
    previous_cwd = Path.cwd()
    previous_home = os.environ.get("TARS_HOME")
    isolated_home = isolated_home or workspace / ".tars-eval-home"
    isolated_home.mkdir(parents=True, exist_ok=True)
    os.environ["TARS_HOME"] = str(isolated_home)
    os.chdir(workspace)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if previous_home is None:
            os.environ.pop("TARS_HOME", None)
        else:
            os.environ["TARS_HOME"] = previous_home


# 将清单内相对路径约束到临时工作区，拒绝绝对路径和目录穿越
def _workspace_path(workspace: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"absolute workspace path is forbidden: {relative}")
    candidate = (workspace / raw).resolve(strict=False)
    root = workspace.resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"workspace path escapes isolated root: {relative}")
    return candidate


# 在隔离工作区创建清单声明的初始文件
def _write_fixtures(workspace: Path, task: EvalTaskSpec) -> None:
    for relative, content in task.fixture_files.items():
        target = _workspace_path(workspace, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


# 对 Agent 最终输出或工作区文件执行确定性评分
def _grade(task: EvalTaskSpec, workspace: Path, output: str) -> tuple[bool, dict[str, JsonValue]]:
    grader = task.grader
    if grader.kind == "output_contains":
        expected = str(grader.expected)
        matched = expected in output
        return matched, {"kind": grader.kind, "matched": matched, "expected": expected}
    if grader.kind == "file_equals":
        assert grader.path is not None
        target = _workspace_path(workspace, grader.path)
        actual = target.read_text(encoding="utf-8") if target.is_file() else None
        matched = actual == grader.expected
        return matched, {"kind": grader.kind, "matched": matched, "path": grader.path}
    if grader.kind == "json_equals":
        assert grader.path is not None
        target = _workspace_path(workspace, grader.path)
        if not target.is_file():
            return False, {"kind": grader.kind, "matched": False, "path": grader.path}
        try:
            actual_json: JsonValue = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False, {
                "kind": grader.kind,
                "matched": False,
                "path": grader.path,
                "reason": "invalid_or_unreadable_json",
            }
        matched = actual_json == grader.expected
        return matched, {"kind": grader.kind, "matched": matched, "path": grader.path}
    if grader.kind == "files_equal":
        mismatches: list[str] = []
        for relative, expected in grader.files.items():
            target = _workspace_path(workspace, relative)
            actual = target.read_text(encoding="utf-8") if target.is_file() else None
            if actual != expected:
                mismatches.append(relative)
        return not mismatches, {
            "kind": grader.kind,
            "matched": not mismatches,
            "mismatched_paths": cast(JsonValue, mismatches),
        }
    raise ValueError(f"internal adapter does not support grader kind {grader.kind!r}")


# 按带日期的价格快照计算估算成本；任一已用费率缺失时不推算
def _apply_pricing(
    usage: UsageMetrics,
    model: str,
    snapshot: PricingSnapshot | None,
) -> UsageMetrics:
    if snapshot is None:
        return usage
    price: ModelPrice | None = snapshot.models.get(model)
    if price is None:
        return usage
    if any(value is None for value in (
        usage.input_tokens, usage.output_tokens,
        usage.cache_read_input_tokens, usage.cache_creation_input_tokens,
    )):
        return usage
    input_tokens = cast(int, usage.input_tokens)
    output_tokens = cast(int, usage.output_tokens)
    cache_read = cast(int, usage.cache_read_input_tokens)
    cache_creation = cast(int, usage.cache_creation_input_tokens)
    if cache_read and price.cache_read_per_million is None:
        return usage
    if cache_creation and price.cache_creation_per_million is None:
        return usage
    cost = (
        input_tokens * price.input_per_million
        + output_tokens * price.output_per_million
        + cache_read * (price.cache_read_per_million or 0)
        + cache_creation * (price.cache_creation_per_million or 0)
    ) / 1_000_000
    return usage.model_copy(
        update={
            "estimated_cost": round(cost, 10),
            "currency": snapshot.currency,
            "pricing_effective_date": snapshot.effective_date,
        }
    )


# 将 AgentRunner 终态映射为评测状态，区分任务失败与模型基础设施错误
def _status_from_outcome(outcome: RunOutcome, grade_passed: bool) -> EvalStatus:
    if outcome.status == "success" and not outcome.reason:
        return EvalStatus.passed if grade_passed else EvalStatus.failed
    if outcome.reason and outcome.reason.startswith("llm_"):
        return EvalStatus.error
    return EvalStatus.failed


def _prepare_skill(task: EvalTaskSpec, workspace: Path) -> tuple[list[str], str]:
    base_tools = list(dict.fromkeys(task.tool_whitelist))
    if set(base_tools) & _COORDINATION_TOOLS:
        raise ValueError("task.tool_whitelist must contain business tools, not coordination tools")
    # Fixtures cannot override the fixed Skill or role definitions of the comparison.
    if (workspace / ".tars").exists():
        raise ValueError("agent task fixtures must not contain .tars configuration")
    template = "$ARGUMENTS"
    allowed = base_tools
    if task.agent_mode == "orchestrated":
        skill = SkillLoader(workspace_root=workspace).resolve("orchestrate")
        if skill is None:
            raise ValueError("builtin orchestrate Skill is unavailable")
        template = skill.system_prompt_template
        allowed = base_tools + sorted(_COORDINATION_TOOLS)
    body = _EVAL_SYSTEM_PROMPT + "\n\n" + template
    target = workspace / ".tars" / "skills" / "eval-task.md"
    target.parent.mkdir(parents=True)
    target.write_text(
        "---\nname: eval-task\nallowed_tools:\n"
        + "".join(f"  - {name}\n" for name in allowed)
        + "---\n" + body + "\n", encoding="utf-8",
    )
    return allowed, hashlib.sha256(body.encode()).hexdigest()


async def _wait_for_tree(service: RuntimeService, registry: BackgroundTaskRegistry,
                         run_id: str) -> None:
    await service.supervisor.wait(run_id)
    # A child can create a grandchild while this batch is being awaited.
    while tasks := [task for task, _ in registry.all() if not task.done()]:
        await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)


async def _tree_records(database: Database, session_id: str) -> list[dict[str, JsonValue]]:
    from sqlalchemy import select

    from tars_agent.core.persistence import RunRecord

    async with database.session() as sql:
        records = (await sql.scalars(
            select(RunRecord).where(RunRecord.session_id == session_id)
        )).all()
        return [{
            "run_id": row.id, "parent_run_id": row.parent_run_id, "kind": row.kind,
            "status": row.status, "reason": row.reason,
        } for row in records]


def _copy_evidence(source: Path, destination: Path) -> dict[str, JsonValue]:
    """Copy closed attempt files without following agent-created links or junctions."""
    if source.is_symlink() or source.resolve() != source:
        raise OSError("temporary attempt root changed ownership")
    if destination.resolve().is_relative_to(source):
        raise OSError("evidence destination must be outside the temporary attempt")
    inventory: dict[str, JsonValue] = {}
    skipped: list[str] = []
    for current, directories, files in os.walk(source, followlinks=False):
        parent = Path(current)
        for name in list(directories):
            path = parent / name
            if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(source):
                directories.remove(name)
                skipped.append(path.relative_to(source).as_posix())
        for name in files:
            path = parent / name
            relative = path.relative_to(source)
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                skipped.append(relative.as_posix())
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            with target.open("rb") as stream:
                inventory[relative.as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"files_sha256": inventory, "skipped_links": cast(JsonValue, skipped)}


def _write_attempt_record(destination: Path, result: TaskAttemptResult) -> None:
    target = destination / "attempt.json"
    temporary = destination / f".attempt-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(result.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _record_write_failed(result: TaskAttemptResult, error: OSError) -> None:
    result.evaluation["attempt_record_write_error"] = str(error)
    result.evaluation["evidence_finalization"] = "failed"
    result.status = EvalStatus.error
    result.score = 0.0
    result.goal_completed = False if result.goal_completed is not None else None
    if result.error_type != "evidence_copy_failed":
        result.error_type = "evidence_write_failed"
        result.error = str(error)


class InternalEvalAdapter:
    def __init__(
        self, config: TarsConfig, *, pricing_snapshot: PricingSnapshot | None = None,
        artifact_root: Path | None = None,
    ) -> None:
        self._config = copy.deepcopy(config)
        self._pricing_snapshot = pricing_snapshot
        self._artifact_root = (
            artifact_root or Path.cwd() / "build" / "eval-evidence"
        ).expanduser().resolve()

    async def prepare_tasks(self, manifest: EvalSuiteManifest) -> list[EvalTaskSpec]:
        from tars_agent.core.eval.runner import AdapterUnavailableError

        if self._config.sandbox.mode != "required":
            raise AdapterUnavailableError("suite requires sandbox_mode=required")
        preflight = await DockerRuntime(self._config.sandbox).preflight()
        if not preflight.available:
            raise AdapterUnavailableError(
                f"required sandbox is unavailable: {preflight.reason or 'sandbox unavailable'}"
            )
        return list(manifest.tasks)

    async def run_attempt(
        self, task: EvalTaskSpec, *, repetition: int, eval_run_id: str,
    ) -> TaskAttemptResult:
        started_at = utc_now()
        clock = asyncio.get_running_loop()
        started_clock = clock.time()
        accumulator = _EventAccumulator()
        cleanup = CleanupMetrics()
        usage = UsageMetrics()
        status = EvalStatus.error
        error_type: str | None = None
        error: str | None = None
        output = ""
        collateral_damage: bool | None = None
        outcome: RunOutcome | None = None
        runtime: RuntimeRouter | None = None
        service: RuntimeService | None = None
        registry: BackgroundTaskRegistry | None = None
        database: Database | None = None
        hub: DurableEventHub | None = None
        manager: PermissionManager | None = None
        run_id: str | None = None
        session_id: str | None = None
        requests_before: int | None = None
        ledger = RequestLedger(self._config.llm.request_budget_path)
        cancelled = False
        timed_out = False
        execution_started = False
        records: list[dict[str, JsonValue]] = []
        protected_paths: list[str] = []
        temporary = Path(tempfile.mkdtemp(prefix="tars-eval-")).resolve()
        workspace = temporary / "workspace"
        workspace.mkdir()
        state = temporary / "state"
        state.mkdir()
        suite_key = hashlib.sha256(eval_run_id.encode()).hexdigest()[:12]
        attempt_key = hashlib.sha256(task.id.encode()).hexdigest()[:12]
        evidence = (
            self._artifact_root / suite_key / f"{attempt_key}-r{repetition}-{uuid.uuid4().hex}"
        )
        evidence.mkdir(parents=True, exist_ok=False)
        evaluation: dict[str, JsonValue] = {
            "agent_mode": task.agent_mode, "execution_path": "runtime_service",
            "evidence_directory": str(evidence), "split": task.metadata.get("split"),
            "business_tool_whitelist": cast(JsonValue, list(task.tool_whitelist)),
        }
        try:
            with _isolated_process_context(workspace, isolated_home=state / ".tars-eval-home"):
                try:
                    if self._config.sandbox.mode != "required":
                        raise ValueError("internal evaluation requires sandbox.mode=required")
                    if not task.tool_whitelist:
                        raise ValueError("internal evaluation requires an explicit tool whitelist")
                    _write_fixtures(workspace, task)
                    protected = task.metadata.get("protected_paths", [])
                    if not isinstance(protected, list) or any(
                        not isinstance(path, str) or path not in task.fixture_files
                        for path in protected
                    ):
                        raise ValueError("protected_paths must name declared fixture files")
                    protected_paths = cast(list[str], protected)
                    allowed, prompt_hash = _prepare_skill(task, workspace)
                    evaluation["effective_tool_whitelist"] = cast(JsonValue, allowed)
                    evaluation["skill_template_sha256"] = prompt_hash
                    if not (self._config.llm.api_key or self._config.llm.anthropic_api_key):
                        status = EvalStatus.skipped
                        error_type = "missing_credentials"
                        error = (
                            "No trusted model credential configured; no model call was attempted"
                        )
                    else:
                        manager = PermissionManager(
                            policies={name: ToolPolicy(default=PermissionDecision.ALLOW)
                                      for name in allowed},
                            timeout_s=min(task.timeout_s, self._config.permission.timeout_s or 60),
                        )

                        async def deny_unapproved_request(event: BaseModel) -> None:
                            if isinstance(event, PermissionRequestedEvent):
                                assert manager is not None
                                manager.respond(event.request_id, event.session_id, "deny_once")

                        runtime = build_runtime_router(self._config.sandbox)
                        if not (await runtime.preflight()).available:
                            raise RuntimeError(
                                "required sandbox unavailable before model execution"
                            )
                        database = Database(state / "state.db")
                        await database.create_schema()
                        bus = EventBus()
                        bus.subscribe(accumulator.record)
                        bus.subscribe(deny_unapproved_request)
                        hub = DurableEventHub(database)
                        await hub.start()
                        bus.subscribe(hub.handle)
                        registry = BackgroundTaskRegistry(database, bus)
                        service = RuntimeService(
                            database,
                            lambda: AgentRunner(
                                self._config, bus=bus, permission_manager=manager,
                                tool_runtime=runtime, task_registry=registry,
                            ),
                            bus, artifacts_root=state / "artifacts", subagent_registry=registry,
                            tool_runtime=runtime, llm_config=self._config.llm,
                        )
                        session = await service.create_session("chat", workspace_root=workspace)
                        session_id = session.id
                        requests_before = ledger.counts()["real"]
                        execution_started = True
                        # The outer guard leaves room for cancellation and evidence saving.
                        async with asyncio.timeout(task.timeout_s):
                            submission = await service.submit_message(
                                session_id, f"/eval-task {task.goal}",
                                client_message_id=f"eval-{uuid.uuid4().hex}",
                            )
                            run_id = submission.run_id
                            await _wait_for_tree(service, registry, run_id)
                        snapshot = await service.get_run(run_id)
                        result = snapshot.result or {}
                        outcome = RunOutcome(
                            status="success" if snapshot.status == "succeeded" else snapshot.status,
                            result=str(result.get("text", "")), reason=snapshot.reason,
                            steps=int(result.get("steps", 0)),
                        )
                        output = outcome.result
                        # This is only a candidate status; grading happens after
                        # the complete run tree and physical cleanup are verified.
                        status = _status_from_outcome(outcome, True)
                        if outcome.status != "success" or outcome.reason:
                            error_type = outcome.reason or "agent_failed"
                            error = f"RuntimeService finished with status={snapshot.status}"
                except TimeoutError:
                    timed_out = True
                    status, error_type = EvalStatus.error, "timeout"
                    error = f"task exceeded timeout_s={task.timeout_s:g}"
                except asyncio.CancelledError:
                    cancelled = True
                    status, error_type = EvalStatus.error, "cancelled"
                    error = "evaluation cancelled"
                except Exception as exc:
                    status, error_type, error = EvalStatus.error, "internal_adapter_error", str(exc)
                finally:
                    async def close_execution() -> None:
                        nonlocal error, error_type, status, records, outcome, output, run_id
                        failures: list[str] = []
                        if manager is not None and session_id is not None:
                            manager.cancel_session(session_id, reason="eval_finished")
                        if service is not None:
                            try:
                                await service.shutdown()
                            except Exception as exc:
                                cleanup.runtime_cleanup_completed = False
                                failures.append(str(exc))
                        if runtime is not None:
                            evaluation["runtime_cleanup_attempted"] = True
                            try:
                                await runtime.cleanup()
                                if runtime.pending_cleanup_run_ids():
                                    raise RuntimeError("sandbox resources still require cleanup")
                                confirmation = await runtime.confirm_cleanup()
                                evaluation["runtime_cleanup_confirmation"] = {
                                    "confirmed": confirmation.confirmed,
                                    "source": confirmation.source,
                                    "scope_id": confirmation.scope_id,
                                    "remaining_resource_ids": list(
                                        confirmation.remaining_resource_ids,
                                    ),
                                    "reason": confirmation.reason,
                                }
                                if not failures:
                                    cleanup.runtime_cleanup_completed = confirmation.confirmed
                                if confirmation.confirmed is not True:
                                    failures.append(
                                        "physical sandbox cleanup unconfirmed: "
                                        + (confirmation.reason or confirmation.source)
                                    )
                            except Exception as exc:
                                cleanup.runtime_cleanup_completed = False
                                failures.append(str(exc))
                        if hub is not None:
                            try:
                                await hub.stop()
                            except Exception as exc:
                                cleanup.runtime_cleanup_completed = False
                                failures.append(str(exc))
                        if database is not None:
                            try:
                                if session_id is not None:
                                    records = await _tree_records(database, session_id)
                                if run_id is None:
                                    run_id = next((str(row["run_id"]) for row in records
                                                   if row["parent_run_id"] is None), None)
                                if outcome is None and run_id is not None and service is not None:
                                    snapshot = await service.get_run(run_id)
                                    result = snapshot.result or {}
                                    outcome = RunOutcome(
                                        status=snapshot.status, result=str(result.get("text", "")),
                                        reason=snapshot.reason, steps=int(result.get("steps", 0)),
                                    )
                                    output = outcome.result
                            finally:
                                await database.dispose()
                        if failures:
                            cleanup.error = "; ".join(failures)
                            if status == EvalStatus.passed:
                                status, error_type = EvalStatus.error, "runtime_cleanup_failed"
                                error = cleanup.error

                    closing = asyncio.create_task(close_execution())
                    while not closing.done():
                        try:
                            await asyncio.shield(closing)
                        except asyncio.CancelledError:
                            cancelled = True
                    closing.result()
        except Exception as exc:
            status, error_type, error = EvalStatus.error, "runtime_cleanup_failed", str(exc)
            cleanup.runtime_cleanup_completed = False
            cleanup.error = str(exc)

        evaluation["parent_run_id"] = run_id
        evaluation["runs"] = cast(JsonValue, records)
        children = [row for row in records if row["run_id"] != run_id]
        terminal = bool(records) and all(row["status"] in _TERMINAL for row in records)
        if registry is not None:
            terminal = terminal and registry.active_count == 0 and not any(
                registry.cleanup_is_pending(str(row["run_id"])) for row in records
            )
        evaluation["tree_terminal"] = terminal
        evaluation["subagent_count"] = len(children)
        evaluation["models_observed"] = cast(JsonValue, sorted(accumulator.models))
        evaluation["observed_model_calls"] = accumulator.model_calls_started
        evaluation["spawn_roles"] = cast(JsonValue, accumulator.spawn_roles)
        evaluation["workflow_stages"] = cast(
            JsonValue, [asdict(stage) for stage in accumulator.spawn_stages],
        )
        evaluation["workflow_validation_source"] = "tool_results_and_subagent_lifecycle_order"
        evaluation["usage_by_run"] = {
            key: cast(JsonValue, value.usage().model_dump(mode="json"))
            for key, value in accumulator.by_run.items()
        }
        evaluation["observed_tokens"] = {
            "input": accumulator.input_tokens, "output": accumulator.output_tokens,
            "cache_read": accumulator.cache_read_input_tokens,
            "cache_creation": accumulator.cache_creation_input_tokens,
        }
        if execution_started:
            usage = accumulator.usage()
            try:
                request_count = ledger.counts()["real"] - cast(int, requests_before)
                evaluation["model_request_reservations"] = request_count
                if request_count > accumulator.usage_events:
                    usage = UsageMetrics()
            except Exception:
                evaluation["model_request_reservations"] = None
                usage = UsageMetrics()
            same_model = accumulator.models <= {self._config.llm.default_model}
            if not same_model:
                usage = UsageMetrics()
                if status == EvalStatus.passed:
                    status, error_type, error = EvalStatus.error, "model_mismatch", "mixed models"
            usage = _apply_pricing(usage, self._config.llm.default_model, self._pricing_snapshot)
            workflow = accumulator.workflow_followed(run_id or "")
            evaluation["workflow_followed"] = (
                workflow if task.agent_mode == "orchestrated" else None
            )
            if status == EvalStatus.passed and (
                not terminal or any(row["status"] != "succeeded" for row in children)
                or (task.agent_mode == "orchestrated" and (not children or not workflow))
            ):
                status, error_type = EvalStatus.failed, "run_tree_incomplete"
                error = "the required workflow or a descendant did not complete successfully"
        evaluation["grading_performed"] = False
        if (outcome is not None and terminal and cleanup.runtime_cleanup_completed is True
                and status in {EvalStatus.passed, EvalStatus.failed}):
            try:
                grade_passed, grade = _grade(task, workspace, output)
                evaluation["grader"] = grade
                evaluation["grading_performed"] = True
                if protected_paths:
                    changed = []
                    for relative in protected_paths:
                        path = _workspace_path(workspace, relative)
                        if (not path.is_file() or path.read_text(encoding="utf-8")
                                != task.fixture_files[relative]):
                            changed.append(relative)
                    collateral_damage = bool(changed)
                    evaluation["changed_protected_paths"] = cast(JsonValue, changed)
                    grade_passed = grade_passed and not collateral_damage
                if status == EvalStatus.passed and not grade_passed:
                    status = EvalStatus.failed
            except Exception as exc:
                status, error_type, error = EvalStatus.error, "grading_failed", str(exc)
        copied = False
        try:
            evaluation["evidence_inventory"] = _copy_evidence(temporary, evidence)
            (evidence / "task.json").write_text(task.model_dump_json(indent=2), encoding="utf-8")
            copied = True
        except OSError as exc:
            status, error_type, error = EvalStatus.error, "evidence_copy_failed", str(exc)
        cleanup.workspace_removed = False
        evaluation["retained_temporary_root"] = str(temporary)
        result_record = TaskAttemptResult(
            task_id=task.id, source_task_id=task.source_task_id, repetition=repetition,
            status=status, run_terminal_status=outcome.status if outcome else "not_started",
            goal_completed=status == EvalStatus.passed if outcome else None,
            collateral_damage=collateral_damage,
            step_count=outcome.steps if outcome else None,
            started_at=started_at, finished_at=utc_now(),
            latency_ms=max(0, int((clock.time() - started_clock) * 1000)),
            score=None if status == EvalStatus.skipped else float(status == EvalStatus.passed),
            output=output, error_type=error_type, error=error, usage=usage,
            tools=accumulator.tools() if execution_started else ToolMetrics(),
            sandbox=SandboxMetrics(
                requested=True if execution_started else None,
                network_mode=(
                    "disabled" if "workspace_sandbox" in accumulator.backends else "not_observed"
                ),
                observed_backends=sorted(accumulator.backends), isolated_workspace=True,
                fallback_used="host" in accumulator.backends if accumulator.backends else None,
                timed_out=(
                    timed_out or bool(accumulator.tool_timeouts)
                ) if execution_started else None,
                cleanup_completed=cleanup.runtime_cleanup_completed,
            ),
            cleanup=cleanup, evaluation=evaluation,
        )
        # Save a valid, explicitly unfinished manifest before deleting any original
        # files. Atomic replacement preserves it if the final write later fails.
        pending = result_record.model_copy(deep=True)
        pending.status = EvalStatus.error
        pending.score = 0.0
        pending.goal_completed = False if outcome else None
        pending.error_type = "evidence_finalization_pending"
        pending.error = "attempt evidence is not finalized"
        pending.evaluation["evidence_finalization"] = "pending"
        try:
            _write_attempt_record(evidence, pending)
        except OSError as exc:
            _record_write_failed(result_record, exc)
            if cancelled:
                raise asyncio.CancelledError() from exc
            return result_record
        if copied and (not execution_started or cleanup.runtime_cleanup_completed is True):
            try:
                if temporary.is_symlink() or temporary.resolve() != temporary:
                    raise OSError("temporary attempt root changed ownership")
                shutil.rmtree(temporary)
                cleanup.workspace_removed = not temporary.exists()
                evaluation.pop("retained_temporary_root", None)
            except OSError as exc:
                cleanup.workspace_removed = False
                cleanup.error = str(exc)
                if status == EvalStatus.passed:
                    status, error_type = EvalStatus.error, "workspace_cleanup_failed"
                    error = str(exc)
        evaluation["evidence_finalization"] = "complete" if copied else "copy_failed"
        result_record.status = status
        result_record.error_type = error_type
        result_record.error = error
        result_record.score = (
            None if status == EvalStatus.skipped else float(status == EvalStatus.passed)
        )
        result_record.goal_completed = status == EvalStatus.passed if outcome else None
        result_record.cleanup = cleanup
        result_record.evaluation = evaluation
        result_record.finished_at = utc_now()
        result_record.latency_ms = max(0, int((clock.time() - started_clock) * 1000))
        try:
            _write_attempt_record(evidence, result_record)
        except OSError as exc:
            _record_write_failed(result_record, exc)
        if cancelled:
            raise asyncio.CancelledError()
        return result_record


__all__ = ["InternalEvalAdapter"]
