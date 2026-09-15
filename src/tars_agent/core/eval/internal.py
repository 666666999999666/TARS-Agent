from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from pydantic import BaseModel, JsonValue

from tars_agent.core.artifacts import ArtifactStore
from tars_agent.core.bus.events import (
    ContextCompactedEvent,
    LlmModelSelectedEvent,
    LlmUsageEvent,
    PermissionDeniedEvent,
    PermissionRequestedEvent,
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
from tars_agent.core.permissions.manager import PermissionManager
from tars_agent.core.permissions.policy import PermissionDecision, ToolPolicy
from tars_agent.core.persistence.request_budget import RequestLedger
from tars_agent.core.runner import AgentRunner, RunOutcome
from tars_agent.core.tools.runtime import DockerRuntime, RuntimeRouter, build_runtime_router

_EVAL_SYSTEM_PROMPT = (
    "You are running a reproducible local evaluation. Work only inside the provided "
    "workspace, use only the exposed tools, and follow the task literally. Do not access "
    "credentials, the network, parent directories, or unrelated files."
)


class _EventAccumulator:
    # 初始化仅包含本次 attempt 的可验证计数器
    def __init__(self) -> None:
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

    # 从 EventBus 事件累积 token、工具耗时和实际执行后端
    async def record(self, event: BaseModel) -> None:
        if isinstance(event, LlmModelSelectedEvent):
            self.model_calls_started += 1
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
def _isolated_process_context(workspace: Path) -> Iterator[None]:
    previous_cwd = Path.cwd()
    previous_home = os.environ.get("TARS_HOME")
    isolated_home = workspace / ".tars-eval-home"
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


class InternalEvalAdapter:
    # 保存真实 AgentRunner 配置和可选价格快照，不注入测试模型
    def __init__(
        self,
        config: TarsConfig,
        *,
        pricing_snapshot: PricingSnapshot | None = None,
    ) -> None:
        self._config = copy.deepcopy(config)
        self._pricing_snapshot = pricing_snapshot

    # 内部套件的任务已完整写在清单中，无需动态发现
    async def prepare_tasks(self, manifest: EvalSuiteManifest) -> list[EvalTaskSpec]:
        if self._config.sandbox.mode != "required":
            from tars_agent.core.eval.runner import AdapterUnavailableError

            raise AdapterUnavailableError("suite requires sandbox_mode=required")
        preflight = await DockerRuntime(self._config.sandbox).preflight()
        if not preflight.available:
            from tars_agent.core.eval.runner import AdapterUnavailableError

            reason = preflight.reason or "sandbox unavailable"
            raise AdapterUnavailableError(f"required sandbox is unavailable: {reason}")
        return list(manifest.tasks)

    # 在临时工作区调用真实 AgentRunner，收集事件、确定性评分并验证清理
    async def run_attempt(
        self,
        task: EvalTaskSpec,
        *,
        repetition: int,
        eval_run_id: str,
    ) -> TaskAttemptResult:
        started_at = utc_now()
        started_clock = asyncio.get_running_loop().time()
        accumulator = _EventAccumulator()
        cleanup = CleanupMetrics()
        usage = UsageMetrics()
        tools = ToolMetrics()
        output = ""
        error_type: str | None = None
        error: str | None = None
        score: float | None = None
        evaluation: dict[str, JsonValue] = {}
        status = EvalStatus.error
        runtime_cleanup_completed: bool | None = None
        execution_started = False
        outcome: RunOutcome | None = None
        runtime: RuntimeRouter | None = None
        manager: PermissionManager | None = None
        agent_run_id = f"{eval_run_id}-{task.id}-r{repetition}"
        requests_before: int | None = None
        ledger = RequestLedger(self._config.llm.request_budget_path)
        workspace = Path(tempfile.mkdtemp(prefix="tars-eval-")).resolve()

        try:
            if self._config.sandbox.mode != "required":
                raise ValueError("internal evaluation requires sandbox.mode=required")
            if not task.tool_whitelist:
                raise ValueError("internal evaluation requires an explicit tool whitelist")
            _write_fixtures(workspace, task)
            if not (self._config.llm.api_key or self._config.llm.anthropic_api_key):
                status = EvalStatus.skipped
                error_type = "missing_credentials"
                error = "No trusted model credential configured; no model call was attempted"
            else:
                manager = PermissionManager(
                    policies={
                        name: ToolPolicy(default=PermissionDecision.ALLOW)
                        for name in task.tool_whitelist
                    },
                    timeout_s=min(task.timeout_s, self._config.permission.timeout_s or 60.0),
                )

                async def deny_unapproved_request(event: BaseModel) -> None:
                    # The manifest authorizes listed tools inside the required sandbox.
                    # Any remaining request (outside-root or host fallback) is outside that grant.
                    if isinstance(event, PermissionRequestedEvent):
                        assert manager is not None
                        manager.respond(event.request_id, event.session_id, "deny_once")

                runtime = build_runtime_router(self._config.sandbox)
                preflight = await runtime.preflight()
                if not preflight.available:
                    raise RuntimeError("required sandbox unavailable before model execution")
                bus = EventBus()
                bus.subscribe(accumulator.record)
                bus.subscribe(deny_unapproved_request)
                runner = AgentRunner(
                    self._config,
                    bus=bus,
                    permission_manager=manager,
                    tool_runtime=runtime,
                )
                requests_before = ledger.counts()["real"]
                execution_started = True
                with _isolated_process_context(workspace):
                    outcome = await runner.run_and_capture(
                        task.goal,
                        run_id=agent_run_id,
                        system_prompt_override=_EVAL_SYSTEM_PROMPT,
                        tool_whitelist=task.tool_whitelist,
                        workspace_root=workspace,
                        session_id=agent_run_id,
                        history=[{"role": "user", "content": task.goal}],
                        artifact_store=ArtifactStore(workspace / ".tars-eval-runs"),
                        session_notes="",
                    )
                if outcome.status == "cancelled":
                    raise asyncio.CancelledError()
                output = outcome.result
                grade_passed, evaluation = _grade(task, workspace, output)
                status = _status_from_outcome(outcome, grade_passed)
                score = 1.0 if status == EvalStatus.passed else 0.0
                if outcome.status != "success":
                    error_type = outcome.reason or "agent_failed"
                    error = f"AgentRunner finished with status={outcome.status}"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status = EvalStatus.error
            error_type = "internal_adapter_error"
            error = str(exc)
        finally:
            try:
                if manager is not None:
                    manager.cancel_session(agent_run_id, reason="eval_finished")
                if runtime is not None:
                    await runtime.cleanup()
                    # cleanup() returns no independently verified resource-removal result.
                    # Record the attempt, but keep cleanup confirmation unknown.
                    evaluation["runtime_cleanup_attempted"] = True
            except Exception as exc:
                runtime_cleanup_completed = False
                cleanup.error = str(exc)
                if status == EvalStatus.passed:
                    status = EvalStatus.error
                    score = 0.0
                    error_type = "runtime_cleanup_failed"
                    error = "Runtime cleanup failed after task execution"
            if execution_started:
                observed_usage = accumulator.usage()
                if requests_before is not None:
                    try:
                        request_count = ledger.counts()["real"] - requests_before
                        evaluation["model_request_reservations"] = request_count
                        if request_count > accumulator.usage_events:
                            observed_usage = UsageMetrics()
                    except Exception:
                        observed_usage = UsageMetrics()
                        evaluation["model_request_reservations"] = None
                usage = _apply_pricing(
                    observed_usage,
                    self._config.llm.default_model,
                    self._pricing_snapshot,
                )
                tools = accumulator.tools()
            cleanup.runtime_cleanup_completed = runtime_cleanup_completed
            try:
                if workspace.resolve(strict=False) != workspace or workspace.is_symlink():
                    raise OSError(
                        "Temporary workspace ownership changed; refusing recursive removal"
                    )
                if workspace.exists():
                    shutil.rmtree(workspace)
                cleanup.workspace_removed = not workspace.exists()
            except OSError as exc:
                cleanup.workspace_removed = False
                cleanup.error = str(exc)
                if status == EvalStatus.passed:
                    status = EvalStatus.error
                    score = 0.0
                    error_type = "workspace_cleanup_failed"
                    error = "Temporary evaluation workspace could not be removed"

        elapsed = max(0, int((asyncio.get_running_loop().time() - started_clock) * 1000))
        return TaskAttemptResult(
            task_id=task.id,
            source_task_id=task.source_task_id,
            repetition=repetition,
            status=status,
            run_terminal_status=(outcome.status if outcome is not None else "not_started"),
            goal_completed=(status == EvalStatus.passed if outcome is not None else None),
            collateral_damage=None,
            step_count=(outcome.steps if outcome is not None else None),
            started_at=started_at,
            finished_at=utc_now(),
            latency_ms=elapsed,
            score=score,
            output=output,
            error_type=error_type,
            error=error,
            usage=usage,
            tools=tools,
            sandbox=SandboxMetrics(
                requested=(True if execution_started else None),
                network_mode=(
                    "disabled" if "workspace_sandbox" in accumulator.backends else "not_observed"
                ),
                observed_backends=sorted(accumulator.backends),
                isolated_workspace=True,
                fallback_used=("host" in accumulator.backends if accumulator.backends else None),
                oom_killed=None,
                timed_out=(bool(accumulator.tool_timeouts) if accumulator.calls_started else None),
                cleanup_completed=runtime_cleanup_completed,
            ),
            cleanup=cleanup,
            evaluation=evaluation,
        )


__all__ = ["InternalEvalAdapter"]
