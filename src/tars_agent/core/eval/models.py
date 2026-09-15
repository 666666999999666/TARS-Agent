from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, model_validator


class EvalStatus(StrEnum):
    passed = "passed"
    failed = "failed"
    skipped = "skipped"
    error = "error"


class GraderSpec(BaseModel):
    kind: Literal[
        "output_contains",
        "file_equals",
        "json_equals",
        "files_equal",
        "pytest",
    ]
    path: str | None = None
    expected: JsonValue = None
    files: dict[str, str] = Field(default_factory=dict)

    # 校验评分器所需字段，避免运行结束后才发现清单不完整
    @model_validator(mode="after")
    def validate_required_fields(self) -> GraderSpec:
        if self.kind in {"file_equals", "json_equals"} and not self.path:
            raise ValueError(f"grader kind {self.kind!r} requires path")
        if self.kind == "file_equals" and not isinstance(self.expected, str):
            raise ValueError("grader kind 'file_equals' requires a string expected value")
        if self.kind == "files_equal" and not self.files:
            raise ValueError("grader kind 'files_equal' requires files")
        if self.kind == "output_contains" and not isinstance(self.expected, str):
            raise ValueError("grader kind 'output_contains' requires a string expected value")
        return self


class EvalTaskSpec(BaseModel):
    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    timeout_s: float = Field(default=180.0, gt=0)
    repetitions: int | None = Field(default=None, ge=1)
    tool_whitelist: list[str] = Field(default_factory=list)
    fixture_files: dict[str, str] = Field(default_factory=dict)
    grader: GraderSpec
    source_task_id: str | None = None
    pytest_node_id: str | None = None
    cleanup_assertion: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ModelConfigReference(BaseModel):
    source: Literal["runtime_config", "external_experiment", "not_used"]
    reference: str = Field(min_length=1)
    provider: str | None = None
    model: str | None = None


class ModelPrice(BaseModel):
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)
    cache_read_per_million: float | None = Field(default=None, ge=0)
    cache_creation_per_million: float | None = Field(default=None, ge=0)


class PricingSnapshot(BaseModel):
    effective_date: date
    currency: str = Field(default="USD", min_length=1)
    source: str = Field(min_length=1)
    models: dict[str, ModelPrice] = Field(min_length=1)


class EvalSuiteManifest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    suite_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    adapter: Literal["internal"]
    execution_mode: Literal["runtime_cases", "agent_tasks"] | None = None
    model_config_ref: ModelConfigReference
    default_repetitions: int = Field(default=1, ge=1)
    sandbox_mode: Literal["required"] = "required"
    tasks: list[EvalTaskSpec] = Field(default_factory=list)
    pricing_snapshot: PricingSnapshot | None = None

    # 校验套件的适配器专属字段及任务 ID 唯一性
    @model_validator(mode="after")
    def validate_adapter_fields(self) -> EvalSuiteManifest:
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        if self.adapter == "internal" and self.execution_mode is None:
            raise ValueError("internal suite requires execution_mode")
        if self.adapter == "internal" and not self.tasks:
            raise ValueError("internal suite requires at least one task")
        if self.execution_mode == "agent_tasks" and any(
            not task.tool_whitelist for task in self.tasks
        ):
            raise ValueError("internal agent tasks require an explicit non-empty tool_whitelist")
        if self.execution_mode == "runtime_cases" and any(
            task.pytest_node_id is None or task.grader.kind != "pytest" for task in self.tasks
        ):
            raise ValueError("runtime cases require pytest_node_id and pytest grader")
        if self.execution_mode == "runtime_cases" and self.model_config_ref.source != "not_used":
            raise ValueError("runtime cases require model_config_ref.source='not_used'")
        return self


class RunProvenance(BaseModel):
    collected_at: str
    git_sha: str | None
    git_dirty: bool | None
    tree_digest: str
    lock_hash: str | None
    config_hash: str
    repository_root: str
    platform: str
    platform_release: str
    python_version: str
    model: str | None
    model_config_ref: ModelConfigReference
    docker_image: str | None
    docker_image_digest: str | None
    config: dict[str, JsonValue]
    external_evidence: dict[str, JsonValue] = Field(default_factory=dict)


class UsageMetrics(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    currency: str | None = None
    pricing_effective_date: date | None = None
    llm_latency_ms: int | None = Field(default=None, ge=0)


class ToolMetrics(BaseModel):
    calls_started: int | None = Field(default=None, ge=0)
    calls_succeeded: int | None = Field(default=None, ge=0)
    calls_failed: int | None = Field(default=None, ge=0)
    elapsed_ms: int | None = Field(default=None, ge=0)
    errors: int | None = Field(default=None, ge=0)
    refusals: int | None = Field(default=None, ge=0)
    tool_names: list[str] = Field(default_factory=list)


class SandboxMetrics(BaseModel):
    mode: Literal["required"] = "required"
    requested: bool | None = None
    network_mode: Literal["disabled", "not_observed"] = "not_observed"
    observed_backends: list[str] = Field(default_factory=list)
    isolated_workspace: bool | None = None
    fallback_used: bool | None = None
    oom_killed: bool | None = None
    timed_out: bool | None = None
    cleanup_completed: bool | None = None


class CleanupMetrics(BaseModel):
    runtime_cleanup_completed: bool | None = None
    workspace_removed: bool | None = None
    error: str | None = None


class TaskAttemptResult(BaseModel):
    task_id: str
    source_task_id: str | None = None
    repetition: int = Field(ge=1)
    status: EvalStatus
    run_terminal_status: str | None = None
    goal_completed: bool | None = None
    collateral_damage: bool | None = None
    step_count: int | None = Field(default=None, ge=0)
    started_at: str
    finished_at: str
    latency_ms: int = Field(ge=0)
    score: float | None = Field(default=None, ge=0, le=1)
    output: str = ""
    error_type: str | None = None
    error: str | None = None
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    tools: ToolMetrics = Field(default_factory=ToolMetrics)
    sandbox: SandboxMetrics = Field(default_factory=SandboxMetrics)
    cleanup: CleanupMetrics = Field(default_factory=CleanupMetrics)
    evaluation: dict[str, JsonValue] = Field(default_factory=dict)


class EvalSummary(BaseModel):
    total_attempts: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    errors: int = Field(ge=0)
    pass_rate: float | None = Field(default=None, ge=0, le=1)
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)
    currency: str | None = None


class TaskSelectionRecord(BaseModel):
    algorithm: str
    source_task_ids: list[str]
    seed: str | None = None
    sample_size: int | None = Field(default=None, ge=1)
    repeat_first: int | None = Field(default=None, ge=0)
    repeated_runs: int | None = Field(default=None, ge=1)
    default_repetitions: int = Field(ge=1)
    selected_ids_digest: str | None = None
    source_task_ids_digest: str | None = None
    environment_url: str | None = None
    apis_url: str | None = None
    mcp_url: str | None = None


class EvalRunResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    suite_id: str
    suite_name: str
    suite_manifest: str
    adapter: Literal["internal"]
    execution_mode: Literal["runtime_cases", "agent_tasks"] | None = None
    started_at: str
    finished_at: str
    provenance: RunProvenance
    model_config_ref: ModelConfigReference
    selected_task_ids: list[str]
    selection: TaskSelectionRecord
    pricing_snapshot: PricingSnapshot | None = None
    attempts: list[TaskAttemptResult]
    summary: EvalSummary


# 从逐次任务结果计算汇总，通过率以全部计划 attempt 为分母并单列基础设施状态
def summarize_attempts(attempts: list[TaskAttemptResult]) -> EvalSummary:
    passed = sum(attempt.status == EvalStatus.passed for attempt in attempts)
    failed = sum(attempt.status == EvalStatus.failed for attempt in attempts)
    skipped = sum(attempt.status == EvalStatus.skipped for attempt in attempts)
    errors = sum(attempt.status == EvalStatus.error for attempt in attempts)
    known_input = [
        attempt.usage.input_tokens
        for attempt in attempts
        if attempt.usage.input_tokens is not None
    ]
    known_output = [
        attempt.usage.output_tokens
        for attempt in attempts
        if attempt.usage.output_tokens is not None
    ]
    known_costs = [
        attempt.usage.estimated_cost
        for attempt in attempts
        if attempt.usage.estimated_cost is not None
    ]
    currencies = {
        attempt.usage.currency for attempt in attempts if attempt.usage.currency is not None
    }
    costs_complete = (
        bool(attempts) and len(known_costs) == len(attempts)
        and len(currencies) == 1
        and all(attempt.usage.currency is not None for attempt in attempts)
    )
    currency = next(iter(currencies)) if costs_complete else None
    return EvalSummary(
        total_attempts=len(attempts),
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        pass_rate=(passed / len(attempts) if attempts else None),
        total_input_tokens=(
            sum(known_input) if attempts and len(known_input) == len(attempts) else None
        ),
        total_output_tokens=(
            sum(known_output) if attempts and len(known_output) == len(attempts) else None
        ),
        estimated_cost=(sum(known_costs) if costs_complete else None),
        currency=currency,
    )


__all__ = [
    "CleanupMetrics",
    "EvalRunResult",
    "EvalStatus",
    "EvalSuiteManifest",
    "EvalSummary",
    "EvalTaskSpec",
    "GraderSpec",
    "ModelPrice",
    "ModelConfigReference",
    "PricingSnapshot",
    "RunProvenance",
    "SandboxMetrics",
    "TaskAttemptResult",
    "TaskSelectionRecord",
    "ToolMetrics",
    "UsageMetrics",
    "summarize_attempts",
]
