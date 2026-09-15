from __future__ import annotations

import asyncio
import os
import subprocess
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from tars_agent.core.config import TarsConfig
from tars_agent.core.eval import internal
from tars_agent.core.eval.models import (
    EvalStatus,
    EvalSuiteManifest,
    EvalTaskSpec,
    GraderSpec,
    ModelConfigReference,
    ModelPrice,
    PricingSnapshot,
    TaskAttemptResult,
    UsageMetrics,
    summarize_attempts,
)
from tars_agent.core.eval.provenance import compute_tree_digest, safe_config_snapshot
from tars_agent.core.eval.runner import load_manifest
from tars_agent.core.eval.runtime_cases import InternalRuntimeCaseAdapter


# 功能：验证内置确定性套件确实包含 12 个唯一任务且全部显式收窄工具
# 设计：读取正式 manifest 而非复制测试数据，防止交付清单与模型校验脱节
def test_internal_manifest_has_twelve_explicit_tasks() -> None:
    root = Path(__file__).resolve().parents[2]

    manifest = load_manifest(root / "evals" / "internal-deterministic.json")

    assert manifest.adapter == "internal"
    assert manifest.execution_mode == "runtime_cases"
    assert manifest.model_config_ref.source == "not_used"
    assert len(manifest.tasks) == 12
    assert len({task.id for task in manifest.tasks}) == 12
    assert all(task.pytest_node_id for task in manifest.tasks)
    assert all(task.grader.kind == "pytest" for task in manifest.tasks)


# 功能：验证零模型 runtime adapter 真实执行固定 pytest 节点并记录 JUnit 与清理证据
# 设计：只运行 12 项白名单中的正常读写节点，断言零 token、stdout hash 和临时目录清理
async def test_runtime_case_adapter_runs_allowlisted_node() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / "evals" / "internal-deterministic.json")
    adapter = InternalRuntimeCaseAdapter(root)
    tasks = await adapter.prepare_tasks(manifest)

    result = await adapter.run_attempt(tasks[0], repetition=1, eval_run_id="eval-test")

    assert result.status == EvalStatus.passed
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert result.evaluation["stdout_sha256"]
    assert result.evaluation["cleanup_assertion_passed"] is True
    assert result.cleanup.workspace_removed is True
    assert result.sandbox.isolated_workspace is None


# 功能：验证 runtime case 的 manifest timeout 会终止 pytest 进程树并记录明确终态
# 设计：把固定白名单节点超时缩短到启动时间以下，断言不会依赖外层 runner 才能清理
async def test_runtime_case_adapter_enforces_task_timeout() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / "evals" / "internal-deterministic.json")
    adapter = InternalRuntimeCaseAdapter(root)
    task = manifest.tasks[0].model_copy(update={"timeout_s": 0.001})

    result = await adapter.run_attempt(task, repetition=1, eval_run_id="eval-timeout")

    assert result.status == EvalStatus.error
    assert result.error_type == "pytest_timeout"
    assert result.sandbox.timed_out is True
    assert result.cleanup.runtime_cleanup_completed is True
    assert result.cleanup.workspace_removed is True


# 功能：验证内部任务缺少显式工具白名单时 manifest 会被拒绝
# 设计：直接构造最小清单触发跨字段校验，锁定空列表不会被 AgentRunner 解释成允许全部工具
def test_internal_manifest_rejects_empty_tool_whitelist() -> None:
    with pytest.raises(ValidationError, match="tool_whitelist"):
        EvalSuiteManifest(
            suite_id="unsafe",
            name="unsafe",
            adapter="internal",
            execution_mode="agent_tasks",
            model_config_ref=ModelConfigReference(
                source="runtime_config",
                reference="test",
            ),
            tasks=[
                EvalTaskSpec(
                    id="task",
                    goal="answer",
                    grader=GraderSpec(kind="output_contains", expected="answer"),
                )
            ],
        )


# 功能：验证缺少真实模型凭证时内部适配器明确 skip 且仍移除临时工作区
# 设计：不注入假模型，只隔离临时路径并删除 API key，确认不会生成伪造通过结果
async def test_internal_adapter_skips_without_model_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "attempt-workspace"

    # 创建受测试适配器可删除的固定临时目录
    def fake_mkdtemp(*, prefix: str) -> str:
        assert prefix == "tars-eval-"
        workspace.mkdir()
        return str(workspace)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(internal.tempfile, "mkdtemp", fake_mkdtemp)
    adapter = internal.InternalEvalAdapter(TarsConfig())
    task = EvalTaskSpec(
        id="read",
        goal="Read clue.txt",
        tool_whitelist=["read_file"],
        fixture_files={"clue.txt": "real\n"},
        grader=GraderSpec(kind="output_contains", expected="real"),
    )

    result = await adapter.run_attempt(task, repetition=1, eval_run_id="eval-test")

    assert result.status == EvalStatus.skipped
    assert result.error_type == "missing_credentials"
    assert result.score is None
    assert result.usage.input_tokens is None
    assert result.tools.calls_started is None
    assert result.cleanup.runtime_cleanup_completed is None
    assert result.cleanup.workspace_removed is True
    assert not workspace.exists()


# 功能：验证费用只依据嵌入结果的带日期价格快照计算
# 设计：覆盖输入、输出和缓存四类 token，断言币种、日期和精确算式同时进入结果
def test_pricing_snapshot_calculates_recorded_usage() -> None:
    usage = UsageMetrics(
        input_tokens=1_000,
        output_tokens=500,
        cache_read_input_tokens=2_000,
        cache_creation_input_tokens=250,
    )
    snapshot = PricingSnapshot(
        effective_date=date(2026, 8, 24),
        currency="USD",
        source="unit-test snapshot",
        models={
            "model-a": ModelPrice(
                input_per_million=3,
                output_per_million=15,
                cache_read_per_million=0.3,
                cache_creation_per_million=3.75,
            )
        },
    )

    priced = internal._apply_pricing(usage, "model-a", snapshot)

    assert priced.estimated_cost == 0.0120375
    assert priced.currency == "USD"
    assert priced.pricing_effective_date == date(2026, 8, 24)


# 功能：验证工作树摘要会随未忽略的未跟踪文件内容改变
# 设计：在临时 Git 仓库内比较写文件前后摘要，覆盖当前脏树而不依赖 HEAD 提交
def test_tree_digest_includes_untracked_worktree_content(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)

    before = compute_tree_digest(tmp_path)
    (tmp_path / "new.txt").write_text("evidence\n", encoding="utf-8")
    after = compute_tree_digest(tmp_path)

    assert before != after


# 功能：验证配置快照不会记录 MCP URL、请求头、环境变量或 trace 文件路径
# 设计：注入明显的敏感占位值后检查序列化文本，锁定 provenance 的脱敏边界
def test_safe_config_snapshot_excludes_secret_bearing_fields() -> None:
    from tars_agent.core.config import McpServerConfig

    config = TarsConfig()
    config.trace.file = "secret-trace-path"
    config.mcp.servers = [
        McpServerConfig(
            name="site",
            transport="streamable_http",
            trusted=True,
            url="https://secret.example/mcp",
            headers={"Authorization": "Bearer secret"},
            env={"TOKEN": "secret"},
        )
    ]

    snapshot_text = str(safe_config_snapshot(config))

    assert "site" in snapshot_text
    assert "secret" not in snapshot_text
    assert "Authorization" not in snapshot_text


# 功能：验证 pass rate 以全部计划 attempt 为分母并仍单列 skip/error
# 设计：构造四种状态且 token 未知，确保报告不会在存在跳过项时误写 100%
def test_summary_separates_failures_from_infrastructure_states() -> None:
    attempts = [
        _attempt("a", EvalStatus.passed),
        _attempt("b", EvalStatus.failed),
        _attempt("c", EvalStatus.skipped),
        _attempt("d", EvalStatus.error),
    ]

    summary = summarize_attempts(attempts)

    assert summary.pass_rate == 0.25
    assert summary.skipped == 1
    assert summary.errors == 1
    assert summary.total_input_tokens is None


# 构造只用于汇总口径测试的最小 attempt
def _attempt(task_id: str, status: EvalStatus) -> TaskAttemptResult:
    return TaskAttemptResult(
        task_id=task_id,
        repetition=1,
        status=status,
        started_at="2026-08-24T00:00:00+00:00",
        finished_at="2026-08-24T00:00:00+00:00",
        latency_ms=0,
    )



def test_empty_usage_accumulator_keeps_unknown() -> None:
    usage = internal._EventAccumulator().usage()
    assert usage.input_tokens is None
    assert usage.output_tokens is None


def test_incomplete_usage_cannot_generate_zero_cost() -> None:
    snapshot = PricingSnapshot(
        effective_date=date(2026, 9, 14), currency="USD", source="test",
        models={"m": ModelPrice(input_per_million=1, output_per_million=2)},
    )
    assert internal._apply_pricing(UsageMetrics(), "m", snapshot).estimated_cost is None


def test_summary_does_not_label_partial_usage_as_total() -> None:
    known = _attempt("known", EvalStatus.passed)
    known.usage = UsageMetrics(input_tokens=10, output_tokens=2)
    unknown = _attempt("unknown", EvalStatus.error)
    result = summarize_attempts([known, unknown])
    assert result.total_input_tokens is None
    assert result.total_output_tokens is None


def test_summary_never_adds_different_currencies() -> None:
    first = _attempt("a", EvalStatus.passed)
    first.usage = UsageMetrics(estimated_cost=1, currency="USD")
    second = _attempt("b", EvalStatus.passed)
    second.usage = UsageMetrics(estimated_cost=2, currency="CNY")
    result = summarize_attempts([first, second])
    assert result.estimated_cost is None
    assert result.currency is None


@pytest.mark.parametrize("reason", ["llm_protocol_error", "llm_total_timeout", "llm_request_budget_exhausted"])
def test_provider_infrastructure_errors_are_not_task_success(reason: str) -> None:
    from tars_agent.core.runner import RunOutcome

    outcome = RunOutcome(status="success", result="matches", reason=reason)
    assert internal._status_from_outcome(outcome, True) == EvalStatus.error


async def test_internal_write_task_has_permission_and_keeps_fixed_budget_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.llm.types import LlmResponse, ToolCallBlock
    from tars_agent.core.runner import AgentRunner
    from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter

    monkeypatch.setenv("TARS_HOME", str(tmp_path / "trusted-home"))
    config = TarsConfig()
    config.llm.api_key = "unit-test-placeholder"
    fixed_budget = config.llm.request_budget_path
    observed_budget_paths = []
    observed_homes = []
    runtime = RuntimeRouter(FakeRuntime(), allow_host_fallback=False)
    monkeypatch.setattr(internal, "build_runtime_router", lambda sandbox: runtime)

    class ScriptedProvider:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, tool_schemas, bus, run_id, **kwargs):
            observed_homes.append(Path(os.environ["TARS_HOME"]))
            self.calls += 1
            if self.calls == 1:
                return LlmResponse(
                    stop_reason="tool_use",
                    tool_calls=[ToolCallBlock("write-1", "write_file", {"path": "result.txt", "content": "verified"})],
                )
            return LlmResponse(stop_reason="end_turn", text="done")

    def runner_factory(actual_config, **kwargs):
        observed_budget_paths.append(actual_config.llm.request_budget_path)
        assert kwargs["permission_manager"].evaluate("write_file", {}) == "allow"
        return AgentRunner(actual_config, provider=ScriptedProvider(), **kwargs)

    monkeypatch.setattr(internal, "AgentRunner", runner_factory)
    adapter = internal.InternalEvalAdapter(config)
    config.llm.request_budget_path = tmp_path / "mutated.sqlite3"
    task = EvalTaskSpec(
        id="write", goal="Create result.txt with verified", tool_whitelist=["write_file"],
        grader=GraderSpec(kind="file_equals", path="result.txt", expected="verified"),
    )
    result = await adapter.run_attempt(task, repetition=1, eval_run_id="test")
    assert result.status == EvalStatus.passed
    assert result.tools.calls_succeeded == 1
    assert result.usage.input_tokens is None
    assert result.cleanup.runtime_cleanup_completed is None
    assert result.sandbox.cleanup_completed is None
    assert result.cleanup.workspace_removed is True
    assert result.evaluation["runtime_cleanup_attempted"] is True
    assert observed_budget_paths == [fixed_budget]
    assert observed_homes and all(home.name == ".tars-eval-home" for home in observed_homes)
    assert Path(os.environ["TARS_HOME"]) == tmp_path / "trusted-home"


async def test_internal_cleanup_exception_cannot_leave_passed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.runner import RunOutcome
    from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter

    config = TarsConfig()
    config.llm.api_key = "unit-test-placeholder"
    config.llm.request_budget_path = tmp_path / "budget.sqlite3"

    class CleanupFailure(FakeRuntime):
        async def cleanup(self):
            raise OSError("cleanup not confirmed")

    class FinishedRunner:
        def __init__(self, *args, **kwargs):
            pass

        async def run_and_capture(self, *args, **kwargs):
            return RunOutcome(status="success", result="done", reason=None)

    monkeypatch.setattr(internal, "AgentRunner", FinishedRunner)
    monkeypatch.setattr(internal, "build_runtime_router", lambda sandbox: RuntimeRouter(CleanupFailure(), allow_host_fallback=False))
    task = EvalTaskSpec(
        id="finish", goal="say done", tool_whitelist=["read_file"],
        grader=GraderSpec(kind="output_contains", expected="done"),
    )
    result = await internal.InternalEvalAdapter(config).run_attempt(task, repetition=1, eval_run_id="test")
    assert result.status == EvalStatus.error
    assert result.score == 0
    assert result.cleanup.runtime_cleanup_completed is False
    assert result.error_type == "runtime_cleanup_failed"


async def test_internal_adapter_requires_required_mode_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TarsConfig()
    config.sandbox.mode = "preferred"
    config.llm.api_key = "unit-test-placeholder"
    config.llm.request_budget_path = tmp_path / "budget.sqlite3"
    task = EvalTaskSpec(
        id="x", goal="say done", tool_whitelist=["write_file"],
        grader=GraderSpec(kind="output_contains", expected="done"),
    )
    result = await internal.InternalEvalAdapter(config).run_attempt(task, repetition=1, eval_run_id="test")
    assert result.status == EvalStatus.error
    assert "required" in result.error
    assert result.evaluation.get("model_request_reservations") is None
    assert not config.llm.request_budget_path.exists()


async def test_internal_cancellation_restores_process_context_and_cleans_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter

    monkeypatch.setenv("TARS_HOME", str(tmp_path / "trusted-home"))
    original_home = os.environ["TARS_HOME"]
    original_cwd = Path.cwd()
    config = TarsConfig()
    config.llm.api_key = "unit-test-placeholder"
    started = asyncio.Event()
    workspaces = []

    class WaitingRunner:
        def __init__(self, *args, **kwargs):
            pass

        async def run_and_capture(self, *args, **kwargs):
            from tars_agent.core.runner import RunOutcome
            workspaces.append(kwargs["workspace_root"])
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return RunOutcome(status="cancelled", result="", reason="cancelled")

    monkeypatch.setattr(internal, "AgentRunner", WaitingRunner)
    monkeypatch.setattr(internal, "build_runtime_router", lambda sandbox: RuntimeRouter(FakeRuntime(), allow_host_fallback=False))
    task = EvalTaskSpec(
        id="x", goal="wait", tool_whitelist=["read_file"],
        grader=GraderSpec(kind="output_contains", expected="done"),
    )
    operation = asyncio.create_task(internal.InternalEvalAdapter(config).run_attempt(task, repetition=1, eval_run_id="test"))
    await started.wait()
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert os.environ["TARS_HOME"] == original_home
    assert Path.cwd() == original_cwd
    assert workspaces and not workspaces[0].exists()


def test_provenance_does_not_hash_private_dotenv_variants(tmp_path: Path) -> None:
    initial = compute_tree_digest(tmp_path)
    (tmp_path / ".env.local").write_text("PRIVATE=one", encoding="utf-8")
    assert compute_tree_digest(tmp_path) == initial
    (tmp_path / ".env.example").write_text("PUBLIC_EXAMPLE=", encoding="utf-8")
    assert compute_tree_digest(tmp_path) != initial


@pytest.mark.parametrize("request_limit", [75, 100, None])
def test_safe_config_snapshot_records_actual_request_limit(request_limit: int | None) -> None:
    config = TarsConfig()
    config.llm.request_limit = request_limit
    config.llm.api_key = "do-not-export-this-key"
    snapshot = safe_config_snapshot(config)
    assert snapshot["llm"]["request_limit"] == request_limit
    assert "do-not-export-this-key" not in str(snapshot)
