from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Literal

import pytest

from tars_agent.cli.commands import eval as cli_eval
from tars_agent.core.config import TarsConfig
from tars_agent.core.eval.models import (
    EvalRunResult,
    EvalStatus,
    ModelConfigReference,
    RunProvenance,
    SandboxMetrics,
    TaskAttemptResult,
    TaskSelectionRecord,
    summarize_attempts,
)
from tars_agent.core.eval.report import render_markdown_report
from tars_agent.core.eval.runner import (
    load_artifact_result,
    load_manifest,
    load_result,
    write_eval_artifacts,
    write_result,
)


# 构造报告与 JSON 往返测试共用的最小真实结果模型
def _result() -> EvalRunResult:
    model_ref = ModelConfigReference(
        source="runtime_config",
        reference="TarsConfig.llm",
        provider="anthropic",
        model="model-a",
    )
    attempt = TaskAttemptResult(
        task_id="task-1",
        repetition=1,
        status=EvalStatus.passed,
        run_terminal_status="success",
        goal_completed=True,
        step_count=2,
        started_at="2026-08-24T00:00:00+00:00",
        finished_at="2026-08-24T00:00:01+00:00",
        latency_ms=1000,
        score=1,
        sandbox=SandboxMetrics(
            requested=True,
            network_mode="disabled",
            observed_backends=["workspace_sandbox"],
            isolated_workspace=True,
            fallback_used=False,
            timed_out=False,
            cleanup_completed=True,
        ),
    )
    return EvalRunResult(
        run_id="eval-1",
        suite_id="suite-1",
        suite_name="Suite One",
        suite_manifest="evals/suite.json",
        adapter="internal",
        started_at="2026-08-24T00:00:00+00:00",
        finished_at="2026-08-24T00:00:01+00:00",
        provenance=RunProvenance(
            collected_at="2026-08-24T00:00:00+00:00",
            git_sha="a" * 40,
            git_dirty=True,
            tree_digest="b" * 64,
            lock_hash="c" * 64,
            config_hash="d" * 64,
            repository_root=".",
            platform="Windows",
            platform_release="11",
            python_version="3.12.0",
            model="model-a",
            model_config_ref=model_ref,
            docker_image="kama:1",
            docker_image_digest=None,
            config={"sandbox": {"network_mode": "disabled"}},
        ),
        model_config_ref=model_ref,
        selected_task_ids=["task-1"],
        selection=TaskSelectionRecord(
            algorithm="manifest_order",
            source_task_ids=["task-1"],
            default_repetitions=1,
        ),
        attempts=[attempt],
        summary=summarize_attempts([attempt]),
    )


# 功能：验证结果 JSON 能按 Pydantic schema 完整写入并重新加载
# 设计：使用正式原子写函数做往返，覆盖新增 provenance、selection 与沙箱字段
def test_eval_result_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "result.json"

    write_result(path, _result())
    loaded = load_result(path)

    assert loaded.suite_id == "suite-1"
    assert loaded.provenance.config_hash == "d" * 64
    assert loaded.attempts[0].sandbox.cleanup_completed is True


# 功能：验证 artifact 目录同时落 manifest 快照、schema 与 result 三件套
# 设计：调用正式 artifact 写入函数并从目录重新读取结果，锁定 CLI 主路径不是单 JSON 文件
def test_eval_artifact_directory_contains_required_files(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / "evals" / "internal-deterministic.json")
    artifact = tmp_path / "artifact"

    write_eval_artifacts(artifact, manifest, _result())

    assert {path.name for path in artifact.iterdir()} == {
        "manifest.json",
        "schema.json",
        "result.json",
    }
    assert load_artifact_result(artifact).run_id == "eval-1"
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(artifact.iterdir())
    )
    assert str(root.resolve()) not in artifact_text


# 功能：验证 Markdown 报告包含关键仓库与任务证据
# 设计：从强类型结果渲染，检查 SHA、脏树、终态表格和 unknown 边界声明
def test_markdown_report_surfaces_evidence_boundary() -> None:
    rendered = render_markdown_report(_result())

    assert "Dirty worktree: `yes`" in rendered
    assert "task-1" in rendered
    assert "workspace_sandbox" in rendered
    assert "does not infer" in rendered


# 功能：验证 `kama eval report` 不加载运行时配置即可分发
# 设计：替换命令函数并让 get_config 一旦调用就失败，锁定离线报告路径无配置副作用
def test_cli_eval_report_dispatches_without_loading_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_main = importlib.import_module("tars_agent.cli.main")
    calls: list[tuple[Path, str, Path | None]] = []

    # 捕获 report 子命令解析出的参数
    def fake_report(
        result_path: Path,
        *,
        format: Literal["json", "md"],
        output: Path | None,
    ) -> None:
        calls.append((result_path, format, output))

    # 若离线报告错误加载配置则立即暴露
    def fail_get_config() -> TarsConfig:
        raise AssertionError("get_config must not be called")

    monkeypatch.setattr(cli_main, "cmd_eval_report", fake_report)
    monkeypatch.setattr(cli_main, "get_config", fail_get_config)
    monkeypatch.setattr(
        sys,
        "argv",
        ["kama", "eval", "report", "result.json", "--format", "json"],
    )

    cli_main.main()

    assert calls == [(Path("result.json"), "json", None)]


# 功能：验证 `kama eval run` 解析 suite/output/model 并沿用现有配置初始化风格
# 设计：替换执行边界避免模型调用，仅检查 argparse 到命令函数的完整参数传递
def test_cli_eval_run_dispatches_paths_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    cli_main = importlib.import_module("tars_agent.cli.main")
    config = TarsConfig()
    calls: list[tuple[TarsConfig, Path, Path, str | None]] = []

    # 捕获 run 子命令解析出的路径与模型覆盖
    def fake_run(
        received_config: TarsConfig,
        *,
        suite: Path,
        output: Path,
        model: str | None,
    ) -> None:
        calls.append((received_config, suite, output, model))

    monkeypatch.setattr(cli_main, "get_config", lambda: config)
    monkeypatch.setattr(cli_main, "setup_logging", lambda _config: None)
    monkeypatch.setattr(cli_main, "cmd_eval_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kama",
            "eval",
            "run",
            "--suite",
            "evals/internal-deterministic.json",
            "--output",
            "out/artifact",
            "--model",
            "model-a",
        ],
    )

    cli_main.main()

    assert calls == [
        (
            config,
            Path("evals/internal-deterministic.json"),
            Path("out/artifact"),
            "model-a",
        )
    ]


# 功能：验证 Eval 含失败、错误或跳过时 CLI 返回退出码 1
# 设计：替换异步执行边界返回一个真实失败结果，区分运行结果失败与命令输入错误
def test_eval_run_returns_one_when_attempt_is_not_passed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_attempt = _result().attempts[0].model_copy(
        update={"status": EvalStatus.failed, "score": 0.0}
    )
    result = _result().model_copy(
        update={
            "attempts": [failed_attempt],
            "summary": summarize_attempts([failed_attempt]),
        }
    )

    # 返回含真实失败终态的强类型结果，不触发文件或模型调用
    async def fake_run(*_args: object, **_kwargs: object) -> EvalRunResult:
        return result

    monkeypatch.setattr(cli_eval, "run_eval_suite", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        cli_eval.cmd_eval_run(
            TarsConfig(),
            suite=Path("suite.json"),
            output=tmp_path / "artifact",
        )

    assert exc_info.value.code == 1


# 功能：验证 manifest、schema 或 I/O 输入错误时 CLI 返回退出码 2
# 设计：让执行边界抛出 ValueError，锁定使用错误不会与任务失败共用退出码
def test_eval_run_returns_two_for_cli_or_schema_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 模拟清单校验失败且不创建任何 artifact
    async def invalid_suite(*_args: object, **_kwargs: object) -> EvalRunResult:
        raise ValueError("invalid suite")

    monkeypatch.setattr(cli_eval, "run_eval_suite", invalid_suite)

    with pytest.raises(SystemExit) as exc_info:
        cli_eval.cmd_eval_run(
            TarsConfig(),
            suite=Path("missing.json"),
            output=tmp_path / "artifact",
        )

    assert exc_info.value.code == 2
