from __future__ import annotations

import asyncio
import copy
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from tars_agent.core.config import TarsConfig
from tars_agent.core.eval.models import (
    CleanupMetrics,
    EvalRunResult,
    EvalStatus,
    EvalSuiteManifest,
    EvalTaskSpec,
    RunProvenance,
    SandboxMetrics,
    TaskAttemptResult,
    TaskSelectionRecord,
    ToolMetrics,
    UsageMetrics,
    summarize_attempts,
)
from tars_agent.core.eval.provenance import collect_provenance


class AdapterUnavailableError(RuntimeError):
    pass


class EvalAdapter(Protocol):
    # 解析适配器动态任务并返回本次实际选择的任务清单
    async def prepare_tasks(self, manifest: EvalSuiteManifest) -> list[EvalTaskSpec]: ...

    # 执行一次真实任务或真实外部评估并返回可审计结果
    async def run_attempt(
        self,
        task: EvalTaskSpec,
        *,
        repetition: int,
        eval_run_id: str,
    ) -> TaskAttemptResult: ...


# 返回当前 UTC 时间的 ISO 8601 字符串
def utc_now() -> str:
    return datetime.now(UTC).isoformat()


# 从 JSON 文件加载并严格校验 Eval 套件清单
def load_manifest(path: Path) -> EvalSuiteManifest:
    try:
        return EvalSuiteManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read suite manifest {path}: {exc}") from exc
    except ValidationError as exc:
        raise ValueError(f"invalid suite manifest {path}: {exc}") from exc


# 从 JSON 文件加载并严格校验 Eval 运行结果
def load_result(path: Path) -> EvalRunResult:
    try:
        return EvalRunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read eval result {path}: {exc}") from exc
    except ValidationError as exc:
        raise ValueError(f"invalid eval result {path}: {exc}") from exc


# 从 artifact 目录读取 result.json，同时兼容直接传入旧结果文件
def load_artifact_result(path: Path) -> EvalRunResult:
    result_path = path / "result.json" if path.is_dir() else path
    return load_result(result_path)


# 原子写入任意 Eval 文本 artifact
def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# 通过同目录临时文件原子写入 Eval 结果，避免中断留下半个 JSON
def write_result(path: Path, result: EvalRunResult) -> None:
    _write_text(path, result.model_dump_json(indent=2, exclude_none=False) + "\n")


# 写入可独立审计的 manifest 快照、JSON Schema 和运行结果三件套
def write_eval_artifacts(
    artifact_dir: Path,
    manifest: EvalSuiteManifest,
    result: EvalRunResult,
) -> None:
    if artifact_dir.exists() and not artifact_dir.is_dir():
        raise ValueError(f"artifact output is not a directory: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        artifact_dir / "manifest.json",
        manifest.model_dump_json(indent=2, exclude_none=False) + "\n",
    )
    schema = {
        "schema_version": "1.0",
        "manifest": EvalSuiteManifest.model_json_schema(),
        "result": EvalRunResult.model_json_schema(),
    }
    _write_text(
        artifact_dir / "schema.json",
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
    )
    write_result(artifact_dir / "result.json", result)


# 从起始路径向上寻找 Git worktree 根目录，找不到时使用起始目录
def find_repository_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


# 构造适配器准备阶段的明确 skipped/error 结果
def _setup_result(status: EvalStatus, error_type: str, message: str) -> TaskAttemptResult:
    now = utc_now()
    return TaskAttemptResult(
        task_id="adapter_setup",
        repetition=1,
        status=status,
        started_at=now,
        finished_at=now,
        latency_ms=0,
        error_type=error_type,
        error=message,
        usage=UsageMetrics(),
        tools=ToolMetrics(),
        sandbox=SandboxMetrics(),
        cleanup=CleanupMetrics(),
    )


# 构造一次未进入适配器正常返回路径的错误结果
def _attempt_error(
    task: EvalTaskSpec,
    repetition: int,
    started_at: str,
    started_clock: float,
    error_type: str,
    message: str,
) -> TaskAttemptResult:
    elapsed = max(0, int((asyncio.get_running_loop().time() - started_clock) * 1000))
    return TaskAttemptResult(
        task_id=task.id,
        source_task_id=task.source_task_id,
        repetition=repetition,
        status=EvalStatus.error,
        started_at=started_at,
        finished_at=utc_now(),
        latency_ms=elapsed,
        error_type=error_type,
        error=message,
        usage=UsageMetrics(),
        tools=ToolMetrics(),
        sandbox=SandboxMetrics(timed_out=(error_type == "timeout")),
        cleanup=CleanupMetrics(),
    )


class EvaluationRunner:
    # 保存不可变运行输入，由 run 顺序执行以避免外部评估器共享状态冲突
    def __init__(
        self,
        manifest: EvalSuiteManifest,
        manifest_path: Path,
        provenance: RunProvenance,
        adapter: EvalAdapter,
    ) -> None:
        self._manifest = manifest
        self._manifest_path = manifest_path
        self._provenance = provenance
        self._adapter = adapter

    # 顺序执行套件全部重复项，并将基础设施异常转成显式结果而非伪造分数
    async def run(self) -> EvalRunResult:
        run_id = f"eval-{uuid.uuid4().hex}"
        started_at = utc_now()
        attempts: list[TaskAttemptResult] = []
        tasks: list[EvalTaskSpec] = []
        try:
            tasks = await self._adapter.prepare_tasks(self._manifest)
        except AdapterUnavailableError as exc:
            attempts.append(_setup_result(EvalStatus.skipped, "adapter_unavailable", str(exc)))
        except Exception as exc:
            attempts.append(_setup_result(EvalStatus.error, "adapter_setup_error", str(exc)))

        for task in tasks:
            repetitions = task.repetitions or self._manifest.default_repetitions
            for repetition in range(1, repetitions + 1):
                started_clock = asyncio.get_running_loop().time()
                attempt_started_at = utc_now()
                try:
                    guard_timeout = task.timeout_s
                    if self._manifest.execution_mode == "runtime_cases":
                        guard_timeout += 10.0
                    async with asyncio.timeout(guard_timeout):
                        attempt = await self._adapter.run_attempt(
                            task,
                            repetition=repetition,
                            eval_run_id=run_id,
                        )
                    if attempt.task_id != task.id or attempt.repetition != repetition:
                        raise RuntimeError("adapter returned mismatched task identity")
                    attempts.append(attempt)
                except TimeoutError:
                    attempts.append(
                        _attempt_error(
                            task,
                            repetition,
                            attempt_started_at,
                            started_clock,
                            "timeout",
                            f"task exceeded timeout_s={task.timeout_s:g}",
                        )
                    )
                except Exception as exc:
                    attempts.append(
                        _attempt_error(
                            task,
                            repetition,
                            attempt_started_at,
                            started_clock,
                            "adapter_error",
                            str(exc),
                        )
                    )

        selected_task_ids = [task.source_task_id or task.id for task in tasks]
        selection = TaskSelectionRecord(
            algorithm="manifest_order",
            source_task_ids=selected_task_ids,
            default_repetitions=self._manifest.default_repetitions,
        )
        return EvalRunResult(
            run_id=run_id,
            suite_id=self._manifest.suite_id,
            suite_name=self._manifest.name,
            suite_manifest=str(self._manifest_path),
            adapter=self._manifest.adapter,
            execution_mode=self._manifest.execution_mode,
            started_at=started_at,
            finished_at=utc_now(),
            provenance=self._provenance,
            model_config_ref=self._manifest.model_config_ref,
            selected_task_ids=selected_task_ids,
            selection=selection,
            pricing_snapshot=self._manifest.pricing_snapshot,
            attempts=attempts,
            summary=summarize_attempts(attempts),
        )


# 根据清单类型构造内部零模型回归或真实任务适配器
def build_adapter(
    manifest: EvalSuiteManifest,
    config: TarsConfig,
    *,
    repository_root: Path | None = None,
) -> EvalAdapter:
    if manifest.adapter == "internal" and manifest.execution_mode == "agent_tasks":
        from tars_agent.core.eval.internal import InternalEvalAdapter

        return InternalEvalAdapter(config, pricing_snapshot=manifest.pricing_snapshot)
    if manifest.adapter == "internal" and manifest.execution_mode == "runtime_cases":
        from tars_agent.core.eval.runtime_cases import InternalRuntimeCaseAdapter

        return InternalRuntimeCaseAdapter(repository_root or find_repository_root(Path.cwd()))
    raise ValueError("unsupported internal execution mode")


# 加载清单、采集运行前证据并执行一次完整 Eval
async def run_eval_suite(
    suite_path: Path,
    artifact_dir: Path,
    config: TarsConfig,
    *,
    repository_root: Path | None = None,
) -> EvalRunResult:
    manifest = load_manifest(suite_path)
    root = find_repository_root(repository_root or Path.cwd())
    execution_config = copy.deepcopy(config)
    execution_config.sandbox.mode = manifest.sandbox_mode
    resolved_artifact_dir = artifact_dir.resolve()
    if resolved_artifact_dir == root or resolved_artifact_dir in root.parents:
        raise ValueError("artifact directory cannot be the repository root or its ancestor")
    if resolved_artifact_dir.exists() and not resolved_artifact_dir.is_dir():
        raise ValueError(f"artifact output is not a directory: {resolved_artifact_dir}")
    model_ref = manifest.model_config_ref
    model: str | None = model_ref.model
    if manifest.execution_mode == "agent_tasks":
        model = execution_config.llm.default_model
        model_ref = model_ref.model_copy(update={"model": model})
        manifest = manifest.model_copy(update={"model_config_ref": model_ref})
    provenance = collect_provenance(
        root,
        execution_config,
        model=model,
        model_config_ref=model_ref,
        eval_config={
            "suite_id": manifest.suite_id,
            "adapter": manifest.adapter,
            "execution_mode": manifest.execution_mode,
            "sandbox_mode": manifest.sandbox_mode,
            "default_repetitions": manifest.default_repetitions,
            "suite_source": _public_suite_source(suite_path, root),
        },
        exclude=(resolved_artifact_dir,),
    )
    adapter = build_adapter(manifest, execution_config, repository_root=root)
    manifest_snapshot = Path("manifest.json")
    result = await EvaluationRunner(manifest, manifest_snapshot, provenance, adapter).run()
    write_eval_artifacts(resolved_artifact_dir, manifest, result)
    return result


# 将 suite 来源压缩为仓库内相对路径，避免公开 artifact 泄露本机绝对目录
def _public_suite_source(suite_path: Path, root: Path) -> str:
    resolved = suite_path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.name


__all__ = [
    "AdapterUnavailableError",
    "EvalAdapter",
    "EvaluationRunner",
    "build_adapter",
    "find_repository_root",
    "load_manifest",
    "load_artifact_result",
    "load_result",
    "run_eval_suite",
    "utc_now",
    "write_result",
    "write_eval_artifacts",
]
