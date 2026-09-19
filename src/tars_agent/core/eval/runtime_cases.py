from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET  # type: ignore[import-untyped]
from pydantic import JsonValue

from tars_agent.core.eval.models import (
    CleanupMetrics,
    EvalStatus,
    EvalSuiteManifest,
    EvalTaskSpec,
    SandboxMetrics,
    TaskAttemptResult,
    ToolMetrics,
    UsageMetrics,
)
from tars_agent.core.eval.runner import AdapterUnavailableError, utc_now

ALLOWED_RUNTIME_CASES: dict[str, str] = {
    "normal-read-write": (
        "tests/unit/test_eval_runtime_cases.py::test_runtime_case_normal_read_write"
    ),
    "path-escape": (
        "tests/unit/test_sandbox_runtime.py::"
        "test_resolve_rejects_absolute_and_parent_escape"
    ),
    "symlink-escape": (
        "tests/unit/test_sandbox_runtime.py::test_resolve_rejects_symlink_escape"
    ),
    "permission-refusal": (
        "tests/unit/test_permission_manager.py::"
        "test_wrong_session_or_invalid_decision_does_not_resolve"
    ),
    "sandbox-unavailable": (
        "tests/unit/test_sandbox_runtime.py::"
        "test_unavailable_sandbox_denial_starts_no_host_process"
    ),
    "required-no-host-fallback": (
        "tests/unit/test_sandbox_runtime.py::test_required_sandbox_disables_host_fallback"
    ),
    "bash-timeout-cleanup": (
        "tests/unit/test_eval_runtime_cases.py::"
        "test_runtime_case_bash_timeout_cleans_child_process"
    ),
    "idempotent-message": (
        "tests/unit/test_runtime_service.py::"
        "test_submit_returns_before_execution_and_deduplicates_message_id"
    ),
    "daemon-crash-recovery": (
        "tests/integration/test_daemon_crash_recovery.py::"
        "test_hard_kill_marks_run_interrupted_without_replaying_side_effect"
    ),
    "subagent-cleanup": (
        "tests/integration/test_daemon_crash_recovery.py::"
        "test_hard_kill_and_restart_interrupts_active_subagent_without_resuming_it"
    ),
    "compact-transaction": (
        "tests/unit/test_runtime_service.py::test_manual_compaction_is_transactional_and_auditable"
    ),
    "mcp-invalid-result": (
        "tests/unit/test_mcp_client_v2.py::test_binary_content_is_omitted_from_context"
    ),
}

_NODE_PATTERN = re.compile(r"^tests/[A-Za-z0-9_./-]+\.py::test_[A-Za-z0-9_]+$")


# 删除可能让零模型用例意外调用外部服务的凭证环境变量
def _sanitized_environment(repository_root: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    }
    environment["PYTHONPATH"] = str(repository_root / "src")
    return environment


# 从 pytest JUnit XML 读取测试、失败、错误和跳过数量
def _junit_counts(path: Path) -> tuple[int, int, int, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    return tests, failures, errors, skipped


# 截取 pytest 输出末尾并限制 artifact 体积
def _output_summary(output: bytes, *, limit: int = 2_000) -> str:
    decoded = output.decode("utf-8", errors="replace").strip()
    if len(decoded) <= limit:
        return decoded
    return "…" + decoded[-(limit - 1) :]


# 让 pytest 及其后代进入可整体终止的独立进程组
def _process_group_options() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


# 跨平台终止 pytest 完整进程树，并等待直接子进程被回收
async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        await process.wait()
        return
    pid = process.pid
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(killer.wait(), timeout=5.0)
        except TimeoutError:
            killer.kill()
            await killer.wait()
    else:
        kill_group = getattr(os, "killpg", None)
        sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
        if kill_group is None:
            process.kill()
        else:
            try:
                kill_group(pid, sigkill)
            except ProcessLookupError:
                pass
            except PermissionError:
                process.kill()
    if process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:
            process.kill()
            await process.wait()


class InternalRuntimeCaseAdapter:
    # 固定仓库根与当前解释器，禁止 manifest 提供命令、参数或外部路径
    def __init__(self, repository_root: Path) -> None:
        self._repository_root = repository_root.resolve()

    # 校验每项严格匹配代码内白名单及仓库内 pytest node id
    async def prepare_tasks(self, manifest: EvalSuiteManifest) -> list[EvalTaskSpec]:
        if importlib.util.find_spec("pytest") is None:
            raise AdapterUnavailableError("pytest is unavailable in the current interpreter")
        for task in manifest.tasks:
            expected = ALLOWED_RUNTIME_CASES.get(task.id)
            if expected is None or task.pytest_node_id != expected:
                raise ValueError(f"runtime case is not allowlisted: {task.id}")
            node_id = task.pytest_node_id
            if not _NODE_PATTERN.fullmatch(node_id):
                raise ValueError(f"invalid pytest node id: {node_id}")
            file_part = node_id.split("::", 1)[0]
            test_file = (self._repository_root / file_part).resolve(strict=True)
            if self._repository_root not in test_file.parents:
                raise ValueError(f"pytest node escapes repository: {node_id}")
        return list(manifest.tasks)

    # 用当前解释器隔离执行一个白名单 pytest node，并记录真实 JUnit 与清理证据
    async def run_attempt(
        self,
        task: EvalTaskSpec,
        *,
        repetition: int,
        eval_run_id: str,
    ) -> TaskAttemptResult:
        del eval_run_id
        node_id = task.pytest_node_id
        expected = ALLOWED_RUNTIME_CASES.get(task.id)
        if node_id is None or node_id != expected:
            raise ValueError(f"runtime case is not allowlisted: {task.id}")
        started_at = utc_now()
        started_clock = asyncio.get_running_loop().time()
        temporary = tempfile.TemporaryDirectory(prefix="tars-eval-pytest-")
        temporary_path = Path(temporary.name)
        junit_path = temporary_path / "junit.xml"
        process: asyncio.subprocess.Process | None = None
        stdout = b""
        stderr = b""
        exit_code: int | None = None
        status = EvalStatus.error
        error_type: str | None = None
        error: str | None = None
        timed_out = False
        evaluation: dict[str, JsonValue] = {
            "pytest_node_id": node_id,
            "cleanup_assertion": task.cleanup_assertion,
        }
        cleanup = CleanupMetrics()
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pytest",
                node_id,
                "-q",
                "--disable-warnings",
                "--maxfail=1",
                f"--junitxml={junit_path}",
                cwd=self._repository_root,
                env=_sanitized_environment(self._repository_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_process_group_options(),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=task.timeout_s,
                )
            except TimeoutError:
                timed_out = True
                await _terminate_process_tree(process)
                stdout, stderr = await process.communicate()
                error_type = "pytest_timeout"
                error = f"pytest exceeded timeout_s={task.timeout_s:g}"
            except asyncio.CancelledError:
                await _terminate_process_tree(process)
                cleanup.runtime_cleanup_completed = True
                raise
            exit_code = process.returncode
            cleanup.runtime_cleanup_completed = True
            if timed_out:
                evaluation.update(
                    {
                        "exit_code": exit_code,
                        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                        "stdout_summary": _output_summary(stdout),
                        "stderr_summary": _output_summary(stderr),
                        "cleanup_assertion_passed": False,
                    }
                )
            elif not junit_path.is_file():
                error_type = "missing_junit"
                error = f"pytest exited {exit_code} without JUnit evidence"
            else:
                tests, failures, errors, skipped = _junit_counts(junit_path)
                evaluation.update(
                    {
                        "exit_code": exit_code,
                        "tests": tests,
                        "failures": failures,
                        "errors": errors,
                        "skipped": skipped,
                        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                        "stdout_summary": _output_summary(stdout),
                        "stderr_summary": _output_summary(stderr),
                    }
                )
                if exit_code == 0 and tests == 1 and skipped == 0:
                    status = EvalStatus.passed
                    evaluation["cleanup_assertion_passed"] = True
                elif exit_code == 0 and skipped:
                    status = EvalStatus.skipped
                    error_type = "pytest_skipped"
                    error = "allowlisted runtime case was skipped"
                else:
                    status = EvalStatus.failed
                    error_type = "pytest_failed"
                    error = f"pytest exited {exit_code}"
                    evaluation["cleanup_assertion_passed"] = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_type = "pytest_runner_error"
            error = str(exc)
        finally:
            temporary.cleanup()
            cleanup.workspace_removed = not temporary_path.exists()

        elapsed = max(0, int((asyncio.get_running_loop().time() - started_clock) * 1000))
        return TaskAttemptResult(
            task_id=task.id,
            source_task_id=node_id,
            repetition=repetition,
            status=status,
            run_terminal_status=(
                "pytest_passed"
                if status == EvalStatus.passed
                else (
                    "pytest_skipped"
                    if status == EvalStatus.skipped
                    else ("pytest_timeout" if timed_out else "pytest_failed")
                )
            ),
            goal_completed=(
                status == EvalStatus.passed
                if status in {EvalStatus.passed, EvalStatus.failed}
                else None
            ),
            collateral_damage=None,
            step_count=None,
            started_at=started_at,
            finished_at=utc_now(),
            latency_ms=elapsed,
            score=(
                1.0
                if status == EvalStatus.passed
                else (0.0 if status == EvalStatus.failed else None)
            ),
            output=_output_summary(stdout),
            error_type=error_type,
            error=error,
            usage=UsageMetrics(
                input_tokens=0,
                output_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                llm_latency_ms=0,
            ),
            tools=ToolMetrics(),
            sandbox=SandboxMetrics(
                requested=None,
                network_mode="not_observed",
                isolated_workspace=None,
                fallback_used=None,
                oom_killed=None,
                timed_out=timed_out,
                cleanup_completed=cleanup.runtime_cleanup_completed,
            ),
            cleanup=cleanup,
            evaluation=evaluation,
        )


__all__ = ["ALLOWED_RUNTIME_CASES", "InternalRuntimeCaseAdapter"]
