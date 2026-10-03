from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from scripts import qa


def test_missing_qa_dependency_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(qa.QaError, match="Missing required dependencies"):
        qa.require("pytest_timeout")


def test_changed_coverage_cannot_switch_repository_base() -> None:
    with pytest.raises(qa.QaError, match="cannot replace"):
        qa.fixed_base("HEAD")


def test_missing_repository_baseline_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa, "ROOT", tmp_path)
    with pytest.raises(qa.QaError, match="missing"):
        qa.fixed_base()


def test_baseline_follows_initialization_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=tmp_path, text=True).strip()

    git("init", "--initial-branch=main")
    git("config", "user.name", "Baseline Test")
    git("config", "user.email", "baseline@example.test")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "LICENSE").write_text("Test license\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "Initialize")
    baseline = git("rev-parse", "HEAD")
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "Add module")
    monkeypatch.setattr(qa, "ROOT", tmp_path)
    assert qa.fixed_base() == baseline
    assert qa.fixed_base(baseline) == baseline
    with pytest.raises(qa.QaError, match="cannot replace"):
        qa.fixed_base(git("rev-parse", "HEAD"))


def test_failed_command_keeps_diagnostics_and_fails_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = [sys.executable, "-c",
               "import sys; print('output'); print('diagnostic', file=sys.stderr); sys.exit(7)"]
    monkeypatch.setattr(qa, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(qa, "quick", lambda base: qa.run(command, label="failure"))
    monkeypatch.setattr(sys, "argv", ["qa.py", "quick"])

    assert qa.main() == 1
    assert (tmp_path / "failure.log").read_text().splitlines() == ["output", "diagnostic"]
    record = json.loads((tmp_path / "failure.command.json").read_text())
    assert record["command"] == command
    assert record["returncode"] == 7
    assert record["cwd"] == str(qa.ROOT)


def test_command_start_failure_also_keeps_a_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = [str(tmp_path / "missing-executable")]
    monkeypatch.setattr(qa, "ARTIFACT_ROOT", tmp_path)
    with pytest.raises(OSError):
        qa.run(command, label="not-started")
    assert "FileNotFoundError" in (tmp_path / "not-started.log").read_text(encoding="utf-8")
    record = json.loads((tmp_path / "not-started.command.json").read_text())
    assert record["command"] == command
    assert record["returncode"] is None


def test_command_record_does_not_overwrite_the_tools_json_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qa, "ARTIFACT_ROOT", tmp_path)
    report = tmp_path / "diff-coverage.json"
    command = [sys.executable, "-c",
               "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('{\"coverage\": 81}')",
               str(report)]
    qa.run(command, label="diff-coverage")
    assert json.loads(report.read_text()) == {"coverage": 81}
    assert json.loads((tmp_path / "diff-coverage.command.json").read_text())["returncode"] == 0


def test_upload_copies_only_qa_evidence_and_redacts_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qa, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setenv("EXAMPLE_API_KEY", "private-test-value")
    raw = "diagnostic private-test-value api_key=another-secret https://user:pass@example.test/x"
    (tmp_path / "mypy.log").write_text(raw, encoding="utf-8")
    (tmp_path / "private.db").write_text("private data", encoding="utf-8")

    qa.prepare_artifacts()

    assert {path.name for path in (tmp_path / "upload").iterdir()} == {"mypy.log"}
    uploaded = (tmp_path / "upload/mypy.log").read_text(encoding="utf-8")
    assert "diagnostic" in uploaded
    for secret in ("private-test-value", "another-secret", "user:pass"):
        assert secret not in uploaded
    assert (tmp_path / "mypy.log").read_text(encoding="utf-8") == raw


def test_ci_requires_security_and_real_docker_recovery_without_skipped_success() -> None:
    workflow = yaml.safe_load((qa.ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    quality = workflow["jobs"]["quality"]
    assert set(quality["strategy"]["matrix"]["os"]) == {"ubuntu-latest", "windows-latest"}
    steps = quality["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    assert "check_public_content.py --history --refs HEAD" in commands
    assert "-m bandit -c pyproject.toml -r src scripts -ll -ii" in commands
    assert "-m pip_audit --local" in commands
    assert "scripts/qa.py coverage" in commands
    assert "docker image inspect --format '{{.Id}}'" in commands
    assert "^sha256:[0-9a-f]{64}$" in commands
    assert 'CORE_RECOVERY_IMAGE=$image_id' in commands
    roots = []
    recovery_indexes = []
    for test in ("test_docker_core_recovery.py", "test_docker_worker_path_boundary.py"):
        selected = [(index, step) for index, step in enumerate(steps) if test in step.get("run", "")]
        assert len(selected) == 1
        index, step = selected[0]
        recovery_indexes.append(index)
        assert step["if"] == "runner.os == 'Linux'"
        assert step["env"]["RUN_CORE_RECOVERY_DOCKER"] == "1"
        assert not step.get("continue-on-error", False)
        run = step["run"]
        assert 'test ! -e "$CORE_RECOVERY_ROOT"' in run
        assert "len(cases) == 1" in run and "('skipped', 'failure', 'error')" in run
        assert "result['passed'] is True" in run
        assert "result['image'] == os.environ['CORE_RECOVERY_IMAGE']" in run
        roots.append(next(line for line in run.splitlines() if line.startswith("export CORE_RECOVERY_ROOT=")))
    assert len(set(roots)) == 2
    assert all("/build/core-recovery/" in root for root in roots)
    cleanup_index, cleanup = next((index, step) for index, step in enumerate(steps)
                                  if "orphaned TARS-Agent containers" in step["name"])
    assert cleanup_index > max(recovery_indexes)
    assert cleanup["if"] == "runner.os == 'Linux' && always()"
    assert "exit 1" in cleanup["run"]
    web_commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["web-e2e"]["steps"])
    assert "npm audit --audit-level=high" in web_commands


def test_ci_independent_install_uses_locked_wheel_and_unchanged_verifier_before_cleanup() -> None:
    workflow = yaml.safe_load((qa.ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["quality"]["steps"]
    selected = [(index, step) for index, step in enumerate(steps)
                if step["name"] == "Verify complete independent wheel installation (Ubuntu)"]
    assert len(selected) == 1
    index, step = selected[0]
    assert step["if"] == "runner.os == 'Linux'" and not step.get("continue-on-error", False)
    run = step["run"]
    assert 'case_root="$RUNNER_TEMP/tars-installed-' in run
    assert 'test ! -e "$case_root"' in run
    assert "uv export --locked --no-default-groups --no-emit-project" in run
    assert 'uv pip sync --python "$case_root/venv/bin/python" --require-hashes' in run
    assert 'uv pip install --python "$case_root/venv/bin/python" --no-deps "${wheels[0]}"' in run
    assert 'cmp scripts/verify_installed.py "$case_root/verify_installed.py"' in run
    assert "unset PYTHONPATH PYTHONHOME" in run and 'cd "$case_root"' in run
    assert "TARS_CONFIG=str(root / 'empty-config.toml')" in run
    assert "TARS_SANDBOX_MODE='required', TARS_SANDBOX_IMAGE=image" in run
    assert "'?mode=ro'" in run and "requests == 0" in run
    for requirement in ("four_entrypoint_help", "schema_resources", "installed_core", "installed_worker",
                        "installed_web_health", "core_stopped", "core_port_closed", "core_control_removed",
                        "web_stopped", "web_port_closed"):
        assert requirement in run
    cleanup_index = next(index for index, step in enumerate(steps)
                         if step["name"] == "Assert and remove orphaned TARS-Agent containers")
    assert index < cleanup_index
    assert any(step["name"] == "Smoke-test isolated wheel install" and "if" not in step for step in steps)


def test_complete_install_evidence_is_sanitized_for_ci_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setenv("EXAMPLE_API_KEY", "fixture-private-value")
    result = {"installed_core": "passed", "model_requests": 0, "diagnostic": "fixture-private-value"}
    (tmp_path / "installed-verification.json").write_text(json.dumps(result), encoding="utf-8")
    qa.prepare_artifacts()
    uploaded = json.loads((tmp_path / "upload/installed-verification.json").read_text(encoding="utf-8"))
    assert uploaded == {"installed_core": "passed", "model_requests": 0, "diagnostic": "[REDACTED]"}
