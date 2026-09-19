from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
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
    assert "FileNotFoundError" in (tmp_path / "not-started.log").read_text()
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
