from __future__ import annotations

import importlib.util
import subprocess
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
