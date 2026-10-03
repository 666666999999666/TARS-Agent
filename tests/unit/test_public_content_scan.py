from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts import check_public_content as scanner


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=True, text=True, encoding="utf-8",
    )
    return result.stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--initial-branch=main")
    _git(tmp_path, "config", "user.name", "Public Scan Test")
    _git(tmp_path, "config", "user.email", "scan@example.test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Public fixture\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore", "README.md")
    _git(tmp_path, "commit", "-m", "Public baseline")
    return tmp_path


def _scan(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    program = """import sys
from pathlib import Path
from scripts import check_public_content as scanner
scanner.ROOT = Path(sys.argv.pop(1))
raise SystemExit(scanner.main())
"""
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-c", program, str(root), *args],
        cwd=scanner.ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )


def test_default_head_scans_branch_secret_removed_from_current_tree(repository: Path) -> None:
    _git(repository, "switch", "-c", "candidate")
    canary = hashlib.sha256(b"generated delivery branch regression fixture").hexdigest()
    secret_path = repository / "temporary.py"
    secret_path.write_text('api_key = "' + canary + '"\n', encoding="utf-8")
    _git(repository, "add", "temporary.py")
    _git(repository, "commit", "-m", "Add synthetic historical canary")
    secret_path.unlink()
    _git(repository, "add", "temporary.py")
    _git(repository, "commit", "-m", "Remove synthetic historical canary")
    head = _git(repository, "rev-parse", "HEAD")

    baseline = _scan(repository, "--history", "--refs", "main")
    assert baseline.returncode == 0, baseline.stderr
    candidate = _scan(repository, "--history")
    assert candidate.returncode == 1, candidate.stderr
    report = json.loads((repository / "build/qa/public-content.json").read_text(encoding="utf-8"))
    assert report["refs"] == ["HEAD"]
    assert report["resolved_refs"] == {"HEAD": head}
    assert any(item["path"] == "temporary.py" and item["object"] != "worktree" for item in report["findings"])
    assert canary not in candidate.stdout + candidate.stderr + json.dumps(report)


@pytest.mark.parametrize("reference", ["HEAD", "refs/pull/17/merge", "raw-sha", "HEAD~0"])
def test_history_accepts_real_commitish_values(repository: Path, reference: str) -> None:
    head = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-ref", "refs/pull/17/merge", head)
    chosen = head if reference == "raw-sha" else reference
    result = _scan(repository, "--history", "--refs", chosen)
    assert result.returncode == 0, result.stderr
    report = json.loads((repository / "build/qa/public-content.json").read_text(encoding="utf-8"))
    assert report["resolved_refs"] == {chosen: head}
    assert report["history_blobs"] == 2


@pytest.mark.parametrize("refs", [["missing"], ["HEAD", "missing"], ["HEAD^{tree}"], ["--help"]])
def test_invalid_or_mixed_refs_fail_instead_of_scanning_only_known_refs(repository: Path, refs: list[str]) -> None:
    result = _scan(repository, "--history", "--refs", *refs)
    assert result.returncode != 0
    assert not (repository / "build/qa/public-content.json").exists()
    assert "does not resolve to a commit" in result.stderr or "expected at least one argument" in result.stderr


def test_explicit_refs_without_history_are_not_silently_ignored(repository: Path) -> None:
    result = _scan(repository, "--refs", "HEAD")
    assert result.returncode == 2
    assert "--refs requires --history" in result.stderr
