from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
from scripts import final_delivery as delivery


@pytest.fixture
def source_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    delivery.git(repo, "init", "--initial-branch=main")
    delivery.git(repo, "config", "user.name", "Delivery Test")
    delivery.git(repo, "config", "user.email", "delivery@example.test")
    delivery.git(repo, "config", "commit.gpgsign", "false")
    delivery.git(repo, "config", "core.autocrlf", "false")
    (repo / ".gitattributes").write_text("* text eol=lf\n", encoding="utf-8")
    (repo / ".gitignore").write_text("build/\n__pycache__/\n", encoding="utf-8")
    (repo / "LICENSE").write_text("Project test license.\n", encoding="utf-8")
    delivery.git(repo, "add", ".")
    delivery.git(repo, "commit", "-m", "Initialize repository")
    baseline = delivery.resolve(repo, "HEAD")
    (repo / "README.md").write_text("Accepted main content.\n", encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    scripts = repo / "scripts"
    scripts.mkdir()
    scanner = Path(delivery.__file__).with_name("check_public_content.py")
    (scripts / scanner.name).write_bytes(scanner.read_bytes())
    delivery.git(repo, "add", ".")
    delivery.git(repo, "commit", "-m", "Add project")
    return repo, baseline, delivery.resolve(repo, "HEAD")


def test_main_and_annotated_tag_export_and_restore(
    source_repo: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, baseline, main = source_repo
    delivery.git(repo, "branch", "work-in-progress", baseline)
    delivery.git(repo, "remote", "add", "origin", "https://example.test/user/project.git")
    delivery.git(repo, "tag", "-a", "stable-fixture", "-m", "Accepted fixture", main)
    tag_oid = delivery.git(repo, "rev-parse", "refs/tags/stable-fixture").decode().strip()
    source_refs = delivery.git(repo, "show-ref")
    output = tmp_path / "delivery"
    monkeypatch.setattr(sys, "argv", [
        "final_delivery.py", "--repo", str(repo), "--main-ref", main,
        "--stable-tag", "stable-fixture", "--output", str(output),
    ])
    assert delivery.main() == 0
    assert delivery.git(repo, "show-ref") == source_refs
    manifest = json.loads((output / "handoff-manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == main
    assert manifest["baseline_commit"] == baseline
    assert manifest["refs"] == {delivery.MAIN_REF: main, "refs/tags/stable-fixture": tag_oid}
    assert verification["bundle"]["exact_refs"] == manifest["refs"]
    assert verification["bundle"]["baseline_commit"] == baseline
    assert [row["branch"] for row in verification["bundle"]["restored_branches"]] == ["main"]
    assert (output / "handoff-manifest.sha256").read_text(encoding="ascii").startswith(
        delivery.file_sha256(output / "handoff-manifest.json") + "  "
    )
    with zipfile.ZipFile(next(output.glob("*.zip"))) as archive:
        assert archive.read("TARS-Agent/README.md") == b"Accepted main content.\n"


def test_preflight_accepts_main_without_tag(source_repo: tuple[Path, str, str]) -> None:
    repo, _, main = source_repo
    assert delivery.preflight(repo, main, None) == (main, {delivery.MAIN_REF: main})


@pytest.mark.parametrize("problem", ["wrong_tag", "wrong_main", "dirty", "wrong_branch"])
def test_preflight_rejects_unqualified_candidate(
    source_repo: tuple[Path, str, str], problem: str,
) -> None:
    repo, baseline, main = source_repo
    selected, tag = main, None
    if problem == "wrong_tag":
        delivery.git(repo, "tag", "other-version", baseline)
        tag = "other-version"
    elif problem == "wrong_main":
        selected = baseline
    elif problem == "dirty":
        (repo / "README.md").write_text("Uncommitted content\n", encoding="utf-8")
    else:
        delivery.git(repo, "switch", "-c", "candidate")
    with pytest.raises(delivery.DeliveryError):
        delivery.preflight(repo, selected, tag)


@pytest.mark.parametrize("name", [".env", "state.db", "build/result.txt"])
def test_preflight_rejects_committed_private_files(
    source_repo: tuple[Path, str, str], name: str,
) -> None:
    repo, _, _ = source_repo
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("local fixture content\n", encoding="utf-8")
    delivery.git(repo, "add", "--force", name)
    delivery.git(repo, "commit", "-m", "Private fixture")
    with pytest.raises(delivery.DeliveryError, match="private"):
        delivery.preflight(repo, delivery.resolve(repo, "HEAD"), None)
