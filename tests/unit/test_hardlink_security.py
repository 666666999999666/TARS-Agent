from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from tars_agent.sandbox import worker


def _linked_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-original", encoding="utf-8")
    target = workspace / "链接文件.txt"
    os.link(outside, target)
    assert target.stat().st_nlink > 1
    return workspace, outside, target


@pytest.mark.parametrize("operation", ["read", "write"])
def test_existing_hardlink_is_rejected_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    workspace, outside, target = _linked_workspace(tmp_path)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("existing multi-link files must be rejected before opening")

    monkeypatch.setattr(worker.os, "open", forbidden_open)
    with pytest.raises(worker.SandboxPolicyError, match="multiple hard links"):
        if operation == "read":
            worker._read_file({"path": target.name}, workspace, 32)
        else:
            worker._write_file({"path": target.name, "content": "replacement"}, workspace)
    assert outside.read_text(encoding="utf-8") == "outside-original"
    assert target.read_text(encoding="utf-8") == "outside-original"


@pytest.mark.parametrize("operation", ["read", "write"])
def test_opened_descriptor_is_checked_if_entry_becomes_a_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_text("inside-original", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-original", encoding="utf-8")
    real_open = os.open
    descriptors: list[int] = []

    def swapped_open(path, flags, mode=0o777, **kwargs):
        if Path(path) == target:
            assert not flags & os.O_TRUNC
            target.unlink()
            os.link(outside, target)
        descriptor = real_open(path, flags, mode, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(worker.os, "open", swapped_open)
    with pytest.raises(worker.SandboxPolicyError, match="multiple hard links"):
        if operation == "read":
            worker._read_file({"path": target.name}, workspace, 32)
        else:
            worker._write_file({"path": target.name, "content": "replacement"}, workspace)
    assert outside.read_text(encoding="utf-8") == "outside-original"
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


async def test_worker_reports_hardlink_rejection_as_policy_error(tmp_path: Path) -> None:
    workspace, outside, target = _linked_workspace(tmp_path)
    for name in ("read_file", "write_file"):
        result = await worker.execute_payload(
            {"tool_name": name, "params": {"path": target.name, "content": "replacement"}},
            workspace, sandboxed=False,
        )
        assert result["is_error"] is True
        assert result["error_type"] == "sandbox_policy_denied"
    assert outside.read_text(encoding="utf-8") == "outside-original"


def test_single_link_chinese_file_creates_and_replaces_content(tmp_path: Path) -> None:
    relative = "中文目录/报告.txt"
    result = worker._write_file({"path": relative, "content": "第一份完整内容"}, tmp_path)
    assert result["is_error"] is False
    result = worker._write_file({"path": relative, "content": "短"}, tmp_path)
    assert result["is_error"] is False
    assert worker._read_file({"path": relative}, tmp_path, 100)["content"] == "短"


def test_read_stays_bounded_after_descriptor_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * 100_000)
    real_fdopen = os.fdopen
    read_sizes = []

    class ReadSpy:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            read_sizes.append(size)
            return self.stream.read(size)

    def tracked_fdopen(*args, **kwargs):
        return ReadSpy(real_fdopen(*args, **kwargs))

    monkeypatch.setattr(worker.os, "fdopen", tracked_fdopen)
    result = worker._read_file({"path": path.name}, tmp_path, 32)
    assert read_sizes == [33]
    assert result["content"] == "x" * 32 + "\n[truncated]"
    assert result["truncated"] is True


@pytest.mark.skipif(os.name != "nt" and getattr(os, "geteuid", lambda: -1)() == 0,
                    reason="root bypasses ordinary file write permissions")
def test_readonly_target_keeps_original_content(tmp_path: Path) -> None:
    target = tmp_path / "readonly.txt"
    target.write_text("original", encoding="utf-8")
    target.chmod(stat.S_IREAD)
    try:
        with pytest.raises(PermissionError):
            worker._write_file({"path": target.name, "content": "replacement"}, tmp_path)
        assert target.read_text(encoding="utf-8") == "original"
    finally:
        target.chmod(stat.S_IREAD | stat.S_IWRITE)
