from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tars_agent.sandbox.worker import (
    SandboxPolicyError,
    _terminate_process,
    execute_payload,
    resolve_workspace_path,
)


def _command(script: str) -> str:
    args = [sys.executable, "-c", script]
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


@pytest.mark.parametrize("path", ["NUL", "NUL ", "CON.txt", "folder/COM1", r"\\.\pipe\x", "a.txt:stream"])
def test_worker_rejects_device_and_stream_paths(tmp_path: Path, path: str) -> None:
    before = set(tmp_path.iterdir())
    with pytest.raises(SandboxPolicyError):
        resolve_workspace_path(tmp_path, path, write=True)
    assert set(tmp_path.iterdir()) == before


def test_worker_allows_absolute_path_inside_root(tmp_path: Path) -> None:
    target = tmp_path / "中文.txt"
    target.write_text("内容", encoding="utf-8")
    assert resolve_workspace_path(tmp_path, str(target)) == target.resolve()


async def test_worker_reads_only_limit_plus_one(tmp_path: Path) -> None:
    target = tmp_path / "large.txt"
    target.write_bytes(b"x" * 1_000_000)
    original_open = os.fdopen
    reads = []

    class Handle:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def read(self, size=-1):
            reads.append(size)
            return self.handle.read(size)

    def bounded_open(descriptor, *args, **kwargs):
        return Handle(original_open(descriptor, *args, **kwargs))

    with patch.object(os, "fdopen", bounded_open):
        result = await execute_payload(
            {"tool_name": "read_file", "params": {"path": "large.txt"}, "output_limit_bytes": 32},
            tmp_path, sandboxed=False,
        )
    assert reads == [33]
    assert result["truncated"] is True
    assert len(result["content"]) < 100


async def test_worker_timeout_kills_delayed_child(tmp_path: Path) -> None:
    marker = tmp_path / "late.txt"
    script = f"import pathlib,time;time.sleep(1.0);pathlib.Path({str(marker)!r}).write_text('late')"
    result = await execute_payload(
        {"tool_name": "bash", "params": {"command": _command(script), "timeout": 0.2}},
        tmp_path, sandboxed=False,
    )
    assert result["error_type"] == "timeout"
    await asyncio.sleep(1.1)
    assert not marker.exists()


async def test_worker_parent_exit_cannot_leave_descendant(tmp_path: Path) -> None:
    marker = tmp_path / "orphan.txt"
    child = f"import pathlib,time;time.sleep(0.8);pathlib.Path({str(marker)!r}).write_text('orphan')"
    script = f"import subprocess,sys;subprocess.Popen([sys.executable,'-c',{child!r}])"
    result = await execute_payload(
        {"tool_name": "bash", "params": {"command": _command(script), "timeout": 0.2}},
        tmp_path, sandboxed=False,
    )
    assert result.get("error_type") in {None, "timeout"}
    await asyncio.sleep(1.0)
    assert not marker.exists()


async def test_worker_cancellation_kills_delayed_child(tmp_path: Path) -> None:
    marker = tmp_path / "cancelled.txt"
    ready = tmp_path / "ready.txt"
    script = (f"import pathlib,time;pathlib.Path({str(ready)!r}).write_text('ready');"
              f"time.sleep(0.8);pathlib.Path({str(marker)!r}).write_text('late')")
    task = asyncio.create_task(execute_payload(
        {"tool_name": "bash", "params": {"command": _command(script), "timeout": 5}},
        tmp_path, sandboxed=False,
    ))
    try:
        async with asyncio.timeout(3):
            while not ready.exists():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.9)
        assert not marker.exists()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_worker_chinese_output(tmp_path: Path) -> None:
    result = await execute_payload(
        {"tool_name": "bash", "params": {"command": _command("print('中文输出')")}},
        tmp_path, sandboxed=False,
    )
    assert not result["is_error"]
    assert "中文输出" in result["content"]


async def test_posix_cleanup_kills_group_even_after_shell_exit() -> None:
    process = AsyncMock()
    process.returncode = 0
    process.pid = 12345
    with patch("tars_agent.sandbox.worker.os.name", "posix"), patch(
        "tars_agent.sandbox.worker.os.killpg", create=True
    ) as killpg, patch("tars_agent.sandbox.worker.signal.SIGKILL", 9, create=True):
        await _terminate_process(process)
    killpg.assert_called_once_with(12345, 9)
    process.wait.assert_awaited_once()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction boundary")
def test_worker_rejects_windows_junction_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    marker = outside / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")
    link = workspace / "junction"
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                            capture_output=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    try:
        with pytest.raises(SandboxPolicyError):
            resolve_workspace_path(workspace, "junction/marker.txt", write=True)
        assert marker.read_text(encoding="utf-8") == "unchanged"
    finally:
        os.rmdir(link)
