from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tars_agent.core.tools.builtin import BashTool, ListDirTool, WriteFileTool
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter, ToolCallContext


def _context(root: Path, tool_use_id: str = "tool-1") -> ToolCallContext:
    return ToolCallContext(
        run_id="run-1",
        session_id="sess-1",
        tool_use_id=tool_use_id,
        workspace_root=root,
    )


def _runtime() -> tuple[RuntimeRouter, FakeRuntime]:
    fake = FakeRuntime()
    return RuntimeRouter(fake), fake


@pytest.mark.asyncio
async def test_bash_success_stdout(tmp_path: Path) -> None:
    runtime, fake = _runtime()
    result = await BashTool(runtime).invoke(
        {"command": "echo hello"},
        context=_context(tmp_path),
    )
    assert not result.is_error
    assert "hello" in result.content
    assert result.backend == "workspace_sandbox"
    assert len(fake.requests) == 1


@pytest.mark.asyncio
async def test_bash_nonzero_exit_is_not_retryable(tmp_path: Path) -> None:
    runtime, _ = _runtime()
    result = await BashTool(runtime).invoke(
        {"command": "exit 2"},
        context=_context(tmp_path),
    )
    assert result.is_error
    assert "[exit 2]" in result.content
    assert result.retryable is False


@pytest.mark.asyncio
async def test_bash_timeout(tmp_path: Path) -> None:
    runtime, _ = _runtime()
    result = await BashTool(runtime).invoke(
        {
            "command": subprocess.list2cmdline(
                [sys.executable, "-c", "import time; time.sleep(5)"]
            ),
            "timeout": 1,
        },
        context=_context(tmp_path),
    )
    assert result.is_error
    assert result.error_type == "timeout"


@pytest.mark.asyncio
async def test_write_file_creates_relative_file(tmp_path: Path) -> None:
    runtime, _ = _runtime()
    result = await WriteFileTool(runtime).invoke(
        {"path": "a/b/out.txt", "content": "hello world"},
        context=_context(tmp_path),
    )
    assert not result.is_error
    assert (tmp_path / "a" / "b" / "out.txt").read_text() == "hello world"


@pytest.mark.asyncio
async def test_write_file_rejects_workspace_escape(tmp_path: Path) -> None:
    runtime, _ = _runtime()
    result = await WriteFileTool(runtime).invoke(
        {"path": "../secret.txt", "content": "x"},
        context=_context(tmp_path),
    )
    assert result.is_error
    assert result.error_type == "sandbox_policy_denied"


@pytest.mark.asyncio
async def test_list_dir_shows_files_and_respects_depth(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("x")
    child = tmp_path / "child"
    child.mkdir()
    grandchild = child / "grandchild"
    grandchild.mkdir()
    (grandchild / "deep.txt").write_text("x")
    runtime, _ = _runtime()
    result = await ListDirTool(runtime).invoke(
        {"path": ".", "max_depth": 1},
        context=_context(tmp_path),
    )
    assert not result.is_error
    assert "foo.py" in result.content
    assert "child" in result.content
    assert "deep.txt" not in result.content


@pytest.mark.asyncio
async def test_workspace_tool_without_context_fails_closed() -> None:
    runtime, fake = _runtime()
    result = await BashTool(runtime).invoke({"command": "echo unsafe"})
    assert result.is_error
    assert result.error_type == "sandbox_policy_denied"
    assert fake.requests == []
