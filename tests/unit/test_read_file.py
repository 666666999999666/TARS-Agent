from __future__ import annotations

from pathlib import Path

from tars_agent.core.tools.builtin.read_file import ReadFileTool
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter, ToolCallContext


async def _read(root: Path, path: str):  # type: ignore[no-untyped-def]
    tool = ReadFileTool(RuntimeRouter(FakeRuntime()))
    return await tool.invoke(
        {"path": path},
        context=ToolCallContext("run", "session", "tool", root),
    )


async def test_read_existing_file(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    result = await _read(tmp_path, "hello.txt")
    assert not result.is_error
    assert result.content == "hello world"


async def test_file_not_found_is_structured_error(tmp_path: Path) -> None:
    result = await _read(tmp_path, "missing.txt")
    assert result.is_error
    assert result.error_type == "not_found"


async def test_path_traversal_is_denied(tmp_path: Path) -> None:
    result = await _read(tmp_path, "../secret.txt")
    assert result.is_error
    assert result.error_type == "sandbox_policy_denied"


async def test_truncation_over_512kb(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_bytes(b"x" * (600 * 1024))
    result = await _read(tmp_path, "big.txt")
    assert not result.is_error
    assert result.content.endswith("[truncated]")


async def test_empty_file_returns_empty_content(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    result = await _read(tmp_path, "empty.txt")
    assert not result.is_error
    assert result.content == ""
