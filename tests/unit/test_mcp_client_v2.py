from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types

from tars_agent.core.config import McpServerConfig
from tars_agent.core.mcp.client import (
    McpClient,
    McpServerUnavailableError,
    _expand_headers,
    _stdio_environment,
)


def test_stdio_environment_does_not_inherit_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy")
    env = _stdio_environment({"EXPLICIT": "yes"})
    assert "ANTHROPIC_API_KEY" not in env
    assert "HTTP_PROXY" not in env
    assert env["EXPLICIT"] == "yes"


def test_header_env_reference_supports_safe_embedded_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_TOKEN", "token-value")
    assert _expand_headers({"Authorization": "${MCP_TOKEN}"}) == {
        "Authorization": "token-value"
    }
    assert _expand_headers({"Authorization": "Bearer ${MCP_TOKEN}"}) == {
        "Authorization": "Bearer token-value"
    }
    assert _expand_headers({"X-Composite": "${MCP_TOKEN}:${MCP_TOKEN}"}) == {
        "X-Composite": "token-value:token-value"
    }
    with pytest.raises(ValueError, match="missing environment variable"):
        _expand_headers({"Authorization": "${MISSING}"})
    with pytest.raises(ValueError, match="invalid MCP header"):
        _expand_headers({"Authorization": "Bearer ${INVALID-NAME}"})


async def test_structured_content_is_used_without_text() -> None:
    config = McpServerConfig(name="x", trusted=True, command="python")
    wrapper = McpClient(config)
    sdk = MagicMock()
    sdk.call_tool = AsyncMock(
        return_value=types.CallToolResult(
            content=[],
            structured_content={"count": 1},
            is_error=False,
        )
    )
    wrapper._client = sdk
    result = await wrapper.call_tool("count", {})
    assert result.content == '{"count": 1}'
    assert not result.is_error


async def test_binary_content_is_omitted_from_context() -> None:
    config = McpServerConfig(name="x", trusted=True, command="python")
    wrapper = McpClient(config)
    sdk = MagicMock()
    sdk.call_tool = AsyncMock(
        return_value=types.CallToolResult(
            content=[types.ImageContent(data="BASE64SECRET", mime_type="image/png")],
        )
    )
    wrapper._client = sdk
    result = await wrapper.call_tool("image", {})
    assert "BASE64SECRET" not in result.content
    assert "image content omitted" in result.content


async def test_connect_stdio_uses_official_client_and_minimal_env() -> None:
    config = McpServerConfig(
        name="x",
        trusted=True,
        command="python",
        args=["server.py"],
        env={"EXPLICIT": "yes"},
    )
    fake_sdk = MagicMock()
    fake_sdk.__aenter__ = AsyncMock(return_value=fake_sdk)
    with patch("mcp.Client", return_value=fake_sdk), patch(
        "mcp.client.stdio.stdio_client", return_value=MagicMock()
    ) as transport:
        wrapper = McpClient(config)
        await wrapper.connect()
    parameters = transport.call_args.args[0]
    assert Path(parameters.command).is_absolute()
    assert Path(parameters.command).is_file()
    assert parameters.args == ["server.py"]
    assert parameters.env["EXPLICIT"] == "yes"
    assert "ANTHROPIC_API_KEY" not in parameters.env


async def test_connect_failure_unwinds_partially_entered_sdk_client() -> None:
    config = McpServerConfig(name="x", trusted=True, command="python")
    fake_sdk = MagicMock()
    failure = RuntimeError("discovery failed")
    fake_sdk.__aenter__ = AsyncMock(side_effect=failure)
    fake_sdk.__aexit__ = AsyncMock()

    with patch("mcp.Client", return_value=fake_sdk), patch(
        "mcp.client.stdio.stdio_client", return_value=MagicMock()
    ):
        wrapper = McpClient(config)
        with pytest.raises(Exception, match="discovery failed"):
            await wrapper.connect()

    fake_sdk.__aexit__.assert_awaited_once_with(
        RuntimeError,
        failure,
        failure.__traceback__,
    )


async def test_missing_stdio_command_fails_before_starting_sdk_transport(
    tmp_path: Path,
) -> None:
    config = McpServerConfig(
        name="missing",
        trusted=True,
        command=str(tmp_path / "missing-server"),
    )
    with patch("mcp.Client") as client_factory:
        with pytest.raises(McpServerUnavailableError, match="command was not found"):
            await McpClient(config).connect()
    client_factory.assert_not_called()


@pytest.mark.parametrize("command", ["./server", "tools/server", r"tools\server", r"C:server"])
def test_mcp_rejects_directory_relative_commands(command: str) -> None:
    from tars_agent.core.mcp.client import _resolve_stdio_command
    with pytest.raises(ValueError, match="absolute path or a bare"):
        _resolve_stdio_command(command)


def test_mcp_path_excludes_project_executable(tmp_path: Path) -> None:
    import os

    from tars_agent.core.mcp.client import _resolve_stdio_command
    project = tmp_path / "project"
    trusted = tmp_path / "trusted"
    project.mkdir()
    trusted.mkdir()
    name = "server.exe" if os.name == "nt" else "server"
    for directory in (project, trusted):
        binary = directory / name
        binary.write_text("test", encoding="utf-8")
        binary.chmod(0o755)
    resolved = _resolve_stdio_command(name, environ={"PATH": os.pathsep.join([str(project), str(trusted)]), "PATHEXT": ".EXE"}, cwd=project)
    assert resolved == str((trusted / name).resolve())


async def test_mcp_timeout_changes_runtime_health() -> None:
    import asyncio
    wrapper = McpClient(McpServerConfig(name="x", trusted=True, command="python", tool_timeout_s=0.01))
    sdk = MagicMock()
    sdk.call_tool = AsyncMock(side_effect=lambda *args, **kwargs: None)
    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()
    sdk.call_tool = blocked
    wrapper._client = sdk
    wrapper.health_status = "connected"
    with pytest.raises(McpServerUnavailableError, match="timed out"):
        await wrapper.call_tool("slow", {})
    assert wrapper.health_status == "degraded"
    assert "timed out" in wrapper.last_error


async def test_mcp_cancel_one_call_does_not_cancel_concurrent_call() -> None:
    import asyncio
    wrapper = McpClient(McpServerConfig(name="x", trusted=True, command="python"))
    entered = asyncio.Event()
    sdk = MagicMock()
    async def call(name, arguments, **kwargs):
        if name == "slow":
            entered.set()
            await asyncio.Event().wait()
        return types.CallToolResult(content=[types.TextContent(text="done")])
    sdk.call_tool = call
    wrapper._client = sdk
    slow = asyncio.create_task(wrapper.call_tool("slow", {}))
    await entered.wait()
    slow.cancel()
    with pytest.raises(asyncio.CancelledError):
        await slow
    assert (await wrapper.call_tool("fast", {})).content == "done"
    assert "external effects may continue" in wrapper.last_error
