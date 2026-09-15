from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tars_agent.core.config import McpServerConfig
from tars_agent.core.mcp.client import McpServerUnavailableError, McpToolDef
from tars_agent.core.mcp.server import McpServerManager


def _config(name: str) -> McpServerConfig:
    return McpServerConfig(name=name, trusted=True, command="python")


async def test_one_server_failure_does_not_block_healthy_server() -> None:
    failed = MagicMock()
    failed.connect = AsyncMock(side_effect=McpServerUnavailableError("offline"))
    failed.close = AsyncMock()
    healthy = MagicMock()
    healthy.connect = AsyncMock()
    healthy.list_tools = AsyncMock(
        return_value=[McpToolDef("echo", "Echo", {"type": "object"})]
    )
    healthy.close = AsyncMock()
    healthy.protocol_version = "2025-11-25"
    with patch(
        "tars_agent.core.mcp.server.McpClient",
        side_effect=[failed, healthy],
    ):
        manager = McpServerManager()
        await manager.start_all([_config("failed"), _config("healthy")])

    assert [tool.name for tool in manager.get_tools()] == ["healthy__echo"]
    statuses = {status.name: status for status in manager.statuses()}
    assert statuses["failed"].status == "failed"
    assert "offline" in (statuses["failed"].error or "")
    assert statuses["healthy"].status == "connected"
    assert statuses["healthy"].tool_count == 1
    await manager.stop_all()
    healthy.close.assert_awaited_once()


async def test_duplicate_server_names_fail_before_starting_any_process() -> None:
    with patch("tars_agent.core.mcp.server.McpClient") as client_factory:
        manager = McpServerManager()
        with pytest.raises(ValueError, match="duplicate MCP server name"):
            await manager.start_all([_config("same"), _config("same")])
    client_factory.assert_not_called()


async def test_cancelled_discovery_closes_unregistered_client() -> None:
    import asyncio
    client = MagicMock()
    client.connect = AsyncMock()
    client.list_tools = AsyncMock(side_effect=asyncio.CancelledError)
    client.close = AsyncMock()
    with patch("tars_agent.core.mcp.server.McpClient", return_value=client):
        manager = McpServerManager()
        with pytest.raises(asyncio.CancelledError):
            await manager.start_all([_config("cancelled")])
    client.close.assert_awaited_once()
    assert not manager.get_tools()


async def test_status_reflects_call_failure_after_startup() -> None:
    client = MagicMock()
    client.connect = AsyncMock()
    client.list_tools = AsyncMock(return_value=[])
    client.close = AsyncMock()
    client.protocol_version = "2025-11-25"
    client.health_status = "connected"
    client.last_error = None
    with patch("tars_agent.core.mcp.server.McpClient", return_value=client):
        manager = McpServerManager()
        await manager.start_all([_config("x")])
    client.health_status = "degraded"
    client.last_error = "MCP tool call timed out"
    assert manager.statuses()[0].status == "degraded"
    assert manager.statuses()[0].error == "MCP tool call timed out"
    await manager.stop_all()
