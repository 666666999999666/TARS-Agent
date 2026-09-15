from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import ToolCallBlock
from tars_agent.core.mcp.client import (
    McpCallResult,
    McpClient,
    McpServerUnavailableError,
    McpToolDef,
)
from tars_agent.core.mcp.tool import McpTool
from tars_agent.core.tools.invocation import invoke_tool
from tars_agent.core.tools.registry import ToolRegistry


def _make_tool() -> tuple[McpTool, MagicMock]:
    client = MagicMock(spec=McpClient)
    client.call_tool = AsyncMock(return_value=McpCallResult("ok", False))
    tool = McpTool(
        client,
        "server",
        McpToolDef(
            name="counter",
            description="count once",
            input_schema={"type": "object", "properties": {}},
        ),
    )
    return tool, client


async def test_success_maps_official_result() -> None:
    tool, client = _make_tool()
    result = await tool.invoke({"value": 1})
    assert result.content == "ok"
    assert result.backend == "external"
    assert result.retryable is False
    client.call_tool.assert_awaited_once_with("counter", {"value": 1})


async def test_is_error_is_not_retryable_or_replayed() -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(return_value=McpCallResult("business failure", True))
    result = await tool.invoke({})
    assert result.is_error
    assert result.error_type == "remote_tool_error"
    assert result.retryable is False
    assert client.call_tool.await_count == 1


async def test_transport_failure_is_not_retryable() -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(side_effect=McpServerUnavailableError("server exited"))
    result = await tool.invoke({})
    assert result.is_error
    assert result.retryable is False
    assert client.call_tool.await_count == 1


async def test_mcp_side_effect_failure_is_not_transparently_replayed() -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(return_value=McpCallResult("business failure", True))
    registry = ToolRegistry()
    registry.register(tool)
    result = await invoke_tool(
        registry,
        ToolCallBlock(id="tool-1", name=tool.name, input={}),
        EventBus(),
        run_id="run-1",
    )
    assert result.is_error
    assert client.call_tool.await_count == 1


async def test_mcp_transport_failure_is_not_transparently_replayed() -> None:
    tool, client = _make_tool()
    client.call_tool = AsyncMock(side_effect=McpServerUnavailableError("timed out"))
    registry = ToolRegistry()
    registry.register(tool)
    result = await invoke_tool(
        registry,
        ToolCallBlock(id="tool-2", name=tool.name, input={}),
        EventBus(),
        run_id="run-2",
    )
    assert result.is_error
    assert result.retryable is False
    assert client.call_tool.await_count == 1


def test_schema_and_prefix() -> None:
    tool, _ = _make_tool()
    assert tool.name == "server__counter"
    assert tool.input_schema["type"] == "object"
