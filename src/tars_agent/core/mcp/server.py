from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace

from tars_agent.core.config import McpServerConfig
from tars_agent.core.mcp.client import McpClient
from tars_agent.core.mcp.tool import McpTool
from tars_agent.core.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class McpServerStatus:
    name: str
    transport: str
    status: str
    tool_count: int
    protocol_version: str | None = None
    error: str | None = None


class McpServerManager:
    def __init__(self) -> None:
        self._clients: dict[str, McpClient] = {}
        self._tools: list[McpTool] = []
        self._statuses: dict[str, McpServerStatus] = {}

    async def start_all(self, servers: list[McpServerConfig]) -> None:
        names = [config.name for config in servers]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError(f"duplicate MCP server name: {', '.join(duplicate_names)}")
        for config in servers:
            client = McpClient(config)
            try:
                await client.connect()
                definitions = await client.list_tools()
                tool_names = [definition.name for definition in definitions]
                duplicate_tools = sorted(
                    {name for name in tool_names if tool_names.count(name) > 1}
                )
                if duplicate_tools:
                    raise ValueError(
                        "duplicate MCP tool name in "
                        f"server '{config.name}': {', '.join(duplicate_tools)}"
                    )
                qualified = {f"{config.name}__{definition.name}" for definition in definitions}
                existing = {tool.name for tool in self._tools}
                collisions = sorted(qualified & existing)
                if collisions:
                    raise ValueError(f"MCP tool name collision: {', '.join(collisions)}")
                tools = [McpTool(client, config.name, definition) for definition in definitions]
                self._tools.extend(tools)
                self._clients[config.name] = client
                self._statuses[config.name] = McpServerStatus(
                    name=config.name,
                    transport=config.transport,
                    status="connected",
                    tool_count=len(tools),
                    protocol_version=client.protocol_version,
                )
            except (Exception, asyncio.CancelledError) as exc:
                try:
                    await client.close()
                except Exception:
                    log.exception(
                        "mcp: server '%s' cleanup failed after startup error",
                        config.name,
                    )
                if isinstance(exc, asyncio.CancelledError):
                    raise
                self._statuses[config.name] = McpServerStatus(
                    name=config.name,
                    transport=config.transport,
                    status="failed",
                    tool_count=0,
                    error=str(exc),
                )
                log.exception("mcp: server '%s' failed to start", config.name)

    def register_tools(self, registry: ToolRegistry) -> None:
        for tool in self._tools:
            registry.register(tool)

    def get_tools(self) -> list[McpTool]:
        return list(self._tools)

    def statuses(self) -> list[McpServerStatus]:
        result = []
        for name in sorted(self._statuses):
            status = self._statuses[name]
            client = self._clients.get(name)
            health = getattr(client, "health_status", None)
            if client is not None and isinstance(health, str):
                status = replace(status, status=health, error=client.last_error)
            result.append(status)
        return result

    async def stop_all(self) -> None:
        for name, client in list(self._clients.items()):
            try:
                await client.close()
            except Exception:
                log.exception("mcp: error closing server '%s'", name)
        for name, status in self._statuses.items():
            if name in self._clients:
                self._statuses[name] = replace(status, status="stopped")
        self._clients.clear()
        self._tools.clear()


__all__ = ["McpServerManager", "McpServerStatus"]
