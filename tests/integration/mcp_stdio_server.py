from __future__ import annotations

import asyncio

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("kama-test-stdio")
calls = 0


@server.tool()
def add(a: int, b: int) -> dict[str, int]:
    """Add two integers."""
    return {"sum": a + b}


@server.tool()
async def count_side_effect(mode: str) -> dict[str, int]:
    """Increment once, then fail or wait long enough for the client timeout."""
    global calls
    calls += 1
    if mode == "fail":
        raise ToolError("intentional failure after side effect")
    if mode == "timeout":
        await asyncio.sleep(30)
    return {"calls": calls}


@server.tool()
def get_count() -> dict[str, int]:
    """Return the number of side-effect tool body entries."""
    return {"calls": calls}


if __name__ == "__main__":
    server.run()
