from __future__ import annotations

import sys

from mcp.server import MCPServer

server = MCPServer("kama-test-http")


@server.tool()
def multiply(a: int, b: int) -> dict[str, int]:
    """Multiply two integers."""
    return {"product": a * b}


if __name__ == "__main__":
    server.run(
        "streamable-http",
        host="127.0.0.1",
        port=int(sys.argv[1]),
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
    )
