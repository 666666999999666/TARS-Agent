"""Task-bound remote-code MCP bridge; no benchmark management tools are exposed."""

from __future__ import annotations

import argparse
import asyncio
from urllib.parse import urlsplit

import httpx


def loopback_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or not parsed.port
    ):
        raise ValueError("AppWorld environment must be an explicit loopback HTTP endpoint")
    return value.rstrip("/")


class TaskCodeBridge:
    """One immutable task binding and one in-flight execution per bridge."""

    def __init__(self, environment_url: str, task_id: str) -> None:
        self.url = loopback_url(environment_url)
        self.task_id = task_id
        self._lock = asyncio.Lock()

    async def execute(self, code: str) -> str:
        if not code.strip() or len(code.encode("utf-8")) > 64 * 1024:
            raise ValueError("code must contain 1 to 65536 UTF-8 bytes")
        async with self._lock:
            async with httpx.AsyncClient(timeout=150, trust_env=False) as client:
                response = await client.post(
                    self.url + "/execute",
                    json={"task_id": self.task_id, "code": code},
                )
                response.raise_for_status()
                result = response.json()
                output = result.get("output")
                if not isinstance(output, str):
                    raise ValueError("AppWorld returned an invalid execution response")
                return output


def main() -> None:
    # Import the optional server entry point only for this explicitly selected process.
    from mcp.server import MCPServer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-url", required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    bridge = TaskCodeBridge(args.environment_url, args.task_id)
    server = MCPServer("tars-appworld-code")

    @server.tool()
    async def appworld_execute(code: str) -> str:
        """Execute Python in the current AppWorld task. Use public apis and print results.

        Variables persist between calls. Obtain public API documentation through
        apis.api_docs; complete the task through apis.supervisor.complete_task.
        Only task-world APIs are allowed. Do not access files, evaluator internals,
        task ground truth, external networks, or real accounts.
        """
        return await bridge.execute(code)

    server.run("stdio")


if __name__ == "__main__":
    main()
