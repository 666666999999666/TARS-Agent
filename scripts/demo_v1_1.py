from __future__ import annotations

import argparse
import asyncio
import json

from tars_agent.core.transport.socket_client import SocketClient


async def _command(
    client: SocketClient,
    method: str,
    params: dict[str, object],
) -> dict[str, object]:
    result = await client.send_command(method, params)
    print(json.dumps({"method": method, "result": result}, ensure_ascii=False, indent=2))
    return result


async def demo(host: str, port: int) -> None:
    """Run a read-only, API-key-free V1.1 protocol demonstration."""
    client = SocketClient(host, port)
    await client.connect()
    loop_task = asyncio.create_task(client.run_event_loop())
    try:
        await _command(client, "core.ping", {"client": "demo-v1.1"})
        await _command(client, "session.list", {"limit": 5, "offset": 0})
        await _command(client, "mcp.status", {})
    finally:
        await client.close()
        await asyncio.gather(loop_task, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TARS-Agent V1.1 read-only demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7437, type=int)
    args = parser.parse_args()
    asyncio.run(demo(args.host, args.port))


if __name__ == "__main__":
    main()
