#!/usr/bin/env python3
"""通过公开 Wire Protocol 演示 V1.0 持久化 Session/Run 生命周期。"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from tars_agent.core.transport.socket_client import SocketClient


async def demo(host: str, port: int) -> None:
    client = SocketClient(host, port)
    await client.connect()
    loop_task = asyncio.create_task(client.run_event_loop())

    async def print_envelope(envelope: dict[str, object]) -> None:
        print(json.dumps(envelope, ensure_ascii=False))

    client.on_event_envelope(print_envelope)
    try:
        ping = await client.send_command("core.ping", {"client": "demo-v1"})
        print("ping", json.dumps(ping, ensure_ascii=False))
        created = await client.send_command(
            "session.create",
            {"mode": "chat", "title": "V1 demo"},
        )
        session_id = str(created["session_id"])
        await client.send_command(
            "event.subscribe",
            {"topics": ["*"], "session_id": session_id, "after_cursor": 0},
        )
        submitted = await client.send_command(
            "session.send_message",
            {
                "session_id": session_id,
                "content": "Reply with a short confirmation.",
                "client_message_id": f"demo-{uuid.uuid4().hex}",
            },
        )
        run_id = str(submitted["run_id"])
        while True:
            run = await client.send_command("run.get", {"run_id": run_id})
            if run["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                print("run", json.dumps(run, ensure_ascii=False))
                break
            await asyncio.sleep(0.1)
    finally:
        await client.close()
        await asyncio.gather(loop_task, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7437)
    args = parser.parse_args()
    asyncio.run(demo(args.host, args.port))


if __name__ == "__main__":
    main()
