from __future__ import annotations

import asyncio
import json
import sys

from tars_agent.core.config import TarsConfig
from tars_agent.core.transport.socket_client import IpcError, SocketClient


async def _run_command(config: TarsConfig, method: str, run_id: str) -> int:
    client = SocketClient(config.host, config.port)
    loop_task: asyncio.Task[None] | None = None
    try:
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())
        result = await client.send_command(method, {"run_id": run_id})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ConnectionRefusedError, OSError, IpcError) as exc:
        if isinstance(exc, IpcError) and exc.code == -32033:
            print("已请求取消，尚未确认停止；后台继续清理", file=sys.stderr)
            print(json.dumps(exc.data, ensure_ascii=False), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()
        if loop_task is not None:
            await asyncio.gather(loop_task, return_exceptions=True)


def cmd_run_status(config: TarsConfig, run_id: str) -> None:
    sys.exit(asyncio.run(_run_command(config, "run.get", run_id)))


def cmd_run_cancel(config: TarsConfig, run_id: str) -> None:
    sys.exit(asyncio.run(_run_command(config, "run.cancel", run_id)))


def cmd_run_metrics(config: TarsConfig, run_id: str) -> None:
    sys.exit(asyncio.run(_run_command(config, "run.metrics", run_id)))
