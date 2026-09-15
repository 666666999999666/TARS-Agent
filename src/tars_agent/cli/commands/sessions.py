from __future__ import annotations

import asyncio
import sys

from tars_agent.core.config import TarsConfig
from tars_agent.core.transport.socket_client import IpcError, SocketClient


async def _list_sessions(config: TarsConfig, *, status: str | None = None) -> int:
    client = SocketClient(config.host, config.port)
    try:
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())
        params: dict[str, object] = {"limit": 50}
        if status is not None:
            params["status"] = status
        result = await client.send_command("session.list", params)
        sessions = result.get("sessions", [])
        if not sessions:
            print("no sessions")
        else:
            print(f"{'SESSION':<24} {'STATUS':<10} {'ACTIVE RUN':<24} TITLE")
            for session in sessions:
                print(
                    f"{str(session['session_id']):<24} "
                    f"{str(session['status']):<10} "
                    f"{str(session.get('active_run_id') or '-'):<24} "
                    f"{session.get('title') or '-'}"
                )
        await client.close()
        await asyncio.gather(loop_task, return_exceptions=True)
        return 0
    except (ConnectionRefusedError, OSError, IpcError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_sessions_list(config: TarsConfig, *, status: str | None = None) -> None:
    sys.exit(asyncio.run(_list_sessions(config, status=status)))
