from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Protocol

from tars_agent.core.transport.socket_client import SocketClient


class CoreReader(Protocol):
    async def ping(self) -> dict[str, Any]: ...

    async def list_sessions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]: ...

    async def get_session(self, session_id: str) -> dict[str, Any]: ...

    async def get_run(self, run_id: str) -> dict[str, Any]: ...

    def stream_events(
        self,
        session_id: str,
        *,
        after_cursor: int,
    ) -> AsyncIterator[dict[str, Any]]: ...


class SocketCoreReader:
    """Thin read-only adapter: every operation is a Wire Protocol V2 read."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port

    async def _command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        client = SocketClient(self._host, self._port)
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())
        try:
            return await client.send_command(method, params)
        finally:
            await client.close()
            await asyncio.gather(loop_task, return_exceptions=True)

    async def ping(self) -> dict[str, Any]:
        return await self._command("core.ping", {"client": "tars-web/0.8.0"})

    async def list_sessions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        result = await self._command(
            "session.list",
            {"limit": limit, "offset": offset},
        )
        sessions = result.get("sessions", [])
        return list(sessions) if isinstance(sessions, list) else []

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return await self._command("session.get", {"session_id": session_id})

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return await self._command("run.get", {"run_id": run_id})

    async def stream_events(
        self,
        session_id: str,
        *,
        after_cursor: int,
    ) -> AsyncIterator[dict[str, Any]]:
        client = SocketClient(self._host, self._port)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        overflow = asyncio.Event()
        delivered_cursor = after_cursor

        async def receive(envelope: dict[str, Any]) -> None:
            try:
                queue.put_nowait(envelope)
            except asyncio.QueueFull:
                overflow.set()

        client.on_event_envelope(receive)
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())
        try:
            await client.send_command(
                "event.subscribe",
                {
                    "topics": ["*"],
                    "session_id": session_id,
                    "after_cursor": after_cursor,
                },
            )
            while True:
                if overflow.is_set():
                    yield {
                        "kind": "overflow",
                        "protocol_version": 2,
                        "reason": "browser_queue_full",
                        # Resume from the last envelope that crossed the Web
                        # adapter boundary, not from SocketClient.last_cursor.
                        # The latter also includes envelopes rejected by this
                        # full browser queue and would create a permanent gap.
                        "last_cursor": delivered_cursor,
                    }
                    return
                get_task = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {get_task, loop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_task in done:
                    envelope = get_task.result()
                    yield envelope
                    cursor = envelope.get("cursor") or envelope.get("last_cursor")
                    if isinstance(cursor, int) and cursor > delivered_cursor:
                        delivered_cursor = cursor
                    if envelope.get("kind") == "overflow":
                        # Core closes the subscription after this terminal
                        # marker.  Do not append a lower-cursor disconnected
                        # marker that would overwrite the browser resume point.
                        return
                    continue
                get_task.cancel()
                await asyncio.gather(get_task, return_exceptions=True)
                if overflow.is_set():
                    continue
                yield {
                    "kind": "core.disconnected",
                    "protocol_version": 2,
                    "last_cursor": delivered_cursor,
                }
                return
        finally:
            await client.close()
            loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)


__all__ = ["CoreReader", "SocketCoreReader"]
