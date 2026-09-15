from __future__ import annotations

import asyncio
import subprocess
from collections.abc import AsyncIterator
from typing import Any

from tars_agent.core.transport.socket_client import SocketClient
from tars_agent.web.core import SocketCoreReader


async def _command(port: int, method: str, params: dict[str, object]) -> dict[str, object]:
    client = SocketClient("127.0.0.1", port)
    await client.connect()
    loop_task = asyncio.create_task(client.run_event_loop())
    try:
        return await client.send_command(method, params)
    finally:
        await client.close()
        await asyncio.gather(loop_task, return_exceptions=True)


async def test_read_only_web_adapter_roundtrips_real_core_without_state_mutation(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    del running_daemon
    created = await _command(
        free_port,
        "session.create",
        {"mode": "chat", "title": "web roundtrip"},
    )
    session_id = str(created["session_id"])
    reader = SocketCoreReader("127.0.0.1", free_port)

    listed = await reader.list_sessions()
    before = next(session for session in listed if session["session_id"] == session_id)
    detail = await reader.get_session(session_id)
    after = next(
        session
        for session in await reader.list_sessions()
        if session["session_id"] == session_id
    )

    assert detail["session"]["session_id"] == session_id
    assert detail["latest_run"] is None
    assert detail["latest_cursor"] >= 1
    assert before["status"] == after["status"] == "ready"
    assert before["updated_at"] == after["updated_at"]


async def _next_event_type(
    stream: AsyncIterator[dict[str, Any]],
    event_type: str,
) -> dict[str, Any]:
    async for envelope in stream:
        if envelope.get("event", {}).get("type") == event_type:
            return envelope
    raise AssertionError(f"stream ended before {event_type}")


async def test_web_and_tui_wire_client_observe_one_session_without_cross_talk(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    del running_daemon
    first = await _command(free_port, "session.create", {"mode": "chat", "title": "one"})
    second = await _command(free_port, "session.create", {"mode": "chat", "title": "two"})
    first_id = str(first["session_id"])
    second_id = str(second["session_id"])
    reader = SocketCoreReader("127.0.0.1", free_port)
    first_cursor = int((await reader.get_session(first_id))["latest_cursor"])
    second_cursor = int((await reader.get_session(second_id))["latest_cursor"])
    after_cursor = max(first_cursor, second_cursor)

    tui = SocketClient("127.0.0.1", free_port)
    await tui.connect()
    tui_envelopes: list[dict[str, Any]] = []
    first_message_seen = asyncio.Event()

    async def collect(envelope: dict[str, Any]) -> None:
        tui_envelopes.append(envelope)
        if envelope.get("event", {}).get("type") == "session.message_received":
            first_message_seen.set()

    tui.on_event_envelope(collect)
    tui_loop = asyncio.create_task(tui.run_event_loop())
    web_stream = reader.stream_events(first_id, after_cursor=after_cursor)
    web_event_task = asyncio.create_task(
        _next_event_type(web_stream, "session.message_received")
    )
    try:
        await tui.send_command(
            "event.subscribe",
            {
                "topics": ["*"],
                "session_id": first_id,
                "after_cursor": after_cursor,
            },
        )
        # Start the independent Web subscription before producing either Session event.
        await asyncio.sleep(0.05)
        await _command(
            free_port,
            "session.send_message",
            {"session_id": second_id, "content": "must stay isolated"},
        )
        await _command(
            free_port,
            "session.send_message",
            {"session_id": first_id, "content": "visible to both"},
        )
        web_envelope = await asyncio.wait_for(web_event_task, timeout=5)
        await asyncio.wait_for(first_message_seen.wait(), timeout=5)

        assert web_envelope["session_id"] == first_id
        assert web_envelope["event"]["content"] == "visible to both"
        assert tui_envelopes
        assert all(envelope.get("session_id") == first_id for envelope in tui_envelopes)
        assert not any(
            envelope.get("event", {}).get("content") == "must stay isolated"
            for envelope in tui_envelopes
        )
    finally:
        if not web_event_task.done():
            web_event_task.cancel()
        await asyncio.gather(web_event_task, return_exceptions=True)
        await web_stream.aclose()
        await tui.close()
        await asyncio.gather(tui_loop, return_exceptions=True)
