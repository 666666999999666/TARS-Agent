from __future__ import annotations

from typing import Any

from tars_agent.web import core as web_core


class _BurstSocketClient:
    def __init__(self, _host: str, _port: int) -> None:
        self._handler = None
        self.last_cursor = 0

    def on_event_envelope(self, handler):  # type: ignore[no-untyped-def]
        self._handler = handler

    async def connect(self) -> None:
        return None

    async def send_command(
        self,
        _method: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        assert self._handler is not None
        for cursor in range(11, 268):
            self.last_cursor = cursor
            await self._handler(
                {
                    "kind": "event",
                    "protocol_version": 2,
                    "event_schema_version": 1,
                    "cursor": cursor,
                    "session_id": "session-1",
                    "run_id": None,
                    "occurred_at": "2026-08-13T10:00:00Z",
                    "event": {"type": "test.event"},
                }
            )
        return {"subscription_id": "subscription-1"}

    async def run_event_loop(self) -> None:
        # send_command creates the complete burst; stay alive until the reader
        # closes this synthetic connection.
        import asyncio

        await asyncio.Future()

    async def close(self) -> None:
        return None


async def test_overflow_resumes_from_last_delivered_cursor(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(web_core, "SocketClient", _BurstSocketClient)
    reader = web_core.SocketCoreReader("127.0.0.1", 7437)
    stream = reader.stream_events("session-1", after_cursor=10)

    envelope = await anext(stream)

    assert envelope["kind"] == "overflow"
    assert envelope["last_cursor"] == 10
    await stream.aclose()


class _UpstreamOverflowSocketClient(_BurstSocketClient):
    async def send_command(
        self,
        _method: str,
        _params: dict[str, Any],
    ) -> dict[str, Any]:
        assert self._handler is not None
        self.last_cursor = 75
        await self._handler(
            {
                "kind": "overflow",
                "protocol_version": 2,
                "reason": "replay_limit",
                "last_cursor": 75,
            }
        )
        return {"subscription_id": "subscription-1"}

    async def run_event_loop(self) -> None:
        return None


async def test_upstream_overflow_is_terminal_and_keeps_resume_cursor(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(web_core, "SocketClient", _UpstreamOverflowSocketClient)
    reader = web_core.SocketCoreReader("127.0.0.1", 7437)

    envelopes = [
        envelope
        async for envelope in reader.stream_events("session-1", after_cursor=10)
    ]

    assert envelopes == [
        {
            "kind": "overflow",
            "protocol_version": 2,
            "reason": "replay_limit",
            "last_cursor": 75,
        }
    ]
