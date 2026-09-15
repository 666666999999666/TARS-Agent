from __future__ import annotations

from typing import Any

from tars_agent.core.transport.message_submission import submit_message_with_retry
from tars_agent.core.transport.socket_client import IpcDisconnectedError


async def test_response_lost_replay_reuses_client_message_id() -> None:
    attempts: list[dict[str, Any]] = []

    async def disconnected(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "session.send_message"
        attempts.append(dict(params))
        raise IpcDisconnectedError("response lost")

    async def reconnected(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "session.send_message"
        attempts.append(dict(params))
        return {"run_id": "run-original", "deduplicated": True}

    async def reconnect():  # type: ignore[no-untyped-def]
        return reconnected

    result = await submit_message_with_retry(
        disconnected,
        session_id="sess-1",
        content="do it once",
        client_message_id="msg-stable",
        reconnect=reconnect,
    )

    assert result["run_id"] == "run-original"
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert attempts[0]["client_message_id"] == "msg-stable"


async def test_retry_payload_isolated_from_first_transport_mutation() -> None:
    async def disconnected(_method: str, params: dict[str, Any]) -> dict[str, Any]:
        params["client_message_id"] = "transport-mutated"
        raise IpcDisconnectedError("response lost")

    async def reconnected(_method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert params["client_message_id"] == "msg-stable"
        return {"run_id": "run-original"}

    async def reconnect():  # type: ignore[no-untyped-def]
        return reconnected

    result = await submit_message_with_retry(
        disconnected,
        session_id="sess-1",
        content="do it once",
        client_message_id="msg-stable",
        reconnect=reconnect,
    )

    assert result["run_id"] == "run-original"
