from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from tars_agent.core.transport.socket_client import IpcDisconnectedError

type SendCommand = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
type Reconnect = Callable[[], Awaitable[SendCommand]]


def new_client_message_id() -> str:
    """Create the stable identity of one logical user submission."""

    return f"msg-{uuid.uuid4().hex}"


async def submit_message_with_retry(
    send_command: SendCommand,
    *,
    session_id: str,
    content: str,
    client_message_id: str,
    reconnect: Reconnect | None = None,
) -> dict[str, Any]:
    """Submit once, or reconnect and replay the same logical message once.

    JSON-RPC request IDs describe transport attempts and therefore change on a
    reconnect. ``client_message_id`` is deliberately held in one immutable
    payload so Core can deduplicate a response-lost replay to the original Run.
    """

    payload: dict[str, Any] = {
        "session_id": session_id,
        "content": content,
        "client_message_id": client_message_id,
    }
    try:
        return await send_command("session.send_message", dict(payload))
    except IpcDisconnectedError:
        if reconnect is None:
            raise
        retry_send = await reconnect()
        return await retry_send("session.send_message", dict(payload))


__all__ = [
    "Reconnect",
    "SendCommand",
    "new_client_message_id",
    "submit_message_with_retry",
]
