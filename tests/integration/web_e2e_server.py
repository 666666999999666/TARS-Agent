from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import uvicorn

from tars_agent.web.app import create_app
from tars_agent.web.auth import LocalWebAuth
from tars_agent.web.core import SocketCoreReader


class DisconnectOnceCoreReader(SocketCoreReader):
    """Inject one transport EOF while preserving real Core events and replay.

    The first target subscription forwards exactly one envelope obtained from the
    real ``event.subscribe`` call and then ends.  A subsequent subscription is
    accepted only when FastAPI received that envelope's cursor through the native
    EventSource ``Last-Event-ID`` header.  All replayed payloads still come from
    the real Core and its durable event store.
    """

    def __init__(self, host: str, port: int, target_session_id: str) -> None:
        super().__init__(host, port)
        self._target_session_id = target_session_id
        self._disconnect_cursor: int | None = None

    async def stream_events(
        self,
        session_id: str,
        *,
        after_cursor: int,
    ) -> AsyncIterator[dict[str, Any]]:
        if session_id != self._target_session_id:
            async for envelope in super().stream_events(
                session_id,
                after_cursor=after_cursor,
            ):
                yield envelope
            return

        if self._disconnect_cursor is None:
            async for envelope in super().stream_events(
                session_id,
                after_cursor=after_cursor,
            ):
                cursor = envelope.get("cursor")
                if not isinstance(cursor, int):
                    raise AssertionError("real Core did not provide a durable event cursor")
                self._disconnect_cursor = cursor
                yield envelope
                # End the HTTP response after one real Core envelope.  The browser
                # must reconnect through native EventSource behavior.
                return
            raise AssertionError("real Core subscription ended before its first event")

        if after_cursor != self._disconnect_cursor:
            raise AssertionError(
                "EventSource did not resume with the first real event cursor: "
                f"expected {self._disconnect_cursor}, got {after_cursor}"
            )
        async for envelope in super().stream_events(
            session_id,
            after_cursor=after_cursor,
        ):
            yield envelope


if __name__ == "__main__":
    token = os.environ["TARS_WEB_BOOTSTRAP_TOKEN"]
    core_port = int(os.environ["TARS_PORT"])
    reconnect_session_id = os.environ["TARS_WEB_E2E_RECONNECT_SESSION_ID"]
    uvicorn.run(
        create_app(
            DisconnectOnceCoreReader(
                "127.0.0.1",
                core_port,
                reconnect_session_id,
            ),
            auth=LocalWebAuth(bootstrap_token=token),
        ),
        host="127.0.0.1",
        port=int(os.environ.get("TARS_WEB_E2E_PORT", "7438")),
        access_log=False,
        log_level="warning",
    )
