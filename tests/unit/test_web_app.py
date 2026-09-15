from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from tars_agent.web.app import COOKIE_NAME, create_app
from tars_agent.web.auth import LocalWebAuth


class FakeCore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.stream_after_cursor = -1

    async def ping(self) -> dict[str, Any]:
        self.calls.append(("ping", None))
        return {"protocol_version": 2}

    async def list_sessions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self.calls.append(("list_sessions", (limit, offset)))
        return [
            {
                "session_id": "sess-one",
                "title": "<script>alert(1)</script>",
                "status": "ready",
                "updated_at": "2026-08-13T10:00:00Z",
            }
        ]

    async def get_session(self, session_id: str) -> dict[str, Any]:
        self.calls.append(("get_session", session_id))
        return {
            "session": {"session_id": session_id, "status": "ready"},
            "latest_run": None,
            "latest_cursor": 3,
        }

    async def get_run(self, run_id: str) -> dict[str, Any]:
        self.calls.append(("get_run", run_id))
        return {"run_id": run_id, "status": "succeeded"}

    async def stream_events(
        self,
        session_id: str,
        *,
        after_cursor: int,
    ) -> AsyncIterator[dict[str, Any]]:
        self.calls.append(("stream_events", session_id))
        self.stream_after_cursor = after_cursor
        for cursor in (after_cursor + 1, after_cursor + 2):
            yield {
                "kind": "event",
                "protocol_version": 2,
                "event_schema_version": 1,
                "cursor": cursor,
                "session_id": session_id,
                "run_id": "run-one",
                "occurred_at": "2026-08-13T10:00:00Z",
                "event": {
                    "type": "llm.token",
                    "token": "<img src=x onerror=alert(1)>",
                },
            }


def _authenticated_client() -> tuple[TestClient, FakeCore, LocalWebAuth]:
    core = FakeCore()
    auth, bootstrap_token = LocalWebAuth.issue()
    client = TestClient(create_app(core, auth=auth, allowed_host="testserver"))
    response = client.post(
        "/api/bootstrap",
        json={"token": bootstrap_token},
        headers={"Origin": "http://testserver"},
    )
    assert response.status_code == 204
    return client, core, auth


def test_bootstrap_sets_http_only_strict_non_secure_cookie_and_is_single_use() -> None:
    core = FakeCore()
    auth, bootstrap_token = LocalWebAuth.issue()
    with TestClient(create_app(core, auth=auth, allowed_host="testserver")) as client:
        response = client.post(
            "/api/bootstrap",
            json={"token": bootstrap_token},
            headers={"Origin": "http://testserver"},
        )
        assert response.status_code == 204
        cookie = response.headers["set-cookie"]
        assert f"{COOKIE_NAME}=" in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=strict" in cookie
        assert "Secure" not in cookie
        assert "Referrer-Policy" in response.headers
        retry = client.post(
            "/api/bootstrap",
            json={"token": bootstrap_token},
            headers={"Origin": "http://testserver"},
        )
        assert retry.status_code == 401


def test_api_requires_cookie_and_exposes_only_read_routes() -> None:
    core = FakeCore()
    auth, _ = LocalWebAuth.issue()
    with TestClient(create_app(core, auth=auth, allowed_host="testserver")) as client:
        assert client.get("/api/sessions").status_code == 401

    client, core, _ = _authenticated_client()
    with client:
        listed = client.get("/api/sessions")
        assert listed.status_code == 200
        assert listed.json()["sessions"][0]["title"].startswith("<script>")
        assert client.get("/api/sessions/sess-one").status_code == 200
        assert client.get("/api/runs/run-one").status_code == 200
        for path in (
            "/api/sessions",
            "/api/sessions/sess-one",
            "/api/runs/run-one",
            "/api/events?session_id=sess-one",
        ):
            assert client.post(path, headers={"Origin": "http://testserver"}).status_code == 405

    assert {name for name, _ in core.calls} <= {
        "list_sessions",
        "get_session",
        "get_run",
    }


def test_host_and_origin_are_enforced() -> None:
    core = FakeCore()
    auth, bootstrap_token = LocalWebAuth.issue()
    with TestClient(create_app(core, auth=auth, allowed_host="testserver")) as client:
        assert client.get("/healthz", headers={"Host": "evil.example"}).status_code == 400
        assert client.get("/healthz", headers={"Host": "[::1]"}).status_code == 400
        response = client.post(
            "/api/bootstrap",
            json={"token": bootstrap_token},
            headers={"Origin": "http://evil.example"},
        )
        assert response.status_code == 403
        missing = client.post(
            "/api/bootstrap",
            json={"token": bootstrap_token},
        )
        assert missing.status_code == 403


def test_sse_uses_last_event_id_and_preserves_payload_as_json_text() -> None:
    client, core, _ = _authenticated_client()
    with client:
        with client.stream(
            "GET",
            "/api/events?session_id=sess-one",
            headers={"Last-Event-ID": "40"},
        ) as response:
            body = "".join(response.iter_text())
    assert response.status_code == 200
    assert core.stream_after_cursor == 40
    assert "id: 41" in body and "id: 42" in body
    assert body.index("id: 41") < body.index("id: 42")
    data_lines = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    envelope = json.loads(data_lines[0])
    assert envelope["event"]["token"] == "<img src=x onerror=alert(1)>"


def test_sse_keepalive_does_not_cancel_pending_core_subscription(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class SlowCore(FakeCore):
        async def stream_events(
            self,
            session_id: str,
            *,
            after_cursor: int,
        ) -> AsyncIterator[dict[str, Any]]:
            await asyncio.sleep(0.03)
            yield {
                "kind": "event",
                "protocol_version": 2,
                "event_schema_version": 1,
                "cursor": after_cursor + 1,
                "session_id": session_id,
                "run_id": None,
                "occurred_at": "2026-08-13T10:00:00Z",
                "event": {"type": "session.resumed"},
            }

    monkeypatch.setattr("tars_agent.web.app.KEEPALIVE_S", 0.01)
    core = SlowCore()
    auth, bootstrap_token = LocalWebAuth.issue()
    with TestClient(create_app(core, auth=auth, allowed_host="testserver")) as client:
        bootstrap = client.post(
            "/api/bootstrap",
            json={"token": bootstrap_token},
            headers={"Origin": "http://testserver"},
        )
        assert bootstrap.status_code == 204
        with client.stream("GET", "/api/events?session_id=sess-one") as response:
            body = "".join(response.iter_text())

    assert ": keepalive" in body
    assert "session.resumed" in body
