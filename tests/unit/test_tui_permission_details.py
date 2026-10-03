from __future__ import annotations

import json
from typing import Any

import pytest
from textual.widgets import Button, Static, TextArea

from tars_agent.core.transport.socket_client import IpcDisconnectedError, IpcError
from tars_agent.tui.app import ChatTextArea, PermissionSelect, TarsTuiApp
from tars_agent.tui.permission_details import PermissionDetailsScreen


class Client:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.error = error

    async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if self.error is not None:
            raise self.error
        return {"ok": True}


def isolated_app(monkeypatch: pytest.MonkeyPatch, client: Client) -> TarsTuiApp:
    app = TarsTuiApp("127.0.0.1", 9999)

    async def no_connection() -> None:
        pass

    monkeypatch.setattr(app, "_socket_loop", no_connection)
    monkeypatch.setattr(app, "_build_slash_items", lambda: [])
    app._client = client  # type: ignore[assignment]
    app._session_id = "session"
    app._active_run_id = "run"
    app._busy = True
    return app


def request(tool_name: str, params: dict[str, Any], kind: str = "tool") -> dict[str, Any]:
    return {
        "type": "permission.requested", "session_id": "session", "run_id": "run",
        "request_id": "permission-1", "tool_use_id": "tool-1", "tool_name": tool_name,
        "params": params, "param_preview": "short summary", "request_kind": kind,
        "platform_shell": "cmd.exe", "warning": "Docker isolation unavailable",
    }


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("tool_name, field, kind", [
    ("bash", "command", "tool"), ("write_file", "content", "tool"),
    ("spawn_agent", "prompt", "tool"), ("bash", "command", "host_fallback"),
])
async def test_long_approval_parameters_are_readable_without_approving(
    monkeypatch: pytest.MonkeyPatch, theme: str, tool_name: str, field: str, kind: str,
) -> None:
    client = Client()
    app = isolated_app(monkeypatch, client)
    app.theme = theme
    params = {field: "中文 [bold red] literal\n" * 180, "last": "FINAL-PARAMETER-MARKER"}
    async with app.run_test(size=(80, 24)) as pilot:
        app._handle_event_inner(request(tool_name, params, kind))
        await pilot.pause()
        selector = app.query_one(PermissionSelect)
        prompt = app.query_one("#prompt", ChatTextArea)
        assert selector.region.bottom <= prompt.region.y
        assert prompt.region.bottom <= app.screen.size.height
        assert "v 查看完整参数" in str(selector.content)
        await pilot.press("v")
        await pilot.pause()
        assert isinstance(app.screen, PermissionDetailsScreen)
        editor = app.screen.query_one(TextArea)
        assert editor.read_only and editor.soft_wrap
        assert json.loads(editor.text) == params
        assert "[bold red]" in editor.text
        await pilot.press("ctrl+end", "y", "n")
        await pilot.pause()
        assert editor.scroll_y > 0
        assert json.loads(editor.text) == params
        close = app.screen.query_one(Button)
        assert 0 <= editor.region.y < editor.region.bottom <= close.region.y
        assert close.region.bottom <= app.screen.size.height
        assert client.calls == []
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, PermissionDetailsScreen)
        assert app.focused is selector
        assert app._permission_details is None
        assert client.calls == []
        await pilot.press("n")
        await pilot.pause()
        assert client.calls == [("permission.respond", {
            "request_id": "permission-1", "session_id": "session", "decision": "deny_once",
        })]


@pytest.mark.parametrize("resolution", ["timeout", "allow_once", "replay"])
async def test_details_close_when_approval_expires_or_connection_replays(
    monkeypatch: pytest.MonkeyPatch, resolution: str,
) -> None:
    client = Client()
    app = isolated_app(monkeypatch, client)
    async with app.run_test(size=(80, 24)) as pilot:
        app._handle_event_inner(request("write_file", {"path": "file.txt", "content": "data"}))
        await pilot.pause()
        selector = app.query_one(PermissionSelect)
        await pilot.press("v")
        await pilot.pause()
        if resolution == "replay":
            app._begin_permission_replay()
        else:
            app._resolve_permission("permission-1", resolution)
        await pilot.pause()
        assert not isinstance(app.screen, PermissionDetailsScreen)
        assert app._permission_details is None
        app.on_permission_select_details_requested(PermissionSelect.DetailsRequested(selector))
        await pilot.pause()
        assert app._permission_details is None
        assert client.calls == []


async def test_permission_resolved_before_detail_mount_cannot_leave_stale_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Client()
    app = isolated_app(monkeypatch, client)
    async with app.run_test(size=(80, 24)) as pilot:
        app._handle_event_inner(request("write_file", {"content": "data"}))
        await pilot.pause()
        selector = app.query_one(PermissionSelect)
        app.on_permission_select_details_requested(PermissionSelect.DetailsRequested(selector))
        app._resolve_permission("permission-1", "timeout")
        await pilot.pause()
        assert not isinstance(app.screen, PermissionDetailsScreen)
        assert app._permission_details is None
        assert client.calls == []


@pytest.mark.parametrize("error, rejected", [
    (IpcError(-32602, "未找到 Skill /missing；本次任务未启动"), True),
    (IpcDisconnectedError(), False),
    (TimeoutError("response timeout"), False),
    (IpcError(-32603, "Internal error"), False),
])
async def test_submission_rejection_is_distinct_from_unknown_outcome(
    monkeypatch: pytest.MonkeyPatch, error: Exception, rejected: bool,
) -> None:
    client = Client(error)
    app = isolated_app(monkeypatch, client)
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", ChatTextArea)
        prompt.text = ""
        prompt.disabled = True
        await app._do_send_message("/missing arguments", "message-id")
        await pilot.pause()
        messages = "\n".join(str(widget.content) for widget in app.query(Static))
        assert ("请求被拒绝" in messages) is rejected
        assert ("send not confirmed" in messages) is not rejected
        assert prompt.text == ("/missing arguments" if rejected else "")
        assert not prompt.disabled and not app._busy
        assert len(client.calls) == 1
        assert client.calls[0][1]["client_message_id"] == "message-id"
