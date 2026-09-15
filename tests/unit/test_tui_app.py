from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from rich.markdown import Markdown
from textual.widget import Widget

from tars_agent.tui.app import (
    LLMStreamBlock,
    PermissionBlock,
    TarsTuiApp,
    ToolCallBlock,
    _param_summary,
    _preview,
)


# 功能：验证 _preview 超出长度时截断并追加省略号
# 设计：不依赖任何 TUI 组件，纯函数测试
def test_preview_truncates() -> None:
    assert _preview("abcde", 3) == "abc…"
    assert _preview("ab", 5) == "ab"


# 功能：验证工具参数摘要优先展示工具最关键字段
# 设计：覆盖 read_file/bash/note_save 三类常见工具，避免工具块摘要退化成整段 JSON
def test_param_summary_prefers_key_fields() -> None:
    assert _param_summary("read_file", {"path": "README.md"}) == "path='README.md'"
    assert _param_summary("bash", {"command": "echo hi", "timeout": 1}) == "command='echo hi'"
    assert _param_summary("note_save", {"content": "Python 3.12"}) == "content='Python 3.12'"


# 功能：验证 llm.token 事件累积到 LLMStreamBlock，不连续 token 各自新开一块
# 设计：monkey-patch _append 收集追加的 widgets，断言 token 追加到同一块；
#       发送非 token 事件后新 block 被重置，下一个 token 开启新块
def test_llm_tokens_accumulate_in_block() -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({"type": "llm.token", "token": "Hello", "run_id": "r", "ts": "t"})
    app._handle_event({"type": "llm.token", "token": " world", "run_id": "r", "ts": "t"})

    assert len(appended) == 1  # same block reused
    assert isinstance(appended[0], LLMStreamBlock)
    assert appended[0]._text == "Hello world"  # type: ignore[attr-defined]


# 功能：验证 LLMStreamBlock 结束时会把累积文本渲染为 Rich Markdown
# 设计：直接调用 finalize_markdown，断言 renderable 类型，覆盖 Markdown polish 的核心行为
def test_llm_block_finalize_renders_markdown() -> None:
    block = LLMStreamBlock()
    block.append_token("## Title\n\n- one\n\n```python\nprint('hi')\n```")
    block.finalize_markdown()
    assert isinstance(block.content, Markdown)


# 功能：验证非 token 事件后 _current_llm 被重置，下一个 token 开启新块
# 设计：插入 step.started 中断流，验证之前的 block 被 finalize，之后的 llm.token 创建新 LLMStreamBlock
def test_llm_block_resets_after_non_token_event() -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({"type": "llm.token", "token": "A", "run_id": "r", "ts": "t"})
    app._handle_event({"type": "step.started", "run_id": "r", "step": 2, "ts": "t"})
    app._handle_event({"type": "llm.token", "token": "B", "run_id": "r", "ts": "t"})

    llm_blocks = [w for w in appended if isinstance(w, LLMStreamBlock)]
    assert len(llm_blocks) == 2
    assert llm_blocks[0]._finalized  # type: ignore[attr-defined]


# 功能：验证 run.started 事件追加 Static widget 且包含 run_id 和 goal
# 设计：monkey-patch _append，断言追加的 widget 的 renderable 包含关键字段
def test_run_started_appends_widget_with_content() -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "run.started", "run_id": "run-abc", "goal": "do the thing", "ts": "t"
    })

    assert len(appended) == 1
    rendered = appended[0].content
    assert "run-abc" in rendered
    assert "do the thing" in rendered


# 功能：验证 run.finished success 追加包含 "completed" 的 widget
# 设计：monkey-patch _append，检查 rendered 内容包含 completed 和 green
def test_run_finished_success_shows_completed() -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "run.finished", "run_id": "r", "status": "success", "steps": 3, "ts": "t"
    })

    rendered = appended[0].content
    assert "completed" in rendered
    assert "green" in rendered


# 功能：验证 run.finished failed 追加包含 "failed" 和 red 的 widget
# 设计：与 success 对称，检查颜色标记差异
def test_run_finished_failed_shows_red() -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "run.finished", "run_id": "r", "status": "failed",
        "steps": 1, "reason": "llm_error", "ts": "t"
    })

    rendered = appended[0].content
    assert "failed" in rendered
    assert "red" in rendered


# 功能：验证 tool.call_started 追加 ToolCallBlock，call_finished 更新其结果
# 设计：直接调用 _handle_event 两次，通过 _pending_tool_blocks 验证状态流转
def test_tool_call_started_and_finished() -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event({
        "type": "tool.call_started",
        "tool_use_id": "uid-1",
        "tool_name": "bash",
        "params": {"command": "echo hi"},
        "run_id": "r", "ts": "t",
    })
    assert "uid-1" in app._pending_tool_blocks  # type: ignore[attr-defined]

    app._handle_event({
        "type": "tool.call_finished",
        "tool_use_id": "uid-1",
        "tool_name": "bash",
        "elapsed_ms": 42,
        "output": "hi",
        "run_id": "r", "ts": "t",
    })
    assert "uid-1" not in app._pending_tool_blocks  # type: ignore[attr-defined]
    block = appended[0]
    assert isinstance(block, ToolCallBlock)
    assert block._finished  # type: ignore[attr-defined]
    assert block._output == "hi"  # type: ignore[attr-defined]


# 功能：验证 note_save 成功完成时工具块摘要显示 remembered
# 设计：直接操作 ToolCallBlock，覆盖 note_save 的特殊低噪声展示策略
def test_note_save_tool_block_shows_remembered() -> None:
    block = ToolCallBlock("note_save", {"content": "Python 3.12"})
    block.set_result("saved", 3)
    assert "remembered" in block._summary()  # type: ignore[attr-defined]


# 功能：验证提交用户输入时会追加 user turn，并进入 busy 状态
# 设计：用 fake client 替代 SocketClient，直接调用 on_chat_text_area_submitted，
#       覆盖 TextArea 清空内容 + 设置 busy 占位符的核心状态迁移
async def test_input_submit_waits_for_durable_event_and_disables_prompt() -> None:
    class _FakeArea:
        def __init__(self) -> None:
            self.disabled = False
            self.border_title = ""
            self.text = "hello"

    class _FakeEvent:
        def __init__(self, area: _FakeArea) -> None:
            self.value = area.text
            self.text_area = area

    sent: list[tuple[str, dict]] = []

    class _FakeClient:
        async def send_command(self, method: str, params: dict) -> dict:
            sent.append((method, params))
            return {"run_id": "run-1"}

    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]
    app._update_header = lambda state: None  # type: ignore[method-assign]
    app._client = _FakeClient()  # type: ignore[assignment]
    app._session_id = "sess-1"

    area = _FakeArea()
    event = _FakeEvent(area)
    await app.on_chat_text_area_submitted(event)  # type: ignore[arg-type]

    assert app._busy  # type: ignore[attr-defined]
    assert area.disabled
    assert area.text == ""
    assert "agent is working" in area.border_title.lower()
    assert appended == []
    for _ in range(20):
        if sent:
            break
        await asyncio.sleep(0)
    assert sent[0][0] == "session.send_message"
    assert sent[0][1]["client_message_id"].startswith("msg-")
    app._handle_event_inner(
        {
            "type": "session.message_received",
            "session_id": "sess-1",
            "content": "hello",
        }
    )
    assert appended[0].content == "[bold]>[/bold] hello"


async def test_overflow_envelope_uses_last_cursor_without_event_cursor() -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda widget: appended.append(widget)  # type: ignore[method-assign]
    app._last_cursor = 7

    await app._handle_event_envelope(
        {
            "kind": "overflow",
            "protocol_version": 2,
            "reason": "subscriber_queue_full",
            "last_cursor": 19,
        }
    )

    assert app._last_cursor == 19
    assert "reconnecting from cursor 19" in str(appended[0].content)


async def test_reconnect_resumes_session_and_refreshes_interrupted_run_state() -> None:
    calls: list[tuple[str, dict]] = []

    async def send_command(method: str, params: dict) -> dict:
        calls.append((method, params))
        return {
            "session": {"session_id": "sess-1", "status": "ready"},
            "active_run": {"run_id": "run-old", "status": "interrupted"},
            "latest_cursor": 12,
        }

    app = TarsTuiApp("127.0.0.1", 9999)
    app._session_id = "sess-1"
    app._busy = True

    await app._attach_session(send_command)

    assert calls == [("session.resume", {"session_id": "sess-1"})]
    assert app._session_id == "sess-1"
    assert not app._busy


async def test_new_session_uses_tui_process_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict]] = []
    monkeypatch.chdir(tmp_path)

    async def send_command(method: str, params: dict) -> dict:
        calls.append((method, params))
        return {"session_id": "sess-1"}

    app = TarsTuiApp("127.0.0.1", 9999)
    await app._attach_session(send_command)

    assert calls == [
        (
            "session.create",
            {"mode": "chat", "workspace_root": str(tmp_path.resolve())},
        )
    ]
    assert app._session_id == "sess-1"


# 功能：验证未知事件类型不抛异常并以通用 JSON 安全降级
# 设计：发送带 Rich 标记的未知事件，断言原始字段可见且标记已转义
@pytest.mark.parametrize("event_type", ["some.unknown.type", "session.unrecognized"])
def test_unknown_event_renders_generic_escaped_json(event_type: str) -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda w: appended.append(w)  # type: ignore[method-assign]

    app._handle_event(
        {
            "type": event_type,
            "run_id": "r",
            "payload": "[bold]must remain text[/bold]",
            "ts": "t",
        }
    )

    assert len(appended) == 1
    rendered = str(appended[0].content)
    assert f"unknown event {event_type}" in rendered
    assert '"run_id": "r"' in rendered
    assert "\\[bold]must remain text\\[/bold]" in rendered


async def test_unsupported_event_schema_renders_compatibility_error() -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda widget: appended.append(widget)  # type: ignore[method-assign]

    await app._handle_event_envelope(
        {
            "kind": "compatibility.error",
            "protocol_version": 2,
            "cursor": 23,
            "received_protocol_version": 2,
            "received_event_schema_version": 99,
            "reason": "unsupported event envelope version",
        }
    )

    assert app._last_cursor == 23
    assert len(appended) == 1
    rendered = str(appended[0].content)
    assert "event compatibility error" in rendered
    assert '"received_event_schema_version": 99' in rendered


def test_host_fallback_permission_displays_full_command_shell_and_warning() -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    appended: list[Widget] = []
    app._append = lambda widget: appended.append(widget)  # type: ignore[method-assign]
    app._mount_permission_select = lambda _select: None  # type: ignore[method-assign]
    app._handle_event_inner(
        {
            "type": "permission.requested",
            "request_id": "perm-1",
            "request_kind": "host_fallback",
            "tool_use_id": "tool-1",
            "tool_name": "bash",
            "params": {"command": "echo full-command-that-must-not-be-truncated"},
            "param_preview": "echo full...",
            "platform_shell": "cmd.exe",
            "warning": "network visibility is not isolated",
        }
    )
    block = appended[0]
    assert isinstance(block, PermissionBlock)
    pending = block._pending_text()  # type: ignore[attr-defined]
    assert "echo full-command-that-must-not-be-truncated" in pending
    assert "cmd.exe" in pending
    assert "network visibility is not isolated" in pending


async def test_tui_disconnect_does_not_resend_message() -> None:
    from tars_agent.core.transport.socket_client import IpcDisconnectedError
    calls = []
    class Client:
        async def send_command(self, method: str, params: dict) -> dict:
            calls.append((method, params))
            raise IpcDisconnectedError()
    app = TarsTuiApp("127.0.0.1", 9999)
    app._client = Client()  # type: ignore[assignment]
    app._session_id = "session"
    app._append = lambda _widget: None  # type: ignore[method-assign]
    app._update_header = lambda _state: None  # type: ignore[method-assign]
    await app._do_send_message("hello", "same-message")
    assert len(calls) == 1
    assert calls[0][1]["client_message_id"] == "same-message"


async def test_tui_cancel_timeout_keeps_running_input_state() -> None:
    from tars_agent.core.transport.socket_client import IpcError
    class Client:
        async def send_command(self, method: str, params: dict) -> dict:
            assert method == "run.cancel"
            assert params == {"run_id": "run-current"}
            raise IpcError(-32033, "pending", {"cancellation_requested": True})
    app = TarsTuiApp("127.0.0.1", 9999)
    app._client = Client()  # type: ignore[assignment]
    app._active_run_id = "run-current"
    app._busy = True
    app._cancellation_requested = True
    shown: list[Widget] = []
    app._append = lambda widget: shown.append(widget)  # type: ignore[method-assign]
    await app._cancel_current_run()
    assert app._busy
    assert app._active_run_id == "run-current"
    assert "尚未确认停止" in str(shown[-1].content)


def test_replayed_previous_waiting_event_does_not_unlock_current_run() -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    app._busy = True
    app._active_run_id = "new-run"
    app._handle_event_inner({"type": "session.waiting_for_input", "last_run_id": "old-run"})
    assert app._busy
    assert app._active_run_id == "new-run"


async def test_welcome_shows_current_brand_and_accurate_keyboard_help(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from textual.content import Content as Text
    from textual.widgets import Static

    app = TarsTuiApp("127.0.0.1", 9999)
    async def no_connection() -> None:
        pass
    monkeypatch.setattr(app, "_socket_loop", no_connection)
    monkeypatch.setattr(app, "_build_slash_items", lambda: [])
    async with app.run_test(size=(80, 24)):
        welcome = Text.from_markup(str(app.query_one("#banner", Static).content)).plain
        assert welcome.splitlines()[0] == "TARS-Agent"
        assert "Ctrl+X 停止当前任务" in welcome
        assert "Ctrl+Q 退出界面，后台任务继续运行" in welcome
        assert "Ctrl+C" not in welcome
        assert "█" not in welcome
        assert len(welcome.splitlines()) == 4
    actions = {binding.key: binding.action for binding in app.BINDINGS}
    assert actions["ctrl+x"] == "stop"
    assert actions["ctrl+q"] == "quit"


@pytest.mark.parametrize("event_type", ["session.created", "session.resumed"])
@pytest.mark.parametrize("busy", [False, True])
def test_known_session_replay_keeps_rpc_state_and_active_stream(
    monkeypatch: pytest.MonkeyPatch, event_type: str, busy: bool,
) -> None:
    from tars_agent.tui.app import ChatTextArea

    app = TarsTuiApp("127.0.0.1", 9999)
    app._session_id = "current-session"
    app._busy = busy
    app._active_run_id = "current-run" if busy else None
    app._cancellation_requested = busy
    prompt = ChatTextArea(show_line_numbers=False)
    prompt.disabled = True
    prompt.border_title = "current input state"
    monkeypatch.setattr(app, "_prompt", lambda: prompt)
    appended: list[Widget] = []
    monkeypatch.setattr(app, "_append", appended.append)
    stream = LLMStreamBlock()
    stream.append_token("before")
    app._current_llm = stream

    app._handle_event_inner({"type": event_type, "session_id": "current-session", "mode": "chat"})

    assert appended == []
    assert app._session_id == "current-session"
    assert app._busy == busy
    assert app._active_run_id == ("current-run" if busy else None)
    assert app._cancellation_requested == busy
    assert prompt.disabled
    assert prompt.border_title == "current input state"
    assert app._current_llm is stream
    assert not stream._finalized
    app._handle_event_inner({"type": "llm.token", "token": "after"})
    assert stream._text == "beforeafter"


@pytest.mark.parametrize("event_type", ["tool.execution_started", "step.finished"])
def test_execution_notifications_do_not_dump_metadata_or_complete_run(
    monkeypatch: pytest.MonkeyPatch, event_type: str,
) -> None:
    app = TarsTuiApp("127.0.0.1", 9999)
    app._busy = True
    app._active_run_id = "active-run"
    pending = ToolCallBlock("read_file", {"path": "report.md"})
    app._pending_tool_blocks["tool-1"] = pending
    appended: list[Widget] = []
    monkeypatch.setattr(app, "_append", appended.append)
    app._handle_event_inner({
        "type": event_type, "run_id": "active-run", "tool_use_id": "tool-1", "step": 1,
        "tool_name": "read_file", "container_id": "container-id-must-not-fill-chat",
        "backend": "workspace_sandbox", "attempt": 1,
    })
    assert appended == []
    assert app._busy
    assert app._active_run_id == "active-run"
    assert app._pending_tool_blocks["tool-1"] is pending
    assert not pending._finished


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_approval_group_keeps_action_choices_visible_and_input_disabled(
    monkeypatch: pytest.MonkeyPatch, theme: str,
) -> None:
    from textual.containers import VerticalScroll
    from textual.content import Content as Text
    from textual.widgets import Static

    from tars_agent.tui.app import ChatTextArea, PermissionSelect

    app = TarsTuiApp("127.0.0.1", 9999)
    app.theme = theme
    async def no_connection() -> None:
        pass
    monkeypatch.setattr(app, "_socket_loop", no_connection)
    monkeypatch.setattr(app, "_build_slash_items", lambda: [])
    async with app.run_test(size=(80, 24)) as pilot:
        log = app.query_one("#log-view", VerticalScroll)
        await log.mount(*(Static(f"prior log line {index}") for index in range(80)))
        app._session_id = "session"
        app._active_run_id = "run"
        app._busy = True
        class ApprovalClient:
            async def send_command(self, method: str, params: dict) -> dict:
                return {"ok": True}
        app._client = ApprovalClient()  # type: ignore[assignment]
        app._handle_event_inner({
            "type": "permission.requested", "session_id": "session", "run_id": "run",
            "request_id": "permission", "request_kind": "tool", "tool_use_id": "tool",
            "tool_name": "write_file", "params": {"path": "report.md", "content": "data"},
            "param_preview": "path='report.md'",
        })
        await pilot.pause()
        group = app.query_one(PermissionSelect)
        prompt = app.query_one("#prompt", ChatTextArea)
        text = Text.from_markup(str(group.content)).plain
        assert "write_file" in text and "request_kind=tool" in text and "path='report.md'" in text
        assert "Allow once" in text and "Deny session" in text
        assert 0 <= group.region.y < group.region.bottom <= prompt.region.y
        assert prompt.region.bottom <= app.screen.size.height
        assert app.focused is group
        assert prompt.disabled and prompt.has_class("permission-waiting")
        assert prompt.styles.opacity == 1.0
        assert prompt.styles.border_title_style.bold
        assert prompt.styles.border_title_color != prompt.styles.border_title_background
        assert "等待审批" in str(prompt.border_title)
        assert app._busy and app._active_run_id == "run"
        await pilot.press("n")
        await pilot.pause()
        assert not prompt.has_class("permission-waiting")
        assert app._busy and app._active_run_id == "run"


class _PermissionRecoveryClient:
    def __init__(self, states: dict[str, str] | None = None, fail_response: bool = False) -> None:
        self.states = states or {}
        self.fail_response = fail_response
        self.calls: list[tuple[str, dict]] = []
    async def send_command(self, method: str, params: dict) -> dict:
        from tars_agent.core.transport.socket_client import IpcDisconnectedError
        self.calls.append((method, params))
        if method == "run.get":
            return {"status": self.states[params["run_id"]]}
        if self.fail_response:
            raise IpcDisconnectedError()
        return {"ok": True}


def _permission_request(request_id: str, run_id: str) -> dict:
    return {"type": "permission.requested", "session_id": "session", "run_id": run_id,
            "request_id": request_id, "request_kind": "tool", "tool_use_id": request_id + "-tool",
            "tool_name": "write_file", "params": {"path": request_id + ".txt"},
            "param_preview": "path='" + request_id + ".txt'"}


def _isolated_permission_app(monkeypatch: pytest.MonkeyPatch, client: _PermissionRecoveryClient) -> TarsTuiApp:
    app = TarsTuiApp("127.0.0.1", 9999)
    async def no_connection() -> None:
        pass
    monkeypatch.setattr(app, "_socket_loop", no_connection)
    monkeypatch.setattr(app, "_build_slash_items", lambda: [])
    app._client = client  # type: ignore[assignment]
    app._session_id = "session"
    app._busy = True
    app._active_run_id = "current"
    return app


@pytest.mark.parametrize(("event_type", "decision"), [
    ("permission.denied", "deny_once"), ("permission.denied", "timeout"),
    ("permission.granted", "allow_once"),
])
async def test_fast_permission_resolution_invalidates_unfinished_mount(
    monkeypatch: pytest.MonkeyPatch, event_type: str, decision: str,
) -> None:
    from tars_agent.tui.app import PermissionSelect
    client = _PermissionRecoveryClient()
    app = _isolated_permission_app(monkeypatch, client)
    async with app.run_test(size=(80, 24)) as pilot:
        app._handle_event(_permission_request("old", "old-run"))
        stale = app._permission_selects["old"]
        app._handle_event({"type": event_type, "request_id": "old", "run_id": "old-run",
                           "tool_use_id": "old-tool", "decision": decision})
        await pilot.pause()
        assert list(app.query(PermissionSelect)) == []
        assert not app._pending_permission_blocks
        assert app.focused is not stale
        assert not stale._valid
        await app.on_permission_select_decided(PermissionSelect.Decided(stale, "old", "allow_once"))
        assert client.calls == []
        assert app._busy and app._active_run_id == "current"


async def test_resolution_removes_only_its_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from tars_agent.tui.app import PermissionSelect
    app = _isolated_permission_app(monkeypatch, _PermissionRecoveryClient())
    async with app.run_test(size=(80, 30)) as pilot:
        app._handle_event(_permission_request("old", "old-run"))
        app._handle_event(_permission_request("current-request", "current"))
        app._handle_event({"type": "permission.denied", "request_id": "old", "decision": "timeout"})
        await pilot.pause()
        groups = list(app.query(PermissionSelect))
        assert len(groups) == 1 and groups[0]._request_id == "current-request"
        assert set(app._pending_permission_blocks) == {"current-request"}
        assert app.focused is groups[0]


@pytest.mark.parametrize("boundary_first", [False, True])
async def test_large_replay_restores_only_current_pending_permission(
    monkeypatch: pytest.MonkeyPatch, boundary_first: bool,
) -> None:
    from tars_agent.tui.app import PermissionSelect
    client = _PermissionRecoveryClient({"crashed-run": "interrupted", "current": "running"})
    app = _isolated_permission_app(monkeypatch, client)
    replay = []
    for index in range(60):
        request_id = f"old-{index}"
        replay.append(_permission_request(request_id, f"old-run-{index}"))
        replay.append({"type": "permission.granted" if index % 2 else "permission.denied",
                       "request_id": request_id, "decision": "allow_once" if index % 2 else "timeout"})
    replay.extend([_permission_request("crashed", "crashed-run"),
                   _permission_request("current-request", "current")])
    boundary = {"replayed_count": len(replay), "high_water_cursor": len(replay) * 2}
    async with app.run_test(size=(80, 30)) as pilot:
        app._begin_permission_replay()
        if boundary_first:
            app._set_permission_replay_boundary(boundary)
        for index, event in enumerate(replay, start=1):
            await app._handle_event_envelope({"kind": "event", "cursor": index * 2, "event": event})
            app._handle_event(event)
            if index % 20 == 0:
                await pilot.pause()
                assert list(app.query(PermissionSelect)) == []
        if not boundary_first:
            app._set_permission_replay_boundary(boundary)
        task = app._permission_replay_task
        if task is not None:
            await task
        await pilot.pause()
        groups = list(app.query(PermissionSelect))
        assert len(groups) == 1 and groups[0]._request_id == "current-request"
        assert set(app._pending_permission_blocks) == {"current-request"}
        assert app.focused is groups[0]
        assert not app._permission_replaying
        assert app._busy and app._active_run_id == "current"
        assert all(method == "run.get" for method, _ in client.calls)


async def test_unconfirmed_decision_remains_recoverable(monkeypatch: pytest.MonkeyPatch) -> None:
    from tars_agent.tui.app import PermissionSelect
    client = _PermissionRecoveryClient({"current": "running"}, fail_response=True)
    app = _isolated_permission_app(monkeypatch, client)
    async with app.run_test(size=(80, 24)) as pilot:
        app._handle_event(_permission_request("current-request", "current"))
        await pilot.pause()
        original = app.query_one(PermissionSelect)
        await app.on_permission_select_decided(PermissionSelect.Decided(original, "current-request", "allow_once"))
        assert "current-request" in app._pending_permission_blocks
        assert "current-request" not in app._resolved_permission_ids
        app._begin_permission_replay()
        app._set_permission_replay_boundary({"replayed_count": 0, "high_water_cursor": 0})
        task = app._permission_replay_task
        if task is not None:
            await task
        await pilot.pause()
        assert app.query_one(PermissionSelect) is original
        assert original._valid and not original.disabled
        assert app.focused is original


async def test_truncated_replay_does_not_activate_an_approval_before_later_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.tui.app import PermissionSelect
    app = _isolated_permission_app(monkeypatch, _PermissionRecoveryClient({"current": "running"}))
    async with app.run_test(size=(80, 30)) as pilot:
        app._begin_permission_replay()
        app._set_permission_replay_boundary({"replayed_count": 1, "high_water_cursor": 10,
                                             "replay_truncated": True})
        event = _permission_request("old", "current")
        await app._handle_event_envelope({"kind": "event", "cursor": 1, "event": event})
        app._handle_event(event)
        await pilot.pause()
        assert not list(app.query(PermissionSelect))
        assert app._permission_replaying
        app._begin_permission_replay()
        app._set_permission_replay_boundary({"replayed_count": 1, "high_water_cursor": 10,
                                             "replay_truncated": False})
        event = {"type": "permission.granted", "request_id": "old", "decision": "allow_once"}
        await app._handle_event_envelope({"kind": "event", "cursor": 2, "event": event})
        app._handle_event(event)
        task = app._permission_replay_task
        if task is not None:
            await task
        await pilot.pause()
        assert not list(app.query(PermissionSelect))
        assert not app._pending_permission_blocks
        assert not app._permission_replaying


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    values = [channel / 255 for channel in rgb]
    linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
              for channel in values]
    return sum(weight * value for weight, value in zip((0.2126, 0.7152, 0.0722), linear))


async def test_semantic_ansi_colors_follow_actual_theme_switch_with_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.segment import Segment
    from rich.style import Style
    from textual.color import Color
    from textual.filter import ANSIToTruecolor

    from tars_agent.tui.app import ChatTextArea
    app = _isolated_permission_app(monkeypatch, _PermissionRecoveryClient())
    original = [Segment("state", Style.parse(name)) for name in ("cyan", "yellow", "green", "red")]
    seen = []
    async with app.run_test(size=(80, 24)) as pilot:
        for theme, backgrounds in (("textual-dark", ((18, 18, 18), (48, 48, 48))),
                                   ("textual-light", ((224, 224, 224), (216, 216, 216)))):
            app.theme = theme
            await pilot.pause()
            converter = next(item for item in app._filters if isinstance(item, ANSIToTruecolor))
            assert converter.enabled
            converted = converter.apply(original, Color(*backgrounds[0]))
            colors = [tuple(segment.style.color.triplet) for segment in converted]
            seen.append(colors)
            for color in colors:
                for background in backgrounds:
                    light, dark = sorted((_relative_luminance(color), _relative_luminance(background)), reverse=True)
                    assert (light + 0.05) / (dark + 0.05) >= 4.5
            prompt = app.query_one("#prompt", ChatTextArea)
            prompt.disabled = False
            prompt.focus()
            await pilot.pause()
            assert prompt.styles.border_title_color != prompt.styles.border_title_background
        assert seen[0] != seen[1]
        assert not app.native_ansi_color


async def test_repeated_ctrl_x_dispatches_only_one_cancel_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    from tars_agent.tui.app import ChatTextArea
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    class Client:
        async def send_command(self, method: str, params: dict) -> dict:
            calls.append((method, params))
            entered.set()
            await release.wait()
            return {"status": "cancelled"}
    app = _isolated_permission_app(monkeypatch, Client())  # type: ignore[arg-type]
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt", ChatTextArea)
        prompt.disabled = False
        prompt.focus()
        await pilot.press("ctrl+x")
        await asyncio.wait_for(entered.wait(), 1)
        await pilot.press("ctrl+x")
        assert calls == [("run.cancel", {"run_id": "current"})]
        release.set()
        await pilot.pause()
        await pilot.press("ctrl+x")
        assert len(calls) == 1


async def test_actual_static_widget_output_changes_color_with_theme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The command runner sets NO_COLOR; the real colored ConPTY does not.
    monkeypatch.delenv("NO_COLOR", raising=False)
    from textual.geometry import Region
    from textual.widgets import Static

    from tars_agent.tui.app import PermissionSelect
    app = _isolated_permission_app(monkeypatch, _PermissionRecoveryClient())
    async with app.run_test(size=(100, 40)) as pilot:
        app._handle_event_inner({"type": "run.started", "run_id": "RUN-COLOR-ID", "goal": "goal"})
        app._handle_event_inner({"type": "subagent.started", "run_id": "child", "description": "CHILD-COLOR-TEXT"})
        app._handle_event_inner({"type": "run.finished", "run_id": "failed-run", "status": "failed", "steps": 1})
        app._handle_event_inner({"type": "run.finished", "run_id": "cancelled-run", "status": "cancelled", "steps": 1})
        app._handle_event_inner(_permission_request("color-request", "current"))
        tool = ToolCallBlock("read_file", {"path": "report.md"})
        await app.mount(tool, before="#prompt")
        tool.set_result("failed output", 1, is_error=True)
        await pilot.pause()
        targets = []
        for marker in ("RUN-COLOR-ID", "CHILD-COLOR-TEXT", "已停止"):
            widget = next(item for item in app.query(Static)
                          if isinstance(item.content, str) and marker in item.content)
            targets.append((widget, marker))
        targets += [(app.query_one(".run-err", Static), "failed"),
                    (app.query_one(PermissionSelect), "Allow once")]
        summary = next(item for item in tool.query(Static) if "failed" in str(item.content))
        targets.append((summary, "failed"))
        phase_colors = []
        for theme in ("textual-dark", "textual-light", "textual-dark"):
            app.theme = theme
            await pilot.pause()
            colors = []
            for widget, marker in targets:
                strips = widget.render_lines(Region(0, 0, widget.region.width, widget.region.height))
                segments = [segment for strip in strips for segment in strip]
                matches = [segment for segment in segments if marker in segment.text]
                assert matches, (theme, marker, widget.size, [segment.text for segment in segments])
                segment = matches[0]
                assert segment.style is not None and segment.style.color is not None
                rgb = tuple(segment.style.color.get_truecolor())
                background = (tuple(segment.style.bgcolor.get_truecolor()) if segment.style.bgcolor
                              else app.screen.styles.background.rgb)
                light, dark = sorted((_relative_luminance(rgb), _relative_luminance(background)), reverse=True)
                assert (light + 0.05) / (dark + 0.05) >= 4.5, (theme, marker, rgb, background)
                colors.append(rgb)
            phase_colors.append(colors)
        assert phase_colors[0] != phase_colors[1]
        assert phase_colors[0] == phase_colors[2]
