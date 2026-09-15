from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from rich.markdown import Markdown
from rich.markup import escape
from rich.terminal_theme import TerminalTheme
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Label, Static, TextArea

from tars_agent.core.config import TarsConfig
from tars_agent.core.skills.loader import SkillLoader
from tars_agent.core.transport.message_submission import (
    SendCommand,
    new_client_message_id,
)
from tars_agent.core.transport.socket_client import IpcError, SocketClient

log = logging.getLogger(__name__)


_ANSI_DARK = TerminalTheme(
    (18, 18, 18), (224, 224, 224),
    [(18, 18, 18), (255, 130, 130), (111, 217, 159), (255, 211, 106),
     (137, 188, 255), (217, 166, 255), (99, 218, 231), (224, 224, 224)],
)
_ANSI_LIGHT = TerminalTheme(
    (224, 224, 224), (32, 32, 32),
    [(248, 248, 248), (151, 32, 45), (22, 92, 47), (112, 64, 0),
     (21, 71, 128), (111, 39, 125), (0, 83, 99), (32, 32, 32)],
)


def _preview(s: str, n: int) -> str:
    return s[:n] + "…" if len(s) > n else s

def _params_str(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, indent=2)


# 从工具参数中提取最适合摘要展示的关键字段
def _param_summary(tool_name: str, params: dict[str, Any], max_len: int = 72) -> str:
    keys_by_tool = {
        "read_file": ("path",),
        "write_file": ("path",),
        "list_dir": ("path", "max_depth"),
        "bash": ("command",),
        "note_save": ("content",),
    }
    keys = keys_by_tool.get(tool_name, ())
    parts = [f"{key}={params[key]!r}" for key in keys if key in params]
    if not parts:
        parts = [f"{key}={value!r}" for key, value in list(params.items())[:2]]
    return _preview(", ".join(parts), max_len)


class LLMStreamBlock(Static):
    """在同一个 Static widget 中累积 LLM 流式 token。"""

    DEFAULT_CSS = "LLMStreamBlock { padding: 0 2; color: $text; }"

    # 初始化为空文本块
    def __init__(self) -> None:
        super().__init__("")
        self._text = ""
        self._finalized = False

    # 追加一个 token 并刷新显示
    def append_token(self, token: str) -> None:
        if self._finalized:
            return
        self._text += token
        self.update(self._text)

    # 将累积文本渲染为 Markdown，供流式块结束后显示
    def finalize_markdown(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._text.strip():
            self.update(Markdown(self._text, code_theme="monokai"))


class ToolCallBlock(Widget):
    """可折叠的工具调用块：折叠时显示摘要，点击后展开完整 params 和 output。"""

    DEFAULT_CSS = """
    ToolCallBlock { height: auto; padding: 0 2; color: $text-muted; }
    ToolCallBlock > .summary { color: $text-muted; }
    ToolCallBlock > .detail { display: none; padding: 0 2 0 4; color: $text-muted; }
    ToolCallBlock.expanded > .detail { display: block; }
    """

    # 初始化工具调用信息
    def __init__(self, tool_name: str, params: dict[str, Any]) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._params = params
        self._params_full = _params_str(params)
        self._output = ""
        self._elapsed_ms = 0
        self._is_error = False
        self._finished = False

    def compose(self) -> ComposeResult:
        yield Static(self._summary(), classes="summary")
        yield Static("", classes="detail")

    # 生成摘要行文本
    def _summary(self) -> str:
        if self._tool_name == "note_save" and self._finished and not self._is_error:
            return f"  [ansi_green]remembered[/ansi_green]  [dim]{self._elapsed_ms}ms[/dim]"

        params_pre = _param_summary(self._tool_name, self._params)
        line = f"  [dim]tool[/dim] [bold]{self._tool_name}[/bold]"
        if params_pre:
            line += f"  [dim]{params_pre}[/dim]"
        if self._finished:
            color = "ansi_red" if self._is_error else "ansi_green"
            status = "failed" if self._is_error else "done"
            hint = "  [dim](click to expand)[/dim]" if self._output else ""
            line += f"  [{color}]{status}[/{color}]  [dim]{self._elapsed_ms}ms[/dim]{hint}"
        return line

    # 工具调用完成时更新结果并刷新摘要（widget 未挂载时跳过 DOM 更新）
    def set_result(self, output: str, elapsed_ms: int, *, is_error: bool = False) -> None:
        self._output = output
        self._elapsed_ms = elapsed_ms
        self._is_error = is_error
        self._finished = True
        if self.children:
            self.query_one(".summary", Static).update(self._summary())

    # 点击时切换展开/折叠状态
    def on_click(self) -> None:
        if not self._finished:
            return
        if "expanded" in self.classes:
            self.remove_class("expanded")
        else:
            detail = self.query_one(".detail", Static)
            detail.update(
                f"[dim]params[/dim]\n{self._params_full}\n\n"
                f"[dim]output[/dim]\n{self._output}\n\n"
                f"[dim]elapsed:[/dim] {self._elapsed_ms}ms"
            )
            self.add_class("expanded")


class PermissionSelect(Static):
    """同一可见审批组包含动作摘要和选项，使用普通键盘焦点。"""

    can_focus = True

    DEFAULT_CSS = """
    PermissionSelect {
        height: auto;
        color: $text;
        background: $surface;
        border: round $foreground;
        padding: 0 1;
        margin: 1 2 0 2;
    }
    """

    _CHOICES: tuple[tuple[str, str, str], ...] = (
        ("allow_once",   "Allow once",   "y / 1"),
        ("allow_session", "Allow same parameters this session", "a / 2"),
        ("deny_once",    "Deny",         "n / 3"),
        ("deny_session",  "Deny session",  "d / 4"),
    )
    _KEY_MAP: dict[str, str] = {
        "y": "allow_once",  "1": "allow_once",
        "a": "allow_session","2": "allow_session",
        "n": "deny_once",   "3": "deny_once",
        "d": "deny_session", "4": "deny_session",
    }

    # 用户作出权限决策时发布，携带工具 ID 和决策字符串
    class Decided(Message):
        # 初始化决策消息，存储控件引用、工具 ID 和决策
        def __init__(self, widget: PermissionSelect, request_id: str, decision: str) -> None:
            self.widget = widget
            self.request_id = request_id
            self.decision = decision
            super().__init__()

    # 初始化控件，存储工具 ID（用于 IPC 回复）
    def __init__(
        self,
        request_id: str,
        choices: tuple[tuple[str, str, str], ...] | None = None,
        *,
        context: str = "",
    ) -> None:
        super().__init__("")
        self._request_id = request_id
        self._approval_context = context
        self._valid = True
        if choices is not None:
            self._CHOICES = choices
            self._KEY_MAP = {
                key: decision
                for decision, _, keys in choices
                for key in keys.replace(" ", "").split("/")
            }
        self._cursor = 0

    def deactivate(self) -> None:
        self._valid = False
        self.disabled = True
        self.display = False
        if self.is_mounted:
            self.remove()

    def suspend(self) -> None:
        self.disabled = True
        self.display = False

    def activate(self) -> None:
        if not self._valid:
            return
        self.disabled = False
        self.display = True
        if self.is_mounted:
            self.focus()
            self.call_after_refresh(self.scroll_visible, animate=False, force=True)

    def on_mount(self) -> None:
        if not self._valid:
            self.remove()
            return
        self.update(self._render_ui())
        if self.disabled:
            return
        self.focus()
        log.debug(
            "PermissionSelect.on_mount  can_focus=%s  focused_after=%r",
            self.can_focus,
            self.app.focused,
        )
        self.app.call_after_refresh(self._log_deferred_focus)
        self.call_after_refresh(self.scroll_visible, animate=False, force=True)

    # 在下一帧记录焦点是否真正转移到本控件
    def _log_deferred_focus(self) -> None:
        log.debug(
            "PermissionSelect.deferred_focus  app.focused=%r  has_focus=%s  focusable=%s",
            self.app.focused,
            self.has_focus,
            self.focusable,
        )

    # 焦点到达时记录，用于确认 focus() 是否真正生效
    def on_focus(self, event: events.Focus) -> None:
        log.debug(
            "PermissionSelect.on_focus  has_focus=%s  app.focused=%r",
            self.has_focus,
            self.app.focused,
        )

    # 焦点离开时记录，用于追踪是否被其他控件抢走焦点
    def on_blur(self, event: events.Blur) -> None:
        log.debug("PermissionSelect.on_blur  app.focused=%r", self.app.focused)

    # 生成带光标高亮的选项列表文本
    def _render_ui(self) -> str:
        lines: list[str] = [self._approval_context, ""] if self._approval_context else []
        for i, (_, label, key_hint) in enumerate(self._CHOICES):
            if i == self._cursor:
                lines.append(f"  [bold ansi_cyan]❯ {label}[/bold ansi_cyan]  [dim]{key_hint}[/dim]")
            else:
                lines.append(f"    {label}  [dim]{key_hint}[/dim]")
        lines.append("[dim]  ↑↓ navigate   enter confirm[/dim]")
        return "\n".join(lines)

    # 方向键导航；快捷键直接选择；enter 确认光标位置
    def on_key(self, event: events.Key) -> None:
        log.debug("PermissionSelect.on_key  key=%r  char=%r", event.key, event.character)
        key = event.key
        if key in ("up", "k"):
            event.stop()
            self._cursor = (self._cursor - 1) % len(self._CHOICES)
            self.update(self._render_ui())
        elif key in ("down", "j"):
            event.stop()
            self._cursor = (self._cursor + 1) % len(self._CHOICES)
            self.update(self._render_ui())
        elif key == "enter":
            event.stop()
            self._pick(self._CHOICES[self._cursor][0])
        else:
            decision = self._KEY_MAP.get(key)
            if decision is not None:
                event.stop()
                self._pick(decision)

    # 发布决策消息，由宿主 App 负责 IPC 回复和控件清理
    def _pick(self, decision: str) -> None:
        if not self._valid or self.disabled:
            return
        log.debug("PermissionSelect._pick  decision=%s", decision)
        self.post_message(self.Decided(self, self._request_id, decision))


class PermissionBlock(Static):
    """日志里的权限审批摘要"""

    _LABEL_MAP: dict[str, str] = {
        "allow_once":   "allowed (once)",
        "allow_session": "allowed (same parameters this session)",
        "allow_host_once": "host allowed (once)",
        "deny_once":    "denied",
        "deny_session":  "denied (session)",
        "timeout":      "⏱ timed out",
    }
    LABEL_MAP = _LABEL_MAP

    # 子类提交消息：用户作出权限决策时发布
    class Resolved(Message):
        def __init__(self, block: PermissionBlock, decision: str) -> None:
            self.block = block
            self.decision = decision
            super().__init__()

    # 初始化审批块，记录工具 ID、名称和参数预览
    def __init__(
        self,
        tool_use_id: str,
        tool_name: str,
        param_preview: str,
        approval_details: str = "",
    ) -> None:
        self._tool_use_id = tool_use_id
        self._tool_name = tool_name
        self._param_preview = param_preview
        self._approval_details = approval_details
        self._resolved = False
        super().__init__(self._pending_text(), classes="log-line")

    def _pending_text(self) -> str:
        preview = f"  [dim]{self._param_preview}[/dim]" if self._param_preview else ""
        details = f"\n{self._approval_details}" if self._approval_details else ""
        return (
            f"[bold ansi_red]? permission[/bold ansi_red]  [bold]{self._tool_name}[/bold]"
            f"{preview}{details}"
        )

    # 将块收缩为单行摘要并发布 Resolved 消息
    def _resolve(self, decision: str) -> None:
        if self._resolved:
            return
        self._resolved = True
        allowed = decision in ("allow_once", "allow_session", "allow_host_once")
        icon = ("[bold ansi_green]✓[/bold ansi_green]" if allowed
                else "[bold ansi_red]✗[/bold ansi_red]")
        label = self._LABEL_MAP.get(decision, decision)
        preview = f"  [dim]{self._param_preview}[/dim]" if self._param_preview else ""
        self.update(
            f"{icon} permission  [bold]{self._tool_name}[/bold]{preview}  [dim]{label}[/dim]"
        )
        self.post_message(self.Resolved(self, decision))


class SlashCompleteWidget(Static):
    """斜杠命令自动补全弹出框：输入 / 时显示可用 skill 列表并支持键盘筛选与选择。"""

    can_focus = False

    DEFAULT_CSS = """
    SlashCompleteWidget {
        height: auto;
        padding: 0 1;
        margin: 0 2;
        background: $surface;
        border: round $surface-lighten-2;
    }
    """

    # 用户选中某条命令时发布
    class Selected(Message):
        # 初始化，携带被选中的 skill 名称
        def __init__(self, skill_name: str) -> None:
            self.skill_name = skill_name
            super().__init__()

    # 初始化，接收全量 (name, description) 列表
    def __init__(self, items: list[tuple[str, str]]) -> None:
        super().__init__("")
        self._all_items = items
        self._filtered: list[tuple[str, str]] = list(items)
        self._cursor = 0

    # 根据查询字符串筛选列表，重置光标并重新渲染
    def set_query(self, query: str) -> None:
        q = query.lower()
        self._filtered = [(n, d) for n, d in self._all_items if not q or q in n.lower()]
        self._cursor = min(self._cursor, max(0, len(self._filtered) - 1))
        if self.is_attached:
            self._redraw()

    # 向上移动光标并重新渲染
    def move_up(self) -> None:
        if self._filtered:
            self._cursor = (self._cursor - 1) % len(self._filtered)
            self._redraw()

    # 向下移动光标并重新渲染
    def move_down(self) -> None:
        if self._filtered:
            self._cursor = (self._cursor + 1) % len(self._filtered)
            self._redraw()

    # 选中当前光标项并发布 Selected 消息
    def select_current(self) -> None:
        if self._filtered:
            self.post_message(self.Selected(self._filtered[self._cursor][0]))

    # 返回当前是否有可选项
    def has_selection(self) -> bool:
        return len(self._filtered) > 0

    def on_mount(self) -> None:
        self._redraw()

    # 渲染筛选后的命令列表，高亮当前光标项
    def _redraw(self) -> None:
        if not self._filtered:
            self.update("[dim]  no matching commands[/dim]")
            return
        lines: list[str] = []
        for i, (name, desc) in enumerate(self._filtered):
            desc_part = f"  [dim]{desc}[/dim]" if desc else ""
            if i == self._cursor:
                lines.append(f"  [bold ansi_cyan]❯ /{name}[/bold ansi_cyan]{desc_part}")
            else:
                lines.append(f"    [ansi_cyan]/{name}[/ansi_cyan]{desc_part}")
        lines.append("[dim]  ↑↓ navigate   tab/enter select   esc dismiss[/dim]")
        self.update("\n".join(lines))


class ChatTextArea(TextArea):
    """支持 Enter 提交、Cmd/Shift/Alt+Enter 换行的多行聊天输入框。"""

    DEFAULT_CSS = """
    ChatTextArea {
        height: auto;
        min-height: 3;
        max-height: 12;
        border: round $surface-lighten-2;
        background: $background;
        padding: 0 1;
        margin: 1 2;
        scrollbar-size-vertical: 1;
    }
    ChatTextArea:focus {
        border: round $primary;
        border-title-color: $foreground;
        border-title-background: $background;
        border-title-style: bold;
        background: $background;
    }
    """

    # 子类自定义的提交消息，供宿主 App 监听
    class Submitted(Message):
        def __init__(self, area: ChatTextArea) -> None:
            self.text_area = area
            self.value = area.text
            super().__init__()

    # 输入内容以 / 开头且无空格时发布，query 为 / 之后的字符串（可为空串）；None 表示收起弹窗
    class SlashChanged(Message):
        def __init__(self, query: str | None) -> None:
            self.query = query
            super().__init__()

    # 文本变化时检测 / 前缀，通知宿主 App 更新自动补全弹窗
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = self.text
        if text.startswith("/") and " " not in text:
            self.post_message(ChatTextArea.SlashChanged(query=text[1:]))
        else:
            self.post_message(ChatTextArea.SlashChanged(query=None))

    # Enter 提交；↑↓/Tab/Esc 路由到自动补全弹窗；Cmd/Shift/Alt+Enter 插入换行；其余键交回 TextArea
    async def _on_key(self, event: events.Key) -> None:
        key = event.key

        popup: SlashCompleteWidget | None = None
        try:
            popup = self.app.query_one(SlashCompleteWidget)
        except NoMatches:
            popup = None

        if key == "enter":
            event.stop()
            event.prevent_default()
            if popup is not None and popup.has_selection():
                popup.select_current()
                return
            if self.text.strip():
                self.post_message(self.Submitted(self))
            return
        if key in ("alt+enter", "shift+enter", "ctrl+j", "super+enter"):
            event.stop()
            event.prevent_default()
            if not self.read_only:
                self.insert("\n")
            return
        if popup is not None:
            if key == "up":
                event.stop()
                event.prevent_default()
                popup.move_up()
                return
            elif key == "down":
                event.stop()
                event.prevent_default()
                popup.move_down()
                return
            elif key == "tab":
                event.stop()
                event.prevent_default()
                popup.select_current()
                return
            elif key == "escape":
                event.stop()
                event.prevent_default()
                self.post_message(ChatTextArea.SlashChanged(query=None))
                return
        await super()._on_key(event)


class TarsTuiApp(App[None]):
    """TARS-Agent TUI：终端滚屏风格，实时展示 agent 执行过程。"""

    TITLE = "TARS-Agent"
    BINDINGS = [
        Binding("ctrl+q", "quit", "退出界面，后台继续"),
        Binding("ctrl+x", "stop", "停止当前任务", priority=True),
    ]
    CSS = """
    Screen { background: $background; }
    #header {
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    #log-view {
        height: 1fr;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
    }
    #banner { color: $text; padding: 1 2 0 2; }
    #prompt.permission-waiting:disabled {
        opacity: 1;
        text-opacity: 1;
        border: round $foreground;
        border-title-color: $foreground;
        border-title-background: $background;
        border-title-style: bold;
    }
    Static.user-turn { color: $text; padding: 1 2 0 2; }
    Static.run-header { color: $text-muted; padding: 1 2 0 2; }
    Static.step-divider { color: $text-muted; padding: 0 2; }
    Static.run-ok { color: ansi_green; padding: 0 2 1 2; }
    Static.run-err { color: ansi_red; padding: 0 2 1 2; }
    Static.usage { padding: 0 2; }
    Static.log-line { padding: 0 2; }
    """

    _BANNER = (
        "[bold]TARS-Agent[/bold]\n"
        "输入消息开始对话；输入 / 使用技能。\n"
        "Ctrl+X 停止当前任务。\n"
        "Ctrl+Q 退出界面，后台任务继续运行。"
    )
    _PROMPT_HINT = "输入消息 · Enter 发送 · Shift+Enter 换行"
    _PERMISSION_HINT = "等待审批 · 输入已暂停"

    # 初始化连接参数和 TUI 内部状态
    def __init__(
        self,
        host: str,
        port: int,
        resume_session_id: str | None = None,
    ) -> None:
        super().__init__(ansi_color=False)
        self.ansi_theme_dark = _ANSI_DARK
        self.ansi_theme_light = _ANSI_LIGHT
        self._host = host
        self._port = port
        self._workspace_root = str(Path.cwd().resolve())
        self._requested_session_id = resume_session_id
        self._client: SocketClient | None = None
        self._current_llm: LLMStreamBlock | None = None
        self._pending_tool_blocks: dict[str, ToolCallBlock] = {}
        self._pending_permission_blocks: dict[str, PermissionBlock] = {}
        self._session_id: str | None = None
        self._last_cursor = 0
        self._busy = False
        self._active_run_id: str | None = None
        self._cancellation_requested = False
        self._permission_runs: dict[str, str] = {}
        self._permission_selects: dict[str, PermissionSelect] = {}
        self._resolved_permission_ids: set[str] = set()
        self._permission_replaying = False
        self._permission_replay_expected: int | None = None
        self._permission_replay_high_water = 0
        self._permission_replay_truncated = False
        self._permission_replay_seen: set[int] = set()
        self._permission_replay_generation = 0
        self._permission_replay_task: asyncio.Task[None] | None = None
        self._last_context_pct: float = 0.0
        self._slash_items: list[tuple[str, str]] = []
        self._subagent_run_ids: dict[str, str] = {}  # child run_id -> description
        self._subagent_start_times: dict[str, float] = {}  # child run_id -> start time

    def compose(self) -> ComposeResult:
        yield Label("[bold]TARS-Agent[/bold]  [dim]connecting...[/dim]", id="header")
        yield VerticalScroll(id="log-view")
        yield ChatTextArea(id="prompt", show_line_numbers=False)

    def on_mount(self) -> None:
        self._slash_items = self._build_slash_items()
        self._append(Static(self._BANNER, id="banner"))
        self.run_worker(self._socket_loop(), exclusive=True, name="socket")
        prompt = self.query_one("#prompt", ChatTextArea)
        prompt.disabled = True
        prompt.border_title = "connecting..."

    # 构建斜杠命令候选列表：内建命令 + 所有已注册 skill
    def _build_slash_items(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = [("compact", "compress context window")]
        try:
            loader = SkillLoader()
            for skill in loader.list_all_skills():
                desc = skill.description.splitlines()[0] if skill.description else ""
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                items.append((skill.name, desc))
        except Exception:
            pass
        return items

    # 根据 / 前缀查询字符串挂载、更新或移除自动补全弹窗
    def on_chat_text_area_slash_changed(self, event: ChatTextArea.SlashChanged) -> None:
        query = event.query
        if query is None:
            try:
                self.query_one(SlashCompleteWidget).remove()
            except NoMatches:
                pass
            return
        try:
            popup = self.query_one(SlashCompleteWidget)
            popup.set_query(query)
        except NoMatches:
            popup = SlashCompleteWidget(self._slash_items)
            self.mount(popup, before="#prompt")
            popup.set_query(query)

    # 用户选中自动补全项后将 /{name} 填入输入框并移除弹窗
    def on_slash_complete_widget_selected(self, event: SlashCompleteWidget.Selected) -> None:
        prompt = self._prompt()
        if prompt is not None:
            prompt.text = f"/{event.skill_name} "
            prompt.move_cursor(prompt.document.end)
        try:
            self.query_one(SlashCompleteWidget).remove()
        except NoMatches:
            pass

    # 记录按键焦点；当 PermissionSelect 失去焦点后作为兜底处理权限快捷键
    def on_key(self, event: events.Key) -> None:
        log.debug("App.on_key  key=%r  focused=%r", event.key, self.focused)
        if not self._pending_permission_blocks:
            return
        try:
            select = next((item for item in self._permission_selects.values()
                           if item.is_mounted and not item.disabled), None)
            if select is None:
                return
            if select.has_focus:
                return  # PermissionSelect 有焦点时自行处理，事件不会冒泡到这里
            key = event.key
            decision = select._KEY_MAP.get(key)
            if decision:
                event.stop()
                select._pick(decision)
            elif key in ("up", "k"):
                event.stop()
                select._cursor = (select._cursor - 1) % len(select._CHOICES)
                select.update(select._render_ui())
            elif key in ("down", "j"):
                event.stop()
                select._cursor = (select._cursor + 1) % len(select._CHOICES)
                select.update(select._render_ui())
            elif key == "enter":
                event.stop()
                select._pick(select._CHOICES[select._cursor][0])
        except Exception:
            pass

    # 退出只断开客户端；Session 与 Run 生命周期由显式 close/cancel 命令管理
    async def action_quit(self) -> None:
        self.exit()

    def action_stop(self) -> None:
        if self._busy and not self._cancellation_requested:
            self._cancellation_requested = True
            self.run_worker(self._cancel_current_run(), name="cancel", exclusive=False)

    async def _cancel_current_run(self) -> None:
        client = self._client
        if client is None:
            self._cancellation_requested = False
            self._append(Static(
                "[ansi_yellow]Disconnected; cancellation is not confirmed.[/ansi_yellow]"
            ))
            return
        try:
            run_id = self._active_run_id
            if run_id is None and self._session_id is not None:
                snapshot = await client.send_command(
                    "session.get", {"session_id": self._session_id},
                )
                run_id = snapshot.get("session", {}).get("active_run_id")
            if run_id is None:
                self._cancellation_requested = False
                return
            self._active_run_id = run_id
            self._append(Static(
                "[ansi_yellow]已请求取消，尚未确认停止[/ansi_yellow]", classes="log-line",
            ))
            result = await client.send_command("run.cancel", {"run_id": run_id})
            if result.get("status") not in {"succeeded", "failed", "cancelled", "interrupted"}:
                self._append(Static("[ansi_yellow]已请求取消，尚未确认停止[/ansi_yellow]"))
                return
            self._append(Static(
                f"[ansi_yellow]停止清理已结束；任务状态：{escape(str(result.get('status')))}[/ansi_yellow]",
                classes="log-line",
            ))
            self._restore_input()
        except IpcError as exc:
            if exc.code == -32033:
                message = "已请求取消，尚未确认停止；后台继续清理"
            else:
                self._cancellation_requested = False
                message = f"停止未确认：{exc}"
            self._append(Static(
                f"[ansi_yellow]{escape(message)}[/ansi_yellow]", classes="log-line",
            ))
        except (OSError, RuntimeError, TimeoutError) as exc:
            self._cancellation_requested = False
            self._append(Static(f"[ansi_yellow]停止未确认：{escape(str(exc))}[/ansi_yellow]"))

    def _resolve_permission(self, request_id: str, decision: str) -> None:
        if not request_id:
            return
        self._resolved_permission_ids.add(request_id)
        self._permission_runs.pop(request_id, None)
        selector = self._permission_selects.pop(request_id, None)
        if selector is not None:
            selector.deactivate()
        block = self._pending_permission_blocks.pop(request_id, None)
        if block is not None:
            block._resolve(decision)

    def _clear_run_permissions(self, run_id: str | None, decision: str = "closed") -> None:
        for request_id, owner in tuple(self._permission_runs.items()):
            if owner == run_id:
                self._resolve_permission(request_id, decision)

    def _refresh_permission_input(self) -> None:
        prompt = self._prompt()
        if prompt is None:
            return
        pending = bool(self._pending_permission_blocks)
        prompt.disabled = self._permission_replaying or pending or self._busy
        prompt.read_only = False
        prompt.set_class(pending, "permission-waiting")
        prompt.border_title = (
            "恢复会话中..." if self._permission_replaying else self._PERMISSION_HINT if pending
            else "agent is working..." if self._busy else self._PROMPT_HINT
        )
        if not prompt.disabled:
            prompt.focus()

    def _restore_input(self) -> None:
        self._clear_run_permissions(self._active_run_id)
        self._busy = False
        self._active_run_id = None
        self._cancellation_requested = False
        self._refresh_permission_input()
        self._update_header("ready")

    # 将输入框提交内容发送给当前 chat session；用 worker 发送，避免 await 阻塞 App 消息泵
    async def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        content = event.value.strip()
        if not content:
            return
        # 检测 /compact 指令
        if content == "/compact":
            event.text_area.text = ""
            if self._client is not None and self._session_id is not None and not self._busy:
                self.run_worker(self._do_compact(), name="compact", exclusive=False)
            return
        if self._client is None or self._session_id is None or self._busy:
            self._append(Static(
                "[ansi_yellow]agent busy or disconnected[/ansi_yellow]", classes="log-line",
            ))
            return
        self._busy = True
        self._active_run_id = None
        self._cancellation_requested = False
        prompt = event.text_area
        prompt.text = ""
        prompt.disabled = True
        prompt.read_only = False
        prompt.border_title = "agent is working..."
        self._update_header("running")
        client_message_id = new_client_message_id()
        self.run_worker(
            self._do_send_message(content, client_message_id),
            name="send_message",
            exclusive=False,
        )

    # 在 worker 中执行手动压缩命令，完成后显示结果横幅
    async def _do_compact(self) -> None:
        if self._client is None or self._session_id is None:
            return
        self._append(Static("[dim]⚡ compacting context...[/dim]", classes="log-line"))
        try:
            result = await self._client.send_command(
                "session.compact",
                {"session_id": self._session_id, "focus": ""},
            )
            summary_tokens = result.get("summary_tokens", 0)
            saved_tokens = result.get("saved_tokens", 0)
            self._last_context_pct = 0.0
            self._append(Static(
                f"[bold ansi_cyan]⚡ Context compacted[/bold ansi_cyan]"
                f"  [dim]summary={summary_tokens} tokens  saved≈{saved_tokens} tokens[/dim]",
                classes="log-line",
            ))
        except (IpcError, RuntimeError, OSError) as e:
            self._append(Static(f"[ansi_red]compact error: {e}[/ansi_red]", classes="log-line"))

    # 在 worker 中执行 IPC 发送，使 App 消息泵在 agent 运行期间仍能处理键盘/焦点等消息
    async def _do_send_message(self, content: str, client_message_id: str) -> None:
        client = self._client
        if client is None or self._session_id is None:
            return
        try:
            result = await client.send_command("session.send_message", {
                "session_id": self._session_id,
                "content": content,
                "client_message_id": client_message_id,
            })
            if self._busy:
                self._active_run_id = str(result["run_id"])
        except (IpcError, RuntimeError, OSError) as e:
            self._busy = False
            prompt = self._prompt()
            if prompt is not None:
                prompt.disabled = False
                prompt.read_only = False
                prompt.remove_class("permission-waiting")
                prompt.border_title = self._PROMPT_HINT
            self._update_header("ready")
            self._append(Static(
                f"[ansi_red]send not confirmed: {escape(str(e))}; reconnect restores state only. "
                "The message was not sent again.[/ansi_red]", classes="log-line",
            ))

    # 处理内联审批控件的用户决策：发送 IPC 响应并恢复输入框
    async def on_permission_select_decided(self, msg: PermissionSelect.Decided) -> None:
        request_id = msg.request_id
        if (request_id not in self._pending_permission_blocks
                or self._permission_selects.get(request_id) is not msg.widget):
            msg.widget.deactivate()
            return
        if self._permission_replaying or msg.widget.disabled:
            return
        msg.widget.suspend()
        if self._client is None:
            return
        try:
            response = await self._client.send_command("permission.respond", {
                "request_id": request_id, "session_id": self._session_id, "decision": msg.decision,
            })
        except (IpcError, RuntimeError, OSError):
            # The server may still be waiting. Preserve this request for cursor
            # recovery instead of guessing whether an ambiguous response was applied.
            self._append(Static(
                "[ansi_yellow]审批响应尚未确认，重连后核对。[/ansi_yellow]", classes="log-line",
            ))
            return
        self._resolve_permission(request_id, msg.decision if response.get("ok") else "expired")
        self._refresh_permission_input()

    # 向日志视图追加一个 widget 并滚动到底部
    def _append(self, widget: Widget) -> None:
        log_view = self.query_one("#log-view", VerticalScroll)
        log_view.mount(widget)
        log_view.scroll_end(animate=False)

    # 结束当前 LLM 流式块（下一个 token 将开启新块）
    def _break_llm(self) -> None:
        if self._current_llm is not None:
            self._current_llm.finalize_markdown()
        self._current_llm = None

    # 将选择控件挂载到 Screen 顶层（#prompt 之前），避免 VerticalScroll 争抢焦点
    def _mount_permission_select(self, select: PermissionSelect) -> None:
        if self._permission_selects.get(select._request_id) is not select:
            select.deactivate()
            return
        if self._permission_replaying:
            select.suspend()
            return
        select.activate()
        if not select.is_mounted:
            self.mount(select, before="#prompt")

    def _begin_permission_replay(self) -> None:
        self._permission_replay_generation += 1
        if self._permission_replay_task is not None:
            self._permission_replay_task.cancel()
            self._permission_replay_task = None
        self._permission_replaying = True
        self._permission_replay_expected = None
        self._permission_replay_truncated = False
        self._permission_replay_seen.clear()
        for select in self._permission_selects.values():
            select.suspend()
        self._refresh_permission_input()

    def _set_permission_replay_boundary(self, result: dict[str, Any]) -> None:
        self._permission_replay_truncated = bool(result.get("replay_truncated", False))
        self._permission_replay_expected = int(result.get("replayed_count", 0))
        self._permission_replay_high_water = int(result.get("high_water_cursor", self._last_cursor))
        self._maybe_finish_permission_replay()

    def _maybe_finish_permission_replay(self) -> None:
        expected = self._permission_replay_expected
        if (not self._permission_replaying or expected is None or self._permission_replay_task
                or self._permission_replay_truncated):
            return
        received = sum(cursor <= self._permission_replay_high_water
                       for cursor in self._permission_replay_seen)
        if received >= expected:
            self._permission_replay_task = asyncio.create_task(
                self._finish_permission_replay(self._permission_replay_generation)
            )

    async def _finish_permission_replay(self, generation: int) -> None:
        try:
            client = self._client
            for run_id in set(self._permission_runs.values()):
                if client is None:
                    return
                try:
                    result = await client.send_command("run.get", {"run_id": run_id})
                except IpcError as exc:
                    if exc.code != -32030:
                        raise
                    result = {"status": "missing"}
                if generation != self._permission_replay_generation:
                    return
                if result.get("status") not in {"queued", "running"}:
                    self._clear_run_permissions(run_id)
            if generation != self._permission_replay_generation:
                return
            self._permission_replaying = False
            for select in tuple(self._permission_selects.values()):
                self._mount_permission_select(select)
            self._refresh_permission_input()
            self._update_header("running" if self._busy else "ready")
        except (IpcError, OSError, RuntimeError):
            log.exception("pending permission recovery is not yet confirmed")
        finally:
            if generation == self._permission_replay_generation:
                self._permission_replay_task = None

    # 安全获取输入框，便于组件测试中未挂载时跳过 UI 操作
    def _prompt(self) -> ChatTextArea | None:
        try:
            return self.query_one("#prompt", ChatTextArea)
        except Exception:
            return None

    # 生成 context 占用率的彩色进度条字符串
    def _render_ctx_bar(self, pct: float) -> str:
        filled = int(pct * 20)
        bar = "█" * filled + "░" * (20 - filled)
        label = f"ctx:{pct * 100:.1f}%"
        if pct >= 0.85:
            color = "bold ansi_red"
        elif pct >= 0.70:
            color = "ansi_yellow"
        else:
            color = "dim"
        return f"[{color}]{label} {bar}[/{color}]"

    # 根据连接和运行状态刷新顶部标题
    def _update_header(self, state: str) -> None:
        try:
            header = self.query_one("#header", Label)
        except NoMatches:
            return
        session = f"  [dim]{self._session_id}[/dim]" if self._session_id else ""
        color = {
            "ready": "ansi_green",
            "running": "ansi_yellow",
            "disconnected": "ansi_red",
            "connecting": "dim",
        }.get(state, "dim")
        header.update(
            f"[bold]TARS-Agent[/bold]  [dim]{self._host}:{self._port}[/dim]"
            f"{session}  [{color}]{state}[/{color}]"
        )

    async def _attach_session(self, send_command: SendCommand) -> None:
        """Create once, then refresh durable Session state on every reconnect."""

        if self._session_id is None and self._requested_session_id is None:
            created = await send_command(
                "session.create",
                {"mode": "chat", "workspace_root": self._workspace_root},
            )
            self._session_id = str(created["session_id"])
            self._busy = False
            return

        session_id = self._session_id or self._requested_session_id
        assert session_id is not None
        resumed = await send_command("session.resume", {"session_id": session_id})
        self._session_id = str(resumed["session"]["session_id"])
        active_run = resumed.get("active_run")
        self._busy = bool(
            active_run and active_run.get("status") in {"queued", "running"}
        )
        self._active_run_id = (
            str(active_run["run_id"]) if self._busy and isinstance(active_run, dict) else None
        )
        if not self._busy:
            self._cancellation_requested = False

    # 管理 SocketClient 生命周期：连接、订阅事件、断线重连
    async def _socket_loop(self) -> None:
        header = self.query_one("#header", Label)

        while True:
            client = SocketClient(self._host, self._port)
            self._client = None
            try:
                await client.connect()
            except (ConnectionRefusedError, OSError):
                log.warning("connection refused %s:%s, retrying", self._host, self._port)
                self._update_header("disconnected")
                await asyncio.sleep(2)
                continue

            log.info("connected to %s:%s", self._host, self._port)
            self._client = client
            self._update_header("connecting")
            loop_task = asyncio.create_task(client.run_event_loop())

            async def on_event(event: dict[str, Any]) -> None:
                self._handle_event(event)

            client.on_event(on_event)

            client.on_event_envelope(self._handle_event_envelope)

            try:
                loop_task.add_done_callback(
                    lambda t: log.error("loop_task failed: %s", t.exception())
                    if not t.cancelled() and t.exception() is not None
                    else None
                )
                self._begin_permission_replay()
                await self._attach_session(client.send_command)

                params: dict[str, Any] = {
                    "topics": [
                        "session.*",
                        "run.*",
                        "step.*",
                        "tool.*",
                        "llm.token",
                        "llm.usage",
                        "log.*",
                        "permission.*",
                        "context.*",
                        "subagent.*",
                        "skill.*",
                    ],
                    "session_id": self._session_id,
                    "after_cursor": self._last_cursor,
                }
                subscription = await client.send_command("event.subscribe", params)
                self._set_permission_replay_boundary(subscription)
                log.info(
                    "session attached session_id=%s after_cursor=%d",
                    self._session_id,
                    self._last_cursor,
                )
                self._refresh_permission_input()
                if not self._permission_replaying:
                    self._update_header("running" if self._busy else "ready")
                await loop_task
            except IpcError as e:
                header.update(f"[bold]TARS-Agent[/bold]  [ansi_red]subscribe error: {e}[/ansi_red]")
            finally:
                if self._permission_replay_task is not None:
                    self._permission_replay_task.cancel()
                    await asyncio.gather(self._permission_replay_task, return_exceptions=True)
                    self._permission_replay_task = None
                for select in self._permission_selects.values():
                    select.suspend()
                if not loop_task.done():
                    loop_task.cancel()
                self._client = None
                prompt = self._prompt()
                if prompt is not None:
                    prompt.disabled = True
                    prompt.read_only = False
                    prompt.border_title = "disconnected, retrying..."
                self._break_llm()
                await client.close()

            self._update_header("disconnected")
            await asyncio.sleep(2)

    async def _handle_event_envelope(self, envelope: dict[str, Any]) -> None:
        """Track resumable cursors without assuming every envelope is an event."""

        cursor = envelope.get("cursor")
        if isinstance(cursor, int):
            self._last_cursor = max(self._last_cursor, cursor)
        kind = envelope.get("kind")
        if kind == "event" and self._permission_replaying and isinstance(cursor, int):
            self._permission_replay_seen.add(cursor)
        if kind == "compatibility.error":
            rendered = json.dumps(envelope, ensure_ascii=False, sort_keys=True)
            log.error("event compatibility error: %s", rendered)
            try:
                self._append(
                    Static(
                        "[bold ansi_red]event compatibility error[/bold ansi_red]\n"
                        f"{escape(rendered)}",
                        classes="run-err",
                    )
                )
            except Exception:
                log.debug("cannot render compatibility error during TUI teardown")
            return
        if kind != "overflow":
            return
        last_cursor = envelope.get("last_cursor")
        if isinstance(last_cursor, int):
            self._last_cursor = max(self._last_cursor, last_cursor)
        reason = str(envelope.get("reason", "event subscription overflow"))
        log.warning(
            "event subscription overflow reason=%s resume_cursor=%d",
            reason,
            self._last_cursor,
        )
        try:
            self._append(
                Static(
                    "[ansi_yellow]event stream overflowed; reconnecting from "
                    f"cursor {self._last_cursor}[/ansi_yellow]",
                    classes="log-line",
                )
            )
        except Exception:
            # The socket loop can receive overflow during App teardown, when the
            # log widget no longer exists.  Cursor recovery must still succeed.
            log.debug("cannot render overflow notice during TUI teardown")

    # 根据事件 type 路由到对应渲染逻辑；捕获异常防止 socket loop 因单个事件崩溃
    def _handle_event(self, event: dict[str, Any]) -> None:
        try:
            self._handle_event_inner(event)
        except Exception:
            log.exception("_handle_event crashed  event_type=%s", event.get("type", "?"))
        finally:
            self._maybe_finish_permission_replay()

    # 实际的事件路由逻辑
    def _handle_event_inner(self, event: dict[str, Any]) -> None:
        t = event.get("type", "")

        if t in {"session.created", "session.resumed", "tool.execution_started", "step.finished"}:
            # These V2 notifications carry no additional user action. In particular,
            # backend/container metadata does not belong in the conversation log.
            # Current run/input state remains owned by the attach RPC and terminal events.
            return

        if t == "llm.token":
            token = event.get("token", "")
            if self._current_llm is None:
                llm_block = LLMStreamBlock()
                self._append(llm_block)
                self._current_llm = llm_block
            self._current_llm.append_token(token)
            return

        self._break_llm()

        if t == "session.message_received":
            content = str(event.get("content", ""))
            self._append(Static(f"[bold]>[/bold] {content}", classes="user-turn"))

        elif t == "session.waiting_for_input":
            if (self._active_run_id is None
                    or event.get("last_run_id") == self._active_run_id):
                self._restore_input()

        elif t == "session.closed":
            self._active_run_id = None
            self._cancellation_requested = False
            self._busy = False
            prompt = self._prompt()
            if prompt is not None:
                prompt.disabled = True
                prompt.read_only = False
                prompt.border_title = "session closed"
            self._update_header("disconnected")

        elif t == "run.started":
            run_id = event.get("run_id", "")
            if run_id not in self._subagent_run_ids and self._active_run_id is None:
                self._active_run_id = str(run_id)
            goal = event.get("goal", "")
            self._append(Static(
                f"[dim]run[/dim]  [ansi_cyan]{run_id}[/ansi_cyan]  [dim]{_preview(goal, 96)}[/dim]",
                classes="run-header",
            ))

        elif t == "skill.invoked":
            skill_name = event.get("skill_name", "")
            arguments = event.get("arguments", "")
            args_preview = _preview(arguments, 80) if arguments else ""
            args_part = f"  [dim]{args_preview}[/dim]" if args_preview else ""
            self._append(Static(
                f"[bold ansi_cyan]/{skill_name}[/bold ansi_cyan]{args_part}",
                classes="log-line",
            ))

        elif t == "subagent.started":
            run_id = event.get("run_id", "")
            description = event.get("description", "")
            self._subagent_run_ids[run_id] = description
            self._subagent_start_times[run_id] = time.monotonic()
            short_id = run_id[:8] if len(run_id) >= 8 else run_id
            self._append(Static(
                f"[dim]┌─[/dim] [ansi_cyan]{_preview(description, 72)}[/ansi_cyan]"
                f"  [dim]{short_id}[/dim]",
                classes="log-line",
            ))

        elif t == "subagent.finished":
            run_id = event.get("run_id", "")
            self._clear_run_permissions(str(run_id))
            if not self._busy:
                self._restore_input()
            status = event.get("status", "")
            description = self._subagent_run_ids.pop(run_id, event.get("description", ""))
            start = self._subagent_start_times.pop(run_id, None)
            elapsed = f"  [dim]{time.monotonic() - start:.1f}s[/dim]" if start is not None else ""
            desc_part = f"[ansi_cyan]{_preview(description, 72)}[/ansi_cyan]{elapsed}"
            if status == "success":
                self._append(Static(
                    f"[dim]└─[/dim] [bold ansi_green]✓[/bold ansi_green] {desc_part}",
                    classes="log-line",
                ))
            else:
                self._append(Static(
                    f"[dim]└─[/dim] [bold ansi_red]✗[/bold ansi_red] {desc_part}",
                    classes="log-line",
                ))

        elif t == "step.started":
            run_id = event.get("run_id", "")
            if run_id in self._subagent_run_ids:
                return
            step = event.get("step", "")
            self._append(Static(
                f"[dim]step {step}[/dim]",
                classes="step-divider",
            ))

        elif t == "tool.call_started":
            tool_use_id = str(event.get("tool_use_id", ""))
            tool_name = str(event.get("tool_name", ""))
            params = event.get("params") or {}
            run_id = event.get("run_id", "")
            tc_block = ToolCallBlock(tool_name, params)
            if run_id in self._subagent_run_ids:
                tc_block.styles.padding = (0, 2, 0, 6)
            self._pending_tool_blocks[tool_use_id] = tc_block
            self._append(tc_block)

        elif t == "tool.call_finished":
            tool_use_id = str(event.get("tool_use_id", ""))
            elapsed_ms = int(event.get("elapsed_ms") or 0)
            output = str(event.get("output") or "")
            if tool_use_id in self._pending_tool_blocks:
                tc_done = self._pending_tool_blocks.pop(tool_use_id)
                tc_done.set_result(output, elapsed_ms)

        elif t == "tool.call_failed":
            tool_use_id = str(event.get("tool_use_id", ""))
            elapsed_ms = int(event.get("elapsed_ms") or 0)
            error_msg = str(event.get("error_message") or "")
            if tool_use_id in self._pending_tool_blocks:
                tc_done = self._pending_tool_blocks.pop(tool_use_id)
                tc_done.set_result(error_msg, elapsed_ms, is_error=True)

        elif t == "run.finished":
            self._clear_run_permissions(str(event.get("run_id", "")))
            self._refresh_permission_input()
            status = event.get("status", "")
            steps = event.get("steps", 0)
            reason = escape(str(event.get("reason") or ""))
            if status in {"success", "succeeded"}:
                self._append(Static(
                    f"[bold ansi_green]✓ completed[/bold ansi_green]  [dim]{steps} steps[/dim]",
                    classes="run-ok",
                ))
            elif status == "cancelled":
                self._append(Static("[ansi_yellow]已停止[/ansi_yellow]", classes="log-line"))
            else:
                detail = f"  [dim]{reason}[/dim]" if reason else ""
                self._append(Static(
                    f"[bold ansi_red]✗ failed[/bold ansi_red]{detail}  [dim]{steps} steps[/dim]",
                    classes="run-err",
                ))
            if event.get("run_id") == self._active_run_id:
                self._restore_input()

        elif t == "llm.usage":
            run_id = event.get("run_id", "")
            if run_id in self._subagent_run_ids:
                return
            pct = float(event.get("context_pct") or 0.0)
            self._last_context_pct = pct
            ctx_bar = self._render_ctx_bar(pct)
            self._append(Static(
                f"[dim]  tokens  "
                f"in={event.get('input_tokens')} "
                f"out={event.get('output_tokens')} "
                f"cache={event.get('cache_read_input_tokens')}[/dim]"
                f"  {ctx_bar}",
                classes="usage",
            ))

        elif t == "context.compacted":
            orig = event.get("original_tokens", 0)
            summary = event.get("summary_tokens", 0)
            self._last_context_pct = 0.0
            self._append(Static(
                f"[bold ansi_cyan]⚡ Context compacted[/bold ansi_cyan]"
                f"  [dim]original≈{orig} tokens → summary={summary} tokens[/dim]",
                classes="log-line",
            ))

        elif t == "permission.requested":
            request_id = str(event.get("request_id", ""))
            if (request_id in self._resolved_permission_ids
                    or request_id in self._pending_permission_blocks):
                return
            tool_use_id = str(event.get("tool_use_id", ""))
            tool_name = str(event.get("tool_name", ""))
            param_preview = str(event.get("param_preview", ""))
            try:
                _focused_repr = repr(self.focused)
            except Exception:
                _focused_repr = "?"
            log.info(
                "permission.requested tool=%s id=%s  app.focused=%s",
                tool_name, request_id, _focused_repr,
            )
            request_kind = str(event.get("request_kind", "tool"))
            approval_details = ""
            if request_kind == "host_fallback":
                params = event.get("params", {})
                full_command = (
                    str(params.get("command", ""))
                    if tool_name == "bash" and isinstance(params, dict)
                    else json.dumps(params, ensure_ascii=False, sort_keys=True)
                )
                approval_details = (
                    f"  [bold]shell:[/bold] {escape(str(event.get('platform_shell', 'unknown')))}\n"
                    f"  [bold]command/parameters:[/bold] {escape(full_command)}\n"
                    f"  [bold ansi_red]WARNING:[/bold ansi_red] "
                    f"{escape(str(event.get('warning', 'Docker isolation is unavailable.')))}"
                )
            perm_block = PermissionBlock(
                tool_use_id,
                tool_name,
                param_preview,
                approval_details,
            )
            self._pending_permission_blocks[request_id] = perm_block
            self._permission_runs[request_id] = str(event.get("run_id", ""))
            prompt = self._prompt()
            if prompt is not None:
                prompt.disabled = True
                prompt.add_class("permission-waiting")
                prompt.border_title = self._PERMISSION_HINT
            self._append(perm_block)
            choices = None
            if request_kind == "host_fallback":
                choices = (
                    ("allow_host_once", "Run on host once", "y / 1"),
                    ("deny_once", "Deny", "n / 2"),
                )
            context = (
                f"[bold]待审批：{escape(tool_name)}[/bold] · request_kind={escape(request_kind)}\n"
                f"参数：{escape(_preview(param_preview, 160)) if param_preview else '（无摘要）'}"
            )
            if approval_details:
                context += "\n" + approval_details
            select = PermissionSelect(request_id, choices, context=context)
            self._permission_selects[request_id] = select
            self._mount_permission_select(select)
            log.debug(
                "PermissionSelect mounted before #prompt  pending=%d",
                len(self._pending_permission_blocks),
            )

        elif t in {"permission.denied", "permission.granted"}:
            request_id = str(event.get("request_id", ""))
            fallback = "deny_once" if t == "permission.denied" else "allow_once"
            self._resolve_permission(request_id, str(event.get("decision") or fallback))
            self._refresh_permission_input()

        elif t == "log.line":
            level = event.get("level", "INFO")
            color = ("bold ansi_red" if level == "ERROR"
                     else "ansi_yellow" if level == "WARNING" else "dim")
            self._append(Static(
                f"[{color}]{level}[/{color}]  "
                f"[dim]{event.get('source', '')}[/dim]  {event.get('message', '')}",
                classes="log-line",
            ))

        else:
            # Event Schema 1 permits new event types. Preserve observability
            # without assigning semantics that this client does not know.
            rendered = json.dumps(event, ensure_ascii=False, sort_keys=True)
            self._append(
                Static(
                    f"[dim]unknown event {escape(str(t or '?'))}[/dim]\n"
                    f"{escape(rendered)}",
                    classes="log-line",
                )
            )


# TUI 入口：读取配置并启动 TarsTuiApp
def run(config: TarsConfig, resume_session_id: str | None = None) -> None:
    app = TarsTuiApp(config.host, config.port, resume_session_id=resume_session_id)
    app.run()
