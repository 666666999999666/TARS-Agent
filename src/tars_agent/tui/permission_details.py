from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea


class PermissionDetailsScreen(ModalScreen[None]):
    """Read-only parameters for one pending approval; closing never approves it."""

    BINDINGS = [
        Binding("escape", "close", "返回审批", priority=True),
        Binding("ctrl+home", "first", "参数开头", priority=True),
        Binding("ctrl+end", "last", "参数末尾", priority=True),
    ]
    DEFAULT_CSS = """
    PermissionDetailsScreen { align: center middle; background: $background; }
    PermissionDetailsScreen > Vertical {
        width: 95%; height: 90%; background: $surface;
        color: $text; border: round $foreground; padding: 0 1;
    }
    PermissionDetailsScreen .details-title { height: auto; color: $text; }
    PermissionDetailsScreen TextArea { height: 1fr; border: solid $foreground; }
    PermissionDetailsScreen Button { height: 3; margin-top: 0; }
    """

    def __init__(self, request_id: str, tool_name: str, params_json: str) -> None:
        super().__init__()
        self.request_id = request_id
        self._tool_name = tool_name
        self._params_json = params_json
        self._valid = True

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                f"完整参数：{self._tool_name}\n审批：{self.request_id}\n"
                "PgUp/PgDn 滚动 · Ctrl+Home/End 首尾",
                classes="details-title", markup=False,
            )
            yield TextArea(
                self._params_json, read_only=True, soft_wrap=True,
                show_line_numbers=False, id="approval-parameters",
            )
            yield Button("返回审批 (Esc)", id="close-approval-details")

    def on_mount(self) -> None:
        if not self._valid:
            self.dismiss(None)
            return
        self.query_one(TextArea).focus()

    def invalidate(self) -> None:
        if not self._valid:
            return
        self._valid = False
        if self.is_mounted:
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)

    def action_first(self) -> None:
        self.query_one(TextArea).move_cursor((0, 0))

    def action_last(self) -> None:
        editor = self.query_one(TextArea)
        editor.move_cursor(editor.document.end)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-approval-details":
            event.stop()
            self.action_close()
