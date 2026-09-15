from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock

from tars_agent.web import cli


def test_open_browser_waits_until_listener_is_ready(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connect = MagicMock(side_effect=[OSError("not ready"), connection])
    opened = MagicMock()
    monkeypatch.setattr(cli.socket, "create_connection", connect)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli.webbrowser, "open", opened)

    cli._open_when_listening("http://127.0.0.1:7438/#bootstrap=secret", "127.0.0.1", 7438)

    assert connect.call_count == 2
    opened.assert_called_once()


def test_main_drops_raw_bootstrap_references_before_server_loop(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    raw_token = "raw-bootstrap-secret"
    auth = MagicMock()
    observed_locals: dict[str, object] = {}

    monkeypatch.setattr(sys, "argv", ["tars-web"])
    monkeypatch.setattr(
        cli,
        "get_config",
        lambda: MagicMock(host="127.0.0.1", port=7437),
    )
    monkeypatch.setattr(cli.LocalWebAuth, "issue", lambda: (auth, raw_token))
    monkeypatch.setattr(cli, "create_app", lambda *_args, **_kwargs: MagicMock())

    def inspect_server_caller(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        observed_locals.update(frame.f_back.f_locals)

    monkeypatch.setattr(cli.uvicorn, "run", inspect_server_caller)

    cli.main()

    assert "bootstrap_token" not in observed_locals
    assert "url" not in observed_locals
    assert raw_token not in observed_locals.values()
