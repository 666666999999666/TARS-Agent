from __future__ import annotations

import importlib
import sys

import pytest

from tars_agent.core.config import TarsConfig

cli = importlib.import_module("tars_agent.cli.main")


@pytest.mark.parametrize("arguments", [
    ["run"], ["run", "--goal", "   "], ["run", "--goal", "goal", "status", "run-1"],
])
def test_invalid_task_arguments_fail_before_loading_configuration(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["tars", *arguments])

    def unexpected_config() -> TarsConfig:
        raise AssertionError("Invalid arguments must fail before initializing user state")

    monkeypatch.setattr(cli, "get_config", unexpected_config)
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2


@pytest.mark.parametrize("resume", [None, "sess-existing"])
def test_chat_dispatches_with_optional_resume(resume, monkeypatch: pytest.MonkeyPatch) -> None:
    config = TarsConfig()
    calls = []
    args = ["tars", "chat"] + (["--resume", resume] if resume else [])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(cli, "get_config", lambda: config)
    monkeypatch.setattr(cli, "setup_logging", lambda _: None)
    monkeypatch.setattr(cli, "cmd_chat", lambda cfg, **kw: calls.append((cfg, kw)))
    cli.main()
    assert calls == [(config, {"resume_session_id": resume})]


def test_goal_dispatches_exact_task_text(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TarsConfig()
    calls = []
    monkeypatch.setattr(sys, "argv", ["tars", "run", "--goal", "读取 中文文件.txt"])
    monkeypatch.setattr(cli, "get_config", lambda: config)
    monkeypatch.setattr(cli, "setup_logging", lambda _: None)
    monkeypatch.setattr(cli, "cmd_run", lambda goal, cfg: calls.append((goal, cfg)))
    cli.main()
    assert calls == [("读取 中文文件.txt", config)]


@pytest.mark.parametrize("action", ["status", "cancel", "metrics"])
def test_existing_run_management_commands_dispatch(
    action: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TarsConfig()
    observed: list[tuple[TarsConfig, str]] = []
    monkeypatch.setattr(sys, "argv", ["tars", "run", action, "run-1"])
    monkeypatch.setattr(cli, "get_config", lambda: config)
    monkeypatch.setattr(cli, "setup_logging", lambda _config: None)
    monkeypatch.setattr(cli, f"cmd_run_{action}", lambda cfg, run_id: observed.append((cfg, run_id)))

    cli.main()

    assert observed == [(config, "run-1")]
