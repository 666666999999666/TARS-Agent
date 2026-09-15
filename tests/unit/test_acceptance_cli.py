from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.acceptance_cli import permitted_answer


@pytest.mark.parametrize(("tool", "params", "kind", "expected"), [
    ("write_file", {"path": "result.txt"}, "tool", "y"),
    ("write_file", {"path": "blocked.txt"}, "tool", "n"),
    ("write_file", {"path": "other.txt"}, "tool", "n"),
    ("read_file", {"path": "../outside.txt"}, "tool", "n"),
    ("write_file", {"path": "result.txt"}, "host_fallback", "n"),
    ("spawn_agent", {}, "tool", "n"),
    ("bash", {"command": "python safe.py", "timeout": 30}, "tool", "y"),
    ("bash", {"command": "python other.py", "timeout": 30}, "tool", "n"),
    ("bash", {"command": "python safe.py", "timeout": 31}, "tool", "n"),
])
def test_cli_acceptance_approves_only_this_fixture_parameters(
    tmp_path: Path, tool: str, params: dict, kind: str, expected: str,
) -> None:
    assert permitted_answer({"tool_name": tool, "params": params, "request_kind": kind},
                            tmp_path, {"result.txt"}, {"python safe.py"}, {"blocked.txt"}) == expected


@pytest.mark.parametrize(("arguments", "workflow"), [([], "all"), (["--workflow", "chat"], "chat")])
def test_cli_acceptance_plan_does_not_create_state_or_read_user_config(tmp_path: Path, arguments, workflow) -> None:
    project = Path(__file__).resolve().parents[2]
    output = tmp_path / "must-not-exist"
    env = {**os.environ, "TARS_HOME": str(tmp_path / "missing-home")}
    completed = subprocess.run([sys.executable, "-B", str(project / "scripts/acceptance_cli.py"),
                                "--output-root", str(output), *arguments], cwd=tmp_path, env=env,
                               capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["mode"] == "plan_only_no_requests"
    assert report["selected_workflow"] == workflow
    assert any(case.startswith("goal") for case in report["cases"]) == (workflow == "all")
    assert not output.exists()


def test_cli_acceptance_uses_selected_installed_interpreter(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from scripts import acceptance_cli

    selected = tmp_path / "installed" / "python.exe"
    selected.parent.mkdir()
    selected.touch()  # Command routing only; this fixture is never executed.
    entry = selected.with_name("tars.exe" if os.name == "nt" else "tars")
    entry.touch()
    observed = []
    monkeypatch.setattr(acceptance_cli, "CliPty", lambda command, *args: (
        observed.append(command) or SimpleNamespace(record={})
    ))
    harness = acceptance_cli.CliAcceptance(tmp_path / "evidence", selected)
    harness.start_cli("chat", ["chat"])
    assert observed == [[str(entry.resolve()), "chat"]]
    assert harness.core.report["client_pythonpath_inherited"] is False


@pytest.mark.parametrize("prompt", ["first\nsecond", "first\rsecond"])
async def test_chat_acceptance_rejects_multiline_before_any_terminal_input(tmp_path, prompt) -> None:
    from scripts.acceptance_cli import CliAcceptance

    class TerminalMustNotBeUsed:
        async def idle(self):
            raise AssertionError("Multiline input should be rejected before using the terminal")

    harness = CliAcceptance(tmp_path / "evidence", workflow="chat")
    with pytest.raises(ValueError, match="single input line"):
        await harness.chat_turn(TerminalMustNotBeUsed(), prompt, files=set())
