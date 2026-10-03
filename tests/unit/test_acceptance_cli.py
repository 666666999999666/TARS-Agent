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


def failure_result(tool_name, parameters, error_class, error_message, *, executed=True, answer="y"):
    return {
        "run": {"id": "run", "status": "succeeded"},
        "tools": [{"id": "run:tool", "tool_name": tool_name, "parameters": json.dumps(parameters),
                   "status": "failed", "error_class": error_class, "error_message": error_message,
                   "started_at": "started" if executed else None,
                   "backend": "workspace_sandbox" if executed else "host"}],
        "approval_decisions": [{"run_id": "run", "tool_use_id": "tool", "answer": answer}],
    }


@pytest.mark.parametrize("failure", [None, "permission_denied", "not_found", "runtime_error", "wrong_path", "not_started", "extra_tool"])
def test_path_escape_requires_the_actual_sandbox_boundary_failure(failure):
    from scripts.acceptance_cli import check_expected_failure

    result = failure_result("read_file", {"path": "../outside.txt"}, "sandbox_policy_denied",
                            "path escapes workspace: ../outside.txt")
    tool = result["tools"][0]
    if failure in {"permission_denied", "not_found", "runtime_error"}:
        tool["error_class"] = failure
    elif failure == "wrong_path":
        tool["parameters"] = json.dumps({"path": "missing.txt"})
    elif failure == "not_started":
        tool["started_at"] = None
    elif failure == "extra_tool":
        result["tools"].append(dict(tool))

    def validate():
        check_expected_failure(result, tool_name="read_file", parameters={"path": "../outside.txt"},
                               error_class="sandbox_policy_denied", executed=True,
                               error_prefix="path escapes workspace: ../outside.txt")

    if failure is None:
        validate()
    else:
        with pytest.raises(AssertionError):
            validate()


@pytest.mark.parametrize("failure", [None, "denied", "different_command", "different_exit", "different_approval", "model_failed", "host"])
def test_bash_failure_requires_the_approved_command_to_exit_seven(failure):
    from scripts.acceptance_cli import check_expected_failure

    command = "python -c 'raise SystemExit(7)'"
    result = failure_result("bash", {"command": command, "timeout": 30}, "runtime_error", "[exit 7]\n")
    tool = result["tools"][0]
    if failure == "denied":
        tool.update(error_class="permission_denied", started_at=None)
    elif failure == "different_command":
        tool["parameters"] = json.dumps({"command": "exit 1", "timeout": 30})
    elif failure == "different_exit":
        tool["error_message"] = "[exit 17]\n"
    elif failure == "different_approval":
        result["approval_decisions"][0]["tool_use_id"] = "another-tool"
    elif failure == "model_failed":
        result["run"]["status"] = "failed"
    elif failure == "host":
        tool["backend"] = "host"

    def validate():
        check_expected_failure(result, tool_name="bash", parameters={"command": command, "timeout": 30},
                               error_class="runtime_error", error_prefix="[exit 7]\n", executed=True, decision="y")

    if failure is None:
        validate()
    else:
        with pytest.raises(AssertionError):
            validate()


@pytest.mark.parametrize("failure", [None, "executed", "no_cli_answer", "wrong_path"])
def test_refusal_requires_cli_denial_and_no_tool_execution(failure):
    from scripts.acceptance_cli import check_expected_failure

    result = failure_result("write_file", {"path": "blocked.txt", "content": "blocked"},
                            "permission_denied", "Permission was not granted (denied or expired).",
                            executed=False, answer="n")
    if failure == "executed":
        result["tools"][0]["started_at"] = "started"
    elif failure == "no_cli_answer":
        result["approval_decisions"] = []
    elif failure == "wrong_path":
        result["tools"][0]["parameters"] = json.dumps({"path": "another.txt", "content": "blocked"})

    def validate():
        check_expected_failure(result, tool_name="write_file", parameters={"path": "blocked.txt", "content": "blocked"},
                               error_class="permission_denied", error_prefix="Permission was not granted",
                               executed=False, decision="n")

    if failure is None:
        validate()
    else:
        with pytest.raises(AssertionError):
            validate()


async def test_goal_does_not_record_pass_before_semantic_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.acceptance_cli import CliAcceptance, check_expected_failure

    harness = CliAcceptance(tmp_path / "evidence")

    async def nothing(*_args, **_kwargs):
        return None

    async def wrong_failure(*_args, **_kwargs):
        return failure_result("read_file", {"path": "../outside.txt"}, "not_found", "missing")

    monkeypatch.setattr(harness, "ping", nothing)
    monkeypatch.setattr(harness, "run_ids", lambda: set())
    monkeypatch.setattr(harness, "start_cli", lambda *_args: SimpleNamespace(finish=nothing))
    monkeypatch.setattr(harness, "drive_run", wrong_failure)
    monkeypatch.setattr(harness, "check_source", lambda *_args: None)
    result = await harness.goal("goal-path-escape", "fixed prompt", 1, files=set())
    with pytest.raises(AssertionError):
        check_expected_failure(result, tool_name="read_file", parameters={"path": "../outside.txt"},
                               error_class="sandbox_policy_denied", error_prefix="path escapes workspace: ../outside.txt",
                               executed=True)
    persisted = json.loads((harness.root / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["cases"] == [{"name": "goal-path-escape", "status": "pending_validation",
                                    "run_id": "run", "exit_code": 1}]


@pytest.mark.parametrize("failure", [None, "failed", "memory_read", "no_write"])
def test_restored_chat_requires_successful_memory_only_write(failure):
    from scripts.acceptance_cli import check_file_turn

    result = {"run": {"status": "succeeded"}, "tools": [{
        "tool_name": "write_file", "parameters": {"path": "chat-restored.txt"},
        "status": "succeeded", "started_at": "started", "backend": "workspace_sandbox",
    }]}
    if failure == "failed":
        result["run"]["status"] = "failed"
    elif failure == "memory_read":
        result["tools"].append({**result["tools"][0], "tool_name": "read_file"})
    elif failure == "no_write":
        result["tools"] = []
    if failure is None:
        check_file_turn(result, "chat-restored.txt", read_source=False)
    else:
        with pytest.raises(AssertionError):
            check_file_turn(result, "chat-restored.txt", read_source=False)


@pytest.mark.parametrize("failure", [None, "not_allowed", "wrong_backend", "wrong_content", "failed"])
def test_file_turn_checks_in_process_notes_separately_from_docker_files(failure):
    from scripts.acceptance_cli import check_file_turn

    result = {"run": {"status": "succeeded"}, "tools": [
        {"tool_name": "write_file", "parameters": {"path": "restored.txt"},
         "status": "succeeded", "started_at": "started", "backend": "workspace_sandbox"},
        {"tool_name": "note_save", "parameters": {"content": "remember MARKER"},
         "status": "succeeded", "started_at": "started", "backend": "in_process"},
    ]}
    marker = None if failure == "not_allowed" else "MARKER"
    if failure == "wrong_backend":
        result["tools"][1]["backend"] = "host"
    elif failure == "wrong_content":
        result["tools"][1]["parameters"] = {"content": "unrelated"}
    elif failure == "failed":
        result["tools"][1]["status"] = "failed"
    if failure is None:
        check_file_turn(result, "restored.txt", read_source=False, note_marker=marker)
    else:
        with pytest.raises(AssertionError):
            check_file_turn(result, "restored.txt", read_source=False, note_marker=marker)


@pytest.mark.parametrize("failure", [None, "no_cancel", "not_started", "wrong_command"])
def test_cancel_requires_the_requested_started_tool_and_sent_interrupt(failure):
    from scripts.acceptance_cli import check_cancelled_turn

    result = {"run": {"status": "cancelled"}, "cancel_sent": True, "tools": [{
        "tool_name": "bash", "parameters": {"command": "fixed probe"},
        "status": "cancelled", "started_at": "started", "backend": "workspace_sandbox",
    }]}
    if failure == "no_cancel":
        result["cancel_sent"] = False
    elif failure == "not_started":
        result["tools"][0]["started_at"] = None
    elif failure == "wrong_command":
        result["tools"][0]["parameters"]["command"] = "another probe"
    if failure is None:
        check_cancelled_turn(result, "fixed probe")
    else:
        with pytest.raises(AssertionError):
            check_cancelled_turn(result, "fixed probe")


def workspace_guard_result(workspace):
    result = failure_result("read_file", {"path": "../outside.txt"}, "permission_denied",
                            "Permission was not granted (denied or expired).", executed=False)
    result["run"]["session_id"] = "session"
    result.update(session={"id": "session", "workspace_root": str(workspace)}, core_launch_id="core-one",
                  permission_requests=[], approval_decisions=[])
    refused = result["tools"][0]
    refused.update(run_id="run", created_at="2026-09-29 10:30:03")
    result["tools"].insert(0, {
        "id": "run:control", "run_id": "run", "tool_name": "read_file",
        "parameters": json.dumps({"path": "源数据.txt"}), "status": "succeeded",
        "started_at": "2026-09-29 10:30:01", "finished_at": "2026-09-29 10:30:02",
        "created_at": "2026-09-29 10:30:00", "backend": "workspace_sandbox",
        "result": json.dumps({"content": "1: local-control-marker"}),
    })
    return result


@pytest.mark.parametrize("failure", [
    None, "wrong_workspace", "wrong_session", "wrong_run", "missing_core", "no_control_read",
    "control_not_started", "control_host", "wrong_marker", "control_after_refusal", "wrong_path",
    "wrong_error", "outside_started", "approval_prompt", "cli_denial",
])
def test_cli_workspace_guard_requires_automatic_refusal_after_same_run_control_read(tmp_path, failure):
    from scripts.acceptance_cli import check_permission_workspace_guard

    result = workspace_guard_result(tmp_path)
    normal, refused = result["tools"]
    if failure == "wrong_workspace":
        result["session"]["workspace_root"] = str(tmp_path / "other")
    elif failure == "wrong_session":
        result["session"]["id"] = "other-session"
    elif failure == "wrong_run":
        normal["run_id"] = "earlier-run"
    elif failure == "missing_core":
        result["core_launch_id"] = ""
    elif failure == "no_control_read":
        result["tools"] = [refused]
    elif failure == "control_not_started":
        normal["started_at"] = None
    elif failure == "control_host":
        normal["backend"] = "host"
    elif failure == "wrong_marker":
        normal["result"] = json.dumps({"content": "another fixture"})
    elif failure == "control_after_refusal":
        normal["finished_at"] = "2026-09-29 10:30:04"
    elif failure == "wrong_path":
        refused["parameters"] = json.dumps({"path": "../another.txt"})
    elif failure == "wrong_error":
        refused["error_class"] = "not_found"
    elif failure == "outside_started":
        refused["started_at"] = "2026-09-29 10:30:03"
    elif failure == "approval_prompt":
        result["permission_requests"] = [{"request_id": "prompted"}]
    elif failure == "cli_denial":
        result["approval_decisions"] = [{"answer": "n"}]

    if failure is None:
        check_permission_workspace_guard(result, tmp_path, "local-control-marker")
    else:
        with pytest.raises(AssertionError):
            check_permission_workspace_guard(result, tmp_path, "local-control-marker")


@pytest.mark.parametrize("failure", [None, "changed", "result_content", "terminal_content", "result_digest", "terminal_digest"])
def test_cli_outside_sentinel_must_remain_unchanged_and_undisclosed(tmp_path, failure):
    import hashlib

    from scripts.acceptance_cli import check_outside_sentinel

    original = b"outside-sentinel-fixed-synthetic-value"
    digest = hashlib.sha256(original).hexdigest()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(original)
    result, terminal = {}, ""
    if failure == "changed":
        outside.write_bytes(b"changed")
    elif failure == "result_content":
        result = {"run": {"result": original.decode()}}
    elif failure == "terminal_content":
        terminal = original.decode()
    elif failure == "result_digest":
        result = {"run": {"result": digest}}
    elif failure == "terminal_digest":
        terminal = digest
    if failure is None:
        check_outside_sentinel(outside, original, result, terminal)
    else:
        with pytest.raises(AssertionError):
            check_outside_sentinel(outside, original, result, terminal)


async def test_drive_run_preserves_empty_approval_evidence_and_bound_session(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import acceptance_cli

    harness = acceptance_cli.CliAcceptance(tmp_path / "evidence")
    expected = workspace_guard_result(harness.workspace)
    harness.core.control = SimpleNamespace(launch_id="core-one")

    def rows(query, _params=()):
        if query.startswith("SELECT * FROM runs"):
            return [expected["run"]]
        if "tool_invocations" in query:
            return expected["tools"]
        if "FROM sessions" in query:
            return [expected["session"]]
        if "event_type='permission.requested'" in query:
            return []
        if "llm.usage" in query:
            return [{"event_type": "llm.usage", "payload": "{}"}]
        raise AssertionError(query)

    async def no_wait(*_args):
        return None

    monkeypatch.setattr(harness, "rows", rows)
    monkeypatch.setattr(acceptance_cli.asyncio, "sleep", no_wait)
    terminal = SimpleNamespace(pump=lambda: None)
    result = await harness.drive_run(terminal, set(), files={"源数据.txt"})
    acceptance_cli.check_permission_workspace_guard(result, harness.workspace, "local-control-marker")
    assert result["approval_decisions"] == [] and result["permission_requests"] == []
    recorded = json.loads((harness.root / "runs" / "run.json").read_text(encoding="utf-8"))
    assert recorded["session"] == expected["session"] and recorded["core_launch_id"] == "core-one"
