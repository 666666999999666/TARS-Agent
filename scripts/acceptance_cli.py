"""Isolated real CLI acceptance; no request is sent without --execute."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT))

from scripts.acceptance_workflows import (  # noqa: E402
    TERMINAL,
    Workflows,
    counts,
    database,
    fixed_configuration,
    json_file,
    now,
)

from tars_agent.core.eval.provenance import compute_tree_digest  # noqa: E402

DEFAULT_OUTPUT = PROJECT / "build" / "v1-cli" / "real-cli"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")


class CliPty:
    def __init__(self, command: list[str], workspace: Path, home: Path, port: int, output: Path):
        from winpty import PTY, Backend

        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        home.mkdir(parents=True, exist_ok=True)
        client_config = output / "client-config.toml"
        client_config.write_text(f"[core]\nhost='127.0.0.1'\nport={port}\n", encoding="utf-8")
        allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE",
                   "APPDATA", "LOCALAPPDATA", "COMSPEC"}
        env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        env.update(TARS_HOME=str(home), TARS_CONFIG=str(client_config), TARS_PORT=str(port),
                   TARS_LOG_FILE=str(output / "client.log"), PYTHONUTF8="1",
                   PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", TERM="xterm-256color")
        self.pty = PTY(160, 42, backend=Backend.ConPTY)
        arguments = " " + subprocess.list2cmdline(command[1:])
        if not self.pty.spawn(command[0], cmdline=arguments, cwd=str(workspace),
                              env="\0".join(f"{key}={value}" for key, value in env.items()) + "\0"):
            raise RuntimeError("CLI ConPTY did not start")
        self.text = ""
        self.inputs: list[dict[str, Any]] = []
        self.record = {"command": command, "workspace": str(workspace), "pid": self.pty.pid,
                       "streams": "stdout and stderr combined by Windows ConPTY",
                       "started_at": now(), "exit_code": None}
        json_file(output / "process.json", self.record)

    def pump(self) -> None:
        try:
            data = self.pty.read(blocking=False)
        except EOFError:
            data = ""
        if data:
            self.text += data
            with (self.output / "terminal.ansi.log").open("a", encoding="utf-8") as stream:
                stream.write(data)

    def send(self, value: str, reason: str) -> None:
        if not self.pty.isalive():
            raise RuntimeError("CLI exited before the requested input")
        self.inputs.append({"at": now(), "text": value, "reason": reason})
        json_file(self.output / "inputs.json", self.inputs)
        self.pty.write(value)

    def approval_ready(self, request_id: str) -> bool:
        plain = ANSI.sub("", self.text)
        position = plain.find(f"[approval: {request_id}]")
        tail = plain[position:] if position >= 0 else ""
        return "y=allow" in tail and "n=deny" in tail and " > " in tail

    async def idle(self, timeout: float = 15) -> None:
        deadline = time.monotonic() + timeout
        while self.pty.isalive() and time.monotonic() < deadline:
            self.pump()
            if ANSI.sub("", self.text).rstrip("\r\n").endswith("> "):
                return
            await asyncio.sleep(0.05)
        raise TimeoutError("CLI chat did not show its idle input prompt")

    async def finish(self, expected: int, timeout: float = 25) -> int:
        deadline = time.monotonic() + timeout
        while self.pty.isalive() and time.monotonic() < deadline:
            self.pump()
            await asyncio.sleep(0.05)
        self.pump()
        if self.pty.isalive():
            raise TimeoutError("CLI did not exit")
        code = self.pty.get_exitstatus()
        self.record.update(exit_code=code, finished_at=now())
        json_file(self.output / "process.json", self.record)
        if code != expected:
            raise AssertionError(f"CLI exit code {code}; expected {expected}")
        return int(code)

    async def cleanup(self) -> None:
        if self.record["exit_code"] is not None:
            # finish() already observed process exit. A released ConPTY handle can
            # reject later queries; that is not a still-running CLI process.
            try:
                self.pty.cancel_io()
            except Exception as exc:
                self.record["handle_release_note"] = type(exc).__name__
            self.record.update(cleaned=True, cleanup_method="process exit already observed")
            json_file(self.output / "process.json", self.record)
            return
        if self.pty.isalive():
            self.send("\x03", "cleanup: interrupt only this CLI")
            await asyncio.sleep(0.2)
        if self.pty.isalive():
            self.send("\x1a\r", "cleanup: Windows EOF")
            for _ in range(100):
                self.pump()
                if not self.pty.isalive():
                    break
                await asyncio.sleep(0.05)
        if self.pty.isalive():
            taskkill = Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"
            await asyncio.to_thread(subprocess.run, [str(taskkill), "/PID", str(self.pty.pid), "/T", "/F"],
                                    capture_output=True, timeout=10, check=False)
        self.record.update(cleaned=not self.pty.isalive(), finished_at=now())
        if not self.pty.isalive():
            self.record["exit_code"] = self.pty.get_exitstatus()
        self.pty.cancel_io()
        json_file(self.output / "process.json", self.record)
        if not self.record["cleaned"]:
            raise RuntimeError("Owned CLI process cleanup is unconfirmed")


def permitted_answer(event: dict[str, Any], workspace: Path, allowed_files: set[str],
                     allowed_commands: set[str], deny_files: set[str]) -> str:
    if event.get("request_kind", "tool") != "tool":
        return "n"
    params = event.get("params", {})
    tool = event.get("tool_name")
    if tool == "bash":
        timeout = params.get("timeout", 30)
        return "y" if (params.get("command") in allowed_commands and type(timeout) in {int, float}
                       and 0 < timeout <= 30) else "n"
    if tool not in {"read_file", "write_file", "list_dir"}:
        return "n"
    raw = Path(str(params.get("path", ".")))
    try:
        relative = (raw if raw.is_absolute() else workspace / raw).resolve().relative_to(workspace.resolve()).as_posix()
    except (ValueError, OSError):
        return "n"
    if relative in deny_files:
        return "n"
    return "y" if relative in allowed_files else "n"


def tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    parameters = tool["parameters"]
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    assert isinstance(parameters, dict), "tool parameters were not recorded as an object"
    return parameters


def check_expected_failure(
    result: dict[str, Any], *, tool_name: str, parameters: dict[str, Any],
    error_class: str, error_prefix: str, executed: bool, decision: str | None = None,
) -> None:
    assert result["run"]["status"] == "succeeded", "model did not finish handling the expected tool failure"
    assert len(result["tools"]) == 1, "expected exactly one attempt of the specified tool"
    tool = result["tools"][0]
    assert tool["tool_name"] == tool_name
    actual = tool_parameters(tool)
    assert all(actual.get(key) == value for key, value in parameters.items()), "different tool parameters were attempted"
    assert tool["status"] == "failed" and tool["error_class"] == error_class, "the tool failed for a different reason"
    assert str(tool.get("error_message", "")).startswith(error_prefix), "expected tool failure evidence was not recorded"
    assert bool(tool.get("started_at")) == executed, "tool execution did not match the expected scenario"
    if executed:
        assert tool["backend"] == "workspace_sandbox", "tool did not execute in Docker"
    if decision is not None:
        assert any(
            item["answer"] == decision
            and item["run_id"] + ":" + item["tool_use_id"] == tool["id"]
            for item in result["approval_decisions"]
        ), "the specified tool did not receive the expected CLI approval answer"


def check_file_turn(
    result: dict[str, Any], output: str, *, read_source: bool, note_marker: str | None = None,
) -> None:
    assert result["run"]["status"] == "succeeded", "file task did not succeed"
    notes = [tool for tool in result["tools"] if tool["tool_name"] == "note_save"]
    for note in notes:
        assert note_marker is not None, "this file task did not request session memory"
        assert note["status"] == "succeeded" and note.get("started_at")
        assert note["backend"] == "in_process", "note_save used an unexpected backend"
        content = tool_parameters(note).get("content")
        assert isinstance(content, str) and note_marker in content, "saved note lost the source marker"
    tools = [tool for tool in result["tools"] if tool["tool_name"] != "note_save"]
    assert tools and all(tool["status"] == "succeeded" and tool.get("started_at")
                         and tool["backend"] == "workspace_sandbox" for tool in tools), "file task did not complete all tools in Docker"
    calls = [(tool["tool_name"], tool_parameters(tool).get("path")) for tool in tools]
    assert ("write_file", output) in calls, "expected write_file did not execute"
    if read_source:
        assert ("read_file", "源数据.txt") in calls and ("read_file", output) in calls, "source/read-back tools did not execute"
    else:
        assert all(name == "write_file" and path == output for name, path in calls), "memory task fetched its answer from a tool"


def check_cancelled_turn(result: dict[str, Any], command: str) -> None:
    assert result["run"]["status"] == "cancelled" and result["cancel_sent"], "cancellation was not sent after the start marker"
    assert len(result["tools"]) == 1, "cancellation scenario executed extra tools"
    tool = result["tools"][0]
    assert tool["tool_name"] == "bash" and tool_parameters(tool).get("command") == command
    assert tool.get("started_at") and tool["backend"] == "workspace_sandbox", "cancellation did not interrupt a started Docker tool"
    assert tool["status"] == "cancelled", "started tool was not recorded as cancelled"


def check_permission_workspace_guard(result: dict[str, Any], workspace: Path, marker: str) -> None:
    """Verify the production pre-execution path guard, separately from Docker's worker guard."""
    run = result["run"]
    assert run["status"] == "succeeded", "model did not finish after the boundary refusal"
    assert result["core_launch_id"], "Core identity was not recorded"
    assert result["session"]["id"] == run["session_id"]
    assert Path(result["session"]["workspace_root"]).resolve() == workspace.resolve(), "the Run used a different workspace"
    tools = result["tools"]
    assert len(tools) == 2, "expected a normal read followed by exactly one boundary read"
    normal, refused = tools
    assert all(tool["run_id"] == run["id"] and tool["tool_name"] == "read_file" for tool in tools)
    assert tool_parameters(normal).get("path") == "源数据.txt"
    assert normal["status"] == "succeeded" and normal.get("started_at")
    assert normal["backend"] == "workspace_sandbox", "the control read did not execute in Docker"
    normal_output = normal["result"]
    if isinstance(normal_output, str):
        normal_output = json.loads(normal_output)
    assert marker in normal_output["content"], "the control read did not return this fixture's marker"
    assert normal["finished_at"] <= refused["created_at"], "the control read did not finish before the boundary attempt"
    assert tool_parameters(refused) == {"path": "../outside.txt"}, "a different boundary request was attempted"
    assert refused["status"] == "failed" and refused["error_class"] == "permission_denied"
    assert str(refused["error_message"]).startswith("Permission was not granted"), "unexpected failure evidence"
    assert refused["started_at"] is None, "the entry guard allowed the outside read to start"
    assert not result["permission_requests"], "a prompted approval refusal is not the automatic workspace guard"
    assert not result["approval_decisions"], "CLI approval answers cannot prove an automatic workspace refusal"


def check_outside_sentinel(outside: Path, original: bytes, result: dict[str, Any], terminal: str) -> None:
    digest = hashlib.sha256(original).hexdigest()
    observed = outside.read_bytes()
    assert observed == original and hashlib.sha256(observed).hexdigest() == digest, "outside sentinel changed"
    visible = json.dumps(result, ensure_ascii=False) + terminal
    assert original.decode("utf-8") not in visible and digest not in visible, "the outside sentinel was disclosed"


class CliAcceptance:
    def __init__(self, output_root: Path, client_python: Path | None = None, workflow: str = "all"):
        self.workflow = workflow
        self.client_python = str((client_python or Path(sys.executable)).expanduser().resolve())
        if not Path(self.client_python).is_file():
            raise ValueError("Client Python must be an existing interpreter")
        if client_python is not None:
            entry = Path(self.client_python).with_name("tars.exe" if os.name == "nt" else "tars")
            if not entry.is_file():
                raise ValueError("The selected installed environment must contain its tars console entry")
            self.cli_prefix = [str(entry)]
        else:
            self.cli_prefix = [self.client_python, "-B", "-m", "tars_agent.cli"]
        self.core = Workflows(False, None, output_root=output_root)
        self.root = self.core.directory
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.marker = "V1-CLI-" + secrets.token_hex(16)
        (self.workspace / "源数据.txt").write_text(self.marker + "\n", encoding="utf-8")
        self.ptys: list[CliPty] = []
        self.answered: set[str] = set()
        self.core.report.update(kind="real_cli_acceptance", marker=self.marker,
                                workspace=str(self.workspace), client_python=self.client_python,
                                selected_workflow=workflow,
                                cli_prefix=self.cli_prefix,
                                client_pythonpath_inherited=False, cli_processes=[],
                                ui_boundary="actual python -m tars_agent.cli through Windows ConPTY",
                                harness_control="only owned file/command approval through CLI y/n; production Provider unchanged")
        self.core.report.pop("all_nine_complete", None)

    def rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with database(self.core.home / "state.db") as connection:
            return [dict(row) for row in connection.execute(sql, params)]

    def run_ids(self) -> set[str]:
        return {row["id"] for row in self.rows("SELECT id FROM runs WHERE parent_run_id IS NULL")}

    def execution_ids(self) -> tuple[set[str], set[str]]:
        return ({row["id"] for row in self.rows("SELECT id FROM runs")},
                {row["id"] for row in self.rows("SELECT id FROM tool_invocations")})

    def start_cli(self, name: str, arguments: list[str]) -> CliPty:
        command = [*self.cli_prefix, *arguments]
        pty = CliPty(command, self.workspace, self.root / "client-home", self.core.port,
                     self.root / "cli" / name)
        self.ptys.append(pty)
        self.core.report["cli_processes"].append(pty.record)
        self.core.checkpoint()
        return pty

    async def ping(self, label: str) -> None:
        pong = await self.core.rpc("core.ping", {"client": "acceptance-cli"})
        assert self.core.control is not None and pong["launch_id"] == self.core.control.launch_id, "Core identity changed during CLI acceptance"
        self.core.report.setdefault("pings", []).append({"label": label, "at": now(),
                                                          "launch_id": pong["launch_id"],
                                                          "server_version": pong["server_version"]})

    def check_source(self, label: str) -> None:
        runtime_hash = compute_tree_digest(PROJECT / "src")
        script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.core.report.setdefault("source_checks", []).append({
            "label": label, "runtime_source_sha256": runtime_hash, "script_sha256": script_hash,
        })
        assert runtime_hash == self.core.report["runtime_source_sha256"], "Runtime source changed during real CLI acceptance"
        assert script_hash == self.core.report["script_sha256"], "Acceptance script changed during execution"

    async def drive_run(self, pty: CliPty, before: set[str], *, files: set[str],
                        commands: set[str] | None = None, denied: set[str] | None = None,
                        cancel_marker: str | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + 180
        run = None
        cancelled = False
        while time.monotonic() < deadline:
            pty.pump()
            candidates = [row for row in self.rows("SELECT * FROM runs WHERE parent_run_id IS NULL ORDER BY created_at")
                          if row["id"] not in before]
            if len(candidates) > 1:
                raise AssertionError("CLI submitted more than one Run for one input")
            if candidates:
                run = candidates[0]
                events = self.rows("SELECT payload FROM events WHERE run_id=? AND event_type='permission.requested' ORDER BY cursor",
                                   (run["id"],))
                pending = [json.loads(row["payload"]) for row in events
                           if json.loads(row["payload"])["request_id"] not in self.answered]
                if pending and pty.approval_ready(pending[0]["request_id"]):
                    event = pending[0]
                    answer = permitted_answer(event, self.workspace, files, commands or set(), denied or set())
                    self.answered.add(event["request_id"])
                    self.core.decisions.append({"request_id": event["request_id"], "run_id": run["id"],
                                                "tool_use_id": event["tool_use_id"],
                                                "tool_name": event["tool_name"], "answer": answer, "at": now()})
                    pty.send(answer + "\r", "parameter-scoped tool approval: " + event["request_id"])
                if cancel_marker and (self.workspace / cancel_marker).is_file() and not cancelled:
                    pty.send("\x03", "cancel after actual tool start marker")
                    cancelled = True
                if run["status"] in TERMINAL:
                    tools = self.rows("SELECT * FROM tool_invocations WHERE run_id=? ORDER BY created_at,id", (run["id"],))
                    unfinished = [tool["id"] for tool in tools if tool["status"] in {"queued", "running"}]
                    sessions = self.rows("SELECT id,workspace_root FROM sessions WHERE id=?", (run["session_id"],))
                    assert len(sessions) == 1 and self.core.control is not None
                    result = {"run": run, "tools": tools, "cancel_sent": cancelled,
                              "session": sessions[0], "core_launch_id": self.core.control.launch_id,
                              "permission_requests": [json.loads(row["payload"]) for row in events],
                              "approval_decisions": [item for item in self.core.decisions if item["run_id"] == run["id"]],
                              "terminal_tool_check": {"passed": not unfinished, "run_id": run["id"],
                                                      "unfinished_tool_ids": unfinished}}
                    self.core.report.setdefault("submissions", []).append({
                        "session_id": run["session_id"], "run_id": run["id"], "via": "actual CLI",
                    })
                    result["model_events"] = self.rows(
                        "SELECT event_type,payload FROM events WHERE run_id=? AND event_type IN ('llm.model_selected','llm.usage') ORDER BY cursor",
                        (run["id"],),
                    )
                    json_file(self.root / "runs" / (run["id"] + ".json"), result)
                    assert not unfinished, "terminal Run has unfinished tools"
                    assert any(event["event_type"] == "llm.usage" for event in result["model_events"]), "real model usage was not observed"
                    await asyncio.sleep(0.2)
                    pty.pump()
                    return result
            if not pty.pty.isalive():
                raise RuntimeError("CLI exited before an owned terminal Run was observed")
            await asyncio.sleep(0.05)
        raise TimeoutError("Real CLI task exceeded 180 seconds")

    def check_file(self, name: str) -> None:
        data = (self.workspace / name).read_text(encoding="utf-8")
        assert self.marker in data, f"actual output {name} does not contain the source marker"

    def cancel_command(self, prefix: str) -> str:
        source = (f"from pathlib import Path; import time; Path('{prefix}-started.txt').write_text('started'); "
                  f"time.sleep(20); Path('{prefix}-forbidden.txt').write_text('must-not-exist')")
        return "python -c " + shlex.quote(source)

    async def goal(self, name: str, prompt: str, exit_code: int, *, files: set[str],
                   commands: set[str] | None = None, denied: set[str] | None = None,
                   cancel_marker: str | None = None) -> dict[str, Any]:
        case = {"name": name, "status": "running"}
        self.core.report["cases"].append(case)
        self.core.checkpoint()
        await self.ping(name + "-before")
        before = self.run_ids()
        pty = self.start_cli(name, ["run", "--goal", prompt])
        result = await self.drive_run(pty, before, files=files, commands=commands,
                                      denied=denied, cancel_marker=cancel_marker)
        await pty.finish(exit_code)
        await self.ping(name + "-after-cli-exit")
        self.check_source(name)
        case.update(status="pending_validation", run_id=result["run"]["id"], exit_code=exit_code)
        self.core.checkpoint()
        return result

    def pass_goal(self, name: str) -> None:
        case = next(case for case in self.core.report["cases"] if case["name"] == name)
        assert case["status"] == "pending_validation"
        case["status"] = "passed"
        self.core.checkpoint()

    async def chat_turn(self, pty: CliPty, prompt: str, **policy: Any) -> dict[str, Any]:
        if "\n" in prompt or "\r" in prompt:
            raise ValueError("CLI chat acceptance requires a single input line")
        await pty.idle()
        before = self.run_ids()
        pty.send(prompt + "\r", "new synthetic user task")
        result = await self.drive_run(pty, before, **policy)
        assert pty.pty.isalive(), "chat did not remain available for another turn"
        await pty.idle()
        self.check_source(result["run"]["id"])
        return result

    async def execute(self) -> int:
        report = self.core.report
        try:
            config, _home, ledger = fixed_configuration()
            allowed_env = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE",
                           "APPDATA", "LOCALAPPDATA", "COMSPEC"}
            identity_env = {key: value for key, value in os.environ.items() if key.upper() in allowed_env}
            identity_env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            identity = subprocess.check_output([
                self.client_python, "-B", "-c",
                "import json,sys,tars_agent; print(json.dumps({'python':sys.executable,'package':tars_agent.__file__}))",
            ], cwd=self.workspace, env=identity_env, text=True, encoding="utf-8", timeout=15)
            report["client_identity"] = json.loads(identity)
            report.update(model=config.llm.default_model, budget_before=counts(ledger),
                          source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                          source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip()),
                          source_tree_sha256=compute_tree_digest(PROJECT),
                          runtime_source_sha256=compute_tree_digest(PROJECT / "src"),
                          script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            await self.core.start(guarded=False)
            if self.workflow == "all":
                success = await self.goal("goal-success", "只用read_file读取源数据.txt，把其中完整标记write_file写入goal-result.txt，再read_file读回核对，结束。", 0,
                                         files={"源数据.txt", "goal-result.txt"})
                check_file_turn(success, "goal-result.txt", read_source=True)
                self.check_file("goal-result.txt")
                self.pass_goal("goal-success")
                refused = await self.goal("goal-refused", "只尝试一次write_file写入blocked.txt，内容为blocked；若拒绝就说明并结束，不换工具或路径。", 1,
                                          files=set(), denied={"blocked.txt"})
                assert not (self.workspace / "blocked.txt").exists()
                check_expected_failure(
                    refused, tool_name="write_file", parameters={"path": "blocked.txt", "content": "blocked"},
                    error_class="permission_denied", error_prefix="Permission was not granted",
                    executed=False, decision="n",
                )
                self.pass_goal("goal-refused")
                fail_command = "python -c " + shlex.quote("raise SystemExit(7)")
                failed = await self.goal("goal-tool-failed", "仅调用一次bash执行下面精确命令，timeout=30；若失败就如实说明并结束，不重试：\n" + fail_command, 1,
                                         files=set(), commands={fail_command})
                check_expected_failure(
                    failed, tool_name="bash", parameters={"command": fail_command, "timeout": 30},
                    error_class="runtime_error", error_prefix="[exit 7]\n", executed=True, decision="y",
                )
                self.pass_goal("goal-tool-failed")
                outside = self.root / "outside.txt"
                outside.write_text("outside-sentinel-" + secrets.token_hex(16), encoding="utf-8")
                outside_content = outside.read_bytes()
                escaped = await self.goal(
                    "goal-path-escape",
                    "本次验证入口的工作区边界。先仅调用一次 read_file 读取源数据.txt，等成功后，"
                    "再仅调用一次 read_file 读取 ../outside.txt；允许工具自动拒绝访问，"
                    "随后如实说明并结束，不改用其他工具或路径。",
                    1, files={"源数据.txt"},
                )
                check_permission_workspace_guard(escaped, self.workspace, self.marker)
                check_outside_sentinel(outside, outside_content, escaped, (
                    self.root / "cli" / "goal-path-escape" / "terminal.ansi.log"
                ).read_text(encoding="utf-8"))
                report["outside_sentinel_unchanged"] = True
                next(case for case in report["cases"] if case["name"] == "goal-path-escape").update(
                    boundary="permission_workspace_guard", docker_worker_boundary="not_tested_by_cli",
                    same_run_control_read=True, no_approval_prompt_or_answer=True,
                    outside_sentinel_unchanged=True, outside_sentinel_disclosed=False,
                )
                self.pass_goal("goal-path-escape")
                cancel = self.cancel_command("goal")
                cancelled = await self.goal("goal-cancelled", "仅调用一次bash执行下面精确命令，timeout=30：\n" + cancel, 130,
                                            files=set(), commands={cancel}, cancel_marker="goal-started.txt")
                check_cancelled_turn(cancelled, cancel)
            chat = self.start_cli("chat", ["chat"])
            await asyncio.sleep(0.5)
            for number in range(1, 4):
                output = f"chat-{number}.txt"
                prompt = (f"只用read_file读取源数据.txt，把完整标记write_file写入{output}并读回，记住这个标记，结束这一轮。")
                result = await self.chat_turn(chat, prompt, files={"源数据.txt", output})
                check_file_turn(result, output, read_source=True, note_marker=self.marker)
                self.check_file(output)
                if number == 1:
                    session_id = result["run"]["session_id"]
                else:
                    assert result["run"]["session_id"] == session_id, "chat switched sessions between turns"
            session_id = result["run"]["session_id"]
            chat_cancel = self.cancel_command("chat")
            result = await self.chat_turn(chat, "仅调用一次bash执行下面精确命令，timeout=30：" + chat_cancel,
                                          files=set(), commands={chat_cancel}, cancel_marker="chat-started.txt")
            assert result["run"]["session_id"] == session_id
            check_cancelled_turn(result, chat_cancel)
            result = await self.chat_turn(chat, "取消已经结束。仅根据会话记忆，把完整标记write_file写入chat-after-cancel.txt并结束。",
                                          files={"chat-after-cancel.txt"})
            assert result["run"]["session_id"] == session_id
            check_file_turn(result, "chat-after-cancel.txt", read_source=False, note_marker=self.marker)
            self.check_file("chat-after-cancel.txt")
            chat.send("\x1a\r", "idle Windows EOF: preserve session")
            await chat.finish(0)
            await self.ping("chat-after-eof")
            before_resume = self.execution_ids()
            resumed = self.start_cli("chat-resumed", ["chat", "--resume", session_id])
            await asyncio.sleep(0.5)
            await resumed.idle()
            assert self.execution_ids() == before_resume, "resume replayed an old task or tool before new input"
            result = await self.chat_turn(resumed, "仅根据会话记忆，把完整标记write_file写入chat-restored.txt并结束。",
                                          files={"chat-restored.txt"})
            assert result["run"]["session_id"] == session_id
            check_file_turn(result, "chat-restored.txt", read_source=False, note_marker=self.marker)
            after_resume = self.execution_ids()
            assert after_resume[0] - before_resume[0] == {result["run"]["id"]}, "resume submitted an additional Run"
            assert after_resume[1] - before_resume[1] == {tool["id"] for tool in result["tools"]}, "resume executed an additional historical tool"
            self.check_file("chat-restored.txt")
            resumed.send("\x1a\r", "idle Windows EOF after restored task")
            await resumed.finish(0)
            await self.ping("chat-resumed-after-eof")
            await asyncio.sleep(21)
            if self.workflow == "all":
                assert not (self.workspace / "goal-forbidden.txt").exists()
                self.pass_goal("goal-cancelled")
            assert not (self.workspace / "chat-forbidden.txt").exists()
            report["cases"].append({"name": "chat-three-turns-cancel-continue-resume", "status": "passed", "session_id": session_id})
            report["status"] = "passed"
        except Exception as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        finally:
            for pty in self.ptys:
                try:
                    await pty.cleanup()
                except Exception as exc:
                    report.setdefault("cleanup_issues", []).append(type(exc).__name__)
            try:
                await self.core.stop()
                if (self.core.home / "state.db").is_file():
                    report["submissions"] = [
                        {"run_id": row["id"], "session_id": row["session_id"], "via": "actual CLI"}
                        for row in self.rows("SELECT id,session_id FROM runs")
                    ]
                    await self.core.verify_owned_containers()
                if self.core.port:
                    try:
                        _reader, writer = await asyncio.open_connection("127.0.0.1", self.core.port)
                    except OSError:
                        report["owned_core_port_closed"] = True
                    else:
                        writer.close()
                        await writer.wait_closed()
                        raise RuntimeError("Owned Core port remained open after cleanup")
            except Exception as exc:
                report.setdefault("cleanup_issues", []).append(type(exc).__name__)
            if self.core.ledger.is_file():
                report["budget_after"] = counts(self.core.ledger)
            report["finished_at"] = now()
            if report.get("cleanup_issues"):
                report["status"] = "failed_cleanup"
            self.core.checkpoint()
        print(json.dumps({"status": report["status"], "manifest": str(self.root / "manifest.json")}, ensure_ascii=False))
        return 0 if report["status"] == "passed" else 1


async def pty_self_test(output_root: Path) -> int:
    harness = CliAcceptance(output_root)
    home = harness.root / "client-home"
    code = ("import sys; print('READY',flush=True); "
            "print('FIRST='+input(),flush=True); print('SECOND='+input(),flush=True); "
            "print('WAITING',flush=True);\ntry: input()\nexcept KeyboardInterrupt: sys.exit(130)")
    pty = CliPty([sys.executable, "-u", "-c", code], harness.workspace, home, 1, harness.root / "pty-keys")

    async def text_seen(terminal: CliPty, text: str) -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            terminal.pump()
            if text in terminal.text:
                return
            await asyncio.sleep(0.05)
        raise TimeoutError("ConPTY self-test marker not received: " + text)

    terminals = [pty]
    try:
        await text_seen(pty, "READY")
        pty.send("y\r", "transport self-test y")
        await text_seen(pty, "FIRST=y")
        pty.send("n\r", "transport self-test n")
        await text_seen(pty, "SECOND=n")
        await text_seen(pty, "WAITING")
        pty.send("\x03", "transport self-test Ctrl+C")
        await pty.finish(130)
        eof = CliPty([sys.executable, "-u", "-c", "import sys; print('EOF-READY',flush=True); sys.exit(0 if sys.stdin.readline()=='' else 2)"],
                     harness.workspace, home, 1, harness.root / "pty-eof")
        terminals.append(eof)
        await text_seen(eof, "EOF-READY")
        eof.send("\x1a\r", "transport self-test Windows EOF")
        await eof.finish(0)
        harness.core.report.update(kind="cli_pty_transport_self_test", status="passed", model_calls=0,
                                    note="ConPTY keys/exit codes only; not real CLI acceptance")
    finally:
        for terminal in terminals:
            await terminal.cleanup()
        harness.core.checkpoint()
    print(json.dumps({"status": harness.core.report["status"], "root": str(harness.root), "model_calls": 0}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Create only new synthetic fixtures; no config or model request")
    parser.add_argument("--pty-self-test", action="store_true", help="Only test ConPTY y/n, Ctrl+C and EOF with a synthetic local process")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workflow", choices=["all", "chat"], default="all")
    parser.add_argument("--client-python", type=Path, help="Use this installed environment's sibling tars console entry; default is python -m")
    args = parser.parse_args()
    if args.pty_self_test:
        return asyncio.run(pty_self_test(args.output_root))
    if not args.execute and not args.prepare:
        print(json.dumps({"mode": "plan_only_no_requests", "output_root": str(args.output_root),
                          "selected_workflow": args.workflow,
                          "client_python": str(args.client_python or sys.executable),
                          "cases": (["goal-success-0", "goal-refused-1", "goal-tool-failed-1", "goal-path-escape-1", "goal-cancel-130"]
                                    if args.workflow == "all" else []) + ["chat-three-turns-cancel-continue-eof-resume"]}))
        return 0
    harness = CliAcceptance(args.output_root, args.client_python, args.workflow)
    if not args.execute:
        harness.core.report["status"] = "prepared_no_requests"
        harness.core.checkpoint()
        print(json.dumps({"status": "prepared_no_requests", "root": str(harness.root)}, ensure_ascii=False))
        return 0
    return asyncio.run(harness.execute())


if __name__ == "__main__":
    raise SystemExit(main())
