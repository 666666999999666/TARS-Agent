"""Public CLI processes against a local protocol peer; no model calls or Core state."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.cli_core_stub import CliCoreStub

pytestmark = pytest.mark.local_integration


async def run_cli(core: CliCoreStub, tmp_path: Path, *arguments: str, stdin: bytes = b"", keep_stdin_open: bool = False):
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("TARS_", "KAMA_", "ANTHROPIC_", "OPENAI_", "DEEPSEEK_"))
        and not key.endswith("_API_KEY")
    }
    environment.update({
        "TARS_HOST": "127.0.0.1", "TARS_PORT": str(core.port),
        "TARS_HOME": str(tmp_path / "cli-home"),
        "TARS_CONFIG": str(tmp_path / "empty-config.toml"),
        "TARS_LOG_FILE": "", "TARS_TRACE_ENABLED": "false",
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
    })
    (tmp_path / "empty-config.toml").write_text("", encoding="utf-8")
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tars_agent.cli", *arguments,
        cwd=tmp_path, env=environment, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        if keep_stdin_open:
            assert process.stdin is not None and process.stdout is not None and process.stderr is not None
            if stdin:
                process.stdin.write(stdin)
                await process.stdin.drain()
            await asyncio.wait_for(process.wait(), timeout=30)
            stdout, stderr = await process.stdout.read(), await process.stderr.read()
        else:
            stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout=30)
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.returncode is None:
            process.kill()
            await process.wait()
    return process.returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


async def test_public_goal_command_returns_real_peer_result_and_cwd(tmp_path):
    async with CliCoreStub() as core:
        code, stdout, stderr = await run_cli(core, tmp_path, "run", "--goal", "read sample")
        assert code == 0, stderr
        assert "wire-final" in stdout
        assert "run-1" in stderr and core.session_id in stderr
        create = next(params for method, params in core.calls if method == "session.create")
        assert create["mode"] == "one_shot"
        assert Path(create["workspace_root"]).resolve() == tmp_path.resolve()
        assert not any(method in {"core.shutdown", "session.close"} for method, _ in core.calls)


async def test_public_goal_script_gets_nonzero_after_recovered_tool_failure(tmp_path):
    async def submit(core, run_id):
        core.tool_failures[run_id] = 1
        await core.finish(run_id, text="repair succeeded after failed tool")

    async with CliCoreStub(submit) as core:
        code, stdout, stderr = await run_cli(core, tmp_path, "run", "--goal", "repair")
        assert code == 1, (stdout, stderr)
        assert "repair succeeded" in stdout


async def test_public_non_tty_goal_does_not_accept_piped_approval(tmp_path):
    async def submit(core, run_id):
        await core.permission(run_id, "write-needs-approval")

    async def denied(core, params):
        core.tool_failures[core.main_run_id] = 1
        await core.finish(core.main_run_id, text="not executed")

    async with CliCoreStub(submit) as core:
        core.on_permission = denied
        code, stdout, stderr = await run_cli(core, tmp_path, "run", "--goal", "write", stdin=b"y\n")
        assert code == 1, (stdout, stderr)
        assert [row["decision"] for row in core.approval_answers] == ["deny_once"]
        assert not any(method in {"core.shutdown", "session.close"} for method, _ in core.calls)


async def test_public_chat_eof_exits_without_waiting_for_input_thread(tmp_path):
    async with CliCoreStub() as core:
        code, stdout, stderr = await run_cli(core, tmp_path, "chat")
        assert code == 0, (stdout, stderr)
        assert not any(method in {"core.shutdown", "session.close", "run.cancel"} for method, _ in core.calls)


async def test_public_chat_resume_eof_preserves_the_session(tmp_path):
    async with CliCoreStub() as core:
        code, stdout, stderr = await run_cli(core, tmp_path, "chat", "--resume", core.session_id)
        assert code == 0, (stdout, stderr)
        assert not any(method in {"session.create", "session.close", "core.shutdown"} for method, _ in core.calls)

async def test_public_goal_exits_even_when_stdin_remains_open(tmp_path):
    async with CliCoreStub() as core:
        code, stdout, stderr = await run_cli(core, tmp_path, "run", "--goal", "finish", keep_stdin_open=True)
        assert code == 0, (stdout, stderr)
        assert "wire-final" in stdout


async def test_public_chat_disconnect_does_not_join_a_blocked_input_thread(tmp_path):
    async def submit(core, run_id):
        core.writers[-1].close()

    async with CliCoreStub(submit) as core:
        code, stdout, stderr = await run_cli(core, tmp_path, "chat", stdin=b"hello\n", keep_stdin_open=True)
        assert code == 1, (stdout, stderr)
        assert sum(method == "session.send_message" for method, _ in core.calls) == 1

async def wait_native_command(core: CliCoreStub, terminal, method: str, count: int = 1) -> None:
    async with asyncio.timeout(15):
        while sum(name == method for name, _ in core.calls) < count:
            terminal.pump()
            if not terminal.pty.isalive():
                raise AssertionError(f"CLI exited before {method}: {terminal.text[-1500:]}")
            await asyncio.sleep(0.02)

@pytest.mark.skipif(os.name != "nt", reason="Windows ConPTY keyboard regression")
async def test_conpty_chat_ctrl_c_continues_and_ctrl_z_exits(tmp_path):
    pytest.importorskip("winpty")
    from scripts.acceptance_cli import CliPty

    async def submit(core, run_id):
        if run_id == "run-2":
            await core.finish(run_id, text="second task completed")

    async with CliCoreStub(submit) as core:
        terminal = CliPty(
            [sys.executable, "-B", "-m", "tars_agent.cli", "chat"],
            tmp_path, tmp_path / "client-home", core.port, tmp_path / "native-chat",
        )
        try:
            await terminal.idle()
            terminal.send("first task\r", "submit controlled first task")
            await wait_native_command(core, terminal, "run.get")
            terminal.send("\x03", "interrupt the active CLI task")
            await wait_native_command(core, terminal, "run.cancel")
            await terminal.idle()
            assert terminal.pty.isalive(), "chat exited after Ctrl+C instead of returning to input"
            terminal.send("second task\r", "continue in the same CLI session")
            await wait_native_command(core, terminal, "session.send_message", 2)
            await terminal.idle()
            assert core.runs["run-1"]["status"] == "cancelled"
            assert core.runs["run-2"]["status"] == "succeeded"
            assert [params["content"] for name, params in core.calls if name == "session.send_message"] == ["first task", "second task"]
            terminal.send("\x1a\r", "Windows EOF at the idle prompt")
            assert await terminal.finish(0) == 0
            assert not any(name in {"session.close", "core.shutdown"} for name, _ in core.calls)
        finally:
            await terminal.cleanup()


@pytest.mark.skipif(os.name != "nt", reason="Windows ConPTY keyboard regression")
async def test_conpty_goal_ctrl_c_exits_130(tmp_path):
    pytest.importorskip("winpty")
    from scripts.acceptance_cli import CliPty

    async def submit(core, run_id):
        pass

    async with CliCoreStub(submit) as core:
        terminal = CliPty(
            [sys.executable, "-B", "-m", "tars_agent.cli", "run", "--goal", "wait for cancellation"],
            tmp_path, tmp_path / "client-home", core.port, tmp_path / "native-goal",
        )
        try:
            await wait_native_command(core, terminal, "run.get")
            terminal.send("\x03", "interrupt the active once-only task")
            await wait_native_command(core, terminal, "run.cancel")
            assert await terminal.finish(130) == 130
            assert core.runs["run-1"]["status"] == "cancelled"
        finally:
            await terminal.cleanup()
