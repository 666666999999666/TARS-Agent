from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tars_agent.core.control import read_control_file
from tars_agent.core.transport.socket_client import IpcError, SocketClient


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _start_daemon(
    port: int,
    home: Path,
    *,
    scenario: str = "side_effect",
    marker: Path | None = None,
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.update(
        {
            "TARS_PORT": str(port),
            "TARS_HOME": str(home),
            "TARS_LOG_FILE": "",
            "TARS_LOG_LEVEL": "WARNING",
            "TARS_TRACE_ENABLED": "false",
            "TARS_SANDBOX_MODE": "preferred",
            "TARS_CRASH_SCENARIO": scenario,
        }
    )
    if marker is not None:
        env["TARS_CRASH_MARKER"] = str(marker)
    return subprocess.Popen(
        [sys.executable, "-m", "tests.integration.crash_daemon"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


async def _wait_listening(proc: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise AssertionError(
                f"daemon exited early ({proc.returncode})\n"
                f"stdout={stdout.decode(errors='replace')}\n"
                f"stderr={stderr.decode(errors='replace')}"
            )
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(0.05)
        else:
            writer.close()
            await writer.wait_closed()
            return
    raise AssertionError("daemon did not start within 45 seconds")


async def _connect(port: int) -> tuple[SocketClient, asyncio.Task[None]]:
    client = SocketClient("127.0.0.1", port)
    await client.connect()
    loop = asyncio.create_task(client.run_event_loop())
    return client, loop


async def _wait_for(
    predicate: Any,
    *,
    timeout: float = 5.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


async def _stop_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()
    await asyncio.to_thread(proc.wait, 5)


@pytest.mark.timeout(120)
async def test_hard_kill_marks_run_interrupted_without_replaying_side_effect(
    tmp_path: Path,
) -> None:
    port = _free_port()
    home = tmp_path / "kama-home"
    first = _start_daemon(port, home)
    client: SocketClient | None = None
    reader_task: asyncio.Task[None] | None = None
    restarted: subprocess.Popen[bytes] | None = None
    second_client: SocketClient | None = None
    second_reader: asyncio.Task[None] | None = None
    try:
        await _wait_listening(first, port)
        client, reader_task = await _connect(port)
        created = await client.send_command(
            "session.create",
            {"mode": "chat", "workspace_root": str(tmp_path)},
        )
        session_id = str(created["session_id"])
        tool_finished = asyncio.Event()

        async def observe(envelope: dict[str, Any]) -> None:
            event = envelope.get("event", {})
            if event.get("type") == "tool.call_finished":
                tool_finished.set()

        client.on_event_envelope(observe)
        await client.send_command(
            "event.subscribe",
            {"topics": ["*"], "session_id": session_id, "after_cursor": 0},
        )
        submitted = await client.send_command(
            "session.send_message",
            {
                "session_id": session_id,
                "content": "save and pause",
                "client_message_id": "crash-window-1",
            },
        )
        run_id = str(submitted["run_id"])
        await asyncio.wait_for(tool_finished.wait(), timeout=5.0)
        await _wait_for(
            lambda: (home / "artifacts" / "sessions" / session_id / "notes.md").exists()
        )
        before = (home / "artifacts" / "sessions" / session_id / "notes.md").read_text(
            encoding="utf-8"
        )
        running = await client.send_command("run.get", {"run_id": run_id})
        assert running["status"] == "running"
        assert running["side_effects_started"] is True

        first.kill()
        await asyncio.to_thread(first.wait, 5)
        await asyncio.gather(reader_task, return_exceptions=True)
        client = None
        reader_task = None

        restarted = _start_daemon(port, home)
        await _wait_listening(restarted, port)
        second_client, second_reader = await _connect(port)
        recovered = await second_client.send_command("run.get", {"run_id": run_id})
        assert recovered["status"] == "interrupted"
        assert recovered["reason"] == "daemon_restarted"
        assert recovered["side_effects_started"] is True

        await asyncio.sleep(0.1)
        after = (home / "artifacts" / "sessions" / session_id / "notes.md").read_text(
            encoding="utf-8"
        )
        assert after == before
        assert after.count("written exactly once before crash") == 1

        try:
            await second_client.send_command(
                "session.retry",
                {"run_id": run_id, "confirm_side_effects": False},
            )
        except IpcError as exc:
            assert exc.code == -32032
        else:
            raise AssertionError("retry without side-effect confirmation was accepted")

        control = read_control_file(home / "control" / f"tars-core-{port}.json")
        assert control is not None
        await second_client.send_command("core.shutdown", {"token": control.token})
        await asyncio.to_thread(restarted.wait, 5)
        restarted = None
    finally:
        if client is not None:
            await client.close()
        if reader_task is not None:
            await asyncio.gather(reader_task, return_exceptions=True)
        if second_client is not None:
            await second_client.close()
        if second_reader is not None:
            await asyncio.gather(second_reader, return_exceptions=True)
        if first.poll() is None:
            await _stop_process(first)
        if restarted is not None:
            await _stop_process(restarted)


@pytest.mark.timeout(120)
async def test_hard_kill_and_restart_interrupts_active_subagent_without_resuming_it(
    tmp_path: Path,
) -> None:
    port = _free_port()
    home = tmp_path / "subagent-home"
    marker = tmp_path / "subagent-provider-calls.log"
    first = _start_daemon(port, home, scenario="subagent", marker=marker)
    client: SocketClient | None = None
    reader_task: asyncio.Task[None] | None = None
    restarted: subprocess.Popen[bytes] | None = None
    second_client: SocketClient | None = None
    second_reader: asyncio.Task[None] | None = None
    try:
        await _wait_listening(first, port)
        client, reader_task = await _connect(port)
        created = await client.send_command(
            "session.create",
            {"mode": "chat", "workspace_root": str(tmp_path)},
        )
        session_id = str(created["session_id"])
        permission_ready = asyncio.Event()
        child_started = asyncio.Event()
        observed: dict[str, str] = {}

        async def observe(envelope: dict[str, Any]) -> None:
            event = envelope.get("event", {})
            if (
                event.get("type") == "permission.requested"
                and event.get("tool_name") == "spawn_agent"
            ):
                observed["request_id"] = str(event["request_id"])
                permission_ready.set()
            elif event.get("type") == "subagent.started":
                observed["child_run_id"] = str(event["run_id"])
                child_started.set()

        client.on_event_envelope(observe)
        await client.send_command(
            "event.subscribe",
            {"topics": ["*"], "session_id": session_id, "after_cursor": 0},
        )
        submitted = await client.send_command(
            "session.send_message",
            {
                "session_id": session_id,
                "content": "start a child and remain active",
                "client_message_id": "subagent-crash-window",
            },
        )
        parent_run_id = str(submitted["run_id"])
        await asyncio.wait_for(permission_ready.wait(), timeout=5)
        response = await client.send_command(
            "permission.respond",
            {
                "request_id": observed["request_id"],
                "session_id": session_id,
                "decision": "allow_once",
            },
        )
        assert response["ok"] is True
        await asyncio.wait_for(child_started.wait(), timeout=5)
        child_run_id = observed["child_run_id"]

        deadline = asyncio.get_running_loop().time() + 5
        while True:
            child_state = await client.send_command("run.get", {"run_id": child_run_id})
            if child_state["status"] == "running":
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("subagent did not reach running state")
            await asyncio.sleep(0.01)
        await _wait_for(
            lambda: marker.exists() and child_run_id in marker.read_text(encoding="utf-8")
        )
        calls_before_restart = marker.read_text(encoding="utf-8")

        first.kill()
        await asyncio.to_thread(first.wait, 5)
        await asyncio.gather(reader_task, return_exceptions=True)
        client = None
        reader_task = None

        restarted = _start_daemon(port, home, scenario="subagent", marker=marker)
        await _wait_listening(restarted, port)
        second_client, second_reader = await _connect(port)
        parent = await second_client.send_command("run.get", {"run_id": parent_run_id})
        child = await second_client.send_command("run.get", {"run_id": child_run_id})
        assert parent["status"] == "interrupted"
        assert parent["reason"] == "daemon_restarted"
        assert child["kind"] == "subagent"
        assert child["status"] == "interrupted"
        assert child["reason"] == "daemon_restarted"

        await asyncio.sleep(0.2)
        assert marker.read_text(encoding="utf-8") == calls_before_restart

        control = read_control_file(home / "control" / f"tars-core-{port}.json")
        assert control is not None
        await second_client.send_command("core.shutdown", {"token": control.token})
        await asyncio.to_thread(restarted.wait, 5)
        restarted = None
    finally:
        if client is not None:
            await client.close()
        if reader_task is not None:
            await asyncio.gather(reader_task, return_exceptions=True)
        if second_client is not None:
            await second_client.close()
        if second_reader is not None:
            await asyncio.gather(second_reader, return_exceptions=True)
        if first.poll() is None:
            await _stop_process(first)
        if restarted is not None:
            await _stop_process(restarted)
