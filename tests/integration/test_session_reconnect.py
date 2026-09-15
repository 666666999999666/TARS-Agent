from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tars_agent.core.control import read_control_file
from tars_agent.core.transport.socket_client import SocketClient


def _start_delayed_daemon(port: int, home: Path) -> subprocess.Popen[bytes]:
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
        }
    )
    return subprocess.Popen(
        [sys.executable, "-m", "tests.integration.delayed_daemon"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


async def _wait_listening(proc: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + 10.0
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
    raise AssertionError("daemon did not start within 10 seconds")


async def _wait_terminal(client: SocketClient, run_id: str) -> dict[str, Any]:
    for _ in range(100):
        result = await client.send_command("run.get", {"run_id": run_id})
        if result["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return result
        await asyncio.sleep(0.01)
    raise AssertionError(f"run did not finish: {run_id}")


# 功能：验证客户端断开不会关闭 Session 或取消 Run，重连后按 cursor 仅补齐未见事件
# 设计：第一连接创建并提交后立刻断开；第二连接 resume 同一 Session，从首连接 cursor 续传并查询终态
async def test_disconnect_keeps_run_and_cursor_resume_has_no_duplicates(
    free_port: int,
    tmp_path: Path,
) -> None:
    proc = _start_delayed_daemon(free_port, tmp_path / "kama-home")
    first: SocketClient | None = None
    first_loop: asyncio.Task[None] | None = None
    second: SocketClient | None = None
    second_loop: asyncio.Task[None] | None = None
    try:
        await _wait_listening(proc, free_port)
        first = SocketClient("127.0.0.1", free_port)
        await first.connect()
        first_loop = asyncio.create_task(first.run_event_loop())
        created = await first.send_command(
            "session.create",
            {"mode": "chat", "workspace_root": str(tmp_path)},
        )
        session_id = str(created["session_id"])
        run_started = asyncio.Event()

        async def observe_first(envelope: dict[str, Any]) -> None:
            if envelope.get("event", {}).get("type") == "run.started":
                run_started.set()

        first.on_event_envelope(observe_first)
        await first.send_command(
            "event.subscribe",
            {"topics": ["*"], "session_id": session_id, "after_cursor": 0},
        )
        submitted = await first.send_command(
            "session.send_message",
            {
                "session_id": session_id,
                "content": "disconnect while running",
                "client_message_id": "disconnect-running-1",
            },
        )
        run_id = str(submitted["run_id"])
        await asyncio.wait_for(run_started.wait(), timeout=5.0)
        running = await first.send_command("run.get", {"run_id": run_id})
        assert running["status"] == "running"
        first_cursor = first.last_cursor
        assert first_cursor > 0

        await first.close()
        await asyncio.gather(first_loop, return_exceptions=True)
        first = None
        first_loop = None

        second = SocketClient("127.0.0.1", free_port)
        await second.connect()
        received_cursors: list[int] = []
        run_finished = asyncio.Event()

        async def collect(envelope: dict[str, Any]) -> None:
            received_cursors.append(int(envelope["cursor"]))
            event = envelope.get("event", {})
            if event.get("type") == "run.finished" and event.get("run_id") == run_id:
                run_finished.set()

        second.on_event_envelope(collect)
        second_loop = asyncio.create_task(second.run_event_loop())
        resumed = await second.send_command("session.resume", {"session_id": session_id})
        assert resumed["session"]["session_id"] == session_id
        await second.send_command(
            "event.subscribe",
            {
                "topics": ["*"],
                "session_id": session_id,
                "after_cursor": first_cursor,
            },
        )
        terminal = await _wait_terminal(second, run_id)
        assert terminal["status"] == "succeeded"
        assert terminal["result"] == {
            "text": "completed after disconnect",
            "steps": 1,
        }

        await asyncio.wait_for(run_finished.wait(), timeout=2.0)
        assert received_cursors == sorted(set(received_cursors))
        assert all(cursor > first_cursor for cursor in received_cursors)
        session = await second.send_command("session.resume", {"session_id": session_id})
        assert session["session"]["status"] == "ready"
        control = read_control_file(tmp_path / "kama-home" / "control" / f"tars-core-{free_port}.json")
        assert control is not None
        await second.send_command("core.shutdown", {"token": control.token})
        await asyncio.to_thread(proc.wait, 5)
        assert proc.returncode == 0
    finally:
        if first is not None:
            await first.close()
        if first_loop is not None:
            await asyncio.gather(first_loop, return_exceptions=True)
        if second is not None:
            await second.close()
        if second_loop is not None:
            await asyncio.gather(second_loop, return_exceptions=True)
        if proc.poll() is None:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 2)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait, 2)
