from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from tars_agent.core.control import read_control_file
from tars_agent.core.transport.socket_client import SocketClient

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_port(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"process exited ({process.returncode})\n"
                f"stdout: {stdout.decode(errors='replace')}\n"
                f"stderr: {stderr.decode(errors='replace')}"
            )
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.05)
            continue
        writer.close()
        await writer.wait_closed()
        del reader
        return
    raise TimeoutError(f"service did not listen on port {port}")


async def _command(port: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    client = SocketClient("127.0.0.1", port)
    await client.connect()
    loop_task = asyncio.create_task(client.run_event_loop())
    try:
        return await client.send_command(method, params)
    finally:
        await client.close()
        await asyncio.gather(loop_task, return_exceptions=True)


async def main() -> int:
    core_port = _free_port()
    web_port = _free_port()
    token = "e2e-bootstrap-token"
    test_home = Path(tempfile.mkdtemp(prefix="tars-web-e2e-"))
    python_executable = sys.executable
    core_env = {key: value for key, value in os.environ.items()
                if not key.startswith(("TARS_", "KAMA_", "ANTHROPIC_", "OPENAI_"))}
    core_env.update(
        {
            "TARS_PORT": str(core_port),
            "TARS_HOME": str(test_home),
            "TARS_CONFIG": str(test_home / "config.toml"),
            "TARS_SANDBOX_MODE": "preferred",  # IPC/browser test; not Docker evidence.
            "TARS_LOG_FILE": "",
            "TARS_LOG_LEVEL": "WARNING",
            "TARS_TRACE_ENABLED": "false",
        }
    )
    core = subprocess.Popen(
        [python_executable, "-m", "tars_agent.core"],
        cwd=ROOT,
        env=core_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    web: subprocess.Popen[bytes] | None = None
    try:
        await _wait_for_port(core_port, core)
        await _command(
            core_port,
            "session.create",
            {
                "mode": "chat",
                "title": "malicious session <script>alert(1)</script>",
                "workspace_root": str(ROOT),
            },
        )
        reconnect_session = await _command(
            core_port,
            "session.create",
            {
                "mode": "chat",
                "title": "reconnect session",
                "workspace_root": str(ROOT),
            },
        )
        reconnect_session_id = str(reconnect_session["session_id"])
        # Persist a second genuine event before the browser connects.  The E2E
        # server will interrupt replay after session.created, so the browser must
        # resume from Last-Event-ID to receive this session.resumed envelope.
        await _command(
            core_port,
            "session.resume",
            {"session_id": reconnect_session_id},
        )
        web_env = core_env | {
            "TARS_WEB_BOOTSTRAP_TOKEN": token,
            "TARS_WEB_E2E_PORT": str(web_port),
            "TARS_WEB_E2E_RECONNECT_SESSION_ID": reconnect_session_id,
        }
        web = subprocess.Popen(
            [python_executable, str(ROOT / "tests/integration/web_e2e_server.py")],
            cwd=ROOT,
            env=web_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await _wait_for_port(web_port, web)
        playwright_env = os.environ.copy() | {
            "TARS_WEB_BOOTSTRAP_TOKEN": token,
            "TARS_WEB_E2E_URL": f"http://127.0.0.1:{web_port}",
        }
        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError("npm is not available")
        completed = subprocess.run(
            [npm, "run", "test:e2e"],
            cwd=WEB_DIR,
            env=playwright_env,
            check=False,
        )
        return completed.returncode
    finally:
        if web is not None and web.poll() is None:
            web.terminate()
            try:
                web.wait(timeout=5)
            except subprocess.TimeoutExpired:
                web.kill()
                web.wait()
        if core.poll() is None:
            try:
                control = read_control_file(test_home / "control" / f"tars-core-{core_port}.json")
                if control is None:
                    raise RuntimeError("missing test Core control file")
                await _command(core_port, "core.shutdown", {"token": control.token})
                await asyncio.to_thread(core.wait, 5)
            except Exception:
                core.terminate()
                try:
                    core.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    core.kill()
                    core.wait()
        evidence = Path(os.environ.get("TARS_WEB_EVIDENCE_DIR", ROOT / "build/qa")) / "web-e2e-processes.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(json.dumps({"core_pid": core.pid, "core_stopped": core.poll() is not None,
                                       "web_pid": web.pid if web else None,
                                       "web_stopped": web is None or web.poll() is not None}), encoding="utf-8")
        if test_home.exists():
            shutil.rmtree(test_home, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
