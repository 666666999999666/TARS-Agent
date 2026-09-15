"""Verify an installed distribution from outside its source checkout, without model calls."""
from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.resources
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _port_closed(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def _pid_running(pid: int) -> bool:
    if os.name == "nt":
        executable = Path(os.environ["SystemRoot"]) / "System32" / "tasklist.exe"
        checked = subprocess.run(
            [str(executable), "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW, check=True,
        )
        return any(len(row) > 1 and row[1] == str(pid) for row in csv.reader(checked.stdout.splitlines()))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def verify_installed_core(home: Path, output: Path, results: dict[str, object]) -> None:
    from tars_agent.core.control import read_control_file, remove_control_file
    from tars_agent.core.transport.socket_client import SocketClient

    port = free_port()
    workspace = home / "core-workspace"
    workspace.mkdir(exist_ok=True)
    empty_config = workspace / "empty-config.toml"
    empty_config.write_text("", encoding="utf-8")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.upper().startswith(("TARS_", "KAMA_", "ANTHROPIC_", "OPENAI_", "DEEPSEEK_"))
        and not key.upper().endswith("_API_KEY") and key.upper() not in {"PYTHONPATH", "PYTHONHOME"}
    }
    image = os.environ.get("TARS_SANDBOX_IMAGE", "tars-agent-sandbox:0.8.0")
    environment.update({
        "TARS_HOME": str(home), "TARS_CONFIG": str(empty_config),
        "TARS_HOST": "127.0.0.1", "TARS_PORT": str(port),
        "TARS_SANDBOX_MODE": "required", "TARS_SANDBOX_IMAGE": image,
        "TARS_TRACE_ENABLED": "false",
        "TARS_LOG_FILE": "", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
    })
    control_path = home / "control" / f"tars-core-{port}.json"
    assert not control_path.exists(), "selected Core control path already exists"
    logs_before = set((home / "logs").glob("core-launch-*.log"))
    results.update(core_port=port, core_sandbox_mode="required", core_image=image,
                   installed_core="not_verified",
                   core_python=sys.executable, core_start_outer_timeout_s=75)
    core_pid: int | None = None

    def invoke(action: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "tars_agent.cli", "core", action], cwd=workspace,
            env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=75 if action == "start" else 20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
        )

    def owned_control():
        control = read_control_file(control_path)
        created = set((home / "logs").glob("core-launch-*.log")) - logs_before
        launch_ids = {path.name[len("core-launch-"):-len(".log")] for path in created}
        if (control is not None and control.launch_id in launch_ids
                and control.host == "127.0.0.1" and control.port == port):
            return control
        return None

    async def ping() -> dict[str, object]:
        client = SocketClient("127.0.0.1", port)
        await asyncio.wait_for(client.connect(), timeout=3)
        reader = asyncio.create_task(client.run_event_loop())
        try:
            return await client.send_command("core.ping", {"client": "installed-verifier"}, timeout_s=3)
        finally:
            await client.close()
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)

    try:
        started = invoke("start")
        assert started.returncode == 0 and started.stdout.strip().startswith("started"), "installed core start failed; inspect its isolated HOME logs"
        control = owned_control()
        assert control is not None, "installed Core launch ownership was not established"
        core_pid = control.pid
        results["core_start_cli"] = "passed"
        results["core_pid"] = core_pid
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        assert asyncio.run(ping()).get("launch_id") == control.launch_id, "Core RPC launch identity mismatch"
        status = invoke("status")
        assert status.returncode == 0 and status.stdout.strip().startswith("running"), "installed core status RPC failed"
        results["core_status_rpc"] = "passed"
        stopped = invoke("stop")
        assert stopped.returncode == 0 and stopped.stdout.strip().startswith("stopped"), "installed core stop failed"
        results["core_stop_cli"] = "passed"
    finally:
        control = owned_control()
        if control is not None:
            core_pid = control.pid
            results["core_pid"] = core_pid
            try:
                invoke("stop")
            except subprocess.SubprocessError:
                results["core_cleanup_retry_failed"] = True
        deadline = time.monotonic() + 10
        while core_pid is not None and time.monotonic() < deadline:
            if not _pid_running(core_pid) and _port_closed(port):
                break
            time.sleep(0.1)
        results["core_stopped"] = core_pid is not None and not _pid_running(core_pid)
        results["core_port_closed"] = _port_closed(port)
        control = owned_control()
        if control is not None and results["core_stopped"] and results["core_port_closed"]:
            remove_control_file(control.token, control_path)
        results["core_control_removed"] = not control_path.exists()
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    assert results["core_stopped"] and results["core_port_closed"] and results["core_control_removed"], "installed Core cleanup was not fully confirmed"
    results["installed_core"] = "passed"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import tars_agent
    from tars_agent.core.persistence import (
        CURRENT_SCHEMA_REVISION,
        bootstrap_state,
        read_schema_revision,
    )

    prefix = Path(sys.prefix).resolve()
    module = Path(tars_agent.__file__).resolve()
    assert module.is_relative_to(prefix), "package was borrowed from a source checkout"
    assert tars_agent.__version__ == "0.8.0"
    home = Path(os.environ["TARS_HOME"]).resolve()
    assert home.is_relative_to(prefix.parent) and home != prefix, "HOME must belong to the isolated installation case"
    home.mkdir(parents=True, exist_ok=True)
    scripts = Path(sys.executable).parent
    results: dict[str, object] = {"prefix": str(prefix), "module": str(module), "version": tars_agent.__version__}
    for command in ("tars", "tars-core", "tars-tui", "tars-web"):
        executable = scripts / (command + ".exe" if os.name == "nt" else command)
        completed = subprocess.run([str(executable), "--help"], capture_output=True, timeout=15, check=True)
        assert b"usage" in completed.stdout.lower()
    results["four_entrypoint_help"] = "passed"

    async def schema() -> None:
        bootstrap = await bootstrap_state(home / "state.db")
        try:
            assert read_schema_revision(home / "state.db") == CURRENT_SCHEMA_REVISION
        finally:
            await bootstrap.database.dispose()
    asyncio.run(schema())
    results["schema_resources"] = "passed"
    verify_installed_core(home, args.output, results)
    resources = importlib.resources.files("tars_agent.sandbox")
    assert resources.joinpath("Dockerfile").read_bytes(), "packaged sandbox Dockerfile is unreadable"
    workspace = home / "worker-workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "marker.txt").write_text("installed-worker-marker", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(resources.joinpath("worker.py")), "--workspace", str(workspace)],
        input=json.dumps({"tool_name": "read_file", "params": {"path": "marker.txt"}}).encode() + b"\n",
        capture_output=True, timeout=10, check=True,
    )
    worker_result = json.loads(completed.stdout)
    assert not worker_result["is_error"] and "installed-worker-marker" in worker_result["content"]
    results["installed_worker"] = "passed"
    assert importlib.resources.files("tars_agent.web").joinpath("static/index.html").is_file()
    port = free_port()
    web_environment = os.environ.copy()
    web_environment["TARS_PORT"] = str(free_port())
    process = subprocess.Popen([sys.executable, "-c",
                                "from tars_agent.web.cli import main; main()", "--port", str(port)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=web_environment)
    results["web_pid"] = process.pid
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Installed Web service exited before ready")
            try:
                with opener.open(  # fixed loopback, explicitly bypass system proxies
                    f"http://127.0.0.1:{port}/healthz", timeout=4.0
                ) as response:
                    if response.status == 200:
                        health = json.loads(response.read())
                        assert health["core"] == "disconnected"
                        results["web_health_without_core"] = health
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.05)
        else:
            raise TimeoutError("Installed Web service did not become ready")
        results["installed_web_health"] = "passed"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        results["web_stopped"] = process.poll() is not None
        close_deadline = time.monotonic() + 5
        while time.monotonic() < close_deadline:
            with socket.socket() as check:
                check.settimeout(0.2)
                closed = check.connect_ex(("127.0.0.1", port)) != 0
            if closed:
                break
            time.sleep(0.05)
        results["web_port_closed"] = closed
        assert closed, "installed Web port remained open"
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
