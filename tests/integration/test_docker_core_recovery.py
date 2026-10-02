"""Opt-in, scoped A/B crash-recovery acceptance; never runs Docker at collection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from tars_agent.core.control import CORE_LAUNCH_ID_ENV, DaemonControl, read_control_file
from tars_agent.core.tools.runtime.recovery import SandboxResourceRecord, validate_container
from tars_agent.core.transport.socket_client import SocketClient
from tests.integration.python_process import python_module_command

pytestmark = pytest.mark.docker
_LATE_DELAY = 45.0


def _docker(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *arguments], capture_output=True, text=True, encoding="utf-8",
        timeout=15, check=False,
    )


def _inspect(container_id: str) -> dict[str, Any] | None:
    assert re.fullmatch(r"[0-9a-f]{64}", container_id)
    result = _docker("inspect", container_id)
    if result.returncode:
        absent = _docker("ps", "--all", "--quiet", "--no-trunc", "--filter", f"id={container_id}")
        if absent.returncode == 0 and not absent.stdout.strip():
            return None
        raise AssertionError("Docker could not confirm container identity or absence")
    objects = json.loads(result.stdout)
    assert len(objects) == 1 and objects[0]["Id"] == container_id
    return objects[0]


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@dataclass
class OwnedCore:
    home: Path
    port: int
    launch_id: str
    process: subprocess.Popen[bytes]
    log: BinaryIO
    client: SocketClient | None = None
    reader: asyncio.Task[None] | None = None
    control: DaemonControl | None = None

    async def ready(self) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"Core exited before ready; inspect {self.home / 'core.log'}")
            control = read_control_file(self.home / "control" / f"tars-core-{self.port}.json")
            if control is not None and control.launch_id == self.launch_id:
                client = SocketClient("127.0.0.1", self.port)
                try:
                    await client.connect()
                except OSError:
                    await asyncio.sleep(0.05)
                    continue
                reader = asyncio.create_task(client.run_event_loop())
                self.client, self.reader, self.control = client, reader, control
                pong = await client.send_command("core.ping", {})
                assert pong["launch_id"] == self.launch_id
                if control.pid != self.process.pid:
                    raise AssertionError(
                        "refuse an unverified trampoline child: "
                        f"launcher_pid={self.process.pid}, core_pid={control.pid}"
                    )
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f"Core startup timed out; inspect {self.home / 'core.log'}")

    async def disconnect(self) -> None:
        if self.client is not None:
            try:
                await self.client.close()
            finally:
                self.client = None
        if self.reader is not None:
            if not self.reader.done():
                self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
            self.reader = None

    async def stop(self) -> None:
        try:
            if self.process.poll() is None and self.client is not None:
                assert self.control is not None
                pong = await self.client.send_command("core.ping", {})
                assert pong["launch_id"] == self.launch_id
                await self.client.send_command("core.shutdown", {"token": self.control.token})
                await asyncio.to_thread(self.process.wait, 20)
            elif self.process.poll() is None:
                # This is a process object created by this test, never a discovered PID.
                self.process.terminate()
                await asyncio.to_thread(self.process.wait, 10)
        finally:
            try:
                if self.process.poll() is None:
                    self.process.kill()
                    await asyncio.to_thread(self.process.wait, 10)
                await self.disconnect()
            finally:
                self.log.close()


def _start(home: Path, image: str, port: int | None = None) -> OwnedCore:
    home.mkdir(parents=True, exist_ok=True)
    chosen_port = port or _port()
    launch_id = secrets.token_hex(24)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("TARS_", "KAMA_", "ANTHROPIC_", "OPENAI_"))}
    env.update({
        "TARS_HOME": str(home), "TARS_CONFIG": str(home / "config.toml"),
        "TARS_HOST": "127.0.0.1", "TARS_PORT": str(chosen_port),
        "TARS_SANDBOX_MODE": "required", "TARS_SANDBOX_IMAGE": image,
        "TARS_LOG_FILE": "", "TARS_LOG_LEVEL": "WARNING", "TARS_TRACE_ENABLED": "false",
        CORE_LAUNCH_ID_ENV: launch_id, "RECOVERY_PROBE_CALLS": str(home / "provider-calls.jsonl"),
    })
    log = (home / "core.log").open("ab")
    try:
        process = subprocess.Popen(
            python_module_command("tests.integration.recovery_daemon"),
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    except BaseException:
        log.close()
        raise
    return OwnedCore(home, chosen_port, launch_id, process, log)


def _workspace(root: Path, *, heartbeat: bool) -> Path:
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    skills = workspace / ".tars" / "skills"
    skills.mkdir(parents=True)
    (skills / "recovery_probe.md").write_text(
        "---\nname: recovery_probe\ndescription: isolated recovery verification\n"
        "allowed_tools:\n  - bash\n---\n$ARGUMENTS\n", encoding="utf-8",
    )
    common = "from pathlib import Path\nimport time\nPath('started.txt').write_text('started')\n"
    if heartbeat:
        code = common + (
            "deadline = time.monotonic() + 110\ncount = 0\n"
            "while time.monotonic() < deadline and not Path('stop.txt').exists():\n"
            "    count += 1\n    Path('heartbeat.txt').write_text(str(count))\n"
            "    time.sleep(0.1)\n"
        )
    else:
        code = common + f"time.sleep({_LATE_DELAY})\nPath('late.txt').write_text('late')\n"
    (workspace / "probe.py").write_text(code, encoding="utf-8")
    return workspace


async def _wait_file(path: Path, timeout: float = 25) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"probe did not create {path.name}")


async def _submit(core: OwnedCore, workspace: Path) -> tuple[str, SandboxResourceRecord]:
    assert core.client is not None
    client = core.client
    created = await client.send_command("session.create", {
        "mode": "chat", "workspace_root": str(workspace),
    })
    session_id = created["session_id"]
    approval: dict[str, Any] = {}
    approval_ready = asyncio.Event()

    async def observe(envelope: dict[str, Any]) -> None:
        event = envelope.get("event", {})
        if event.get("type") == "permission.requested":
            approval.update(event)
            approval_ready.set()

    client.on_event_envelope(observe)
    await client.send_command("event.subscribe", {"topics": ["*"], "session_id": session_id})
    submitted = await client.send_command("session.send_message", {
        "session_id": session_id, "content": "/recovery_probe Run the fixed probe.",
        "client_message_id": "fixed-probe",
    })
    await asyncio.wait_for(approval_ready.wait(), timeout=15)
    assert approval["session_id"] == session_id and approval["tool_name"] == "bash"
    assert approval["request_kind"] == "tool"
    assert approval["params"] == {"command": "python probe.py", "timeout": 120}
    await client.send_command("permission.respond", {
        "request_id": approval["request_id"], "session_id": session_id, "decision": "allow_once",
    })
    await _wait_file(workspace / "started.txt")
    paths = list((core.home / "control" / "sandbox-resources").glob("*.json"))
    assert len(paths) == 1
    record = SandboxResourceRecord.model_validate_json(paths[0].read_bytes())
    assert record.phase == "confirmed" and record.container_id is not None
    assert record.run_id == submitted["run_id"] and record.launch_id == core.launch_id
    home_id = hashlib.sha256(os.path.normcase(str(core.home.resolve())).encode()).hexdigest()
    assert record.home_id == home_id and Path(record.workspace_root) == workspace.resolve()
    inspected = await asyncio.to_thread(_inspect, record.container_id)
    assert inspected is not None
    validate_container(record, inspected)
    assert inspected["State"]["Running"] is True
    return record.run_id, record


@pytest.mark.timeout(240)
async def test_real_core_restart_reclaims_a_without_interrupting_b() -> None:
    if os.environ.get("RUN_CORE_RECOVERY_DOCKER") != "1":
        pytest.skip("explicit RUN_CORE_RECOVERY_DOCKER=1 is required")
    image = os.environ.get("CORE_RECOVERY_IMAGE", "")
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", image), "pin CORE_RECOVERY_IMAGE to an image ID"
    requested_root = os.environ.get("CORE_RECOVERY_ROOT")
    assert requested_root, "set a fresh CORE_RECOVERY_ROOT beneath build/core-recovery"
    project = Path(__file__).resolve().parents[2]
    root = Path(requested_root).resolve()
    allowed = (project / "build" / "core-recovery").resolve()
    assert allowed.is_relative_to(project.resolve())
    assert root != allowed and root.is_relative_to(allowed) and not root.exists()
    root.mkdir(parents=True)
    evidence: dict[str, Any] = {"image": image, "root": str(root), "passed": False,
                                "fallback_cleanup": [], "errors": []}
    cores: list[OwnedCore] = []
    records: list[SandboxResourceRecord] = []
    try:
        b_workspace = _workspace(root / "B", heartbeat=True)
        b = _start(root / "B" / "home", image)
        cores.append(b)
        await b.ready()
        b_run, b_record = await _submit(b, b_workspace)
        records.append(b_record)
        b_before = await asyncio.to_thread(_inspect, b_record.container_id)
        assert b_before is not None

        a_workspace = _workspace(root / "A", heartbeat=False)
        a = _start(root / "A" / "home", image)
        cores.append(a)
        await a.ready()
        a_run, a_record = await _submit(a, a_workspace)
        records.append(a_record)
        late_deadline = time.monotonic() + _LATE_DELAY + 1
        calls_before = (a.home / "provider-calls.jsonl").read_bytes()
        evidence["A"] = a_record.model_dump()
        evidence["B"] = b_record.model_dump()
        (root / "identities.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

        assert a.control is not None and a.client is not None
        if a.control.pid != a.process.pid:
            raise AssertionError("refuse to kill an unverified trampoline child")
        assert (await a.client.send_command("core.ping", {}))["launch_id"] == a.launch_id
        assert a.process.poll() is None
        a.process.kill()  # Only the verified process object created above.
        await asyncio.to_thread(a.process.wait, 10)
        await a.disconnect()
        leftover = await asyncio.to_thread(_inspect, a_record.container_id)
        assert leftover is not None and leftover["State"]["Running"] is True
        validate_container(a_record, leftover)
        evidence["orphan_observed_before_restart"] = True

        restarted = _start(a.home, image, a.port)
        cores.append(restarted)
        await restarted.ready()
        assert restarted.client is not None and b.client is not None
        recovered = await restarted.client.send_command("run.get", {"run_id": a_run})
        assert recovered["status"] == "interrupted" and recovered["reason"] == "daemon_restarted"
        assert await asyncio.to_thread(_inspect, a_record.container_id) is None
        assert not list((a.home / "control" / "sandbox-resources").glob("*.json"))
        assert (a.home / "provider-calls.jsonl").read_bytes() == calls_before

        b_after = await asyncio.to_thread(_inspect, b_record.container_id)
        assert b_after is not None and b_after["State"]["Running"] is True
        validate_container(b_record, b_after)
        assert b_after["State"]["StartedAt"] == b_before["State"]["StartedAt"]
        assert (await b.client.send_command("core.ping", {}))["launch_id"] == b.launch_id
        heartbeat_before = (b_workspace / "heartbeat.txt").read_bytes()
        await asyncio.sleep(0.5)
        assert (b_workspace / "heartbeat.txt").read_bytes() != heartbeat_before
        while time.monotonic() < late_deadline:
            await asyncio.sleep(0.2)
        assert not (a_workspace / "late.txt").exists()
        assert (a.home / "provider-calls.jsonl").read_bytes() == calls_before
        evidence["no_late_write"] = True
        evidence["no_replay"] = True
        evidence["B_identity_and_heartbeat_preserved"] = True

        (b_workspace / "stop.txt").write_text("stop", encoding="utf-8")
        deadline = time.monotonic() + 20
        while True:
            result = await b.client.send_command("run.get", {"run_id": b_run})
            if result["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                break
            assert time.monotonic() < deadline, "B did not finish its normal cleanup"
            await asyncio.sleep(0.1)
        assert result["status"] == "succeeded"
        assert await asyncio.to_thread(_inspect, b_record.container_id) is None
        evidence["B_normal_cleanup"] = True
        evidence["passed"] = True
    except BaseException as exc:
        evidence["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Teardown evidence never turns a failed production recovery into a pass.
        for core in reversed(cores):
            try:
                await core.stop()
            except Exception as exc:
                evidence["errors"].append(f"Core teardown: {type(exc).__name__}: {exc}")
                evidence["passed"] = False
        # Also retain identities persisted before _submit could finish observing them.
        recorded = {record.resource_id: record for record in records}
        for core in cores:
            for path in (core.home / "control" / "sandbox-resources").glob("*.json"):
                try:
                    record = SandboxResourceRecord.model_validate_json(path.read_bytes())
                    if record.container_id is not None:
                        recorded[record.resource_id] = record
                    else:
                        evidence["errors"].append(f"unresolved create intent: {record.resource_id}")
                        evidence["passed"] = False
                except Exception as exc:
                    evidence["errors"].append(f"record read: {type(exc).__name__}: {exc}")
                    evidence["passed"] = False
        for record in recorded.values():
            assert record.container_id is not None
            try:
                engine = await asyncio.to_thread(_docker, "info", "--format", "{{.ID}}")
                assert engine.returncode == 0 and engine.stdout.strip() == record.engine_id
                inspected = await asyncio.to_thread(_inspect, record.container_id)
                if inspected is not None:
                    validate_container(record, inspected)
                    removed = await asyncio.to_thread(_docker, "rm", "--force", record.container_id)
                    assert removed.returncode == 0
                    assert await asyncio.to_thread(_inspect, record.container_id) is None
                    evidence["fallback_cleanup"].append(record.container_id)
                    evidence["passed"] = False
            except Exception as exc:
                evidence["errors"].append(f"Docker teardown: {type(exc).__name__}: {exc}")
                evidence["passed"] = False
        (root / "results.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        assert evidence["passed"], f"Recovery did not pass; inspect {root / 'results.json'}"
