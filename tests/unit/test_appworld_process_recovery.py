from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tars_agent.core.eval import appworld


def recorded_worker(
    directory: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[int], asyncio.Event, asyncio.Event]:
    directory.mkdir()
    argv = ["python", "owned-worker"]
    identity = {"pid": 12345, "started": "created", "argv": argv, "executable": "python"}
    appworld.write_json(directory / "process.json", {
        "phase": "running", "argv": argv, "identity": identity,
    })
    monkeypatch.setattr(appworld, "_worker_argv", lambda *args: argv)
    exited, signalled = asyncio.Event(), asyncio.Event()
    signals: list[int] = []

    def kill(pid: int, number: int) -> None:
        assert pid == identity["pid"]
        signals.append(number)
        signalled.set()

    monkeypatch.setattr(appworld, "os", SimpleNamespace(name="posix", kill=kill, fsync=os.fsync))
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(appworld, "process_identity", lambda pid: None if exited.is_set() else identity)
    return identity, signals, exited, signalled


async def test_recovery_does_not_retire_worker_before_term_exit_is_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "attempt"
    _, signals, exited, signalled = recorded_worker(directory, monkeypatch)
    stopping = asyncio.create_task(appworld.stop_recorded_worker(directory))
    try:
        await asyncio.wait_for(signalled.wait(), 2)
        assert signals == [signal.SIGTERM]
        assert json.loads((directory / "process.json").read_text())["phase"] == "running"
        assert not stopping.done()
        exited.set()
        await asyncio.wait_for(stopping, 2)
        assert json.loads((directory / "process.json").read_text())["phase"] == "retired"
    finally:
        exited.set()
        await stopping


async def test_recovery_waits_after_kill_and_refuses_to_retire_a_live_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "attempt"
    _, signals, _, _ = recorded_worker(directory, monkeypatch)
    monkeypatch.setattr(appworld, "_WORKER_EXIT_WAIT_S", 0.0)
    with pytest.raises(RuntimeError, match="exit was not confirmed"):
        await appworld.stop_recorded_worker(directory)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert json.loads((directory / "process.json").read_text())["phase"] == "running"


@pytest.mark.parametrize("identity_change", ["argv", "executable", "reused_pid"])
async def test_recovery_rechecks_identity_immediately_before_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity_change: str,
) -> None:
    directory = tmp_path / "attempt"
    identity, signals, _, _ = recorded_worker(directory, monkeypatch)
    monkeypatch.setattr(appworld, "_WORKER_EXIT_WAIT_S", 0.0)
    checks_after_term = 0

    def current_identity(pid: int) -> dict[str, Any]:
        nonlocal checks_after_term
        if signals:
            checks_after_term += 1
            if checks_after_term > 1:
                change = {"started": "reused"} if identity_change == "reused_pid" else {
                    identity_change: ["foreign"] if identity_change == "argv" else "foreign",
                }
                return {**identity, **change}
        return identity

    monkeypatch.setattr(appworld, "process_identity", current_identity)
    if identity_change == "reused_pid":
        await appworld.stop_recorded_worker(directory)
    else:
        with pytest.raises(RuntimeError, match="identity changed"):
            await appworld.stop_recorded_worker(directory)
    assert signals == [signal.SIGTERM]  # Never signal a process with a different identity.
    phase = json.loads((directory / "process.json").read_text())["phase"]
    assert phase == ("retired" if identity_change == "reused_pid" else "running")


async def test_recovery_identity_query_failure_preserves_running_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "attempt"
    identity, signals, _, _ = recorded_worker(directory, monkeypatch)

    def inaccessible_after_term(pid: int) -> dict[str, Any]:
        if signals:
            raise PermissionError("cannot confirm worker identity")
        return identity

    monkeypatch.setattr(appworld, "process_identity", inaccessible_after_term)
    with pytest.raises(PermissionError, match="cannot confirm"):
        await appworld.stop_recorded_worker(directory)
    assert signals == [signal.SIGTERM]
    assert json.loads((directory / "process.json").read_text())["phase"] == "running"


@pytest.mark.skipif(sys.platform != "linux", reason="requires a real Linux process and /proc")
@pytest.mark.parametrize("ignore_term", [False, True])
async def test_linux_recovery_waits_for_real_worker_exit_without_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ignore_term: bool,
) -> None:
    directory = tmp_path / "attempt"
    directory.mkdir()
    code = (
        "import signal,time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + "print('ready',flush=True); time.sleep(60)"
    )
    argv = [sys.executable, "-c", code]
    process = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE)
    try:
        assert process.stdout is not None
        assert await asyncio.wait_for(process.stdout.readline(), 5) == b"ready\n"
        identity = await asyncio.to_thread(appworld.process_identity, process.pid)
        assert identity is not None and identity["argv"] == argv
        appworld.write_json(directory / "process.json", {
            "phase": "running", "argv": argv, "identity": identity,
        })
        monkeypatch.setattr(appworld, "_worker_argv", lambda *args: argv)
        monkeypatch.setattr(appworld, "_WORKER_EXIT_WAIT_S", 0.2)
        await asyncio.wait_for(appworld.stop_recorded_worker(directory), 3)
        await asyncio.wait_for(process.wait(), 1)
        assert process.returncode == -(signal.SIGKILL if ignore_term else signal.SIGTERM)
        assert appworld.process_identity(process.pid) is None
        assert json.loads((directory / "process.json").read_text())["phase"] == "retired"
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
