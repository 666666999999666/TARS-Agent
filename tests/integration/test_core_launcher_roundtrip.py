from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tars_agent.core.control import read_control_file


@pytest.mark.timeout(90)
def test_launcher_verifies_ready_and_releases_control_on_stop(tmp_path: Path, free_port: int) -> None:
    """Exercise the installed CLI outside the source directory with no model call."""
    home = tmp_path / "runtime-home"
    env = os.environ.copy()
    env.update({
        "TARS_HOME": str(home), "TARS_CONFIG": str(home / "config.toml"),
        "TARS_HOST": "127.0.0.1", "TARS_PORT": str(free_port),
        "TARS_SANDBOX_MODE": "preferred", "TARS_TRACE_ENABLED": "false",
    })
    control_path = home / "control" / f"tars-core-{free_port}.json"
    def invoke(action: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "tars_agent.cli", "core", action],
            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
            timeout=65 if action == "start" else 25,
        )
    try:
        started = invoke("start")
        assert started.returncode == 0, started.stderr + started.stdout
        control = read_control_file(control_path)
        assert control is not None and control.launch_id
        assert f"pid={control.pid}" in started.stdout
        status = invoke("status")
        assert status.returncode == 0 and "running" in status.stdout
        stopped = invoke("stop")
        assert stopped.returncode == 0, stopped.stderr + stopped.stdout
        assert not control_path.exists()
    finally:
        if control_path.exists():
            invoke("stop")


def test_core_help_exits_without_initializing_home(tmp_path: Path) -> None:
    home = tmp_path / "must-not-be-created"
    env = os.environ.copy()
    env["TARS_HOME"] = str(home)
    env["TARS_CONFIG"] = str(home / "config.toml")
    help_result = subprocess.run(
        [sys.executable, "-m", "tars_agent.core", "--help"],
        env=env, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=10,
    )
    assert help_result.returncode == 0
    assert "tars-core" in help_result.stdout
    assert not home.exists()
