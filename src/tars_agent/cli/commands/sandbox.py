from __future__ import annotations

import asyncio
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path

from tars_agent.core.config import TarsConfig
from tars_agent.core.tools.runtime import DockerRuntime


def cmd_sandbox_build(config: TarsConfig) -> None:
    resources = files("tars_agent.sandbox")
    with tempfile.TemporaryDirectory(prefix="tars-sandbox-build-") as directory:
        root = Path(directory)
        for name in ("Dockerfile", "worker.py"):
            (root / name).write_bytes(resources.joinpath(name).read_bytes())
        completed = subprocess.run(
            [config.sandbox.docker_binary, "build", "--file", str(root / "Dockerfile"),
             "--tag", config.sandbox.image, str(root)],
            check=False,
        )
    raise SystemExit(completed.returncode)



async def _doctor(config: TarsConfig) -> int:
    runtime = DockerRuntime(config.sandbox)
    status = await runtime.preflight()
    print(f"backend: {status.backend}")
    print(f"image: {config.sandbox.image}")
    print(f"available: {'yes' if status.available else 'no'}")
    if status.reason:
        print(f"reason: {status.reason}")
    for key, value in status.details.items():
        print(f"{key}: {value}")
    return 0 if status.available else 1


def cmd_sandbox_doctor(config: TarsConfig) -> None:
    raise SystemExit(asyncio.run(_doctor(config)))


__all__ = ["cmd_sandbox_build", "cmd_sandbox_doctor"]
