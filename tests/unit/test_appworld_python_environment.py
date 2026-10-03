from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tars_agent.core.eval import appworld


@pytest.mark.parametrize("platform", ["posix", "nt"])
def test_worker_launcher_selects_environment_entrypoint_by_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str,
) -> None:
    base = tmp_path / "base-python"
    entrypoint = tmp_path / "venv-python"
    base.write_bytes(b"base interpreter")
    entrypoint.write_bytes(b"environment entrypoint")
    packages = tmp_path / "site-packages"
    packages.mkdir()
    monkeypatch.setattr(appworld, "os", SimpleNamespace(name=platform))
    monkeypatch.setattr(
        appworld, "sys", SimpleNamespace(executable=str(entrypoint), _base_executable=str(base)),
    )
    distribution = SimpleNamespace(locate_file=lambda _name: packages)
    monkeypatch.setattr(appworld.importlib.metadata, "distribution", lambda _name: distribution)
    environment = appworld.python_environment()
    assert environment["base_executable"] == str(base if platform == "nt" else entrypoint)
    assert environment["site_packages"] == str(packages)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink venv semantics require POSIX")
def test_appworld_module_launcher_keeps_venv_prefix_and_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", "--symlinks", str(venv)],
        check=True, capture_output=True, timeout=30,
    )
    entrypoint = venv / "bin/python"
    packages = Path(subprocess.check_output(
        [str(entrypoint), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True, timeout=10,
    ).strip())
    (packages / "appworld_launcher_probe.py").write_text(
        "import json, sys\nprint(json.dumps({'prefix': sys.prefix, 'marker': 'venv-only'}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        appworld, "sys", SimpleNamespace(
            executable=str(entrypoint), _base_executable=getattr(sys, "_base_executable"),
        ),
    )
    distribution = SimpleNamespace(locate_file=lambda _name: packages)
    monkeypatch.setattr(appworld.importlib.metadata, "distribution", lambda _name: distribution)
    environment = appworld.python_environment()
    argv = appworld.module_argv("appworld_launcher_probe", [], environment)
    reply = json.loads(subprocess.check_output(argv, cwd=tmp_path, text=True, timeout=10))
    assert Path(reply["prefix"]) == venv
    assert reply["marker"] == "venv-only"
