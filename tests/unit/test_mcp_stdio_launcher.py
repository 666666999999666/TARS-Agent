from __future__ import annotations

import json
import os
import subprocess
import venv
from pathlib import Path

import pytest

from tars_agent.core.mcp.client import _resolve_stdio_command, _stdio_environment


@pytest.mark.parametrize("selection", ["absolute", "path"])
def test_launcher_keeps_invocation_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str,
) -> None:
    root = tmp_path.resolve()
    project, trusted = root / "project", root / "trusted"
    project.mkdir()
    trusted.mkdir()
    launcher = trusted / ("python.exe" if os.name == "nt" else "python")
    target = root / "base-python"
    for executable in (launcher, target):
        executable.write_text("launcher fixture", encoding="utf-8")
        executable.chmod(0o755)
    original_resolve = Path.resolve

    def resolve(path: Path, strict: bool = False) -> Path:
        # Simulate a launcher symlink even where creating symlinks is unavailable.
        return target if path == launcher else original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)
    command = str(launcher) if selection == "absolute" else launcher.name
    resolved = _resolve_stdio_command(
        command, environ={"PATH": str(trusted), "PATHEXT": ".EXE"}, cwd=project,
    )
    assert resolved == str(launcher)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink-based virtual environment")
@pytest.mark.parametrize("selection", ["absolute", "path"])
def test_resolved_launcher_preserves_venv_packages(tmp_path: Path, selection: str) -> None:
    project, environment_root = tmp_path / "project", tmp_path / "venv"
    project.mkdir()
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment_root)
    launcher = environment_root / "bin" / "python"
    assert launcher.is_symlink()
    environment = _stdio_environment({"PATH": str(launcher.parent)})
    layout = subprocess.run(
        [str(launcher), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        cwd=project, env=environment, capture_output=True, text=True, check=True, timeout=20,
    )
    site_packages = Path(layout.stdout.strip())
    site_packages.mkdir(parents=True, exist_ok=True)
    (site_packages / "_tars_mcp_venv_probe.py").write_text(
        'VALUE = "venv-only-package"\n', encoding="utf-8",
    )
    command = str(launcher) if selection == "absolute" else "python"
    resolved = _resolve_stdio_command(command, environ=environment, cwd=project)
    result = subprocess.run(
        [resolved, "-I", "-c",
         "import json, sys, _tars_mcp_venv_probe as probe; "
         "print(json.dumps({'prefix': sys.prefix, 'marker': probe.VALUE}))"],
        cwd=project, env=environment, capture_output=True, text=True, check=True, timeout=20,
    )
    observed = json.loads(result.stdout)
    assert Path(observed["prefix"]).resolve() == environment_root.resolve()
    assert observed["marker"] == "venv-only-package"


@pytest.mark.skipif(os.name != "posix", reason="POSIX executable symlink boundary")
def test_path_rejects_launcher_link_into_project(tmp_path: Path) -> None:
    project, trusted = tmp_path / "project", tmp_path / "trusted"
    project.mkdir()
    trusted.mkdir()
    target = project / "server"
    target.write_text("project executable fixture", encoding="utf-8")
    target.chmod(0o755)
    (trusted / "server").symlink_to(target)
    with pytest.raises(FileNotFoundError, match="trusted PATH"):
        _resolve_stdio_command("server", environ={"PATH": str(trusted)}, cwd=project)
