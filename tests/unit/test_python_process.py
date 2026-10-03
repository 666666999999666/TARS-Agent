from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest
import typing_extensions

from tests.integration import python_process


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_posix_command_keeps_virtual_environment_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str,
) -> None:
    launcher = tmp_path / "venv" / "bin" / "python"
    native = tmp_path / "native-python"
    native.write_text("native interpreter fixture", encoding="utf-8")
    original_resolve = Path.resolve

    def resolve(path: Path, strict: bool = False) -> Path:
        # Model a POSIX venv symlink on platforms that cannot create one.
        if path == launcher:
            return original_resolve(native, strict=strict)
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(
        python_process, "sys",
        SimpleNamespace(platform=platform, executable=str(launcher), _base_executable=str(native)),
    )
    command = python_process.python_module_command("tests.integration.offline_core")
    assert command[0] == str(launcher)
    assert command[1:3] == ["-I", "-c"]


def test_windows_command_uses_native_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = tmp_path / "native-python.exe"
    native.write_text("native interpreter fixture", encoding="utf-8")
    monkeypatch.setattr(
        python_process, "sys",
        SimpleNamespace(platform="win32", executable="venv-python.exe", _base_executable=str(native)),
    )
    command = python_process.python_module_command("tests.integration.offline_core")
    assert command[0] == str(native.resolve())
    assert command[1:3] == ["-I", "-c"]


def test_windows_bootstrap_prefers_environment_packages_before_pth_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_packages, system_packages = tmp_path / "environment", tmp_path / "system"
    environment_packages.mkdir()
    system_packages.mkdir()
    for packages, value in ((environment_packages, "environment"), (system_packages, "system")):
        (packages / "_tars_dependency_probe.py").write_text(
            f"VALUE = {value!r}\n", encoding="utf-8",
        )
    (environment_packages / "priority.pth").write_text(
        "import _tars_dependency_probe; assert _tars_dependency_probe.VALUE == 'environment'\n",
        encoding="utf-8",
    )
    (environment_packages / "_tars_environment_probe.py").write_text(
        "import json, sys, _tars_dependency_probe as dependency\n"
        "print(json.dumps({'value': dependency.VALUE, 'argv': sys.argv}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        python_process, "sys",
        SimpleNamespace(
            platform="win32", executable=sys.executable,
            _base_executable=getattr(sys, "_base_executable", sys.executable),
        ),
    )
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(environment_packages))
    arguments = ("two words", "quotes' and \"double\"", "back\\slash")
    command = python_process.python_module_command("_tars_environment_probe", *arguments)
    # Reproduce a pre-existing system package ahead of a manually added venv path.
    command[-1] = f"import sys; sys.path.insert(0, {str(system_packages)!r}); " + command[-1]
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=True, timeout=20,
    )
    observed = json.loads(result.stdout)
    assert observed["value"] == "environment"
    assert observed["argv"][1:] == list(arguments)
    assert "AssertionError" not in result.stderr


def test_module_command_retains_dependencies_and_process_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(python_process, "__file__", str(project / "tests/integration/python_process.py"))
    (project / "_tars_process_probe.py").write_text(
        "import json, os, sys, pydantic_core, typing_extensions\n"
        "from typing_extensions import Sentinel\n"
        "from mcp.server import MCPServer\n"
        "print(json.dumps({'pid': os.getpid(), 'prefix': sys.prefix, "
        "'typing_extensions': typing_extensions.__file__, 'argv': sys.argv}))\n",
        encoding="utf-8",
    )
    arguments = ("two words", "quotes' and \"double\"")
    with subprocess.Popen(
        python_process.python_module_command("_tars_process_probe", *arguments),
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
            raise
        assert process.returncode == 0, stderr
        observed = json.loads(stdout)
        assert observed["pid"] == process.pid
    assert observed["argv"][1:] == list(arguments)
    assert Path(observed["typing_extensions"]).resolve() == Path(typing_extensions.__file__).resolve()
    if os.name == "posix":
        assert Path(observed["prefix"]).resolve() == Path(sys.prefix).resolve()


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink-based virtual environment")
def test_posix_command_preserves_real_virtual_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, environment_root = tmp_path / "project", tmp_path / "venv"
    project.mkdir()
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment_root)
    launcher = environment_root / "bin" / "python"
    assert launcher.is_symlink()
    layout = subprocess.run(
        [str(launcher), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        cwd=project, capture_output=True, text=True, check=True, timeout=20,
    )
    packages = Path(layout.stdout.strip())
    (packages / "_tars_venv_only_dependency.py").write_text(
        'VALUE = "venv-only-package"\n', encoding="utf-8",
    )
    (project / "_tars_venv_probe.py").write_text(
        "import json, sys, _tars_venv_only_dependency as dependency\n"
        "print(json.dumps({'prefix': sys.prefix, 'marker': dependency.VALUE}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(python_process, "__file__", str(project / "tests/integration/python_process.py"))
    monkeypatch.setattr(
        python_process, "sys",
        SimpleNamespace(platform=sys.platform, executable=str(launcher), _base_executable=str(launcher.resolve())),
    )
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(packages))
    result = subprocess.run(
        python_process.python_module_command("_tars_venv_probe"),
        cwd=project, capture_output=True, text=True, check=True, timeout=20,
    )
    observed = json.loads(result.stdout)
    assert Path(observed["prefix"]).resolve() == environment_root.resolve()
    assert observed["marker"] == "venv-only-package"
