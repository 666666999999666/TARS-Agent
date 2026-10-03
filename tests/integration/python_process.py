"""Launch a real Python process while retaining the test environment's dependencies."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path


def python_module_command(module: str, *arguments: str) -> list[str]:
    # POSIX venv launchers may be symlinks: keep their path so Python finds pyvenv.cfg.
    interpreter = sys.executable
    package_bootstrap = ""
    if sys.platform == "win32":
        # uv's Windows venv python.exe is a parent trampoline. A direct interpreter
        # gives Popen ownership of the actual process handle, including during teardown.
        interpreter = str(Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True))
        packages = str(Path(sysconfig.get_path("purelib")).resolve(strict=True))
        # Prefer the active environment even during imports triggered by its .pth files.
        package_bootstrap = (
            f"sys.path.insert(0, {packages!r}); "
            f"site.addsitedir({packages!r}); "
        )
    project = Path(__file__).resolve().parents[2]
    bootstrap = (
        "import runpy,site,sys; "
        f"{package_bootstrap}"
        f"sys.path.insert(0, {str(project)!r}); "
        f"sys.argv = {[module, *arguments]!r}; "
        f"runpy.run_module({module!r}, run_name='__main__')"
    )
    return [interpreter, "-I", "-c", bootstrap]
