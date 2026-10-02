"""Launch a real Python process while retaining the test environment's dependencies."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path


def python_module_command(module: str, *arguments: str) -> list[str]:
    # uv's Windows venv python.exe is a parent trampoline. A direct interpreter
    # gives Popen ownership of the actual process handle, including during teardown.
    interpreter = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    packages = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    project = Path(__file__).resolve().parents[2]
    bootstrap = (
        "import runpy,site,sys; "
        f"site.addsitedir({str(packages)!r}); "
        f"sys.path.insert(0, {str(project)!r}); "
        f"sys.argv = {[module, *arguments]!r}; "
        f"runpy.run_module({module!r}, run_name='__main__')"
    )
    return [str(interpreter), "-I", "-c", bootstrap]
