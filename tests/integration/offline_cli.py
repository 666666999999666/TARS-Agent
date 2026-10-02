"""Exercise the production CLI launcher with only its child entrypoint substituted."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from tars_agent.cli.main import main

if __name__ == "__main__":
    original_popen = subprocess.Popen

    def start_offline_core(args: list[str], *positional: Any, **kwargs: Any) -> Any:
        assert args == [sys.executable, "-m", "tars_agent.core"]
        command = [sys.executable, str(Path(__file__).with_name("offline_core.py"))]
        return original_popen(command, *positional, **kwargs)

    subprocess.Popen = start_offline_core  # type: ignore[assignment,misc]
    main()
