from __future__ import annotations

import sys

from tars_agent.cli.client import run_client
from tars_agent.core.config import TarsConfig


def cmd_run(goal: str, config: TarsConfig) -> None:
    sys.exit(run_client(config, goal=goal))
