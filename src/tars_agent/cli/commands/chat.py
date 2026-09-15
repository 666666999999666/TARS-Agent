from __future__ import annotations

import sys

from tars_agent.cli.client import run_client
from tars_agent.core.config import TarsConfig


def cmd_chat(config: TarsConfig, *, resume_session_id: str | None = None) -> None:
    sys.exit(run_client(config, resume_session_id=resume_session_id))
