from __future__ import annotations

import os
from pathlib import Path


def tars_home() -> Path:
    """Return the canonical local state root, allowing isolated test homes."""

    configured = os.environ.get("TARS_HOME", "~/.tars-baseline")
    return Path(configured).expanduser().resolve()


__all__ = ["tars_home"]
