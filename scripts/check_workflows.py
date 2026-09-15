"""Check CI jobs without silently skipping a missing YAML dependency."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    paths = list((ROOT / ".github/workflows").glob("*.yml"))
    if not paths:
        raise ValueError("No CI workflow")
    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("jobs"), dict):
            raise ValueError(f"Invalid workflow {path.name}")
        for job in document["jobs"].values():
            if not isinstance(job, dict) or not job.get("timeout-minutes"):
                raise ValueError("Every job requires a bounded timeout")
        if "ANTHROPIC_API_KEY:" in path.read_text(encoding="utf-8"):
            raise ValueError("Default baseline CI must not invoke real models")
    print("Workflow structure passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
