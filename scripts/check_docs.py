"""Check required project documentation and local links."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("README.md", "RUNBOOK.md", "WIRE_PROTOCOL.md", "LICENSE",
            "docs/baseline/VERIFICATION_SUMMARY.md", "docs/baseline/ACCEPTANCE.md",
            "docs/baseline/ARCHITECTURE.md", "docs/baseline/DEMOS.md",
            "docs/baseline/LIMITATIONS.md", "docs/baseline/MODEL_SETUP.md",
            "docs/baseline/REAL_WORKFLOWS.md", "scripts/FINAL_DELIVERY.md")


def main() -> int:
    failures: list[str] = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            failures.append(f"missing {name}")
    for path in [ROOT / name for name in REQUIRED if name.endswith(".md")]:
        if not path.is_file():
            continue
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            local = target.split("#", 1)[0].strip("<>")
            if local and not (path.parent / local).exists():
                failures.append(f"broken link in {path.relative_to(ROOT)}: {local}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
