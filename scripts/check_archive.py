"""Check committed resources without executing or installing an archive."""
from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

REQUIRED = {
    "tars_agent/__init__.py", "tars_agent/py.typed",
    "tars_agent/core/persistence/alembic/env.py",
    "tars_agent/core/persistence/alembic/script.py.mako",
    "tars_agent/core/persistence/alembic/versions/0001_initial_state.py",
    "tars_agent/sandbox/worker.py", "tars_agent/sandbox/Dockerfile",
    "tars_agent/web/static/index.html", "tars_agent/core/agents/builtin/executor.toml",
    "tars_agent/core/skills/builtin/init.md",
}


def inspect(path: Path) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        prefix = ""
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            names = {member.name for member in archive.getmembers()}
        roots = {name.split("/", 1)[0] for name in names}
        if len(roots) != 1:
            raise ValueError("sdist must contain one root")
        root = roots.pop()
        prefix = root + "/src/"
        for required in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
            if root + "/" + required not in names:
                raise ValueError(f"sdist missing {required}")
    else:
        raise ValueError(f"unsupported archive: {path.name}")
    missing = sorted(prefix + name for name in REQUIRED if prefix + name not in names)
    if missing:
        raise ValueError("missing package resources: " + ", ".join(missing))
    for name in names:
        parts = Path(name).parts
        if Path(name).is_absolute() or ".." in parts:
            raise ValueError("unsafe archive path")
        if any(part in {".git", ".venv", "node_modules", ".env", ".kama", ".tars-baseline", "__pycache__"} for part in parts):
            raise ValueError(f"private/generated path included: {name}")
    print(f"{path.name}: resource and path checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    for path in parser.parse_args().archives:
        inspect(path)


if __name__ == "__main__":
    main()
