"""Reproducible QA entry point; missing tools and skipped required gates never pass."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "build" / "qa"
LINE_GATE, BRANCH_GATE, COMBINED_GATE, DIFF_GATE = 65.0, 49.0, 62.0, 80.0


class QaError(RuntimeError):
    pass


def run(command: list[str], *, label: str) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    (ARTIFACT_ROOT / f"{label}.log").write_bytes(result.stdout + result.stderr)
    print(f"{label}: {'passed' if result.returncode == 0 else 'failed'} ({result.returncode})")
    if result.returncode:
        raise QaError(f"{label} failed; see build/qa/{label}.log")


def python(*args: str) -> list[str]:
    return [sys.executable, *args]


def require(*modules: str) -> None:
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise QaError("Missing required dependencies: " + ", ".join(missing)
                      + "; user preparation: uv sync --locked --group qa --group security")


def fixed_base(requested: str | None = None) -> str:
    checked = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=ROOT,
                             capture_output=True, check=False)
    roots = checked.stdout.decode("ascii").splitlines()
    if checked.returncode or len(roots) != 1:
        raise QaError("A single repository baseline is missing; clone the complete history")
    base = roots[0]
    if requested is not None and requested != base:
        raise QaError("--diff-base cannot replace the repository root commit")
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT,
                             capture_output=True, check=False)
    if shallow.returncode or shallow.stdout.strip() != b"false":
        raise QaError("Git coverage gates require the complete repository history")
    return base


def quick(base: str) -> None:
    require("ruff", "mypy", "yaml")
    run(["uv", "lock", "--check", "--offline"], label="lock")
    run(python("-m", "ruff", "check", "src", "tests", "scripts"), label="ruff")
    run(python("-m", "mypy", "src"), label="mypy")
    for name in ("gen_protocol_doc", "check_architecture", "check_docs", "check_workflows"):
        extra = ["--check"] if name == "gen_protocol_doc" else []
        run(python(f"scripts/{name}.py", *extra), label=name)
    run(["git", "diff", "--check", base], label="whitespace")


def tests(*, coverage: bool = False) -> None:
    require("pytest", "pytest_asyncio", "pytest_timeout", "yaml")
    command = python("-m", "pytest", "tests", "-m",
                     "not integration and not docker and not external and not anthropic",
                     "--timeout=60", "--junitxml=build/qa/junit.xml")
    if coverage:
        require("pytest_cov", "coverage", "diff_cover")
        command += ["--cov=tars_agent", "--cov-branch", "--cov-report=term",
                    "--cov-report=json:build/qa/coverage.json",
                    "--cov-report=xml:build/qa/coverage.xml"]
    run(command, label="coverage-tests" if coverage else "tests")


def check_coverage(base: str) -> None:
    totals = json.loads((ARTIFACT_ROOT / "coverage.json").read_text(encoding="utf-8"))["totals"]
    covered, lines = totals["covered_lines"], totals["num_statements"]
    branches, possible = totals["covered_branches"], totals["num_branches"]
    if not lines or not possible:
        raise QaError("Coverage report has no line or branch opportunities")
    values = {"line": covered / lines * 100, "branch": branches / possible * 100,
              "combined": (covered + branches) / (lines + possible) * 100}
    for kind, limit in (("line", LINE_GATE), ("branch", BRANCH_GATE), ("combined", COMBINED_GATE)):
        if values[kind] + 1e-9 < limit:
            raise QaError(f"{kind} coverage {values[kind]:.2f}% is below {limit}%")
    # Compare with the repository's initialization commit; never replace it with HEAD.
    dirty = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", "src"],
                           cwd=ROOT, capture_output=True, check=True)
    if dirty.stdout.strip():
        raise QaError("Stage or commit all new source before formal changed-line coverage")
    run([str(Path(sys.executable).with_name("diff-cover.exe" if os.name == "nt" else "diff-cover")),
         "build/qa/coverage.xml", f"--compare-branch={base}", f"--fail-under={DIFF_GATE:g}",
         "--format=json:build/qa/diff-coverage.json"], label="diff-coverage")
    (ARTIFACT_ROOT / "coverage-gates.json").write_text(json.dumps(values, indent=2), encoding="utf-8")


def build() -> None:
    require("build", "hatchling", "twine")
    # --no-isolation never installs build dependencies on the user's behalf.
    entry = Path(sys.executable).with_name("pyproject-build.exe" if os.name == "nt" else "pyproject-build")
    run([str(entry), "--no-isolation", "--sdist", "--wheel", "--outdir", "build/qa/dist"], label="build")
    archives = sorted((ARTIFACT_ROOT / "dist").glob("*"))
    artifacts = [str(p) for p in archives if p.suffix == ".whl" or p.name.endswith(".tar.gz")]
    if len(artifacts) != 2:
        raise QaError("Expected one wheel and one sdist; inspect build/qa/dist")
    run(python("-m", "twine", "check", *artifacts), label="metadata")
    run(python("scripts/check_archive.py", *artifacts), label="archive")


def security() -> None:
    require("bandit", "detect_secrets", "pip_audit")
    failures: list[str] = []
    jobs = [
        (python("-m", "bandit", "-c", "pyproject.toml", "-r", "src", "scripts", "-ll", "-ii",
                "-f", "json", "-o", "build/qa/bandit.json"), "bandit"),
        (python("scripts/check_public_content.py", "--history"), "public-history"),
        (python("-m", "pip_audit", "--local", "--format", "json", "--output", "build/qa/pip-audit.json"), "pip-audit"),
        (["npm.cmd" if os.name == "nt" else "npm", "--prefix", "web", "audit", "--audit-level=high", "--json"], "npm-audit"),
    ]
    for command, label in jobs:
        try:
            run(command, label=label)
        except (OSError, QaError) as exc:
            failures.append(str(exc))
    if failures:
        raise QaError("; ".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=["quick", "tests", "coverage", "build", "security", "full"])
    parser.add_argument("--diff-base")
    args = parser.parse_args()
    try:
        base = fixed_base(args.diff_base)
        if args.profile in {"quick", "full"}:
            quick(base)
        if args.profile == "tests":
            tests()
        if args.profile in {"coverage", "full"}:
            tests(coverage=True)
            check_coverage(base)
        if args.profile in {"build", "full"}:
            build()
        if args.profile in {"security", "full"}:
            security()
    except (OSError, ValueError, KeyError, QaError, subprocess.SubprocessError) as exc:
        print(f"QA failed: {exc}", file=sys.stderr)
        return 1
    print("Selected QA gates passed. Docker, real-model and manual TUI acceptance are separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
