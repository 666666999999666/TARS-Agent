"""Run the agreed local experiments; never sends requests without --execute."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from tars_agent.core.config import get_config
from tars_agent.core.control import CoreHomeLock
from tars_agent.core.eval.appworld import DEEPSEEK_PROFILE, load_deepseek_config
from tars_agent.core.eval.appworld_comparison import TRAIN_TASK_IDS, bind_shared_budget
from tars_agent.core.eval.models import EvalSuiteManifest
from tars_agent.core.eval.runner import run_eval_suite
from tars_agent.core.persistence.request_budget import RequestLedger

ROOT = Path(__file__).resolve().parents[1]
PAIRED_MODES = {"deepseek-train-paired": "train", "deepseek-dev-paired": "dev"}


def prepare_suite(mode: str) -> EvalSuiteManifest:
    if mode in PAIRED_MODES:
        name = f"appworld-deepseek-{PAIRED_MODES[mode]}-paired.json"
    else:
        name = "appworld-test-normal.json" if mode.startswith("appworld") else "internal-agent-tasks.json"
    suite = EvalSuiteManifest.model_validate_json((ROOT / "evals" / name).read_bytes())
    if mode in PAIRED_MODES:
        spec = suite.appworld
        if (suite.model_config_ref.reference != DEEPSEEK_PROFILE or spec is None
                or (spec.dataset, spec.workers, spec.max_steps, spec.task_timeout_s,
                    spec.infrastructure_retries, spec.task_limit, spec.paired_comparison)
                != (PAIRED_MODES[mode], 1, 60, 900, 1, None, True)
                or (spec.dataset == "train" and tuple(spec.task_ids or []) != TRAIN_TASK_IDS)
                or (spec.dataset == "dev" and spec.task_ids is not None)):
            raise ValueError("DeepSeek manifest differs from the fixed paired comparison")
    elif mode == "internal-dev":
        suite.tasks = [task for task in suite.tasks if task.metadata.get("split") == "development"]
        suite.default_repetitions = 1
        suite.suite_id += "-development"
    elif mode == "appworld-dev":
        assert suite.appworld is not None
        suite.appworld = suite.appworld.model_copy(update={"dataset": "dev", "task_limit": 4})
        suite.suite_id += "-development"
    elif mode == "appworld-train":
        assert suite.appworld is not None
        suite.appworld = suite.appworld.model_copy(update={"dataset": "train", "task_limit": 4})
        suite.suite_id += "-train-readiness"
    return suite


async def execute(mode: str, output: Path, resume: bool, batch_root: Path | None = None) -> int:
    if mode == "cli":
        if resume:
            raise ValueError("CLI acceptance starts fresh rounds; resume is not supported")
        return await execute_cli(output)
    suite = prepare_suite(mode)
    if output.exists() and not resume:
        raise ValueError("Output already exists; choose a new directory to preserve prior evidence")
    deepseek_profile = mode in PAIRED_MODES
    if resume and not (mode.startswith("appworld") or deepseek_profile):
        raise ValueError("Only the checkpointed AppWorld adapter supports resume")
    if deepseek_profile:
        if batch_root is None:
            raise ValueError("paired comparison requires --batch-root shared by train and dev")
        batch_root = batch_root.resolve()
        if output.resolve() != batch_root / PAIRED_MODES[mode]:
            raise ValueError("paired evidence must be --batch-root/train or --batch-root/dev")
        config = load_deepseek_config()
        assert suite.appworld is not None
        config.agent.max_steps = suite.appworld.max_steps
    else:
        # Keep the trusted original ledger/credentials; overrides live only in this experiment process.
        os.environ["TARS_LLM_CONTEXT_BUDGET_TOKENS"] = "131072"
        os.environ["TARS_COMPACT_THRESHOLD"] = "0"
        config = get_config()
    config.compaction.auto_threshold = 0
    ledger = RequestLedger(config.llm.request_budget_path, limit=config.llm.request_limit)
    before = ledger.counts()["real"]
    if suite.adapter == "internal":
        allowance = sum(
            (task.repetitions or suite.default_repetitions) * config.agent.max_steps
            * config.llm.attempts * (4 if task.agent_mode == "orchestrated" else 1)
            for task in suite.tasks
        )
        calculated = before + allowance
        config.llm.request_limit = (calculated if config.llm.request_limit is None else
                                    min(config.llm.request_limit, calculated))
    if mode in {"internal", "appworld"} or deepseek_profile:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8",
        )
        if dirty.strip():
            raise ValueError("Commit the frozen candidate locally before the formal experiment")
    if deepseek_profile:
        assert batch_root is not None
        owner = CoreHomeLock(batch_root / "batch-controller-state")
        owner.acquire()
        try:
            bind_shared_budget(batch_root, output, PAIRED_MODES[mode], config, ROOT)
            return await _execute_prepared(suite, output, resume, config, ledger, before)
        finally:
            owner.release()
    return await _execute_prepared(suite, output, resume, config, ledger, before)


async def _execute_prepared(suite, output, resume, config, ledger, before) -> int:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "suite.json"
    serialized = suite.model_dump_json(indent=2)
    if resume and path.read_text(encoding="utf-8") != serialized:
        raise ValueError("Resume suite differs from the frozen suite")
    if not resume:
        path.write_text(serialized, encoding="utf-8")
    result = await run_eval_suite(path, output / "evidence", config, repository_root=ROOT)
    print(json.dumps({"summary": result.summary.model_dump(mode="json"),
                      "benchmark": result.benchmark,
                      "ledger_delta": ledger.counts()["real"] - before,
                      "artifact": str(output / "evidence" / "result.json")}, ensure_ascii=False))
    if result.adapter == "appworld":
        return 0 if result.benchmark.get("complete") is True else 1
    return int(bool(result.summary.failed or result.summary.errors or result.summary.skipped))


async def execute_cli(output: Path) -> int:
    # A directly executed file has scripts/, not the repository, on sys.path.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.acceptance_cli import CliAcceptance

    if output.exists():
        raise ValueError("CLI evidence directory must be new")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8",
    )
    if dirty.strip():
        raise ValueError("Commit the frozen candidate before formal CLI acceptance")
    os.environ["TARS_LLM_CONTEXT_BUDGET_TOKENS"] = "131072"
    os.environ["TARS_COMPACT_THRESHOLD"] = "0"
    os.environ["TARS_MAX_STEPS"] = "20"
    config = get_config()
    ledger = RequestLedger(config.llm.request_budget_path, limit=config.llm.request_limit)
    before = ledger.counts()["real"]
    calculated = before + 3 * 11 * 20 * config.llm.attempts
    limit = calculated if config.llm.request_limit is None else min(calculated, config.llm.request_limit)
    os.environ["TARS_LLM_REQUEST_LIMIT"] = str(limit)
    output.mkdir(parents=True)
    rounds = []
    for number in range(1, 4):
        harness = CliAcceptance(output / f"round-{number}")
        code = await harness.execute()
        rounds.append({"round": number, "exit_code": code,
                       "manifest": str(harness.root / "manifest.json")})
        (output / "rounds.json").write_text(json.dumps({
            "expected_rounds": 3, "expected_scenario_types": 6,
            "request_limit": limit, "ledger_before": before,
            "rounds": rounds, "complete": len(rounds) == 3 and all(r["exit_code"] == 0 for r in rounds),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if code:
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["internal-dev", "internal", "appworld-train", "appworld-dev", "appworld", "cli", *PAIRED_MODES])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-root", type=Path)
    args = parser.parse_args()
    if not args.execute:
        if args.mode == "cli":
            print(json.dumps({"mode": "prepare_only", "rounds": 3, "scenario_types": 6}))
            return 0
        suite = prepare_suite(args.mode)
        print(suite.model_dump_json(indent=2))
        return 0
    return asyncio.run(execute(args.mode, args.output.resolve(), args.resume, args.batch_root))


if __name__ == "__main__":
    raise SystemExit(main())
