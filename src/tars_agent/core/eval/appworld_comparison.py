"""Frozen task selection, shared budgets and reporting for one completion-prompt A/B."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tars_agent.core.config import TarsConfig
from tars_agent.core.eval.appworld import config_fingerprint, write_json
from tars_agent.core.eval.appworld_prompts import COMPARISON_ID, template_sha256
from tars_agent.core.eval.models import AppWorldSpec
from tars_agent.core.eval.provenance import compute_tree_digest
from tars_agent.core.persistence.cost_budget import CostLedger
from tars_agent.core.persistence.request_budget import (
    RequestLedger,
    validate_existing_request_ledger,
)

TRAIN_SCENARIOS = ("692c77d", "22cc237", "27e1026", "76f2c72")
TRAIN_TASK_IDS = tuple(sorted(f"{scenario}_{variant}" for scenario in TRAIN_SCENARIOS
                              for variant in (1, 2, 3)))
REQUEST_ALLOWANCE = (12 + 57) * 2 * 60 * 2 * 2


def validate_comparison_selection(
    spec: AppWorldSpec, selected: list[str], metadata: dict[str, Any],
) -> None:
    if spec.dataset == "train":
        if tuple(selected) != TRAIN_TASK_IDS:
            raise ValueError("paired train must use the twelve frozen tasks")
    elif (spec.dataset != "dev" or len(selected) != 57
          or selected != sorted(metadata["task_ids"])):
        raise ValueError("paired dev must use the complete official 57-task split")
    groups: dict[str, list[int]] = {}
    for task_id in selected:
        scenario, number = metadata["scenarios"][task_id]
        groups.setdefault(scenario, []).append(number)
    expected = 4 if spec.dataset == "train" else 19
    if len(groups) != expected or any(sorted(numbers) != [1, 2, 3] for numbers in groups.values()):
        raise ValueError("paired comparison requires every variant of each selected scenario")


def bind_shared_budget(
    batch_root: Path, output: Path, dataset: str, config: TarsConfig, repository_root: Path,
) -> dict[str, Any]:
    """Called under the batch owner lock. Resume can never grant a new request allowance."""
    if dataset not in {"train", "dev"}:
        raise ValueError("comparison has only train and dev stages")
    if config.llm.cost_budget_path is None:
        raise ValueError("paired comparison requires its already initialized 50-CNY cost ledger")
    ledger_path = validate_existing_request_ledger(config.llm.request_budget_path)
    counts = RequestLedger(ledger_path).counts()["real"]
    identity = ledger_path.stat()
    cost = CostLedger(config.llm.cost_budget_path).summary()
    base = copy.deepcopy(config)
    base.llm.request_limit = None
    binding = {
        "comparison": COMPARISON_ID,
        "tree_digest": compute_tree_digest(repository_root),
        "config_fingerprint_without_request_limit": config_fingerprint(base),
        "request_ledger_identity": [str(ledger_path), identity.st_dev, identity.st_ino],
        "cost_budget_id": cost["budget_id"],
        "cost_policy": cost["policy"],
        "train_task_ids": list(TRAIN_TASK_IDS),
        "dev_task_count": 57,
        "request_allowance": REQUEST_ALLOWANCE,
        "prompt_templates": {variant: template_sha256(variant) for variant in ("A", "B")},
    }
    contract_path = batch_root / "batch.json"
    sentinel = batch_root / "batch.identity.json"
    record: dict[str, Any]
    if contract_path.exists():
        record = json.loads(contract_path.read_text(encoding="utf-8"))
        if record["binding"] != binding or not sentinel.is_file():
            raise ValueError("paired batch inputs changed or its identity is missing")
        seal = json.loads(sentinel.read_text(encoding="utf-8"))
        if (seal["initial_requests"] != record["initial_requests"]
                or seal["request_limit"] != record["request_limit"]
                or counts < record["request_high_water"]):
            raise ValueError("paired batch request budget changed or was rolled back")
        if (config.llm.request_limit is not None
                and config.llm.request_limit < record["request_limit"]):
            raise ValueError("trusted request limit is lower than this frozen batch")
    else:
        if dataset == "dev":
            raise ValueError("complete the paired train stage before starting dev")
        if sentinel.exists() or output.exists():
            raise ValueError("paired batch lost its request budget; refusing a fresh allowance")
        limit = counts + REQUEST_ALLOWANCE
        if config.llm.request_limit is not None:
            limit = min(limit, config.llm.request_limit)
        if limit <= counts:
            raise ValueError("paired batch has no remaining request allowance")
        record = {
            "binding": binding, "initial_requests": counts, "request_limit": limit,
            "request_high_water": counts, "stages": {},
        }
        # A partial initialization blocks recovery instead of silently opening another budget.
        write_json(sentinel, {"initial_requests": counts, "request_limit": limit})
    expected_output = str(output.resolve())
    if dataset == "dev":
        _require_completed_train(record, config)
    previous = record["stages"].get(dataset)
    if previous is not None and previous != expected_output:
        raise ValueError("paired stage is already bound to a different evidence directory")
    record["stages"][dataset] = expected_output
    record["request_high_water"] = counts
    write_json(contract_path, record)
    config.llm.request_limit = int(record["request_limit"])
    return record


def _require_completed_train(record: dict[str, Any], config: TarsConfig) -> None:
    train_output = record["stages"].get("train")
    if train_output is None:
        raise ValueError("complete the paired train stage before starting dev")
    evidence = Path(train_output) / "evidence"
    try:
        result = json.loads((evidence / "result.json").read_text(encoding="utf-8"))
        state = json.loads((evidence / "checkpoint.json").read_text(encoding="utf-8"))
        frozen = json.loads((evidence / "freeze.json").read_text(encoding="utf-8"))
        expected = copy.deepcopy(config)
        expected.llm.request_limit = record["request_limit"]
        benchmark = result["benchmark"]
        expected_keys = {f"{variant}:{task_id}" for task_id in TRAIN_TASK_IDS
                         for variant in ("A", "B")}
        valid = (
            benchmark["complete"] is True and not benchmark["errors"]
            and benchmark["dataset"] == "train"
            and benchmark["expected_tasks"] == benchmark["completed_tasks"] == 24
            and result["run_id"] == state["run_id"] and set(state["results"]) == expected_keys
            and result["selected_task_ids"] == list(TRAIN_TASK_IDS)
            and frozen == state["binding"]
            and frozen["manifest"]["appworld"]["paired_comparison"] is True
            and frozen["manifest"]["appworld"]["dataset"] == "train"
            and frozen["manifest"]["appworld"]["task_ids"] == list(TRAIN_TASK_IDS)
            and frozen["tree_digest"] == record["binding"]["tree_digest"]
            and frozen["base_config_fingerprint"] == config_fingerprint(expected)
        )
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError("paired train must be complete and match this frozen batch before dev")


def failure_category(requirement: str, label: str | None = None) -> str:
    normalized = " ".join(requirement.strip().lower().split()).rstrip(".")
    if normalized == "assert answers match":
        return "answer_contract"
    if normalized.startswith("assert model changes match "):
        return "mixed_change_boundary"
    if normalized == "assert no model changes" or label == "no_op_pass":
        return "preservation_check"
    return "unclassified_requirement"


def _task_details(
    raw: dict[str, Any], selected: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    individual = raw.get("individual")
    if not isinstance(individual, dict) or set(individual) != set(selected):
        raise ValueError("official paired scores do not cover exactly the frozen tasks")
    rows = []
    categories: Counter[str] = Counter()
    for task_id in selected:
        result = individual[task_id]
        if type(result.get("success")) is not bool:
            raise ValueError("official task success is missing")
        failures = result.get("failures")
        if not isinstance(failures, list) or not isinstance(result.get("passes"), list):
            raise ValueError("official detailed requirement checks are missing")
        failed_checks = []
        for check in failures:
            requirement = check.get("requirement")
            if not isinstance(requirement, str):
                raise ValueError("official failed requirement is missing")
            label = check.get("label")
            category = failure_category(requirement, label)
            categories[category] += 1
            failed_checks.append({"requirement": requirement, "label": label, "category": category})
        rows.append({"task_id": task_id, "success": result["success"],
                     "failed_checks": failed_checks})
    unknown = categories["unclassified_requirement"] + categories["mixed_change_boundary"]
    return rows, {
        "failed_check_categories": dict(categories),
        "collateral_failure_count": None if unknown else categories["preservation_check"],
        "collateral_count_note": "unclassified failures require review" if unknown else None,
    }


def comparison_summary(
    metrics: dict[str, dict[str, Any]], selected: list[str], attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in ("A", "B"):
        rows, checks = _task_details(metrics[variant], selected)
        variant_attempts = [item for item in attempts
                            if item["evaluation"].get("prompt_variant") == variant]
        usage_complete = all(item["evaluation"].get("usage_complete") is True
                             for item in variant_attempts)
        token_names = ("input_tokens", "output_tokens", "cache_read_input_tokens",
                       "cache_creation_input_tokens")
        variants[variant] = {
            "official_metrics": metrics[variant]["aggregate"],
            "tasks": rows, **checks,
            "runtime_terminal_states": dict(Counter(item.get("run_terminal_status") or "unknown"
                                                     for item in variant_attempts)),
            "all_attempts": len(variant_attempts),
            "latency_ms": sum(item["latency_ms"] for item in variant_attempts),
            "http_requests": sum((item["evaluation"].get("model_requests") or {}).get("reserved", 0)
                                 for item in variant_attempts),
            "usage_complete": usage_complete,
            "confirmed_usage_totals": {
                name: sum((item["evaluation"].get("confirmed_usage") or {}).get(name, 0) or 0
                          for item in variant_attempts)
                for name in token_names
            },
            "usage": {name: sum(item["usage"].get(name, 0) or 0 for item in variant_attempts)
                      if usage_complete else None for name in token_names},
        }
    paired = []
    for index, task_id in enumerate(selected):
        a, b = (variants[variant]["tasks"][index]["success"] for variant in ("A", "B"))
        outcome = ("both_pass" if a and b else "both_fail" if not a and not b
                   else "B_win" if b else "A_win")
        paired.append({"task_id": task_id, "A_success": a, "B_success": b, "outcome": outcome})
    a, b = (variants[variant] for variant in ("A", "B"))
    known_collateral = (a["collateral_failure_count"] is not None
                        and b["collateral_failure_count"] is not None)
    better = (b["official_metrics"]["task_goal_completion"]
              > a["official_metrics"]["task_goal_completion"])
    eligible = bool(known_collateral and better
                    and b["collateral_failure_count"] <= a["collateral_failure_count"])
    return {
        "comparison": COMPARISON_ID, "variants": variants, "paired_tasks": paired,
        "outcomes": dict(Counter(row["outcome"] for row in paired)),
        "B_eligible_pending_engineering_acceptance": eligible,
        "selection_note": ("B additionally requires final engineering acceptance" if eligible else
                           "retain A; no demonstrated eligible B improvement"),
        "repetitions": 1,
        "interpretation": "one paired observation; not a stable average improvement estimate",
        "details_sha256": hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest(),
    }
