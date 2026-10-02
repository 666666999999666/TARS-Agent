"""Recompute the published AppWorld summary offline; this does not rerun its evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "docs/evaluation/results.json"
TOKEN_KEYS = (
    "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
)
OUTCOMES = ("A_win", "B_win", "both_pass", "both_fail")
TASK_ID = re.compile(r"[0-9a-f]{7}_[123]\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SummaryError(ValueError):
    """The public summary is incomplete or internally inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def exact_keys(value: Any, names: str, label: str) -> None:
    require(isinstance(value, dict) and set(value) == set(names.split()),
            f"{label}: unexpected or missing fields")


def integer(value: Any, label: str) -> None:
    require(type(value) is int and value >= 0, f"{label}: expected nonnegative integer")


def digest(value: Any, label: str) -> None:
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None,
            f"{label}: expected SHA-256")


def usage_cost(usage: dict[str, int]) -> int:
    """Frozen peak prices in nano-CNY per token, never a provider invoice."""
    return (usage["input_tokens"] * 2000 + usage["output_tokens"] * 8000
            + usage["cache_read_input_tokens"] * 40)


def validate_usage(usage: Any) -> None:
    exact_keys(usage, " ".join(TOKEN_KEYS), "confirmed_usage")
    for name, value in usage.items():
        integer(value, name)
    require(usage["cache_creation_input_tokens"] == 0,
            "This frozen price calculation requires zero cache-creation tokens")


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute both percentage metrics from task results, not Run terminal states."""
    variants: dict[str, Any] = {}
    for variant in ("A", "B"):
        selected = [r for r in records if r["variant"] == variant]
        scenarios = sorted({r["scenario_id"] for r in selected})
        passed = sum(r["official_pass"] for r in selected)
        scenario_passed = sum(all(r["official_pass"] for r in selected
                                  if r["scenario_id"] == scenario) for scenario in scenarios)
        usage = {key: sum(r["confirmed_usage"][key] for r in selected) for key in TOKEN_KEYS}
        variants[variant] = {
            "task_count": len(selected), "passed_tasks": passed,
            "scenario_count": len(scenarios), "passed_scenarios": scenario_passed,
            "tgc_percent": round(100 * passed / len(selected), 1),
            "sgc_percent": round(100 * scenario_passed / len(scenarios), 1),
            "runtime_status_counts": dict(sorted(Counter(r["runtime_status"] for r in selected).items())),
            "runtime_reason_counts": dict(sorted(Counter(r["runtime_reason"] for r in selected).items())),
            "official_failed_checks": sum(r["official_failed_checks"] for r in selected),
            "median_latency_seconds": statistics.median(r["latency_ms"] for r in selected) / 1000,
            "http_attempts": sum(r["http_attempts"] for r in selected),
            "confirmed_usage": usage,
            "estimated_confirmed_nano_cny": usage_cost(usage),
        }
    by_task: dict[str, dict[str, bool]] = {}
    for record in records:
        by_task.setdefault(record["task_id"], {})[record["variant"]] = record["official_pass"]
    paired = dict.fromkeys(OUTCOMES, 0)
    for pair in by_task.values():
        outcome = ("both_pass" if pair["A"] else "both_fail") if pair["A"] == pair["B"] else (
            "A_win" if pair["A"] else "B_win")
        paired[outcome] += 1
    return {"variants": variants, "paired_outcomes": paired}


def validate_stage(stage: Any, split: str) -> dict[str, Any]:
    exact_keys(stage, "selection task_ids task_ids_sha256 data_sha256 records aggregate "
               "raw_automatic_collateral_failed_checks source_evidence_sha256", split)
    ids = stage["task_ids"]
    expected = 12 if split == "train" else 57
    require(isinstance(ids, list) and len(ids) == expected, f"{split}: expected {expected} tasks")
    require(all(isinstance(task, str) and TASK_ID.fullmatch(task) for task in ids),
            f"{split}: invalid task ID")
    require(ids == sorted(set(ids)), f"{split}: task IDs must be sorted and unique")
    expected_digest = hashlib.sha256(("\n".join(ids) + "\n").encode("ascii")).hexdigest()
    require(stage["task_ids_sha256"] == expected_digest, f"{split}: task list digest mismatch")
    digest(stage["data_sha256"], f"{split}.data_sha256")
    scenes = {task.rsplit("_", 1)[0] for task in ids}
    require(all({f"{scene}_{i}" for i in (1, 2, 3)} <= set(ids) for scene in scenes),
            f"{split}: incomplete three-variant scenario")
    selection = "four_preselected_scenarios" if split == "train" else "complete_official_dev"
    require(stage["selection"] == selection, f"{split}: wrong selection declaration")
    if split == "train":
        require(scenes == {"692c77d", "22cc237", "27e1026", "76f2c72"},
                "train: fixed diagnostic scenarios changed")
    evidence = stage["source_evidence_sha256"]
    exact_keys(evidence, "result checkpoint freeze official_A official_B", f"{split}.evidence")
    for label, value in evidence.items():
        digest(value, label)
    raw = stage["raw_automatic_collateral_failed_checks"]
    exact_keys(raw, "A B", f"{split}.raw_automatic_collateral_failed_checks")
    require(raw == ({"A": 1, "B": None} if split == "train" else {"A": None, "B": None}),
            f"{split}: preserve the raw automatic unknown values")
    records = stage["records"]
    require(isinstance(records, list) and len(records) == expected * 2,
            f"{split}: expected {expected * 2} paired records")
    seen: set[tuple[str, str]] = set()
    for record in records:
        exact_keys(record, "split task_id scenario_id variant official_pass official_failed_checks "
                   "runtime_status runtime_reason latency_ms http_attempts confirmed_usage "
                   "usage_complete attempt_sha256", f"{split}.record")
        require(record["split"] == split and record["task_id"] in ids
                and record["variant"] in ("A", "B"), f"{split}: invalid record identity")
        key = (record["task_id"], record["variant"])
        require(key not in seen, f"{split}: duplicate task/variant record")
        seen.add(key)
        require(record["scenario_id"] == record["task_id"].rsplit("_", 1)[0],
                f"{split}: scenario mismatch")
        require(type(record["official_pass"]) is bool, f"{split}: pass must be boolean")
        integer(record["official_failed_checks"], "official_failed_checks")
        require(record["official_pass"] == (record["official_failed_checks"] == 0),
                f"{split}: official pass/failed-check mismatch")
        reason = record["runtime_reason"]
        require(reason in ("completed", "context_budget_exceeded", "max_tokens"),
                f"{split}: unrecognized or unredacted runtime reason")
        require(record["runtime_status"] == ("succeeded" if reason == "completed" else "failed"),
                f"{split}: runtime reason/status mismatch")
        for field in ("latency_ms", "http_attempts"):
            integer(record[field], field)
        require(0 < record["http_attempts"] <= 120, f"{split}: HTTP limit violated")
        require(record["usage_complete"] is True, f"{split}: usage is not fully confirmed")
        validate_usage(record["confirmed_usage"])
        digest(record["attempt_sha256"], "attempt_sha256")
    require(seen == {(task, variant) for task in ids for variant in ("A", "B")},
            f"{split}: missing pair")
    result = aggregate_records(records)
    require(stage["aggregate"] == result, f"{split}: aggregate mismatch")
    return result


def validate_manual_review(review: Any, records: list[dict[str, Any]]) -> None:
    exact_keys(review, "scope source_sha256 definition counted_checks counts "
               "new_B_affected_tasks removed_in_B_tasks shared_affected_tasks", "manual_review")
    require(review["scope"] == "dev", "manual review: wrong split")
    digest(review["source_sha256"], "manual review source")
    require(isinstance(review["definition"], str), "manual review: missing counting definition")
    by_id = {(r["task_id"], r["variant"]): r for r in records}
    seen: set[tuple[str, str, int]] = set()
    tasks: dict[str, set[str]] = {"A": set(), "B": set()}
    counts = dict.fromkeys(("A", "B"), 0)
    require(isinstance(review["counted_checks"], list), "manual review: missing checks")
    for check in review["counted_checks"]:
        exact_keys(check, "task_id variant failure_index category trace_sha256", "manual check")
        require(check["variant"] in ("A", "B"), "manual check: invalid variant")
        integer(check["failure_index"], "failure_index")
        record = by_id.get((check["task_id"], check["variant"]))
        require(record is not None and check["failure_index"] < record["official_failed_checks"],
                "manual check: no matching official failed check")
        key = (check["task_id"], check["variant"], check["failure_index"])
        require(key not in seen, "manual check: duplicate check")
        seen.add(key)
        require(check["category"] in ("actual_extra_model_changes", "extra_and_missing_target_records",
                                      "actual_out_of_scope_added_records"), "manual check: invalid category")
        digest(check["trace_sha256"], "trace_sha256")
        tasks[check["variant"]].add(check["task_id"])
        counts[check["variant"]] += 1
    calculated = {v: {"failed_checks": counts[v], "affected_tasks": len(tasks[v])} for v in ("A", "B")}
    require(review["counts"] == calculated, "manual review: counts mismatch")
    require(review["new_B_affected_tasks"] == sorted(tasks["B"] - tasks["A"]),
            "manual review: new B regressions mismatch")
    require(review["removed_in_B_tasks"] == sorted(tasks["A"] - tasks["B"]),
            "manual review: removed tasks mismatch")
    require(review["shared_affected_tasks"] == sorted(tasks["A"] & tasks["B"]),
            "manual review: shared tasks mismatch")


def validate_cost(cost: Any, stages: dict[str, Any]) -> None:
    exact_keys(cost, "unit estimate_basis frozen_peak_nano_cny_per_token cap_nano_cny "
               "http_attempts confirmed_nano_cny unknown_reserved_nano_cny remaining_nano_cny "
               "earlier_interrupted_batch", "cost")
    require(cost["unit"] == "nano_CNY" and cost["estimate_basis"] == "frozen_peak_prices_not_account_invoice",
            "cost: missing estimate qualification")
    require(cost["frozen_peak_nano_cny_per_token"] == {
        "input_tokens": 2000, "output_tokens": 8000, "cache_read_input_tokens": 40,
    }, "cost: frozen price mismatch")
    for name in ("cap_nano_cny", "http_attempts", "confirmed_nano_cny",
                 "unknown_reserved_nano_cny", "remaining_nano_cny"):
        integer(cost[name], name)
    require(cost["cap_nano_cny"] == 50_000_000_000, "cost: approved cap changed")
    old = cost["earlier_interrupted_batch"]
    exact_keys(old, "http_attempts settled_http_attempts unknown_http_attempts confirmed_usage "
               "confirmed_nano_cny unknown_reserved_nano_cny", "earlier_interrupted_batch")
    for name in set(old) - {"confirmed_usage"}:
        integer(old[name], name)
    validate_usage(old["confirmed_usage"])
    require(old["http_attempts"] == old["settled_http_attempts"] + old["unknown_http_attempts"],
            "cost: earlier attempt count mismatch")
    require(old["confirmed_nano_cny"] == usage_cost(old["confirmed_usage"]),
            "cost: earlier confirmed cost mismatch")
    require(old["unknown_reserved_nano_cny"] == old["unknown_http_attempts"] * 2_162_688_000,
            "cost: unknown use must retain its conservative reservation")
    all_variants = [v for stage in stages.values() for v in stage["variants"].values()]
    require(cost["http_attempts"] == old["http_attempts"] + sum(v["http_attempts"] for v in all_variants),
            "cost: total request count mismatch")
    require(cost["confirmed_nano_cny"] == old["confirmed_nano_cny"]
            + sum(v["estimated_confirmed_nano_cny"] for v in all_variants), "cost: confirmed total mismatch")
    require(cost["unknown_reserved_nano_cny"] == old["unknown_reserved_nano_cny"],
            "cost: unknown reserve mismatch")
    require(cost["remaining_nano_cny"] + cost["confirmed_nano_cny"] + cost["unknown_reserved_nano_cny"]
            == cost["cap_nano_cny"], "cost: remaining budget mismatch")


def verify_summary(data: Any) -> dict[str, Any]:
    exact_keys(data, "schema_version experiment source_evidence_sha256 stages manual_review cost limitations",
               "summary")
    require(data["schema_version"] == 1, "unsupported schema version")
    experiment = data["experiment"]
    exact_keys(experiment, "comparison executed_date execution_source_commit source_tree_sha256 "
               "appworld_source_commit appworld_data_version image_sha256 model protocol base_url "
               "thinking model_fallback workers max_steps task_timeout_seconds max_output_tokens "
               "context_budget max_http_attempts_per_step infrastructure_retries prompt_template_sha256",
               "experiment")
    for name in ("execution_source_commit", "appworld_source_commit"):
        require(isinstance(experiment[name], str) and re.fullmatch(r"[0-9a-f]{40}", experiment[name]) is not None,
                f"{name}: expected full Git commit")
    for name in ("source_tree_sha256", "image_sha256"):
        digest(experiment[name], name)
    exact_keys(experiment["prompt_template_sha256"], "A B", "prompt templates")
    for value in experiment["prompt_template_sha256"].values():
        digest(value, "prompt template")
    require(experiment["prompt_template_sha256"]["A"] != experiment["prompt_template_sha256"]["B"],
            "A and B must have different completion prompts")
    fixed = {"comparison": "completion-contract-v1", "executed_date": "2026-09-30",
             "appworld_data_version": "0.2.0", "model": "deepseek-flash", "protocol": "anthropic_messages",
             "base_url": "https://api.deepseek.com/anthropic", "thinking": "provider_default",
             "model_fallback": False, "workers": 1, "max_steps": 60, "task_timeout_seconds": 900,
             "max_output_tokens": 8192, "context_budget": 131072,
             "max_http_attempts_per_step": 2, "infrastructure_retries": 1}
    require(all(experiment[k] == v for k, v in fixed.items()), "frozen experiment configuration mismatch")
    exact_keys(data["source_evidence_sha256"], "delivery_summary manual_review", "source evidence")
    for name, value in data["source_evidence_sha256"].items():
        digest(value, name)
    exact_keys(data["stages"], "train dev", "stages")
    stages = {split: validate_stage(data["stages"][split], split) for split in ("train", "dev")}
    validate_manual_review(data["manual_review"], data["stages"]["dev"]["records"])
    validate_cost(data["cost"], stages)
    require(isinstance(data["limitations"], list) and all(isinstance(x, str) for x in data["limitations"]),
            "limitations: expected list of strings")
    return stages


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.summary.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
        # json.loads accepts NaN/Infinity by default; serialized evidence must not.
        require(not any(not math.isfinite(v) for v in _numbers(data)), "non-finite JSON number")
        stages = verify_summary(data)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Invalid summary: {exc}", file=sys.stderr)
        return 1
    for split, stage in stages.items():
        cells = [f"{v}: {s['passed_tasks']}/{s['task_count']} TGC={s['tgc_percent']:.1f}% "
                 f"SGC={s['sgc_percent']:.1f}%" for v, s in stage["variants"].items()]
        print(f"{split}: " + "; ".join(cells))
        print("  paired: " + ", ".join(f"{k}={v}" for k, v in stage["paired_outcomes"].items()))
    print("Verified published-summary consistency only; no model call or official reevaluation.")
    return 0


def _numbers(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [v for item in value.values() for v in _numbers(item)]
    if isinstance(value, list):
        return [v for item in value for v in _numbers(item)]
    return [value] if isinstance(value, float) else []


if __name__ == "__main__":
    raise SystemExit(main())
