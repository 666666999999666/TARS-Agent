"""Public evidence remains complete, paired and distinguishable from runtime status."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import socket
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("verify_eval_summary", ROOT / "scripts/verify_eval_summary.py")
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture
def summary() -> dict[str, Any]:
    return json.loads((ROOT / "docs/evaluation/results.json").read_text(encoding="utf-8"))


def test_published_results_recompute_without_network_or_writes(
    summary: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Verification must not connect to services or write files")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    verified = VERIFIER.verify_summary(summary)
    assert verified["dev"]["variants"]["A"]["passed_tasks"] == 20
    assert verified["dev"]["variants"]["B"]["passed_tasks"] == 54
    assert verified["dev"]["paired_outcomes"] == {
        "A_win": 0, "B_win": 34, "both_pass": 20, "both_fail": 3,
    }
    assert VERIFIER.main([]) == 0
    assert "no model call or official reevaluation" in capsys.readouterr().out


@pytest.mark.parametrize("mutation,match", [
    ("missing", "114 paired records"),
    ("duplicate", "duplicate task/variant"),
    ("pass", "official pass/failed-check mismatch"),
    ("aggregate", "aggregate mismatch"),
    ("scenario", "scenario mismatch"),
    ("usage", "aggregate mismatch"),
    ("private_field", "unexpected or missing fields"),
    ("unknown_to_zero", "preserve the raw automatic unknown"),
])
def test_rejects_incomplete_or_misleading_records(
    summary: dict[str, Any], mutation: str, match: str,
) -> None:
    stage = summary["stages"]["dev"]
    if mutation == "missing":
        stage["records"].pop()
    elif mutation == "duplicate":
        stage["records"][-1] = copy.deepcopy(stage["records"][0])
    elif mutation == "pass":
        stage["records"][0]["official_pass"] = not stage["records"][0]["official_pass"]
    elif mutation == "aggregate":
        stage["aggregate"]["variants"]["A"]["tgc_percent"] = 100.0
    elif mutation == "scenario":
        stage["records"][0]["scenario_id"] = "0000000"
    elif mutation == "usage":
        stage["records"][0]["confirmed_usage"]["output_tokens"] += 1
    elif mutation == "private_field":
        stage["records"][0]["raw_answer"] = "must never be published"
    elif mutation == "unknown_to_zero":
        stage["raw_automatic_collateral_failed_checks"]["B"] = 0
    with pytest.raises(VERIFIER.SummaryError, match=match):
        VERIFIER.verify_summary(summary)


def test_rejects_incomplete_scenario_even_with_updated_list_digest(summary: dict[str, Any]) -> None:
    stage = summary["stages"]["dev"]
    stage["task_ids"][0] = "0000000_1"
    stage["task_ids"].sort()
    stage["task_ids_sha256"] = hashlib.sha256(
        ("\n".join(stage["task_ids"]) + "\n").encode("ascii")
    ).hexdigest()
    with pytest.raises(VERIFIER.SummaryError, match="incomplete three-variant scenario"):
        VERIFIER.verify_summary(summary)


def test_runtime_success_does_not_become_official_task_success(summary: dict[str, Any]) -> None:
    records = summary["stages"]["dev"]["records"]
    assert sum(r["runtime_status"] == "succeeded" for r in records if r["variant"] == "A") == 57
    assert sum(r["official_pass"] for r in records if r["variant"] == "A") == 20
    failed_b = [r for r in records if r["variant"] == "B" and r["runtime_status"] == "failed"]
    assert len(failed_b) == 1
    assert failed_b[0]["runtime_reason"] == "max_tokens"
    assert not failed_b[0]["official_pass"]


@pytest.mark.parametrize("mutation,match", [
    ("duplicate_check", "duplicate check"),
    ("hide_regression", "new B regressions mismatch"),
    ("drop_unknown_cost", "unknown use must retain"),
    ("exclude_failed_batch", "total request count mismatch"),
])
def test_retains_negative_results_and_unknown_cost(
    summary: dict[str, Any], mutation: str, match: str,
) -> None:
    if mutation == "duplicate_check":
        summary["manual_review"]["counted_checks"].append(summary["manual_review"]["counted_checks"][0])
    elif mutation == "hide_regression":
        summary["manual_review"]["new_B_affected_tasks"] = []
    elif mutation == "drop_unknown_cost":
        summary["cost"]["earlier_interrupted_batch"]["unknown_reserved_nano_cny"] = 0
    elif mutation == "exclude_failed_batch":
        summary["cost"]["http_attempts"] -= 98
    with pytest.raises(VERIFIER.SummaryError, match=match):
        VERIFIER.verify_summary(summary)


def test_cli_fails_for_duplicate_json_keys(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    assert VERIFIER.main([str(path)]) == 1
    assert "duplicate JSON object key" in capsys.readouterr().err


def test_cli_fails_cleanly_for_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert VERIFIER.main([str(tmp_path / "absent.json")]) == 1
    assert "Invalid summary" in capsys.readouterr().err
