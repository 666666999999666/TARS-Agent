from __future__ import annotations

import copy
import json

import pytest

from tars_agent.core.eval import appworld
from tars_agent.core.eval.appworld_comparison import (
    REQUEST_ALLOWANCE,
    TRAIN_TASK_IDS,
    bind_shared_budget,
    comparison_summary,
)
from tars_agent.core.eval.appworld_prompts import prompt_template, template_sha256
from tars_agent.core.persistence.cost_budget import CostLedger
from tars_agent.core.persistence.request_budget import RequestLedger
from tests.unit.test_appworld_recovery import Batch, FakeWorld


def paired_batch(tmp_path, monkeypatch):
    batch = Batch(tmp_path, monkeypatch, count=1)
    batch.ids = [f"case{scene}_{variant}" for scene in range(2) for variant in (1, 2, 3)]
    batch.manifest.appworld.paired_comparison = True
    return batch


async def test_paired_schedule_scorers_and_disk_resume_preserve_each_arm(tmp_path, monkeypatch):
    batch = paired_batch(tmp_path, monkeypatch)
    first = await batch.run()
    assert first.benchmark["complete"] and first.benchmark["expected_tasks"] == 12
    assert batch.max_parallel == 1 and batch.scorings == 2 and not FakeWorld.live
    assert [(job["task_id"], job["prompt_variant"]) for job in batch.calls] == [
        (task, variant) for _, task, variant in appworld.task_schedule(batch.ids, paired=True)
    ]
    assert len({job["attempt_root"] for job in batch.calls}) == 12
    canonical = list((batch.artifacts / "variants").rglob("state.db"))
    assert len(canonical) == 12
    before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in canonical}
    second = await batch.run()  # Actual on-disk JSON round trip, not an in-memory fake state.
    assert second.benchmark["complete"] and len(batch.calls) == 12
    assert before == {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in canonical}
    frozen = json.loads((batch.artifacts / "freeze.json").read_text())
    assert frozen["completion_prompts"]["templates"] == {
        variant: template_sha256(variant) for variant in ("A", "B")
    }


async def test_paired_interruption_recovers_same_variant_then_continues_schedule(tmp_path, monkeypatch):
    batch = paired_batch(tmp_path, monkeypatch)
    batch.behavior = "interrupt"
    interrupted = await batch.run()
    assert not interrupted.benchmark["complete"] and len(batch.calls) == 1
    resumed = await batch.run()
    assert resumed.benchmark["complete"]
    assert [job["prompt_variant"] for job in batch.calls[:3]] == ["A", "A", "B"]
    assert len(batch.calls) == 13 and not FakeWorld.live
    attempts = json.loads((batch.artifacts / "checkpoint.json").read_text())["attempts"]
    assert attempts[0]["evaluation"]["prompt_variant"] == "A"
    assert attempts[0]["status"] == "error" and attempts[1]["repetition"] == 2


async def test_budget_denial_pauses_without_scoring_partial_world(tmp_path, monkeypatch):
    batch = paired_batch(tmp_path, monkeypatch)
    batch.behavior = "cost_denied"
    result = await batch.run()
    assert result.benchmark["paused_reason"] == "llm_request_budget_exhausted"
    assert not result.benchmark["complete"] and result.benchmark["completed_tasks"] == 0
    assert len(batch.calls) == 1 and batch.scorings == 0 and not FakeWorld.live


@pytest.mark.parametrize("field", ["prompt_variant", "prompt_template_sha256"])
async def test_paired_output_receipt_cannot_be_relabelled(tmp_path, monkeypatch, field):
    batch = paired_batch(tmp_path, monkeypatch)
    await batch.run()
    marker = next((batch.artifacts / "variants" / "A").rglob(".tars-checkpoint.json"))
    receipt = json.loads(marker.read_text())
    receipt["binding"][field] = "B" if field == "prompt_variant" else template_sha256("B")
    appworld.write_json(marker, receipt)
    with pytest.raises(ValueError, match="output changed"):
        await batch.run()
    assert len(batch.calls) == 12


def test_original_prompt_hash_is_unchanged_and_only_completion_clause_differs():
    assert template_sha256("A") == "eb33913ffe8c2b13306787797b8230423327480a6be7eb9a848693db4f93d8f2"
    a, b = prompt_template("A"), prompt_template("B")
    start = a.index("apis.supervisor.complete_task(")
    end = a.index("Do not access filesystem")
    assert b[:start] == a[:start]
    assert b[b.index("Do not access filesystem"):] == a[end:]
    assert "for action-only tasks omit answer" in b


def test_shared_request_allowance_created_once_and_not_refreshed_on_resume(tmp_path):
    from tars_agent.core.config import TarsConfig

    config = TarsConfig()
    config.llm.request_limit = None
    config.llm.request_budget_path = tmp_path / "original.sqlite3"
    ledger = RequestLedger(config.llm.request_budget_path)
    ledger.reserve()
    cost = CostLedger.initialize(tmp_path / "cost.sqlite3", request_budget_path=ledger.path)
    config.llm.cost_budget_path = cost.path
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "source.py").write_text("frozen")
    batch_root = tmp_path / "batch"
    initial = bind_shared_budget(batch_root, batch_root / "train", "train", config, repo)
    ledger.reserve()
    resumed = bind_shared_budget(batch_root, batch_root / "train", "train", config, repo)
    assert initial["request_limit"] == resumed["request_limit"] == 1 + REQUEST_ALLOWANCE
    assert resumed["initial_requests"] == 1 and resumed["request_high_water"] == 2
    assert cost.summary()["attempt_count"] == 0
    with pytest.raises(ValueError, match="train must be complete"):
        bind_shared_budget(batch_root, batch_root / "dev", "dev", config, repo)
    assert "dev" not in json.loads((batch_root / "batch.json").read_text())["stages"]
    with pytest.raises(ValueError, match="different evidence directory"):
        bind_shared_budget(batch_root, batch_root / "train-again", "train", config, repo)
    evidence = batch_root / "train" / "evidence"
    frozen = {
        "tree_digest": initial["binding"]["tree_digest"],
        "base_config_fingerprint": appworld.config_fingerprint(config),
        "manifest": {"appworld": {"paired_comparison": True, "dataset": "train",
                                    "task_ids": list(TRAIN_TASK_IDS)}},
    }
    appworld.write_json(evidence / "freeze.json", frozen)
    appworld.write_json(evidence / "checkpoint.json", {
        "run_id": "finished-train", "binding": frozen,
        "results": {f"{variant}:{task_id}": {} for task_id in TRAIN_TASK_IDS for variant in ("A", "B")},
    })
    appworld.write_json(evidence / "result.json", {
        "run_id": "finished-train", "selected_task_ids": list(TRAIN_TASK_IDS),
        "benchmark": {"complete": True, "errors": [], "dataset": "train",
                      "expected_tasks": 24, "completed_tasks": 24},
    })
    dev = bind_shared_budget(batch_root, batch_root / "dev", "dev", config, repo)
    assert dev["request_limit"] == initial["request_limit"]
    assert dev["stages"] == {"train": str(batch_root / "train"), "dev": str(batch_root / "dev")}


def scores(success: bool, requirement: str | None = None):
    return {"aggregate": {"task_goal_completion": 100.0 if success else 0.0,
                          "scenario_goal_completion": 100.0 if success else 0.0},
            "individual": {"a_1": {"success": success, "passes": [], "failures": [] if success
                                     else [{"requirement": requirement}]}}}


def test_pair_report_preserves_requirements_and_unknown_collateral_blocks_selection():
    metrics = {"A": scores(False, "assert answers match."), "B": scores(True)}
    report = comparison_summary(metrics, ["a_1"], [])
    assert report["outcomes"] == {"B_win": 1}
    assert report["B_eligible_pending_engineering_acceptance"] is True
    assert report["variants"]["A"]["tasks"][0]["failed_checks"] == [
        {"requirement": "assert answers match.", "label": None, "category": "answer_contract"}
    ]
    unknown = copy.deepcopy(metrics)
    unknown["A"]["individual"]["a_1"]["failures"][0]["requirement"] = "assert unspecified change."
    report = comparison_summary(unknown, ["a_1"], [])
    assert report["variants"]["A"]["collateral_failure_count"] is None
    assert report["B_eligible_pending_engineering_acceptance"] is False
    with pytest.raises(ValueError, match="exactly the frozen"):
        comparison_summary(metrics, ["a_1", "missing_1"], [])


async def test_comparison_official_scoring_returns_details_without_changing_canonical_outputs(
    tmp_path, monkeypatch,
):
    from tars_agent.core.eval.models import AppWorldSpec

    data = tmp_path / "data"
    data.mkdir()
    root = tmp_path / "variant-A"
    canonical = root / "official" / "outputs" / "run" / "tasks" / "a_1"
    (canonical / "dbs").mkdir(parents=True)
    (canonical / "dbs" / "state").write_text("saved-before-scoring")
    before = appworld.output_digest(canonical)
    metrics = scores(True)
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        assert "save_reports=False" in args[-1] and "include_details=True" in args[-1]
        assert args[args.index("--network") + 1] == "none"
        attempt = next((root / "official" / "scoring-attempts").iterdir())
        appworld.write_json(attempt / "metrics.json", metrics)
        return ""

    monkeypatch.setattr(appworld, "command", command)
    spec = AppWorldSpec(data_root=str(data), dataset="train", workers=1, paired_comparison=True)
    assert await appworld.score_outputs(spec, "pinned-image", data, root, "run", ["a_1"]) == metrics
    assert appworld.output_digest(canonical) == before
    timestamp = (root / "official" / "metrics.json").stat().st_mtime_ns
    assert await appworld.score_outputs(spec, "pinned-image", data, root, "run", ["a_1"]) == metrics
    assert len(calls) == 1 and (root / "official" / "metrics.json").stat().st_mtime_ns == timestamp
    appworld.write_json(root / "official" / "metrics.json", scores(False, "assert answers match."))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        await appworld.score_outputs(spec, "pinned-image", data, root, "run", ["a_1"])
