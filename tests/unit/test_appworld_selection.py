from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from scripts import run_internship_evals as launcher

from tars_agent.core.eval import appworld
from tars_agent.core.eval.models import AppWorldSpec
from tars_agent.core.persistence.cost_budget import CostLedger
from tars_agent.core.persistence.request_budget import RequestLedger
from tests.unit.test_appworld_recovery import Batch, FakeWorld


@pytest.mark.parametrize("dataset", ["train", "dev"])
def test_explicit_ids_preserve_order_and_require_official_membership(dataset):
    metadata = {"task_ids": ["a_1", "b_1", "c_1"]}
    spec = AppWorldSpec(data_root="unused", dataset=dataset, task_ids=["c_1", "a_1"])
    assert appworld.select_task_ids(spec, metadata) == ["c_1", "a_1"]
    spec.task_ids = ["not-official_1"]
    with pytest.raises(ValueError, match="official dataset members"):
        appworld.select_task_ids(spec, metadata)


@pytest.mark.parametrize("update", [
    {"dataset": "test_normal", "task_ids": ["a_1"]},
    {"dataset": "dev", "task_ids": []},
    {"dataset": "dev", "task_ids": ["a_1", "a_1"]},
    {"dataset": "dev", "task_ids": [""]},
    {"dataset": "dev", "task_ids": [" a_1"]},
    {"dataset": "dev", "task_ids": ["a_1"], "task_limit": 1},
])
def test_invalid_or_test_normal_subsets_rejected(update):
    with pytest.raises(ValidationError):
        AppWorldSpec.model_validate({"data_root": "unused", **update})


def test_existing_default_and_task_limit_selection_stay_compatible():
    metadata = {"task_ids": ["z_1", "a_1", "b_1"]}
    full = AppWorldSpec(data_root="unused")
    assert full.task_ids is None and appworld.select_task_ids(full, metadata) == metadata["task_ids"]
    limited = AppWorldSpec(data_root="unused", dataset="dev", task_limit=2)
    assert appworld.select_task_ids(limited, metadata) == ["z_1", "a_1"]
    full.task_ids = ["a_1"]  # Runtime boundary also rejects validation bypass.
    with pytest.raises(ValueError, match="train/dev"):
        appworld.select_task_ids(full, metadata)


def prepare_batch(tmp_path, monkeypatch):
    batch = Batch(tmp_path, monkeypatch, count=7)
    data_root = batch.manifest.appworld.data_root
    batch.manifest = launcher.prepare_suite("deepseek-train-paired")
    batch.manifest.appworld.data_root = data_root
    batch.ids = list(reversed(launcher.TRAIN_TASK_IDS)) + ["other_1", "another_1"]
    ledger = RequestLedger(batch.config.llm.request_budget_path, limit=None)
    ledger.reserve()
    cost = CostLedger.initialize(tmp_path / "cost.sqlite3", request_budget_path=ledger.path)
    batch.config = appworld.load_deepseek_config(private_env=tmp_path / "missing.env", environment={
        "DEEPSEEK_API_KEY": "unit-placeholder", "DEEPSEEK_REQUEST_BUDGET_PATH": str(ledger.path),
        "DEEPSEEK_COST_BUDGET_PATH": str(cost.path),
    })
    return batch, ledger


async def test_paired_train_uses_pinned_ids_and_retains_original_request_count(tmp_path, monkeypatch):
    batch, ledger = prepare_batch(tmp_path, monkeypatch)
    result = await batch.run()
    assert result.selected_task_ids == list(launcher.TRAIN_TASK_IDS)
    assert [(job["task_id"], job["prompt_variant"]) for job in batch.calls] == [
        (task, variant) for _, task, variant in appworld.task_schedule(
            list(launcher.TRAIN_TASK_IDS), paired=True,
        )
    ]
    assert all(job["profile"] == appworld.DEEPSEEK_PROFILE and job["model"] == appworld.DEEPSEEK_MODEL
               and job["request_limit"] == 5761 and job["cost_budget_path"] == str(batch.config.llm.cost_budget_path)
               for job in batch.calls)
    assert result.benchmark["complete"] is True and not FakeWorld.live
    assert result.benchmark["official_metrics"] is None  # No mixture of the A/B scores.
    assert set(result.benchmark["comparison"]["variants"]) == {"A", "B"}
    assert ledger.counts()["real"] == 1
    frozen = json.loads((batch.artifacts / "freeze.json").read_text())
    assert frozen["manifest"]["appworld"]["task_ids"] == list(launcher.TRAIN_TASK_IDS)
    assert frozen["model_protocol"]["provider_profile"] == appworld.DEEPSEEK_PROFILE


async def test_deepseek_rejects_repeated_scenario_variants_before_starting_workers(tmp_path, monkeypatch):
    batch, _ = prepare_batch(tmp_path, monkeypatch)
    batch.ids = ["a_1", "a_2", "b_1", "c_1", "d_1"]
    batch.manifest.appworld.task_ids = batch.ids
    with pytest.raises(ValueError, match="twelve frozen"):
        await batch.run()
    assert not batch.calls and not FakeWorld.live


async def test_deepseek_limited_run_pauses_without_filling_remaining_dev_results(tmp_path, monkeypatch):
    batch, ledger = prepare_batch(tmp_path, monkeypatch)
    batch.behavior = "rate_limit"
    result = await batch.run()
    assert result.benchmark["paused_reason"] == "llm_rate_limited"
    assert not result.benchmark["complete"] and len(batch.calls) == 1
    state = json.loads((batch.artifacts / "checkpoint.json").read_text())
    assert not state["results"] and state["requests_before"] == 1
    assert ledger.counts()["real"] == 1 and not FakeWorld.live
