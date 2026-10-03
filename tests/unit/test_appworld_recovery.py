from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tars_agent.core.config import TarsConfig
from tars_agent.core.eval import appworld
from tars_agent.core.eval.models import EvalStatus, EvalSuiteManifest, RunProvenance


class FakeWorld:
    worlds: dict[str, FakeWorld] = {}
    live: dict[str, FakeWorld] = {}
    serial = 0

    def __init__(self, image_id: str, data: Path, output: Path, owner: str) -> None:
        self.output, self.owner = output, owner
        self.url = ""
        self.container_id: str | None = None
        self.network: str | None = None

    async def start(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        type(self).serial += 1
        self.url = f"http://127.0.0.1:{10000 + self.serial}"
        self.container_id, self.network = self.url, self.owner
        self.worlds[str(self.output)] = self
        self.live[self.url] = self

    async def close(self) -> None:
        original = self.worlds.get(str(self.output), self)
        self.live.pop(original.url, None)
        original.container_id = original.network = None
        self.container_id = self.network = None


class Batch:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 2,
                 workers: int = 1) -> None:
        FakeWorld.worlds, FakeWorld.live, FakeWorld.serial = {}, {}, 0
        self.artifacts = tmp_path / "artifacts"
        data = tmp_path / "appworld" / "data"
        data.mkdir(parents=True)
        (data / "fixture").write_text("synthetic data")
        self.ids = [f"case{index}_1" for index in range(count)]
        self.config = TarsConfig()
        self.config.llm.request_limit = None
        self.config.llm.request_budget_path = tmp_path / "ledger.sqlite3"
        self.manifest = EvalSuiteManifest.model_validate({
            "suite_id": "synthetic-appworld", "name": "synthetic", "adapter": "appworld",
            "execution_mode": "agent_tasks", "tasks": [],
            "model_config_ref": {"source": "runtime_config", "reference": "frozen"},
            "appworld": {"data_root": str(data.parent), "dataset": "dev", "workers": workers},
        })
        self.provenance = RunProvenance(
            collected_at="2026-09-29", git_sha="fixed", git_dirty=True, tree_digest="fixed-tree",
            lock_hash="lock", config_hash="config", repository_root=str(tmp_path), platform="test",
            platform_release="test", python_version="3.12", model=self.config.llm.default_model,
            model_config_ref=self.manifest.model_config_ref, docker_image="fake", docker_image_digest="image",
            config={},
        )
        self.calls: list[dict[str, Any]] = []
        self.scorings = 0
        self.behavior = "success"
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.active_urls: set[str] = set()
        self.max_parallel = 0

        async def image_metadata(spec: Any, root: Path) -> tuple[str, dict[str, Any]]:
            metadata = {"task_ids": self.ids, "scenarios": {
                task_id: [task_id.rsplit("_", 1)[0], int(task_id.rsplit("_", 1)[1])]
                for task_id in self.ids
            }}
            appworld.validate_task_set(spec.dataset, metadata)
            return "image", metadata

        monkeypatch.setattr(appworld, "image_metadata", image_metadata)
        monkeypatch.setattr(appworld, "WorldContainer", FakeWorld)
        monkeypatch.setattr(appworld, "run_worker_process", self.worker)
        monkeypatch.setattr(appworld, "score_outputs", self.score)

    async def worker(self, job: dict[str, Any], directory: Path) -> dict[str, Any]:
        checkpoint = json.loads((self.artifacts / "checkpoint.json").read_text())
        result_key = (f"{job['prompt_variant']}:{job['task_id']}"
                      if self.manifest.appworld.paired_comparison else job["task_id"])
        assert checkpoint["inflight"][result_key]["job"] == job  # Durable before launch.
        directory.mkdir(parents=True, exist_ok=False)
        appworld.write_json(directory / "job.json", job)
        appworld.write_json(directory / "process.json", {"phase": "finished", "argv": []})
        self.calls.append(job)
        self.started.set()
        url = job["environment_url"]
        assert url not in self.active_urls
        self.active_urls.add(url)
        self.max_parallel = max(self.max_parallel, len(self.active_urls))
        try:
            if self.behavior == "blocked":
                await self.release.wait()
            else:
                await asyncio.sleep(0.01)
            if self.behavior == "interrupt":
                self.behavior = "success"
                raise RuntimeError("simulated controller interruption")
            world = FakeWorld.live[url]
            output = world.output / "outputs" / job["experiment_name"] / "tasks" / job["task_id"]
            (output / "dbs").mkdir(parents=True)
            (output / "dbs" / "state.db").write_text(job["task_id"])
            raw = {"task_id": job["task_id"], "experiment_name": job["experiment_name"],
                   "job_fingerprint": job["job_fingerprint"], "run_terminal_status": "succeeded",
                   "agent_started": True, "run_id": "run-" + job["task_id"],
                   "infrastructure_error": False, "model_requests": {"reserved": 1},
                   **appworld.prompt_binding(job),
                   "saved": self.behavior != "save_failure", "closed": True}
            if self.behavior == "no_run":
                raw.update(agent_started=False, run_id=None, infrastructure_error=True,
                           model_requests={"reserved": 0}, error="bridge startup failed")
            if self.behavior == "task_failure":
                raw.update(run_terminal_status="failed", reason="task_timeout")
            if self.behavior in {"rate_limit", "rate_limit_once", "model_mismatch", "cost_denied"}:
                raw.update(run_terminal_status="failed", reason=(
                    "llm_model_mismatch" if self.behavior == "model_mismatch" else
                    "llm_request_budget_exhausted" if self.behavior == "cost_denied"
                    else "llm_rate_limited"
                ))
                if self.behavior == "rate_limit_once":
                    self.behavior = "success"
            if self.behavior == "save_once":
                raw["saved"] = False
                self.behavior = "success"
            appworld.write_json(directory / "result.json", raw)
            if self.behavior == "result_then_interrupt":
                self.behavior = "success"
                raise RuntimeError("result durable, controller checkpoint not yet written")
            return raw
        finally:
            self.active_urls.discard(url)

    async def score(self, spec: Any, image: str, data: Path, artifacts: Path,
                    run_id: str, selected: list[str]) -> dict[str, float]:
        self.scorings += 1
        assert not FakeWorld.live
        for task_id in selected:
            path = artifacts / "official" / "outputs" / run_id / "tasks" / task_id / "dbs" / "state.db"
            assert path.read_text() == task_id
        aggregate = {"task_goal_completion": 0.0, "scenario_goal_completion": 0.0}
        if spec.paired_comparison:
            return {"aggregate": aggregate, "individual": {
                task_id: {"success": False, "passes": [],
                          "failures": [{"requirement": "assert answers match."}]}
                for task_id in selected
            }}
        return aggregate

    async def run(self) -> Any:
        return await appworld.run_appworld_suite(self.manifest, self.config, self.provenance, self.artifacts)


async def test_complete_batch_has_isolated_concurrent_worlds_and_independent_official_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=4, workers=2)
    result = await batch.run()
    assert result.benchmark["complete"] is True
    assert result.benchmark["official_metrics"]["task_goal_completion"] == 0.0
    assert result.summary.passed == 4  # Runtime completion is separate from official task success.
    assert all(item.score is None and item.goal_completed is None for item in result.attempts)
    assert batch.max_parallel == 2 and not FakeWorld.live
    assert batch.scorings == 1
    assert len({job["environment_url"] for job in batch.calls}) == 2
    resumed = await batch.run()
    assert resumed.benchmark["complete"] is True
    assert len(batch.calls) == 4  # Completed tasks are never re-run for a better score.


async def test_partial_scenario_selection_masks_reported_sgc_and_preserves_official_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.eval.report import render_markdown_report

    batch = Batch(tmp_path, monkeypatch, count=6)
    batch.ids = [f"{scenario}_{variant}" for scenario in ("first", "second") for variant in (1, 2, 3)]
    assert batch.manifest.appworld is not None
    batch.manifest.appworld.dataset = "train"
    batch.manifest.appworld.task_limit = 4
    original_aggregate = {"task_goal_completion": 75.0, "scenario_goal_completion": 50.0}

    async def score(*args: Any, **kwargs: Any) -> dict[str, float]:
        appworld.write_json(batch.artifacts / "official" / "aggregate.json", original_aggregate)
        return original_aggregate

    monkeypatch.setattr(appworld, "score_outputs", score)
    result = await batch.run()
    benchmark = result.benchmark
    assert benchmark["complete"] is True and benchmark["completed_tasks"] == 4
    assert benchmark["official_source_scenarios_complete"] is True
    assert benchmark["selected_scenario_variants_complete"] is False
    assert "scenario_variants_complete" not in benchmark
    assert benchmark["official_metrics"] == {
        "task_goal_completion": 75.0, "scenario_goal_completion": None,
    }
    assert json.loads((batch.artifacts / "official" / "aggregate.json").read_text()) == original_aggregate
    report = render_markdown_report(result)
    assert "| Scenario Goal Completion | — |" in report
    assert "SGC 不可解释" in report and "official/aggregate.json" in report


@pytest.mark.parametrize("behavior, expected_attempts", [("interrupt", 2), ("result_then_interrupt", 1)])
async def test_resume_consumes_interrupted_attempt_or_recovers_durable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behavior: str, expected_attempts: int,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=1)
    batch.behavior = behavior
    interrupted = await batch.run()
    assert interrupted.benchmark["complete"] is False
    assert json.loads((batch.artifacts / "checkpoint.json").read_text())["inflight"]
    resumed = await batch.run()
    assert resumed.benchmark["complete"] is True
    assert len(batch.calls) == expected_attempts
    assert len(resumed.attempts) == expected_attempts
    assert (batch.artifacts / "attempts" / batch.ids[0] / "attempt-1" / "job.json").exists()
    if expected_attempts == 2:
        assert resumed.attempts[0].status == EvalStatus.error


async def test_resume_after_canonical_publish_before_checkpoint_does_not_overwrite_or_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=1)
    original = appworld.write_json
    fail = True

    def checkpoint_failure(path: Path, value: Any) -> None:
        nonlocal fail
        if path.name == "checkpoint.json" and value.get("results") and fail:
            fail = False
            raise OSError("checkpoint disk interruption")
        original(path, value)

    monkeypatch.setattr(appworld, "write_json", checkpoint_failure)
    first = await batch.run()
    assert first.benchmark["complete"] is False
    canonical = next((batch.artifacts / "official").rglob("state.db"))
    first_mtime = canonical.stat().st_mtime_ns
    resumed = await batch.run()
    assert resumed.benchmark["complete"] is True
    assert len(batch.calls) == 1 and len(resumed.attempts) == 1
    assert canonical.stat().st_mtime_ns == first_mtime


async def test_unconfirmed_old_worker_blocks_world_cleanup_and_the_next_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=1)
    batch.behavior = "interrupt"
    assert (await batch.run()).benchmark["complete"] is False
    checkpoint = batch.artifacts / "checkpoint.json"
    original = checkpoint.read_bytes()
    worlds_before = FakeWorld.serial

    async def refuse_recovery(directory: Path) -> None:
        raise RuntimeError("AppWorld worker exit was not confirmed; refusing recovery")

    async def unexpected_world_cleanup(world: FakeWorld) -> None:
        pytest.fail("old world must remain intact while its worker exit is unconfirmed")

    monkeypatch.setattr(appworld, "stop_recorded_worker", refuse_recovery)
    monkeypatch.setattr(FakeWorld, "close", unexpected_world_cleanup)
    with pytest.raises(RuntimeError, match="exit was not confirmed"):
        await batch.run()
    assert checkpoint.read_bytes() == original
    assert len(batch.calls) == 1 and FakeWorld.serial == worlds_before


@pytest.mark.parametrize("behavior, complete, calls", [("save_once", True, 2), ("save_failure", False, 2)])
async def test_save_failure_retires_world_and_respects_one_infrastructure_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behavior: str, complete: bool, calls: int,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=1)
    batch.behavior = behavior
    result = await batch.run()
    assert result.benchmark["complete"] is complete
    assert len(batch.calls) == calls
    assert batch.calls[0]["environment_url"] != batch.calls[1]["environment_url"]
    assert not FakeWorld.live
    assert batch.scorings == int(complete)


async def test_completed_output_tampering_is_rejected_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=1)
    await batch.run()
    canonical = next((batch.artifacts / "official").rglob("state.db"))
    canonical.write_text("modified after completion")
    with pytest.raises(ValueError, match="output changed"):
        await batch.run()
    assert canonical.read_text() == "modified after completion" and len(batch.calls) == 1


async def test_endpoint_drift_cannot_resume_a_frozen_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=1)
    await batch.run()
    batch.config.llm.base_url = "https://different.invalid/private?token=not-to-be-shown"
    with pytest.raises(ValueError, match="frozen"):
        await batch.run()
    assert len(batch.calls) == 1
    frozen = (batch.artifacts / "freeze.json").read_text()
    assert "not-to-be-shown" not in frozen


async def test_cancellation_closes_worlds_and_same_directory_cannot_run_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=1)
    batch.behavior = "blocked"
    running = asyncio.create_task(batch.run())
    await batch.started.wait()
    with pytest.raises(RuntimeError, match="lock|in use"):
        await batch.run()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert not FakeWorld.live
    assert json.loads((batch.artifacts / "checkpoint.json").read_text())["inflight"]


async def test_saved_initial_world_without_an_agent_run_is_not_a_completed_benchmark_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=1)
    batch.behavior = "no_run"
    result = await batch.run()
    assert len(batch.calls) == 2  # One bounded infrastructure retry.
    assert result.benchmark["completed_tasks"] == 0
    assert result.benchmark["complete"] is False and batch.scorings == 0
    assert all(item.status == EvalStatus.error for item in result.attempts)
    assert not (batch.artifacts / "official").exists()


async def test_real_started_task_failure_is_saved_and_scored_without_quality_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=1)
    batch.behavior = "task_failure"
    result = await batch.run()
    assert len(batch.calls) == 1 and batch.scorings == 1
    assert result.benchmark["complete"] is True
    assert result.summary.failed == 1 and result.attempts[0].score is None


@pytest.mark.parametrize("reason", ["rate_limit", "model_mismatch"])
async def test_rate_limit_or_model_mismatch_pauses_before_allocating_the_next_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=3)
    batch.behavior = reason
    result = await batch.run()
    expected = "llm_rate_limited" if reason == "rate_limit" else "llm_model_mismatch"
    assert result.benchmark["paused_reason"] == expected
    assert result.benchmark["complete"] is False and result.benchmark["completed_tasks"] == 0
    assert len(batch.calls) == 1 and batch.scorings == 0 and not FakeWorld.live
    assert result.attempts[0].error_type == expected
    state = json.loads((batch.artifacts / "checkpoint.json").read_text())
    assert state["paused_reason"] == expected and not state["results"] and not state["inflight"]
    assert not (batch.artifacts / "official").exists()


async def test_rate_limited_resume_retries_the_unfinished_task_once_and_preserves_budget_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.persistence.request_budget import RequestLedger

    batch = Batch(tmp_path, monkeypatch, count=2)
    ledger = RequestLedger(batch.config.llm.request_budget_path, limit=None)
    ledger.reserve()
    batch.behavior = "rate_limit_once"
    first = await batch.run()
    first_attempt = batch.artifacts / "attempts" / batch.ids[0] / "attempt-1" / "result.json"
    original = first_attempt.read_bytes()
    state_before = json.loads((batch.artifacts / "checkpoint.json").read_text())
    ledger.reserve()  # Other recorded history must not increase this checkpoint's cap.
    resumed = await batch.run()
    state_after = json.loads((batch.artifacts / "checkpoint.json").read_text())
    assert first.benchmark["complete"] is False and resumed.benchmark["complete"] is True
    assert [job["task_id"] for job in batch.calls] == [batch.ids[0], batch.ids[0], batch.ids[1]]
    assert state_after["request_limit"] == state_before["request_limit"]
    assert first_attempt.read_bytes() == original
    assert len(resumed.attempts) == 3 and resumed.attempts[0].error_type == "llm_rate_limited"
    assert not FakeWorld.live


@pytest.mark.parametrize("retries", [0, 1])
async def test_limit_pause_resume_never_resets_attempt_slots_or_skips_the_unfinished_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retries: int,
) -> None:
    batch = Batch(tmp_path, monkeypatch, count=2)
    assert batch.manifest.appworld is not None
    batch.manifest.appworld.infrastructure_retries = retries
    batch.behavior = "rate_limit"
    for _ in range(retries + 1):
        assert (await batch.run()).benchmark["paused_reason"] == "llm_rate_limited"
    worlds_before = FakeWorld.serial
    exhausted = await batch.run()
    assert exhausted.benchmark["paused_reason"] == "infrastructure_retries_exhausted"
    assert exhausted.benchmark["complete"] is False
    assert len(batch.calls) == retries + 1 and {job["task_id"] for job in batch.calls} == {batch.ids[0]}
    assert FakeWorld.serial == worlds_before and not FakeWorld.live
    assert len(exhausted.attempts) == retries + 1 and batch.scorings == 0


async def test_guard_drift_cannot_resume_a_frozen_batch(tmp_path, monkeypatch):
    batch = Batch(tmp_path, monkeypatch, count=1)
    batch.config.llm.expected_model = batch.config.llm.default_model
    await batch.run()
    batch.config.llm.expected_model = ""
    with pytest.raises(ValueError, match="frozen"):
        await batch.run()
    assert len(batch.calls) == 1
