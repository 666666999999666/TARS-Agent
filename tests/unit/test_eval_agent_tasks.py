from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tars_agent.core.bus.events import LlmModelSelectedEvent, LlmUsageEvent
from tars_agent.core.config import TarsConfig
from tars_agent.core.eval import internal
from tars_agent.core.eval.models import EvalStatus, EvalTaskSpec, GraderSpec
from tars_agent.core.eval.runner import load_manifest
from tars_agent.core.llm.types import LlmResponse, ToolCallBlock
from tars_agent.core.runner import AgentRunner
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter


class ScriptedProvider:
    def __init__(self, model: str, *, background: bool = False, hang: bool = False,
                 fail_planner: bool = False, mutate_input: bool = False) -> None:
        self.model = model
        self.background = background
        self.hang = hang
        self.fail_planner = fail_planner
        self.mutate_input = mutate_input
        self.root: str | None = None
        self.calls: Counter[str] = Counter()
        self.schemas: dict[str, set[str]] = {}
        self.root_finished = asyncio.Event()
        self.hanging_request_started = asyncio.Event()
        self.hanging_request_run_id: str | None = None

    async def chat(self, messages: list[dict[str, Any]], tool_schemas: list[dict[str, Any]],
                   bus: Any, run_id: str, **kwargs: Any) -> LlmResponse:
        self.root = self.root or run_id
        self.calls[run_id] += 1
        count = self.calls[run_id]
        names = {tool["name"] for tool in tool_schemas}
        self.schemas[run_id] = names
        await bus.publish(LlmModelSelectedEvent(run_id=run_id, model=self.model, ts="2026-09-29T00:00:00Z"))
        if run_id != self.root and "write_file" in names and self.background:
            await self.root_finished.wait()
            if self.hang:
                self.hanging_request_run_id = run_id
                self.hanging_request_started.set()
                await asyncio.Event().wait()
        if run_id != self.root and "规划专家" in (kwargs.get("system") or "") and self.fail_planner:
            raise RuntimeError("scripted planner failure")
        await bus.publish(LlmUsageEvent(
            run_id=run_id, input_tokens=10, output_tokens=2,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
            ts="2026-09-29T00:00:00Z",
        ))
        if run_id == self.root and "spawn_agent" in names:
            if count <= 3:
                role = ["planner", "executor", "reviewer"][count - 1]
                return LlmResponse(stop_reason="tool_use", tool_calls=[ToolCallBlock(
                    f"spawn-{count}", "spawn_agent", {
                        "description": role, "subagent_type": role, "prompt": "write result.txt",
                        "run_in_background": self.background and role == "executor",
                    },
                )])
        elif "write_file" in names and count == 1:
            writes = [ToolCallBlock(
                "write", "write_file", {"path": "result.txt", "content": "verified"},
            )]
            if self.mutate_input:
                writes.append(ToolCallBlock(
                    "change-input", "write_file", {"path": "input.txt", "content": "changed"},
                ))
            return LlmResponse(stop_reason="tool_use", tool_calls=writes)
        return LlmResponse(stop_reason="end_turn", text="done")


def setup_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
                  cleanup_confirmed: bool | None = True, **options: Any) -> tuple[
    internal.InternalEvalAdapter, ScriptedProvider,
]:
    monkeypatch.setenv("TARS_HOME", str(tmp_path / "trusted-home"))
    config = TarsConfig()
    config.llm.api_key = "unit-test-placeholder"
    config.llm.request_budget_path = tmp_path / "ledger.sqlite3"
    provider = ScriptedProvider(config.llm.default_model, **options)
    monkeypatch.setattr(internal, "build_runtime_router", lambda sandbox: RuntimeRouter(
        FakeRuntime(cleanup_confirmed=cleanup_confirmed), allow_host_fallback=False,
    ))
    original_mkdtemp = internal.tempfile.mkdtemp

    def scoped_mkdtemp(**kwargs: Any) -> str:
        kwargs.setdefault("dir", tmp_path)
        return original_mkdtemp(**kwargs)

    monkeypatch.setattr(internal.tempfile, "mkdtemp", scoped_mkdtemp)

    def runner_factory(actual_config: TarsConfig, **kwargs: Any) -> AgentRunner:
        async def root_finished(event: Any) -> None:
            if event.type == "run.finished" and event.run_id == provider.root:
                provider.root_finished.set()
        kwargs["bus"].subscribe(root_finished)
        return AgentRunner(actual_config, provider=provider, **kwargs)

    monkeypatch.setattr(internal, "AgentRunner", runner_factory)
    return internal.InternalEvalAdapter(config, artifact_root=tmp_path / "evidence"), provider


def task(mode: str = "single", timeout_s: float = 5) -> EvalTaskSpec:
    return EvalTaskSpec(
        id="write-" + mode, source_task_id="write", agent_mode=mode,  # type: ignore[arg-type]
        goal="Write verified to result.txt", timeout_s=timeout_s,
        tool_whitelist=["read_file", "list_dir", "write_file"],
        fixture_files={"input.txt": "input sentinel\n"},
        grader=GraderSpec(kind="file_equals", path="result.txt", expected="verified"),
        metadata={"protected_paths": ["input.txt"]},
    )


@pytest.mark.parametrize("mode, calls, children", [("single", 2, 0), ("orchestrated", 8, 3)])
async def test_attempt_uses_durable_submission_and_counts_entire_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, calls: int, children: int,
) -> None:
    adapter, provider = setup_adapter(tmp_path, monkeypatch)
    result = await adapter.run_attempt(task(mode), repetition=1, eval_run_id="paired")
    assert result.status == EvalStatus.passed, result.error
    assert result.evaluation["execution_path"] == "runtime_service"
    assert result.evaluation["tree_terminal"] is True
    assert result.evaluation["subagent_count"] == children
    assert result.usage.input_tokens == calls * 10
    assert result.usage.output_tokens == calls * 2
    assert sum(provider.calls.values()) == calls
    assert ("spawn_agent" in provider.schemas[provider.root]) is (mode == "orchestrated")
    if children:
        assert result.evaluation["workflow_followed"] is True
        assert result.evaluation["spawn_roles"] == ["planner", "executor", "reviewer"]
        assert all(names <= provider.schemas[provider.root] for names in provider.schemas.values())
    evidence = Path(result.evaluation["evidence_directory"])
    assert (evidence / "workspace" / "result.txt").read_text() == "verified"
    assert (evidence / "workspace" / "input.txt").read_text() == "input sentinel\n"
    assert json.loads((evidence / "attempt.json").read_text())["status"] == "passed"
    with sqlite3.connect(evidence / "state" / "state.db") as database:
        assert database.execute("SELECT COUNT(*) FROM turns").fetchone()[0] == 1
        assert database.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == children + 1
        assert database.execute("SELECT raw_content FROM turns").fetchone()[0].startswith("/eval-task ")
        assert database.execute("SELECT COUNT(*) FROM events").fetchone()[0] > 0
    assert result.cleanup.workspace_removed is True


async def test_grading_and_accounting_wait_for_background_child_after_parent_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, provider = setup_adapter(tmp_path, monkeypatch, background=True)
    result = await adapter.run_attempt(task("orchestrated"), repetition=1, eval_run_id="late")
    assert provider.root_finished.is_set()
    # The executor finishes after the parent, so accounting must wait but the
    # claimed planner -> executor -> reviewer workflow must not pass.
    assert result.status == EvalStatus.failed and result.error_type == "run_tree_incomplete"
    assert result.evaluation["workflow_followed"] is False
    assert result.evaluation["tree_terminal"] is True
    assert result.usage.input_tokens == 80
    assert (Path(result.evaluation["evidence_directory"]) / "workspace" / "result.txt").read_text() == "verified"


async def test_timeout_cancels_descendants_and_preserves_observed_usage_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, provider = setup_adapter(tmp_path, monkeypatch, background=True, hang=True)
    case = task("orchestrated", timeout_s=0.3)
    timers: list[asyncio.Timeout] = []

    def controlled_attempt_timeout(delay: float | None) -> asyncio.Timeout:
        assert delay == case.timeout_s
        timer = asyncio.timeout(None)
        timers.append(timer)
        return timer

    # Keep the real Timeout implementation, but arm it only after the intended
    # request has started. Coverage overhead must not move expiry before that event.
    adapter_asyncio = SimpleNamespace(**vars(asyncio))
    adapter_asyncio.timeout = controlled_attempt_timeout
    monkeypatch.setattr(internal, "asyncio", adapter_asyncio)
    operation = asyncio.create_task(adapter.run_attempt(case, repetition=1, eval_run_id="timeout"))
    try:
        async with asyncio.timeout(30):  # Guard only against a broken fixture/deadlock.
            await provider.hanging_request_started.wait()
        hanging_run = provider.hanging_request_run_id
        assert hanging_run is not None and hanging_run != provider.root
        assert len(timers) == 1 and timers[0].when() is None
        timers[0].reschedule(asyncio.get_running_loop().time())
        result = await asyncio.wait_for(operation, 30)
    finally:
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
    assert provider.root_finished.is_set()
    assert result.error_type == "timeout"
    assert result.status == EvalStatus.error
    assert result.evaluation["observed_model_calls"] > 0
    assert result.evaluation["observed_tokens"]["input"] > 0
    assert result.usage.input_tokens is None  # The hanging call supplied no complete usage.
    assert result.evaluation["usage_by_run"][hanging_run]["input_tokens"] is None
    assert result.evaluation["observed_model_calls"] == sum(provider.calls.values())
    assert result.evaluation["tree_terminal"] is True
    assert next(row["status"] for row in result.evaluation["runs"] if row["run_id"] == hanging_run) == "cancelled"
    assert next(row["status"] for row in result.evaluation["runs"] if row["run_id"] == provider.root) == "succeeded"
    evidence = Path(result.evaluation["evidence_directory"])
    assert (evidence / "workspace" / "input.txt").exists()
    saved = json.loads((evidence / "attempt.json").read_text())
    assert saved["error_type"] == "timeout" and saved["evaluation"]["observed_model_calls"] > 0
    assert result.cleanup.workspace_removed is True


@pytest.mark.parametrize("confirmed", [False, None])
async def test_cleanup_residue_or_unknown_blocks_grading_and_preserves_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed: bool | None,
) -> None:
    adapter, _ = setup_adapter(tmp_path, monkeypatch, cleanup_confirmed=confirmed)

    def unexpected_grade(*args: Any) -> Any:
        raise AssertionError("grading must not start before physical cleanup confirmation")

    monkeypatch.setattr(internal, "_grade", unexpected_grade)
    result = await adapter.run_attempt(task(), repetition=1, eval_run_id="unconfirmed")
    assert result.status == EvalStatus.error and result.error_type == "runtime_cleanup_failed"
    assert result.cleanup.runtime_cleanup_completed is confirmed
    assert result.evaluation["runtime_cleanup_confirmation"]["source"] == "fake_runtime"
    assert result.evaluation["grading_performed"] is False
    assert result.cleanup.workspace_removed is False
    original = Path(result.evaluation["retained_temporary_root"])
    assert (original / "workspace" / "result.txt").read_text() == "verified"


async def test_failed_spawn_requests_cannot_stand_in_for_completed_role_stages() -> None:
    from tars_agent.core.bus.events import (
        RunFinishedEvent,
        ToolCallFailedEvent,
        ToolCallStartedEvent,
    )

    events = internal._EventAccumulator()
    for role in ("planner", "executor", "reviewer"):
        await events.record(ToolCallStartedEvent(
            run_id="root", tool_use_id=role, tool_name="spawn_agent",
            params={"subagent_type": role}, ts="2026-09-29T00:00:00Z",
        ))
        await events.record(ToolCallFailedEvent(
            run_id="root", tool_use_id=role, tool_name="spawn_agent", error_class="runtime_error",
            error_message="could not create child", elapsed_ms=0, ts="2026-09-29T00:00:00Z",
        ))
    await events.record(RunFinishedEvent(
        run_id="root", status="success", steps=3, ts="2026-09-29T00:00:00Z",
    ))
    assert events.spawn_roles == ["planner", "executor", "reviewer"]
    assert events.workflow_followed("root") is False


async def test_first_manifest_write_failure_keeps_original_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = setup_adapter(tmp_path, monkeypatch)

    def fail_record(*args: Any) -> None:
        raise OSError("manifest storage unavailable")

    monkeypatch.setattr(internal, "_write_attempt_record", fail_record)
    result = await adapter.run_attempt(task(), repetition=1, eval_run_id="manifest-failure")
    assert result.status == EvalStatus.error and result.error_type == "evidence_write_failed"
    assert result.cleanup.workspace_removed is False
    original = Path(result.evaluation["retained_temporary_root"])
    assert (original / "workspace" / "result.txt").read_text() == "verified"


async def test_final_manifest_write_failure_keeps_original_evidence_and_no_passed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = setup_adapter(tmp_path, monkeypatch)
    original_write = internal._write_attempt_record
    writes = 0

    def fail_final(destination: Path, result: Any) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("final manifest replace failed")
        original_write(destination, result)

    monkeypatch.setattr(internal, "_write_attempt_record", fail_final)
    result = await adapter.run_attempt(task(), repetition=1, eval_run_id="final-write-failure")
    assert result.status == EvalStatus.error and result.error_type == "evidence_write_failed"
    evidence = Path(result.evaluation["evidence_directory"])
    assert (evidence / "workspace" / "result.txt").read_text() == "verified"
    saved = json.loads((evidence / "attempt.json").read_text())
    assert saved["status"] == "error"
    assert saved["evaluation"]["evidence_finalization"] == "pending"


async def test_failed_child_prevents_success_even_if_final_file_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = setup_adapter(tmp_path, monkeypatch, fail_planner=True)
    result = await adapter.run_attempt(task("orchestrated"), repetition=1, eval_run_id="child-failure")
    assert result.status == EvalStatus.failed
    assert result.error_type == "run_tree_incomplete"
    assert result.evaluation["grader"]["matched"] is True
    assert any(row["status"] == "failed" for row in result.evaluation["runs"])


async def test_failed_evidence_copy_keeps_original_attempt_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = setup_adapter(tmp_path, monkeypatch)

    def fail_copy(source: Path, destination: Path) -> dict[str, Any]:
        raise OSError("simulated evidence storage failure")

    monkeypatch.setattr(internal, "_copy_evidence", fail_copy)
    result = await adapter.run_attempt(task(), repetition=1, eval_run_id="retain")
    assert result.error_type == "evidence_copy_failed"
    assert result.cleanup.workspace_removed is False
    original = Path(result.evaluation["retained_temporary_root"])
    assert (original / "workspace" / "result.txt").read_text() == "verified"


async def test_correct_output_cannot_hide_changes_to_protected_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _ = setup_adapter(tmp_path, monkeypatch, mutate_input=True)
    result = await adapter.run_attempt(task(), repetition=1, eval_run_id="protected")
    assert result.status == EvalStatus.failed
    assert result.evaluation["grader"]["matched"] is True
    assert result.collateral_damage is True
    assert result.evaluation["changed_protected_paths"] == ["input.txt"]


def test_task_suite_has_eight_matched_tasks_four_categories_and_48_attempts() -> None:
    manifest = load_manifest(Path(__file__).parents[2] / "evals" / "internal-agent-tasks.json")
    pairs: dict[str, list[EvalTaskSpec]] = defaultdict(list)
    for item in manifest.tasks:
        pairs[item.source_task_id].append(item)
    assert len(pairs) == 8
    assert len(manifest.tasks) * manifest.default_repetitions == 48
    split_counts: Counter[str] = Counter()
    categories: set[str] = set()
    for pair in pairs.values():
        assert len(pair) == 2 and {item.agent_mode for item in pair} == {"single", "orchestrated"}
        left, right = pair
        assert left.goal == right.goal
        assert left.fixture_files == right.fixture_files
        assert left.grader == right.grader
        assert left.tool_whitelist == right.tool_whitelist
        assert left.metadata == right.metadata
        split_counts[left.metadata["split"]] += 1
        categories.add(left.metadata["category"])
    assert split_counts == {"development": 4, "holdout": 4}
    assert len(categories) == 4
