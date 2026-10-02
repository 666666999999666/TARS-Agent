from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from tars_agent.core.eval import appworld
from tars_agent.core.eval.appworld_bridge import TaskCodeBridge, loopback_url
from tars_agent.core.eval.models import AppWorldSpec, EvalStatus, EvalSuiteManifest
from tars_agent.core.eval.runner import load_manifest


@pytest.mark.parametrize("url", [
    "https://127.0.0.1:8000", "http://example.com:8000", "http://127.0.0.1",
    "http://user:password@127.0.0.1:8000", "http://127.0.0.1:8000/evaluate",
    "http://127.0.0.1:8000?redirect=x", "file:///tmp/world",
])
def test_bridge_rejects_non_loopback_or_management_url(url: str) -> None:
    with pytest.raises(ValueError):
        loopback_url(url)


async def test_bridge_binds_task_and_exposes_only_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"output": "result"})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(handle), **kwargs,
    ))
    bridge = TaskCodeBridge("http://127.0.0.1:8123", "fixed-task")
    assert await bridge.execute("print(1)") == "result"
    assert len(requests) == 1
    assert requests[0].url.path == "/execute"
    assert b'"task_id":"fixed-task"' in requests[0].content
    with pytest.raises(ValueError):
        await bridge.execute("x" * 65537)
    assert len(requests) == 1


def test_test_normal_disallows_task_sampling() -> None:
    with pytest.raises(ValidationError, match="complete"):
        AppWorldSpec(data_root="data", dataset="test_normal", task_limit=4)
    assert AppWorldSpec(data_root="data", dataset="dev", task_limit=4).task_limit == 4


def test_manifest_keeps_appworld_explicit_and_full() -> None:
    manifest = load_manifest(Path("evals/appworld-test-normal.json"))
    assert manifest.adapter == "appworld"
    assert manifest.appworld is not None
    assert manifest.appworld.task_limit is None
    assert manifest.tasks == []
    modified = manifest.model_dump()
    modified["default_repetitions"] = 3
    with pytest.raises(ValidationError, match="once"):
        EvalSuiteManifest.model_validate(modified)


def test_runtime_result_does_not_invent_official_goal_score() -> None:
    raw = {"saved": True, "closed": True, "run_terminal_status": "failed",
           "agent_started": True, "run_id": "run", "infrastructure_error": False,
           "model_requests": {"reserved": 1},
           "reason": "llm_request_budget_exhausted"}
    result = appworld.attempt_result(raw, "a", 1)
    assert result.status == EvalStatus.error  # A shared budget stop is an unfinished experiment.
    assert result.score is None and result.goal_completed is None
    assert result.usage.input_tokens is None
    raw["saved"] = False
    assert appworld.attempt_result(raw, "a", 1).status == EvalStatus.error


def test_mount_normalization_preserves_windows_and_posix_identity() -> None:
    assert appworld.canonical_mount("/run/desktop/mnt/host/c/Users/Test") == "c:/users/test"
    assert appworld.canonical_mount("C:\\Users\\Test\\") == "c:/users/test"
    assert appworld.canonical_mount("/run/Other") != appworld.canonical_mount("/run/other")


async def test_cleanup_refuses_foreign_identity_before_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    data = tmp_path / "data"
    data.mkdir()
    world = appworld.WorldContainer("image-a", data, tmp_path / "out", "owner-a")
    world.container_id = "a" * 64

    def fake_command(args: list[str], **kwargs: Any) -> str:
        calls.append(args)
        if args[1] == "ps":
            return "a" * 64
        return '[{"Id":"wrong", "Image":"image-a", "Config":{"Labels":{}}}]'

    monkeypatch.setattr(appworld, "command", fake_command)
    with pytest.raises(RuntimeError, match="ownership"):
        await world.close()
    assert all("rm" not in item for item in calls)


def test_data_fingerprint_detects_any_modified_fixture(tmp_path: Path) -> None:
    (tmp_path / "fixture").write_bytes(b"first")
    first = appworld.data_digest(tmp_path)
    (tmp_path / "fixture").write_bytes(b"second")
    assert appworld.data_digest(tmp_path) != first


@pytest.mark.parametrize("change", [None, "missing", "duplicate_variant", "missing_scenario"])
def test_pinned_test_normal_requires_all_168_ids_and_56_complete_scenarios(change: str | None) -> None:
    ids = [f"scenario{index}_{number}" for index in range(56) for number in (1, 2, 3)]
    metadata = {"task_ids": ids, "scenarios": {
        task_id: [task_id.rsplit("_", 1)[0], int(task_id.rsplit("_", 1)[1])] for task_id in ids
    }}
    if change == "missing":
        ids.pop()
    elif change == "duplicate_variant":
        metadata["scenarios"][ids[-1]][1] = 1
    elif change == "missing_scenario":
        metadata["scenarios"][ids[-1]][0] = "scenario0"
    if change:
        with pytest.raises(ValueError, match="168 tasks"):
            appworld.validate_task_set("test_normal", metadata)
    else:
        appworld.validate_task_set("test_normal", metadata)
        assert metadata["task_count"] == 168 and metadata["scenario_count"] == 56
        assert metadata["scenario_variants_complete"] is True
        assert appworld.selected_scenarios_complete(ids, metadata) is True
        aggregate = {"task_goal_completion": 75.0, "scenario_goal_completion": 50.0}
        assert appworld.reported_official_metrics(
            aggregate, selected_scenario_variants_complete=True,
        ) == aggregate


def test_selected_scenarios_are_checked_independently_of_complete_source() -> None:
    ids = [f"{scenario}_{variant}" for scenario in ("first", "second") for variant in (1, 2, 3)]
    selected = ids[:4]
    metadata = {"task_ids": ids, "scenarios": {
        task_id: [task_id.rsplit("_", 1)[0], int(task_id.rsplit("_", 1)[1])] for task_id in ids
    }}
    appworld.validate_task_set("train", metadata)
    assert metadata["scenario_variants_complete"] is True
    assert appworld.selected_scenarios_complete(selected, metadata) is False
    aggregate = {"task_goal_completion": 75.0, "scenario_goal_completion": 50.0}
    metrics = appworld.reported_official_metrics(
        aggregate, selected_scenario_variants_complete=False,
    )
    assert metrics["task_goal_completion"] == 75.0
    assert metrics["scenario_goal_completion"] is None
    assert aggregate["scenario_goal_completion"] == 50.0
