"""Explicit real-model, real-Docker runtime test; this does not claim TUI acceptance."""
from __future__ import annotations

import asyncio
import copy
import json
import uuid
from pathlib import Path

import pytest

from tars_agent.core.config import get_config
from tars_agent.core.events.bus import EventBus
from tars_agent.core.permissions.manager import PermissionManager
from tars_agent.core.permissions.policy import PermissionDecision, ToolPolicy
from tars_agent.core.persistence import Database
from tars_agent.core.runner import AgentRunner
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.tools.runtime import initialize_runtime_router

pytestmark = pytest.mark.integration


async def test_run_e2e_reads_file_and_succeeds(
    tmp_path: Path, pytestconfig: pytest.Config,
) -> None:
    if not pytestconfig.getoption("--real-model"):
        pytest.skip("requires explicit --real-model authorization")
    config = copy.deepcopy(get_config())
    if not (config.llm.api_key or config.llm.anthropic_api_key):
        pytest.skip("trusted model credential not configured")
    config.agent.max_steps = 5
    config.llm.max_tokens = min(config.llm.max_tokens, 512)
    config.sandbox.mode = "required"
    marker = uuid.uuid4().hex
    (tmp_path / "sample.txt").write_text(f"File marker: {marker}\n", encoding="utf-8")
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    bus = EventBus()
    runtime = None
    service = None
    try:
        runtime = await initialize_runtime_router(config.sandbox)
        manager = PermissionManager(policies={
            name: ToolPolicy(default=PermissionDecision.ALLOW if name == "read_file" else PermissionDecision.DENY)
            for name in ("read_file", "write_file", "list_dir", "bash", "note_save",
                         "task_create", "task_get", "task_update", "task_list")
        })
        service = RuntimeService(
            database,
            lambda: AgentRunner(config, bus=bus, tool_runtime=runtime, permission_manager=manager),
            bus,
            artifacts_root=tmp_path / "artifacts",
            tool_runtime=runtime,
        )
        session = await service.create_session("chat", workspace_root=tmp_path)
        submitted = await service.submit_message(
            session.id, "Use read_file to read sample.txt and report its exact file marker.",
        )
        await asyncio.wait_for(service.supervisor.wait(submitted.run_id), timeout=180)
        snapshot = await service.get_run(submitted.run_id)
        assert snapshot.status == "succeeded", snapshot.reason
        assert marker in snapshot.result["text"]
        path = tmp_path / "artifacts/sessions" / session.id / "runs" / submitted.run_id / "events.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert events[0]["type"] == "run.started"
        assert events[-1]["type"] == "run.finished"
        assert events[-1]["status"] == "success"
        assert all(event["run_id"] == submitted.run_id for event in events)
        assert any(event["type"] == "tool.call_finished" and event["tool_name"] == "read_file" for event in events)
        assert any(event["type"] == "llm.usage" for event in events)
    finally:
        if service is not None:
            await service.shutdown()
        if runtime is not None:
            await runtime.cleanup()
        await database.dispose()
