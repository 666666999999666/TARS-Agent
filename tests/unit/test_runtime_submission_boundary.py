from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tars_agent.core.bus.events import SessionMessageReceivedEvent
from tars_agent.core.config import TarsConfig
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import LlmResponse
from tars_agent.core.persistence import Database, StateRepository
from tars_agent.core.runner import AgentRunner, RunOutcome
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter


async def terminal(runtime: RuntimeService, run_id: str) -> str:
    async with asyncio.timeout(3):
        while True:
            snapshot = await runtime.get_run(run_id)
            if snapshot.status not in {"queued", "running"}:
                return snapshot.status
            await asyncio.sleep(0.01)


async def test_empty_skill_whitelist_survives_durable_submission_to_provider(
    tmp_path: Path,
) -> None:
    schemas: list[list[dict[str, Any]]] = []
    class Provider:
        async def chat(self, **kwargs: Any) -> LlmResponse:
            schemas.append(kwargs["tool_schemas"])
            return LlmResponse(stop_reason="end_turn", text="no tools were granted")
    db = Database(tmp_path / "state.db")
    await db.create_schema()
    bus = EventBus()
    runner = AgentRunner(TarsConfig(), bus=bus, provider=Provider(),
                         tool_runtime=RuntimeRouter(FakeRuntime()))
    runtime = RuntimeService(db, lambda: runner, bus, artifacts_root=tmp_path / "artifacts")
    skills = tmp_path / ".tars" / "skills"
    skills.mkdir(parents=True)
    (skills / "no-tools.md").write_text(
        "---\nname: no-tools\ndescription: text only\nallowed_tools: []\n---\n"
        "respond without tools", encoding="utf-8",
    )
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        result = await runtime.submit_message(session.id, "/no-tools", client_message_id="empty")
        assert await terminal(runtime, result.run_id) == "succeeded"
        async with db.session() as sql:
            persisted = await StateRepository(sql).get_run(result.run_id)
            assert persisted is not None
            assert persisted.execution_options["tool_whitelist"] == []
        assert schemas == [[]]
    finally:
        await runtime.shutdown()
        await db.dispose()


class MarkerRunner:
    def __init__(self, marker: Path) -> None:
        self.marker = marker
    async def run_and_capture(self, goal: str, **kwargs: Any) -> RunOutcome:
        with self.marker.open("a", encoding="utf-8") as stream:
            stream.write("executed once\n")
        return RunOutcome(status="success", result="done", reason=None)


@pytest.mark.parametrize("disconnect", [False, True])
async def test_committed_submission_is_scheduled_despite_notification_failure_or_disconnect(
    tmp_path: Path, disconnect: bool,
) -> None:
    db = Database(tmp_path / "state.db")
    await db.create_schema()
    marker = tmp_path / "effect.txt"
    bus = EventBus()
    observed = asyncio.Event()
    release = asyncio.Event()
    async def failing_subscriber(event: Any) -> None:
        if isinstance(event, SessionMessageReceivedEvent):
            observed.set()
            if disconnect:
                await release.wait()
            else:
                raise OSError("notification delivery failed")
    bus.subscribe(failing_subscriber)
    runtime = RuntimeService(db, lambda: MarkerRunner(marker), bus,  # type: ignore[arg-type,return-value]
                             artifacts_root=tmp_path / "artifacts")
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submission = asyncio.create_task(runtime.submit_message(
            session.id, "write marker", client_message_id="same-logical-message",
        ))
        await observed.wait()
        if disconnect:
            submission.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submission
            release.set()
        else:
            await submission
        # An ambiguous transport retry must reuse the already scheduled Run.
        retried = await runtime.submit_message(
            session.id, "write marker", client_message_id="same-logical-message",
        )
        assert retried.deduplicated
        assert await terminal(runtime, retried.run_id) == "succeeded"
        assert marker.read_text(encoding="utf-8") == "executed once\n"
    finally:
        release.set()
        await runtime.shutdown()
        await db.dispose()
