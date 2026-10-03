from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSessionTransaction

from tars_agent.core.bus.envelope import HandlerError
from tars_agent.core.bus.events import SessionMessageReceivedEvent
from tars_agent.core.config import TarsConfig
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import LlmResponse
from tars_agent.core.persistence import Database, StateRepository
from tars_agent.core.runner import AgentRunner, RunOutcome
from tars_agent.core.runtime.service import RUN_INVALID_STATE, RuntimeService
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


class RetryMarkerRunner(MarkerRunner):
    def __init__(self, marker: Path) -> None:
        super().__init__(marker)
        self.first_attempt = True

    async def run_and_capture(self, goal: str, **kwargs: Any) -> RunOutcome:
        if self.first_attempt:
            self.first_attempt = False
            return RunOutcome(status="failed", result="", reason="synthetic failure")
        return await super().run_and_capture(goal, **kwargs)


async def failed_retry_case(tmp_path: Path) -> tuple[Database, RuntimeService, str]:
    db = Database(tmp_path / "state.db")
    await db.create_schema()
    runner = RetryMarkerRunner(tmp_path / "effect.txt")
    runtime = RuntimeService(db, lambda: runner, EventBus(),  # type: ignore[arg-type,return-value]
                             artifacts_root=tmp_path / "artifacts")
    session = await runtime.create_session("chat", workspace_root=tmp_path)
    original = await runtime.submit_message(session.id, "retry marker")
    await runtime.supervisor.wait(original.run_id)
    assert (await runtime.get_run(original.run_id)).status == "failed"
    return db, runtime, original.run_id


@pytest.mark.parametrize("shutdown_during_commit", [False, True])
async def test_retry_commit_survives_cancelled_caller_and_is_owned_until_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shutdown_during_commit: bool,
) -> None:
    db, runtime, original_id = await failed_retry_case(tmp_path)
    committing = asyncio.Event()
    release_commit = asyncio.Event()
    committed = asyncio.Event()
    transaction_exit = AsyncSessionTransaction.__aexit__
    armed = True

    async def controlled_transaction_exit(
        transaction: AsyncSessionTransaction, exc_type: Any, exc_value: Any, traceback: Any,
    ) -> None:
        nonlocal armed
        if armed and exc_type is None:
            armed = False
            committing.set()
            await release_commit.wait()
            # This is the project's actual Database/SQLite commit, running in
            # SQLAlchemy's shielded context-manager task after caller cancellation.
            await transaction_exit(transaction, exc_type, exc_value, traceback)
            committed.set()
        else:
            await transaction_exit(transaction, exc_type, exc_value, traceback)

    monkeypatch.setattr(AsyncSessionTransaction, "__aexit__", controlled_transaction_exit)
    shutdown: asyncio.Task[None] | None = None
    retry = asyncio.create_task(runtime.retry_run(original_id))
    try:
        await asyncio.wait_for(committing.wait(), timeout=1)
        retry.cancel()
        with pytest.raises(asyncio.CancelledError):
            await retry
        if shutdown_during_commit:
            shutdown = asyncio.create_task(runtime.shutdown())
            await asyncio.sleep(0)
            assert not shutdown.done()
        release_commit.set()
        await asyncio.wait_for(committed.wait(), timeout=1)
        original = await runtime.get_run(original_id)
        async with db.session() as sql:
            latest = await StateRepository(sql).latest_run(original.session_id)
            assert latest is not None
            assert latest.id != original_id
            assert latest.retry_of_run_id == original_id
            retry_id = latest.id
        if shutdown is not None:
            await asyncio.wait_for(shutdown, timeout=3)
            assert (await runtime.get_run(retry_id)).status in {"succeeded", "cancelled"}
            assert not runtime.supervisor.active_run_ids()
        else:
            assert await terminal(runtime, retry_id) == "succeeded"
            assert (tmp_path / "effect.txt").read_text(encoding="utf-8") == "executed once\n"
        snapshot = await runtime.get_session(original.session_id)
        assert snapshot.status == "ready"
        assert snapshot.active_run_id is None
        assert not runtime._submissions
    finally:
        release_commit.set()
        if shutdown is not None:
            await asyncio.gather(shutdown, return_exceptions=True)
        await runtime.shutdown()
        await db.dispose()


async def test_retry_after_shutdown_rejects_without_creating_attempt(tmp_path: Path) -> None:
    db, runtime, original_id = await failed_retry_case(tmp_path)
    try:
        await runtime.shutdown()
        with pytest.raises(HandlerError) as captured:
            await runtime.retry_run(original_id)
        assert captured.value.code == RUN_INVALID_STATE
        assert str(captured.value) == "runtime is shutting down"
        original = await runtime.get_run(original_id)
        async with db.session() as sql:
            latest = await StateRepository(sql).latest_run(original.session_id)
            assert latest is not None
            assert latest.id == original_id
        assert (await runtime.get_session(original.session_id)).status == "ready"
    finally:
        await runtime.shutdown()
        await db.dispose()
