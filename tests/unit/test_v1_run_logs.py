from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from tars_agent.core.bus.events import RunStartedEvent, SkillInvokedEvent
from tars_agent.core.events.bus import EventBus
from tars_agent.core.persistence import Database
from tars_agent.core.runner import RunOutcome
from tars_agent.core.runtime.service import RuntimeService


async def test_parallel_runs_record_only_own_events_and_committed_terminal(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    bus, both_running = EventBus(), asyncio.Barrier(2)

    class Runner:
        async def run_and_capture(self, goal: str, **kwargs: Any) -> RunOutcome:
            await both_running.wait()
            await bus.publish(RunStartedEvent(run_id=kwargs["run_id"], goal=goal, ts="test"))
            return RunOutcome(status="success", result=goal, reason=None)

    runtime = RuntimeService(
        database, Runner, bus,  # type: ignore[arg-type]
        artifacts_root=tmp_path / "artifacts",
    )
    try:
        first = await runtime.create_session("chat", workspace_root=tmp_path)
        second = await runtime.create_session("chat", workspace_root=tmp_path)
        runs = await asyncio.gather(
            runtime.submit_message(first.id, "first"), runtime.submit_message(second.id, "second"),
        )
        await asyncio.gather(*(runtime.supervisor.wait(run.run_id) for run in runs))
        for session, run in zip((first, second), runs, strict=True):
            path = tmp_path / "artifacts" / "sessions" / session.id / "runs" / run.run_id / "events.jsonl"
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            assert {event["run_id"] for event in events} == {run.run_id}
            terminal = [event for event in events if event["type"] == "run.finished"]
            assert len(terminal) == 1 and terminal[0]["status"] == "success"
            assert (await runtime.get_run(run.run_id)).status == "succeeded"
        assert not runtime._event_writers
    finally:
        await runtime.shutdown()
        await database.dispose()


async def test_recovery_records_interruption_without_executing_a_task(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()

    def forbidden_runner() -> Any:
        raise AssertionError("recovery must not replay execution")

    runtime = RuntimeService(database, forbidden_runner, EventBus(), artifacts_root=tmp_path / "artifacts")
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        queued = await runtime._create_message_run(
            session.id, "interrupted work", client_message_id="original", run_id=None,
        )
        assert await runtime.recover_interrupted() == 1
        assert await runtime.recover_interrupted() == 0
        path = tmp_path / "artifacts" / "sessions" / session.id / "runs" / queued.run_id / "events.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(events) == 1
        assert events[0]["status"] == "interrupted"
        assert events[0]["reason"] == "daemon_restarted"
        assert (await runtime.get_session(session.id)).status == "ready"
    finally:
        await runtime.shutdown()
        await database.dispose()


async def _assert_preparation_failure_finalized(
    runtime: RuntimeService, session_id: str, run_id: str, tmp_path: Path,
) -> None:
    assert (await runtime.get_run(run_id)).status == "failed"
    session = await runtime.get_session(session_id)
    assert session.status == "ready"
    assert session.active_run_id is None
    path = tmp_path / "artifacts" / "sessions" / session_id / "runs" / run_id / "events.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    terminal = [event for event in events if event["type"] == "run.finished"]
    assert len(terminal) == 1 and terminal[0]["status"] == "failed"
    assert {event["run_id"] for event in events} == {run_id}
    assert not runtime._event_writers


async def test_skill_event_failure_finalizes_run_and_closes_own_log(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    bus = EventBus()
    skills = tmp_path / ".tars" / "skills"
    skills.mkdir(parents=True)
    (skills / "broken-event.md").write_text("Summarize $ARGUMENTS", encoding="utf-8")

    async def fail_skill_event(event: Any) -> None:
        if isinstance(event, SkillInvokedEvent):
            raise RuntimeError("skill event observer unavailable")

    bus.subscribe(fail_skill_event)

    def forbidden_runner() -> Any:
        raise AssertionError("execution must not begin after preparation failed")

    runtime = RuntimeService(database, forbidden_runner, bus, artifacts_root=tmp_path / "artifacts")
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "/broken-event a short note")
        await asyncio.gather(runtime.supervisor.wait(submitted.run_id), return_exceptions=True)
        await _assert_preparation_failure_finalized(runtime, session.id, submitted.run_id, tmp_path)
    finally:
        await runtime.shutdown()
        await database.dispose()


async def test_runner_factory_failure_finalizes_run_and_closes_own_log(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()

    def broken_runner() -> Any:
        raise RuntimeError("runner factory unavailable")

    runtime = RuntimeService(
        database, broken_runner, EventBus(), artifacts_root=tmp_path / "artifacts",
    )
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        submitted = await runtime.submit_message(session.id, "execute a task")
        await asyncio.gather(runtime.supervisor.wait(submitted.run_id), return_exceptions=True)
        await _assert_preparation_failure_finalized(runtime, session.id, submitted.run_id, tmp_path)
    finally:
        await runtime.shutdown()
        await database.dispose()
