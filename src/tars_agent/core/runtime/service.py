from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select, update

from tars_agent.core.artifacts import ArtifactStore
from tars_agent.core.bus.envelope import HandlerError
from tars_agent.core.bus.events import (
    ContextCompactedEvent,
    RunFinishedEvent,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionMessageReceivedEvent,
    SessionResumedEvent,
    SessionWaitingForInputEvent,
    SkillInvokedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    ToolExecutionStartedEvent,
)
from tars_agent.core.compact.budget import (
    ContextBudget,
    ContextBudgetError,
    is_user_input,
    message_estimate,
    validate_tool_pairs,
)
from tars_agent.core.compact.compactor import CompactionResult, Compactor
from tars_agent.core.config import LlmConfig
from tars_agent.core.events.bus import EventBus
from tars_agent.core.events.writer import EventWriter
from tars_agent.core.llm.base import LLMProvider
from tars_agent.core.persistence import (
    CompactionRecord,
    Database,
    MessageRecord,
    RunRecord,
    SessionRecord,
    StateRepository,
    ToolInvocationRecord,
    TurnRecord,
)
from tars_agent.core.persistence.models import SessionMode
from tars_agent.core.processes import finish_cleanup
from tars_agent.core.runner import AgentRunner, RunOutcome
from tars_agent.core.runs import new_run_id
from tars_agent.core.runtime.supervisor import RunSupervisor
from tars_agent.core.skills.loader import SkillLoader
from tars_agent.core.subagent.registry import BackgroundTaskRegistry
from tars_agent.core.tools.runtime import RuntimeCleanupPending, RuntimeRouter

log = logging.getLogger(__name__)

SESSION_NOT_FOUND = -32010
SESSION_CLOSED = -32011
SESSION_BUSY = -32012
RUN_NOT_FOUND = -32030
RUN_INVALID_STATE = -32031
RUN_SIDE_EFFECT_CONFIRMATION_REQUIRED = -32032
RUN_CANCEL_TIMEOUT = -32033
CANCEL_WAIT_SECONDS = 15.0
CLEANUP_RETRY_SECONDS = 0.5
SHUTDOWN_WAIT_SECONDS = 15.0
COMPACTION_FAILED = -32020

_TERMINAL_RUN_STATUSES = frozenset(
    {"succeeded", "failed", "cancelled", "interrupted"}
)
_SIDE_EFFECT_TOOLS = frozenset(
    {
        "bash",
        "write_file",
        "note_save",
        "task_create",
        "task_update",
        "spawn_agent",
        "agent_cancel",
    }
)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass(frozen=True, slots=True)
class SubmitRunResult:
    run_id: str
    status: str
    deduplicated: bool = False


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    id: str
    mode: str
    status: str
    title: str
    workspace_root: str | None
    active_run_id: str | None
    created_at: str
    updated_at: str
    closed_at: str | None


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    id: str
    session_id: str
    turn_id: str | None
    parent_run_id: str | None
    retry_of_run_id: str | None
    kind: str
    attempt: int
    status: str
    reason: str | None
    side_effects_started: bool
    result: dict[str, Any] | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class RuntimeService:
    """Own durable Session/Turn/Run state and transaction boundaries."""

    def __init__(
        self,
        database: Database,
        runner_factory: Callable[[], AgentRunner],
        bus: EventBus,
        *,
        artifacts_root: Path,
        supervisor: RunSupervisor | None = None,
        subagent_registry: BackgroundTaskRegistry | None = None,
        tool_runtime: RuntimeRouter | None = None,
        compaction_provider_factory: Callable[[], LLMProvider] | None = None,
        llm_config: LlmConfig | None = None,
    ) -> None:
        self._database = database
        self._runner_factory = runner_factory
        self._bus = bus
        self._supervisor = supervisor or RunSupervisor()
        self._subagent_registry = subagent_registry
        self._tool_runtime = tool_runtime
        self._pending_resources: set[str] = set()
        self._failed_cleanup_roots: set[str] = set()
        self._cleanup_stopping = asyncio.Event()
        self._artifact_store = ArtifactStore(artifacts_root / "sessions")
        self._event_writers: dict[str, EventWriter] = {}
        self._submission_locks: dict[str, asyncio.Lock] = {}
        self._submissions: set[asyncio.Task[SubmitRunResult]] = set()
        self._closed = False
        self._terminal_lock = asyncio.Lock()
        self._cancellations: dict[str, asyncio.Task[None]] = {}
        self._cancel_pending: dict[str, set[str]] = {}
        self._compaction_provider_factory = compaction_provider_factory
        self._compaction_provider: LLMProvider | None = None
        self._llm_config = llm_config or LlmConfig()
        self._bus.subscribe(self._observe_tool_event)
        if subagent_registry is not None:
            subagent_registry.set_cleanup_pending_handler(self._request_failed_cleanup)

    @property
    def supervisor(self) -> RunSupervisor:
        return self._supervisor

    async def recover_interrupted(self) -> int:
        async with self._database.session() as sql:
            runs = await StateRepository(sql).list_runs_with_statuses(("queued", "running"))
        # Recovery records interruption; it never replays a tool or a user submission.
        for run in runs:
            await self._finalize_without_outcome(run.id, "interrupted", "daemon_restarted")
        return len(runs)

    async def _open_run_writer(self, run_id: str, session_id: str) -> None:
        if run_id in self._event_writers:
            return
        writer = EventWriter(
            self._artifact_store.run_dir(session_id, run_id) / "events.jsonl", run_id=run_id,
        )
        try:
            await writer.__aenter__()
        except OSError:
            log.exception("run event file unavailable run_id=%s", run_id)
            return
        writer.subscribe(self._bus)
        self._event_writers[run_id] = writer

    async def _publish_run_finished(self, event: RunFinishedEvent) -> None:
        try:
            await self._bus.publish(event)
        finally:
            writer = self._event_writers.pop(event.run_id, None)
            if writer is not None:
                await writer.__aexit__()

    async def create_session(
        self,
        mode: SessionMode,
        *,
        title: str = "",
        workspace_root: Path | None = None,
    ) -> SessionSnapshot:
        now = _now()
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        try:
            root = (workspace_root or Path.cwd()).expanduser().resolve(strict=True)
        except OSError as exc:
            raise HandlerError(-32602, "workspace_root does not exist") from exc
        if not root.is_dir():
            raise HandlerError(-32602, "workspace_root must be an existing directory")
        record = SessionRecord(
            id=session_id,
            mode=mode,
            status="ready",
            title=title,
            workspace_root=str(root),
            created_at=now,
            updated_at=now,
        )
        async with self._database.transaction() as db_session:
            await StateRepository(db_session).add_session(record)
        await self._bus.publish(
            SessionCreatedEvent(session_id=session_id, mode=mode, ts=now.isoformat())
        )
        return self._session_snapshot(record)

    async def submit_message(
        self,
        session_id: str,
        raw_content: str,
        *,
        client_message_id: str | None = None,
        run_id: str | None = None,
    ) -> SubmitRunResult:
        if not raw_content.strip():
            raise HandlerError(-32602, "content must not be empty")
        if self._closed:
            raise HandlerError(RUN_INVALID_STATE, "runtime is shutting down")
        # A connection owns only the wait. Once accepted, the runtime owns the
        # commit-and-schedule operation even if the client loses its response.
        operation = asyncio.create_task(self._submit_message_locked(
            session_id, raw_content, client_message_id=client_message_id, run_id=run_id,
        ), name=f"submit:{session_id}")
        self._submissions.add(operation)
        operation.add_done_callback(self._submission_finished)
        return await asyncio.shield(operation)

    def _submission_finished(self, task: asyncio.Task[SubmitRunResult]) -> None:
        self._submissions.discard(task)
        if not task.cancelled():
            error = task.exception()
            if error is not None and not isinstance(error, HandlerError):
                log.error("durable submission failed", exc_info=error)

    async def _submit_message_locked(
        self, session_id: str, raw_content: str, *,
        client_message_id: str | None, run_id: str | None,
    ) -> SubmitRunResult:
        lock = self._submission_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            result = await self._create_message_run(
                session_id, raw_content, client_message_id=client_message_id, run_id=run_id,
            )
            if result.deduplicated:
                return result
            # No suspension between successful persistence and supervision.
            self._start_run(result.run_id)
            try:
                await self._bus.publish(SessionMessageReceivedEvent(
                    session_id=session_id, content=raw_content, ts=_now().isoformat(),
                ))
            except Exception:
                # Notification is observable but cannot undo committed state or
                # strand a queued Run that an idempotent retry will merely reuse.
                log.exception("message notification failed run_id=%s", result.run_id)
            return result

    async def _create_message_run(
        self,
        session_id: str,
        raw_content: str,
        *,
        client_message_id: str | None,
        run_id: str | None,
    ) -> SubmitRunResult:
        now = _now()
        selected_run_id = run_id or new_run_id()
        turn_id = f"turn-{uuid.uuid4().hex}"
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            session = await repository.get_session(session_id)
            if session is None:
                raise HandlerError(SESSION_NOT_FOUND, "session not found")
            if client_message_id is not None:
                existing_turn = await repository.get_turn_by_client_message(
                    session_id, client_message_id
                )
                if existing_turn is not None:
                    existing_run = await repository.get_run_for_turn(existing_turn.id)
                    if existing_run is None:
                        raise RuntimeError("idempotent turn has no run")
                    return SubmitRunResult(
                        run_id=existing_run.id,
                        status=existing_run.status,
                        deduplicated=True,
                    )
            if session.status == "closed":
                raise HandlerError(SESSION_CLOSED, "session already closed")
            if session.status == "running" or session.active_run_id is not None:
                raise HandlerError(SESSION_BUSY, "session busy")

            effective_content, execution_options = self._resolve_skill(
                raw_content, workspace_root=Path(session.workspace_root or Path.cwd()),
            )

            turn = TurnRecord(
                id=turn_id,
                session_id=session_id,
                client_message_id=client_message_id,
                raw_content=raw_content,
                effective_content=effective_content,
                status="queued",
                created_at=now,
                updated_at=now,
            )
            await repository.add_turn(turn)
            run = RunRecord(
                id=selected_run_id,
                session_id=session_id,
                turn_id=turn_id,
                kind="chat",
                attempt=1,
                status="queued",
                execution_options=execution_options,
                created_at=now,
                updated_at=now,
            )
            await repository.add_run(run)
            sequence = await repository.next_message_sequence(session_id)
            await repository.add_message(
                MessageRecord(
                    session_id=session_id,
                    turn_id=turn_id,
                    run_id=selected_run_id,
                    sequence=sequence,
                    role="user",
                    content=effective_content,
                    committed=False,
                    active=True,
                    created_at=now,
                )
            )
            session.status = "running"
            session.active_run_id = selected_run_id
            session.updated_at = now
            if not session.title:
                session.title = raw_content[:40]

        return SubmitRunResult(run_id=selected_run_id, status="queued")

    def _start_run(self, run_id: str) -> None:
        try:
            self._supervisor.start(run_id, lambda: self._execute_run(run_id))
        except Exception:
            asyncio.create_task(self._mark_launch_failed(run_id))
            raise

    async def _mark_launch_failed(self, run_id: str) -> None:
        await self._finalize_without_outcome(run_id, "interrupted", "supervisor_unavailable")

    async def _execute_run(self, run_id: str) -> None:
        try:
            run, session, turn, history, history_complete = await self._begin_run(run_id)
        except asyncio.CancelledError:
            await self._finalize_without_outcome(run_id, "cancelled", "cancelled")
            return
        except ContextBudgetError as exc:
            await self._finalize_without_outcome(run_id, "failed", str(exc))
            return
        except Exception:
            log.exception("run preparation failed run_id=%s", run_id)
            await self._finalize_without_outcome(run_id, "failed", "runtime_prepare_error")
            return
        try:
            await self._open_run_writer(run_id, session.id)
            options = run.execution_options or {}
            skill_name = options.get("skill_name")
            if isinstance(skill_name, str) and skill_name:
                await self._bus.publish(
                    SkillInvokedEvent(
                        skill_name=skill_name,
                        arguments=str(options.get("skill_arguments", "")),
                        run_id=run_id,
                        ts=_now().isoformat(),
                    )
                )
            runner = self._runner_factory()
            outcome = await runner.run_and_capture(
                turn.effective_content,
                run_id=run_id,
                session_id=session.id,
                workspace_root=Path(session.workspace_root or Path.cwd()),
                history=history,
                history_complete=history_complete,
                artifact_store=self._artifact_store,
                session_notes=self._artifact_store.read_notes(session.id),
                system_prompt_override=self._optional_string(options.get("system_prompt_override")),
                tool_whitelist=self._string_list(options.get("tool_whitelist")),
            )
        except asyncio.CancelledError:
            outcome = RunOutcome(
                status="cancelled",
                result="",
                reason="cancelled",
            )
        except RuntimeCleanupPending:
            await self._mark_cleanup_pending(run_id)
            self._request_failed_cleanup(run_id)
            return
        except Exception:
            log.exception("run preparation or execution failed run_id=%s", run_id)
            outcome = RunOutcome(
                status="failed",
                result="",
                reason="runtime_error",
            )
        await finish_cleanup(
            asyncio.create_task(self._finish_run(run_id, outcome)),
            failure_message="run finalization failed during cancellation",
        )

    async def _begin_run(
        self,
        run_id: str,
    ) -> tuple[RunRecord, SessionRecord, TurnRecord, list[dict[str, Any]], bool]:
        now = _now()
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            run = await repository.get_run(run_id)
            if run is None:
                raise HandlerError(RUN_NOT_FOUND, "run not found")
            if run.status != "queued":
                raise HandlerError(RUN_INVALID_STATE, f"run is {run.status}")
            session = await repository.get_session(run.session_id)
            turn = await repository.get_turn(run.turn_id or "")
            if session is None or turn is None:
                raise RuntimeError("run references missing session or turn")
            run.status = "running"
            run.started_at = now
            run.updated_at = now
            turn.status = "running"
            turn.updated_at = now
            committed, complete = await self._load_context_records(repository, session.id)
            history = [
                {"role": message.role, "content": message.content}
                for message in committed
            ]
            history.append({"role": "user", "content": turn.effective_content})
            return run, session, turn, history, complete

    async def _load_context_records(
        self, repository: StateRepository, session_id: str, *, require_complete: bool = False,
    ) -> tuple[list[MessageRecord], bool]:
        budget = ContextBudget.from_config(self._llm_config)
        bounds = await repository.context_message_bounds(session_id)
        if bounds is None:
            return [], True
        groups: list[list[MessageRecord]] = []
        pending: list[MessageRecord] = []
        used = 0
        pending_cost = 0
        complete = True
        records = repository.iter_context_messages(session_id, through_sequence=bounds[1])
        async for record in records:
            message = {"role": record.role, "content": record.content}
            cost = message_estimate(message)
            if used + pending_cost + cost > budget.input_limit:
                if require_complete or not groups:
                    raise budget.exceeded("loading required history", used + pending_cost + cost)
                complete = False
                pending = []  # Discard only this older, incomplete group from the projection.
                break
            pending.append(record)
            pending_cost += cost
            if is_user_input(message):
                group = list(reversed(pending))
                validate_tool_pairs([{"role": row.role, "content": row.content} for row in group])
                groups.append(group)
                used += pending_cost
                pending, pending_cost = [], 0
        if pending:
            group = list(reversed(pending))
            validate_tool_pairs([{"role": row.role, "content": row.content} for row in group])
            groups.append(group)
        return [row for group in reversed(groups) for row in group], complete

    async def _finish_run(self, run_id: str, outcome: RunOutcome) -> None:
        async with self._terminal_lock:
            await self._finish_run_locked(run_id, outcome)

    async def _finish_run_locked(self, run_id: str, outcome: RunOutcome) -> None:
        now = _now()
        db_status = {
            "success": "succeeded",
            "succeeded": "succeeded",
            "cancelled": "cancelled",
        }.get(outcome.status, "failed")
        session_event: BaseModel | None = None
        run_session_id: str | None = None
        committed_compaction = False
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            run = await repository.get_run(run_id)
            if run is None:
                return
            run_session_id = run.session_id
            if run.status in _TERMINAL_RUN_STATUSES:
                return
            sequence = await repository.next_message_sequence(run.session_id)
            for message in outcome.messages:
                role = message.get("role")
                if role not in {"user", "assistant"}:
                    continue
                await repository.add_message(
                    MessageRecord(
                        session_id=run.session_id,
                        turn_id=run.turn_id,
                        run_id=run_id,
                        sequence=sequence,
                        role=role,
                        content=message.get("content", ""),
                        committed=False,
                        active=True,
                        created_at=now,
                    )
                )
                sequence += 1
            if db_status == "succeeded":
                await repository.set_run_messages_committed(run_id, True)
                if outcome.active_context is not None and outcome.compactions:
                    await self._commit_compacted_context(
                        repository,
                        run,
                        outcome,
                        next_sequence=sequence,
                        now=now,
                    )
                    committed_compaction = True
            run.status = db_status
            run.reason = outcome.reason
            run.result = {"text": outcome.result, "steps": outcome.steps}
            run.finished_at = now
            run.updated_at = now
            await repository.close_unfinished_tool_invocations(
                run_id,
                terminal_status="cancelled" if db_status == "cancelled" else "interrupted",
                reason=outcome.reason or f"run_{db_status}",
                finished_at=now,
            )
            turn = await repository.get_turn(run.turn_id or "")
            if turn is not None:
                turn.status = db_status
                turn.updated_at = now
            session = await repository.get_session(run.session_id)
            if session is not None:
                session.active_run_id = None
                session.updated_at = now
                if session.status == "closed" or session.mode == "one_shot":
                    session.status = "closed"
                    session.closed_at = session.closed_at or now
                    session_event = SessionClosedEvent(
                        session_id=session.id,
                        ts=now.isoformat(),
                    )
                else:
                    session.status = "ready"
                    session_event = SessionWaitingForInputEvent(
                        session_id=session.id,
                        last_run_id=run_id,
                        ts=now.isoformat(),
                    )
        if run_session_id is None:
            return
        if committed_compaction:
            latest_compaction = outcome.compactions[-1]
            compactor = Compactor(
                self._bus,
                self._artifact_store.session_dir(run_session_id),
                run_session_id,
            )
            try:
                compactor.write_summary(latest_compaction.summary_text)
            except OSError:
                log.exception(
                    "failed to update compaction artifact for session %s",
                    run_session_id,
                )
            await self._bus.publish(
                ContextCompactedEvent(
                    session_id=run_session_id,
                    run_id=run_id,
                    original_tokens=latest_compaction.original_token_estimate,
                    summary_tokens=latest_compaction.summary_tokens,
                    ts=now.isoformat(),
                )
            )
        await self._publish_run_finished(
            RunFinishedEvent(
                run_id=run_id,
                status="success" if db_status == "succeeded" else db_status,
                reason=outcome.reason,
                steps=outcome.steps,
                ts=now.isoformat(),
            )
        )
        if session_event is not None:
            await self._bus.publish(session_event)

    async def _commit_compacted_context(
        self,
        repository: StateRepository,
        run: RunRecord,
        outcome: RunOutcome,
        *,
        next_sequence: int,
        now: datetime,
    ) -> None:
        bounds = await repository.context_message_bounds(run.session_id)
        if bounds is None:
            return
        start_sequence, end_sequence = bounds
        await repository.deactivate_messages_through(run.session_id, end_sequence)
        summary_ids: list[int] = []
        for message in outcome.active_context or []:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            record = await repository.add_message(
                MessageRecord(
                    session_id=run.session_id,
                    turn_id=run.turn_id,
                    run_id=run.id,
                    sequence=next_sequence,
                    role=role,
                    content=message.get("content", ""),
                    committed=True,
                    active=True,
                    created_at=now,
                )
            )
            summary_ids.append(record.id)
            next_sequence += 1
        latest = outcome.compactions[-1]
        context_version = await repository.latest_compaction_version(run.session_id) + 1
        await repository.add_compaction(
            CompactionRecord(
                session_id=run.session_id,
                run_id=run.id,
                summary_message_id=summary_ids[0] if summary_ids else None,
                start_sequence=start_sequence,
                end_sequence=end_sequence,
                summary=latest.summary_text,
                original_tokens=latest.original_token_estimate,
                summary_tokens=latest.summary_tokens,
                context_version=context_version,
                created_at=now,
            )
        )

    async def retry_run(
        self,
        run_id: str,
        *,
        confirm_side_effects: bool = False,
    ) -> SubmitRunResult:
        original = await self.get_run(run_id)
        lock = self._submission_locks.setdefault(original.session_id, asyncio.Lock())
        async with lock:
            return await self._create_retry_run(
                run_id,
                confirm_side_effects=confirm_side_effects,
            )

    async def _create_retry_run(
        self,
        run_id: str,
        *,
        confirm_side_effects: bool,
    ) -> SubmitRunResult:
        now = _now()
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            original = await repository.get_run(run_id)
            if original is None:
                raise HandlerError(RUN_NOT_FOUND, "run not found")
            if original.status not in {"failed", "interrupted"}:
                raise HandlerError(
                    RUN_INVALID_STATE,
                    "only failed or interrupted runs can be retried",
                )
            if original.side_effects_started and not confirm_side_effects:
                raise HandlerError(
                    RUN_SIDE_EFFECT_CONFIRMATION_REQUIRED,
                    "retry requires confirm_side_effects=true",
                )
            session = await repository.get_session(original.session_id)
            turn = await repository.get_turn(original.turn_id or "")
            if session is None or turn is None:
                raise RuntimeError("run references missing session or turn")
            if session.status == "closed":
                raise HandlerError(SESSION_CLOSED, "session already closed")
            if session.status == "running" or session.active_run_id is not None:
                raise HandlerError(SESSION_BUSY, "session busy")
            latest = await repository.get_run_for_turn(turn.id)
            attempt = (latest.attempt if latest is not None else original.attempt) + 1
            retry_id = new_run_id()
            retry = RunRecord(
                id=retry_id,
                session_id=original.session_id,
                turn_id=turn.id,
                parent_run_id=original.parent_run_id,
                retry_of_run_id=original.id,
                kind=original.kind,
                attempt=attempt,
                status="queued",
                execution_options=dict(original.execution_options or {}),
                created_at=now,
                updated_at=now,
            )
            await repository.add_run(retry)
            sequence = await repository.next_message_sequence(original.session_id)
            await repository.add_message(
                MessageRecord(
                    session_id=original.session_id,
                    turn_id=turn.id,
                    run_id=retry_id,
                    sequence=sequence,
                    role="user",
                    content=turn.effective_content,
                    committed=False,
                    active=True,
                    created_at=now,
                )
            )
            turn.status = "queued"
            turn.updated_at = now
            session.status = "running"
            session.active_run_id = retry_id
            session.updated_at = now
        self._start_run(retry_id)
        return SubmitRunResult(run_id=retry_id, status="queued")

    def _request_failed_cleanup(self, run_id: str) -> None:
        self._pending_resources.add(run_id)
        if self._closed or any(run_id in ids for ids in self._cancel_pending.values()):
            return
        self._failed_cleanup_roots.add(run_id)
        self._start_cleanup_task(run_id)

    def _start_cleanup_task(self, run_id: str) -> asyncio.Task[None]:
        cleanup = self._cancellations.get(run_id)
        if cleanup is None:
            self._cancel_pending[run_id] = {run_id}
            cleanup = asyncio.create_task(self._cancel_tree(run_id), name=f"cancel:{run_id}")
            self._cancellations[run_id] = cleanup
            cleanup.add_done_callback(lambda task: self._cancel_done(run_id, task))
        return cleanup

    async def _mark_cleanup_pending(self, run_id: str) -> None:
        self._pending_resources.add(run_id)
        async with self._database.transaction() as sql:
            await sql.execute(update(RunRecord).where(
                RunRecord.id == run_id, RunRecord.status.in_(("queued", "running")),
            ).values(reason="cleanup_pending", updated_at=_now()))

    async def _confirm_run_cleanup(self, run_id: str) -> bool:
        if self._tool_runtime is None:
            return run_id not in self._pending_resources
        while not self._cleanup_stopping.is_set():
            try:
                await self._tool_runtime.cleanup_run(run_id)
                if run_id in self._tool_runtime.pending_cleanup_run_ids():
                    raise RuntimeCleanupPending(run_id)
            except RuntimeCleanupPending:
                await self._mark_cleanup_pending(run_id)
                try:
                    await asyncio.wait_for(self._cleanup_stopping.wait(), CLEANUP_RETRY_SECONDS)
                except TimeoutError:
                    pass
            else:
                self._pending_resources.discard(run_id)
                if self._subagent_registry is not None:
                    self._subagent_registry.complete_cleanup(run_id)
                return True
        return False

    async def cancel_run(self, run_id: str) -> RunSnapshot:
        # Own cleanup outside the RPC task. Disconnects and repeated requests
        # only end/join a waiter; they never inject another cancellation.
        cleanup = self._start_cleanup_task(run_id)
        try:
            async with asyncio.timeout(CANCEL_WAIT_SECONDS):
                await asyncio.shield(cleanup)
                return await self.get_run(run_id)
        except TimeoutError as exc:
            raise HandlerError(
                RUN_CANCEL_TIMEOUT,
                "cancellation requested; cleanup is still running",
                data={
                    "run_id": run_id,
                    "cancellation_requested": True,
                    "pending_run_ids": sorted(self._cancel_pending.get(run_id, {run_id})),
                },
            ) from exc

    def _cancel_done(self, run_id: str, task: asyncio.Task[None]) -> None:
        self._cancellations.pop(run_id, None)
        self._cancel_pending.pop(run_id, None)
        self._failed_cleanup_roots.discard(run_id)
        if not task.cancelled() and task.exception() is not None:
            log.error("run cleanup failed run_id=%s", run_id, exc_info=task.exception())

    async def _run_tree(self, run_id: str) -> list[RunRecord]:
        # No repository pagination: descendants can outlive a finished parent.
        async with self._database.session() as session:
            records = list((await session.scalars(select(RunRecord))).all())
        selected = {run_id}
        while True:
            found = {record.id for record in records if record.parent_run_id in selected}
            if found <= selected:
                break
            selected.update(found)
        return [record for record in records if record.id in selected]

    async def _cancel_tree(self, run_id: str) -> None:
        await self.get_run(run_id)
        registry = self._subagent_registry
        if registry is not None:
            registry.block_descendants({run_id})
        records = await self._run_tree(run_id)
        ids = {record.id for record in records}
        if registry is not None:
            registry.block_descendants(ids)
        self._cancel_pending[run_id] = set(ids)
        # Send all cancellation signals concurrently so one slow child does not
        # prevent its siblings or the parent from beginning their own cleanup.
        async def stop(record: RunRecord) -> None:
            if record.kind == "subagent" and registry is not None:
                try:
                    await registry.cancel(record.id, session_id=record.session_id)
                except RuntimeCleanupPending:
                    await self._mark_cleanup_pending(record.id)
            elif self._supervisor.cancel(record.id):
                try:
                    await self._supervisor.wait(record.id)
                except asyncio.CancelledError:
                    pass
            if not await self._confirm_run_cleanup(record.id):
                return
            failed = record.id == run_id and run_id in self._failed_cleanup_roots
            await self._finalize_without_outcome(
                record.id, "failed" if failed else "cancelled",
                "cleanup_failed" if failed else "cancelled",
            )
            self._cancel_pending[run_id].discard(record.id)

        results = await asyncio.gather(
            *(stop(record) for record in records), return_exceptions=True,
        )
        failures = [str(result) for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError("run tree cleanup failed: " + "; ".join(failures))

    async def _finalize_without_outcome(
        self,
        run_id: str,
        status: str,
        reason: str,
    ) -> None:
        async with self._terminal_lock:
            await self._finalize_without_outcome_locked(run_id, status, reason)

    async def _finalize_without_outcome_locked(
        self, run_id: str, status: str, reason: str,
    ) -> None:
        now = _now()
        session_event: BaseModel | None = None
        async with self._database.session() as sql:
            existing = await StateRepository(sql).get_run(run_id)
            if existing is None:
                raise HandlerError(RUN_NOT_FOUND, "run not found")
        if existing.status in _TERMINAL_RUN_STATUSES:
            return
        if existing.kind == "subagent":
            if self._subagent_registry is None:
                raise RuntimeError("subagent owner is not configured")
            await self._subagent_registry.start_recording(
                run_id, self._artifact_store.run_dir(existing.session_id, run_id) / "events.jsonl",
            )
            await self._subagent_registry.finish(run_id, status=status, reason=reason)
            return
        await self._open_run_writer(run_id, existing.session_id)
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            run = await repository.get_run(run_id)
            if run is None:
                raise HandlerError(RUN_NOT_FOUND, "run not found")
            if run.status in _TERMINAL_RUN_STATUSES:
                return
            run.status = status
            run.reason = reason
            run.finished_at = now
            run.updated_at = now
            await repository.close_unfinished_tool_invocations(
                run_id,
                terminal_status="cancelled" if status == "cancelled" else "interrupted",
                reason=reason,
                finished_at=now,
            )
            turn = await repository.get_turn(run.turn_id or "")
            if turn is not None:
                turn.status = status
                turn.updated_at = now
            session = await repository.get_session(run.session_id)
            if (
                session is not None
                and session.status != "closed"
            ):
                session.active_run_id = None
                session.updated_at = now
                if session.mode == "one_shot":
                    session.status = "closed"
                    session.closed_at = session.closed_at or now
                    session_event = SessionClosedEvent(session_id=session.id, ts=now.isoformat())
                else:
                    session.status = "ready"
                    session_event = SessionWaitingForInputEvent(
                        session_id=session.id, last_run_id=run_id, ts=now.isoformat(),
                    )
        await self._publish_run_finished(RunFinishedEvent(
            run_id=run_id, status="success" if status == "succeeded" else status,
            reason=reason, steps=0, ts=now.isoformat(),
        ))
        if session_event is not None:
            await self._bus.publish(session_event)

    async def get_run(self, run_id: str) -> RunSnapshot:
        async with self._database.session() as db_session:
            run = await StateRepository(db_session).get_run(run_id)
            if run is None:
                raise HandlerError(RUN_NOT_FOUND, "run not found")
            return self._run_snapshot(run)

    async def get_session(self, session_id: str) -> SessionSnapshot:
        async with self._database.session() as db_session:
            session = await StateRepository(db_session).get_session(session_id)
            if session is None:
                raise HandlerError(SESSION_NOT_FOUND, "session not found")
            return self._session_snapshot(session)

    async def get_session_with_latest_run(
        self,
        session_id: str,
    ) -> tuple[SessionSnapshot, RunSnapshot | None]:
        async with self._database.session() as db_session:
            repository = StateRepository(db_session)
            session = await repository.get_session(session_id)
            if session is None:
                raise HandlerError(SESSION_NOT_FOUND, "session not found")
            run = await repository.latest_run(session_id)
            return (
                self._session_snapshot(session),
                self._run_snapshot(run) if run is not None else None,
            )

    async def list_sessions(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionSnapshot]:
        async with self._database.session() as db_session:
            records = await StateRepository(db_session).list_sessions(
                status=status,
                limit=limit,
                offset=offset,
            )
            return [self._session_snapshot(record) for record in records]

    async def resume_session(self, session_id: str) -> SessionSnapshot:
        session = await self.get_session(session_id)
        if session.status == "closed":
            raise HandlerError(SESSION_CLOSED, "session already closed")
        await self._bus.publish(
            SessionResumedEvent(session_id=session_id, ts=_now().isoformat())
        )
        return session

    async def latest_event_cursor(self, session_id: str) -> int:
        async with self._database.session() as db_session:
            repository = StateRepository(db_session)
            if await repository.get_session(session_id) is None:
                raise HandlerError(SESSION_NOT_FOUND, "session not found")
            return await repository.latest_event_cursor(session_id=session_id)

    async def get_history(
        self,
        session_id: str,
        *,
        after_sequence: int | None = None,
        limit: int = 2_000,
    ) -> list[dict[str, Any]]:
        async with self._database.session() as db_session:
            repository = StateRepository(db_session)
            if await repository.get_session(session_id) is None:
                raise HandlerError(SESSION_NOT_FOUND, "session not found")
            messages = await repository.list_messages(
                session_id,
                after_sequence=after_sequence,
                limit=limit,
            )
            return [
                {
                    "sequence": message.sequence,
                    "role": message.role,
                    "content": message.content,
                    "run_id": message.run_id,
                }
                for message in messages
            ]

    async def compact_session(
        self,
        session_id: str,
        *,
        focus: str = "",
    ) -> CompactionResult:
        lock = self._submission_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session, messages, start_sequence, end_sequence = (
                await self._load_manual_compaction_input(session_id)
            )
            provider = self._get_compaction_provider()
            compactor = Compactor(
                self._bus,
                self._artifact_store.session_dir(session_id),
                session_id,
                context_budget=ContextBudget.from_config(self._llm_config),
            )
            try:
                result = await compactor.compact_messages(messages, provider, focus=focus)
            except ContextBudgetError as exc:
                raise HandlerError(COMPACTION_FAILED, str(exc)) from exc
            if result is None:
                raise HandlerError(COMPACTION_FAILED, "context compaction failed")
            context_version = await self._commit_manual_compaction(
                session_id,
                result,
                start_sequence=start_sequence,
                end_sequence=end_sequence,
            )
            compactor.write_summary(result.summary_text)
            await self._bus.publish(
                ContextCompactedEvent(
                    session_id=session_id,
                    run_id=None,
                    original_tokens=result.original_token_estimate,
                    summary_tokens=result.summary_tokens,
                    ts=_now().isoformat(),
                )
            )
            log.info(
                "manual context compaction committed session=%s version=%d",
                session_id,
                context_version,
            )
            return result

    async def _load_manual_compaction_input(
        self,
        session_id: str,
    ) -> tuple[SessionRecord, list[dict[str, Any]], int, int]:
        async with self._database.session() as db_session:
            repository = StateRepository(db_session)
            session = await repository.get_session(session_id)
            if session is None:
                raise HandlerError(SESSION_NOT_FOUND, "session not found")
            if session.status == "closed":
                raise HandlerError(SESSION_CLOSED, "session already closed")
            if session.status == "running" or session.active_run_id is not None:
                raise HandlerError(SESSION_BUSY, "session busy")
            try:
                records, _complete = await self._load_context_records(
                    repository, session_id, require_complete=True,
                )
            except ContextBudgetError as exc:
                raise HandlerError(COMPACTION_FAILED, str(exc)) from exc
            if not records:
                raise HandlerError(COMPACTION_FAILED, "session has no active context")
            messages = [
                {"role": record.role, "content": record.content}
                for record in records
            ]
            return session, messages, records[0].sequence, records[-1].sequence

    def _get_compaction_provider(self) -> LLMProvider:
        if self._compaction_provider is not None:
            return self._compaction_provider
        if self._compaction_provider_factory is None:
            raise HandlerError(COMPACTION_FAILED, "compaction provider is unavailable")
        try:
            provider = self._compaction_provider_factory()
        except SystemExit as exc:
            raise HandlerError(COMPACTION_FAILED, str(exc)) from exc
        self._compaction_provider = provider
        return provider

    async def _commit_manual_compaction(
        self,
        session_id: str,
        result: CompactionResult,
        *,
        start_sequence: int,
        end_sequence: int,
    ) -> int:
        now = _now()
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            session = await repository.get_session(session_id)
            if session is None:
                raise HandlerError(SESSION_NOT_FOUND, "session not found")
            if session.status != "ready" or session.active_run_id is not None:
                raise HandlerError(SESSION_BUSY, "session changed during compaction")
            bounds = await repository.context_message_bounds(session_id)
            if bounds is None or bounds != (start_sequence, end_sequence):
                raise HandlerError(COMPACTION_FAILED, "active context changed during compaction")
            await repository.deactivate_messages_through(session_id, end_sequence)
            next_sequence = await repository.next_message_sequence(session_id)
            summary = await repository.add_message(
                MessageRecord(
                    session_id=session_id,
                    sequence=next_sequence,
                    role="user",
                    content=result.summary_text,
                    committed=True,
                    active=True,
                    created_at=now,
                )
            )
            await repository.add_message(
                MessageRecord(
                    session_id=session_id,
                    sequence=next_sequence + 1,
                    role="assistant",
                    content="Understood, I'll continue from this summary.",
                    committed=True,
                    active=True,
                    created_at=now,
                )
            )
            context_version = await repository.latest_compaction_version(session_id) + 1
            await repository.add_compaction(
                CompactionRecord(
                    session_id=session_id,
                    summary_message_id=summary.id,
                    start_sequence=start_sequence,
                    end_sequence=end_sequence,
                    summary=result.summary_text,
                    original_tokens=result.original_token_estimate,
                    summary_tokens=result.summary_tokens,
                    context_version=context_version,
                    created_at=now,
                )
            )
            session.updated_at = now
            return context_version

    async def close_session(self, session_id: str) -> SessionSnapshot:
        lock = self._submission_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            return await self._close_session_locked(session_id)

    async def _close_session_locked(self, session_id: str) -> SessionSnapshot:
        session = await self.get_session(session_id)
        if session.active_run_id is not None:
            await self.cancel_run(session.active_run_id)
        if self._subagent_registry is not None:
            await self._subagent_registry.cancel_session(session_id)
        now = _now()
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            record = await repository.get_session(session_id)
            if record is None:
                raise HandlerError(SESSION_NOT_FOUND, "session not found")
            record.status = "closed"
            record.active_run_id = None
            record.updated_at = now
            record.closed_at = now
            snapshot = self._session_snapshot(record)
        await self._bus.publish(
            SessionClosedEvent(session_id=session_id, ts=now.isoformat())
        )
        return snapshot

    async def shutdown(self) -> None:
        self._closed = True
        self._cleanup_stopping.set()
        shutdown = asyncio.create_task(self._shutdown_owned_tasks())
        try:
            done, _ = await asyncio.wait({shutdown}, timeout=SHUTDOWN_WAIT_SECONDS)
            if shutdown not in done:
                shutdown.cancel()
                shutdown.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None
                )
                raise RuntimeError("runtime shutdown timed out; resource cleanup is not confirmed")
            await shutdown
        finally:
            # An incomplete cleanup keeps its DB status; closing a file is not completion.
            for writer in tuple(self._event_writers.values()):
                await writer.__aexit__()
            self._event_writers.clear()

    async def _shutdown_owned_tasks(self) -> None:
        if self._submissions:
            await asyncio.gather(*tuple(self._submissions), return_exceptions=True)
        if self._cancellations:
            await asyncio.gather(*tuple(self._cancellations.values()), return_exceptions=True)
        active = self._supervisor.active_run_ids()
        await self._supervisor.shutdown()
        for run_id in active:
            if run_id not in self._pending_resources:
                await self._finalize_without_outcome(run_id, "cancelled", "core_shutdown")
        if self._subagent_registry is not None:
            await self._subagent_registry.shutdown()
        provider = self._compaction_provider
        if provider is not None:
            close = getattr(provider, "close", None)
            if close is not None:
                await close()
            self._compaction_provider = None

    async def _observe_tool_event(self, event: BaseModel) -> None:
        if isinstance(event, ToolCallStartedEvent):
            await self._record_tool_requested(event)
        elif isinstance(event, ToolExecutionStartedEvent):
            await self._record_tool_started(event)
        elif isinstance(event, ToolCallFinishedEvent):
            await self._record_tool_finished(event, succeeded=True)
        elif isinstance(event, ToolCallFailedEvent):
            await self._record_tool_finished(event, succeeded=False)

    async def _record_tool_requested(self, event: ToolCallStartedEvent) -> None:
        invocation_id = self._invocation_id(event.run_id, event.tool_use_id)
        parameters = dict(event.params)
        digest = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            if await repository.get_run(event.run_id) is None:
                return
            if await repository.get_tool_invocation(invocation_id) is not None:
                return
            await repository.add_tool_invocation(
                ToolInvocationRecord(
                    id=invocation_id,
                    run_id=event.run_id,
                    tool_name=event.tool_name,
                    parameters=parameters,
                    parameter_digest=digest,
                    backend=self._tool_backend(event.tool_name),
                    status="queued",
                    may_have_side_effects=self._may_have_side_effects(event.tool_name),
                    retryable=False,
                )
            )

    async def _record_tool_started(self, event: ToolExecutionStartedEvent) -> None:
        invocation_id = self._invocation_id(event.run_id, event.tool_use_id)
        now = _now()
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            invocation = await repository.get_tool_invocation(invocation_id)
            if invocation is None:
                return
            invocation.status = "running"
            invocation.backend = event.backend
            invocation.started_at = now
            if invocation.may_have_side_effects:
                run = await repository.get_run(event.run_id)
                if run is not None:
                    run.side_effects_started = True
                    run.updated_at = now

    async def _record_tool_finished(
        self,
        event: ToolCallFinishedEvent | ToolCallFailedEvent,
        *,
        succeeded: bool,
    ) -> None:
        invocation_id = self._invocation_id(event.run_id, event.tool_use_id)
        now = _now()
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            invocation = await repository.get_tool_invocation(invocation_id)
            if invocation is None:
                return
            invocation.status = "succeeded" if succeeded else "failed"
            invocation.retryable = event.retryable
            invocation.finished_at = now
            if isinstance(event, ToolCallFinishedEvent):
                invocation.result = {"content": event.output}
            else:
                invocation.error_class = event.error_class
                invocation.error_message = event.error_message

    def _resolve_skill(
        self, content: str, *, workspace_root: Path,
    ) -> tuple[str, dict[str, Any]]:
        if not content.startswith("/"):
            return content, {}
        parts = content[1:].split(None, 1)
        if not parts:
            raise HandlerError(-32602, "skill name is required; use /<skill> [arguments]")
        skill_name = parts[0]
        arguments = parts[1] if len(parts) > 1 else ""
        loader = SkillLoader(workspace_root=workspace_root)
        skill = loader.resolve(skill_name)
        if skill is None:
            return content, {}
        effective = loader.render_prompt(skill, arguments)
        return effective, {
            "skill_name": skill_name,
            "skill_arguments": arguments,
            "tool_whitelist": list(skill.allowed_tools),
        }

    @staticmethod
    def _invocation_id(run_id: str, tool_use_id: str) -> str:
        return f"{run_id}:{tool_use_id}"

    @staticmethod
    def _may_have_side_effects(tool_name: str) -> bool:
        return tool_name in _SIDE_EFFECT_TOOLS or "__" in tool_name

    @staticmethod
    def _tool_backend(tool_name: str) -> str:
        return "external" if "__" in tool_name else "host"

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _string_list(value: Any) -> list[str] | None:
        if not isinstance(value, list):
            return None
        return [item for item in value if isinstance(item, str)]

    @staticmethod
    def _session_snapshot(record: SessionRecord) -> SessionSnapshot:
        return SessionSnapshot(
            id=record.id,
            mode=record.mode,
            status=record.status,
            title=record.title,
            workspace_root=record.workspace_root,
            active_run_id=record.active_run_id,
            created_at=record.created_at.isoformat(),
            updated_at=record.updated_at.isoformat(),
            closed_at=_iso(record.closed_at),
        )

    @staticmethod
    def _run_snapshot(record: RunRecord) -> RunSnapshot:
        return RunSnapshot(
            id=record.id,
            session_id=record.session_id,
            turn_id=record.turn_id,
            parent_run_id=record.parent_run_id,
            retry_of_run_id=record.retry_of_run_id,
            kind=record.kind,
            attempt=record.attempt,
            status=record.status,
            reason=record.reason,
            side_effects_started=record.side_effects_started,
            result=record.result,
            created_at=record.created_at.isoformat(),
            started_at=_iso(record.started_at),
            finished_at=_iso(record.finished_at),
        )


__all__ = [
    "COMPACTION_FAILED",
    "RUN_INVALID_STATE",
    "RUN_NOT_FOUND",
    "RUN_SIDE_EFFECT_CONFIRMATION_REQUIRED",
    "RunSnapshot",
    "RuntimeService",
    "SESSION_BUSY",
    "SESSION_CLOSED",
    "SESSION_NOT_FOUND",
    "SessionSnapshot",
    "SubmitRunResult",
]
