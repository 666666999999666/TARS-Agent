from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import update

from tars_agent.core.bus.events import SubagentFinishedEvent
from tars_agent.core.context import ExecutionContext
from tars_agent.core.events.bus import EventBus
from tars_agent.core.events.writer import EventWriter
from tars_agent.core.persistence import Database, RunRecord, StateRepository
from tars_agent.core.processes import finish_cleanup
from tars_agent.core.tools.runtime.models import RuntimeCleanupPending

log = logging.getLogger(__name__)

_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class SubagentRunSnapshot:
    run_id: str
    session_id: str
    parent_run_id: str
    status: str
    result: str
    reason: str | None


@dataclass(slots=True)
class ActiveSubagent:
    task: asyncio.Task[None]
    context: ExecutionContext
    session_id: str
    parent_run_id: str


class BackgroundTaskRegistry:
    """Daemon-level owner for active subagent tasks; durable results live in SQLite."""

    def __init__(self, database: Database, bus: EventBus) -> None:
        self._database = database
        self._bus = bus
        self._tasks: dict[str, ActiveSubagent] = {}
        self._writers: dict[str, EventWriter] = {}
        self._terminal_lock = asyncio.Lock()
        self._closed = False
        self._finalizers: set[asyncio.Task[None]] = set()
        self._parents: dict[str, str] = {}
        self._blocked: set[str] = set()
        self._cleanup_pending: set[str] = set()
        self._cleanup_pending_handler: Callable[[str], None] | None = None
        self._cancellations: dict[str, asyncio.Task[bool]] = {}

    async def start_recording(self, run_id: str, path: Path) -> None:
        """Keep this Run's event file open until its committed terminal notification."""
        if run_id in self._writers:
            return
        writer = EventWriter(path, run_id=run_id)
        try:
            await writer.__aenter__()
        except OSError:
            log.exception("subagent event file could not be opened run_id=%s", run_id)
            return
        writer.subscribe(self._bus)
        self._writers[run_id] = writer

    async def _close_writer(self, run_id: str) -> None:
        writer = self._writers.pop(run_id, None)
        if writer is not None:
            await writer.__aexit__()

    def set_cleanup_pending_handler(self, callback: Callable[[str], None]) -> None:
        self._cleanup_pending_handler = callback

    def cleanup_is_pending(self, run_id: str) -> bool:
        return run_id in self._cleanup_pending

    def mark_cleanup_pending(self, run_id: str) -> None:
        self._cleanup_pending.add(run_id)
        if self._cleanup_pending_handler is not None:
            self._cleanup_pending_handler(run_id)

    def complete_cleanup(self, run_id: str) -> None:
        self._cleanup_pending.discard(run_id)

    async def release_when_idle(self, run_id: str, cleanup: Callable[[], Awaitable[None]]) -> None:
        async def finish() -> None:
            while True:
                descendants = self.descendant_ids({run_id})
                tasks = [entry.task for child_id, entry in self._tasks.items()
                         if child_id in descendants and not entry.task.done()]
                if not tasks:
                    break
                await asyncio.gather(*tasks, return_exceptions=True)
            await cleanup()

        if not any(child_id in self.descendant_ids({run_id}) and not entry.task.done()
                   for child_id, entry in self._tasks.items()):
            await cleanup()
            return
        operation = asyncio.create_task(finish())
        self._finalizers.add(operation)
        operation.add_done_callback(self._finalizer_finished)

    def _finalizer_finished(self, operation: asyncio.Task[None]) -> None:
        self._finalizers.discard(operation)
        if not operation.cancelled() and operation.exception() is not None:
            log.error("subagent resource finalizer failed", exc_info=operation.exception())

    def descendant_ids(self, run_ids: set[str]) -> set[str]:
        result = set(run_ids)
        while True:
            added = {run_id for run_id, parent in self._parents.items() if parent in result}
            if added <= result:
                return result
            result.update(added)

    def block_descendants(self, run_ids: set[str]) -> set[str]:
        """Synchronously close the spawn gate before cancellation performs database I/O."""
        blocked = self.descendant_ids(run_ids)
        self._blocked.update(blocked)
        return blocked

    def assert_can_spawn(self, parent_run_id: str) -> None:
        if self._closed:
            raise RuntimeError("subagent registry is closed")
        if parent_run_id in self._blocked:
            raise RuntimeError("subagent parent cancellation has been requested")

    @property
    def active_count(self) -> int:
        return sum(not entry.task.done() for entry in self._tasks.values())

    async def create_run(
        self,
        *,
        run_id: str,
        session_id: str,
        parent_run_id: str,
        description: str,
        prompt: str,
        depth: int,
        background: bool,
    ) -> None:
        self.assert_can_spawn(parent_run_id)
        self._parents[run_id] = parent_run_id
        now = _now()
        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            parent = await repository.get_run(parent_run_id)
            if parent is None or parent.session_id != session_id:
                raise RuntimeError("parent run does not belong to the subagent session")
            self.assert_can_spawn(parent_run_id)
            if parent.status in {"cancelled", "interrupted"}:
                raise RuntimeError("subagent parent is cancelled or interrupted")
            await repository.add_run(
                RunRecord(
                    id=run_id,
                    session_id=session_id,
                    parent_run_id=parent_run_id,
                    kind="subagent",
                    attempt=1,
                    status="queued",
                    execution_options={
                        "description": description,
                        "prompt": prompt,
                        "depth": depth,
                        "background": background,
                    },
                    created_at=now,
                    updated_at=now,
                )
            )

    async def mark_running(self, run_id: str) -> None:
        await _complete_before_cancelling(self._mark_running(run_id))

    async def _mark_running(self, run_id: str) -> None:
        now = _now()
        async with self._database.transaction() as db_session:
            await db_session.execute(update(RunRecord).where(
                RunRecord.id == run_id,
                RunRecord.kind == "subagent",
                RunRecord.status == "queued",
            ).values(status="running", started_at=now, updated_at=now))

    async def finish(
        self,
        run_id: str,
        *,
        status: str,
        result: str = "",
        reason: str | None = None,
    ) -> tuple[SubagentRunSnapshot, bool]:
        return await _complete_before_cancelling(
            self._finish(
                run_id,
                status=status,
                result=result,
                reason=reason,
            )
        )

    async def _finish(
        self,
        run_id: str,
        *,
        status: str,
        result: str,
        reason: str | None,
    ) -> tuple[SubagentRunSnapshot, bool]:
        normalized = {
            "success": "succeeded",
            "succeeded": "succeeded",
            "cancelled": "cancelled",
            "interrupted": "interrupted",
        }.get(status, "failed")
        now = _now()
        async with self._terminal_lock:
            async with self._database.transaction() as db_session:
                changed_id = await db_session.scalar(update(RunRecord).where(
                    RunRecord.id == run_id,
                    RunRecord.kind == "subagent",
                    RunRecord.status.in_(("queued", "running")),
                ).values(
                    status=normalized, reason=reason, result={"text": result},
                    finished_at=now, updated_at=now,
                ).returning(RunRecord.id))
                repository = StateRepository(db_session)
                if changed_id is not None:
                    await repository.close_unfinished_tool_invocations(
                        run_id,
                        terminal_status="cancelled" if normalized == "cancelled" else "interrupted",
                        reason=reason or f"run_{normalized}",
                        finished_at=now,
                    )
                record = await repository.get_run(run_id)
                if record is None or record.kind != "subagent":
                    raise RuntimeError(f"subagent run not found: {run_id}")
                snapshot = self._snapshot(record)
            changed = changed_id is not None
            try:
                if changed:
                    await self._bus.publish(SubagentFinishedEvent(
                        run_id=snapshot.run_id,
                        parent_run_id=snapshot.parent_run_id,
                        status="success" if snapshot.status == "succeeded" else snapshot.status,
                        ts=now.isoformat(),
                    ))
            finally:
                await self._close_writer(run_id)
            return snapshot, changed

    def register(
        self,
        run_id: str,
        task: asyncio.Task[None],
        context: ExecutionContext,
        *,
        session_id: str = "",
        parent_run_id: str = "",
    ) -> None:
        if self._closed or parent_run_id in self._blocked or run_id in self._blocked:
            task.cancel()
            raise RuntimeError("subagent parent cancellation has been requested")
        if run_id in self._tasks:
            task.cancel()
            raise RuntimeError(f"subagent task already registered: {run_id}")
        self._parents[run_id] = parent_run_id
        self._tasks[run_id] = ActiveSubagent(
            task=task,
            context=context,
            session_id=session_id,
            parent_run_id=parent_run_id,
        )
        task.add_done_callback(lambda completed: self._task_finished(run_id, completed))

    def get(self, run_id: str) -> tuple[asyncio.Task[None], ExecutionContext] | None:
        """Compatibility accessor for active tasks only."""
        entry = self._tasks.get(run_id)
        if entry is None:
            return None
        return entry.task, entry.context

    def all(self) -> list[tuple[asyncio.Task[None], ExecutionContext]]:
        return [(entry.task, entry.context) for entry in self._tasks.values()]

    async def get_snapshot(
        self,
        run_id: str,
        *,
        session_id: str | None = None,
    ) -> SubagentRunSnapshot | None:
        async with self._database.session() as db_session:
            record = await StateRepository(db_session).get_run(run_id)
            if (
                record is None
                or record.kind != "subagent"
                or (session_id is not None and record.session_id != session_id)
            ):
                return None
            return self._snapshot(record)

    @staticmethod
    def _snapshot(record: RunRecord) -> SubagentRunSnapshot:
        text = (record.result or {}).get("text", "")
        return SubagentRunSnapshot(
            run_id=record.id, session_id=record.session_id,
            parent_run_id=record.parent_run_id or "", status=record.status,
            result=text if isinstance(text, str) else str(text), reason=record.reason,
        )

    async def cancel(
        self,
        run_id: str,
        *,
        session_id: str | None = None,
    ) -> bool:
        snapshot = await self.get_snapshot(run_id, session_id=session_id)
        if snapshot is None:
            return False
        run_ids = self.block_descendants({run_id})
        operations = []
        for child_id in run_ids:
            operation = self._cancellations.get(child_id)
            if (operation is None or operation.cancelled()
                    or (operation.done() and operation.exception() is not None)):
                operation = asyncio.create_task(self._cancel_one(child_id))
                self._cancellations[child_id] = operation
            operations.append(operation)
        results = await asyncio.shield(asyncio.gather(*operations))
        return any(results)

    async def _cancel_one(self, run_id: str) -> bool:
        snapshot = await self.get_snapshot(run_id)
        if snapshot is None or snapshot.status in _TERMINAL:
            return False
        entry = self._tasks.get(run_id)
        if entry is not None and not entry.task.done():
            if not entry.task.cancelling():
                entry.task.cancel()
            await asyncio.gather(entry.task, return_exceptions=True)
        if run_id in self._cleanup_pending:
            raise RuntimeCleanupPending(run_id)
        await self.finish(run_id, status="cancelled", reason="cancelled")
        return True

    async def cancel_session(self, session_id: str) -> None:
        run_ids = [
            run_id
            for run_id, entry in self._tasks.items()
            if entry.session_id == session_id and not entry.task.done()
        ]
        for run_id in run_ids:
            await self.cancel(run_id, session_id=session_id)

    async def shutdown(self) -> None:
        self._closed = True
        try:
            await self._shutdown_tasks()
        finally:
            for run_id in tuple(self._writers):
                await self._close_writer(run_id)

    async def _shutdown_tasks(self) -> None:
        entries = list(self._tasks.items())
        for _, entry in entries:
            if not entry.task.done() and not entry.task.cancelling():
                entry.task.cancel()
        if entries:
            await asyncio.gather(
                *(entry.task for _, entry in entries),
                return_exceptions=True,
            )
        for run_id, _ in entries:
            if self.cleanup_is_pending(run_id):
                continue
            await self.finish(
                run_id,
                status="cancelled",
                reason="core_shutdown",
            )
        self._tasks.clear()
        if self._finalizers:
            await asyncio.gather(*tuple(self._finalizers), return_exceptions=True)

    def _task_finished(self, run_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(run_id, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            log.error(
                "background subagent escaped run_id=%s",
                run_id,
                exc_info=(type(error), error, error.__traceback__),
            )


async def _complete_before_cancelling[T](coro: Coroutine[object, object, T]) -> T:
    """Complete the transaction and notification before propagating repeated cancellation."""
    operation = asyncio.create_task(coro)
    await finish_cleanup(
        operation, failure_message="subagent state update failed during cancellation",
    )
    return operation.result()


__all__ = [
    "ActiveSubagent",
    "BackgroundTaskRegistry",
    "SubagentRunSnapshot",
]
