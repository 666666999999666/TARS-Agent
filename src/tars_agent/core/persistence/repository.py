from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import Select, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from tars_agent.core.persistence.models import (
    EVENT_SCHEMA_VERSION,
    CompactionRecord,
    EventRecord,
    MessageRecord,
    MigrationIssueRecord,
    MigrationMarkerRecord,
    RunRecord,
    SessionRecord,
    StateMetadataRecord,
    ToolInvocationRecord,
    TurnRecord,
)


class StateRepository:
    """Small persistence facade whose caller owns the transaction boundary."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_session(self, record: SessionRecord) -> SessionRecord:
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_session(self, session_id: str) -> SessionRecord | None:
        return await self.session.get(SessionRecord, session_id)

    async def list_sessions(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[SessionRecord]:
        statement = select(SessionRecord)
        if status is not None:
            statement = statement.where(SessionRecord.status == status)
        statement = statement.order_by(SessionRecord.updated_at.desc()).limit(limit).offset(offset)
        return (await self.session.scalars(statement)).all()

    async def add_turn(self, record: TurnRecord) -> TurnRecord:
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_turn(self, turn_id: str) -> TurnRecord | None:
        return await self.session.get(TurnRecord, turn_id)

    async def get_turn_by_client_message(
        self,
        session_id: str,
        client_message_id: str,
    ) -> TurnRecord | None:
        statement = select(TurnRecord).where(
            TurnRecord.session_id == session_id,
            TurnRecord.client_message_id == client_message_id,
        )
        return (await self.session.scalars(statement)).one_or_none()

    async def add_run(self, record: RunRecord) -> RunRecord:
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_run(self, run_id: str) -> RunRecord | None:
        return await self.session.get(RunRecord, run_id)

    async def list_runs(
        self,
        session_id: str,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> Sequence[RunRecord]:
        statement = select(RunRecord).where(RunRecord.session_id == session_id)
        if status is not None:
            statement = statement.where(RunRecord.status == status)
        statement = statement.order_by(RunRecord.created_at.desc()).limit(limit)
        return (await self.session.scalars(statement)).all()

    async def latest_run(self, session_id: str) -> RunRecord | None:
        statement = (
            select(RunRecord)
            .where(RunRecord.session_id == session_id)
            .order_by(RunRecord.created_at.desc())
            .limit(1)
        )
        return (await self.session.scalars(statement)).one_or_none()

    async def get_run_for_turn(self, turn_id: str) -> RunRecord | None:
        statement = (
            select(RunRecord)
            .where(RunRecord.turn_id == turn_id)
            .order_by(RunRecord.attempt.desc())
            .limit(1)
        )
        return (await self.session.scalars(statement)).one_or_none()

    async def list_runs_with_statuses(
        self,
        statuses: Iterable[str],
    ) -> Sequence[RunRecord]:
        values = tuple(statuses)
        if not values:
            return []
        statement = select(RunRecord).where(RunRecord.status.in_(values))
        return (await self.session.scalars(statement)).all()

    async def add_message(self, record: MessageRecord) -> MessageRecord:
        self.session.add(record)
        await self.session.flush()
        return record

    async def add_messages(self, records: Iterable[MessageRecord]) -> None:
        self.session.add_all(records)
        await self.session.flush()

    async def list_run_messages(self, run_id: str) -> Sequence[MessageRecord]:
        statement = (
            select(MessageRecord)
            .where(MessageRecord.run_id == run_id)
            .order_by(MessageRecord.sequence)
        )
        return (await self.session.scalars(statement)).all()

    async def next_message_sequence(self, session_id: str) -> int:
        statement = select(func.max(MessageRecord.sequence)).where(
            MessageRecord.session_id == session_id
        )
        current = await self.session.scalar(statement)
        return int(current) + 1 if current is not None else 0

    async def set_run_messages_committed(self, run_id: str, committed: bool) -> None:
        await self.session.execute(
            update(MessageRecord)
            .where(MessageRecord.run_id == run_id)
            .values(committed=committed)
        )
        await self.session.flush()

    async def deactivate_messages_through(
        self,
        session_id: str,
        end_sequence: int,
    ) -> None:
        await self.session.execute(
            update(MessageRecord)
            .where(
                MessageRecord.session_id == session_id,
                MessageRecord.sequence <= end_sequence,
            )
            .values(active=False)
        )
        await self.session.flush()

    async def latest_compaction_version(self, session_id: str) -> int:
        statement = select(func.max(CompactionRecord.context_version)).where(
            CompactionRecord.session_id == session_id
        )
        value = await self.session.scalar(statement)
        return int(value or 0)

    async def list_messages(
        self,
        session_id: str,
        *,
        committed_only: bool = True,
        active_only: bool = True,
        after_sequence: int | None = None,
        limit: int = 2_000,
    ) -> Sequence[MessageRecord]:
        statement = select(MessageRecord).where(MessageRecord.session_id == session_id)
        if committed_only:
            statement = statement.where(MessageRecord.committed.is_(True))
        if active_only:
            statement = statement.where(MessageRecord.active.is_(True))
        if after_sequence is not None:
            statement = statement.where(MessageRecord.sequence > after_sequence)
        statement = statement.order_by(MessageRecord.sequence).limit(limit)
        return (await self.session.scalars(statement)).all()

    async def add_tool_invocation(
        self,
        record: ToolInvocationRecord,
    ) -> ToolInvocationRecord:
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_tool_invocation(self, invocation_id: str) -> ToolInvocationRecord | None:
        return await self.session.get(ToolInvocationRecord, invocation_id)

    async def list_tool_invocations(self, run_id: str) -> Sequence[ToolInvocationRecord]:
        statement = (
            select(ToolInvocationRecord)
            .where(ToolInvocationRecord.run_id == run_id)
            .order_by(ToolInvocationRecord.created_at)
        )
        return (await self.session.scalars(statement)).all()

    async def close_unfinished_tool_invocations(
        self,
        run_id: str,
        *,
        terminal_status: Literal["cancelled", "interrupted"],
        reason: str,
        finished_at: datetime,
    ) -> None:
        """Close missing results in the owner's terminal transaction, never invent success."""
        await self.session.execute(update(ToolInvocationRecord).where(
            ToolInvocationRecord.run_id == run_id,
            ToolInvocationRecord.status.in_(("queued", "running")),
        ).values(
            status=terminal_status,
            finished_at=finished_at,
            retryable=False,
            error_class=f"{terminal_status}_without_result",
            error_message=(
                f"Run ended ({reason}); no final tool result was received. "
                "Prior side effects are not assumed to be rolled back."
            ),
        ))

    async def append_event(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        session_id: str | None = None,
        run_id: str | None = None,
        event_schema_version: int = EVENT_SCHEMA_VERSION,
    ) -> EventRecord:
        record = EventRecord(
            event_schema_version=event_schema_version,
            session_id=session_id,
            run_id=run_id,
            event_type=event_type,
            payload=payload,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def add_events(self, records: Iterable[EventRecord]) -> Sequence[EventRecord]:
        materialized = list(records)
        if not materialized:
            return materialized
        self.session.add_all(materialized)
        await self.session.flush()
        return materialized

    async def resolve_run_sessions(self, run_ids: Iterable[str]) -> dict[str, str]:
        values = tuple(run_ids)
        if not values:
            return {}
        statement = select(RunRecord.id, RunRecord.session_id).where(
            RunRecord.id.in_(values)
        )
        rows = await self.session.execute(statement)
        return {run_id: session_id for run_id, session_id in rows}

    async def list_events(
        self,
        *,
        after_cursor: int = 0,
        through_cursor: int | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 2_000,
    ) -> Sequence[EventRecord]:
        statement: Select[tuple[EventRecord]] = select(EventRecord).where(
            EventRecord.cursor > after_cursor
        )
        if through_cursor is not None:
            statement = statement.where(EventRecord.cursor <= through_cursor)
        if session_id is not None:
            statement = statement.where(EventRecord.session_id == session_id)
        if run_id is not None:
            statement = statement.where(EventRecord.run_id == run_id)
        statement = statement.order_by(EventRecord.cursor).limit(limit)
        return (await self.session.scalars(statement)).all()

    async def latest_event_cursor(self, *, session_id: str | None = None) -> int:
        statement = select(func.max(EventRecord.cursor))
        if session_id is not None:
            statement = statement.where(EventRecord.session_id == session_id)
        value = await self.session.scalar(statement)
        return int(value or 0)

    async def add_compaction(self, record: CompactionRecord) -> CompactionRecord:
        self.session.add(record)
        await self.session.flush()
        return record

    async def add_migration_issue(
        self,
        record: MigrationIssueRecord,
    ) -> MigrationIssueRecord:
        self.session.add(record)
        await self.session.flush()
        return record

    async def list_migration_issues(
        self,
        source_key: str,
    ) -> Sequence[MigrationIssueRecord]:
        statement = (
            select(MigrationIssueRecord)
            .where(MigrationIssueRecord.source_key == source_key)
            .order_by(MigrationIssueRecord.id)
        )
        return (await self.session.scalars(statement)).all()

    async def add_migration_marker(
        self,
        record: MigrationMarkerRecord,
    ) -> MigrationMarkerRecord:
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_migration_marker(self, source_key: str) -> MigrationMarkerRecord | None:
        return await self.session.get(MigrationMarkerRecord, source_key)

    async def get_metadata(self, key: str) -> StateMetadataRecord | None:
        return await self.session.get(StateMetadataRecord, key)

    async def set_metadata(self, key: str, value: dict[str, Any]) -> StateMetadataRecord:
        record = await self.get_metadata(key)
        if record is None:
            record = StateMetadataRecord(key=key, value=value)
            self.session.add(record)
        else:
            record.value = value
        await self.session.flush()
        return record
