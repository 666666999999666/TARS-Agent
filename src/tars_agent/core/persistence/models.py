from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Final, Literal

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

type JsonObject = dict[str, Any]
SessionMode = Literal["one_shot", "chat"]

SESSION_STATUSES: Final = ("ready", "running", "closed")
TURN_STATUSES: Final = ("queued", "running", "succeeded", "failed", "cancelled", "interrupted")
RUN_STATUSES: Final = ("queued", "running", "succeeded", "failed", "cancelled", "interrupted")
TOOL_INVOCATION_STATUSES: Final = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
)
TOOL_BACKENDS: Final = ("in_process", "workspace_sandbox", "external", "host")
EVENT_SCHEMA_VERSION: Final = 1


def utc_now() -> datetime:
    return datetime.now(UTC)


def _allowed(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({quoted})"


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint(_allowed("status", SESSION_STATUSES), name="ck_sessions_status"),
        Index("ix_sessions_updated_at", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mode: Mapped[str] = mapped_column(String(32), default="chat", server_default="chat")
    status: Mapped[str] = mapped_column(String(16), default="ready", server_default="ready")
    title: Mapped[str] = mapped_column(String(512), default="", server_default="")
    workspace_root: Mapped[str | None] = mapped_column(Text, nullable=True)
    active_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TurnRecord(Base):
    __tablename__ = "turns"
    __table_args__ = (
        CheckConstraint(_allowed("status", TURN_STATUSES), name="ck_turns_status"),
        UniqueConstraint("session_id", "client_message_id", name="uq_turns_client_message"),
        Index("ix_turns_session_created", "session_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    client_message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    raw_content: Mapped[str] = mapped_column(Text)
    effective_content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="queued", server_default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class RunRecord(Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(_allowed("status", RUN_STATUSES), name="ck_runs_status"),
        CheckConstraint("attempt >= 1", name="ck_runs_attempt_positive"),
        Index("ix_runs_session_created", "session_id", "created_at"),
        Index("ix_runs_parent_run_id", "parent_run_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    turn_id: Mapped[str | None] = mapped_column(
        ForeignKey("turns.id", ondelete="SET NULL"), nullable=True
    )
    parent_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    retry_of_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(32), default="chat", server_default="chat")
    attempt: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    status: Mapped[str] = mapped_column(String(16), default="queued", server_default="queued")
    reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    execution_options: Mapped[JsonObject] = mapped_column(JSON, default=dict)
    result: Mapped[JsonObject | None] = mapped_column(JSON, nullable=True)
    side_effects_started: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MessageRecord(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("session_id", "sequence", name="uq_messages_session_sequence"),
        CheckConstraint("sequence >= 0", name="ck_messages_sequence_nonnegative"),
        CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
        Index("ix_messages_active_context", "session_id", "committed", "active", "sequence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    turn_id: Mapped[str | None] = mapped_column(
        ForeignKey("turns.id", ondelete="SET NULL"), nullable=True
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[Any] = mapped_column(JSON)
    committed: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ToolInvocationRecord(Base):
    __tablename__ = "tool_invocations"
    __table_args__ = (
        CheckConstraint(
            _allowed("status", TOOL_INVOCATION_STATUSES),
            name="ck_tool_invocations_status",
        ),
        CheckConstraint(_allowed("backend", TOOL_BACKENDS), name="ck_tool_invocations_backend"),
        Index("ix_tool_invocations_run_created", "run_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    tool_name: Mapped[str] = mapped_column(String(128))
    parameters: Mapped[JsonObject] = mapped_column(JSON)
    parameter_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    backend: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="queued", server_default="queued")
    may_have_side_effects: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    retryable: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    result: Mapped[JsonObject | None] = mapped_column(JSON, nullable=True)
    error_class: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EventRecord(Base):
    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint(
            "event_schema_version >= 1",
            name="ck_events_schema_version_positive",
        ),
        Index("ix_events_session_cursor", "session_id", "cursor"),
        Index("ix_events_run_cursor", "run_id", "cursor"),
    )

    # SQLite requires the exact INTEGER type for rowid-backed autoincrement.
    cursor: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_schema_version: Mapped[int] = mapped_column(
        Integer,
        default=EVENT_SCHEMA_VERSION,
        server_default=str(EVENT_SCHEMA_VERSION),
    )
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(128))
    payload: Mapped[JsonObject] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class CompactionRecord(Base):
    __tablename__ = "compactions"
    __table_args__ = (
        CheckConstraint("start_sequence >= 0", name="ck_compactions_start_nonnegative"),
        CheckConstraint("end_sequence >= start_sequence", name="ck_compactions_range"),
        CheckConstraint("context_version >= 1", name="ck_compactions_version_positive"),
        Index("ix_compactions_session_version", "session_id", "context_version", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    summary_message_id: Mapped[int | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    start_sequence: Mapped[int] = mapped_column(Integer)
    end_sequence: Mapped[int] = mapped_column(Integer)
    summary: Mapped[str] = mapped_column(Text)
    original_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MigrationMarkerRecord(Base):
    __tablename__ = "migration_markers"

    source_key: Mapped[str] = mapped_column(String(256), primary_key=True)
    source_path: Mapped[str] = mapped_column(Text)
    fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    details: Mapped[JsonObject] = mapped_column(JSON, default=dict)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class MigrationIssueRecord(Base):
    __tablename__ = "migration_issues"
    __table_args__ = (Index("ix_migration_issues_source", "source_key", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_key: Mapped[str] = mapped_column(String(256))
    source_path: Mapped[str] = mapped_column(Text)
    line_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    issue_code: Mapped[str] = mapped_column(String(128))
    detail: Mapped[str] = mapped_column(Text)
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class StateMetadataRecord(Base):
    __tablename__ = "state_metadata"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[JsonObject] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
