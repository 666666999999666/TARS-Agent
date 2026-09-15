"""Create the durable runtime state schema.

Revision ID: 0001
Revises:
Create Date: 2026-08-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=32), server_default="chat", nullable=False),
        sa.Column("status", sa.String(length=16), server_default="ready", nullable=False),
        sa.Column("title", sa.String(length=512), server_default="", nullable=False),
        sa.Column("workspace_root", sa.Text(), nullable=True),
        sa.Column("active_run_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('ready', 'running', 'closed')", name="ck_sessions_status"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sessions_updated_at", "sessions", ["updated_at"], unique=False)

    op.create_table(
        "turns",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("client_message_id", sa.String(length=128), nullable=True),
        sa.Column("raw_content", sa.Text(), nullable=False),
        sa.Column("effective_content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="queued", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', "
            "'interrupted')",
            name="ck_turns_status",
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "client_message_id", name="uq_turns_client_message"),
    )
    op.create_index(
        "ix_turns_session_created", "turns", ["session_id", "created_at"], unique=False
    )

    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("turn_id", sa.String(length=64), nullable=True),
        sa.Column("parent_run_id", sa.String(length=64), nullable=True),
        sa.Column("retry_of_run_id", sa.String(length=64), nullable=True),
        sa.Column("kind", sa.String(length=32), server_default="chat", nullable=False),
        sa.Column("attempt", sa.Integer(), server_default="1", nullable=False),
        sa.Column("status", sa.String(length=16), server_default="queued", nullable=False),
        sa.Column("reason", sa.String(length=128), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("side_effects_started", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("attempt >= 1", name="ck_runs_attempt_positive"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', "
            "'interrupted')",
            name="ck_runs_status",
        ),
        sa.ForeignKeyConstraint(["parent_run_id"], ["runs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["retry_of_run_id"], ["runs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["turn_id"], ["turns.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_runs_parent_run_id", "runs", ["parent_run_id"], unique=False)
    op.create_index(
        "ix_runs_session_created", "runs", ["session_id", "created_at"], unique=False
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("turn_id", sa.String(length=64), nullable=True),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column("committed", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("active", sa.Boolean(), server_default="1", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
        sa.CheckConstraint("sequence >= 0", name="ck_messages_sequence_nonnegative"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["turn_id"], ["turns.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "sequence", name="uq_messages_session_sequence"),
    )
    op.create_index(
        "ix_messages_active_context",
        "messages",
        ["session_id", "committed", "active", "sequence"],
        unique=False,
    )

    op.create_table(
        "tool_invocations",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("parameter_digest", sa.String(length=128), nullable=True),
        sa.Column("backend", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="queued", nullable=False),
        sa.Column("may_have_side_effects", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("retryable", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("error_class", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "backend IN ('in_process', 'workspace_sandbox', 'external', 'host')",
            name="ck_tool_invocations_backend",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled', "
            "'interrupted')",
            name="ck_tool_invocations_status",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_tool_invocations_run_created",
        "tool_invocations",
        ["run_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "events",
        sa.Column("cursor", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_schema_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "event_schema_version >= 1", name="ck_events_schema_version_positive"
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("cursor"),
    )
    op.create_index("ix_events_run_cursor", "events", ["run_id", "cursor"], unique=False)
    op.create_index(
        "ix_events_session_cursor", "events", ["session_id", "cursor"], unique=False
    )

    op.create_table(
        "compactions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("summary_message_id", sa.Integer(), nullable=True),
        sa.Column("start_sequence", sa.Integer(), nullable=False),
        sa.Column("end_sequence", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("original_tokens", sa.Integer(), nullable=True),
        sa.Column("summary_tokens", sa.Integer(), nullable=True),
        sa.Column("context_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("context_version >= 1", name="ck_compactions_version_positive"),
        sa.CheckConstraint("end_sequence >= start_sequence", name="ck_compactions_range"),
        sa.CheckConstraint("start_sequence >= 0", name="ck_compactions_start_nonnegative"),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["summary_message_id"], ["messages.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_compactions_session_version",
        "compactions",
        ["session_id", "context_version"],
        unique=True,
    )

    op.create_table(
        "migration_markers",
        sa.Column("source_key", sa.String(length=256), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("fingerprint", sa.String(length=128), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("source_key"),
    )
    op.create_table(
        "migration_issues",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_key", sa.String(length=256), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("line_number", sa.Integer(), nullable=True),
        sa.Column("issue_code", sa.String(length=128), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("raw_content", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_migration_issues_source",
        "migration_issues",
        ["source_key", "id"],
        unique=False,
    )
    op.create_table(
        "state_metadata",
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("state_metadata")
    op.drop_index("ix_migration_issues_source", table_name="migration_issues")
    op.drop_table("migration_issues")
    op.drop_table("migration_markers")
    op.drop_index("ix_compactions_session_version", table_name="compactions")
    op.drop_table("compactions")
    op.drop_index("ix_events_session_cursor", table_name="events")
    op.drop_index("ix_events_run_cursor", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_tool_invocations_run_created", table_name="tool_invocations")
    op.drop_table("tool_invocations")
    op.drop_index("ix_messages_active_context", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_runs_session_created", table_name="runs")
    op.drop_index("ix_runs_parent_run_id", table_name="runs")
    op.drop_table("runs")
    op.drop_index("ix_turns_session_created", table_name="turns")
    op.drop_table("turns")
    op.drop_index("ix_sessions_updated_at", table_name="sessions")
    op.drop_table("sessions")
