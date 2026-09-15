from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect

from tars_agent.core.persistence import (
    BUSY_TIMEOUT_MS,
    CURRENT_SCHEMA_REVISION,
    Base,
    Database,
    EventRecord,
    MessageRecord,
    RunRecord,
    SessionRecord,
    StateRepository,
    TurnRecord,
)


@pytest.mark.asyncio
async def test_create_schema_has_all_runtime_tables_and_sqlite_pragmas(tmp_path: Path) -> None:
    database = Database(tmp_path / "nested" / "state.db")
    try:
        await database.create_schema()
        async with database.engine.connect() as connection:
            table_names = await connection.run_sync(
                lambda sync_connection: set(inspect(sync_connection).get_table_names())
            )
            foreign_keys = (await connection.exec_driver_sql("PRAGMA foreign_keys")).scalar_one()
            busy_timeout = (await connection.exec_driver_sql("PRAGMA busy_timeout")).scalar_one()
            journal_mode = (await connection.exec_driver_sql("PRAGMA journal_mode")).scalar_one()

        assert set(Base.metadata.tables) <= table_names
        assert foreign_keys == 1
        assert busy_timeout == BUSY_TIMEOUT_MS
        assert str(journal_mode).lower() == "wal"
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_transaction_commits_and_rolls_back_atomically(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    try:
        async with database.transaction() as session:
            await StateRepository(session).add_session(_session("committed"))

        with pytest.raises(RuntimeError, match="rollback"):
            async with database.transaction() as session:
                await StateRepository(session).add_session(_session("rolled-back"))
                raise RuntimeError("rollback")

        async with database.session() as session:
            repository = StateRepository(session)
            assert await repository.get_session("committed") is not None
            assert await repository.get_session("rolled-back") is None
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_repository_idempotency_lookup_messages_and_event_cursor(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    now = datetime.now(UTC)
    try:
        async with database.transaction() as session:
            repository = StateRepository(session)
            await repository.add_session(_session("sess-1"))
            await repository.add_turn(
                TurnRecord(
                    id="turn-1",
                    session_id="sess-1",
                    client_message_id="client-1",
                    raw_content="/review plan",
                    effective_content="Review the plan",
                    status="queued",
                    created_at=now,
                    updated_at=now,
                )
            )
            await repository.add_run(
                RunRecord(
                    id="run-1",
                    session_id="sess-1",
                    turn_id="turn-1",
                    kind="chat",
                    attempt=1,
                    status="running",
                )
            )
            await repository.add_messages(
                [
                    MessageRecord(
                        session_id="sess-1",
                        turn_id="turn-1",
                        run_id="run-1",
                        sequence=0,
                        role="user",
                        content="Review the plan",
                        committed=True,
                    ),
                    MessageRecord(
                        session_id="sess-1",
                        turn_id="turn-1",
                        run_id="run-1",
                        sequence=1,
                        role="assistant",
                        content=[{"type": "text", "text": "Done"}],
                        committed=False,
                    ),
                ]
            )
            event_one = await repository.append_event(
                session_id="sess-1",
                run_id="run-1",
                event_type="run.started",
                payload={"type": "run.started"},
            )
            event_two = await repository.append_event(
                session_id="sess-1",
                run_id="run-1",
                event_type="llm.token",
                payload={"type": "llm.token", "token": "x"},
            )
            assert event_one.cursor < event_two.cursor

        async with database.session() as session:
            repository = StateRepository(session)
            turn = await repository.get_turn_by_client_message("sess-1", "client-1")
            messages = await repository.list_messages("sess-1")
            events = await repository.list_events(
                session_id="sess-1", after_cursor=event_one.cursor
            )
            assert turn is not None and turn.id == "turn-1"
            assert [message.sequence for message in messages] == [0]
            assert [event.cursor for event in events] == [event_two.cursor]
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_repository_batches_events_and_bounds_replay_by_high_water(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    try:
        async with database.transaction() as session:
            repository = StateRepository(session)
            await repository.add_session(_session("sess-1"))
            await repository.add_run(
                RunRecord(
                    id="run-1",
                    session_id="sess-1",
                    kind="chat",
                    attempt=1,
                    status="running",
                )
            )
            records = await repository.add_events(
                [
                    EventRecord(
                        session_id="sess-1",
                        run_id="run-1",
                        event_type=f"event.{index}",
                        payload={"type": f"event.{index}"},
                    )
                    for index in range(3)
                ]
            )
            cursors = [record.cursor for record in records]

        async with database.session() as session:
            repository = StateRepository(session)
            assert await repository.resolve_run_sessions(["run-1", "missing"]) == {
                "run-1": "sess-1"
            }
            assert await repository.latest_event_cursor() == cursors[-1]
            replay = await repository.list_events(
                after_cursor=cursors[0],
                through_cursor=cursors[1],
            )
            assert [record.cursor for record in replay] == [cursors[1]]
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_foreign_keys_reject_orphan_records(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    try:
        with pytest.raises(Exception) as error:
            async with database.transaction() as session:
                session.add(
                    EventRecord(
                        session_id="missing",
                        event_type="session.created",
                        payload={"type": "session.created"},
                    )
                )
        assert "FOREIGN KEY constraint failed" in str(error.value)
    finally:
        await database.dispose()


def test_alembic_upgrade_and_downgrade(tmp_path: Path) -> None:
    from alembic import command
    from alembic.config import Config

    database_path = tmp_path / "alembic-state.db"
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")

    command.upgrade(config, "head")
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert set(Base.metadata.tables) <= tables
    assert revision == (CURRENT_SCHEMA_REVISION,)

    command.downgrade(config, "base")
    with sqlite3.connect(database_path) as connection:
        remaining = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert not set(Base.metadata.tables) & remaining


def _session(session_id: str) -> SessionRecord:
    now = datetime.now(UTC)
    return SessionRecord(
        id=session_id,
        mode="chat",
        status="ready",
        title="",
        workspace_root=None,
        created_at=now,
        updated_at=now,
    )
