from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tars_agent.core.persistence.migrations import (
    CURRENT_SCHEMA_REVISION,
    MigrationBackupError,
    MigrationUpgradeError,
    ensure_current_schema,
    read_schema_revision,
    upgrade_database,
)

_FIXED_TIME = datetime(2026, 8, 13, 12, 34, 56, tzinfo=UTC)


def _create_database(path: Path, *, revision: str = "rev_1", value: str = "before") -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE alembic_version (version_num TEXT NOT NULL)")
        connection.execute("INSERT INTO alembic_version VALUES (?)", (revision,))
        connection.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        connection.execute("INSERT INTO sample VALUES (?)", (value,))
        connection.commit()


def _value(path: Path) -> str:
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute("SELECT value FROM sample").fetchone()
    assert row is not None
    return str(row[0])


def _revision(path: Path) -> str:
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert row is not None
    return str(row[0])


def test_success_creates_versioned_verified_backup_before_upgrade(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    _create_database(database_path)
    calls: list[tuple[Path, str]] = []

    def upgrade(path: Path, target: str) -> None:
        calls.append((path, target))
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("UPDATE sample SET value = 'after'")
            connection.execute("UPDATE alembic_version SET version_num = 'rev_2'")
            connection.commit()

    result = upgrade_database(
        database_path,
        upgrade,
        target_revision="rev_2",
        clock=lambda: _FIXED_TIME,
    )

    assert calls == [(database_path.resolve(), "rev_2")]
    assert result.source_revision == "rev_1"
    assert result.final_revision == "rev_2"
    assert result.backup_path is not None
    assert result.backup_path.is_file()
    assert "pre-upgrade-rev_1-to-rev_2-20260813T123456000000Z" in result.backup_path.name
    assert _value(database_path) == "after"
    assert _value(result.backup_path) == "before"
    assert _revision(result.backup_path) == "rev_1"


def test_upgrade_failure_restores_old_database_and_preserves_failed_copy(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    _create_database(database_path)

    def failing_upgrade(path: Path, _target: str) -> None:
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("UPDATE sample SET value = 'partially-upgraded'")
            connection.commit()
        raise RuntimeError("injected alembic failure")

    with pytest.raises(MigrationUpgradeError) as captured:
        upgrade_database(
            database_path,
            failing_upgrade,
            target_revision="rev_2",
            clock=lambda: _FIXED_TIME,
        )

    error = captured.value
    assert error.rollback_succeeded is True
    assert error.backup_path is not None and error.backup_path.is_file()
    assert error.failed_database_path is not None and error.failed_database_path.is_file()
    assert _value(database_path) == "before"
    assert _revision(database_path) == "rev_1"
    assert _value(error.failed_database_path) == "partially-upgraded"


def test_invalid_database_blocks_upgrade_before_callable_runs(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    database_path.write_bytes(b"not a sqlite database")
    called = False

    def upgrade(_path: Path, _target: str) -> None:
        nonlocal called
        called = True

    with pytest.raises(MigrationBackupError):
        upgrade_database(database_path, upgrade)

    assert called is False
    assert database_path.read_bytes() == b"not a sqlite database"


def test_new_database_upgrade_does_not_invent_pre_upgrade_backup(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"

    def bootstrap(path: Path, _target: str) -> None:
        _create_database(path, revision="rev_1", value="new")

    result = upgrade_database(database_path, bootstrap, target_revision="rev_1")

    assert result.backup_path is None
    assert result.source_revision is None
    assert result.final_revision == "rev_1"
    assert _value(database_path) == "new"


def test_failed_new_database_is_moved_aside_and_missing_state_is_restored(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"

    def failing_bootstrap(path: Path, _target: str) -> None:
        _create_database(path, revision="rev_1", value="partial")
        raise RuntimeError("injected bootstrap failure")

    with pytest.raises(MigrationUpgradeError) as captured:
        upgrade_database(database_path, failing_bootstrap, target_revision="rev_1")

    error = captured.value
    assert error.rollback_succeeded is True
    assert error.backup_path is None
    assert error.failed_database_path is not None and error.failed_database_path.is_file()
    assert not database_path.exists()
    assert _value(error.failed_database_path) == "partial"


def test_post_upgrade_corruption_is_treated_as_failure_and_rolled_back(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    _create_database(database_path)

    def corrupting_upgrade(path: Path, _target: str) -> None:
        path.write_bytes(b"broken by injected upgrade")

    with pytest.raises(MigrationUpgradeError) as captured:
        upgrade_database(database_path, corrupting_upgrade, target_revision="rev_2")

    assert captured.value.rollback_succeeded is True
    assert _value(database_path) == "before"
    assert captured.value.failed_database_path is not None
    assert captured.value.failed_database_path.read_bytes() == b"broken by injected upgrade"


def test_backup_names_do_not_collide_for_same_timestamp(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    _create_database(database_path)

    def no_op_upgrade(_path: Path, _target: str) -> None:
        return

    first = upgrade_database(database_path, no_op_upgrade, clock=lambda: _FIXED_TIME)
    second = upgrade_database(database_path, no_op_upgrade, clock=lambda: _FIXED_TIME)

    assert first.backup_path is not None
    assert second.backup_path is not None
    assert first.backup_path != second.backup_path
    assert first.backup_path.is_file()
    assert second.backup_path.is_file()


def test_explicit_target_revision_mismatch_is_rolled_back(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    _create_database(database_path)

    def incomplete_upgrade(path: Path, _target: str) -> None:
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("UPDATE sample SET value = 'changed-without-revision'")
            connection.commit()

    with pytest.raises(MigrationUpgradeError) as captured:
        upgrade_database(database_path, incomplete_upgrade, target_revision="rev_2")

    assert captured.value.rollback_succeeded is True
    assert _revision(database_path) == "rev_1"
    assert _value(database_path) == "before"


def test_packaged_upgrade_is_idempotent_at_current_revision(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"

    first = ensure_current_schema(database_path)
    second = ensure_current_schema(database_path)

    assert first.source_revision is None
    assert first.final_revision == CURRENT_SCHEMA_REVISION
    assert first.backup_path is None
    assert second.source_revision == CURRENT_SCHEMA_REVISION
    assert second.final_revision == CURRENT_SCHEMA_REVISION
    assert second.backup_path is None
    assert read_schema_revision(database_path) == CURRENT_SCHEMA_REVISION
    assert not list((tmp_path / "backups").glob("*.db"))


# 功能：验证已存在的 0001 数据库会先备份再升级到当前 0002 revision
# 设计：用正式 Alembic 环境停在 0001，再调用统一升级入口并检查新增列与备份 revision
def test_packaged_upgrade_from_0001_creates_verified_backup(tmp_path: Path) -> None:
    from alembic import command
    from alembic.config import Config

    database_path = tmp_path / "state.db"
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    command.upgrade(config, "0001")

    result = ensure_current_schema(database_path)

    assert result.source_revision == "0001"
    assert result.final_revision == CURRENT_SCHEMA_REVISION
    assert result.backup_path is not None
    assert read_schema_revision(result.backup_path) == "0001"
    with closing(sqlite3.connect(database_path)) as connection:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
    assert "execution_options" in columns


def test_0003_repairs_messages_from_unsuccessful_runs(tmp_path: Path) -> None:
    from alembic import command
    from alembic.config import Config

    database_path = tmp_path / "state.db"
    config = Config(str(Path(__file__).parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")
    command.upgrade(config, "0002")

    timestamp = "2026-08-13 00:00:00+00:00"
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            """
            INSERT INTO sessions (
                id, mode, status, title, workspace_root, active_run_id,
                created_at, updated_at, closed_at
            ) VALUES ('sess-1', 'chat', 'ready', '', NULL, NULL, ?, ?, NULL)
            """,
            (timestamp, timestamp),
        )
        for run_id, status in (
            ("run-ok", "succeeded"),
            ("run-failed", "failed"),
            ("run-interrupted", "interrupted"),
        ):
            connection.execute(
                """
                INSERT INTO runs (
                    id, session_id, turn_id, parent_run_id, retry_of_run_id,
                    kind, attempt, status, reason, result, side_effects_started,
                    created_at, updated_at, started_at, finished_at, execution_options
                ) VALUES (?, 'sess-1', NULL, NULL, NULL, 'chat', 1, ?, NULL,
                          NULL, 0, ?, ?, NULL, NULL, '{}')
                """,
                (run_id, status, timestamp, timestamp),
            )
        for sequence, run_id in enumerate(
            ("run-ok", "run-failed", "run-interrupted")
        ):
            connection.execute(
                """
                INSERT INTO messages (
                    session_id, turn_id, run_id, sequence, role, content,
                    committed, active, created_at
                ) VALUES ('sess-1', NULL, ?, ?, 'assistant', '"audit"', 1, 1, ?)
                """,
                (run_id, sequence, timestamp),
            )
        connection.commit()

    result = ensure_current_schema(database_path)

    assert result.source_revision == "0002"
    assert result.final_revision == "0003"
    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT runs.status, messages.committed, messages.active
              FROM messages
              JOIN runs ON runs.id = messages.run_id
             ORDER BY messages.sequence
            """
        ).fetchall()
    assert rows == [
        ("succeeded", 1, 1),
        ("failed", 0, 0),
        ("interrupted", 0, 0),
    ]
