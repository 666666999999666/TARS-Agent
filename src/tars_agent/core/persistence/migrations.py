from __future__ import annotations

import os
import re
import shutil
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config

UpgradeCallable = Callable[[Path, str], None]
Clock = Callable[[], datetime]

_SQLITE_TIMEOUT_SECONDS = 5.0
_SAFE_LABEL_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
CURRENT_SCHEMA_REVISION = "0003"


class MigrationOperationError(RuntimeError):
    """Base error for an upgrade that must prevent the daemon from starting."""


class MigrationBackupError(MigrationOperationError):
    """The pre-upgrade database could not be checkpointed or backed up."""


class MigrationUpgradeError(MigrationOperationError):
    """The schema upgrade failed and startup must stop."""

    def __init__(
        self,
        message: str,
        *,
        database_path: Path,
        backup_path: Path | None,
        failed_database_path: Path | None,
        rollback_succeeded: bool,
        rollback_error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.database_path = database_path
        self.backup_path = backup_path
        self.failed_database_path = failed_database_path
        self.rollback_succeeded = rollback_succeeded
        self.rollback_error = rollback_error


@dataclass(frozen=True, slots=True)
class MigrationUpgradeResult:
    """Evidence produced by a completed schema upgrade."""

    database_path: Path
    backup_path: Path | None
    source_revision: str | None
    requested_revision: str
    final_revision: str | None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_label(value: str | None) -> str:
    label = value or "unversioned"
    sanitized = _SAFE_LABEL_PATTERN.sub("-", label).strip("-.")
    return sanitized[:80] or "unversioned"


def _timestamp(clock: Clock) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _versioned_path(
    directory: Path,
    database_path: Path,
    *,
    kind: str,
    source_revision: str | None,
    target_revision: str,
    timestamp: str,
) -> Path:
    stem = database_path.stem
    suffix = database_path.suffix or ".db"
    source = _safe_label(source_revision)
    target = _safe_label(target_revision)
    base = f"{stem}.{kind}-{source}-to-{target}-{timestamp}"
    candidate = directory / f"{base}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{base}-{counter}{suffix}"
        counter += 1
    return candidate


def read_schema_revision(database_path: Path) -> str | None:
    if not database_path.exists():
        return None
    with closing(sqlite3.connect(database_path, timeout=_SQLITE_TIMEOUT_SECONDS)) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'alembic_version'"
        ).fetchone()
        if table is None:
            return None
        rows = connection.execute(
            "SELECT version_num FROM alembic_version ORDER BY version_num"
        ).fetchall()
    revisions = [str(row[0]) for row in rows]
    return "+".join(revisions) if revisions else None


def _validate_sqlite_database(database_path: Path) -> None:
    if not database_path.is_file():
        raise sqlite3.DatabaseError(f"SQLite database does not exist: {database_path}")
    with closing(sqlite3.connect(database_path, timeout=_SQLITE_TIMEOUT_SECONDS)) as connection:
        rows = connection.execute("PRAGMA quick_check").fetchall()
    results = [str(row[0]) for row in rows]
    if results != ["ok"]:
        details = "; ".join(results) if results else "no result"
        raise sqlite3.DatabaseError(f"SQLite quick_check failed: {details}")


def _checkpoint_and_backup(database_path: Path, backup_path: Path) -> None:
    temporary_path = backup_path.with_name(f".{backup_path.name}.{uuid4().hex}.tmp")
    try:
        with closing(
            sqlite3.connect(
                database_path,
                timeout=_SQLITE_TIMEOUT_SECONDS,
                isolation_level=None,
            )
        ) as source:
            checkpoint = source.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is not None and int(checkpoint[0]) != 0:
                raise sqlite3.OperationalError(
                    "SQLite WAL checkpoint was busy; close active database connections"
                )
            with closing(sqlite3.connect(temporary_path)) as destination:
                source.backup(destination)
        _validate_sqlite_database(temporary_path)
        os.replace(temporary_path, backup_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sidecar_paths(database_path: Path) -> tuple[Path, Path]:
    return (
        database_path.with_name(f"{database_path.name}-wal"),
        database_path.with_name(f"{database_path.name}-shm"),
    )


def _preserve_failed_database(
    database_path: Path,
    failed_database_path: Path,
) -> tuple[Path | None, BaseException | None]:
    if not database_path.exists():
        return None, None
    try:
        shutil.copy2(database_path, failed_database_path)
        for source_sidecar in _sidecar_paths(database_path):
            if source_sidecar.exists():
                target_sidecar = failed_database_path.with_name(
                    f"{failed_database_path.name}{source_sidecar.name.removeprefix(database_path.name)}"
                )
                shutil.copy2(source_sidecar, target_sidecar)
    except BaseException as exc:
        return failed_database_path if failed_database_path.exists() else None, exc
    return failed_database_path, None


def _remove_database_and_sidecars(database_path: Path) -> None:
    for path in (*_sidecar_paths(database_path), database_path):
        path.unlink(missing_ok=True)


def _restore_backup(database_path: Path, backup_path: Path | None) -> None:
    if backup_path is None:
        _remove_database_and_sidecars(database_path)
        return

    restore_path = database_path.with_name(f".{database_path.name}.{uuid4().hex}.restore")
    try:
        shutil.copy2(backup_path, restore_path)
        _validate_sqlite_database(restore_path)
        for sidecar_path in _sidecar_paths(database_path):
            sidecar_path.unlink(missing_ok=True)
        os.replace(restore_path, database_path)
        _validate_sqlite_database(database_path)
    finally:
        restore_path.unlink(missing_ok=True)


def upgrade_database(
    database_path: Path,
    upgrade: UpgradeCallable,
    *,
    target_revision: str = "head",
    backup_dir: Path | None = None,
    clock: Clock = _utc_now,
) -> MigrationUpgradeResult:
    """Checkpoint, back up, and upgrade a SQLite database.

    This function is deliberately synchronous and must run before the async
    SQLAlchemy engine is opened. ``upgrade`` must close every connection it
    creates before returning or raising.

    Any backup, upgrade, validation, or rollback failure raises an exception.
    Callers must treat that exception as a startup-blocking condition.
    """

    path = database_path.expanduser().resolve()
    destination_dir = (backup_dir or path.parent / "backups").expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp(clock)

    database_existed = path.exists()
    source_revision: str | None = None
    backup_path: Path | None = None
    if database_existed:
        try:
            source_revision = read_schema_revision(path)
            if target_revision != "head" and source_revision == target_revision:
                _validate_sqlite_database(path)
                return MigrationUpgradeResult(
                    database_path=path,
                    backup_path=None,
                    source_revision=source_revision,
                    requested_revision=target_revision,
                    final_revision=source_revision,
                )
            backup_path = _versioned_path(
                destination_dir,
                path,
                kind="pre-upgrade",
                source_revision=source_revision,
                target_revision=target_revision,
                timestamp=timestamp,
            )
            _checkpoint_and_backup(path, backup_path)
        except BaseException as exc:
            if backup_path is not None:
                backup_path.unlink(missing_ok=True)
            raise MigrationBackupError(
                f"Cannot create a verified pre-upgrade backup for {path}"
            ) from exc

    try:
        upgrade(path, target_revision)
        _validate_sqlite_database(path)
        final_revision = read_schema_revision(path)
        if target_revision != "head" and final_revision != target_revision:
            raise RuntimeError(
                f"Upgrade returned at revision {final_revision!r}; expected {target_revision!r}"
            )
    except BaseException as upgrade_error:
        failed_path = _versioned_path(
            destination_dir,
            path,
            kind="failed-upgrade",
            source_revision=source_revision,
            target_revision=target_revision,
            timestamp=timestamp,
        )
        preserved_path, preservation_error = _preserve_failed_database(path, failed_path)

        rollback_error: BaseException | None = None
        try:
            _restore_backup(path, backup_path)
        except BaseException as exc:
            rollback_error = exc

        combined_rollback_error = rollback_error or preservation_error
        rollback_succeeded = rollback_error is None
        detail = "the previous database was restored" if rollback_succeeded else "rollback failed"
        raise MigrationUpgradeError(
            f"Database upgrade to {target_revision!r} failed; {detail}; startup is blocked",
            database_path=path,
            backup_path=backup_path,
            failed_database_path=preserved_path,
            rollback_succeeded=rollback_succeeded,
            rollback_error=combined_rollback_error,
        ) from upgrade_error

    return MigrationUpgradeResult(
        database_path=path,
        backup_path=backup_path,
        source_revision=source_revision,
        requested_revision=target_revision,
        final_revision=final_revision,
    )


def run_alembic_upgrade(database_path: Path, target_revision: str) -> None:
    """Run the packaged Alembic environment against one SQLite database."""

    path = database_path.expanduser().resolve()
    script_location = Path(__file__).with_name("alembic")
    config = Config()
    config.set_main_option("script_location", str(script_location))
    # ConfigParser treats percent characters as interpolation markers.
    sqlite_url = f"sqlite:///{path.as_posix()}".replace("%", "%%")
    config.set_main_option("sqlalchemy.url", sqlite_url)
    command.upgrade(config, target_revision)


def ensure_current_schema(
    database_path: Path,
    *,
    backup_dir: Path | None = None,
) -> MigrationUpgradeResult:
    """Bring ``database_path`` to the schema revision shipped with this build."""

    return upgrade_database(
        database_path,
        run_alembic_upgrade,
        target_revision=CURRENT_SCHEMA_REVISION,
        backup_dir=backup_dir,
    )


__all__ = [
    "CURRENT_SCHEMA_REVISION",
    "MigrationBackupError",
    "MigrationOperationError",
    "MigrationUpgradeError",
    "MigrationUpgradeResult",
    "UpgradeCallable",
    "ensure_current_schema",
    "read_schema_revision",
    "run_alembic_upgrade",
    "upgrade_database",
]
