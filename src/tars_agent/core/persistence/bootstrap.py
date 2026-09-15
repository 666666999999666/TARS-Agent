from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tars_agent.core.persistence.database import Database
from tars_agent.core.persistence.migrations import MigrationUpgradeResult, ensure_current_schema


@dataclass(frozen=True, slots=True)
class StateBootstrapResult:
    """Resources produced by schema-only startup in the explicitly selected HOME."""

    database: Database
    migration: MigrationUpgradeResult


async def bootstrap_state(
    database_path: Path,
    *,
    backup_dir: Path | None = None,
) -> StateBootstrapResult:
    """Initialize/upgrade this database without scanning or importing legacy files."""
    migration = ensure_current_schema(database_path, backup_dir=backup_dir)
    return StateBootstrapResult(database=Database(database_path), migration=migration)


__all__ = ["StateBootstrapResult", "bootstrap_state"]
