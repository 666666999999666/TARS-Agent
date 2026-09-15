from __future__ import annotations

import json
from pathlib import Path

import pytest

from tars_agent.core.persistence import (
    CURRENT_SCHEMA_REVISION,
    EVENT_SCHEMA_VERSION,
    SessionRecord,
    bootstrap_state,
    read_schema_revision,
)


@pytest.mark.asyncio
async def test_bootstrap_upgrades_without_reading_or_importing_legacy_state(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    legacy_session = sessions_root / "sess-1"
    legacy_session.mkdir(parents=True)
    (legacy_session / "meta.json").write_text(
        json.dumps(
            {
                "id": "sess-1",
                "mode": "chat",
                "status": "waiting_for_input",
                "title": "imported",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "run_ids": [],
            }
        ),
        encoding="utf-8",
    )
    database_path = tmp_path / "state.db"

    sentinel = (legacy_session / "meta.json").read_bytes()
    result = await bootstrap_state(database_path)
    try:
        assert result.migration.final_revision == CURRENT_SCHEMA_REVISION
        assert read_schema_revision(database_path) == CURRENT_SCHEMA_REVISION
        async with result.database.session() as session:
            record = await session.get(SessionRecord, "sess-1")
        assert record is None
        assert (legacy_session / "meta.json").read_bytes() == sentinel
        assert EVENT_SCHEMA_VERSION == 1
    finally:
        await result.database.dispose()

    second = await bootstrap_state(database_path)
    try:
        assert second.migration.backup_path is None
        async with second.database.session() as session:
            assert await session.get(SessionRecord, "sess-1") is None
    finally:
        await second.database.dispose()
