from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from tars_agent.core.persistence import request_budget
from tars_agent.core.persistence.request_budget import (
    RequestLedger,
    validate_existing_request_ledger,
)


def test_existing_ledger_validation_is_read_only_and_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "existing ledger #1.sqlite3"
    ledger = RequestLedger(path)
    ledger.reserve("real")
    ledger.reserve("probe")
    original_bytes = path.read_bytes()
    original_files = set(tmp_path.iterdir())
    connect = sqlite3.connect
    calls = []

    def tracked_connect(database, **kwargs):
        connection = connect(database, **kwargs)
        calls.append((database, kwargs, connection))
        return connection

    monkeypatch.setattr(request_budget.sqlite3, "connect", tracked_connect)
    assert validate_existing_request_ledger(tmp_path / "." / path.name) == path.resolve()
    assert calls[0][:2] == (path.as_uri() + "?mode=ro", {"uri": True})
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        calls[0][2].execute("SELECT 1")
    assert path.read_bytes() == original_bytes
    assert set(tmp_path.iterdir()) == original_files


def test_existing_ledger_allows_home_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ledger.sqlite3"
    RequestLedger(path).counts()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert validate_existing_request_ledger(Path("~/ledger.sqlite3")) == path.resolve()


@pytest.mark.parametrize("kind", ["relative", "missing", "missing_parent", "directory"])
def test_invalid_path_never_opens_or_creates_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    existing = tmp_path / "private-ledger.sqlite3"
    RequestLedger(existing).reserve()
    monkeypatch.chdir(tmp_path)
    paths = {
        "relative": Path(existing.name), "missing": tmp_path / "missing.sqlite3",
        "missing_parent": tmp_path / "absent" / "ledger.sqlite3", "directory": tmp_path,
    }
    before = set(tmp_path.rglob("*"))

    def must_not_connect(*args, **kwargs):
        pytest.fail("Invalid paths must be rejected before opening SQLite")

    monkeypatch.setattr(request_budget.sqlite3, "connect", must_not_connect)
    with pytest.raises(ValueError, match="existing absolute") as captured:
        validate_existing_request_ledger(paths[kind])
    assert "private-ledger" not in str(captured.value)
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("schema", [
    None,
    "CREATE TABLE unrelated(id INTEGER)",
    "CREATE TABLE requests(id INTEGER PRIMARY KEY, kind TEXT NOT NULL)",
    "CREATE TABLE requests(id TEXT PRIMARY KEY, kind TEXT NOT NULL, reserved_at TEXT NOT NULL)",
    "CREATE TABLE requests(id INTEGER PRIMARY KEY, kind TEXT, reserved_at TEXT)",
    "CREATE VIEW requests AS SELECT 1 AS id, 'real' AS kind, 'now' AS reserved_at",
])
def test_fake_ledger_rejected_without_initializing_or_changing_it(tmp_path: Path, schema: str | None) -> None:
    path = tmp_path / "fake.sqlite3"
    if schema is None:
        path.write_bytes(b"not a SQLite database")
    else:
        with closing(sqlite3.connect(path)) as connection:
            connection.execute(schema)
            connection.commit()
    before = path.read_bytes()
    files = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="valid request ledger"):
        validate_existing_request_ledger(path)
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == files


def test_rejected_schema_also_closes_read_only_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "empty.sqlite3"
    path.touch()
    connect = sqlite3.connect
    opened = []

    def track(database, **kwargs):
        connection = connect(database, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(request_budget.sqlite3, "connect", track)
    with pytest.raises(ValueError):
        validate_existing_request_ledger(path)
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    assert path.read_bytes() == b""
