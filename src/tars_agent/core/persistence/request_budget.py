from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

REQUEST_LIMIT = 100
RequestKind = Literal["real", "probe"]


class ModelRequestBudgetExceeded(RuntimeError):
    """The authorized real-request allowance has been exhausted."""


class RequestLedger:
    """A separate safety counter, never a source of session or run state."""

    def __init__(self, path: Path, *, limit: int | None = REQUEST_LIMIT) -> None:
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("Request limit must be a positive integer or None")
        self.path = path.expanduser().resolve()
        self.limit = limit

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS requests ("
            "id INTEGER PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('real','probe')), "
            "reserved_at TEXT NOT NULL)"
        )
        return connection

    def reserve(self, kind: RequestKind = "real") -> int:
        # Commit before the transport can send. Never refund an uncertain network result.
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            count = int(connection.execute(
                "SELECT count(*) FROM requests WHERE kind = ?", (kind,)
            ).fetchone()[0])
            if kind == "real" and self.limit is not None and count >= self.limit:
                raise ModelRequestBudgetExceeded(
                    f"Real model request budget exhausted ({self.limit}); no request was sent"
                )
            connection.execute(
                "INSERT INTO requests(kind, reserved_at) VALUES (?, ?)",
                (kind, datetime.now(UTC).isoformat()),
            )
            connection.commit()
            return count + 1
        finally:
            connection.close()

    def counts(self) -> dict[str, int]:
        connection = self._connect()
        try:
            result = {"real": 0, "probe": 0}
            result.update({str(kind): int(count) for kind, count in connection.execute(
                "SELECT kind, count(*) FROM requests GROUP BY kind"
            )})
            return result
        finally:
            connection.close()
