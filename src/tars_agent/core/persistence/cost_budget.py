"""Fail-closed, explicitly initialized budget for the local DeepSeek experiment.

Amounts are integer nano-CNY. Prices are frozen peak CNY prices per million
tokens from https://api-docs.deepseek.com/zh-cn/quick_start/pricing/.
This is a conservative local estimate, not an account invoice.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from tars_agent.core.control import CoreHomeLock
from tars_agent.core.persistence.request_budget import (
    ModelRequestBudgetExceeded,
    RequestLedger,
    validate_existing_request_ledger,
)

NANO_CNY = 1_000_000_000
CAP_NANO_CNY = 50 * NANO_CNY
CONTEXT_TOKEN_BOUND = 1_048_576
MAX_OUTPUT_TOKENS = 8192
_POLICY = {
    "version": 1, "model": "deepseek-flash",
    "base_url": "https://api.deepseek.com/anthropic",
    "cap_nano_cny": CAP_NANO_CNY,
    "context_token_bound": CONTEXT_TOKEN_BOUND,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "input_nano_cny_per_token": 2000,
    "cache_read_nano_cny_per_token": 40,
    "output_nano_cny_per_token": 8000,
}
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class CostBudgetError(ModelRequestBudgetExceeded):
    """No further model request is permitted until the budget is reconciled."""


def _identity(path: Path) -> list[int | str]:
    stat = path.stat()
    return [str(path.resolve(strict=True)), stat.st_dev, stat.st_ino]


def _cny(value: int) -> str:
    return f"{value // NANO_CNY}.{value % NANO_CNY:09d}"


class CostLedger:
    """Opening never creates a ledger. Lost/corrupt state blocks sending."""

    def __init__(self, path: Path) -> None:
        expanded = path.expanduser()
        if not expanded.is_absolute():
            raise CostBudgetError("Cost budget requires an absolute existing ledger")
        self.path = expanded.resolve()
        self.sentinel = self.path.with_name(self.path.name + ".identity.json")
        self.lock_home = self.path.with_name(self.path.name + ".owner")
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(str(self.path), threading.RLock())

    @contextmanager
    def _owned(self) -> Iterator[None]:
        with self._lock:
            owner = CoreHomeLock(self.lock_home)
            expires = time.monotonic() + 5.0
            while True:
                try:
                    owner.acquire()
                    break
                except RuntimeError:
                    if time.monotonic() >= expires:
                        raise CostBudgetError("Cost budget owner lock unavailable") from None
                    time.sleep(0.01)
            try:
                yield
            except (OSError, sqlite3.Error, ValueError, KeyError, TypeError):
                raise CostBudgetError("Cost budget state is missing, changed or invalid") from None
            finally:
                owner.release()

    @classmethod
    def initialize(cls, path: Path, *, request_budget_path: Path) -> CostLedger:
        """Explicit one-time creation; refuses any existing or partial budget."""
        ledger = cls(path)
        with ledger._owned():
            if ledger.path.exists() or ledger.sentinel.exists():
                raise CostBudgetError("Cost budget already exists; open it without initialization")
            request_path = validate_existing_request_ledger(request_budget_path)
            record = {
                "policy": _POLICY, "budget_id": str(uuid4()),
                "request_identity": _identity(request_path),
                "request_high_water": RequestLedger(request_path).counts()["real"],
                "initialized": False,
            }
            ledger.path.parent.mkdir(parents=True, exist_ok=True)
            # An interrupted initializer cannot silently start another 50-CNY budget.
            with ledger.sentinel.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            with closing(sqlite3.connect(ledger.path)) as connection:
                connection.execute("CREATE TABLE metadata(value TEXT NOT NULL)")
                connection.execute("INSERT INTO metadata VALUES (?)", (json.dumps(record),))
                connection.execute(
                    "CREATE TABLE attempts (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, "
                    "step INTEGER NOT NULL, attempt INTEGER NOT NULL, reserved INTEGER NOT NULL, "
                    "confirmed INTEGER, request_ordinal INTEGER, created_at TEXT NOT NULL, "
                    "usage_json TEXT)"
                )
                connection.commit()
                record.update(initialized=True, db_identity=_identity(ledger.path))
                ledger._seal(connection, record)
        return ledger

    @staticmethod
    def _digest(connection: sqlite3.Connection) -> str:
        rows = connection.execute("SELECT * FROM attempts ORDER BY id").fetchall()
        return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()

    def _seal(self, connection: sqlite3.Connection, record: dict[str, Any]) -> None:
        record["digest"] = self._digest(connection)
        temporary = self.sentinel.with_name(self.sentinel.name + ".tmp")
        for attempt in range(6):
            try:
                # A Windows reader can briefly deny replacement even while the
                # budget's owner lock is held. Retry only publication, never the
                # committed monetary transition or its request reservation.
                with temporary.open("w", encoding="utf-8") as stream:
                    json.dump(record, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(self.sentinel)
                return
            except PermissionError:
                if attempt == 5:
                    raise CostBudgetError(
                        "Cost budget seal publication failed; pending state retained"
                    ) from None
                time.sleep(min(0.05 * 2**attempt, 0.2))

    @contextmanager
    def _validated(self) -> Iterator[tuple[sqlite3.Connection, dict[str, Any]]]:
        with self._owned():
            record = json.loads(self.sentinel.read_text(encoding="utf-8"))
            if (record["initialized"] is not True or record["policy"] != _POLICY
                    or record["db_identity"] != _identity(self.path)):
                raise CostBudgetError("Cost budget identity or frozen policy changed")
            request_path = Path(record["request_identity"][0])
            validate_existing_request_ledger(request_path)
            if record["request_identity"] != _identity(request_path):
                raise CostBudgetError("Original request ledger identity changed")
            count = RequestLedger(request_path).counts()["real"]
            if count < record["request_high_water"]:
                raise CostBudgetError("Original request ledger was rolled back")
            with closing(sqlite3.connect(
                self.path.as_uri() + "?mode=rw", uri=True, timeout=5.0,
            )) as connection:
                metadata = json.loads(
                    connection.execute("SELECT value FROM metadata").fetchone()[0]
                )
                if (metadata["budget_id"] != record["budget_id"]
                        or metadata["policy"] != _POLICY
                        or metadata["request_identity"] != record["request_identity"]
                        or record["digest"] != self._digest(connection)):
                    raise CostBudgetError("Cost budget content changed or was rolled back")
                yield connection, record

    @staticmethod
    def _totals(connection: sqlite3.Connection) -> tuple[int, int, int]:
        confirmed, pending, count = connection.execute(
            "SELECT COALESCE(SUM(confirmed), 0), "
            "COALESCE(SUM(CASE WHEN confirmed IS NULL THEN reserved ELSE 0 END), 0), "
            "COUNT(*) FROM attempts"
        ).fetchone()
        return int(confirmed), int(pending), int(count)

    def reserve_attempt(
        self, request_ledger: RequestLedger, *, max_tokens: int,
        run_id: str, step: int, attempt: int,
    ) -> str:
        if type(max_tokens) is not int or not 0 < max_tokens <= MAX_OUTPUT_TOKENS:
            raise CostBudgetError("Output budget exceeds the frozen cost policy")
        reserved = CONTEXT_TOKEN_BOUND * 2000 + max_tokens * 8000
        with self._validated() as (connection, record):
            if _identity(request_ledger.path) != record["request_identity"]:
                raise CostBudgetError("Cost budget must use the original request ledger")
            confirmed, pending, _ = self._totals(connection)
            if confirmed + pending + reserved > CAP_NANO_CNY:
                raise CostBudgetError("50-CNY model cost budget exhausted; no request was sent")
            reservation = str(uuid4())
            connection.execute(
                "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL)",
                (reservation, run_id, step, attempt, reserved, datetime.now(UTC).isoformat()),
            )
            connection.commit()
            self._seal(connection, record)
            # Both reservations precede sending. A crash between stores keeps the
            # entire cost reserve, never granting another free attempt on resume.
            ordinal = request_ledger.reserve("real")
            connection.execute(
                "UPDATE attempts SET request_ordinal = ? WHERE id = ?", (ordinal, reservation),
            )
            connection.commit()
            record["request_high_water"] = ordinal
            self._seal(connection, record)
            return reservation

    def settle(
        self, reservation: str, *, model: str | None, input_tokens: int, output_tokens: int,
        cache_read_input_tokens: int, cache_creation_input_tokens: int,
    ) -> None:
        values = [input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens]
        if model != _POLICY["model"] or any(type(x) is not int or x < 0 for x in values):
            raise CostBudgetError("Cannot confirm cost without permitted model and valid usage")
        amount = ((input_tokens + cache_creation_input_tokens) * 2000
                  + cache_read_input_tokens * 40 + output_tokens * 8000)
        usage = json.dumps(values)
        with self._validated() as (connection, record):
            row = connection.execute(
                "SELECT reserved, confirmed, request_ordinal, usage_json "
                "FROM attempts WHERE id = ?",
                (reservation,),
            ).fetchone()
            if row is None or row[2] is None or amount > row[0]:
                raise CostBudgetError("Usage cannot be reconciled with its reserved attempt")
            if row[1] is not None:
                if row[1] == amount and row[3] == usage:
                    return
                raise CostBudgetError("Previously settled usage cannot be changed")
            connection.execute(
                "UPDATE attempts SET confirmed = ?, usage_json = ? WHERE id = ?",
                (amount, usage, reservation),
            )
            connection.commit()
            self._seal(connection, record)

    def summary(self) -> dict[str, Any]:
        with self._validated() as (connection, record):
            confirmed, pending, count = self._totals(connection)
            remaining = CAP_NANO_CNY - confirmed - pending
            return {
                "budget_id": record["budget_id"], "policy": dict(_POLICY),
                "cap_nano_cny": CAP_NANO_CNY, "confirmed_nano_cny": confirmed,
                "unknown_reserved_nano_cny": pending, "remaining_nano_cny": remaining,
                "confirmed_cny": _cny(confirmed), "unknown_reserved_cny": _cny(pending),
                "remaining_cny": _cny(remaining), "attempt_count": count,
            }
