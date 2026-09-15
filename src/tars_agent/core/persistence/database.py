from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from tars_agent.core.persistence.models import Base

BUSY_TIMEOUT_MS = 5_000


class Database:
    """Own the async SQLite engine and transaction boundaries for runtime state."""

    def __init__(self, path: Path, *, echo: bool = False) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        url = URL.create("sqlite+aiosqlite", database=str(self.path))
        self.engine = create_async_engine(url, echo=echo)
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._configure_sqlite(self.engine)

    @staticmethod
    def _configure_sqlite(engine: AsyncEngine) -> None:
        @event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragmas(dbapi_connection: object, _connection_record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
                cursor.execute("PRAGMA journal_mode=WAL")
            finally:
                cursor.close()

    async def create_schema(self) -> None:
        """Create tables for tests/bootstrap; release startup should run Alembic instead."""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory.begin() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()

    async def __aenter__(self) -> Database:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.dispose()
