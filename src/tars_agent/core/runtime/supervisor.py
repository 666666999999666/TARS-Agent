from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

log = logging.getLogger(__name__)

RunFactory = Callable[[], Coroutine[Any, Any, None]]


class RunSupervisor:
    """Own every active main/chat run task for deterministic lifecycle control."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False
        self._cancelling: set[str] = set()

    def start(self, run_id: str, factory: RunFactory) -> None:
        if self._closed:
            raise RuntimeError("run supervisor is closed")
        if run_id in self._tasks:
            raise RuntimeError(f"run is already supervised: {run_id}")
        task: asyncio.Task[None] = asyncio.create_task(
            factory(), name=f"tars-run:{run_id}"
        )
        self._tasks[run_id] = task
        task.add_done_callback(lambda completed: self._finish(run_id, completed))

    def _finish(self, run_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(run_id, None)
        self._cancelling.discard(run_id)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            log.error(
                "supervised run escaped with an error run_id=%s",
                run_id,
                exc_info=(type(error), error, error.__traceback__),
            )

    def cancel(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        if run_id not in self._cancelling:
            self._cancelling.add(run_id)
            task.cancel()
        return True

    def is_active(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    def active_run_ids(self) -> tuple[str, ...]:
        return tuple(run_id for run_id, task in self._tasks.items() if not task.done())

    async def wait(self, run_id: str) -> None:
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.shield(task)

    async def shutdown(self) -> None:
        self._closed = True
        tasks = tuple(self._tasks.values())
        for run_id in tuple(self._tasks):
            self.cancel(run_id)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()


__all__ = ["RunFactory", "RunSupervisor"]
