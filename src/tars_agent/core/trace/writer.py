from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any, TextIO

from tars_agent.core.trace.record import TraceRecord

log = logging.getLogger(__name__)
_SENSITIVE_KEYS = {
    "token", "api_key", "anthropic_api_key", "authorization", "proxy_authorization",
    "password", "secret", "client_secret", "access_token", "refresh_token", "shutdown_token",
    "x_api_key",
}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).lower().replace("-", "_") in _SENSITIVE_KEYS
            else _redact(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class TraceSecurityError(ValueError):
    pass


class TraceWriter:
    """Bounded diagnostic output; storage failure cannot interrupt the event bus."""

    def __init__(
        self, path: Path, *, queue_size: int = 256, stop_timeout: float = 2.0,
        max_record_bytes: int = 256 * 1024,
    ) -> None:
        if queue_size <= 0 or stop_timeout <= 0 or max_record_bytes <= 0:
            raise ValueError("Trace limits must be positive")
        self._path = path.expanduser().absolute()
        self._queue: queue.Queue[TraceRecord] = queue.Queue(maxsize=queue_size)
        self._stop_timeout = stop_timeout
        self._max_record_bytes = max_record_bytes
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._failure: BaseException | None = None
        self._accepted = 0
        self._written = 0
        self._dropped = 0

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {"accepted": self._accepted, "written": self._written,
                    "dropped": self._dropped, "failed": self._failure is not None}

    async def start(self) -> None:
        if self._thread is not None or self._failure is not None:
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._drain, name="tars-trace", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._stopping.set()
        thread = self._thread
        if thread is not None:
            await asyncio.to_thread(thread.join, self._stop_timeout)
            if thread.is_alive():
                self._failure = TimeoutError("trace writer did not stop before deadline")
                log.error("Trace writer stop timed out; diagnostics disabled")
            else:
                self._thread = None

    def emit(self, record: TraceRecord) -> None:
        with self._lock:
            if self._failure is not None or self._stopping.is_set():
                self._dropped += 1
                return
            try:
                self._queue.put_nowait(record)
                self._accepted += 1
            except queue.Full:
                self._dropped += 1
                if self._dropped == 1:
                    log.warning("Trace queue full; record dropped, business processing continues")

    def _reject_links(self) -> None:
        for part in (self._path, *self._path.parents):
            if part.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(part)):
                raise TraceSecurityError("Trace path must not contain a symbolic link or junction")

    def _open(self) -> TextIO:
        self._reject_links()
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._reject_links()
        descriptor = os.open(
            self._path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            return os.fdopen(descriptor, "a", encoding="utf-8", errors="replace")
        except BaseException:
            os.close(descriptor)
            raise

    def _drain(self) -> None:
        try:
            with self._open() as handle:
                while not self._stopping.is_set() or not self._queue.empty():
                    try:
                        record = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    try:
                        line = json.dumps(
                            _redact(record.model_dump(mode="json")), ensure_ascii=False,
                        ) + "\n"
                        if len(line.encode("utf-8")) > self._max_record_bytes:
                            with self._lock:
                                self._dropped += 1
                            continue
                        handle.write(line)
                        handle.flush()
                        with self._lock:
                            self._written += 1
                    finally:
                        self._queue.task_done()
        except Exception as exc:
            self._failure = exc
            log.error("Trace storage failed; diagnostics disabled (%s)", type(exc).__name__)
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()
                with self._lock:
                    self._dropped += 1
