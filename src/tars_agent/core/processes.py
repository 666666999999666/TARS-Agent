from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

log = logging.getLogger(__name__)

_IS_WINDOWS = os.name == "nt"
_DEFAULT_GRACE_S = 5.0
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
_SIGTERM = int(getattr(signal, "SIGTERM", 15))
_SIGKILL = int(getattr(signal, "SIGKILL", 9))


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class CleanupDeadline:
    """One absolute deadline shared by every phase of a cleanup operation."""

    expires_at: float

    @classmethod
    def after(cls, timeout_s: float) -> CleanupDeadline:
        if timeout_s <= 0:
            raise ValueError("cleanup timeout must be positive")
        return cls(time.monotonic() + timeout_s)

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    async def wait(
        self,
        task: asyncio.Task[_T],
        *,
        max_wait_s: float | None = None,
    ) -> _T:
        """Wait for ``task`` without letting task cancellation defeat the deadline."""

        if task.done():
            return task.result()
        timeout_s = self.remaining()
        if max_wait_s is not None:
            timeout_s = min(timeout_s, max(0.0, max_wait_s))
        if timeout_s <= 0:
            raise TimeoutError
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
        if task not in done:
            raise TimeoutError
        return task.result()


def cancel_stuck_task(task: asyncio.Task[Any], *, label: str) -> None:
    """Cancel a task and consume its eventual result without awaiting forever."""

    task.cancel()

    def consume_result(done: asyncio.Task[Any]) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.debug("%s failed after cleanup deadline", label, exc_info=True)

    task.add_done_callback(consume_result)


async def finish_cleanup(
    task: asyncio.Task[Any],
    *,
    failure_message: str,
) -> None:
    """Finish a bounded cleanup task before propagating repeated cancellation."""

    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()

    try:
        task.result()
    except Exception:
        if cancellation is None:
            raise
        log.warning(failure_message, exc_info=True)

    if cancellation is not None:
        raise cancellation


class SignalableProcess(Protocol):
    @property
    def pid(self) -> int: ...

    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class ManagedProcess(SignalableProcess, Protocol):

    async def wait(self) -> int: ...


def subprocess_group_kwargs() -> dict[str, Any]:
    """Return platform options that place a child in its own process group."""

    if _IS_WINDOWS:
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


class ManagedPopen(SignalableProcess, Protocol):
    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


def _taskkill_path() -> Path:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    return Path(system_root) / "System32" / "taskkill.exe"


def _wait_popen_until(
    proc: ManagedPopen,
    deadline: CleanupDeadline,
    *,
    max_wait_s: float | None = None,
) -> bool:
    if proc.poll() is not None:
        return True
    timeout_s = deadline.remaining()
    if max_wait_s is not None:
        timeout_s = min(timeout_s, max(0.0, max_wait_s))
    if timeout_s <= 0:
        return proc.poll() is not None
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return proc.poll() is not None
    except OSError:
        log.debug("synchronous process wait failed", exc_info=True)
        return proc.poll() is not None
    return True


def terminate_popen_process_tree(
    proc: subprocess.Popen[bytes],
    *,
    grace_s: float = _DEFAULT_GRACE_S,
    deadline: CleanupDeadline | None = None,
) -> None:
    """Synchronously terminate a grouped ``Popen`` tree within one deadline."""

    if grace_s <= 0:
        raise ValueError("process cleanup grace period must be positive")
    cleanup_deadline = deadline or CleanupDeadline.after(grace_s * 4)
    if _IS_WINDOWS:
        _terminate_windows_popen_tree(
            proc,
            grace_s=grace_s,
            deadline=cleanup_deadline,
        )
    else:
        _terminate_posix_popen_tree(
            proc,
            grace_s=grace_s,
            deadline=cleanup_deadline,
        )


def _terminate_posix_popen_tree(
    proc: ManagedPopen,
    *,
    grace_s: float,
    deadline: CleanupDeadline,
) -> None:
    _signal_posix_group(proc, _SIGTERM)
    _wait_popen_until(proc, deadline, max_wait_s=grace_s)

    _signal_posix_group(proc, _SIGKILL, fallback_root=proc.poll() is None)
    if _wait_popen_until(proc, deadline, max_wait_s=grace_s):
        return
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    if not _wait_popen_until(proc, deadline):
        log.error("process %s did not exit before cleanup deadline", proc.pid)


def _terminate_windows_popen_tree(
    proc: ManagedPopen,
    *,
    grace_s: float,
    deadline: CleanupDeadline,
) -> None:
    if proc.poll() is not None:
        return

    try:
        taskkill = subprocess.Popen(
            [str(_taskkill_path()), "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not _wait_popen_until(taskkill, deadline, max_wait_s=grace_s):
            try:
                taskkill.kill()
            except ProcessLookupError:
                pass
            if not _wait_popen_until(taskkill, deadline, max_wait_s=grace_s):
                log.error("taskkill did not exit before cleanup deadline")
    except OSError:
        log.debug("taskkill process-tree cleanup failed", exc_info=True)

    if _wait_popen_until(proc, deadline, max_wait_s=grace_s):
        return
    if proc.poll() is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    if not _wait_popen_until(proc, deadline):
        log.error("process %s did not exit before cleanup deadline", proc.pid)


async def terminate_process_tree(
    proc: ManagedProcess,
    *,
    grace_s: float = _DEFAULT_GRACE_S,
    deadline: CleanupDeadline | None = None,
) -> None:
    """Terminate a spawned process tree and wait for the root process to exit.

    POSIX children are isolated in a new session and signalled as a process group.
    Windows uses ``taskkill /T /F`` as a best-effort tree cleanup; a Job Object is
    still required to contain descendants that detach or outlive the root.
    """

    if grace_s <= 0:
        raise ValueError("process cleanup grace period must be positive")
    cleanup_deadline = deadline or CleanupDeadline.after(grace_s * 4)
    cleanup = asyncio.create_task(
        _terminate_process_tree(
            proc,
            grace_s=grace_s,
            deadline=cleanup_deadline,
        )
    )
    await finish_cleanup(
        cleanup,
        failure_message="process-tree cleanup failed during cancellation",
    )


async def _terminate_process_tree(
    proc: ManagedProcess,
    *,
    grace_s: float,
    deadline: CleanupDeadline,
) -> None:
    if _IS_WINDOWS:
        await _terminate_windows_tree(proc, grace_s=grace_s, deadline=deadline)
    else:
        await _terminate_posix_tree(proc, grace_s=grace_s, deadline=deadline)


async def _terminate_posix_tree(
    proc: ManagedProcess,
    *,
    grace_s: float,
    deadline: CleanupDeadline | None = None,
) -> None:
    cleanup_deadline = deadline or CleanupDeadline.after(grace_s * 4)
    wait_task = asyncio.create_task(proc.wait())
    _signal_posix_group(proc, _SIGTERM)
    try:
        await cleanup_deadline.wait(wait_task, max_wait_s=grace_s)
    except TimeoutError:
        pass

    # The root can exit before one of its descendants.  Always check the original
    # process group once more and force any remaining members down.
    _signal_posix_group(proc, _SIGKILL, fallback_root=not wait_task.done())
    if not wait_task.done():
        try:
            await cleanup_deadline.wait(wait_task, max_wait_s=grace_s)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await cleanup_deadline.wait(wait_task)
            except TimeoutError:
                log.error("process %s did not exit before cleanup deadline", proc.pid)
                cancel_stuck_task(wait_task, label=f"process {proc.pid} wait")
    else:
        await cleanup_deadline.wait(wait_task)


def _signal_posix_group(
    proc: SignalableProcess,
    sig: int,
    *,
    fallback_root: bool = True,
) -> None:
    killpg = cast(
        Callable[[int, int], None] | None,
        getattr(os, "killpg", None),
    )
    try:
        if killpg is not None:
            killpg(proc.pid, sig)
            return
    except ProcessLookupError:
        pass
    except OSError:
        log.debug("failed to signal process group %s", proc.pid, exc_info=True)

    if not fallback_root or proc.returncode is not None:
        return
    try:
        if sig == _SIGTERM:
            proc.terminate()
        else:
            proc.kill()
    except ProcessLookupError:
        pass


async def _terminate_windows_tree(
    proc: ManagedProcess,
    *,
    grace_s: float,
    deadline: CleanupDeadline | None = None,
) -> None:
    if proc.returncode is not None:
        return

    cleanup_deadline = deadline or CleanupDeadline.after(grace_s * 4)
    try:
        taskkill = await asyncio.create_subprocess_exec(
            str(_taskkill_path()),
            "/PID",
            str(proc.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        taskkill_wait = asyncio.create_task(taskkill.wait())
        try:
            await cleanup_deadline.wait(taskkill_wait, max_wait_s=grace_s)
        except TimeoutError:
            taskkill.kill()
            try:
                await cleanup_deadline.wait(taskkill_wait, max_wait_s=grace_s)
            except TimeoutError:
                log.error("taskkill did not exit before cleanup deadline")
                cancel_stuck_task(taskkill_wait, label="taskkill wait")
    except OSError:
        log.debug("taskkill process-tree cleanup failed", exc_info=True)

    wait_task = asyncio.create_task(proc.wait())
    try:
        await cleanup_deadline.wait(wait_task, max_wait_s=grace_s)
    except TimeoutError:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await cleanup_deadline.wait(wait_task)
        except TimeoutError:
            log.error("process %s did not exit before cleanup deadline", proc.pid)
            cancel_stuck_task(wait_task, label=f"process {proc.pid} wait")
