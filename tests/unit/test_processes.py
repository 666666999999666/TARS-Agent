from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import AsyncMock

import pytest

from tars_agent.core import processes


def test_subprocess_group_kwargs_are_platform_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(processes, "_IS_WINDOWS", False)
    assert processes.subprocess_group_kwargs() == {"start_new_session": True}

    monkeypatch.setattr(processes, "_IS_WINDOWS", True)
    assert "creationflags" in processes.subprocess_group_kwargs()


async def test_windows_tree_cleanup_invokes_taskkill_with_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RootProcess:
        pid = 4242
        returncode: int | None = None
        killed = False

        def terminate(self) -> None:
            raise AssertionError("Windows tree cleanup should use taskkill")

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    taskkill = AsyncMock()
    taskkill.wait = AsyncMock(return_value=0)
    taskkill.kill = AsyncMock()
    create = AsyncMock(return_value=taskkill)
    monkeypatch.setattr(processes.asyncio, "create_subprocess_exec", create)

    root = _RootProcess()
    await processes._terminate_windows_tree(root, grace_s=0.01)

    args = create.await_args.args
    assert args[1:] == ("/PID", "4242", "/T", "/F")
    assert not root.killed


async def test_windows_tree_cleanup_falls_back_to_root_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RootProcess:
        pid = 4242
        returncode: int | None = None

        def __init__(self) -> None:
            self.exited = asyncio.Event()
            self.killed = False

        def terminate(self) -> None:
            raise AssertionError("Windows fallback is forced cleanup")

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9
            self.exited.set()

        async def wait(self) -> int:
            await self.exited.wait()
            return -9

    monkeypatch.setattr(
        processes.asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=OSError("taskkill unavailable")),
    )

    root = _RootProcess()
    await processes._terminate_windows_tree(root, grace_s=0.001)

    assert root.killed


async def test_posix_tree_cleanup_signals_the_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RootProcess:
        pid = 5151
        returncode: int | None = None

        def __init__(self) -> None:
            self.exited = asyncio.Event()

        def terminate(self) -> None:
            raise AssertionError("process-group signal should succeed")

        def kill(self) -> None:
            raise AssertionError("process-group signal should succeed")

        async def wait(self) -> int:
            await self.exited.wait()
            return self.returncode or 0

    root = _RootProcess()
    seen: list[tuple[int, int]] = []

    def fake_killpg(pid: int, sig: int) -> None:
        seen.append((pid, sig))
        if sig == processes._SIGKILL:
            root.returncode = -9
            root.exited.set()

    monkeypatch.setattr(processes.os, "killpg", fake_killpg, raising=False)

    await processes._terminate_posix_tree(root, grace_s=0.05)

    assert seen == [(5151, processes._SIGTERM), (5151, processes._SIGKILL)]


async def test_tree_cleanup_finishes_before_propagating_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def fake_cleanup(
        process: object,
        *,
        grace_s: float,
        deadline: processes.CleanupDeadline,
    ) -> None:
        del process, grace_s, deadline
        started.set()
        await release.wait()
        finished.set()

    monkeypatch.setattr(processes, "_terminate_process_tree", fake_cleanup)
    task = asyncio.create_task(processes.terminate_process_tree(object()))  # type: ignore[arg-type]
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_posix_cleanup_bounds_stuck_root_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StuckRoot:
        pid = 6161
        returncode: int | None = None

        def __init__(self) -> None:
            self.killed = False

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            await asyncio.Event().wait()
            return 0

    monkeypatch.setattr(processes.os, "killpg", lambda pid, sig: None, raising=False)
    root = _StuckRoot()

    await asyncio.wait_for(
        processes._terminate_posix_tree(root, grace_s=0.002),
        timeout=0.1,
    )

    assert root.killed


async def test_windows_cleanup_bounds_stuck_taskkill_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Taskkill:
        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            await asyncio.Event().wait()
            return 0

    class _Root:
        pid = 7171
        returncode: int | None = None

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    taskkill = _Taskkill()
    monkeypatch.setattr(
        processes.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=taskkill),
    )

    await asyncio.wait_for(
        processes._terminate_windows_tree(_Root(), grace_s=0.002),
        timeout=0.1,
    )

    assert taskkill.killed


async def test_windows_cleanup_bounds_stuck_root_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Taskkill:
        def kill(self) -> None:
            raise AssertionError("completed taskkill must not be killed")

        async def wait(self) -> int:
            return 0

    class _StuckRoot:
        pid = 8181
        returncode: int | None = None

        def __init__(self) -> None:
            self.killed = False

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            await asyncio.Event().wait()
            return 0

    monkeypatch.setattr(
        processes.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=_Taskkill()),
    )
    root = _StuckRoot()

    await asyncio.wait_for(
        processes._terminate_windows_tree(root, grace_s=0.002),
        timeout=0.1,
    )

    assert root.killed


def test_sync_posix_cleanup_signals_group_and_uses_bounded_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RootProcess:
        pid = 9191
        returncode: int | None = None

        def __init__(self) -> None:
            self.wait_timeouts: list[float | None] = []

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            raise AssertionError("process-group signal should succeed")

        def kill(self) -> None:
            raise AssertionError("process-group signal should succeed")

        def wait(self, timeout: float | None = None) -> int:
            self.wait_timeouts.append(timeout)
            if len(self.wait_timeouts) == 1:
                raise subprocess.TimeoutExpired("root", timeout)
            self.returncode = -9
            return -9

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        processes.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
        raising=False,
    )
    root = _RootProcess()

    processes._terminate_posix_popen_tree(
        root,
        grace_s=0.01,
        deadline=processes.CleanupDeadline.after(0.05),
    )

    assert signals == [(9191, processes._SIGTERM), (9191, processes._SIGKILL)]
    assert all(timeout is not None and 0 < timeout <= 0.01 for timeout in root.wait_timeouts)


def test_sync_windows_cleanup_uses_trusted_taskkill_and_bounds_stuck_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StuckProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.returncode: int | None = None
            self.killed = False
            self.wait_timeouts: list[float | None] = []

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            raise AssertionError("Windows cleanup should use taskkill")

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            self.wait_timeouts.append(timeout)
            raise subprocess.TimeoutExpired(str(self.pid), timeout)

    root = _StuckProcess(9292)
    taskkill = _StuckProcess(9393)
    created: list[tuple[object, ...]] = []

    def fake_popen(*args: object, **kwargs: object) -> _StuckProcess:
        del kwargs
        created.append(args)
        return taskkill

    monkeypatch.setenv("SystemRoot", r"C:\TrustedWindows")
    monkeypatch.setattr(processes.subprocess, "Popen", fake_popen)

    processes._terminate_windows_popen_tree(
        root,
        grace_s=0.002,
        deadline=processes.CleanupDeadline.after(0.01),
    )

    command = created[0][0]
    assert isinstance(command, list)
    assert command == [
        str(processes.Path(r"C:\TrustedWindows") / "System32" / "taskkill.exe"),
        "/PID",
        "9292",
        "/T",
        "/F",
    ]
    assert taskkill.killed
    assert root.killed
    assert all(timeout is not None and timeout > 0 for timeout in taskkill.wait_timeouts)
    assert all(timeout is not None and timeout > 0 for timeout in root.wait_timeouts)
