from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tars_agent.cli.commands import core
from tars_agent.core.bus.commands import PongResult
from tars_agent.core.config import TarsConfig
from tars_agent.core.control import DaemonControl


async def test_ping_check_uses_core_ping_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncMock()
    client.send_command.return_value = {
        "server_version": "0.0.1",
        "uptime_ms": 12,
        "schema_revision": "0003",
        "received_at": "2026-08-22T00:00:00+00:00",
    }
    monkeypatch.setattr(core, "SocketClient", lambda host, port: client)

    await core._ping_check(TarsConfig(host="127.0.0.1", port=7437))

    client.connect.assert_awaited_once()
    client.send_command.assert_awaited_once_with(
        "core.ping",
        {"client": "core-launcher"},
        timeout_s=2.0,
    )
    client.close.assert_awaited_once()


async def test_ping_check_rejects_non_tars_tcp_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncMock()
    client.send_command.return_value = {"message": "not-tars"}
    monkeypatch.setattr(core, "SocketClient", lambda host, port: client)

    with pytest.raises(core.ValidationError):
        await core._ping_check(TarsConfig())

    client.close.assert_awaited_once()




class _ClosedWriter:
    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


# 功能：验证停止等待会持续轮询，直到 daemon 端口真正关闭
# 设计：首轮返回可关闭 writer、次轮拒绝连接，覆盖已接收关闭请求但尚未退出的窗口
async def test_wait_until_stopped_observes_closed_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def fake_open_connection(host: str, port: int) -> tuple[object, _ClosedWriter]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return object(), _ClosedWriter()
        raise ConnectionRefusedError

    monkeypatch.setattr(core.asyncio, "open_connection", fake_open_connection)
    control = DaemonControl(pid=42, host="127.0.0.1", port=7437, token="token")

    assert await core._wait_until_stopped(control, timeout_s=0.2)
    assert attempts == 2


# 功能：验证 daemon 未在期限内停止时 core stop 以非零状态退出
# 设计：替换控制文件、关闭 RPC 与等待函数，只隔离断言超时分支和错误输出
def test_core_stop_times_out_with_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    control = DaemonControl(pid=42, host="127.0.0.1", port=7437, token="token")
    shutdown_called = False

    async def fake_shutdown(value: DaemonControl) -> None:
        nonlocal shutdown_called
        shutdown_called = value is control

    async def fake_wait(value: DaemonControl, timeout_s: float = 5.0) -> bool:
        assert value is control
        assert timeout_s == 5.0
        return False

    monkeypatch.setattr(core, "control_file_for", lambda port: tmp_path / f"{port}.json")
    monkeypatch.setattr(core, "read_control_file", lambda path: control)
    monkeypatch.setattr(core, "_shutdown_daemon", fake_shutdown)
    monkeypatch.setattr(core, "_wait_until_stopped", fake_wait)

    with pytest.raises(SystemExit) as exc_info:
        core.cmd_core_stop(TarsConfig())

    assert exc_info.value.code == 1
    assert shutdown_called
    assert "did not stop within timeout" in capsys.readouterr().err


# 功能：验证 daemon 启动未就绪时 core start 终止子进程并以非零状态退出
# 设计：用可观测假进程模拟持续运行，确保优先 terminate 且不误报 started
def test_core_start_failure_returns_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeProcess:
        pid = 99

        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return 1

        def kill(self) -> None:
            raise AssertionError("terminate should be sufficient")

    process = _FakeProcess()
    spawned_env: dict[str, str] | None = None
    spawned_kwargs: dict[str, Any] | None = None

    async def refused(config: TarsConfig) -> None:
        raise ConnectionRefusedError

    async def never_ready(
        config: TarsConfig,
        launch_id: str,
        timeout_s: float = 45.0,
    ) -> None:
        assert launch_id == "test-launch"
        return None

    async def cleanup_failed_start(
        proc: Any,
        config: TarsConfig,
        launch_id: str,
    ) -> None:
        proc.terminate()
        proc.wait(timeout=2.0)

    monkeypatch.setattr(core, "_ping_check", refused)
    monkeypatch.setattr(core.secrets, "token_urlsafe", lambda length: "test-launch")
    monkeypatch.setattr(core, "_wait_until_ready", never_ready)
    monkeypatch.setattr(core, "_cleanup_failed_start", cleanup_failed_start)
    monkeypatch.setattr(core, "control_file_for", lambda port: tmp_path / f"{port}.json")
    monkeypatch.setattr(core, "read_control_file", lambda path: None)

    def fake_popen(*args: Any, **kwargs: Any) -> _FakeProcess:
        nonlocal spawned_env, spawned_kwargs
        spawned_env = kwargs["env"]
        spawned_kwargs = kwargs
        return process

    monkeypatch.setattr(core.subprocess, "Popen", fake_popen)

    with pytest.raises(SystemExit) as exc_info:
        core.cmd_core_start(TarsConfig())

    assert exc_info.value.code == 1
    assert process.terminated
    assert spawned_env is not None
    assert spawned_env[core.CORE_LAUNCH_ID_ENV] == "test-launch"
    assert spawned_kwargs is not None
    assert (
        "creationflags" in spawned_kwargs
        or spawned_kwargs.get("start_new_session") is True
    )
    assert "failed to become ready" in capsys.readouterr().err


# 功能：验证 readiness 依赖 launch_id 与端点，不依赖短生命周期 launcher 的 PID
# 设计：控制文件中的 daemon PID 与 launcher 无关，匹配 nonce 后通过 ping 即视为就绪
async def test_wait_until_ready_accepts_matching_launch_not_launcher_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TarsConfig()
    control = DaemonControl(
        pid=202,
        host=config.host,
        port=config.port,
        token="token",
        launch_id="expected-launch",
    )
    pinged = False

    async def successful_ping(value: TarsConfig) -> PongResult:
        nonlocal pinged
        pinged = value is config
        return PongResult(server_version="0.8.0", uptime_ms=0, received_at="now",
                          schema_revision="0003", launch_id="expected-launch")

    monkeypatch.setattr(core, "read_control_file", lambda path: control)
    monkeypatch.setattr(core, "_ping_check", successful_ping)

    assert await core._wait_until_ready(config, "expected-launch", timeout_s=0.1) == control
    assert pinged


# 功能：验证 nonce 或端点不匹配时不会把旧控制文件误认成本次启动
# 设计：直接覆盖 launch_id、host、port 三个识别条件，避免为纯判定逻辑引入等待
def test_matches_launch_rejects_wrong_nonce_or_endpoint() -> None:
    config = TarsConfig()
    matching = DaemonControl(
        pid=202,
        host=config.host,
        port=config.port,
        token="token",
        launch_id="expected-launch",
    )

    assert core._matches_launch(matching, config, "expected-launch")
    assert not core._matches_launch(matching, config, "other-launch")
    assert not core._matches_launch(
        DaemonControl(
            pid=202,
            host="127.0.0.2",
            port=config.port,
            token="token",
            launch_id="expected-launch",
        ),
        config,
        "expected-launch",
    )
    assert not core._matches_launch(
        DaemonControl(
            pid=202,
            host=config.host,
            port=config.port + 1,
            token="token",
            launch_id="expected-launch",
        ),
        config,
        "expected-launch",
    )


def test_terminate_launcher_uses_shared_sync_tree_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = object()
    calls: list[tuple[object, float]] = []

    def fake_cleanup(proc: object, *, grace_s: float) -> None:
        calls.append((proc, grace_s))

    monkeypatch.setattr(core, "terminate_popen_process_tree", fake_cleanup)

    core._terminate_launcher(process)  # type: ignore[arg-type]

    assert calls == [(process, 2.0)]


# 功能：验证失败清理会处理同一 launch 的 daemon 与 launcher
# 设计：模拟 daemon 无法优雅退出，断言只对 nonce 匹配的 daemon PID 发终止信号
async def test_cleanup_failed_start_cleans_matching_daemon_and_launcher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakeProcess:
        pid = 101

        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            raise AssertionError("terminate should be sufficient")

    config = TarsConfig()
    control_path = tmp_path / "control.json"
    control = DaemonControl(
        pid=202,
        host=config.host,
        port=config.port,
        token="token",
        launch_id="expected-launch",
    )
    process = _FakeProcess()
    killed_pids: list[int] = []

    async def failed_shutdown(value: DaemonControl) -> None:
        assert value is control
        raise ConnectionRefusedError

    monkeypatch.setattr(core, "control_file_for", lambda port: control_path)
    monkeypatch.setattr(core, "read_control_file", lambda path: control)
    monkeypatch.setattr(core, "_shutdown_daemon", failed_shutdown)
    monkeypatch.setattr(core.os, "kill", lambda pid, sig: killed_pids.append(pid))
    monkeypatch.setattr(core, "remove_control_file", lambda token, path: path.unlink())
    monkeypatch.setattr(
        core,
        "terminate_popen_process_tree",
        lambda proc, grace_s: (proc.terminate(), proc.wait(timeout=grace_s)),
    )
    control_path.write_text("owned", encoding="utf-8")

    await core._cleanup_failed_start(process, config, "expected-launch")

    assert killed_pids == []  # Only the owned Popen tree is force-terminated.
    assert process.terminated
    assert not control_path.exists()


# 功能：验证失败清理不会终止不属于本次 launch 的 daemon
# 设计：旧控制文件 nonce 不匹配时，daemon 清理路径必须跳过，但 launcher 仍需回收
async def test_cleanup_failed_start_preserves_unmatched_daemon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakeProcess:
        pid = 101

        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            raise AssertionError("terminate should be sufficient")

    config = TarsConfig()
    control = DaemonControl(
        pid=202,
        host=config.host,
        port=config.port,
        token="token",
        launch_id="other-launch",
    )
    process = _FakeProcess()

    monkeypatch.setattr(core, "control_file_for", lambda port: tmp_path / "control.json")
    monkeypatch.setattr(core, "read_control_file", lambda path: control)
    monkeypatch.setattr(
        core,
        "_shutdown_daemon",
        lambda value: pytest.fail("unmatched daemon must not receive shutdown"),
    )
    monkeypatch.setattr(
        core.os,
        "kill",
        lambda pid, sig: pytest.fail("unmatched daemon PID must not be terminated"),
    )
    monkeypatch.setattr(
        core,
        "remove_control_file",
        lambda token, path: pytest.fail("unmatched control file must not be removed"),
    )
    monkeypatch.setattr(
        core,
        "terminate_popen_process_tree",
        lambda proc, grace_s: (proc.terminate(), proc.wait(timeout=grace_s)),
    )

    await core._cleanup_failed_start(process, config, "expected-launch")

    assert process.terminated




async def test_ready_rejects_ping_from_different_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    config = TarsConfig()
    control = DaemonControl(1, config.host, config.port, "token", "expected")
    monkeypatch.setattr(core, "read_control_file", lambda _path: control)
    async def ping(_config: TarsConfig) -> PongResult:
        return PongResult(server_version="0.8.0", uptime_ms=0, received_at="now",
                          schema_revision="0003", launch_id="other")
    monkeypatch.setattr(core, "_ping_check", ping)
    assert await core._wait_until_ready(config, "expected", timeout_s=0.01) is None


class _ReadinessClock:
    def __init__(self) -> None:
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    async def sleep(self, _delay: float) -> None:
        self.elapsed += 5.0


async def test_slow_legal_readiness_is_verified_after_old_ten_second_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ReadinessClock()
    config = TarsConfig()
    control = DaemonControl(pid=202, host=config.host, port=config.port,
                            token="test-control", launch_id="slow-launch")
    attempts = []

    async def slow_ping(_config: TarsConfig) -> PongResult:
        attempts.append(clock.elapsed)
        if clock.elapsed < 12:
            raise ConnectionRefusedError("cold startup still initializing")
        return PongResult(server_version="0.8.0", uptime_ms=0, received_at="now",
                          schema_revision="0003", launch_id="slow-launch")

    monkeypatch.setattr(core, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(core, "asyncio", SimpleNamespace(sleep=clock.sleep))
    monkeypatch.setattr(core, "read_control_file", lambda path: control)
    monkeypatch.setattr(core, "_ping_check", slow_ping)
    assert await core._wait_until_ready(config, "slow-launch") == control
    assert attempts == [0, 5, 10, 15]
    assert clock.elapsed > 10


@pytest.mark.parametrize("owned_control", [True, False])
def test_default_readiness_deadline_still_cleans_only_owned_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owned_control: bool,
) -> None:
    clock = _ReadinessClock()
    config = TarsConfig()
    control = DaemonControl(
        pid=202, host=config.host, port=config.port, token="test-control",
        launch_id="our-launch" if owned_control else "other-launch",
    )
    process = SimpleNamespace(pid=101)
    ping_calls = 0
    shutdown = []
    removed = []
    terminated = []

    async def unavailable_ping(_config: TarsConfig) -> PongResult:
        nonlocal ping_calls
        ping_calls += 1
        if ping_calls == 1:
            raise ConnectionRefusedError
        # A matching control file cannot override a different RPC launch identity.
        return PongResult(server_version="0.8.0", uptime_ms=0, received_at="now",
                          schema_revision="0003", launch_id="wrong-rpc-launch")

    async def shutdown_owned(value: DaemonControl) -> None:
        shutdown.append(value)

    async def stopped(value: DaemonControl, timeout_s: float) -> bool:
        return True

    monkeypatch.setattr(core, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(core, "asyncio", SimpleNamespace(run=asyncio.run, sleep=clock.sleep))
    monkeypatch.setattr(core, "_ping_check", unavailable_ping)
    monkeypatch.setattr(core, "read_control_file", lambda path: control)
    monkeypatch.setattr(core, "control_file_for", lambda port: tmp_path / "control.json")
    monkeypatch.setattr(core.secrets, "token_urlsafe", lambda length: "our-launch")
    monkeypatch.setattr(core.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(core, "_shutdown_daemon", shutdown_owned)
    monkeypatch.setattr(core, "_wait_until_stopped", stopped)
    monkeypatch.setattr(core, "remove_control_file", lambda token, path: removed.append(token))
    monkeypatch.setattr(core, "_terminate_launcher", lambda proc: terminated.append(proc))
    with pytest.raises(SystemExit) as failure:
        core.cmd_core_start(config)
    assert failure.value.code == 1
    assert clock.elapsed == 45
    assert terminated == [process]
    assert shutdown == ([control] if owned_control else [])
    assert removed == ([control.token] if owned_control else [])
