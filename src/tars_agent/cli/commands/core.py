from __future__ import annotations

import asyncio
import os
import secrets
import subprocess
import sys
import time

from pydantic import ValidationError

from tars_agent.core.bus.commands import PongResult
from tars_agent.core.config import TarsConfig
from tars_agent.core.control import (
    CORE_LAUNCH_ID_ENV,
    DaemonControl,
    control_file_for,
    read_control_file,
    remove_control_file,
)
from tars_agent.core.paths import tars_home
from tars_agent.core.processes import (
    subprocess_group_kwargs,
    terminate_popen_process_tree,
)
from tars_agent.core.transport.socket_client import IpcError, SocketClient


# 通过 JSON-RPC core.ping 验证端点确实是可工作的 TARS-Agent daemon，而非仅有 TCP listener
async def _ping_check(config: TarsConfig) -> PongResult:
    client = SocketClient(config.host, config.port)
    await asyncio.wait_for(client.connect(), timeout=2.0)
    event_loop = asyncio.create_task(client.run_event_loop())
    try:
        result = await client.send_command(
            "core.ping",
            {"client": "core-launcher"},
            timeout_s=2.0,
        )
        return PongResult.model_validate(result)
    finally:
        event_loop.cancel()
        try:
            await client.close()
        except OSError:
            pass
        await asyncio.gather(event_loop, return_exceptions=True)


# 通过控制面 RPC 请求 daemon 优雅关闭并释放客户端资源
async def _shutdown_daemon(control: DaemonControl) -> None:
    client = SocketClient(control.host, control.port)
    await asyncio.wait_for(client.connect(), timeout=2.0)
    event_loop = asyncio.create_task(client.run_event_loop())
    try:
        await client.send_command(
            "core.shutdown",
            {"token": control.token},
            timeout_s=5.0,
        )
    finally:
        try:
            await client.close()
        except OSError:
            pass
        await asyncio.gather(event_loop, return_exceptions=True)


# 判断控制文件是否由本次启动写入，且监听端点与请求配置一致
def _matches_launch(
    control: DaemonControl,
    config: TarsConfig,
    launch_id: str,
) -> bool:
    return (
        control.launch_id == launch_id
        and control.host == config.host
        and control.port == config.port
    )


# 等待匹配控制文件和真实 RPC 就绪；45 秒覆盖预检及迁移冷启动，不依赖启动器 PID
async def _wait_until_ready(
    config: TarsConfig,
    launch_id: str,
    timeout_s: float = 45.0,
) -> DaemonControl | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        control = read_control_file(control_file_for(config.port))
        if control is not None and _matches_launch(control, config, launch_id):
            try:
                pong = await _ping_check(config)
                if pong.launch_id == launch_id:
                    return control
            except (
                ConnectionRefusedError,
                IpcError,
                OSError,
                TimeoutError,
                ValidationError,
            ):
                pass
        await asyncio.sleep(0.05)
    return None


# 终止本次创建且仍存活的 launcher，必要时升级为 kill
def _terminate_launcher(proc: subprocess.Popen[bytes]) -> None:
    terminate_popen_process_tree(proc, grace_s=2.0)


# 启动失败时，仅清理 launch_id 和端点都匹配的 daemon，再清理 launcher
async def _cleanup_failed_start(
    proc: subprocess.Popen[bytes],
    config: TarsConfig,
    launch_id: str,
) -> None:
    control_path = control_file_for(config.port)
    control = read_control_file(control_path)
    if control is not None and _matches_launch(control, config, launch_id):
        try:
            await _shutdown_daemon(control)
            await _wait_until_stopped(control, timeout_s=2.0)
        except (ConnectionRefusedError, IpcError, OSError, TimeoutError):
            pass
        remove_control_file(control.token, control_path)
    _terminate_launcher(proc)


# 有界轮询 daemon 监听端口，确认优雅关闭已经完成
async def _wait_until_stopped(
    control: DaemonControl,
    timeout_s: float = 5.0,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            _reader, writer = await asyncio.open_connection(control.host, control.port)
        except (ConnectionRefusedError, OSError):
            current = read_control_file(control_file_for(control.port))
            if current is None or current.token != control.token:
                return True
            await asyncio.sleep(0.05)
            continue
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=0.5)
        except (TimeoutError, OSError):
            pass
        await asyncio.sleep(0.05)
    return False


# 打印 daemon 当前状态（running / not running）
def cmd_core_status(config: TarsConfig) -> None:
    try:
        asyncio.run(_ping_check(config))
        print(f"running  ({config.host}:{config.port})")
    except (
        ConnectionRefusedError,
        IpcError,
        OSError,
        TimeoutError,
        ValidationError,
    ):
        print("not running")


# 在后台启动 daemon，若已在运行则提示并退出
def cmd_core_start(config: TarsConfig) -> None:
    try:
        asyncio.run(_ping_check(config))
        print(f"already running  ({config.host}:{config.port})")
        return
    except (
        ConnectionRefusedError,
        IpcError,
        OSError,
        TimeoutError,
        ValidationError,
    ):
        pass

    launch_id = secrets.token_urlsafe(32)
    child_env = os.environ.copy()
    child_env[CORE_LAUNCH_ID_ENV] = launch_id
    child_env["TARS_HOST"] = config.host
    child_env["TARS_PORT"] = str(config.port)
    launch_log = tars_home() / "logs" / f"core-launch-{launch_id}.log"
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    group_options = subprocess_group_kwargs()
    if sys.platform == "win32":
        group_options["creationflags"] = (
            group_options.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        )
    # Preserve early failures (including config/import errors before logging starts).
    with launch_log.open("ab") as diagnostics:
        proc: subprocess.Popen[bytes] = subprocess.Popen(
            [sys.executable, "-m", "tars_agent.core"],
            env=child_env,
            **group_options,
            stdout=diagnostics,
            stderr=diagnostics,
        )
    control = asyncio.run(_wait_until_ready(config, launch_id))
    if control is None:
        asyncio.run(_cleanup_failed_start(proc, config, launch_id))
        print(f"error: core failed to become ready; inspect {launch_log}",
              file=sys.stderr)
        raise SystemExit(1)
    print(f"started  pid={control.pid}  ({control.host}:{control.port})")


# 通过带本地控制 token 的 RPC 请求 daemon 优雅停止，避免 Windows os.kill 误杀进程
def cmd_core_stop(config: TarsConfig) -> None:
    control_path = control_file_for(config.port)
    control = read_control_file(control_path)
    if control is None:
        print("not running")
        return
    try:
        asyncio.run(_shutdown_daemon(control))
    except (ConnectionRefusedError, OSError):
        remove_control_file(control.token, control_path)
        print("not running")
        return
    except (IpcError, TimeoutError) as exc:
        print(f"error: core shutdown request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if not asyncio.run(_wait_until_stopped(control)):
        print(
            f"error: core did not stop within timeout ({control.host}:{control.port})",
            file=sys.stderr,
        )
        raise SystemExit(1)
    remove_control_file(control.token, control_path)
    print(f"stopped  pid={control.pid}")
