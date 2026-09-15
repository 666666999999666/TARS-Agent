from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tars_agent.core.control import read_control_file


# 功能：验证真实 daemon 响应 core.ping 命令并返回包含版本、uptime、时间戳的 PongResult
# 设计：通过原始 TCP 连接发送 JSON-RPC 帧（不经过任何 SDK 客户端层），直接验证 wire 协议的端到端正确性
async def test_ping_returns_pong(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
    req = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "core.ping",
        "params": {"client": "test/1.0.0"},
    }
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    resp = json.loads(line)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == "test-1"
    assert "result" in resp
    assert resp["result"]["server_version"] == "0.8.0"
    assert resp["result"]["schema_revision"] == "0003"
    assert resp["result"]["event_schema_versions"] == [1]
    assert resp["result"]["uptime_ms"] >= 0
    assert "received_at" in resp["result"]


# 功能：验证调用未注册方法时 daemon 返回 METHOD_NOT_FOUND 错误码（-32601）
# 设计：检查精确的 JSON-RPC 错误码，确认 SocketServer 的路由失败路径符合 JSON-RPC 2.0 规范
async def test_unknown_method_returns_error(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
    req = {
        "jsonrpc": "2.0",
        "id": "test-2",
        "method": "core.nonexistent",
        "params": {},
    }
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    resp = json.loads(line)
    assert "error" in resp
    assert resp["error"]["code"] == -32601  # METHOD_NOT_FOUND


# 功能：验证发送非 JSON 数据时 daemon 返回 PARSE_ERROR（-32700）并不崩溃
# 设计：发送裸文本而非 JSON，检查错误码，确认 daemon 对格式错误输入的健壮性（不因单个坏帧终止服务）
async def test_invalid_json_returns_error(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
    writer.write(b"not valid json\n")
    await writer.drain()

    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    resp = json.loads(line)
    assert "error" in resp
    assert resp["error"]["code"] == -32700  # PARSE_ERROR


# 功能：验证没有 ANTHROPIC_API_KEY 时 Core 仍能启动并完成 ping 与 Session 管理
# 设计：显式删除 key 后启动真实 daemon，依次走 ping/create/history/close，最后用 core.shutdown 清理
async def test_daemon_management_works_without_api_key(
    free_port: int,
    tmp_path: Path,
) -> None:
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env["TARS_PORT"] = str(free_port)
    env["TARS_LOG_FILE"] = ""
    env["TARS_LOG_LEVEL"] = "WARNING"
    env["TARS_TRACE_ENABLED"] = "false"
    env["TARS_SANDBOX_MODE"] = "preferred"
    env["TARS_HOME"] = str(tmp_path / "kama-home")
    proc = subprocess.Popen([sys.executable, "-m", "tars_agent.core"], env=env)

    try:
        # Importing the pinned official MCP SDK is intentionally lazy, but a cold
        # Windows interpreter plus SQLite migration can still exceed five seconds.
        deadline = time.monotonic() + 10.0
        while True:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
                break
            except (ConnectionRefusedError, OSError):
                if time.monotonic() >= deadline or proc.poll() is not None:
                    pytest.fail(f"daemon failed to start without API key (exit={proc.poll()})")
                await asyncio.sleep(0.05)

        async def call(method: str, params: dict[str, object], req_id: str) -> dict[str, object]:
            request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            return json.loads(line)

        pong = await call("core.ping", {"client": "no-key-test"}, "ping")
        assert "result" in pong

        created = await call(
            "session.create",
            {"mode": "chat", "title": "no-key"},
            "create",
        )
        session_id = created["result"]["session_id"]  # type: ignore[index]
        history = await call(
            "session.get_history",
            {"session_id": session_id},
            "history",
        )
        assert history["result"] == {"messages": []}

        compact = await call(
            "session.compact",
            {"session_id": session_id},
            "compact",
        )
        assert compact["error"]["code"] == -32020  # type: ignore[index]
        still_alive = await call("core.ping", {"client": "after-compact"}, "ping-2")
        assert "result" in still_alive

        closed = await call(
            "session.close",
            {"session_id": session_id},
            "close",
        )
        assert closed["result"] == {"status": "closed"}

        control = read_control_file(Path(env["TARS_HOME"]) / "control" / f"tars-core-{free_port}.json")
        assert control is not None
        shutdown = await call("core.shutdown", {"token": control.token}, "shutdown")
        assert shutdown["result"] == {"ok": True}
        writer.close()
        await writer.wait_closed()
        await asyncio.to_thread(proc.wait, 5.0)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
