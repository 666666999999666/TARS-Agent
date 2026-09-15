from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any


async def _call(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    method: str,
    params: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    writer.write((json.dumps(request) + "\n").encode())
    await writer.drain()
    return json.loads(await asyncio.wait_for(reader.readline(), timeout=5.0))


async def _wait_for_terminal(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    run_id: str,
    prefix: str,
) -> dict[str, Any]:
    for index in range(100):
        response = await _call(
            reader,
            writer,
            "run.get",
            {"run_id": run_id},
            f"{prefix}-{index}",
        )
        result = response["result"]
        if result["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return result
        await asyncio.sleep(0.01)
    raise AssertionError(f"run did not reach a terminal state: {run_id}")


# 功能：验证 Wire V2 的幂等异步提交、Run 查询、失败上下文隔离与显式 retry 闭环
# 设计：真实启动无 API Key daemon，使 Runner 快速受控失败，全程只通过 JSON-RPC 观察数据库状态
async def test_durable_runtime_commands_over_ipc(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
    tmp_path: Path,
) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", free_port)
    try:
        created = await _call(
            reader,
            writer,
            "session.create",
            {
                "mode": "chat",
                "title": "durable",
                "workspace_root": str(tmp_path),
            },
            "create",
        )
        session_id = created["result"]["session_id"]
        assert created["result"]["status"] == "ready"

        listed = await _call(
            reader,
            writer,
            "session.list",
            {"limit": 10},
            "list",
        )
        assert listed["result"]["sessions"][0]["session_id"] == session_id

        resumed = await _call(
            reader,
            writer,
            "session.resume",
            {"session_id": session_id},
            "resume",
        )
        assert resumed["result"]["session"]["session_id"] == session_id
        assert resumed["result"]["latest_cursor"] > 0

        submitted = await _call(
            reader,
            writer,
            "session.send_message",
            {
                "session_id": session_id,
                "content": "first content",
                "client_message_id": "client-1",
            },
            "submit",
        )
        run_id = submitted["result"]["run_id"]
        assert submitted["result"]["status"] == "queued"
        assert submitted["result"]["deduplicated"] is False

        duplicate = await _call(
            reader,
            writer,
            "session.send_message",
            {
                "session_id": session_id,
                "content": "must not create another run",
                "client_message_id": "client-1",
            },
            "duplicate",
        )
        assert duplicate["result"]["run_id"] == run_id
        assert duplicate["result"]["deduplicated"] is True

        original = await _wait_for_terminal(reader, writer, run_id, "poll-original")
        assert original["status"] == "failed"
        assert original["attempt"] == 1
        assert original["reason"] == "llm_error"

        history = await _call(
            reader,
            writer,
            "session.get_history",
            {"session_id": session_id},
            "history",
        )
        assert history["result"]["messages"] == []

        retried = await _call(
            reader,
            writer,
            "session.retry",
            {"run_id": run_id},
            "retry",
        )
        retry_id = retried["result"]["run_id"]
        assert retry_id != run_id
        retry = await _wait_for_terminal(reader, writer, retry_id, "poll-retry")
        assert retry["status"] == "failed"
        assert retry["attempt"] == 2
        assert retry["retry_of_run_id"] == run_id
    finally:
        writer.close()
        await writer.wait_closed()
