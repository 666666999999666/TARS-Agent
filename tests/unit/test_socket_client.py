from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tars_agent.core.transport.socket_client import (
    IpcDisconnectedError,
    IpcError,
    SocketClient,
)


async def _start_mock_server(
    handler: Any,
) -> tuple[asyncio.Server, int]:
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port: int = server.sockets[0].getsockname()[1]
    return server, port


# 功能：验证 send_command 向 mock server 发送 JSON-RPC 请求并正确解析响应 result
# 设计：用 asyncio.start_server + port 0 启动内存中的 mock server，避免依赖真实 daemon；
#       loop_task 并发运行 run_event_loop，使 send_command 的 future 能被 _dispatch 解析
async def test_send_command_returns_result() -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        req = json.loads(line)
        resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"pong": True}}
        writer.write(json.dumps(resp).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())

        result = await asyncio.wait_for(
            client.send_command("core.ping", {"client": "test"}),
            timeout=2.0,
        )
        assert result == {"pong": True}

        await loop_task
        await client.close()


# 功能：验证 server 返回 JSON-RPC error 时 send_command 抛出 IpcError 并携带正确错误码
# 设计：mock server 返回 error 对象（code=-32601），断言异常类型和 code 属性，确认客户端的错误路径处理
async def test_send_command_raises_ipc_error() -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        req = json.loads(line)
        resp = {
            "jsonrpc": "2.0",
            "id": req["id"],
            "error": {"code": -32601, "message": "Method not found"},
        }
        writer.write(json.dumps(resp).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())

        with pytest.raises(IpcError) as exc_info:
            await asyncio.wait_for(
                client.send_command("core.nope", {}),
                timeout=2.0,
            )
        assert exc_info.value.code == -32601

        await loop_task
        await client.close()


# 功能：验证 server 推送 kind=event 的消息时，on_event 注册的 handler 能收到 event 字典
# 设计：server 先返回 RPC 响应（解除 send_command 的等待），再推送事件；用 asyncio.Event 等待 handler 被调用，
#       避免 sleep 轮询；断言 event 内容中的 type 字段是否正确
async def test_event_push_routed_to_handler() -> None:
    received_events: list[dict[str, Any]] = []
    push_done = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        req = json.loads(line)
        resp = {"jsonrpc": "2.0", "id": req["id"], "result": {"subscription_id": "sub-1"}}
        writer.write(json.dumps(resp).encode() + b"\n")
        push = {
            "kind": "event",
            "protocol_version": 2,
            "event_schema_version": 1,
            "cursor": 1,
            "session_id": "sess-1",
            "run_id": "r1",
            "occurred_at": "2026-01-01T00:00:00Z",
            "event": {"type": "run.started", "run_id": "r1"},
        }
        writer.write(json.dumps(push).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)

        async def collect(event_data: dict[str, Any]) -> None:
            received_events.append(event_data)
            push_done.set()

        client.on_event(collect)
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())

        await asyncio.wait_for(
            client.send_command("event.subscribe", {"topics": ["run.*"]}),
            timeout=2.0,
        )
        await asyncio.wait_for(push_done.wait(), timeout=2.0)

        assert len(received_events) == 1
        assert received_events[0]["type"] == "run.started"

        await loop_task
        await client.close()


# 功能：验证 server 关闭连接后 run_event_loop 正常退出（不挂起）
# 设计：mock server 立即关闭写流，client readline 收到空字节后退出循环；
#       用 asyncio.wait_for 设置超时，防止 loop 意外挂起导致测试卡死
async def test_run_event_loop_exits_on_server_close() -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.close()
        await writer.wait_closed()

    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)
        await client.connect()
        await asyncio.wait_for(client.run_event_loop(), timeout=2.0)
        await client.close()


# 功能：验证 RPC 等待期间连接断开会收到明确 IpcDisconnectedError，而不是 CancelledError
# 设计：服务端读完请求立即关闭且不响应，覆盖 Web 并发请求需要的确定性失败语义
async def test_pending_command_fails_with_disconnected_error() -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        writer.close()
        await writer.wait_closed()

    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)
        await client.connect()
        loop_task = asyncio.create_task(client.run_event_loop())

        with pytest.raises(IpcDisconnectedError):
            await client.send_command("run.get", {"run_id": "run-missing"})

        await loop_task
        await client.close()


# 功能：验证客户端按 cursor 去重且只接受支持的事件 Schema
# 设计：直接 dispatch 重复 cursor 与未知 Schema envelope，断言 handler 仅收到一次合法事件
async def test_event_cursor_dedup_and_schema_guard() -> None:
    client = SocketClient("127.0.0.1", 1)
    received: list[dict[str, Any]] = []
    envelopes: list[dict[str, Any]] = []

    async def collect(event: dict[str, Any]) -> None:
        received.append(event)

    client.on_event(collect)

    async def collect_envelope(envelope: dict[str, Any]) -> None:
        envelopes.append(envelope)

    client.on_event_envelope(collect_envelope)
    base = {
        "kind": "event",
        "protocol_version": 2,
        "event_schema_version": 1,
        "cursor": 7,
        "occurred_at": "2026-01-01T00:00:00Z",
        "event": {"type": "run.started"},
    }
    await client._dispatch((json.dumps(base) + "\n").encode())
    await client._dispatch((json.dumps(base) + "\n").encode())
    unsupported = {**base, "cursor": 8, "event_schema_version": 99}
    await client._dispatch((json.dumps(unsupported) + "\n").encode())

    assert received == [{"type": "run.started"}]
    assert client.last_cursor == 8
    assert envelopes[-1] == {
        "kind": "compatibility.error",
        "protocol_version": 2,
        "cursor": 8,
        "received_protocol_version": 2,
        "received_event_schema_version": 99,
        "reason": "unsupported event envelope version",
    }


async def test_cancel_timeout_retains_structured_error_data() -> None:
    data = {"run_id": "r", "cancellation_requested": True, "pending_run_ids": ["r", "child"]}
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = json.loads(await reader.readline())
        writer.write(json.dumps({"jsonrpc": "2.0", "id": request["id"], "error": {
            "code": -32033, "message": "pending", "data": data,
        }}).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    server, port = await _start_mock_server(handle)
    async with server:
        client = SocketClient("127.0.0.1", port)
        await client.connect()
        loop = asyncio.create_task(client.run_event_loop())
        try:
            with pytest.raises(IpcError) as caught:
                await client.send_command("run.cancel", {"run_id": "r"})
            assert caught.value.data == data
        finally:
            await client.close()
            await loop
