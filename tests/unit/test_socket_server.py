from __future__ import annotations

import asyncio
import json
import socket

from pydantic import BaseModel

from tars_agent.core.transport.socket_server import ConnectionSender, SocketServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# 功能：验证客户端断开后 SocketServer 调用 broadcaster.unsubscribe(writer) 清理订阅
# 设计：用内联 MockBroadcaster 捕获 unsubscribe 调用并设置 asyncio.Event，避免 sleep 轮询；
#       等待 Event 而非断言调用次数，确保时序正确性而不依赖竞态假设
async def test_broadcaster_unsubscribe_called_on_disconnect() -> None:
    unsubscribed = asyncio.Event()

    class MockBroadcaster:
        def unsubscribe(self, writer: object) -> None:
            unsubscribed.set()

    port = _free_port()
    server = SocketServer("127.0.0.1", port, broadcaster=MockBroadcaster())  # type: ignore[arg-type]
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()
        await writer.wait_closed()

        await asyncio.wait_for(unsubscribed.wait(), timeout=2.0)
    finally:
        await server.stop()


# 功能：验证不传入 broadcaster 时 SocketServer 仍可正常启动和停止（backward-compatible 默认值）
# 设计：直接实例化 SocketServer(host, port)（无 broadcaster），start/stop 不抛异常即为通过；
#       回归测试确保新参数的默认值 None 不破坏现有调用方
async def test_no_broadcaster_server_starts_and_stops() -> None:
    port = _free_port()
    server = SocketServer("127.0.0.1", port)
    await server.start()
    await server.stop()


# 功能：验证 after_response 只在 JSON-RPC 成功响应已经 drain 后触发
# 设计：回调设置 Event，客户端随后读取完整响应，确保 shutdown 等生命周期操作不抢先关闭连接
async def test_after_response_callback_runs_after_success_is_sent() -> None:
    port = _free_port()
    callback_called = asyncio.Event()
    server = SocketServer("127.0.0.1", port)

    async def handler(params: dict[str, object]) -> dict[str, bool]:
        assert params == {}
        return {"ok": True}

    server.register("core.shutdown", handler, after_response=callback_called.set)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        request = {
            "jsonrpc": "2.0",
            "id": "shutdown",
            "method": "core.shutdown",
            "params": {},
        }
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()

        await asyncio.wait_for(callback_called.wait(), timeout=2.0)
        response = json.loads(await asyncio.wait_for(reader.readline(), timeout=2.0))
        assert response["result"] == {"ok": True}
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


# 功能：验证同一连接的并发响应与事件发送始终由唯一 writer 协程串行 drain
# 设计：mock writer 的 drain 记录同时进入数，两个 send 并发执行时最大并发必须保持为一
async def test_connection_sender_serializes_concurrent_writes() -> None:
    class Message(BaseModel):
        value: int

    class Writer:
        def __init__(self) -> None:
            self.frames: list[bytes] = []
            self.active_drains = 0
            self.max_active_drains = 0

        def write(self, data: bytes) -> None:
            self.frames.append(data)

        async def drain(self) -> None:
            self.active_drains += 1
            self.max_active_drains = max(self.max_active_drains, self.active_drains)
            await asyncio.sleep(0.01)
            self.active_drains -= 1

        def get_extra_info(self, name: str, default: object = None) -> object:
            del name
            return default

        def close(self) -> None:
            return None

        async def wait_closed(self) -> None:
            return None

    writer = Writer()
    sender = ConnectionSender(writer)  # type: ignore[arg-type]
    sender.start()
    try:
        await asyncio.gather(
            sender.send(Message(value=1), wait=True),
            sender.send(Message(value=2), wait=True),
        )
    finally:
        await sender.close()

    assert writer.max_active_drains == 1
    assert [json.loads(frame) for frame in writer.frames] == [{"value": 1}, {"value": 2}]
