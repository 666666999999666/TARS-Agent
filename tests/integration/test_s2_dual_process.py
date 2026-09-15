from __future__ import annotations

import asyncio
import subprocess
from typing import Any

from tars_agent.core.transport.socket_client import SocketClient


# 功能：验证 agent.run 命令返回非空 run_id，且 daemon 随即广播 run.started 事件
# 设计：用 SocketClient 封装 IPC 层，asyncio.Event 等待事件而非轮询，
#       timeout=5s 防测试挂起；run.started 在 LLM 调用前触发，无需真实 API Key
async def test_agent_run_returns_run_id_and_emits_started(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    client = SocketClient("127.0.0.1", free_port)
    await client.connect()

    started_event: asyncio.Event = asyncio.Event()
    received: dict[str, Any] = {}

    async def on_event(event: dict[str, Any]) -> None:
        if event.get("type") == "run.started":
            received.update(event)
            started_event.set()

    client.on_event(on_event)
    loop_task = asyncio.create_task(client.run_event_loop())

    try:
        await client.send_command("event.subscribe", {"topics": ["run.*"]})
        result = await client.send_command("agent.run", {"goal": "hello"})

        assert result.get("run_id"), "run_id must be non-empty"
        returned_run_id: str = result["run_id"]

        await asyncio.wait_for(started_event.wait(), timeout=5.0)
        assert received.get("run_id") == returned_run_id
        assert received.get("goal") == "hello"
    finally:
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)
        await client.close()


# 功能：验证两个独立客户端同时订阅后，其中一个触发 agent.run，两个都能收到 run.started 广播
# 设计：两个 SocketClient 并行等待事件（asyncio.gather），确认 IpcEventBroadcaster 的扇出语义；
#       不需要两个客户端都发命令，只验证广播覆盖所有订阅者
async def test_two_clients_both_receive_broadcast(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    client1 = SocketClient("127.0.0.1", free_port)
    client2 = SocketClient("127.0.0.1", free_port)
    await client1.connect()
    await client2.connect()

    event1: asyncio.Event = asyncio.Event()
    event2: asyncio.Event = asyncio.Event()

    async def on_event1(event: dict[str, Any]) -> None:
        if event.get("type") == "run.started":
            event1.set()

    async def on_event2(event: dict[str, Any]) -> None:
        if event.get("type") == "run.started":
            event2.set()

    client1.on_event(on_event1)
    client2.on_event(on_event2)

    loop1 = asyncio.create_task(client1.run_event_loop())
    loop2 = asyncio.create_task(client2.run_event_loop())

    try:
        await client1.send_command("event.subscribe", {"topics": ["run.*"]})
        await client2.send_command("event.subscribe", {"topics": ["run.*"]})
        await client1.send_command("agent.run", {"goal": "broadcast test"})

        await asyncio.wait_for(
            asyncio.gather(event1.wait(), event2.wait()),
            timeout=5.0,
        )
    finally:
        loop1.cancel()
        loop2.cancel()
        await asyncio.gather(loop1, loop2, return_exceptions=True)
        await client1.close()
        await client2.close()


# 功能：验证客户端断开后使用 SQLite cursor 重连，可按 Run scope 回放持久化事件
# 设计：第一连接记录 run.started 的 envelope cursor，第二连接 after_cursor=0 回放同一 Run，
#       断言高水位与回放数量，证明不依赖调试 events.jsonl 文件
async def test_disconnect_and_replay_from_cursor(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    # Phase 1: trigger a run and wait for run.started to be written to disk
    client1 = SocketClient("127.0.0.1", free_port)
    await client1.connect()

    started_event: asyncio.Event = asyncio.Event()
    run_id_holder: list[str] = []
    cursor_holder: list[int] = []

    async def on_event(event: dict[str, Any]) -> None:
        if event.get("type") == "run.started":
            run_id_holder.append(event.get("run_id", ""))
            started_event.set()

    client1.on_event(on_event)

    async def on_envelope(envelope: dict[str, Any]) -> None:
        if envelope.get("event", {}).get("type") == "run.started":
            cursor_holder.append(int(envelope["cursor"]))

    client1.on_event_envelope(on_envelope)
    loop1 = asyncio.create_task(client1.run_event_loop())

    try:
        await client1.send_command("event.subscribe", {"topics": ["run.*"]})
        await client1.send_command("agent.run", {"goal": "replay test"})
        await asyncio.wait_for(started_event.wait(), timeout=5.0)
    finally:
        loop1.cancel()
        await asyncio.gather(loop1, return_exceptions=True)
        await client1.close()

    assert run_id_holder, "run.started was never received"
    assert cursor_holder, "run.started envelope cursor was never received"
    run_id = run_id_holder[0]

    # Phase 2: reconnect from durable cursor 0 and verify the Run-scoped replay
    client2 = SocketClient("127.0.0.1", free_port)
    await client2.connect()
    loop2 = asyncio.create_task(client2.run_event_loop())

    try:
        result = await client2.send_command(
            "event.subscribe",
            {
                "topics": ["run.*"],
                "run_id": run_id,
                "after_cursor": 0,
            },
        )
        assert result.get("replayed_count", 0) > 0, (
            f"Expected replayed_count > 0 for run_id={run_id!r}, got {result}"
        )
        assert result["high_water_cursor"] >= cursor_holder[0]
    finally:
        loop2.cancel()
        await asyncio.gather(loop2, return_exceptions=True)
        await client2.close()
