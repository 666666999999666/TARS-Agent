from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from tars_agent.core.bus.envelope import EventPushEnvelope
from tars_agent.core.bus.events import LlmTokenEvent, RunStartedEvent
from tars_agent.core.context import ExecutionContext
from tars_agent.core.events.bus import EventBus
from tars_agent.core.events.durable import DurableEventHub
from tars_agent.core.llm.types import LlmResponse
from tars_agent.core.loop import AgentLoop
from tars_agent.core.persistence import (
    Database,
    RunRecord,
    SessionRecord,
    StateRepository,
)
from tars_agent.core.tools.registry import ToolRegistry
from tars_agent.core.transport.ipc_broadcaster import IpcEventBroadcaster
from tars_agent.core.transport.socket_server import ConnectionSender


class _CollectingSender:
    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.messages: list[BaseModel] = []

    async def send(self, message: BaseModel, *, wait: bool = False) -> None:
        del wait
        self.messages.append(message)

    async def close(self) -> None:
        return


class _BlockedSender(_CollectingSender):
    def __init__(self, client_id: str) -> None:
        super().__init__(client_id)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def send(self, message: BaseModel, *, wait: bool = False) -> None:
        del wait
        self.entered.set()
        await self.release.wait()
        self.messages.append(message)


class _BurstProvider:
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        del messages, tool_schemas, step, system
        for index in range(600):
            await bus.publish(
                LlmTokenEvent(
                    run_id=run_id,
                    token=str(index),
                    ts="2026-01-01T00:00:00Z",
                )
            )
        return LlmResponse(stop_reason="end_turn", text="done")


async def _runtime(tmp_path: Path) -> tuple[Database, DurableEventHub]:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    async with database.transaction() as session:
        repository = StateRepository(session)
        await repository.add_session(
            SessionRecord(id="sess-1", mode="chat", status="running", title="")
        )
        await repository.add_run(
            RunRecord(
                id="run-1",
                session_id="sess-1",
                kind="chat",
                attempt=1,
                status="running",
            )
        )
    hub = DurableEventHub(database, batch_interval_s=0.001)
    await hub.start()
    return database, hub


def _sender() -> tuple[ConnectionSender, asyncio.StreamWriter]:
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.drain = AsyncMock()
    writer.wait_closed = AsyncMock()
    sender = ConnectionSender(cast(asyncio.StreamWriter, writer))
    sender.start()
    return sender, cast(asyncio.StreamWriter, writer)


# 功能：验证 Broadcaster 只转发 DurableEventHub 已落库且 topic 匹配的版本化 envelope
# 设计：真实 Hub 分配 cursor，mock writer 仅替代 TCP，覆盖 Hub→订阅泵→单写协程完整链路
async def test_subscriber_receives_durable_matching_event(tmp_path: Path) -> None:
    database, hub = await _runtime(tmp_path)
    broadcaster = IpcEventBroadcaster(hub)
    sender, writer = _sender()
    try:
        await broadcaster.subscribe(sender, ["run.*"], run_id="run-1")
        await hub.handle(
            RunStartedEvent(
                run_id="run-1",
                goal="test",
                ts="2026-01-01T00:00:00Z",
            )
        )
        await hub.flush()

        async def written() -> None:
            while writer.write.call_count == 0:  # type: ignore[attr-defined]
                await asyncio.sleep(0)

        await asyncio.wait_for(written(), timeout=1.0)
        data = json.loads(writer.write.call_args[0][0].rstrip(b"\n"))  # type: ignore[attr-defined]
        assert data["kind"] == "event"
        assert data["protocol_version"] == 2
        assert data["event_schema_version"] == 1
        assert data["cursor"] > 0
        assert data["event"]["type"] == "run.started"
    finally:
        await broadcaster.stop()
        await sender.close()
        await hub.stop()
        await database.dispose()


# 功能：验证断开连接会取消其全部事件泵并释放 Hub 订阅
# 设计：同一 sender 建立两个订阅，调用 unsubscribe 后断言 broadcaster 与 hub 均无残留任务
async def test_unsubscribe_releases_all_connection_subscriptions(tmp_path: Path) -> None:
    database, hub = await _runtime(tmp_path)
    broadcaster = IpcEventBroadcaster(hub)
    sender, _writer = _sender()
    try:
        await broadcaster.subscribe(sender, ["run.*"])
        await broadcaster.subscribe(sender, ["tool.*"])
        assert hub.subscription_count == 2

        await broadcaster.unsubscribe(sender)

        assert hub.subscription_count == 0
    finally:
        await sender.close()
        await hub.stop()
        await database.dispose()


async def test_slow_socket_does_not_block_agent_loop_or_fast_clients(
    tmp_path: Path,
) -> None:
    database, hub = await _runtime(tmp_path)
    broadcaster = IpcEventBroadcaster(hub)
    fast_one = _CollectingSender("fast-1")
    fast_two = _CollectingSender("fast-2")
    slow = _BlockedSender("slow")
    bus = EventBus()
    durable_subscription = bus.subscribe(hub.handle)
    try:
        for sender in (fast_one, fast_two, slow):
            await broadcaster.subscribe(
                cast(ConnectionSender, sender),
                ["*"],
                run_id="run-1",
            )

        loop = AgentLoop(
            _BurstProvider(),
            ToolRegistry(),
            bus,
            workspace_root=tmp_path,
        )
        context = ExecutionContext(run_id="run-1", goal="burst", max_steps=1)

        await asyncio.wait_for(loop.run(context), timeout=2.0)
        assert context.status == "success"
        await asyncio.wait_for(slow.entered.wait(), timeout=1.0)
        await hub.flush()

        async def fast_clients_caught_up() -> None:
            while any(
                sum(
                    isinstance(message, EventPushEnvelope)
                    and message.event.get("type") == "llm.token"
                    for message in sender.messages
                )
                < 600
                for sender in (fast_one, fast_two)
            ):
                await asyncio.sleep(0)

        await asyncio.wait_for(fast_clients_caught_up(), timeout=2.0)
        cursor_lists = [
            [
                message.cursor
                for message in sender.messages
                if isinstance(message, EventPushEnvelope)
            ]
            for sender in (fast_one, fast_two)
        ]
        assert cursor_lists[0] == cursor_lists[1]
        assert cursor_lists[0] == sorted(cursor_lists[0])
        assert len(cursor_lists[0]) == len(set(cursor_lists[0]))
    finally:
        durable_subscription.unsubscribe()
        await broadcaster.stop()
        slow.release.set()
        await hub.stop()
        await database.dispose()
