from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tars_agent.core.bus.events import (
    RunFinishedEvent,
    RunStartedEvent,
    StepStartedEvent,
)
from tars_agent.core.events.durable import (
    DurableEventHub,
    SubscriptionOverflow,
)
from tars_agent.core.persistence import (
    Database,
    EventRecord,
    RunRecord,
    SessionRecord,
    StateRepository,
)


class _BarrierReplayHub(DurableEventHub):
    def __init__(
        self,
        database: Database,
        replay_entered: asyncio.Event,
        release_replay: asyncio.Event,
    ) -> None:
        super().__init__(database, batch_interval_s=0.001)
        self._replay_entered = replay_entered
        self._release_replay = release_replay

    async def _load_replay(
        self,
        *,
        after_cursor: int,
        high_water: int,
        session_id: str | None,
        run_id: str | None,
    ) -> list[EventRecord]:
        self._replay_entered.set()
        await self._release_replay.wait()
        return await super()._load_replay(
            after_cursor=after_cursor,
            high_water=high_water,
            session_id=session_id,
            run_id=run_id,
        )


async def _database_with_run(tmp_path: Path) -> Database:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    async with database.transaction() as session:
        repository = StateRepository(session)
        await repository.add_session(
            SessionRecord(
                id="sess-1",
                mode="chat",
                status="running",
                title="",
                workspace_root=str(tmp_path),
            )
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
    return database


# 功能：验证事件在订阅可见前已经分配 SQLite cursor，并携带 Session/Run/Schema 信息
# 设计：发布后 flush，再从订阅和 Repository 两侧读取同一 cursor，证明持久化先于扇出
async def test_persist_before_delivery_assigns_versioned_cursor(tmp_path: Path) -> None:
    database = await _database_with_run(tmp_path)
    hub = DurableEventHub(database, batch_interval_s=0.001)
    await hub.start()
    subscription = await hub.subscribe(topics=["run.*"], run_id="run-1")
    try:
        await hub.handle(
            RunStartedEvent(
                run_id="run-1",
                goal="hello",
                ts="2026-01-01T00:00:00Z",
            )
        )
        await hub.flush()
        envelope = await asyncio.wait_for(subscription.get(), timeout=1.0)

        async with database.session() as session:
            records = await StateRepository(session).list_events(run_id="run-1")

        assert [record.cursor for record in records] == [envelope.cursor]
        assert envelope.protocol_version == 2
        assert envelope.event_schema_version == 1
        assert envelope.session_id == "sess-1"
        assert envelope.run_id == "run-1"
    finally:
        hub.unsubscribe(subscription.id)
        await hub.stop()
        await database.dispose()


# 功能：验证 after_cursor 回放后无缝接上订阅建立期间产生的实时事件且 cursor 递增
# 设计：预置两个历史事件，订阅协程查询回放时并发发布第三个事件，最终统一从订阅读取三条
async def test_replay_and_live_merge_is_cursor_ordered(tmp_path: Path) -> None:
    database = await _database_with_run(tmp_path)
    replay_entered = asyncio.Event()
    release_replay = asyncio.Event()
    hub = _BarrierReplayHub(database, replay_entered, release_replay)
    await hub.start()
    try:
        await hub.handle(
            RunStartedEvent(run_id="run-1", goal="x", ts="2026-01-01T00:00:00Z")
        )
        await hub.handle(
            StepStartedEvent(run_id="run-1", step=1, ts="2026-01-01T00:00:01Z")
        )
        await hub.flush()

        subscribe_task = asyncio.create_task(
            hub.subscribe(topics=["*"], run_id="run-1", after_cursor=0)
        )
        await asyncio.wait_for(replay_entered.wait(), timeout=1.0)
        await hub.handle(
            RunFinishedEvent(
                run_id="run-1",
                status="success",
                steps=1,
                ts="2026-01-01T00:00:02Z",
            )
        )
        await hub.flush()
        release_replay.set()
        subscription = await asyncio.wait_for(subscribe_task, timeout=1.0)

        envelopes = [await asyncio.wait_for(subscription.get(), timeout=1.0) for _ in range(3)]
        assert [item.event["type"] for item in envelopes] == [
            "run.started",
            "step.started",
            "run.finished",
        ]
        assert [item.cursor for item in envelopes] == sorted(
            item.cursor for item in envelopes
        )
    finally:
        await hub.stop()
        await database.dispose()


# 功能：验证慢客户端填满有界 live 队列后被标记 overflow，不会阻塞事件持久化 worker
# 设计：队列容量设为 1 且不消费，连续落库两批事件后读取必须得到 SubscriptionOverflow
async def test_slow_subscriber_overflow_does_not_block_persistence(tmp_path: Path) -> None:
    database = await _database_with_run(tmp_path)
    hub = DurableEventHub(database, batch_interval_s=0.001)
    await hub.start()
    subscription = await hub.subscribe(
        topics=["run.*"],
        run_id="run-1",
        queue_capacity=1,
    )
    try:
        for index in range(2):
            await hub.handle(
                RunStartedEvent(
                    run_id="run-1",
                    goal=str(index),
                    ts="2026-01-01T00:00:00Z",
                )
            )
            await hub.flush()

        with pytest.raises(SubscriptionOverflow):
            await subscription.get()
        async with database.session() as session:
            records = await StateRepository(session).list_events(run_id="run-1")
        assert len(records) == 2
    finally:
        await hub.stop()
        await database.dispose()


# 功能：验证 Session scope 与 topic glob 同时过滤其他事件
# 设计：同一 Run 发布 run 与 step 事件，Session 范围仅订阅 step.*，只应读到 step.started
async def test_session_scope_and_topic_filter(tmp_path: Path) -> None:
    database = await _database_with_run(tmp_path)
    hub = DurableEventHub(database, batch_interval_s=0.001)
    await hub.start()
    subscription = await hub.subscribe(topics=["step.*"], session_id="sess-1")
    try:
        await hub.handle(
            RunStartedEvent(run_id="run-1", goal="x", ts="2026-01-01T00:00:00Z")
        )
        await hub.handle(
            StepStartedEvent(run_id="run-1", step=1, ts="2026-01-01T00:00:01Z")
        )
        await hub.flush()
        envelope = await asyncio.wait_for(subscription.get(), timeout=1.0)
        assert envelope.event["type"] == "step.started"
    finally:
        await hub.stop()
        await database.dispose()
