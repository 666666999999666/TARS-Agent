from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from tars_agent.core.app import CoreApp
from tars_agent.core.bus.events import (
    RunFinishedEvent,
    RunStartedEvent,
    StepStartedEvent,
)
from tars_agent.core.events.bus import EventBus
from tars_agent.core.events.durable import (
    DurableEventHub,
    DurableEventHubError,
    SubscriptionOverflow,
    _PendingEvent,
)
from tars_agent.core.persistence import (
    Database,
    EventRecord,
    RunRecord,
    SessionRecord,
    StateRepository,
)
from tars_agent.core.runtime.service import RuntimeService


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


# F01: callers get Hub errors with the original cause; cancellation of an individual
# caller stays CancelledError. A timeout is a failed assertion, never a way to finish
# the task under test. Only the fixture's failure cleanup cancels remaining tasks.
class _ControlledPersistenceHub(DurableEventHub):
    def __init__(self, database: Database) -> None:
        super().__init__(database, queue_capacity=1, batch_interval_s=60.0)
        self.entered = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]
        self.release[1].set()
        self.error: BaseException | None = None
        self.batch_count = 0

    async def _persist_and_fan_out(self, batch: list[_PendingEvent]) -> None:
        index = self.batch_count
        self.batch_count += 1
        if index < len(self.entered):
            self.entered[index].set()
            await self.release[index].wait()
        if self.error is not None:
            raise self.error
        await super()._persist_and_fan_out(batch)


def _event(goal: str = "first") -> RunStartedEvent:
    return RunStartedEvent(run_id="run-1", goal=goal, ts="2026-01-01T00:00:00Z")


async def _assert_finished(*tasks: asyncio.Task[Any]) -> None:
    _done, pending = await asyncio.wait(tasks, timeout=2.0)
    assert not pending, (
        "tasks still pending BEFORE test cleanup: "
        + ", ".join(sorted(task.get_name() for task in pending))
    )


def _assert_hub_error(
    task: asyncio.Task[Any], cause: BaseException | type[BaseException] | None = None,
) -> DurableEventHubError:
    assert task.done() and not task.cancelled()
    error = task.exception()
    assert isinstance(error, DurableEventHubError), repr(error)
    if isinstance(cause, type):
        assert isinstance(error.__cause__, cause)
    elif cause is not None:
        assert error.__cause__ is cause
    return error


class _FailureCase:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.hub = _ControlledPersistenceHub(database)
        self.tasks: list[asyncio.Task[Any]] = []
        self.worker: asyncio.Task[None] | None = None

    async def begin(self, call: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
        entered = asyncio.Event()

        async def invoke() -> Any:
            entered.set()
            return await call

        task = asyncio.create_task(invoke(), name=name)
        self.tasks.append(task)
        # invoke runs until its first suspension before this waiter can resume.
        await entered.wait()
        return task

    async def signal(self, event: asyncio.Event) -> None:
        waiter = await self.begin(event.wait(), "synchronization signal")
        await _assert_finished(waiter)
        assert waiter.result()

    async def start(self) -> None:
        await self.hub.start()
        self.worker = self.hub._worker
        assert self.worker is not None

    async def first_batch(self) -> asyncio.Task[Any]:
        await self.start()
        await self.hub.handle(_event())
        current = await self.begin(self.hub.flush(), "dequeued flush")
        await self.signal(self.hub.entered[0])
        assert self.hub._queue.empty()  # The barrier is already out of the queue.
        assert not current.done()
        return current

    async def drained(self) -> None:
        join = await self.begin(self.hub._queue.join(), "queue join")
        await _assert_finished(join)
        assert join.result() is None
        assert self.hub._queue.empty()
        assert all(task.done() for task in self.tasks)
        assert self.worker is None or self.worker.done()
        assert self.hub.subscription_count == 0

    async def stop(self, cause: BaseException | type[BaseException] | None = None) -> None:
        stopping = await self.begin(self.hub.stop(), "stop")
        await _assert_finished(stopping)
        if cause is None:
            assert stopping.result() is None
        else:
            _assert_hub_error(stopping, cause)
        await self.drained()


@pytest.fixture
async def failure_case(tmp_path: Path) -> AsyncIterator[_FailureCase]:
    case = _FailureCase(await _database_with_run(tmp_path))
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        yield case
    finally:
        # Assertions above must establish termination first. This only prevents a
        # deliberately red regression from leaking work into the following test.
        owned = set(case.tasks)
        owned.update(task for task in (case.worker, getattr(case.hub, "_stop_task", None))
                     if task is not None)
        for task in owned:
            if not task.done():
                task.cancel()
        await asyncio.gather(*owned, return_exceptions=True)
        await case.database.dispose()
        gc.collect()
        checkpoint = loop.create_future()
        loop.call_soon(checkpoint.set_result, None)
        await checkpoint
        loop.set_exception_handler(previous_handler)
        assert not unhandled, unhandled


async def test_f01_empty_flush_before_start_is_a_noop(failure_case: _FailureCase) -> None:
    flush = await failure_case.begin(failure_case.hub.flush(), "unstarted empty flush")
    await _assert_finished(flush)
    assert flush.result() is None
    await failure_case.stop()


async def test_f01_empty_flush_after_start_finishes(failure_case: _FailureCase) -> None:
    await failure_case.start()
    flush = await failure_case.begin(failure_case.hub.flush(), "running empty flush")
    await _assert_finished(flush)
    assert flush.result() is None
    await failure_case.stop()


async def test_f01_success_flush_waits_for_the_actual_write(failure_case: _FailureCase) -> None:
    flush = await failure_case.first_batch()
    async with failure_case.database.session() as session:
        assert await session.scalar(select(func.count(EventRecord.cursor))) == 0
    assert not flush.done()
    failure_case.hub.release[0].set()
    await _assert_finished(flush)
    assert flush.result() is None
    async with failure_case.database.session() as session:
        assert await session.scalar(select(func.count(EventRecord.cursor))) == 1
    await failure_case.stop()


async def test_f01_flush_does_not_wait_for_a_later_batch(failure_case: _FailureCase) -> None:
    first = await failure_case.first_batch()
    failure_case.hub.release[1].clear()
    await failure_case.hub.handle(_event("later"))
    later = await failure_case.begin(failure_case.hub.flush(), "later flush")
    failure_case.hub.release[0].set()
    await failure_case.signal(failure_case.hub.entered[1])
    await _assert_finished(first)
    assert first.result() is None and not later.done()
    async with failure_case.database.session() as session:
        events = await StateRepository(session).list_events(run_id="run-1")
    assert [event.payload["goal"] for event in events] == ["first"]
    failure_case.hub.release[1].set()
    await _assert_finished(later)
    assert later.result() is None
    await failure_case.stop()


async def test_f01_write_failure_settles_current_queued_and_subscription_waiters(
    failure_case: _FailureCase,
) -> None:
    error = OSError("injected write failure")
    failure_case.hub.error = error
    current = await failure_case.first_batch()
    subscription = await failure_case.hub.subscribe(topics=["*"])
    reading = await failure_case.begin(subscription.get(), "subscription read")
    queued = await failure_case.begin(failure_case.hub.flush(), "queued flush")
    assert failure_case.hub._queue.full()
    failure_case.hub.release[0].set()
    await _assert_finished(current, queued, reading)
    _assert_hub_error(current, error)
    _assert_hub_error(queued, error)
    assert isinstance(reading.exception(), SubscriptionOverflow)
    await failure_case.stop(error)


async def test_f01_full_queue_producers_fail_without_post_failure_enqueue(
    failure_case: _FailureCase,
) -> None:
    error = OSError("injected full-queue failure")
    failure_case.hub.error = error
    current = await failure_case.first_batch()
    queued = await failure_case.begin(failure_case.hub.flush(), "queued flush")
    producer = await failure_case.begin(failure_case.hub.handle(_event("blocked")), "blocked handle")
    blocked = await failure_case.begin(failure_case.hub.flush(), "blocked flush")
    assert failure_case.hub._queue.full()
    assert not producer.done() and not blocked.done()
    failure_case.hub.release[0].set()
    await _assert_finished(current, queued, producer, blocked)
    for task in (current, queued, producer, blocked):
        _assert_hub_error(task, error)
    await failure_case.stop(error)


async def test_f01_failed_worker_rejects_new_work_and_stop_reports_failure(
    failure_case: _FailureCase,
) -> None:
    error = OSError("injected permanent failure")
    failure_case.hub.error = error
    current = await failure_case.first_batch()
    failure_case.hub.release[0].set()
    assert failure_case.worker is not None
    await _assert_finished(failure_case.worker)
    new_handle = await failure_case.begin(failure_case.hub.handle(_event()), "new handle")
    new_flush = await failure_case.begin(failure_case.hub.flush(), "new flush")
    subscribing = await failure_case.begin(failure_case.hub.subscribe(topics=["*"]), "new subscribe")
    stopping = await failure_case.begin(failure_case.hub.stop(), "failed stop")
    await _assert_finished(new_handle, new_flush, subscribing, stopping)
    for task in (new_handle, new_flush, subscribing, stopping):
        _assert_hub_error(task, error)
    await _assert_finished(current)
    _assert_hub_error(current, error)
    await failure_case.drained()


@pytest.mark.parametrize("failure", [False, True], ids=["write-succeeds", "write-fails"])
async def test_f01_cancelled_flush_does_not_cancel_other_callers(
    failure_case: _FailureCase, failure: bool,
) -> None:
    error = OSError("failure after a caller left") if failure else None
    failure_case.hub.error = error
    cancelled = await failure_case.first_batch()
    other = await failure_case.begin(failure_case.hub.flush(), "other flush")
    cancelled.cancel()
    await _assert_finished(cancelled)
    assert cancelled.cancelled()
    assert failure_case.worker is not None and not failure_case.worker.done()
    failure_case.hub.release[0].set()
    await _assert_finished(other)
    if error is None:
        assert other.result() is None
        async with failure_case.database.session() as session:
            assert await session.scalar(select(func.count(EventRecord.cursor))) == 1
    else:
        _assert_hub_error(other, error)
    await failure_case.stop(error)


async def test_f01_cancelled_blocked_enqueue_leaves_no_work_or_warning(
    failure_case: _FailureCase,
) -> None:
    current = await failure_case.first_batch()
    queued = await failure_case.begin(failure_case.hub.flush(), "queued flush")
    blocked = await failure_case.begin(failure_case.hub.flush(), "cancelled blocked flush")
    assert failure_case.hub._queue.full() and not blocked.done()
    blocked.cancel()
    await _assert_finished(blocked)
    assert blocked.cancelled()
    failure_case.hub.release[0].set()
    await _assert_finished(current, queued)
    assert current.result() is None and queued.result() is None
    await failure_case.stop()


async def test_f01_cancelled_worker_settles_inflight_queued_and_blocked_waiters(
    failure_case: _FailureCase,
) -> None:
    current = await failure_case.first_batch()
    queued = await failure_case.begin(failure_case.hub.flush(), "queued flush")
    producer = await failure_case.begin(failure_case.hub.handle(_event()), "blocked handle")
    blocked = await failure_case.begin(failure_case.hub.flush(), "blocked flush")
    assert failure_case.worker is not None
    failure_case.worker.cancel()
    await _assert_finished(failure_case.worker, current, queued, producer, blocked)
    assert failure_case.worker.cancelled()
    for task in (current, queued, producer, blocked):
        _assert_hub_error(task, asyncio.CancelledError)
    await failure_case.stop(asyncio.CancelledError)


async def test_f01_worker_cancelled_before_first_step_still_settles_work(
    failure_case: _FailureCase,
) -> None:
    await failure_case.start()
    assert failure_case.worker is not None
    failure_case.worker.cancel()  # No yield since create_task in start().
    await failure_case.hub.handle(_event())
    flush = await failure_case.begin(failure_case.hub.flush(), "flush after early worker cancel")
    await _assert_finished(failure_case.worker, flush)
    _assert_hub_error(flush, asyncio.CancelledError)
    await failure_case.stop(asyncio.CancelledError)


async def test_f01_worker_cancelled_while_collecting_a_batch_balances_queue(
    failure_case: _FailureCase, monkeypatch: pytest.MonkeyPatch,
) -> None:
    taken = asyncio.Event()
    queue_get = failure_case.hub._queue.get

    async def observed_get() -> object:
        item = await queue_get()
        taken.set()
        return item

    monkeypatch.setattr(failure_case.hub._queue, "get", observed_get)
    await failure_case.start()
    await failure_case.hub.handle(_event())
    await failure_case.signal(taken)
    assert failure_case.hub.batch_count == 0
    assert failure_case.worker is not None
    failure_case.worker.cancel()
    await _assert_finished(failure_case.worker)
    assert failure_case.worker.cancelled()
    await failure_case.stop(asyncio.CancelledError)


async def test_f01_stop_rejects_blocked_and_new_calls_but_drains_accepted_work(
    failure_case: _FailureCase,
) -> None:
    current = await failure_case.first_batch()
    queued = await failure_case.begin(failure_case.hub.flush(), "accepted queued flush")
    blocked = await failure_case.begin(failure_case.hub.flush(), "not-yet-accepted flush")
    producer = await failure_case.begin(failure_case.hub.handle(_event()), "not-yet-accepted handle")
    stopping = await failure_case.begin(failure_case.hub.stop(), "stop during full queue")
    new_flush = await failure_case.begin(failure_case.hub.flush(), "new flush during stop")
    new_handle = await failure_case.begin(failure_case.hub.handle(_event()), "new handle during stop")
    await _assert_finished(blocked, producer, new_flush, new_handle)
    for task in (blocked, producer, new_flush, new_handle):
        _assert_hub_error(task)
    assert not stopping.done() and not current.done()
    failure_case.hub.release[0].set()
    await _assert_finished(current, queued, stopping)
    assert all(task.result() is None for task in (current, queued, stopping))
    async with failure_case.database.session() as session:
        assert await session.scalar(select(func.count(EventRecord.cursor))) == 1
    await failure_case.drained()


async def test_f01_concurrent_stops_do_not_leave_extra_queue_items(
    failure_case: _FailureCase,
) -> None:
    current = await failure_case.first_batch()
    first = await failure_case.begin(failure_case.hub.stop(), "first stop")
    second = await failure_case.begin(failure_case.hub.stop(), "second stop")
    failure_case.hub.release[0].set()
    await _assert_finished(current, first, second)
    assert all(task.result() is None for task in (current, first, second))
    await failure_case.drained()


@pytest.mark.parametrize("failure", [False, True], ids=["write-succeeds", "write-fails"])
async def test_f01_cancelled_stop_caller_does_not_abandon_shutdown(
    failure_case: _FailureCase, failure: bool,
) -> None:
    error = OSError("write failed while stopping") if failure else None
    failure_case.hub.error = error
    current = await failure_case.first_batch()
    stopping = await failure_case.begin(failure_case.hub.stop(), "cancelled stop caller")
    stopping.cancel()
    await _assert_finished(stopping)
    assert stopping.cancelled()
    failure_case.hub.release[0].set()
    assert failure_case.worker is not None
    await _assert_finished(current, failure_case.worker)
    if error is None:
        assert current.result() is None
    else:
        _assert_hub_error(current, error)
    await failure_case.stop(error)


async def test_f01_stop_waiting_for_queue_room_observes_writer_failure(
    failure_case: _FailureCase,
) -> None:
    error = OSError("failed while stop was waiting for room")
    failure_case.hub.error = error
    current = await failure_case.first_batch()
    queued = await failure_case.begin(failure_case.hub.flush(), "queued flush")
    stopping = await failure_case.begin(failure_case.hub.stop(), "stop waiting for room")
    assert not stopping.done() and failure_case.hub._queue.full()
    failure_case.hub.release[0].set()
    await _assert_finished(current, queued, stopping)
    for task in (current, queued, stopping):
        _assert_hub_error(task, error)
    await failure_case.drained()


async def test_f01_stop_saves_accepted_events_without_an_explicit_flush(
    failure_case: _FailureCase,
) -> None:
    await failure_case.start()
    await failure_case.hub.handle(_event("first"))
    await failure_case.hub.handle(_event("second"))
    stopping = await failure_case.begin(failure_case.hub.stop(), "stop without flush")
    await failure_case.signal(failure_case.hub.entered[0])
    assert not stopping.done()
    failure_case.hub.release[0].set()
    await _assert_finished(stopping)
    assert stopping.result() is None
    async with failure_case.database.session() as session:
        events = await StateRepository(session).list_events(run_id="run-1")
    assert [event.payload["goal"] for event in events] == ["first", "second"]
    await failure_case.drained()


async def test_f01_stopped_hub_rejects_new_work(failure_case: _FailureCase) -> None:
    await failure_case.start()
    await failure_case.stop()
    flush = await failure_case.begin(failure_case.hub.flush(), "flush after stop")
    handle = await failure_case.begin(failure_case.hub.handle(_event()), "handle after stop")
    await _assert_finished(flush, handle)
    _assert_hub_error(flush)
    _assert_hub_error(handle)
    await failure_case.drained()


async def test_f01_unexpected_worker_return_rejects_queued_work(
    failure_case: _FailureCase, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def premature_return() -> None:
        return

    monkeypatch.setattr(failure_case.hub, "_run", premature_return)
    await failure_case.start()
    await failure_case.hub.handle(_event())
    flush = await failure_case.begin(failure_case.hub.flush(), "flush after worker return")
    assert failure_case.worker is not None
    await _assert_finished(failure_case.worker, flush)
    error = _assert_hub_error(flush, DurableEventHubError)
    assert "exited unexpectedly" in str(error.__cause__)
    await failure_case.stop(DurableEventHubError)


async def test_f01_unexpected_worker_return_during_stop_is_not_clean_shutdown(
    failure_case: _FailureCase, monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_worker = asyncio.Event()

    async def premature_return() -> None:
        await exit_worker.wait()

    monkeypatch.setattr(failure_case.hub, "_run", premature_return)
    await failure_case.start()
    await failure_case.hub.handle(_event())
    stopping = await failure_case.begin(failure_case.hub.stop(), "stop before unexpected return")
    assert not stopping.done() and failure_case.hub._queue.full()
    exit_worker.set()
    await _assert_finished(stopping)
    error = _assert_hub_error(stopping)
    await failure_case.drained()
    assert isinstance(error.__cause__, DurableEventHubError)
    assert "exited unexpectedly" in str(error.__cause__)


async def test_f01_sqlite_transaction_failure_reaches_session_get_caller(
    failure_case: _FailureCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SQLite itself rejects INSERT; neither the repository nor flush is mocked.
    async with failure_case.database.transaction() as session:
        await session.execute(text(
            "CREATE TRIGGER w02_fail_events BEFORE INSERT ON events "
            "BEGIN SELECT RAISE(ABORT, 'W02 injected SQLite failure'); END"
        ))
    current = await failure_case.first_batch()
    flush_entered = asyncio.Event()
    real_flush = failure_case.hub.flush

    async def observed_flush() -> None:
        flush_entered.set()
        await real_flush()

    def no_runner() -> Any:
        raise AssertionError("This read-only handler must not start an agent or model")

    runtime = RuntimeService(failure_case.database, no_runner, EventBus(),
                             artifacts_root=tmp_path / "artifacts")
    app = CoreApp()  # In-process handler only: no server, daemon or real provider.
    app._runtime = runtime
    app._event_hub = failure_case.hub
    monkeypatch.setattr(failure_case.hub, "flush", observed_flush)
    try:
        caller = await failure_case.begin(
            app._session_get_handler({"session_id": "sess-1"}), "session.get caller",
        )
        await failure_case.signal(flush_entered)
        failure_case.hub.release[0].set()
        await _assert_finished(current, caller)
        for task in (current, caller):
            error = _assert_hub_error(task, IntegrityError)
            assert "W02 injected SQLite failure" in str(error.__cause__)
        async with failure_case.database.session() as session:
            assert await session.scalar(select(func.count(EventRecord.cursor))) == 0
            assert await session.scalar(select(func.count(SessionRecord.id))) == 1
            assert await session.scalar(select(func.count(RunRecord.id))) == 1
        await failure_case.stop(IntegrityError)
    finally:
        await runtime.shutdown()
