from __future__ import annotations

import asyncio
import fnmatch
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC
from typing import Final

from pydantic import BaseModel

from tars_agent.core.bus.envelope import EventPushEnvelope
from tars_agent.core.persistence import (
    EVENT_SCHEMA_VERSION,
    Database,
    EventRecord,
    StateRepository,
)

logger = logging.getLogger(__name__)

PERSISTENCE_QUEUE_CAPACITY: Final = 4_096
SUBSCRIPTION_QUEUE_CAPACITY: Final = 512
MAX_REPLAY_EVENTS: Final = 2_000
MAX_BATCH_SIZE: Final = 64
BATCH_INTERVAL_S: Final = 0.020


class DurableEventHubError(RuntimeError):
    pass


class SubscriptionClosed(DurableEventHubError):
    pass


class SubscriptionOverflow(DurableEventHubError):
    pass


class ReplayLimitReached(DurableEventHubError):
    def __init__(self, next_cursor: int) -> None:
        super().__init__("event replay limit reached")
        self.next_cursor = next_cursor


@dataclass(slots=True)
class DurableSubscription:
    id: str
    topics: tuple[str, ...]
    session_id: str | None
    run_id: str | None
    high_water_cursor: int
    replayed_count: int = 0
    replay_truncated: bool = False
    next_cursor: int = 0
    _replay: deque[EventPushEnvelope] = field(default_factory=deque)
    _live: asyncio.Queue[EventPushEnvelope | object] = field(
        default_factory=lambda: asyncio.Queue(maxsize=SUBSCRIPTION_QUEUE_CAPACITY)
    )
    _closed: bool = False
    _overflowed: bool = False

    def matches(self, envelope: EventPushEnvelope) -> bool:
        event_type = str(envelope.event.get("type", ""))
        if not any(fnmatch.fnmatch(event_type, pattern) for pattern in self.topics):
            return False
        if self.session_id is not None and envelope.session_id != self.session_id:
            return False
        return self.run_id is None or envelope.run_id == self.run_id

    def offer(self, envelope: EventPushEnvelope) -> None:
        if self._closed or self._overflowed or not self.matches(envelope):
            return
        try:
            self._live.put_nowait(envelope)
        except asyncio.QueueFull:
            self._overflowed = True
            self._replace_live_queue(_OVERFLOW)

    def set_replay(
        self,
        envelopes: list[EventPushEnvelope],
        *,
        truncated: bool,
        next_cursor: int,
    ) -> None:
        self._replay.extend(envelopes)
        self.replayed_count = len(envelopes)
        self.replay_truncated = truncated
        self.next_cursor = next_cursor

    async def get(self) -> EventPushEnvelope:
        if self._closed:
            raise SubscriptionClosed("event subscription is closed")
        if self._replay:
            return self._replay.popleft()
        if self.replay_truncated:
            self.replay_truncated = False
            raise ReplayLimitReached(self.next_cursor)
        if self._overflowed:
            raise SubscriptionOverflow("event subscription queue overflowed")
        item = await self._live.get()
        if item is _OVERFLOW:
            raise SubscriptionOverflow("event subscription queue overflowed")
        if item is _CLOSED:
            raise SubscriptionClosed("event subscription is closed")
        assert isinstance(item, EventPushEnvelope)
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._replay.clear()
        self._replace_live_queue(_CLOSED)

    def _replace_live_queue(self, marker: object) -> None:
        while True:
            try:
                self._live.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._live.put_nowait(marker)


@dataclass(slots=True)
class _PendingEvent:
    event: BaseModel


@dataclass(slots=True)
class _FlushBarrier:
    future: asyncio.Future[None]


_STOP = object()
_OVERFLOW = object()
_CLOSED = object()


class DurableEventHub:
    """事件持久化真源：先分配 SQLite cursor，再向有界订阅队列扇出。"""

    def __init__(
        self,
        database: Database,
        *,
        queue_capacity: int = PERSISTENCE_QUEUE_CAPACITY,
        batch_size: int = MAX_BATCH_SIZE,
        batch_interval_s: float = BATCH_INTERVAL_S,
    ) -> None:
        self._database = database
        self._queue: asyncio.Queue[_PendingEvent | _FlushBarrier | object] = asyncio.Queue(
            maxsize=queue_capacity
        )
        self._batch_size = batch_size
        self._batch_interval_s = batch_interval_s
        self._subscriptions: dict[str, DurableSubscription] = {}
        self._worker: asyncio.Task[None] | None = None
        self._latest_cursor = 0
        self._fatal_error: BaseException | None = None
        self._stopping = False

    @property
    def latest_cursor(self) -> int:
        return self._latest_cursor

    @property
    def subscription_count(self) -> int:
        return len(self._subscriptions)

    async def start(self) -> None:
        if self._worker is not None:
            return
        async with self._database.session() as db_session:
            repository = StateRepository(db_session)
            self._latest_cursor = await repository.latest_event_cursor()
        self._stopping = False
        self._worker = asyncio.create_task(self._run(), name="durable-event-writer")

    async def handle(self, event: BaseModel) -> None:
        if self._fatal_error is not None:
            raise DurableEventHubError("event persistence worker failed") from self._fatal_error
        if self._worker is None or self._stopping:
            raise DurableEventHubError("event hub is not running")
        await self._queue.put(_PendingEvent(event))

    async def flush(self) -> None:
        if self._fatal_error is not None:
            raise DurableEventHubError("event persistence worker failed") from self._fatal_error
        if self._worker is None:
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._queue.put(_FlushBarrier(future))
        await future

    async def subscribe(
        self,
        *,
        topics: list[str],
        session_id: str | None = None,
        run_id: str | None = None,
        after_cursor: int = 0,
        queue_capacity: int = SUBSCRIPTION_QUEUE_CAPACITY,
    ) -> DurableSubscription:
        if self._worker is None or self._stopping:
            raise DurableEventHubError("event hub is not running")
        if not topics:
            raise ValueError("topics must not be empty")
        if session_id is not None and run_id is not None:
            raise ValueError("session_id and run_id are mutually exclusive scopes")

        high_water = self._latest_cursor
        subscription = DurableSubscription(
            id=f"sub-{uuid.uuid4().hex[:12]}",
            topics=tuple(topics),
            session_id=session_id,
            run_id=run_id,
            high_water_cursor=high_water,
            next_cursor=after_cursor,
            _live=asyncio.Queue(maxsize=queue_capacity),
        )
        # 注册发生在首次 await 之前；之后落库的 cursor 会进入 live 队列，因此回放与实时无缝衔接。
        self._subscriptions[subscription.id] = subscription
        try:
            records = await self._load_replay(
                after_cursor=after_cursor,
                high_water=high_water,
                session_id=session_id,
                run_id=run_id,
            )
        except BaseException:
            self.unsubscribe(subscription.id)
            raise

        truncated = len(records) > MAX_REPLAY_EVENTS
        scanned = records[:MAX_REPLAY_EVENTS]
        envelopes = [self._from_record(record) for record in scanned]
        envelopes = [envelope for envelope in envelopes if subscription.matches(envelope)]
        next_cursor = scanned[-1].cursor if scanned else after_cursor
        subscription.set_replay(
            envelopes,
            truncated=truncated,
            next_cursor=next_cursor,
        )
        return subscription

    def unsubscribe(self, subscription_id: str) -> None:
        subscription = self._subscriptions.pop(subscription_id, None)
        if subscription is not None:
            subscription.close()

    async def stop(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._stopping = True
        if self._fatal_error is None:
            await self.flush()
            await self._queue.put(_STOP)
        await asyncio.gather(worker, return_exceptions=True)
        self._worker = None
        for subscription_id in list(self._subscriptions):
            self.unsubscribe(subscription_id)

    async def _run(self) -> None:
        stop_requested = False
        try:
            while not stop_requested:
                item = await self._queue.get()
                if item is _STOP:
                    self._queue.task_done()
                    break
                if isinstance(item, _FlushBarrier):
                    if not item.future.done():
                        item.future.set_result(None)
                    self._queue.task_done()
                    continue

                assert isinstance(item, _PendingEvent)
                batch = [item]
                deadline = asyncio.get_running_loop().time() + self._batch_interval_s
                barrier: _FlushBarrier | None = None
                while len(batch) < self._batch_size:
                    timeout = deadline - asyncio.get_running_loop().time()
                    if timeout <= 0:
                        break
                    try:
                        next_item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    except TimeoutError:
                        break
                    if next_item is _STOP:
                        self._queue.task_done()
                        stop_requested = True
                        break
                    if isinstance(next_item, _FlushBarrier):
                        barrier = next_item
                        break
                    assert isinstance(next_item, _PendingEvent)
                    batch.append(next_item)

                await self._persist_and_fan_out(batch)
                for _ in batch:
                    self._queue.task_done()
                if barrier is not None:
                    if not barrier.future.done():
                        barrier.future.set_result(None)
                    self._queue.task_done()
        except BaseException as exc:
            self._fatal_error = exc
            logger.exception("durable event persistence worker failed")
            self._fail_barriers(exc)
            for subscription in self._subscriptions.values():
                subscription._overflowed = True
                subscription._replace_live_queue(_OVERFLOW)
            if isinstance(exc, asyncio.CancelledError):
                raise

    def _fail_barriers(self, exc: BaseException) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(item, _FlushBarrier) and not item.future.done():
                item.future.set_exception(exc)
            self._queue.task_done()

    async def _persist_and_fan_out(self, batch: list[_PendingEvent]) -> None:
        payloads = [item.event.model_dump(mode="json") for item in batch]
        run_ids = {
            str(payload["run_id"])
            for payload in payloads
            if payload.get("run_id") is not None
        }
        parent_run_ids = {
            str(payload["parent_run_id"])
            for payload in payloads
            if payload.get("parent_run_id") is not None
        }
        lookup_ids = run_ids | parent_run_ids

        async with self._database.transaction() as db_session:
            repository = StateRepository(db_session)
            run_sessions = await repository.resolve_run_sessions(lookup_ids)

            records: list[EventRecord] = []
            envelope_run_ids: list[str | None] = []
            envelope_session_ids: list[str | None] = []
            for payload in payloads:
                payload_run_id = _optional_string(payload.get("run_id"))
                parent_run_id = _optional_string(payload.get("parent_run_id"))
                session_id = _optional_string(payload.get("session_id"))
                if session_id is None and payload_run_id is not None:
                    session_id = run_sessions.get(payload_run_id)
                if session_id is None and parent_run_id is not None:
                    session_id = run_sessions.get(parent_run_id)
                stored_run_id = payload_run_id if payload_run_id in run_sessions else None
                record = EventRecord(
                    event_schema_version=EVENT_SCHEMA_VERSION,
                    session_id=session_id,
                    run_id=stored_run_id,
                    event_type=str(payload.get("type", "unknown")),
                    payload=payload,
                )
                records.append(record)
                envelope_run_ids.append(payload_run_id)
                envelope_session_ids.append(session_id)
            await repository.add_events(records)

        envelopes = [
            self._from_record(
                record,
                run_id=envelope_run_ids[index],
                session_id=envelope_session_ids[index],
            )
            for index, record in enumerate(records)
        ]
        if envelopes:
            self._latest_cursor = envelopes[-1].cursor
        for envelope in envelopes:
            for subscription in tuple(self._subscriptions.values()):
                subscription.offer(envelope)

    async def _load_replay(
        self,
        *,
        after_cursor: int,
        high_water: int,
        session_id: str | None,
        run_id: str | None,
    ) -> list[EventRecord]:
        async with self._database.session() as db_session:
            repository = StateRepository(db_session)
            return list(
                await repository.list_events(
                    after_cursor=after_cursor,
                    through_cursor=high_water,
                    session_id=session_id,
                    run_id=run_id,
                    limit=MAX_REPLAY_EVENTS + 1,
                )
            )

    @staticmethod
    def _from_record(
        record: EventRecord,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> EventPushEnvelope:
        occurred_at = record.created_at
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)
        occurred_at_text = record.payload.get("ts")
        if not isinstance(occurred_at_text, str) or not occurred_at_text:
            occurred_at_text = occurred_at.astimezone(UTC).isoformat()
        return EventPushEnvelope(
            event_schema_version=record.event_schema_version,
            cursor=record.cursor,
            session_id=record.session_id if session_id is None else session_id,
            run_id=record.run_id if run_id is None else run_id,
            occurred_at=occurred_at_text,
            event=dict(record.payload),
        )


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
