from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from tars_agent.core.bus.envelope import EventOverflowEnvelope
from tars_agent.core.events.durable import (
    DurableEventHub,
    DurableSubscription,
    ReplayLimitReached,
    SubscriptionClosed,
    SubscriptionOverflow,
)
from tars_agent.core.trace.record import TraceRecord
from tars_agent.core.trace.writer import TraceWriter
from tars_agent.core.transport.socket_server import ConnectionSender

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class IpcEventBroadcaster:
    """把 DurableEventHub 订阅转换为单写协程上的 IPC 推送。"""

    def __init__(self, hub: DurableEventHub, trace: TraceWriter | None = None) -> None:
        self._hub = hub
        self._trace = trace
        self._pumps: dict[ConnectionSender, dict[str, asyncio.Task[None]]] = {}

    async def subscribe(
        self,
        sender: ConnectionSender,
        topics: list[str],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        after_cursor: int = 0,
    ) -> tuple[str, int, int, bool]:
        subscription = await self._hub.subscribe(
            topics=topics,
            session_id=session_id,
            run_id=run_id,
            after_cursor=after_cursor,
        )
        task = asyncio.create_task(
            self._pump(sender, subscription),
            name=f"ipc-event-pump:{subscription.id}",
        )
        self._pumps.setdefault(sender, {})[subscription.id] = task
        def callback(_task: asyncio.Task[None]) -> None:
            self._pump_finished(sender, subscription.id)

        task.add_done_callback(callback)
        return (
            subscription.id,
            subscription.replayed_count,
            subscription.high_water_cursor,
            subscription.replay_truncated,
        )

    async def unsubscribe(self, sender: ConnectionSender) -> None:
        pumps = self._pumps.pop(sender, {})
        for subscription_id, task in pumps.items():
            self._hub.unsubscribe(subscription_id)
            task.cancel()
        if pumps:
            await asyncio.gather(*pumps.values(), return_exceptions=True)

    async def stop(self) -> None:
        for sender in list(self._pumps):
            await self.unsubscribe(sender)

    async def _pump(
        self,
        sender: ConnectionSender,
        subscription: DurableSubscription,
    ) -> None:
        subscription_id = subscription.id
        last_cursor = subscription.next_cursor
        try:
            while True:
                try:
                    envelope = await subscription.get()
                except ReplayLimitReached as exc:
                    await self._send_overflow(
                        sender,
                        reason="replay_limit",
                        last_cursor=exc.next_cursor,
                    )
                    await sender.close()
                    return
                except SubscriptionOverflow:
                    await self._send_overflow(
                        sender,
                        reason="subscriber_queue_full",
                        last_cursor=last_cursor,
                    )
                    await sender.close()
                    return
                except SubscriptionClosed:
                    return

                await sender.send(envelope)
                last_cursor = envelope.cursor
                if self._trace is not None:
                    self._trace.emit(
                        TraceRecord(
                            ts=_now(),
                            direction="CORE→CLIENT",
                            layer="ipc",
                            kind="push",
                            run_id=envelope.run_id,
                            client_id=sender.client_id,
                            data={
                                "sub_id": subscription_id,
                                "event_type": envelope.event.get("type"),
                                "cursor": envelope.cursor,
                            },
                        )
                    )
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("connection closed for subscription %s", subscription_id)
        finally:
            self._hub.unsubscribe(subscription_id)

    def _pump_finished(self, sender: ConnectionSender, subscription_id: str) -> None:
        pumps = self._pumps.get(sender)
        if pumps is None:
            return
        pumps.pop(subscription_id, None)
        if not pumps:
            self._pumps.pop(sender, None)

    @staticmethod
    async def _send_overflow(
        sender: ConnectionSender,
        *,
        reason: str,
        last_cursor: int,
    ) -> None:
        try:
            await asyncio.wait_for(
                sender.send(
                    EventOverflowEnvelope(
                        reason=reason,  # type: ignore[arg-type]
                        last_cursor=last_cursor,
                    ),
                    wait=True,
                ),
                timeout=1.0,
            )
        except (TimeoutError, ConnectionError, OSError):
            logger.debug("unable to deliver overflow marker to %s", sender.client_id)
