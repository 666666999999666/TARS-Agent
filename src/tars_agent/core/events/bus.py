from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel

type EventHandler = Callable[[BaseModel], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class EventSubscription:
    """可显式释放的 EventBus 订阅句柄。"""

    _bus: EventBus
    _subscription_id: int

    def unsubscribe(self) -> None:
        self._bus.unsubscribe(self)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[int, EventHandler] = {}
        self._next_subscription_id = 1

    # 注册事件处理函数并返回可释放句柄
    def subscribe(self, handler: EventHandler) -> EventSubscription:
        subscription_id = self._next_subscription_id
        self._next_subscription_id += 1
        self._subscribers[subscription_id] = handler
        return EventSubscription(self, subscription_id)

    # 幂等移除订阅，已经释放的句柄再次释放不会报错
    def unsubscribe(self, subscription: EventSubscription) -> None:
        if subscription._bus is not self:
            raise ValueError("subscription belongs to another EventBus")
        self._subscribers.pop(subscription._subscription_id, None)

    # 按注册顺序依次调用所有订阅者
    async def publish(self, event: BaseModel) -> None:
        for handler in tuple(self._subscribers.values()):
            await handler(event)
