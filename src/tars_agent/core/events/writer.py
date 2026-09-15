from __future__ import annotations

import logging
from pathlib import Path
from typing import IO

from pydantic import BaseModel

from tars_agent.core.events.bus import EventBus, EventSubscription

logger = logging.getLogger(__name__)


class EventWriter:
    def __init__(self, path: Path, *, run_id: str) -> None:
        self._path = path
        self._run_id = run_id
        self._file: IO[str] | None = None
        self._subscription: EventSubscription | None = None

    # 打开事件文件（追加模式），供 async with 使用
    async def __aenter__(self) -> EventWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")
        return self

    # 关闭事件文件
    async def __aexit__(self, *args: object) -> None:
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None
        if self._file is not None:
            self._file.close()
            self._file = None

    # 将事件序列化为 JSON 行并写入文件，写入失败时记录日志但不抛出异常
    async def handle(self, event: BaseModel) -> None:
        if self._file is None or getattr(event, "run_id", None) != self._run_id:
            return
        try:
            self._file.write(event.model_dump_json() + "\n")
            self._file.flush()
        except (OSError, ValueError) as e:
            logger.error("EventWriter: failed to write event: %s", e)

    # 将 handle 注册为 bus 的订阅者
    def subscribe(self, bus: EventBus) -> None:
        if self._subscription is not None:
            self._subscription.unsubscribe()
        self._subscription = bus.subscribe(self.handle)
