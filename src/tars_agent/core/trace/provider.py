from __future__ import annotations

import dataclasses
import logging
import time
from datetime import UTC, datetime
from typing import Any, cast

from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.base import LLMProvider
from tars_agent.core.llm.types import LlmResponse
from tars_agent.core.trace.record import TraceRecord
from tars_agent.core.trace.writer import TraceWriter


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TracingProvider:
    # 包裹真实 LLMProvider，在每次 chat() 调用前后向 TraceWriter 写入完整 API I/O 记录
    def __init__(
        self,
        inner: LLMProvider,
        trace: TraceWriter,
        *,
        include_payload: bool = False,
    ) -> None:
        self._inner = inner
        self._trace = trace
        self._include_payload = include_payload

    def with_model(self, model: str) -> TracingProvider:
        clone = getattr(self._inner, "with_model", None)
        if not callable(clone):
            raise ValueError("Provider cannot apply the requested profile model")
        return TracingProvider(
            cast(LLMProvider, clone(model)), self._trace, include_payload=self._include_payload,
        )

    async def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            await close()

    def _emit(self, record: TraceRecord) -> None:
        try:
            self._trace.emit(record)
        except Exception:
            logging.getLogger(__name__).exception("Trace observer failed; model call continues")

    # 记录 CORE→LLM 请求，调用真实 provider，记录 LLM→CORE 响应（含延迟）
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
        call_data: dict[str, Any]
        if self._include_payload:
            call_data = {"messages": messages, "tool_schemas": tool_schemas, "system": system}
        else:
            call_data = {
                "message_count": len(messages),
                "tool_count": len(tool_schemas),
            }

        self._emit(
            TraceRecord(
                ts=_now(),
                direction="CORE→LLM",
                layer="llm",
                kind="api_call",
                run_id=run_id,
                step=step,
                data=call_data,
            )
        )

        t0 = time.monotonic()
        result = await self._inner.chat(
            messages, tool_schemas, bus, run_id, step=step, system=system
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        resp_data: dict[str, Any]
        if self._include_payload:
            resp_data = {
                "stop_reason": result.stop_reason,
                "text": result.text,
                "tool_calls": [dataclasses.asdict(tc) for tc in result.tool_calls],
                "usage": dataclasses.asdict(result.usage) if result.usage else {},
                "latency_ms": latency_ms,
            }
        else:
            resp_data = {
                "stop_reason": result.stop_reason,
                "usage": dataclasses.asdict(result.usage) if result.usage else {},
                "latency_ms": latency_ms,
            }

        self._emit(
            TraceRecord(
                ts=_now(),
                direction="LLM→CORE",
                layer="llm",
                kind="api_response",
                run_id=run_id,
                step=step,
                data=resp_data,
            )
        )

        return result
