from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anthropic
import httpx

from tars_agent.core.bus.events import LlmModelSelectedEvent, LlmTokenEvent, LlmUsageEvent
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.budget import BudgetTransport, ModelRequestBudgetExceeded, RequestLedger
from tars_agent.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from tars_agent.core.paths import tars_home

if TYPE_CHECKING:
    from tars_agent.core.config import LlmConfig

_MODEL_CONTEXT_WINDOWS = {
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-opus-4-7": 200_000,
}
_OFFICIAL_BASE_URL = "https://api.anthropic.com"
log = logging.getLogger(__name__)
_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Use the available tools to complete the user's goal. "
    "When the goal is fully achieved, respond with a final answer and do not call any more tools."
)


class ProviderConfigurationError(ValueError):
    pass


class LlmProtocolError(RuntimeError):
    pass


class LlmCallTimeoutError(TimeoutError):
    pass


class LlmStreamInterruptedError(RuntimeError):
    """Once any stream event arrives, the request must never be replayed transparently."""

    def __init__(self, message: str, *, partial_text: str = "") -> None:
        super().__init__(message)
        self.partial_text = partial_text


def _context_window(model: str) -> int:
    return _MODEL_CONTEXT_WINDOWS.get(model, 200_000)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return isinstance(exc, (anthropic.APIConnectionError, httpx.TransportError))


class AnthropicProvider:
    def __init__(
        self, model: str, client: Any = None, *, max_tokens: int = 8192,
        api_key: str = "", base_url: str = "", anthropic_api_key: str = "",
        total_timeout_s: float = 120.0, connect_timeout_s: float = 10.0,
        read_timeout_s: float = 60.0, write_timeout_s: float = 10.0,
        pool_timeout_s: float = 10.0, attempts: int = 2, retry_delay_s: float = 1.0,
        request_budget_path: Path | None = None, request_limit: int | None = 100,
    ) -> None:
        if not model.strip() or attempts not in (1, 2):
            raise ProviderConfigurationError("Model must be nonempty and attempts must be 1 or 2")
        for value in (total_timeout_s, connect_timeout_s, read_timeout_s, write_timeout_s,
                      pool_timeout_s, retry_delay_s):
            if not math.isfinite(value) or value <= 0:
                raise ProviderConfigurationError("Model timeouts and retry delay must be positive")
        if request_limit is not None and (type(request_limit) is not int or request_limit <= 0):
            raise ProviderConfigurationError("Request limit must be a positive integer or None")
        self._request_limit = request_limit
        self._model = model.strip()
        self._max_tokens = max_tokens
        self._total_timeout_s = total_timeout_s
        self._attempts = attempts
        self._retry_delay_s = retry_delay_s
        self._owns_client = client is None
        if client is None:
            dedicated_key = api_key or os.environ.get("TARS_LLM_API_KEY", "")
            endpoint = base_url or _OFFICIAL_BASE_URL
            if base_url:
                key = dedicated_key
                if not key:
                    raise ProviderConfigurationError(
                        "Custom endpoint requires dedicated TARS_LLM_API_KEY"
                    )
            else:
                key = dedicated_key or anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
                if not key:
                    raise ProviderConfigurationError("LLM API key not set")
            ledger = RequestLedger(
                request_budget_path or tars_home() / "acceptance" / "request-budget.sqlite3",
                limit=request_limit,
            )
            timeout = httpx.Timeout(
                connect=connect_timeout_s, read=read_timeout_s,
                write=write_timeout_s, pool=pool_timeout_s,
            )
            transport = BudgetTransport(httpx.AsyncHTTPTransport(retries=0), ledger)
            http_client = httpx.AsyncClient(
                transport=transport, timeout=timeout, follow_redirects=False, trust_env=False,
            )
            self._client: Any = anthropic.AsyncAnthropic(
                api_key=key, base_url=endpoint, max_retries=0,
                timeout=timeout, http_client=http_client,
            )
        else:
            # Injection is for deterministic tests. Production callers use from_config.
            self._client = client

    @classmethod
    def from_config(cls, config: LlmConfig) -> AnthropicProvider:
        return cls(
            config.default_model, max_tokens=config.max_tokens, api_key=config.api_key,
            base_url=config.base_url, anthropic_api_key=config.anthropic_api_key,
            total_timeout_s=config.total_timeout_s, connect_timeout_s=config.connect_timeout_s,
            read_timeout_s=config.read_timeout_s, write_timeout_s=config.write_timeout_s,
            pool_timeout_s=config.pool_timeout_s, attempts=config.attempts,
            retry_delay_s=config.retry_delay_s, request_budget_path=config.request_budget_path,
            request_limit=config.request_limit,
        )

    def with_model(self, model: str) -> AnthropicProvider:
        return AnthropicProvider(
            model, self._client, max_tokens=self._max_tokens,
            total_timeout_s=self._total_timeout_s, attempts=self._attempts,
            retry_delay_s=self._retry_delay_s, request_limit=self._request_limit,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()
            self._owns_client = False

    async def chat(
        self, messages: list[dict[str, object]], tool_schemas: list[dict[str, object]],
        bus: EventBus, run_id: str, *, step: int = 0, system: str | None = None,
    ) -> LlmResponse:
        try:
            async with asyncio.timeout(self._total_timeout_s):
                return await self._chat(messages, tool_schemas, bus, run_id, step, system)
        except TimeoutError as exc:
            raise LlmCallTimeoutError("llm_total_timeout") from exc

    async def _chat(
        self, messages: list[dict[str, object]], tool_schemas: list[dict[str, object]],
        bus: EventBus, run_id: str, step: int, system: str | None,
    ) -> LlmResponse:
        await bus.publish(LlmModelSelectedEvent(
            run_id=run_id, model=self._model, ts=_now(),
        ))
        tools = list(tool_schemas)
        if tools:
            tools = tools[:-1] + [{**tools[-1], "cache_control": {"type": "ephemeral"}}]
        kwargs: dict[str, Any] = {
            "model": self._model, "max_tokens": self._max_tokens,
            "system": [{"type": "text", "text": system or _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        for attempt in range(1, self._attempts + 1):
            received_event = False
            text_parts: list[str] = []
            message_started = False
            message_stopped = False
            opened: set[int] = set()
            completed: set[int] = set()
            block_types: dict[int, str] = {}
            json_parts: dict[int, list[str]] = {}
            try:
                async with self._client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        received_event = True
                        kind = event.type
                        # MessageStream yields convenience text events as well as raw deltas.
                        if kind in ("text", "input_json", "thinking", "signature", "citation"):
                            continue
                        if kind == "message_start":
                            if message_started or message_stopped:
                                raise LlmProtocolError("duplicate message_start")
                            message_started = True
                        elif kind == "content_block_start":
                            index = event.index
                            if (
                                not message_started or message_stopped
                                or index in opened | completed
                                or index != len(opened) + len(completed)
                            ):
                                raise LlmProtocolError("invalid content block start")
                            opened.add(index)
                            block_types[index] = event.content_block.type
                            json_parts[index] = []
                        elif kind == "content_block_delta":
                            if event.index not in opened or message_stopped:
                                raise LlmProtocolError("delta outside an open content block")
                            if event.delta.type == "input_json_delta":
                                if block_types[event.index] != "tool_use":
                                    raise LlmProtocolError("tool delta in non-tool block")
                                json_parts[event.index].append(event.delta.partial_json)
                            if event.delta.type == "text_delta":
                                if block_types[event.index] != "text":
                                    raise LlmProtocolError("text delta belongs to a non-text block")
                                text = event.delta.text
                                text_parts.append(text)
                                await bus.publish(
                                    LlmTokenEvent(run_id=run_id, token=text, ts=_now())
                                )
                        elif kind == "content_block_stop":
                            if event.index not in opened or message_stopped:
                                raise LlmProtocolError("invalid content block stop")
                            if json_parts[event.index]:
                                try:
                                    tool_input = json.loads("".join(json_parts[event.index]))
                                except ValueError as exc:
                                    raise LlmProtocolError(
                                        "invalid or incomplete tool JSON"
                                    ) from exc
                                if not isinstance(tool_input, dict):
                                    raise LlmProtocolError("tool input must be an object")
                            opened.remove(event.index)
                            completed.add(event.index)
                        elif kind == "message_stop":
                            if not message_started or message_stopped or opened:
                                raise LlmProtocolError("message stopped with incomplete content")
                            message_stopped = True
                        elif kind == "message_delta":
                            if not message_started or message_stopped:
                                raise LlmProtocolError("message delta outside a message")
                        elif kind == "error":
                            raise LlmProtocolError("model stream error event")
                    if not message_stopped or opened:
                        raise LlmProtocolError("model stream ended before message_stop")
                    final = await stream.get_final_message()
                    if len(final.content) != len(completed):
                        raise LlmProtocolError("final content does not match completed blocks")
                response = self._response(final, "".join(text_parts))
                usage = response.usage
                assert usage is not None
                await bus.publish(LlmUsageEvent(
                    run_id=run_id, input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                    cache_creation_input_tokens=usage.cache_creation_input_tokens,
                    context_pct=usage.context_pct, ts=_now(),
                ))
                return response
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                cause: BaseException | None = exc
                while cause is not None:
                    if isinstance(cause, ModelRequestBudgetExceeded):
                        raise cause
                    cause = cause.__cause__
                if received_event and _retryable(exc):
                    raise LlmStreamInterruptedError(
                        "llm_stream_interrupted", partial_text="".join(text_parts),
                    ) from exc
                if received_event or not _retryable(exc) or attempt >= self._attempts:
                    raise
                log.warning(
                    "Model transport failed before first event; attempt %d/%d",
                    attempt, self._attempts,
                )
                await asyncio.sleep(self._retry_delay_s)
        raise AssertionError("unreachable attempt loop")

    def _response(self, final: Any, text: str) -> LlmResponse:
        tool_calls: list[ToolCallBlock] = []
        thinking: list[dict[str, object]] = []
        final_text: list[str] = []
        identifiers: set[str] = set()
        for block in final.content:
            if block.type == "tool_use":
                if (not isinstance(block.id, str) or not block.id
                        or block.id in identifiers or not isinstance(block.name, str)
                        or not block.name.strip() or not isinstance(block.input, dict)):
                    raise LlmProtocolError("invalid or duplicate tool call")
                identifiers.add(block.id)
                tool_calls.append(ToolCallBlock(block.id, block.name, block.input))
            elif block.type == "text":
                final_text.append(block.text)
            elif block.type == "thinking":
                thinking.append({"type": "thinking", "thinking": block.thinking,
                                 "signature": block.signature})
            elif block.type == "redacted_thinking":
                thinking.append({"type": "redacted_thinking", "data": block.data})
            else:
                raise LlmProtocolError(f"unsupported content block: {block.type}")
        if "".join(final_text) != text:
            raise LlmProtocolError("stream text does not match final message")
        reason = final.stop_reason
        if reason is None and text.strip() and not tool_calls:
            log.warning("Complete text message omitted stop_reason; accepting end_turn")
            reason = "end_turn"
        if reason == "tool_use" and not tool_calls:
            raise LlmProtocolError("tool_use response did not contain a tool call")
        if reason == "end_turn" and (not text.strip() or tool_calls):
            raise LlmProtocolError("end_turn must contain nonempty text and no tools")
        if not isinstance(reason, str) or not reason:
            raise LlmProtocolError("missing or invalid stop_reason")
        usage = final.usage
        values = [
            getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None),
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
        ]
        if any(type(value) is not int or value < 0 for value in values):
            raise LlmProtocolError("invalid usage counters")
        counts = cast(list[int], values)
        input_tokens, output_tokens, cache_read, cache_create = counts
        context_pct = min(1.0, sum(counts) / _context_window(self._model))
        return LlmResponse(
            stop_reason=reason, text=text, tool_calls=tool_calls, thinking_blocks=thinking,
            usage=UsageStats(input_tokens, output_tokens, cache_read, cache_create, context_pct),
        )
