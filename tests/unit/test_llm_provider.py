from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import BaseModel

from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.provider import (
    AnthropicProvider,
    LlmProtocolError,
    LlmStreamInterruptedError,
    ProviderConfigurationError,
)
from tars_agent.core.llm.types import LlmResponse

# --- helpers -----------------------------------------------------------------


def _make_usage(
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_create: int = 0,
) -> MagicMock:
    u = MagicMock()
    u.input_tokens = input_tokens
    u.output_tokens = output_tokens
    u.cache_read_input_tokens = cache_read
    u.cache_creation_input_tokens = cache_create
    return u


def _make_final(
    stop_reason: str = "end_turn",
    content: list[MagicMock] | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
    cache_create: int = 0,
) -> MagicMock:
    msg = MagicMock()
    msg.stop_reason = stop_reason
    msg.content = content if content is not None else [SimpleNamespace(type="text", text="ok")]
    msg.usage = _make_usage(input_tokens, output_tokens, cache_read, cache_create)
    return msg


class FakeStream:
    """Minimal async context manager that fakes the anthropic streaming interface."""

    def __init__(self, texts: list[str], final: MagicMock) -> None:
        self._texts = texts
        self._final = final

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def _texts_iter(self):
        for text in self._texts:
            yield text

    def __aiter__(self):
        async def events():
            # A pre-stream failure occurs before message_start, not merely before text.
            iterator = self._texts_iter()
            try:
                first = await anext(iterator)
            except StopAsyncIteration:
                first = None
            yield SimpleNamespace(type="message_start")
            content = self._final.content
            for index, block in enumerate(content):
                yield SimpleNamespace(type="content_block_start", index=index, content_block=block)
                if block.type == "text":
                    texts = []
                    if first is not None:
                        texts.append(first)
                        yield SimpleNamespace(type="content_block_delta", index=index,
                                              delta=SimpleNamespace(type="text_delta", text=first))
                    async for text in iterator:
                        texts.append(text)
                        yield SimpleNamespace(type="content_block_delta", index=index,
                                              delta=SimpleNamespace(type="text_delta", text=text))
                    # Tests supply final message content, as the SDK would accumulate it.
                    if not texts and block.text:
                        yield SimpleNamespace(type="content_block_delta", index=index,
                                              delta=SimpleNamespace(type="text_delta", text=block.text))
                    elif texts:
                        block.text = "".join(texts)
                yield SimpleNamespace(type="content_block_stop", index=index)
            yield SimpleNamespace(type="message_delta")
            yield SimpleNamespace(type="message_stop")
        return events()

    async def get_final_message(self) -> MagicMock:
        return self._final


class FailingStream(FakeStream):
    def __init__(
        self,
        texts: list[str],
        final: MagicMock,
        *,
        fail_after_tokens: int,
    ) -> None:
        super().__init__(texts, final)
        self._fail_after_tokens = fail_after_tokens

    async def _texts_iter(self):
        for index, text in enumerate(self._texts):
            if index == self._fail_after_tokens:
                raise httpx.ReadError("stream dropped")
            yield text
        if self._fail_after_tokens >= len(self._texts):
            raise httpx.ReadError("stream dropped")


def _make_provider(
    texts: list[str] | None = None,
    stop_reason: str = "end_turn",
    content: list[MagicMock] | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
) -> tuple[AnthropicProvider, MagicMock]:
    if texts == [] and content is None:
        content = [SimpleNamespace(type="text", text="")]
    final = _make_final(stop_reason, content, input_tokens, output_tokens, cache_read)
    client = MagicMock()
    client.messages.stream.return_value = FakeStream(texts or [], final)
    return AnthropicProvider(model="test-model", client=client), client


async def _chat(
    provider: AnthropicProvider,
    messages: list[dict[str, object]] | None = None,
    tool_schemas: list[dict[str, object]] | None = None,
) -> tuple[LlmResponse, list[BaseModel]]:
    collected: list[BaseModel] = []
    bus = EventBus()

    async def _collect(e: BaseModel) -> None:
        collected.append(e)

    bus.subscribe(_collect)
    result = await provider.chat(
        messages=messages or [],
        tool_schemas=tool_schemas or [],
        bus=bus,
        run_id="r1",
    )
    return result, collected


# --- tests -------------------------------------------------------------------


# 每次模型调用发布模型名，供运行记录和评测核对实际使用的模型。
async def test_model_selected_event_published() -> None:
    provider, _ = _make_provider()
    _, events = await _chat(provider)
    sel = [e for e in events if e.type == "llm.model_selected"]  # type: ignore[attr-defined]
    assert len(sel) == 1
    assert sel[0].model == "test-model"  # type: ignore[attr-defined]
    assert sel[0].run_id == "r1"  # type: ignore[attr-defined]


# 功能：验证流式响应的每个 token 触发独立的 llm.token 事件，内容与顺序均正确
# 设计：使用 FakeStream 控制精确的 token 序列，断言数量和各 token 的值，排除批量合并或跳过情况
async def test_token_events_published_per_chunk() -> None:
    provider, _ = _make_provider(texts=["Hello", " world"])
    _, events = await _chat(provider)
    tokens = [e for e in events if e.type == "llm.token"]  # type: ignore[attr-defined]
    assert len(tokens) == 2
    assert tokens[0].token == "Hello"  # type: ignore[attr-defined]
    assert tokens[1].token == " world"  # type: ignore[attr-defined]


# 功能：验证 llm.usage 事件中的 token 统计字段（input、output、cache_read）正确
# 设计：向 FakeStream 注入固定 usage 值，断言三个字段精确匹配，因为这些字段是 S6 成本计算的数据源
async def test_usage_event_published_after_stream() -> None:
    provider, _ = _make_provider(input_tokens=200, output_tokens=75, cache_read=150)
    _, events = await _chat(provider)
    usage_events = [e for e in events if e.type == "llm.usage"]  # type: ignore[attr-defined]
    assert len(usage_events) == 1
    ue = usage_events[0]
    assert ue.input_tokens == 200  # type: ignore[attr-defined]
    assert ue.output_tokens == 75  # type: ignore[attr-defined]
    assert ue.cache_read_input_tokens == 150  # type: ignore[attr-defined]


async def test_context_pct_counts_cached_prompt_and_next_step_output() -> None:
    final = _make_final(
        input_tokens=100,
        output_tokens=1_000,
        cache_read=120_000,
        cache_create=39_000,
    )
    client = MagicMock()
    client.messages.stream.return_value = FakeStream([], final)
    provider = AnthropicProvider(model="test-model", client=client, context_budget_tokens=200_000)

    response, events = await _chat(provider)

    assert response.usage is not None
    assert response.usage.context_pct == pytest.approx(0.8005)
    usage_event = next(event for event in events if event.type == "llm.usage")  # type: ignore[attr-defined]
    assert usage_event.context_pct == pytest.approx(0.8005)  # type: ignore[attr-defined]


async def test_context_pct_is_capped_at_one() -> None:
    final = _make_final(
        input_tokens=200_000,
        output_tokens=8_000,
        cache_read=50_000,
        cache_create=25_000,
    )
    client = MagicMock()
    client.messages.stream.return_value = FakeStream([], final)
    provider = AnthropicProvider(model="test-model", client=client)

    response, _ = await _chat(provider)

    assert response.usage is not None
    assert response.usage.context_pct == 1.0


# 功能：验证事件发布顺序为 model_selected → token（×N） → usage
# 设计：检查类型列表的首尾元素，聚焦 provider 的时序契约，而非中间 token 事件的顺序（那由流本身决定）
async def test_event_order_model_selected_first_usage_last() -> None:
    provider, _ = _make_provider(texts=["hi"])
    _, events = await _chat(provider)
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert types[0] == "llm.model_selected"
    assert "llm.token" in types
    assert types[-1] == "llm.usage"


# 功能：验证 stop_reason=tool_use 时 final_message content block 被正确解析为 ToolCallBlock
# 设计：注入带 tool_use block 的 final_message，逐字段检查 ToolCallBlock 的 id/name/input，确认解析路径完整
async def test_tool_use_parsed_from_final_message() -> None:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_01"
    tool_block.name = "read_file"
    tool_block.input = {"path": "README.md"}
    provider, _ = _make_provider(stop_reason="tool_use", content=[tool_block])
    result, _ = await _chat(provider)
    assert result.stop_reason == "tool_use"
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "toolu_01"
    assert tc.name == "read_file"
    assert tc.input == {"path": "README.md"}


# 功能：验证 stop_reason=end_turn 时不产生任何工具调用
# 设计：与 test_tool_use_parsed 互补，确认 stop_reason 决策树的另一侧，防止 end_turn 响应被误解析为 tool_use
async def test_end_turn_produces_no_tool_calls() -> None:
    provider, _ = _make_provider(stop_reason="end_turn")
    result, _ = await _chat(provider)
    assert result.stop_reason == "end_turn"
    assert result.tool_calls == []


# 功能：验证多个流式 token 被正确拼接为 LlmResponse.text 字段
# 设计：拼接三段 token，核对完整回复，避免最终消息遗漏流式片段。
async def test_text_accumulated_from_tokens() -> None:
    provider, _ = _make_provider(texts=["foo", "bar", "baz"])
    result, _ = await _chat(provider)
    assert result.text == "foobarbaz"


# 功能：验证缺少 ANTHROPIC_API_KEY 时 provider 初始化立即 SystemExit 而非等到调用时才报错
# 设计：用 monkeypatch 清除环境变量后实例化，确认 fail-fast 行为，防止"幽灵 run"（有 started 但无 finished 事件）
async def test_missing_api_key_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TARS_LLM_API_KEY", raising=False)
    with pytest.raises(ProviderConfigurationError):
        AnthropicProvider(model="any")


# 功能：验证空流式响应不发布任何 llm.token 事件且 result.text 为空字符串
# 设计：texts=[] 覆盖零 token 边界条件，确认 text="" 而非 None，避免调用方对空内容做额外 None 判断
async def test_no_tokens_when_response_is_empty() -> None:
    provider, _ = _make_provider(texts=[])
    with pytest.raises(LlmProtocolError, match="nonempty text"):
        await _chat(provider)


# 功能：验证首 token 前连接失败可透明重试，并只发布成功尝试产生的 token
# 设计：第一次 stream 立即 ReadError，第二次返回完整内容，替换 sleep 以避免测试等待
async def test_stream_retries_only_before_first_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _make_final()
    client = MagicMock()
    client.messages.stream.side_effect = [
        FailingStream([], final, fail_after_tokens=0),
        FakeStream(["ok"], final),
    ]
    provider = AnthropicProvider(model="test-model", client=client)
    monkeypatch.setattr("tars_agent.core.llm.provider.asyncio.sleep", AsyncMock())

    result, events = await _chat(provider)

    assert result.text == "ok"
    assert client.messages.stream.call_count == 2
    assert [event.token for event in events if event.type == "llm.token"] == ["ok"]  # type: ignore[attr-defined]


async def test_stream_connection_failure_has_two_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _make_final()
    client = MagicMock()
    client.messages.stream.return_value = FailingStream(
        [], final, fail_after_tokens=0
    )
    provider = AnthropicProvider(model="test-model", client=client)
    sleep = AsyncMock()
    monkeypatch.setattr("tars_agent.core.llm.provider.asyncio.sleep", sleep)

    with pytest.raises(httpx.ReadError, match="stream dropped"):
        await _chat(provider)

    assert client.messages.stream.call_count == 2
    assert [entry.args[0] for entry in sleep.await_args_list] == [1.0]


# 功能：验证已发布 token 后流中断立即失败，不进行第二次 API 调用
# 设计：stream 先产出 partial 再 ReadError，断言专用异常、调用次数与已发布 token
async def test_stream_after_token_is_not_retried() -> None:
    final = _make_final()
    client = MagicMock()
    client.messages.stream.return_value = FailingStream(
        ["partial", "never"],
        final,
        fail_after_tokens=1,
    )
    provider = AnthropicProvider(model="test-model", client=client)
    events: list[BaseModel] = []
    bus = EventBus()

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    with pytest.raises(LlmStreamInterruptedError) as captured:
        await provider.chat([], [], bus, "r1")

    assert captured.value.partial_text == "partial"
    assert client.messages.stream.call_count == 1
    assert [event.token for event in events if event.type == "llm.token"] == ["partial"]  # type: ignore[attr-defined]
