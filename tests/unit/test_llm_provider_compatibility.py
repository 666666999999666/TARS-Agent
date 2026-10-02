from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import pytest

from tars_agent.core.compact.budget import ContextBudgetError
from tars_agent.core.config import LlmConfig
from tars_agent.core.context import ExecutionContext
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm import provider as provider_module
from tars_agent.core.llm.budget import RequestLedger
from tars_agent.core.llm.provider import (
    AnthropicProvider,
    LlmCallTimeoutError,
    LlmModelMismatchError,
    LlmProtocolError,
    LlmRateLimitError,
    ProviderConfigurationError,
)
from tars_agent.core.loop import AgentLoop
from tars_agent.core.tools.base import BaseTool, ToolResult
from tars_agent.core.tools.registry import ToolRegistry
from tars_agent.core.trace.provider import TracingProvider
from tars_agent.core.trace.writer import TraceWriter
from tests.unit.test_provider_boundary import _client, _sse

MODEL = "deepseek-flash"


def _encode(events: list[dict]) -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
        for event in events
    )


def _events(content: bytes) -> list[dict]:
    return [json.loads(line[6:]) for line in content.decode().splitlines() if line.startswith("data: ")]


def _reply() -> bytes:
    events = _events(_sse("已收到工具结果"))
    events[0]["message"]["model"] = MODEL
    return _encode(events)


def _tool_events() -> list[dict]:
    start = _events(_reply())[0]
    start["message"]["usage"].update(cache_read_input_tokens=3, cache_creation_input_tokens=2)
    events = [
        start,
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "需要调用工具"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": "test-signature"}},
        {"type": "content_block_stop", "index": 0},
    ]
    for index, (identifier, text) in enumerate(
        [("tool-cn", '你好，世界 "引号"'), ("tool-error", "故障")], start=1,
    ):
        argument = json.dumps({"msg": text}, ensure_ascii=False)
        split = len(argument) // 2
        events.append({"type": "content_block_start", "index": index, "content_block": {
            "type": "tool_use", "id": identifier, "name": "echo", "input": {},
        }})
        for fragment in (argument[:split], argument[split:]):
            events.append({"type": "content_block_delta", "index": index,
                           "delta": {"type": "input_json_delta", "partial_json": fragment}})
        events.append({"type": "content_block_stop", "index": index})
    events.extend([
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ])
    return events


class RecordingEcho(BaseTool):
    name = "echo"
    description = "Echo a string"
    input_schema = {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls.append(params)
        return ToolResult(content=f"回显：{params['msg']}", is_error=params["msg"] == "故障")


async def test_compatible_stream_runs_chinese_tools_and_returns_paired_results(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=_encode(_tool_events()) if len(requests) == 1 else _reply(),
                              headers={"content-type": "text/event-stream"})

    tool = RecordingEcho()
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    usage = []

    async def collect(event) -> None:
        if event.type == "llm.usage":
            usage.append(event)

    bus.subscribe(collect)
    context = ExecutionContext(run_id="compat", goal="回显两段文字", max_steps=3)
    async with _client(handler, ledger) as client:
        await AgentLoop(AnthropicProvider(MODEL, client, expected_model=MODEL), registry, bus).run(context)
    assert context.status == "success"
    assert context.result == "已收到工具结果"
    assert tool.calls == [{"msg": '你好，世界 "引号"'}, {"msg": "故障"}]
    assert ledger.counts()["real"] == 2
    assert [entry["model"] for entry in requests] == [MODEL, MODEL]
    assert requests[0]["tools"][0]["input_schema"] == tool.input_schema
    assistant, results = requests[1]["messages"][-2:]
    assert assistant["content"][0] == {
        "type": "thinking", "thinking": "需要调用工具", "signature": "test-signature",
    }
    assert [block["id"] for block in assistant["content"][1:]] == ["tool-cn", "tool-error"]
    assert results == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tool-cn", "content": '回显：你好，世界 "引号"'},
        {"type": "tool_result", "tool_use_id": "tool-error", "content": "回显：故障", "is_error": True},
    ]}
    assert [(entry.input_tokens, entry.output_tokens) for entry in usage] == [(10, 5), (10, 2)]
    assert (usage[0].cache_read_input_tokens, usage[0].cache_creation_input_tokens) == (3, 2)


@pytest.mark.parametrize("fault, reason", [
    ("model_missing", "llm_model_mismatch"), ("model_other", "llm_model_mismatch"),
    ("incomplete_json", "llm_protocol_error"), ("truncated", "llm_protocol_error"),
    ("invalid_usage", "llm_protocol_error"),
])
async def test_invalid_compatible_response_never_executes_tools(tmp_path: Path, fault: str, reason: str) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    events = _tool_events()
    if fault == "model_missing":
        events[0]["message"].pop("model")
    elif fault == "model_other":
        events[0]["message"]["model"] = "paid-model"
    elif fault == "incomplete_json":
        for event in events:
            if event["type"] == "content_block_delta" and event["index"] == 1:
                event["delta"]["partial_json"] = '{"msg":'
    elif fault == "truncated":
        events.pop()
    elif fault == "invalid_usage":
        events[0]["message"]["usage"]["input_tokens"] = -1
    tool = RecordingEcho()
    registry = ToolRegistry()
    registry.register(tool)
    context = ExecutionContext(run_id="invalid", goal="must not execute", max_steps=3)
    async with _client(lambda request: httpx.Response(200, content=_encode(events), headers={"content-type": "text/event-stream"}), ledger) as client:
        await AgentLoop(AnthropicProvider(MODEL, client, expected_model=MODEL), registry, EventBus()).run(context)
    assert context.status == "failed"
    assert context.reason == reason
    assert tool.calls == []
    assert ledger.counts()["real"] == 1


async def test_response_model_is_reported_not_backfilled_and_guard_is_optional(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    events = _events(_reply())
    async with _client(lambda request: httpx.Response(200, content=_encode(events), headers={"content-type": "text/event-stream"}), ledger) as client:
        provider = AnthropicProvider(MODEL, client, expected_model=MODEL)
        assert (await provider.chat([], [], EventBus(), "matched")).model == MODEL
        events[0]["message"].pop("model")
        with pytest.raises(LlmModelMismatchError):
            await provider.chat([], [], EventBus(), "missing")
        unguarded = AnthropicProvider("old-request-model", client)
        assert (await unguarded.chat([], [], EventBus(), "optional")).model is None
    assert ledger.counts()["real"] == 3


@pytest.mark.parametrize("actual_model", [
    None, 123, {"unexpected": "response-field"}, "DEEPSEEK-FLASH", "server-field\nprivate-marker",
])
async def test_model_mismatch_has_explicit_diagnostics_but_fixed_error_text(
    tmp_path: Path, actual_model,
) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    events = _tool_events()
    if actual_model is None:
        events[0]["message"].pop("model")
    else:
        events[0]["message"]["model"] = actual_model
    async with _client(lambda request: httpx.Response(200, content=_encode(events), headers={"content-type": "text/event-stream"}), ledger) as client:
        with pytest.raises(LlmModelMismatchError) as captured:
            await AnthropicProvider(MODEL, client, expected_model=MODEL).chat([], [], EventBus(), "identity")
    error = captured.value
    assert error.expected_model == MODEL
    assert error.actual_model == (actual_model if isinstance(actual_model, str) else None)
    assert str(error) == "Service model is missing or differs from expected_model"
    assert error.args == ("Service model is missing or differs from expected_model",)
    assert "private-marker" not in repr(error)
    assert "DEEPSEEK-FLASH" not in repr(error)
    assert MODEL not in repr(error)
    assert ledger.counts()["real"] == 1


async def test_configured_guard_and_clone_reject_paid_model_before_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = LlmConfig(default_model=MODEL, expected_model=MODEL, api_key="test-placeholder",
                       request_budget_path=tmp_path / "budget.sqlite3")
    provider = AnthropicProvider.from_config(config)
    try:
        clone = provider.with_model(MODEL)
        assert clone._client is provider._client
        assert clone._expected_model == MODEL
        with pytest.raises(ProviderConfigurationError, match="must match"):
            provider.with_model("paid-model")
        factory = MagicMock(side_effect=AssertionError("must reject before client creation"))
        monkeypatch.setattr(provider_module.anthropic, "AsyncAnthropic", factory)
        config.default_model = "paid-model"
        with pytest.raises(ProviderConfigurationError, match="must match"):
            AnthropicProvider.from_config(config)
        factory.assert_not_called()
        assert not config.request_budget_path.exists()
    finally:
        await provider.close()


@pytest.mark.parametrize("value", [None, 123, " ", "deepseek flash"])
def test_invalid_identity_guard_rejected(value) -> None:
    with pytest.raises(ProviderConfigurationError, match="Expected model"):
        AnthropicProvider(MODEL, MagicMock(), expected_model=value)


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 9, 29, 0, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize("header, delay", [
    ("3", 3.0), ("0", 0.0), ("Tue, 29 Sep 2026 00:00:07 GMT", 7.0),
    ("Mon, 28 Sep 2026 00:00:00 GMT", 0.0),
    ("bad-date", 0.25), ("-1", 0.25), ("inf", 0.25), ("nan", 0.25), (None, 0.25),
])
@pytest.mark.parametrize("status", [429, 529])
async def test_retry_after_obeyed_and_both_http_attempts_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, header: str | None, delay: float, status: int,
) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    responses = [
        httpx.Response(status, json={"error": {"type": "rate_limit_error" if status == 429 else "overloaded_error", "message": "limited"}},
                       headers={"retry-after": header} if header is not None else {}),
        httpx.Response(200, content=_reply(), headers={"content-type": "text/event-stream"}),
    ]
    sleep = AsyncMock()
    monkeypatch.setattr(provider_module, "datetime", FixedDateTime)
    monkeypatch.setattr(provider_module.asyncio, "sleep", sleep)
    async with _client(lambda request: responses.pop(0), ledger) as client:
        response = await AnthropicProvider(MODEL, client, expected_model=MODEL, retry_delay_s=0.25).chat([], [], EventBus(), "rate")
    assert response.model == MODEL
    sleep.assert_awaited_once_with(delay)
    assert ledger.counts()["real"] == 2


@pytest.mark.parametrize("attempts", [1, 2])
@pytest.mark.parametrize("status", [429, 529])
async def test_exhausted_rate_limit_has_distinct_loop_reason(tmp_path: Path, attempts: int, status: int) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    context = ExecutionContext(run_id="rate", goal="limited", max_steps=3)
    error = {"type": "rate_limit_error"} if status == 429 else {"type": "overloaded_error", "code": "1305"}
    async with _client(lambda request: httpx.Response(status, json={"error": error}, headers={"retry-after": "0"}), ledger) as client:
        provider = AnthropicProvider(MODEL, client, expected_model=MODEL, attempts=attempts)
        await AgentLoop(provider, ToolRegistry(), EventBus()).run(context)
    assert context.status == "failed"
    assert context.reason == "llm_rate_limited"
    assert ledger.counts()["real"] == attempts


@pytest.mark.parametrize("status", [429, 529])
async def test_retry_after_beyond_remaining_deadline_stops_without_second_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    sleep = AsyncMock()
    monkeypatch.setattr(provider_module.asyncio, "sleep", sleep)
    async with _client(lambda request: httpx.Response(status, json={"error": {}}, headers={"retry-after": "120"}), ledger) as client:
        with pytest.raises(LlmRateLimitError):
            await AnthropicProvider(MODEL, client, total_timeout_s=10).chat([], [], EventBus(), "rate")
    sleep.assert_not_awaited()
    assert ledger.counts()["real"] == 1


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("status", [429, 529])
async def test_rate_limit_wait_deadline_and_external_cancel_never_resend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool, status: int,
) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    waiting = asyncio.Event()
    deadlines: list[asyncio.Timeout] = []
    original_timeout = asyncio.timeout

    def capture_timeout(delay):
        deadline = original_timeout(delay)
        deadlines.append(deadline)
        return deadline

    async def stalled_sleep(delay: float) -> None:
        waiting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(provider_module.asyncio, "sleep", stalled_sleep)
    monkeypatch.setattr(provider_module.asyncio, "timeout", capture_timeout)
    async with _client(lambda request: httpx.Response(status, json={"error": {}}, headers={"retry-after": "0"}), ledger) as client:
        provider = AnthropicProvider(MODEL, client, total_timeout_s=10)
        task = asyncio.create_task(provider.chat([], [], EventBus(), "rate"))
        await asyncio.wait_for(waiting.wait(), 5)
        if cancel:
            task.cancel()
        else:
            # Trigger the actual call deadline after sleep started, without timing assumptions.
            deadlines[0].reschedule(asyncio.get_running_loop().time())
        with pytest.raises(asyncio.CancelledError if cancel else LlmRateLimitError):
            await task
    assert ledger.counts()["real"] == 1


@pytest.mark.parametrize("prefix", ["none", "message_start", "text"])
@pytest.mark.parametrize("error_type", ["rate_limit_error", "overloaded_error"])
async def test_stream_rate_limit_stops_without_retry_and_keeps_only_audit_text(
    tmp_path: Path, prefix: str, error_type: str,
) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    length = {"none": 0, "message_start": 1, "text": 3}[prefix]
    events = _events(_reply())[:length]
    events.append({"type": "error", "error": {"type": error_type, "message": "slow down"}})
    context = ExecutionContext(run_id="rate", goal="limited stream", max_steps=3)
    async with _client(lambda request: httpx.Response(200, content=_encode(events), headers={"content-type": "text/event-stream"}), ledger) as client:
        await AgentLoop(AnthropicProvider(MODEL, client, expected_model=MODEL), ToolRegistry(), EventBus()).run(context)
    assert context.reason == "llm_rate_limited"
    assert ledger.counts()["real"] == 1
    assert not any(message["role"] == "assistant" for message in context.messages)
    partial = [message for message in context.audit_messages if message["role"] == "assistant"]
    if prefix == "text":
        assert partial == [{"role": "assistant", "content": [
            {"type": "text", "text": "已收到工具结果", "partial": True},
        ]}]
    else:
        assert partial == []


async def test_concurrent_calls_have_independent_rate_limit_deadlines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    waiting = asyncio.Event()
    release = asyncio.Event()
    attempts: dict[str, int] = {}

    async def sleep(delay: float) -> None:
        waiting.set()
        await release.wait()

    def handler(request: httpx.Request) -> httpx.Response:
        name = json.loads(request.content)["messages"][0]["content"]
        attempts[name] = attempts.get(name, 0) + 1
        if name == "limited":
            return httpx.Response(429, json={"error": {}}, headers={"retry-after": "0"})
        return httpx.Response(200, content=_reply(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(provider_module.asyncio, "sleep", sleep)
    async with _client(handler, ledger) as client:
        provider = AnthropicProvider(MODEL, client, expected_model=MODEL)
        limited = asyncio.create_task(provider.chat([{"role": "user", "content": "limited"}], [], EventBus(), "limited"))
        await asyncio.wait_for(waiting.wait(), 5)
        success = await provider.chat([{"role": "user", "content": "success"}], [], EventBus(), "success")
        assert success.model == MODEL
        release.set()
        with pytest.raises(LlmRateLimitError):
            await limited
    assert attempts == {"limited": 2, "success": 1}
    assert ledger.counts()["real"] == 3


async def test_overloaded_529_after_first_stream_event_is_not_retried(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")

    class OverloadedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _encode(_events(_reply())[:3])
            response = httpx.Response(529, request=httpx.Request("POST", "http://127.0.0.1/messages"))
            raise anthropic.OverloadedError(
                "overloaded", response=response,
                body={"error": {"type": "overloaded_error", "code": "1305"}},
            )

    async with _client(lambda request: httpx.Response(200, stream=OverloadedStream(), headers={"content-type": "text/event-stream"}), ledger) as client:
        with pytest.raises(LlmRateLimitError) as captured:
            await AnthropicProvider(MODEL, client, expected_model=MODEL).chat([], [], EventBus(), "overloaded")
    assert captured.value.partial_text == "已收到工具结果"
    assert ledger.counts()["real"] == 1


@pytest.mark.parametrize("status, error_type, attempts", [
    (400, "overloaded_error", 1), (401, "rate_limit_error", 1),
    (500, "api_error", 2), (503, "api_error", 2),
    (200, "invalid_request_error", 1), (200, "unknown_error", 1),
])
async def test_other_http_and_stream_errors_are_not_reclassified_as_rate_limits(
    tmp_path: Path, status: int, error_type: str, attempts: int,
) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    body = {"type": "error", "error": {"type": error_type, "message": "test-error"}}
    response = (
        httpx.Response(200, content=_encode([body]), headers={"content-type": "text/event-stream"})
        if status == 200 else httpx.Response(status, json=body)
    )
    async with _client(lambda request: response, ledger) as client:
        with pytest.raises(anthropic.APIStatusError):
            await AnthropicProvider(MODEL, client, retry_delay_s=0.001).chat([], [], EventBus(), "other-error")
    assert ledger.counts()["real"] == attempts


@pytest.mark.parametrize("fault,reason,status", [
    ("rate", "llm_rate_limited", 429),
    ("sdk", "llm_error", 500),
    ("stream", "llm_stream_interrupted", None),
    ("model", "llm_model_mismatch", None),
])
async def test_loop_sdk_failure_logs_only_classification_without_error_body(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, fault: str, reason: str, status: int | None,
) -> None:
    secret = "sk-synthetic-private-key"
    remote_marker = "remote-private-error-body"
    sensitive = secret + " " + remote_marker
    captured = []

    class ObservedProvider(AnthropicProvider):
        async def chat(self, *args, **kwargs):
            try:
                return await super().chat(*args, **kwargs)
            except Exception as exc:
                captured.append(exc)
                raise

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _encode(_events(_reply())[:1])
            raise anthropic.APIConnectionError(
                message=sensitive, request=httpx.Request("POST", "http://127.0.0.1/messages"),
            )

    def handler(request):
        if fault == "stream":
            return httpx.Response(200, stream=BrokenStream(), headers={"content-type": "text/event-stream"})
        if fault == "model":
            events = _events(_reply())
            events[0]["message"]["model"] = sensitive
            return httpx.Response(200, content=_encode(events), headers={"content-type": "text/event-stream"})
        return httpx.Response(status, json={"error": {"type": "rate_limit_error" if fault == "rate" else "api_error", "message": sensitive}})

    caplog.set_level(logging.WARNING)
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    writer = TraceWriter(tmp_path / "trace.jsonl")
    await writer.start()
    context = ExecutionContext(run_id="safe-log", goal=sensitive, max_steps=3)
    try:
        async with _client(handler, ledger) as client:
            inner = ObservedProvider(MODEL, client, expected_model=MODEL, attempts=1)
            await AgentLoop(TracingProvider(inner, writer, include_payload=False),
                            ToolRegistry(), EventBus()).run(context)
    finally:
        await writer.stop()
    assert context.status == "failed" and context.reason == reason
    assert len(captured) == 1 and ledger.counts()["real"] == 1
    # The original exception/SDK cause remains available to the safe probe.
    if fault == "model":
        assert captured[0].actual_model == sensitive
    else:
        source = captured[0].__cause__ if fault in {"rate", "stream"} else captured[0]
        assert sensitive in str(source)
    assert secret not in caplog.text and remote_marker not in caplog.text
    records = [record for record in caplog.records if record.name == "tars_agent.core.loop"]
    assert len(records) == 1 and records[0].exc_info is None
    assert f"reason={reason}" in records[0].getMessage()
    assert f"error_type={type(captured[0]).__name__}" in records[0].getMessage()
    if status is not None:
        assert f"http_status={status}" in records[0].getMessage()
    trace = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")
    assert secret not in trace and remote_marker not in trace
    assert "message_count" in trace and '"messages"' not in trace


@pytest.mark.parametrize("kind,reason", [
    ("context", "context-budget-code"), ("protocol", "llm_protocol_error"),
    ("timeout", "llm_total_timeout"), ("budget", "llm_request_budget_exhausted"),
])
async def test_loop_controlled_model_failure_does_not_log_chained_sdk_message(
    caplog: pytest.LogCaptureFixture, kind: str, reason: str,
) -> None:
    from tars_agent.core.llm.budget import ModelRequestBudgetExceeded

    secret = "sk-synthetic-private-key"
    response = httpx.Response(401, request=httpx.Request("POST", "http://127.0.0.1/messages"))
    sdk_error = anthropic.AuthenticationError(secret + " remote-private-marker", response=response, body={})
    error = {
        "context": ContextBudgetError("context-budget-code"),
        "protocol": LlmProtocolError(secret), "timeout": LlmCallTimeoutError(secret),
        "budget": ModelRequestBudgetExceeded(secret),
    }[kind]
    error.__cause__ = sdk_error
    provider = AsyncMock()
    provider.chat.side_effect = error
    caplog.set_level(logging.WARNING)
    context = ExecutionContext(run_id="controlled-safe-log", goal="synthetic", max_steps=2)
    await AgentLoop(provider, ToolRegistry(), EventBus()).run(context)
    assert context.status == "failed" and context.reason == reason
    assert error.__cause__ is sdk_error
    assert secret not in caplog.text and "remote-private-marker" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert f"error_type={type(error).__name__}" in caplog.text
