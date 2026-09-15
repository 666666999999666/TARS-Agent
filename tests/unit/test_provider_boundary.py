from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from tars_agent.core.config import LlmConfig
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.budget import (
    BudgetTransport,
    ModelRequestBudgetExceeded,
    RequestLedger,
)
from tars_agent.core.llm.provider import (
    AnthropicProvider,
    LlmCallTimeoutError,
    LlmProtocolError,
    LlmStreamInterruptedError,
    ProviderConfigurationError,
)


def _sse(text: str = "hello", stop: str | None = "end_turn") -> bytes:
    events = [
        {"type": "message_start", "message": {
            "id": "msg_test", "type": "message", "role": "assistant", "model": "test",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
         "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    return b"".join(
        ("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode()
        for event in events
    )


def _client(handler, ledger: RequestLedger) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key="test-placeholder", base_url="http://127.0.0.1:18888", max_retries=0,
        http_client=httpx.AsyncClient(
            transport=BudgetTransport(httpx.MockTransport(handler), ledger),
            follow_redirects=False,
        ),
    )


async def test_real_sdk_stream_and_clone_share_request_ledger(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(), headers={"content-type": "text/event-stream"})

    async with _client(handler, ledger) as client:
        provider = AnthropicProvider("one", client)
        response = await provider.chat([], [], EventBus(), "top")
        clone = provider.with_model("two")
        await clone.chat([], [], EventBus(), "child")
    assert response.text == "hello"
    assert [item["model"] for item in captured] == ["one", "two"]
    assert ledger.counts() == {"real": 2, "probe": 0}


@pytest.mark.parametrize("status, count", [(400, 1), (401, 1), (403, 1), (429, 2), (500, 2)])
async def test_http_status_retry_classification(tmp_path: Path, status: int, count: int) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    async with _client(lambda request: httpx.Response(status, json={"error": {"type": "api_error", "message": "test"}}), ledger) as client:
        with pytest.raises(anthropic.APIStatusError):
            await AnthropicProvider("test", client, retry_delay_s=0.001).chat([], [], EventBus(), "x")
    assert ledger.counts()["real"] == count


async def test_incomplete_stream_is_failure_without_retry(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    truncated = _sse().split(b"event: message_stop")[0]
    async with _client(lambda request: httpx.Response(200, content=truncated, headers={"content-type": "text/event-stream"}), ledger) as client:
        with pytest.raises(LlmProtocolError, match="message_stop"):
            await AnthropicProvider("test", client).chat([], [], EventBus(), "x")
    assert ledger.counts()["real"] == 1


@pytest.mark.parametrize("text", ["", "   "])
async def test_blank_reply_is_not_success(tmp_path: Path, text: str) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    async with _client(lambda request: httpx.Response(200, content=_sse(text), headers={"content-type": "text/event-stream"}), ledger) as client:
        with pytest.raises(LlmProtocolError, match="nonempty"):
            await AnthropicProvider("test", client).chat([], [], EventBus(), "x")


async def test_missing_stop_reason_warns_only_after_complete_text(tmp_path: Path, caplog) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    async with _client(lambda request: httpx.Response(200, content=_sse(stop=None), headers={"content-type": "text/event-stream"}), ledger) as client:
        result = await AnthropicProvider("test", client).chat([], [], EventBus(), "x")
    assert result.stop_reason == "end_turn"
    assert "omitted stop_reason" in caplog.text


class BrokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield _sse().split(b"event: content_block_start")[0]
        raise httpx.ReadError("lost after message_start")


async def test_first_nontext_event_already_prevents_retry(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    async with _client(lambda request: httpx.Response(200, stream=BrokenStream(), headers={"content-type": "text/event-stream"}), ledger) as client:
        with pytest.raises(LlmStreamInterruptedError) as failure:
            await AnthropicProvider("test", client).chat([], [], EventBus(), "x")
    assert failure.value.partial_text == ""
    assert ledger.counts()["real"] == 1


async def test_total_deadline_includes_connection_wait(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")

    async def delayed(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, content=_sse())

    async with _client(delayed, ledger) as client:
        with pytest.raises(LlmCallTimeoutError):
            await AnthropicProvider("test", client, total_timeout_s=0.05).chat([], [], EventBus(), "x")
    assert ledger.counts()["real"] == 1


def test_atomic_budget_survives_reopen_and_concurrent_providers(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"

    def reserve(_: int) -> bool:
        try:
            RequestLedger(path).reserve()
            return True
        except ModelRequestBudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(108)))
    assert sum(results) == 100
    assert RequestLedger(path).counts() == {"real": 100, "probe": 0}
    with pytest.raises(ModelRequestBudgetExceeded):
        RequestLedger(path).reserve()


async def test_101st_request_blocked_before_transport_send(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    for _ in range(100):
        ledger.reserve()
    handler = MagicMock(return_value=httpx.Response(200, content=_sse()))
    async with _client(handler, ledger) as client:
        with pytest.raises(ModelRequestBudgetExceeded):
            await AnthropicProvider("test", client).chat([], [], EventBus(), "x")
    handler.assert_not_called()
    assert ledger.counts()["real"] == 100


async def test_probe_counter_is_separate_and_rejects_remote(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    transport = BudgetTransport(httpx.MockTransport(lambda request: httpx.Response(200)), ledger, kind="probe")
    async with httpx.AsyncClient(transport=transport) as client:
        await client.get("http://127.0.0.1/test")
        with pytest.raises(ValueError, match="loopback"):
            await client.get("https://remote.invalid/test")
    assert ledger.counts() == {"real": 0, "probe": 1}


def test_custom_endpoint_never_uses_official_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "official")
    monkeypatch.delenv("TARS_LLM_API_KEY", raising=False)
    with pytest.raises(ProviderConfigurationError, match="dedicated"):
        AnthropicProvider("test", base_url="https://relay.invalid")


async def test_constructor_pins_endpoint_timeouts_and_disables_sdk_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://untrusted.invalid")
    monkeypatch.delenv("TARS_LLM_API_KEY", raising=False)
    config = LlmConfig(anthropic_api_key="official", request_budget_path=tmp_path / "budget.sqlite3")
    provider = AnthropicProvider.from_config(config)
    try:
        assert str(provider._client.base_url).rstrip("/") == "https://api.anthropic.com"
        assert provider._client.max_retries == 0
        assert provider._client.timeout.connect == 10
        assert provider._client.timeout.read == 60
        assert provider._client.timeout.write == 10
        assert provider._client.timeout.pool == 10
        assert provider._client._client.follow_redirects is False
    finally:
        await provider.close()
    assert not config.request_budget_path.exists()


async def test_tool_json_must_be_complete_even_when_sdk_accepts_partial_json(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    lines = _sse(stop="tool_use").decode().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "content_block_start":
                event["content_block"] = {
                    "type": "tool_use", "id": "tool_one", "name": "write_file", "input": {},
                }
                lines[index] = "data: " + json.dumps(event)
            if event["type"] == "content_block_delta":
                event["delta"] = {"type": "input_json_delta", "partial_json": '{"path": "x"'}
                lines[index] = "data: " + json.dumps(event)
    content = ("\n".join(lines) + "\n").encode()
    async with _client(lambda request: httpx.Response(200, content=content, headers={"content-type": "text/event-stream"}), ledger) as client:
        with pytest.raises(LlmProtocolError, match="tool JSON"):
            await AnthropicProvider("test", client).chat([], [], EventBus(), "x")
    assert ledger.counts()["real"] == 1


async def test_cancel_model_wait_never_retries(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    started = asyncio.Event()

    async def wait_forever(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async with _client(wait_forever, ledger) as client:
        task = asyncio.create_task(AnthropicProvider("test", client).chat([], [], EventBus(), "x"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert ledger.counts()["real"] == 1


async def test_redirect_is_not_followed_or_retried(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "budget.sqlite3")
    handler = MagicMock(return_value=httpx.Response(307, headers={"location": "https://other.invalid"}))
    async with _client(handler, ledger) as client:
        with pytest.raises(anthropic.APIStatusError):
            await AnthropicProvider("test", client).chat([], [], EventBus(), "x")
    assert handler.call_count == ledger.counts()["real"] == 1



def test_unlimited_keeps_existing_history_and_counts_past_100(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    original = RequestLedger(path)
    for _ in range(100):
        original.reserve()
    with pytest.raises(ModelRequestBudgetExceeded):
        original.reserve()
    unlimited = RequestLedger(path, limit=None)
    with ThreadPoolExecutor(max_workers=8) as pool:
        reservations = list(pool.map(lambda _: unlimited.reserve(), range(12)))
    assert sorted(reservations) == list(range(101, 113))
    reopened = RequestLedger(path, limit=None)
    assert reopened.counts() == {"real": 112, "probe": 0}
    assert reopened.reserve() == 113
    assert RequestLedger(path).counts()["real"] == 113
    with pytest.raises(ModelRequestBudgetExceeded):
        RequestLedger(path).reserve()


def test_positive_custom_limit_preserves_history(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    first = RequestLedger(path, limit=2)
    assert first.reserve() == 1
    assert first.reserve() == 2
    with pytest.raises(ModelRequestBudgetExceeded):
        first.reserve()
    increased = RequestLedger(path, limit=3)
    assert increased.reserve() == 3
    with pytest.raises(ModelRequestBudgetExceeded):
        increased.reserve()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_ledger_rejects_invalid_limits_without_touching_file(tmp_path: Path, limit) -> None:
    path = tmp_path / "budget.sqlite3"
    with pytest.raises(ValueError, match="positive integer"):
        RequestLedger(path, limit=limit)
    assert not path.exists()


async def test_from_config_and_model_clone_share_unlimited_ledger(tmp_path: Path) -> None:
    config = LlmConfig(
        api_key="unit-placeholder", request_limit=None,
        request_budget_path=tmp_path / "budget.sqlite3",
    )
    provider = AnthropicProvider.from_config(config)
    clone = provider.with_model("another-model")
    try:
        assert clone._client is provider._client
        assert clone._request_limit is None
        transport = provider._client._client._transport
        assert transport._ledger.path == config.request_budget_path
        assert transport._ledger.limit is None
        for _ in range(102):
            transport._ledger.reserve()
        assert RequestLedger(config.request_budget_path).counts()["real"] == 102
    finally:
        await provider.close()
