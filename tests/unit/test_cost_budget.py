from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import anthropic
import httpx
import pytest

from tars_agent.core.config import LlmConfig, TarsConfig, _apply_env
from tars_agent.core.context import ExecutionContext
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm import provider as provider_module
from tars_agent.core.llm.budget import BudgetTransport
from tars_agent.core.llm.provider import (
    AnthropicProvider,
    LlmCallTimeoutError,
    LlmModelMismatchError,
    LlmProtocolError,
    ProviderConfigurationError,
)
from tars_agent.core.loop import AgentLoop
from tars_agent.core.persistence.cost_budget import (
    CAP_NANO_CNY,
    CONTEXT_TOKEN_BOUND,
    CostBudgetError,
    CostLedger,
)
from tars_agent.core.persistence.request_budget import RequestLedger
from tars_agent.core.tools.registry import ToolRegistry

RESERVED = CONTEXT_TOKEN_BOUND * 2000 + 8192 * 8000


@pytest.fixture
def ledgers(tmp_path: Path) -> tuple[RequestLedger, CostLedger]:
    requests = RequestLedger(tmp_path / "requests.sqlite3", limit=None)
    requests.reserve()
    costs = CostLedger.initialize(tmp_path / "costs.sqlite3", request_budget_path=requests.path)
    return requests, costs


def reserve(requests: RequestLedger, costs: CostLedger, *, max_tokens: int = 8192) -> str:
    return costs.reserve_attempt(requests, max_tokens=max_tokens, run_id="run", step=1, attempt=1)


def settle(costs: CostLedger, identifier: str, **overrides: object) -> None:
    kwargs = dict(model="deepseek-flash", input_tokens=100, output_tokens=10,
                  cache_read_input_tokens=20, cache_creation_input_tokens=5)
    kwargs.update(overrides)
    costs.settle(identifier, **kwargs)


def test_exact_integer_cost_and_original_cumulative_requests(ledgers) -> None:
    requests, costs = ledgers
    first = reserve(requests, costs)
    assert requests.counts()["real"] == 2
    assert costs.summary()["unknown_reserved_nano_cny"] == RESERVED
    settle(costs, first)
    settle(costs, first)  # Same complete observation is idempotent, never double-counted.
    summary = CostLedger(costs.path).summary()
    assert summary["confirmed_nano_cny"] == 290800
    assert summary["confirmed_cny"] == "0.000290800"
    assert summary["unknown_reserved_nano_cny"] == 0
    assert summary["remaining_nano_cny"] == CAP_NANO_CNY - 290800
    with closing(sqlite3.connect(costs.path)) as connection:
        assert connection.execute("SELECT request_ordinal FROM attempts").fetchone() == (2,)


def test_uncertain_attempts_exhaust_without_sending_or_reset(ledgers) -> None:
    requests, costs = ledgers
    for _ in range(CAP_NANO_CNY // RESERVED):
        reserve(requests, costs)
    before = requests.counts()["real"]
    with pytest.raises(CostBudgetError, match="exhausted"):
        reserve(requests, CostLedger(costs.path))
    assert requests.counts()["real"] == before
    assert costs.summary()["unknown_reserved_nano_cny"] == before * RESERVED - RESERVED
    with pytest.raises(CostBudgetError, match="already exists"):
        CostLedger.initialize(costs.path, request_budget_path=requests.path)


def test_parallel_admission_never_exceeds_fifty(ledgers) -> None:
    requests, costs = ledgers

    def admit(_: int) -> bool:
        try:
            reserve(requests, CostLedger(costs.path))
            return True
        except CostBudgetError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = sum(pool.map(admit, range(32)))
    assert accepted == CAP_NANO_CNY // RESERVED
    assert requests.counts()["real"] == accepted + 1
    assert costs.summary()["remaining_nano_cny"] >= 0


@pytest.mark.parametrize("mutation", ["missing_db", "missing_sentinel", "policy", "row", "rollback", "request_rollback", "metadata"])
def test_corrupt_or_rolled_back_state_blocks_admission_without_recreation(ledgers, mutation) -> None:
    requests, costs = ledgers
    reserve(requests, costs)
    if mutation == "missing_db":
        costs.path.unlink()
    elif mutation == "missing_sentinel":
        costs.sentinel.unlink()
    elif mutation == "policy":
        record = json.loads(costs.sentinel.read_text())
        record["policy"]["cap_nano_cny"] *= 2
        costs.sentinel.write_text(json.dumps(record))
    else:
        with closing(sqlite3.connect(requests.path if mutation == "request_rollback" else costs.path)) as connection:
            connection.execute({
                "row": "UPDATE attempts SET reserved = 1",
                "rollback": "DELETE FROM attempts",
                "request_rollback": "DELETE FROM requests",
                "metadata": "UPDATE metadata SET value = '{}'",
            }[mutation])
            connection.commit()
    before = requests.counts()["real"]
    with pytest.raises(CostBudgetError):
        reserve(requests, CostLedger(costs.path))
    assert requests.counts()["real"] == before
    if mutation == "missing_db":
        assert not costs.path.exists()
        with pytest.raises(CostBudgetError):
            CostLedger.initialize(costs.path, request_budget_path=requests.path)


@pytest.mark.parametrize("overrides", [
    {"model": "other"}, {"input_tokens": -1}, {"output_tokens": True},
    {"cache_read_input_tokens": None}, {"input_tokens": 9_000_000},
])
def test_invalid_settlement_retains_full_reserve(ledgers, overrides) -> None:
    requests, costs = ledgers
    identifier = reserve(requests, costs)
    with pytest.raises(CostBudgetError):
        settle(costs, identifier, **overrides)
    assert costs.summary()["unknown_reserved_nano_cny"] == RESERVED


def test_cannot_use_different_request_ledger_or_change_settled_usage(ledgers, tmp_path) -> None:
    requests, costs = ledgers
    different = RequestLedger(tmp_path / "other.sqlite3")
    different.counts()
    with pytest.raises(CostBudgetError, match="original"):
        reserve(different, costs)
    identifier = reserve(requests, costs)
    settle(costs, identifier)
    with pytest.raises(CostBudgetError, match="cannot be changed"):
        settle(costs, identifier, output_tokens=11)


@pytest.mark.parametrize("maximum", [0, -1, 8193, True, 1.5])
def test_invalid_output_bound_never_reserves(ledgers, maximum) -> None:
    requests, costs = ledgers
    with pytest.raises(CostBudgetError):
        reserve(requests, costs, max_tokens=maximum)
    assert requests.counts()["real"] == 1


def test_smaller_output_bound_is_honored(ledgers) -> None:
    requests, costs = ledgers
    reserve(requests, costs, max_tokens=32)
    assert costs.summary()["unknown_reserved_nano_cny"] == CONTEXT_TOKEN_BOUND * 2000 + 32 * 8000


@pytest.mark.parametrize("phase", ["request_link", "settlement"])
def test_transient_seal_publication_denial_retries_without_new_reservations(
    ledgers, monkeypatch, phase,
) -> None:
    requests, costs = ledgers
    identifier = reserve(requests, costs) if phase == "settlement" else None
    original_replace = Path.replace
    matching_calls = 0
    fail_at = 2 if phase == "request_link" else 1

    def temporarily_denied(path, destination):
        nonlocal matching_calls
        if path == costs.sentinel.with_name(costs.sentinel.name + ".tmp"):
            matching_calls += 1
            if matching_calls == fail_at:
                raise PermissionError(13, "synthetic sharing violation")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", temporarily_denied)
    if phase == "request_link":
        reserve(requests, costs)
    else:
        settle(costs, identifier)
    assert matching_calls == (3 if phase == "request_link" else 2)
    assert requests.counts()["real"] == 2
    summary = costs.summary()
    assert summary["attempt_count"] == 1
    assert summary["unknown_reserved_nano_cny"] == (RESERVED if phase == "request_link" else 0)
    assert summary["confirmed_nano_cny"] == (0 if phase == "request_link" else 290800)


def test_permanent_seal_publication_denial_retains_pending_state_and_fails_closed(
    ledgers, monkeypatch,
) -> None:
    requests, costs = ledgers
    original_replace = Path.replace
    matching_calls = 0

    def persistently_denied(path, destination):
        nonlocal matching_calls
        if path == costs.sentinel.with_name(costs.sentinel.name + ".tmp"):
            matching_calls += 1
            # Reproduce the real failure after linking the existing request ordinal.
            if matching_calls >= 2:
                raise PermissionError(13, "synthetic persistent sharing violation")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", persistently_denied)
    with pytest.raises(CostBudgetError):
        reserve(requests, costs)
    assert 2 <= matching_calls <= 7
    assert requests.counts()["real"] == 2
    pending = costs.sentinel.with_name(costs.sentinel.name + ".tmp")
    assert pending.is_file()
    pending_record = json.loads(pending.read_text())
    with closing(sqlite3.connect(costs.path)) as connection:
        assert pending_record["digest"] == CostLedger._digest(connection)
        assert connection.execute("SELECT reserved, confirmed, request_ordinal FROM attempts").fetchone() == (RESERVED, None, 2)
    with pytest.raises(CostBudgetError):
        costs.summary()
    with pytest.raises(CostBudgetError):
        reserve(requests, CostLedger(costs.path))
    with pytest.raises(CostBudgetError, match="already exists"):
        CostLedger.initialize(costs.path, request_budget_path=requests.path)
    assert requests.counts()["real"] == 2
    assert pending.read_text() == json.dumps(pending_record, sort_keys=True)


def test_transient_fsync_denial_rewrites_complete_seal_without_another_request(
    ledgers, monkeypatch,
) -> None:
    requests, costs = ledgers
    original_fsync = os.fsync
    calls = 0

    def denied_once(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError(13, "synthetic transient fsync denial")
        return original_fsync(fd)

    monkeypatch.setattr(os, "fsync", denied_once)
    reserve(requests, costs)
    assert calls == 3
    assert requests.counts()["real"] == 2
    assert costs.summary()["attempt_count"] == 1
    assert costs.summary()["unknown_reserved_nano_cny"] == RESERVED


def test_non_permission_io_error_does_not_get_retried_or_refunded(ledgers, monkeypatch) -> None:
    requests, costs = ledgers
    calls = 0

    def io_error(fd):
        nonlocal calls
        calls += 1
        raise OSError(5, "synthetic I/O failure")

    monkeypatch.setattr(os, "fsync", io_error)
    with pytest.raises(CostBudgetError):
        reserve(requests, costs)
    assert calls == 1
    assert requests.counts()["real"] == 1
    with closing(sqlite3.connect(costs.path)) as connection:
        assert connection.execute("SELECT reserved, confirmed FROM attempts").fetchone() == (RESERVED, None)
    with pytest.raises(CostBudgetError):
        costs.summary()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_windows_reader_temporarily_blocks_identity_replace(ledgers, monkeypatch) -> None:
    import ctypes
    from ctypes import wintypes

    requests, costs = ledgers
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    # Permit readers/writers but deliberately omit FILE_SHARE_DELETE, as a
    # concurrent Windows reader or scanner can do while inspecting the seal.
    handle = kernel32.CreateFileW(str(costs.sentinel), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    assert handle not in (None, ctypes.c_void_p(-1).value)
    original_replace = Path.replace
    denials = []

    def observed_replace(path, destination):
        nonlocal handle
        try:
            return original_replace(path, destination)
        except PermissionError as exc:
            if path == costs.sentinel.with_name(costs.sentinel.name + ".tmp") and handle is not None:
                denials.append(exc.winerror)
                assert kernel32.CloseHandle(handle)
                handle = None
            raise

    monkeypatch.setattr(Path, "replace", observed_replace)
    try:
        reserve(requests, costs)
    finally:
        if handle is not None:
            assert kernel32.CloseHandle(handle)
    assert denials and all(code in (5, 32, 33) for code in denials)
    assert requests.counts()["real"] == 2
    assert costs.summary()["unknown_reserved_nano_cny"] == RESERVED


def test_interrupted_initialization_never_automatically_restarts(tmp_path, monkeypatch) -> None:
    requests = RequestLedger(tmp_path / "existing.sqlite3")
    requests.counts()
    costs = CostLedger(tmp_path / "costs.sqlite3")
    original_seal = CostLedger._seal

    def crash_before_final_seal(self, connection, record):
        raise RuntimeError("synthetic crash")

    monkeypatch.setattr(CostLedger, "_seal", crash_before_final_seal)
    with pytest.raises(RuntimeError, match="synthetic"):
        CostLedger.initialize(costs.path, request_budget_path=requests.path)
    monkeypatch.setattr(CostLedger, "_seal", original_seal)
    with pytest.raises(CostBudgetError):
        costs.summary()
    with pytest.raises(CostBudgetError, match="already exists"):
        CostLedger.initialize(costs.path, request_budget_path=requests.path)


def test_configuration_defaults_and_trusted_environment_only(ledgers) -> None:
    _, costs = ledgers
    assert LlmConfig().cost_budget_path is None
    config = TarsConfig()
    _apply_env(config, {"TARS_LLM_COST_BUDGET_PATH": str(costs.path)}, trusted=False)
    assert config.llm.cost_budget_path is None
    _apply_env(config, {"TARS_LLM_COST_BUDGET_PATH": str(costs.path)})
    assert config.llm.cost_budget_path == costs.path
    with pytest.raises(SystemExit, match="existing valid"):
        _apply_env(config, {"TARS_LLM_COST_BUDGET_PATH": str(costs.path.with_name("absent"))})


def sse(*, model="deepseek-flash", input_tokens=10, output_tokens=2, truncated=False) -> bytes:
    events = [
        {"type": "message_start", "message": {
            "id": "msg_test", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": output_tokens}},
        {"type": "message_stop"},
    ]
    return b"".join(("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n").encode() for e in (events[:-1] if truncated else events))


def client_for(handler, requests, costs):
    return anthropic.AsyncAnthropic(
        api_key="synthetic-local-fixture", base_url="https://api.deepseek.com/anthropic", max_retries=0,
        http_client=httpx.AsyncClient(transport=BudgetTransport(httpx.MockTransport(handler), requests, cost_ledger=costs)),
    )


def provider_for(client, costs, **kwargs):
    return AnthropicProvider("deepseek-flash", client, expected_model="deepseek-flash",
                             cost_budget_path=costs.path, retry_delay_s=0.001, **kwargs)


async def test_sdk_retries_settle_only_success_and_clones_share_budget(ledgers) -> None:
    requests, costs = ledgers
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": {"type": "api_error", "message": "synthetic"}})
        return httpx.Response(200, content=sse(), headers={"content-type": "text/event-stream"})

    async with client_for(handler, requests, costs) as client:
        provider = provider_for(client, costs)
        assert (await provider.chat([], [], EventBus(), "parent")).text == "OK"
        await provider.with_model("deepseek-flash").chat([], [], EventBus(), "child")
    summary = costs.summary()
    assert calls == 3 and requests.counts()["real"] == 4
    assert summary["attempt_count"] == 3
    assert summary["confirmed_nano_cny"] == 72000
    assert summary["unknown_reserved_nano_cny"] == RESERVED


@pytest.mark.parametrize("payload, error", [
    (sse(truncated=True), LlmProtocolError), (sse(input_tokens=None), LlmProtocolError),
    (sse(output_tokens=-1), LlmProtocolError), (sse(model="other"), LlmModelMismatchError),
])
async def test_invalid_complete_or_truncated_response_never_refunds(ledgers, payload, error) -> None:
    requests, costs = ledgers
    async with client_for(lambda _: httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"}), requests, costs) as client:
        with pytest.raises(error):
            await provider_for(client, costs).chat([], [], EventBus(), "failure")
    assert requests.counts()["real"] == 2
    assert costs.summary()["unknown_reserved_nano_cny"] == RESERVED


@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancellation_keep_uncertain_reserve(ledgers, cancel) -> None:
    requests, costs = ledgers
    entered = asyncio.Event()

    async def handler(request):
        entered.set()
        await asyncio.sleep(30)
        return httpx.Response(200, content=sse())

    async with client_for(handler, requests, costs) as client:
        task = asyncio.create_task(provider_for(client, costs, total_timeout_s=0.15 if not cancel else 30).chat([], [], EventBus(), "stopped"))
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else LlmCallTimeoutError):
            await task
    assert costs.summary()["unknown_reserved_nano_cny"] == RESERVED
    assert requests.counts()["real"] == 2


async def test_cost_guard_rejects_uninstrumented_client_or_model_switch(ledgers) -> None:
    requests, costs = ledgers
    async with client_for(lambda _: httpx.Response(200, content=sse()), requests, costs) as client:
        with pytest.raises(ProviderConfigurationError):
            AnthropicProvider("other", client, cost_budget_path=costs.path)
    assert costs.summary()["attempt_count"] == 0


async def test_production_from_config_instruments_every_request(ledgers, monkeypatch) -> None:
    requests, costs = ledgers
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, content=sse(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(provider_module.httpx, "AsyncHTTPTransport", lambda **_: httpx.MockTransport(handler))
    provider = AnthropicProvider.from_config(LlmConfig(
        default_model="deepseek-flash", expected_model="deepseek-flash",
        api_key="synthetic-local-fixture", base_url="https://api.deepseek.com/anthropic",
        request_budget_path=requests.path, request_limit=None, cost_budget_path=costs.path,
    ))
    try:
        await provider.chat([], [], EventBus(), "actual-constructor")
    finally:
        await provider.close()
    assert len(sent) == 1
    assert requests.counts()["real"] == 2
    assert costs.summary()["confirmed_nano_cny"] == 36000


async def test_sdk_never_sends_after_cost_exhaustion(ledgers) -> None:
    requests, costs = ledgers
    for _ in range(CAP_NANO_CNY // RESERVED):
        reserve(requests, costs)
    before = requests.counts()["real"]

    def must_not_send(request):
        pytest.fail("Transport must not send after exhausting the local cost budget")

    async with client_for(must_not_send, requests, costs) as client:
        with pytest.raises(CostBudgetError, match="exhausted"):
            await provider_for(client, costs).chat([], [], EventBus(), "blocked")
    assert requests.counts()["real"] == before


async def test_concurrent_provider_calls_have_separate_attempt_identity(ledgers) -> None:
    requests, costs = ledgers
    async with client_for(lambda _: httpx.Response(200, content=sse(), headers={"content-type": "text/event-stream"}), requests, costs) as client:
        provider = provider_for(client, costs)
        await asyncio.gather(*(provider.chat([], [], EventBus(), f"concurrent-{n}") for n in range(5)))
    assert costs.summary()["confirmed_nano_cny"] == 5 * 36000
    assert costs.summary()["unknown_reserved_nano_cny"] == 0
    with closing(sqlite3.connect(costs.path)) as connection:
        assert len(set(connection.execute("SELECT run_id, request_ordinal FROM attempts"))) == 5


async def test_cost_budget_subclass_has_stable_run_reason() -> None:
    class ExhaustedProvider:
        async def chat(self, *args, **kwargs):
            raise CostBudgetError("synthetic exhausted cap")

    context = ExecutionContext(run_id="cost-exhausted", goal="test", max_steps=1)
    await AgentLoop(ExhaustedProvider(), ToolRegistry(), EventBus()).run(context)
    assert context.status == "failed"
    assert context.reason == "llm_request_budget_exhausted"


async def test_cancellation_waits_for_owned_reservation_and_never_sends(ledgers, monkeypatch) -> None:
    requests, costs = ledgers
    entered = threading.Event()
    release = threading.Event()
    original_reserve = costs.reserve_attempt

    def delayed_reserve(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_reserve(*args, **kwargs)

    def must_not_send(request):
        pytest.fail("Cancellation during reservation must prevent the HTTP send")

    monkeypatch.setattr(costs, "reserve_attempt", delayed_reserve)
    async with client_for(must_not_send, requests, costs) as client:
        task = asyncio.create_task(provider_for(client, costs).chat([], [], EventBus(), "cancel-while-reserving"))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert costs.summary()["unknown_reserved_nano_cny"] == RESERVED
    assert requests.counts()["real"] == 2
