from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from tars_agent.core.bus.events import LlmModelSelectedEvent, LlmUsageEvent
from tars_agent.core.config import TarsConfig
from tars_agent.core.eval import appworld
from tars_agent.core.eval import appworld_worker as worker
from tars_agent.core.eval.internal import _EventAccumulator
from tars_agent.core.llm.types import LlmResponse
from tars_agent.core.mcp.client import McpToolDef
from tars_agent.core.mcp.tool import McpTool
from tars_agent.core.persistence.database import Database
from tars_agent.core.persistence.request_budget import ModelRequestBudgetExceeded, RequestLedger
from tars_agent.core.runner import AgentRunner
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter


async def test_request_accounting_is_process_local_and_missing_retry_usage_stays_unknown(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "shared.sqlite3", limit=10)
    counter = worker.ProcessRequestAccounting()
    transport = counter.transport(httpx.MockTransport(lambda request: httpx.Response(200)), ledger)
    ledger.reserve()  # Another worker's reservation is deliberately outside this wrapper.
    request = httpx.Request("POST", "https://not-contacted.invalid/messages")
    await transport.handle_async_request(request)
    await transport.handle_async_request(request)
    assert ledger.counts()["real"] == 3
    assert counter.snapshot() == {"attempted": 2, "reserved": 2, "budget_denied": 0,
                                  "reservation_errors": 0, "inflight": 0}
    events = _EventAccumulator()
    await events.record(LlmModelSelectedEvent(run_id="r", model="m", ts="now"))
    await events.record(LlmUsageEvent(run_id="r", input_tokens=10, output_tokens=2,
                                     cache_read_input_tokens=0, cache_creation_input_tokens=0, ts="now"))
    assert events.usage().input_tokens == 10
    assert counter.usage(events).input_tokens is None  # Two HTTP attempts, only one usage event.
    await events.record(LlmUsageEvent(run_id="r", input_tokens=20, output_tokens=3,
                                     cache_read_input_tokens=0, cache_creation_input_tokens=0, ts="now"))
    assert counter.usage(events).input_tokens == 30
    assert await counter.wait_idle()
    await transport.aclose()


async def test_budget_denial_is_counted_before_reserve_but_not_as_sent(tmp_path: Path) -> None:
    ledger = RequestLedger(tmp_path / "limited.sqlite3", limit=1)
    counter = worker.ProcessRequestAccounting()
    transport = counter.transport(httpx.MockTransport(lambda request: httpx.Response(200)), ledger)
    request = httpx.Request("POST", "https://not-contacted.invalid/messages")
    await transport.handle_async_request(request)
    with pytest.raises(ModelRequestBudgetExceeded):
        await transport.handle_async_request(request)
    assert counter.snapshot()["attempted"] == 2
    assert counter.snapshot()["reserved"] == 1
    assert counter.snapshot()["budget_denied"] == 1
    await transport.aclose()


async def test_worker_accounting_wraps_real_provider_cost_guard_without_double_reservation(
    tmp_path, monkeypatch,
):
    from tars_agent.core.events.bus import EventBus
    from tars_agent.core.llm import provider as provider_module
    from tars_agent.core.persistence.cost_budget import CostLedger
    from tests.unit.test_cost_budget import sse

    ledger = RequestLedger(tmp_path / "original.sqlite3")
    ledger.reserve()
    costs = CostLedger.initialize(tmp_path / "cost.sqlite3", request_budget_path=ledger.path)
    config = appworld.load_deepseek_config(private_env=tmp_path / "absent.env", environment={
        "DEEPSEEK_API_KEY": "synthetic-local-fixture",
        "DEEPSEEK_REQUEST_BUDGET_PATH": str(ledger.path),
        "DEEPSEEK_COST_BUDGET_PATH": str(costs.path),
    })
    before = appworld.config_fingerprint(config)
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, content=sse(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(provider_module.httpx, "AsyncHTTPTransport",
                        lambda **_: httpx.MockTransport(handler))
    with worker.ProcessRequestAccounting() as accounting:
        provider = provider_module.AnthropicProvider.from_config(config.llm)
        try:
            response = await provider.chat([], [], EventBus(), "guarded-worker")
        finally:
            await provider.close()
        assert await accounting.wait_idle()
        assert accounting.snapshot()["reserved"] == 1
    assert response.model == "deepseek-flash" and len(sent) == 1
    assert ledger.counts()["real"] == 2
    assert costs.summary()["attempt_count"] == 1
    assert costs.summary()["confirmed_nano_cny"] == 36000
    assert appworld.config_fingerprint(config) == before  # Mutable spend is not a freeze input.


def make_job(config: TarsConfig, tmp_path: Path) -> dict[str, Any]:
    job = {"task_id": "synthetic_1", "experiment_name": "synthetic-experiment",
           "environment_url": "http://127.0.0.1:8000", "attempt_root": str(tmp_path / "runtime"),
           "model": config.llm.default_model, "max_steps": 4, "task_timeout_s": 5,
           "request_limit": 100, "request_budget_path": str(config.llm.request_budget_path),
           "python_environment": appworld.python_environment()}
    if config.llm.cost_budget_path is not None:
        job["cost_budget_path"] = str(config.llm.cost_budget_path)
    job["config_fingerprint"] = appworld.config_fingerprint(appworld.worker_config(config, job))
    job["job_fingerprint"] = appworld.digest_json(job)
    return job


async def test_worker_keeps_confirmed_tokens_after_later_cost_publication_fault(tmp_path, monkeypatch):
    from tars_agent.core.events.bus import EventBus
    from tars_agent.core.llm import provider as provider_module
    from tars_agent.core.persistence.cost_budget import CostLedger
    from tests.unit.test_cost_budget import sse

    ledger = RequestLedger(tmp_path / "original.sqlite3")
    ledger.reserve()
    costs = CostLedger.initialize(tmp_path / "cost.sqlite3", request_budget_path=ledger.path)
    config = appworld.load_deepseek_config(private_env=tmp_path / "absent.env", environment={
        "DEEPSEEK_API_KEY": "synthetic-local-fixture",
        "DEEPSEEK_REQUEST_BUDGET_PATH": str(ledger.path),
        "DEEPSEEK_COST_BUDGET_PATH": str(costs.path),
    })
    job = make_job(config, tmp_path)
    sent = []

    def handler(request):
        sent.append(request)
        assert len(sent) == 1  # The second cost reservation must fail before another HTTP send.
        response = sse(input_tokens=23, output_tokens=7).replace(
            b'"input_tokens": 23, "output_tokens": 0',
            b'"input_tokens": 23, "output_tokens": 0, '
            b'"cache_read_input_tokens": 5, "cache_creation_input_tokens": 3',
        )
        return httpx.Response(200, content=response, headers={"content-type": "text/event-stream"})

    def failed_publication(*args, **kwargs):
        raise OSError("synthetic cost seal publication failure")

    async def execute(received, selected):
        bus = EventBus()
        events = _EventAccumulator()
        bus.subscribe(events.record)
        provider = provider_module.AnthropicProvider.from_config(selected.llm)
        try:
            response = await provider.chat([], [], bus, "partial-usage", step=1)
            assert response.model == "deepseek-flash"
            monkeypatch.setattr(CostLedger, "_seal", failed_publication)
            with pytest.raises(ModelRequestBudgetExceeded):
                await provider.chat([], [], bus, "partial-usage", step=2)
        finally:
            await provider.close()
        return {
            "run_id": "partial-usage", "run_terminal_status": "failed",
            "reason": "llm_request_budget_exhausted", "saved": True, "closed": True,
        }, events

    monkeypatch.setattr(worker, "get_config", lambda: config)
    monkeypatch.setattr(worker, "_execute_job", execute)
    monkeypatch.setattr(provider_module.httpx, "AsyncHTTPTransport",
                        lambda **_: httpx.MockTransport(handler))
    result = await worker.run_job(job)
    assert len(sent) == 1 and ledger.counts()["real"] == 2
    assert result["model_calls_started"] == 2 and result["model_responses"] == 1
    assert result["usage_complete"] is False and result["usage"]["input_tokens"] is None
    assert result["confirmed_usage"]["input_tokens"] == 23
    assert result["confirmed_usage"]["output_tokens"] == 7
    assert result["confirmed_usage"]["cache_read_input_tokens"] == 5
    assert result["confirmed_usage"]["cache_creation_input_tokens"] == 3
    assert result["infrastructure_error"] is True
    assert result["model_requests"]["reserved"] == 1


async def test_worker_rejects_frozen_endpoint_drift_before_initializing_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TarsConfig()
    config.llm.api_key = "credential-not-in-job"
    config.llm.base_url = "https://first.invalid/secret-path"
    job = make_job(config, tmp_path)
    config.llm.base_url = "https://second.invalid/other-path"
    monkeypatch.setattr(worker, "get_config", lambda: config)
    with pytest.raises(ValueError, match="frozen"):
        await worker.run_job(job)
    assert not (tmp_path / "runtime").exists()
    assert "credential-not-in-job" not in json.dumps(job)
    assert "secret-path" not in json.dumps(job)


async def test_worker_rejects_prompt_template_drift_before_loading_configuration(tmp_path, monkeypatch):
    config = TarsConfig()
    job = make_job(config, tmp_path)
    job.update(prompt_variant="B", prompt_template_sha256=appworld.template_sha256("A"))
    job["job_fingerprint"] = appworld.digest_json({
        key: value for key, value in job.items() if key != "job_fingerprint"
    })
    monkeypatch.setattr(worker, "get_config", lambda: pytest.fail("prompt drift must fail first"))
    with pytest.raises(ValueError, match="completion prompt"):
        await worker.run_job(job)
    assert not (tmp_path / "runtime").exists()


async def test_deepseek_worker_reloads_dedicated_config_without_ordinary_credentials(tmp_path, monkeypatch):
    ledger = RequestLedger(tmp_path / "original.sqlite3")
    ledger.reserve()
    config = appworld.load_deepseek_config(private_env=tmp_path / "missing.env", environment={
        "DEEPSEEK_API_KEY": "unit-deepseek-placeholder", "DEEPSEEK_REQUEST_BUDGET_PATH": str(ledger.path),
    })
    job = make_job(config, tmp_path)
    job["profile"] = appworld.DEEPSEEK_PROFILE
    job["config_fingerprint"] = appworld.config_fingerprint(appworld.worker_config(config, job))
    job["job_fingerprint"] = appworld.digest_json({key: value for key, value in job.items() if key != "job_fingerprint"})

    async def execute(received, selected):
        assert selected.llm.default_model == selected.llm.expected_model == "deepseek-flash"
        assert selected.llm.base_url == appworld.DEEPSEEK_BASE_URL
        assert selected.llm.request_budget_path == ledger.path
        return {"run_id": "limited", "reason": "llm_rate_limited"}, _EventAccumulator()

    monkeypatch.setattr(worker, "get_config", lambda: pytest.fail("ordinary credentials forbidden"))
    monkeypatch.setattr(worker, "load_deepseek_config", lambda: config)
    monkeypatch.setattr(worker, "_execute_job", execute)
    result = await worker.run_job(job)
    assert result["infrastructure_error"] is True
    assert result["model_protocol"]["provider_profile"] == appworld.DEEPSEEK_PROFILE
    assert "unit-deepseek-placeholder" not in json.dumps(result) + json.dumps(job)
    assert ledger.counts()["real"] == 1


@pytest.mark.parametrize("failure", [None, "service", "mcp", "save", "close", "runtime", "database"])
async def test_worker_cleanup_attempts_every_independent_step_and_never_leaks_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARS_HOME", str(tmp_path / "trusted-home"))
    config = TarsConfig()
    config.llm.api_key = "unit-test-placeholder"
    config.llm.request_budget_path = tmp_path / "ledger.sqlite3"
    job = make_job(config, tmp_path)
    calls: list[str] = []
    seen_prompts: list[str] = []
    monkeypatch.setattr(worker, "get_config", lambda: config)

    def api(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        body = json.loads(request.content)
        assert body["task_id"] == "synthetic_1"
        if path == "/initialize":
            assert body["load_ground_truth"] is False
            assert body["raise_on_unsafe_execution"] and body["raise_on_unsafe_syntax"]
            return httpx.Response(200, json={"output": {"task_id": "synthetic_1", "instruction": "synthetic",
                                                       "ground_truth": "DO-NOT-SHOW-THIS"}})
        return httpx.Response(500 if path == "/" + str(failure) else 200, json={"output": None})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(api), **kwargs,
    ))

    class Provider:
        async def chat(self, messages: Any, **kwargs: Any) -> LlmResponse:
            seen_prompts.append(json.dumps(messages))
            return LlmResponse(stop_reason="end_turn", text="done")

    monkeypatch.setattr(worker, "AgentRunner", lambda selected, **kwargs: AgentRunner(
        selected, provider=Provider(), **kwargs,
    ))

    class TrackedRuntime(FakeRuntime):
        async def cleanup(self) -> None:
            calls.append("runtime")
            if failure == "runtime":
                raise RuntimeError("synthetic runtime cleanup failure")

    monkeypatch.setattr(worker, "build_runtime_router", lambda sandbox: RuntimeRouter(
        TrackedRuntime(), allow_host_fallback=False,
    ))

    class TrackedDatabase(Database):
        async def dispose(self) -> None:
            calls.append("database")
            await super().dispose()
            if failure == "database":
                raise RuntimeError("synthetic database close failure")

    monkeypatch.setattr(worker, "Database", TrackedDatabase)

    class TrackedService(RuntimeService):
        async def shutdown(self) -> None:
            calls.append("service")
            await super().shutdown()
            if failure == "service":
                raise RuntimeError("synthetic service shutdown failure")

    monkeypatch.setattr(worker, "RuntimeService", TrackedService)

    class MCP:
        async def start_all(self, servers: Any) -> None:
            calls.append("mcp-start")

        def get_tools(self) -> list[McpTool]:
            return [McpTool(SimpleNamespace(), "appworld", McpToolDef(  # type: ignore[arg-type]
                name="appworld_execute", description="synthetic", input_schema={"type": "object"},
            ))]

        async def stop_all(self) -> None:
            calls.append("mcp")
            if failure == "mcp":
                raise RuntimeError("synthetic MCP shutdown failure")

    monkeypatch.setattr(worker, "McpServerManager", MCP)
    result = await worker.run_job(job)
    assert {"service", "mcp", "/close", "runtime", "database"} <= set(calls)
    assert result["saved"] is (failure not in {"service", "mcp", "save"})
    assert result["closed"] is (failure != "close")
    assert bool(result["cleanup_errors"]) is (failure is not None)
    assert seen_prompts and all("DO-NOT-SHOW-THIS" not in prompt for prompt in seen_prompts)
    assert result["model_requests"]["reserved"] == 0
    assert result["usage_complete"] is False  # This scripted provider made no HTTP requests.
