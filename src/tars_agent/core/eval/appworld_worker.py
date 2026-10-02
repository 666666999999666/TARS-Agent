"""One isolated TARS process per AppWorld attempt; never runs an upstream Agent."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import httpx

from tars_agent.core.config import McpServerConfig, TarsConfig, get_config
from tars_agent.core.eval.appworld import (
    DEEPSEEK_PROFILE,
    config_fingerprint,
    dependency_fingerprint,
    digest_json,
    load_deepseek_config,
    model_protocol_binding,
    module_argv,
    prompt_binding,
    worker_config,
    write_json,
)
from tars_agent.core.eval.appworld_bridge import loopback_url
from tars_agent.core.eval.appworld_prompts import TOOL_NAME, task_prompt, template_sha256
from tars_agent.core.eval.internal import _EventAccumulator
from tars_agent.core.eval.models import UsageMetrics
from tars_agent.core.eval.runner import utc_now
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.budget import BudgetTransport
from tars_agent.core.mcp.server import McpServerManager
from tars_agent.core.permissions.manager import PermissionManager
from tars_agent.core.permissions.policy import PermissionDecision, ToolPolicy
from tars_agent.core.persistence.cost_budget import CostLedger
from tars_agent.core.persistence.database import Database
from tars_agent.core.persistence.request_budget import (
    ModelRequestBudgetExceeded,
    RequestKind,
    RequestLedger,
)
from tars_agent.core.runner import AgentRunner
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.tools.runtime import RuntimeRouter, build_runtime_router


class _CountingLedger(RequestLedger):
    def __init__(self, inner: RequestLedger, counter: ProcessRequestAccounting) -> None:
        super().__init__(inner.path, limit=inner.limit)
        self._inner = inner
        self._counter = counter

    def reserve(self, kind: RequestKind = "real") -> int:
        if kind != "real":
            return self._inner.reserve(kind)
        self._counter.begin()
        outcome = "reservation_errors"
        try:
            result = self._inner.reserve(kind)
            outcome = "reserved"
            return result
        except ModelRequestBudgetExceeded:
            outcome = "budget_denied"
            raise
        finally:
            self._counter.finish(outcome)


class ProcessRequestAccounting:
    """Instrument this worker's existing transport without changing provider APIs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._counts = {
            "attempted": 0,
            "reserved": 0,
            "budget_denied": 0,
            "reservation_errors": 0,
            "inflight": 0,
        }
        self._original: Any = None

    def begin(self) -> None:
        with self._lock:
            self._counts["attempted"] += 1
            self._counts["inflight"] += 1
            self._idle.clear()

    def finish(self, outcome: str) -> None:
        with self._lock:
            self._counts[outcome] += 1
            self._counts["inflight"] -= 1
            if not self._counts["inflight"]:
                self._idle.set()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def transport(
        self, inner: httpx.AsyncBaseTransport, ledger: RequestLedger, *, kind: RequestKind = "real",
        cost_ledger: CostLedger | None = None,
    ) -> BudgetTransport:
        return BudgetTransport(
            inner, _CountingLedger(ledger, self), kind=kind, cost_ledger=cost_ledger,
        )

    def __enter__(self) -> ProcessRequestAccounting:
        from tars_agent.core.llm import provider

        self._original = getattr(provider, "BudgetTransport")
        setattr(provider, "BudgetTransport", self.transport)
        return self

    def __exit__(self, *args: Any) -> None:
        from tars_agent.core.llm import provider

        setattr(provider, "BudgetTransport", self._original)

    async def wait_idle(self) -> bool:
        return await asyncio.to_thread(self._idle.wait, 6.0)

    def usage(self, counter: _EventAccumulator) -> UsageMetrics:
        counts = self.snapshot()
        # A retried or interrupted HTTP request with no usage must never disappear from totals.
        if (
            counts["inflight"]
            or counts["reservation_errors"]
            or counts["reserved"] != counter.usage_events
        ):
            return UsageMetrics()
        return counter.usage()


async def run_job(job: dict[str, Any]) -> dict[str, Any]:
    actual_job_hash = digest_json(
        {key: value for key, value in job.items() if key != "job_fingerprint"}
    )
    if job.get("job_fingerprint") != actual_job_hash:
        raise ValueError("AppWorld job binding changed")
    variant = str(job.get("prompt_variant", "A"))
    if ("prompt_template_sha256" in job
            and job["prompt_template_sha256"] != template_sha256(variant)):
        raise ValueError("AppWorld completion prompt differs from the frozen experiment")
    environment = job["python_environment"]
    if (
        dependency_fingerprint(Path(environment["site_packages"]))
        != environment["dependencies_digest"]
    ):
        raise ValueError("worker dependencies differ from the frozen experiment")
    # Credentials stay in trusted configuration. Endpoint and execution settings are hashed.
    profile = job.get("profile")
    if profile not in {None, DEEPSEEK_PROFILE}:
        raise ValueError("unknown AppWorld model profile")
    if profile == DEEPSEEK_PROFILE:
        base_config = load_deepseek_config()
    else:
        base_config = get_config()
    config = worker_config(base_config, job)
    if config_fingerprint(config) != job.get("config_fingerprint"):
        raise ValueError("trusted worker configuration differs from the frozen experiment")
    if not (config.llm.api_key or config.llm.anthropic_api_key):
        raise RuntimeError("trusted model credentials are unavailable")
    with ProcessRequestAccounting() as accounting:
        result, counter = await _execute_job(job, config)
        reservations_finished = await accounting.wait_idle()
        usage = accounting.usage(counter) if reservations_finished else UsageMetrics()
        confirmed_usage = (
            UsageMetrics(
                input_tokens=counter.input_tokens,
                output_tokens=counter.output_tokens,
                cache_read_input_tokens=counter.cache_read_input_tokens,
                cache_creation_input_tokens=counter.cache_creation_input_tokens,
            )
            if counter.usage_events else UsageMetrics()
        )
        result.update(
            usage=usage.model_dump(mode="json"),
            confirmed_usage=confirmed_usage.model_dump(mode="json"),
            model_requests=accounting.snapshot(),
            usage_complete=usage.input_tokens is not None,
            model_calls_started=counter.model_calls_started,
            model_responses=counter.usage_events,
            model_protocol=model_protocol_binding(config),
            **prompt_binding(job),
        )
        result["agent_started"] = bool(result.get("run_id") and accounting.snapshot()["reserved"])
        result["infrastructure_error"] = bool(
            not result["agent_started"]
            or result.get("error")
            or result.get("cleanup_errors")
            or result.get("reason") in {
                "llm_rate_limited", "llm_model_mismatch", "llm_request_budget_exhausted",
            }
            or (not counter.usage_events and str(result.get("reason") or "").startswith("llm_"))
        )
    return result


async def _execute_job(
    job: dict[str, Any], config: TarsConfig
) -> tuple[dict[str, Any], _EventAccumulator]:
    root = Path(job["attempt_root"]).resolve()
    job_prompt_sha256: str | None = None
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    home = root / "home"
    home.mkdir()
    os.environ["TARS_HOME"] = str(home)
    os.chdir(workspace)
    url = loopback_url(job["environment_url"])
    task_id, experiment = str(job["task_id"]), str(job["experiment_name"])
    started_at = utc_now()
    start_clock = asyncio.get_running_loop().time()
    bus = EventBus()
    counter = _EventAccumulator()
    bus.subscribe(counter.record)
    database = Database(home / "state.db")
    runtime: RuntimeRouter | None = None
    permissions = PermissionManager(
        policies={TOOL_NAME: ToolPolicy(default=PermissionDecision.ALLOW)}
    )
    mcp = McpServerManager()
    service: RuntimeService | None = None
    initialized = saved = closed = False
    run_id: str | None = None
    status = "not_started"
    reason: str | None = None
    result_error: str | None = None
    cleanup_errors: list[str] = []
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        try:
            await database.create_schema()
            runtime = build_runtime_router(config.sandbox)
            response = await client.post(
                url + "/initialize",
                json={
                    "task_id": task_id,
                    "experiment_name": experiment,
                    "load_ground_truth": False,
                    "raise_on_unsafe_syntax": True,
                    "raise_on_unsafe_execution": True,
                    "max_interactions": int(job["max_steps"]),
                    "timeout_seconds": 120,
                },
            )
            response.raise_for_status()
            initialized = True
            task = response.json()["output"]
            if task.get("task_id") != task_id:
                raise ValueError("initialized world belongs to a different task")
            public_task = {
                key: task[key]
                for key in (
                    "task_id",
                    "instruction",
                    "supervisor",
                    "datetime",
                )
                if key in task
            }
            prompt = task_prompt(public_task, str(job.get("prompt_variant", "A")))
            job_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            skill_dir = workspace / ".tars" / "skills"
            skill_dir.mkdir(parents=True)
            (skill_dir / "appworld-eval.md").write_text(
                "---\nname: appworld-eval\ndescription: Isolated AppWorld task\n"
                f"allowed_tools:\n  - {TOOL_NAME}\n---\n$ARGUMENTS\n",
                encoding="utf-8",
            )
            bridge_command = module_argv(
                "tars_agent.core.eval.appworld_bridge",
                ["--environment-url", url, "--task-id", task_id],
                job["python_environment"],
            )
            await mcp.start_all(
                [
                    McpServerConfig(
                        name="appworld",
                        transport="stdio",
                        trusted=True,
                        command=bridge_command[0],
                        args=bridge_command[1:],
                        tool_timeout_s=180,
                    )
                ]
            )
            if {tool.name for tool in mcp.get_tools()} != {TOOL_NAME}:
                raise RuntimeError("AppWorld bridge did not expose exactly its permitted tool")
            service = RuntimeService(
                database,
                lambda: AgentRunner(
                    config,
                    bus=bus,
                    permission_manager=permissions,
                    tool_runtime=runtime,
                    mcp_manager=mcp,
                ),
                bus,
                artifacts_root=root / "tars-artifacts",
                tool_runtime=runtime,
                llm_config=config.llm,
            )
            session = await service.create_session("chat", workspace_root=workspace)
            submitted = await service.submit_message(session.id, "/appworld-eval " + prompt)
            run_id = submitted.run_id
            try:
                async with asyncio.timeout(float(job["task_timeout_s"])):
                    await service.supervisor.wait(run_id)
            except TimeoutError:
                service.supervisor.cancel(run_id)
                await service.supervisor.wait(run_id)
                reason = "task_timeout"
            snapshot = await service.get_run(run_id)
            status, reason = snapshot.status, reason or snapshot.reason
        except Exception as exc:
            result_error = f"{type(exc).__name__}: {exc}"
        finally:

            async def clean(label: str, operation: Any) -> bool:
                try:
                    await operation
                    return True
                except Exception as exc:
                    cleanup_errors.append(f"{label}: {type(exc).__name__}: {exc}")
                    return False

            quiescent = True
            if service is not None:
                quiescent = await clean("runtime shutdown", service.shutdown())
            bridge_stopped = await clean("MCP shutdown", mcp.stop_all())
            if initialized:
                if quiescent and bridge_stopped:
                    try:
                        response = await client.post(url + "/save", json={"task_id": task_id})
                        response.raise_for_status()
                        saved = True
                    except Exception as exc:
                        cleanup_errors.append(f"world save: {type(exc).__name__}: {exc}")
                else:
                    cleanup_errors.append("world save skipped: execution shutdown is unconfirmed")
                try:
                    response = await client.post(url + "/close", json={"task_id": task_id})
                    response.raise_for_status()
                    closed = True
                except Exception as exc:
                    cleanup_errors.append(f"world close: {type(exc).__name__}: {exc}")
            if runtime is not None:
                await clean("tool runtime cleanup", runtime.cleanup())
            await clean("database dispose", database.dispose())
    result = {
        "task_id": task_id,
        "worker_pid": os.getpid(),
        "experiment_name": experiment,
        "job_fingerprint": job["job_fingerprint"],
        "config_fingerprint": job["config_fingerprint"],
        "run_id": run_id,
        "run_submitted": run_id is not None,
        "run_terminal_status": status,
        "reason": reason,
        "error": result_error or ("; ".join(cleanup_errors) if cleanup_errors else None),
        "cleanup_errors": cleanup_errors,
        "initialized": initialized,
        "saved": saved,
        "closed": closed,
        "started_at": started_at,
        "finished_at": utc_now(),
        "latency_ms": int((asyncio.get_running_loop().time() - start_clock) * 1000),
        "tools": counter.tools().model_dump(mode="json"),
        "prompt_sha256": job_prompt_sha256,
    }
    return result, counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    output = args.output.resolve()
    try:
        result = asyncio.run(run_job(job))
    except Exception as exc:
        result = {
            "task_id": job["task_id"],
            "worker_pid": os.getpid(),
            "experiment_name": job["experiment_name"],
            "job_fingerprint": job["job_fingerprint"],
            **prompt_binding(job),
            "saved": False,
            "closed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(output, result)


if __name__ == "__main__":
    main()
