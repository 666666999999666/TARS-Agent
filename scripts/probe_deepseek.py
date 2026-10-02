"""Opt-in DeepSeek official probe with one durable twelve-attempt allowance.

Private synthetic evaluator evidence is retained under --output. Console output and
probe-summary.json contain only validation metadata, never response bodies/headers.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import unicodedata
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tars_agent.core.bus.events import LlmTokenEvent
from tars_agent.core.config import TarsConfig
from tars_agent.core.control import CoreHomeLock
from tars_agent.core.eval.appworld import (
    DEEPSEEK_MODEL,
    DEEPSEEK_PROFILE,
    load_deepseek_config,
)
from tars_agent.core.eval.models import EvalSuiteManifest, TaskAttemptResult
from tars_agent.core.eval.runner import run_eval_suite
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm import provider as provider_module
from tars_agent.core.llm.provider import AnthropicProvider
from tars_agent.core.persistence.request_budget import (
    ModelRequestBudgetExceeded,
    RequestLedger,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGET_PATH = ROOT / "build/internship/deepseek-access-budget.json"
MAX_ATTEMPTS = 12
TOOL_CHECKS = ("read-write-read", "read")
READ_FIXTURE = "来源数据.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_count(path: Path) -> int:
    with closing(sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True)) as connection:
        return int(connection.execute(
            "SELECT count(*) FROM requests WHERE kind='real'",
        ).fetchone()[0])


def ledger_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {"path_sha256": hashlib.sha256(str(resolved).encode()).hexdigest(),
            "device": stat.st_dev, "inode": stat.st_ino}


class ProbeBudget:
    """Lock spans both phases; the original ledger still counts every HTTP attempt."""

    def __init__(self, path: Path, ledger: Path, user_limit: int | None) -> None:
        self.path, self.ledger, self.user_limit = path, ledger.resolve(), user_limit
        self.lock = CoreHomeLock(path.parent / "deepseek-access-lock")
        self.mutex = threading.RLock()
        self.record: dict[str, Any] = {}
        self.limit = 0

    def __enter__(self) -> ProbeBudget:
        self.lock.acquire()
        try:
            identity, count = ledger_identity(self.ledger), read_count(self.ledger)
            sentinel = self.lock.home / "initialized"
            if self.path.exists():
                record = json.loads(self.path.read_text(encoding="utf-8"))
                if (record.get("version") != 1 or record.get("profile") != DEEPSEEK_PROFILE
                        or record.get("ledger") != identity
                        or type(record.get("start_count")) is not int
                        or record["start_count"] < 0
                        or type(record.get("absolute_cap")) is not int
                        or type(record.get("high_water_count")) is not int
                        or record["absolute_cap"] != record["start_count"] + MAX_ATTEMPTS
                        or record["high_water_count"] < record["start_count"]
                        or count < record["high_water_count"]):
                    raise ValueError("probe budget or original request ledger changed")
                self.record = record
            else:
                if sentinel.exists():
                    raise ValueError("initialized probe budget is missing; refusing a new allowance")
                self.record = {"version": 1, "profile": DEEPSEEK_PROFILE, "ledger": identity,
                               "start_count": count, "absolute_cap": count + MAX_ATTEMPTS,
                               "high_water_count": count}
                write_json(self.path, self.record)
            sentinel.touch(exist_ok=True)
            self.limit = min(self.record["absolute_cap"], self.user_limit or self.record["absolute_cap"])
            self.checkpoint()
            return self
        except BaseException:
            self.lock.release()
            raise

    def checkpoint(self) -> int:
        with self.mutex:
            if ledger_identity(self.ledger) != self.record["ledger"]:
                raise ValueError("original request ledger was replaced")
            count = read_count(self.ledger)
            if count < self.record["high_water_count"]:
                raise ValueError("original request ledger regressed")
            self.record["high_water_count"] = count
            write_json(self.path, self.record)
            return count

    def require_remaining(self) -> None:
        if self.checkpoint() >= self.limit:
            raise ModelRequestBudgetExceeded("probe stage cumulative request allowance exhausted")

    def summary(self) -> dict[str, object]:
        count = self.checkpoint()
        return {"start_count": self.record["start_count"],
                "absolute_cap": self.record["absolute_cap"], "effective_cap": self.limit,
                "current_count": count, "cumulative_used": count - self.record["start_count"],
                "remaining": max(0, self.limit - count),
                "record_version": self.record["version"], "profile": DEEPSEEK_PROFILE}

    def __exit__(self, *exc: object) -> None:
        self.lock.release()

    def ledger_type(self) -> type[RequestLedger]:
        budget = self

        class BoundLedger(RequestLedger):
            def __init__(self, path: Path, *, limit: int | None = None) -> None:
                if path.resolve() != budget.ledger:
                    raise ValueError("probe attempted to use another request ledger")
                super().__init__(path, limit=min(limit or budget.limit, budget.limit))

            def _connect(self) -> sqlite3.Connection:
                if ledger_identity(self.path) != budget.record["ledger"]:
                    raise ValueError("original request ledger was replaced")
                return sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=5)

            def reserve(self, kind: str = "real") -> int:
                if kind != "real":
                    raise ValueError("remote probe attempts must count as real requests")
                with budget.mutex, closing(self._connect()) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    count = int(connection.execute(
                        "SELECT count(*) FROM requests WHERE kind='real'",
                    ).fetchone()[0])
                    if count < budget.record["high_water_count"]:
                        raise ValueError("original request ledger regressed")
                    if count >= self.limit:
                        raise ModelRequestBudgetExceeded("probe stage request allowance exhausted")
                    # Write the conservative watermark first. A crash between these commits
                    # fails closed on restart rather than silently granting another attempt.
                    budget.record["high_water_count"] = count + 1
                    write_json(budget.path, budget.record)
                    connection.execute(
                        "INSERT INTO requests(kind,reserved_at) VALUES ('real',?)",
                        (datetime.now(UTC).isoformat(),),
                    )
                    connection.commit()
                    return count + 1

        return BoundLedger


def safe_error(exc: BaseException, key: str) -> dict[str, object]:
    # Only the short exception message, never repr(request), headers, or a traceback.
    message = str(exc).replace(key, "[REDACTED]") if key else str(exc)
    message = re.sub(r"(?i)(authorization|x-api-key|api[_-]?key)\s*[:=]\s*\S+",
                     r"\1=[REDACTED]", message)
    return {"type": type(exc).__name__, "message": message[:500]}


def safe_provider_error(exc: BaseException, key: str) -> dict[str, object]:
    """Extract diagnostic scalars without serializing SDK bodies or headers."""
    def safe_text(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        value = "".join(char for char in value if unicodedata.category(char) not in {"Cc", "Cf", "Cs"})
        if key:
            value = value.replace(key, "[REDACTED]")
        value = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)
        value = re.sub(r"(?i)\b[0-9a-f]{32}\.[A-Za-z0-9_-]+", "[REDACTED]", value)
        value = re.sub(r"(?i)(authorization|x-api-key|api[_-]?key)\s*[:=]\s*\S+",
                       r"\1=[REDACTED]", value)
        return value[:500] if value else None

    result: dict[str, object] = {"type": safe_error(exc, key)["type"],
                                "http_status": None, "service_error_type": None,
                                "service_error_code": None, "service_error_message": None,
                                "retry_after": None, "cause_types": []}
    current: BaseException | None = exc
    seen: set[int] = set()
    causes: list[str] = []
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        if current is not exc:
            causes.append(type(current).__name__)
        status = getattr(current, "status_code", None)
        if result["http_status"] is None and type(status) is int:
            result["http_status"] = status
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        if result["retry_after"] is None and headers is not None:
            result["retry_after"] = safe_text(headers.get("retry-after"))
        for field in ("expected_model", "actual_model"):
            if field not in result and hasattr(current, field):
                value = getattr(current, field)
                if isinstance(value, str):
                    value = value.replace(key, "[REDACTED]") if key else value
                    result[field] = value if re.fullmatch(r"[A-Za-z0-9_./:\[\]-]{1,128}", value) else None
                else:
                    # Missing/non-string service identity remains unknown. In
                    # particular, never fill it using the requested model name.
                    result[field] = None
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            error = body.get("error", body)
            if result["service_error_message"] is None:
                message = error.get("message") if isinstance(error, dict) else None
                if not isinstance(message, str):
                    message = body.get("message")
                result["service_error_message"] = safe_text(message)
            if isinstance(error, dict):
                for field in ("type", "code"):
                    value = error.get(field)
                    target = f"service_error_{field}"
                    if result[target] is None and type(value) in (str, int):
                        value = str(value).replace(key, "[REDACTED]") if key else str(value)
                        # Codes are identifiers, not a place to copy arbitrary remote text.
                        if re.fullmatch(r"[A-Za-z0-9_.:\[\]-]{1,128}", value):
                            result[target] = value
        current = current.__cause__
    result["cause_types"] = causes
    return result


@contextmanager
def quiet_libraries():
    # Production failure logging includes SDK tracebacks; the probe emits its own
    # redacted exception summary and keeps normal synthetic runtime evidence.
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def synthetic_suite(*, tool_check: str = "read-write-read",
                    max_output_tokens: int | None = None,
                    attempts: int | None = None) -> tuple[EvalSuiteManifest, dict[str, str]]:
    if tool_check not in TOOL_CHECKS:
        raise ValueError("unknown synthetic tool check")
    expected = {"marker": secrets.token_hex(16), "text": "读取成功，中文内容必须完整保留。"}
    fixture = READ_FIXTURE if tool_check == "read" else "source.json"
    goal = (
        f"Use read_file once to read {READ_FIXTURE}. Then in your final answer repeat both "
        "the marker and text values exactly as read. Do not write or modify any file."
        if tool_check == "read" else
        "Read source.json with read_file. Copy the entire JSON object exactly "
        "to result.json using write_file. Then read result.json with read_file "
        "to verify it. Do these three operations in order, waiting for each "
        "result. In your final answer include the marker read from the file. "
        "Do not modify source.json."
    )
    suite = EvalSuiteManifest.model_validate({
        "suite_id": "deepseek-access-tools", "name": "DeepSeek synthetic file tool loop",
        "adapter": "internal", "execution_mode": "agent_tasks", "default_repetitions": 1,
        "model_config_ref": {"source": "runtime_config", "reference": DEEPSEEK_PROFILE,
                             "provider": "anthropic-compatible", "model": DEEPSEEK_MODEL},
        "tasks": [{"id": tool_check, "timeout_s": 600, "goal": goal,
                   "tool_whitelist": ["read_file"] if tool_check == "read" else ["read_file", "write_file"],
                   "fixture_files": {fixture: json.dumps(expected, ensure_ascii=False)},
                   "grader": ({"kind": "output_contains", "expected": expected["marker"]}
                              if tool_check == "read" else
                              {"kind": "json_equals", "path": "result.json", "expected": expected}),
                   "metadata": {"protected_paths": [fixture], "probe_options": {
                       "tool_check": tool_check, "max_output_tokens": max_output_tokens,
                       "attempts": attempts, "max_steps": 8, "timeout_s": 600,
                   }}}],
    })
    return suite, expected


def audit_tool_events(database: Path, parent_run_id: str, expected: dict[str, str],
                      *, tool_check: str = "read-write-read") -> dict:
    """Require actual sandbox execution and paired success, in durable cursor order."""
    with closing(sqlite3.connect(database.resolve(strict=True).as_uri() + "?mode=ro", uri=True)) as connection:
        runs = connection.execute("SELECT id,parent_run_id,status FROM runs").fetchall()
        if runs != [(parent_run_id, None, "succeeded")]:
            raise ValueError("synthetic parent did not finish normally as the only Run")
        rows = connection.execute(
            "SELECT cursor,event_type,payload FROM events ORDER BY cursor",
        ).fetchall()
    calls: dict[str, dict] = {}
    completed: list[dict] = []
    for cursor, event_type, payload in rows:
        if not event_type.startswith("tool."):
            continue
        event = json.loads(payload)
        if event.get("run_id") != parent_run_id:
            raise ValueError("unexpected tool Run")
        if event_type not in {"tool.call_started", "tool.execution_started", "tool.call_finished"}:
            raise ValueError("unsuccessful or unexpected tool event")
        tool_id, name = event["tool_use_id"], event["tool_name"]
        if event_type == "tool.call_started":
            if tool_id in calls or name not in {"read_file", "write_file"}:
                raise ValueError("unexpected or duplicate tool call")
            calls[tool_id] = {"tool_use_id": tool_id, "name": name, "params": event["params"],
                              "start": cursor}
        else:
            call = calls.get(tool_id)
            if call is None or call["name"] != name or "finish" in call:
                raise ValueError("unpaired tool event")
            if event_type == "tool.execution_started":
                if "execution" in call or event.get("backend") != "workspace_sandbox":
                    raise ValueError("tool did not execute once in the sandbox")
                call["execution"] = cursor
            else:
                if "execution" not in call:
                    raise ValueError("success without execution")
                call.update(finish=cursor, output=event.get("output", ""))
                completed.append(call)
    if tool_check not in TOOL_CHECKS:
        raise ValueError("unknown synthetic tool check")
    sequence = ([("read_file", READ_FIXTURE)] if tool_check == "read" else
                [("read_file", "source.json"), ("write_file", "result.json"),
                 ("read_file", "result.json")])
    if len(calls) != len(sequence) or len(completed) != len(sequence):
        raise ValueError(f"expected exactly {len(sequence)} completed tool calls")
    previous = -1
    proof = []
    for call, (name, path) in zip(completed, sequence, strict=True):
        raw_path = call["params"].get("path")
        normalized = re.sub(r"^(?:\./)+", "", raw_path) if isinstance(raw_path, str) else None
        if (call["name"] != name or normalized != path
                or not previous < call["start"] < call["execution"] < call["finish"]):
            raise ValueError("tool path or order does not prove the selected tool check")
        content = call["params"].get("content") if name == "write_file" else call["output"]
        if json.loads(content) != expected:
            raise ValueError("tool contents do not match the protected fixture")
        previous = call["finish"]
        proof.append({key: call[key] for key in ("tool_use_id", "start", "execution", "finish")}
                                | {"tool": name, "path": path, "json_matches": True})
    result = {"source": "persisted_state_db_events", "parent_succeeded": True, "sequence": proof}
    if tool_check == "read":
        after_read = [(cursor, event_type) for cursor, event_type, payload in rows
                      if cursor > previous and event_type in {"llm.model_selected", "llm.usage"}
                      and json.loads(payload).get("run_id") == parent_run_id]
        selected = next((cursor for cursor, kind in after_read if kind == "llm.model_selected"), None)
        usage = next((cursor for cursor, kind in after_read if kind == "llm.usage"
                      and selected is not None and cursor > selected), None)
        if usage is None:
            raise ValueError("no completed model response after the actual file read")
        result["followup_model_response"] = {"selected_cursor": selected, "usage_cursor": usage}
    return result


def validate_attempt(attempt: TaskAttemptResult, expected: dict[str, str],
                     *, tool_check: str = "read-write-read") -> dict:
    confirmation = attempt.evaluation.get("runtime_cleanup_confirmation")
    if (attempt.status != "passed" or attempt.run_terminal_status != "success"
            or attempt.evaluation.get("tree_terminal") is not True
            or attempt.cleanup.runtime_cleanup_completed is not True
            or attempt.cleanup.workspace_removed is not True
            or not isinstance(confirmation, dict) or confirmation.get("confirmed") is not True
            or confirmation.get("source") != "docker_instance_inventory"
            or not confirmation.get("scope_id") or confirmation.get("remaining_resource_ids") != []
            or attempt.collateral_damage is not False
            or expected["marker"] not in attempt.output
            or (tool_check == "read" and expected["text"] not in attempt.output)):
        raise ValueError("tool phase lacks success, marker, protected-file or physical-cleanup proof")
    evidence = Path(str(attempt.evaluation["evidence_directory"]))
    for name in ((READ_FIXTURE,) if tool_check == "read" else ("source.json", "result.json")):
        if json.loads((evidence / "workspace" / name).read_text(encoding="utf-8")) != expected:
            raise ValueError("saved synthetic files differ from expected")
    proof = audit_tool_events(evidence / "state/state.db",
                              str(attempt.evaluation["parent_run_id"]), expected,
                              tool_check=tool_check)
    return {"status": "passed", "event_proof": proof, "marker_in_final_answer": True,
            "tool_check": tool_check, "write_verified": tool_check == "read-write-read",
            "chinese_value_in_final_answer": expected["text"] in attempt.output,
            "cleanup": confirmation, "workspace_removed": True,
            "usage": attempt.usage.model_dump(mode="json"), "evidence_directory": str(evidence)}


async def cancellation_probe(config: TarsConfig, budget: ProbeBudget) -> dict:
    budget.require_remaining()
    before = budget.checkpoint()
    provider = AnthropicProvider.from_config(config.llm)
    bus = EventBus()
    first_token = False
    cancelled = False
    error = None
    task: asyncio.Task | None = None

    async def on_event(event):
        nonlocal first_token
        if isinstance(event, LlmTokenEvent) and not first_token:
            first_token = True
            assert task is not None
            task.cancel()

    bus.subscribe(on_event)
    try:
        task = asyncio.create_task(provider.chat(
            [{"role": "user", "content": "Write the integers from 1 through 100, one per line."}],
            [], bus, "deepseek-access-cancel", system="Follow the synthetic formatting request.",
        ))
        try:
            await task
        except asyncio.CancelledError:
            # External cancellation is not evidence that our token handler cancelled it.
            if not first_token:
                raise
            cancelled = True
        except Exception as exc:
            error = safe_error(exc, config.llm.api_key)
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await provider.close()
    closed = provider._client.is_closed() is True
    after = budget.checkpoint()
    passed = first_token and cancelled and closed and after == before + 1 and error is None
    return {"status": "passed" if passed else "failed" if first_token else "unverified",
            "first_token_observed": first_token, "cancelled_error_observed": cancelled,
            "client_closed": closed, "request_count": after - before, "no_resend": after == before + 1,
            "actual_model": None, "actual_model_note": "cancelled before complete response",
            "usage": None, "thinking_blocks_present": None, "error": error}


async def execute(output: Path, config: TarsConfig, budget: ProbeBudget,
                  *, tool_check: str = "read-write-read") -> dict:
    config.agent.max_steps = 8
    config.sandbox.mode = "required"
    config.trace.include_llm_payload = False
    config.llm.request_limit = budget.limit
    expected_model = DEEPSEEK_MODEL
    summary: dict[str, Any] = {"status": "failed", "profile": DEEPSEEK_PROFILE,
                               "expected_model": expected_model, "requested_model": config.llm.default_model,
                               "protocol": "anthropic_messages", "base_url": config.llm.base_url,
                               "tool_check": tool_check, "probe_options": {
                                   "tool_check": tool_check, "max_output_tokens": config.llm.max_tokens,
                                   "attempts": config.llm.attempts, "max_steps": 8, "timeout_s": 600,
                               },
                               "tool_phase": None, "cancellation_phase": {"status": "not_run"}}
    observations = []
    provider_errors = []
    original = AnthropicProvider.chat

    async def observed_chat(self, *args, **kwargs):
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            provider_errors.append(safe_provider_error(exc, config.llm.api_key))
            raise
        observations.append({"actual_model": response.model,
                             "thinking_blocks_present": bool(response.thinking_blocks),
                             "usage": (None if response.usage is None else {
                                 "input_tokens": response.usage.input_tokens,
                                 "output_tokens": response.usage.output_tokens,
                                 "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                                 "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
                             })})
        return response

    try:
        budget.require_remaining()
        if config.llm.expected_model != expected_model or config.llm.default_model != expected_model:
            raise ValueError("probe configuration does not match the selected profile model")
        suite, expected = synthetic_suite(tool_check=tool_check, max_output_tokens=config.llm.max_tokens,
                                         attempts=config.llm.attempts)
        manifest = output / "synthetic-suite.json"
        write_json(manifest, suite.model_dump(mode="json"))
        with patch.object(provider_module, "RequestLedger", budget.ledger_type()):
            before = budget.checkpoint()
            with patch.object(AnthropicProvider, "chat", observed_chat):
                result = await run_eval_suite(manifest, output / "private-evidence", config,
                                              repository_root=ROOT)
            if len(result.attempts) != 1:
                raise ValueError("synthetic evaluator returned an unexpected number of attempts")
            attempt = result.attempts[0]
            summary["tool_phase"] = {
                "status": "failed", "eval_status": str(attempt.status),
                "run_terminal_status": attempt.run_terminal_status,
                "request_count": budget.checkpoint() - before,
                "evidence_directory": attempt.evaluation.get("evidence_directory"),
                "cleanup": attempt.evaluation.get("runtime_cleanup_confirmation"),
                "workspace_removed": attempt.cleanup.workspace_removed,
                "error_type": attempt.error_type,
                "error": safe_error(RuntimeError(attempt.error), config.llm.api_key)
                if attempt.error else None,
            }
            summary["tool_phase"] = validate_attempt(result.attempts[0], expected, tool_check=tool_check)
            summary["tool_phase"]["request_count"] = budget.checkpoint() - before
            if (len(observations) < (2 if tool_check == "read" else 1)
                    or any(row["actual_model"] != expected_model for row in observations)):
                raise ValueError("service model identity was not confirmed by completed responses")
            if budget.checkpoint() < budget.limit:
                summary["cancellation_phase"] = await cancellation_probe(config, budget)
            else:
                summary["cancellation_phase"] = {"status": "unverified", "reason": "budget_exhausted"}
            if summary["cancellation_phase"]["status"] == "passed":
                summary["status"] = "passed"
    except Exception as exc:
        summary["error"] = safe_error(exc, config.llm.api_key)
    finally:
        summary["completed_response_observations"] = observations
        summary["provider_errors"] = provider_errors
        try:
            summary["budget"] = budget.summary()
        except Exception as exc:
            summary["status"] = "failed"
            summary["budget_error"] = safe_error(exc, config.llm.api_key)
        write_json(output / "probe-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-env", type=Path)
    parser.add_argument("--tool-check", choices=TOOL_CHECKS, default="read-write-read")
    parser.add_argument("--max-output-tokens", type=int, help="Probe-only output token limit")
    parser.add_argument("--attempts", type=int, choices=(1, 2), help="Probe-only transport attempt count")
    parser.add_argument("--image", help="Optional pinned existing sandbox image (sha256 full ID)")
    args = parser.parse_args(argv)
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if not args.execute:
        print(json.dumps({"status": "not_executed", "requires": ["--execute"],
                          "profile": DEEPSEEK_PROFILE,
                          "tool_check": args.tool_check, "max_output_tokens": args.max_output_tokens,
                          "attempts": args.attempts,
                          "maximum_cumulative_http_attempts": MAX_ATTEMPTS}))
        return 0
    if args.output is None:
        parser.error("--execute requires a fresh --output under build/internship")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "build/internship") or output.exists():
        parser.error("--output must be a fresh directory under build/internship")
    output.mkdir(parents=True, exist_ok=False)
    key = ""
    try:
        config = copy.deepcopy(load_deepseek_config(private_env=args.private_env))
        key = config.llm.api_key
        if args.max_output_tokens is not None:
            config.llm.max_tokens = args.max_output_tokens
        if args.attempts is not None:
            config.llm.attempts = args.attempts
        if args.image:
            if re.fullmatch(r"sha256:[0-9a-f]{64}", args.image) is None:
                raise ValueError("--image requires a full sha256 image ID")
            config.sandbox.image = args.image
        with ProbeBudget(BUDGET_PATH, config.llm.request_budget_path,
                         config.llm.request_limit) as budget, quiet_libraries():
            summary = asyncio.run(execute(output, config, budget, tool_check=args.tool_check))
    except Exception as exc:
        summary = {"status": "blocked", "error": safe_error(exc, key)}
        write_json(output / "probe-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
