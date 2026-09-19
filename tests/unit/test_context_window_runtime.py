from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event, select

from tars_agent.core.bus.envelope import HandlerError
from tars_agent.core.bus.events import RunFinishedEvent
from tars_agent.core.config import TarsConfig, _apply_env, _apply_toml, _validate_config
from tars_agent.core.eval.provenance import safe_config_snapshot
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from tars_agent.core.persistence import Database, MessageRecord, StateRepository
from tars_agent.core.runner import AgentRunner
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter


class _CapturingProvider:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[LlmResponse] = []

    async def chat(self, messages, tool_schemas, bus, run_id, *, step=0, system=None):
        self.requests.append(copy.deepcopy({
            "messages": messages, "tools": tool_schemas, "system": system,
            "run_id": run_id, "step": step,
        }))
        if self.responses:
            return self.responses.pop(0)
        return LlmResponse(stop_reason="end_turn", text="fixture answer")


class _RuntimeCase:
    def __init__(self, database: Database, root: Path) -> None:
        self.database = database
        self.root = root
        self.config = TarsConfig()
        self.config.compaction.auto_threshold = 0
        self.config.llm.max_tokens = 64
        # Setting these on the existing dataclass also lets the original code run
        # the red tests; it ignores them instead of failing at fixture construction.
        self.config.llm.context_budget_tokens = 500_000
        self.config.llm.context_safety_margin = 64
        self.provider = _CapturingProvider()
        self.bus = EventBus()
        self.runtime = self.new_runtime()
        self.session_id = ""
        self.original_ids: list[int] = []
        self.before: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []

    def new_runtime(self) -> RuntimeService:
        runtime = RuntimeService(
            self.database,
            lambda: AgentRunner(self.config, bus=self.bus,
                                tool_runtime=RuntimeRouter(FakeRuntime(), allow_host_fallback=False),
                                provider=self.provider),
            self.bus, artifacts_root=self.root / "artifacts",
            compaction_provider_factory=lambda: self.provider,
        )
        # Same LlmConfig instance that production passes to Runtime and Runner.
        runtime._llm_config = self.config.llm
        return runtime

    async def reopen(self) -> None:
        await self.runtime.shutdown()
        await self.database.dispose()
        self.database = Database(self.root / "state.db")
        self.bus = EventBus()
        self.runtime = self.new_runtime()

    async def seed(self, messages: list[dict[str, Any]]) -> None:
        async with self.database.transaction() as sql:
            rows = [MessageRecord(session_id=self.session_id, sequence=index,
                                  role=message["role"], content=message["content"],
                                  committed=message.get("committed", True),
                                  active=message.get("active", True))
                    for index, message in enumerate(messages)]
            await StateRepository(sql).add_messages(rows)
            self.original_ids = [row.id for row in rows]
        self.before = await self.snapshot()

    async def snapshot(self) -> list[dict[str, Any]]:
        async with self.database.session() as sql:
            rows = list(await sql.scalars(select(MessageRecord).where(
                MessageRecord.id.in_(self.original_ids),
            ).order_by(MessageRecord.sequence)))
            return [{"id": row.id, "sequence": row.sequence, "role": row.role,
                     "content": copy.deepcopy(row.content), "committed": row.committed,
                     "active": row.active, "session_id": row.session_id} for row in rows]

    async def run(self, goal: str = "CURRENT_INPUT", *, tool: str | None = None):
        skill = self.root / ".tars/skills/fixture.md"
        skill.parent.mkdir(parents=True, exist_ok=True)
        tools = f"\n  - {tool}" if tool else ""
        skill.write_text(f"---\nname: fixture\nallowed_tools:{tools}\n---\n$ARGUMENTS\n",
                         encoding="utf-8")
        _effective, options = self.runtime._resolve_skill(f"/fixture {goal}", workspace_root=self.root)
        assert options["tool_whitelist"] == ([tool] if tool else [])
        finished = asyncio.Event()

        async def observe(event):
            if isinstance(event, RunFinishedEvent):
                finished.set()

        listener = self.bus.subscribe(observe)
        waiter = asyncio.create_task(finished.wait())
        try:
            submitted = await self.runtime.submit_message(self.session_id, f"/fixture {goal}")
            _done, pending = await asyncio.wait([waiter], timeout=10)
            assert not pending, "Runtime did not finish before test cleanup"
            run = await self.runtime.get_run(submitted.run_id)
            self.runs.append({"run_id": run.id, "status": run.status,
                              "reason": run.reason, "result": run.result})
            return run
        finally:
            listener.unsubscribe()
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)


@pytest.fixture
async def runtime_case(tmp_path: Path, request) -> AsyncIterator[_RuntimeCase]:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    case = _RuntimeCase(database, tmp_path)
    session = await case.runtime.create_session("chat", workspace_root=tmp_path)
    case.session_id = session.id
    try:
        yield case
    finally:
        await case.runtime.shutdown()
        after = await case.snapshot()
        destination = os.environ.get("CONTEXT_TEST_EVIDENCE_DIR")
        if destination:
            output = Path(destination)
            output.mkdir(parents=True, exist_ok=True)
            def digest(rows):
                return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            projections = []
            estimates = []
            for model_request in case.provider.requests:
                fixed = json.dumps({key: model_request[key] for key in ("system", "tools")},
                                   ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                estimate = 64 + len(fixed) + sum(
                    32 + len(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                    for message in model_request["messages"]
                )
                allowance = (case.config.llm.context_budget_tokens - case.config.llm.max_tokens
                             - case.config.llm.context_safety_margin)
                estimates.append({"estimated_input": estimate, "input_allowance": allowance,
                                  "fits_local_policy": estimate <= allowance,
                                  "method": "utf8_json_bytes_plus_message_framing_v1"})
                projected = []
                for index, message in enumerate(model_request["messages"]):
                    blocks = message["content"] if isinstance(message["content"], list) else []
                    projected.append({
                        "request_message_index": index,
                        "matching_original_sequences": [row["sequence"] for row in case.before
                            if row["role"] == message["role"] and row["content"] == message["content"]],
                        "tool_use_ids": [b.get("id") for b in blocks if b.get("type") == "tool_use"],
                        "tool_result_ids": [b.get("tool_use_id") for b in blocks if b.get("type") == "tool_result"],
                    })
                projections.append(projected)
            evidence = {"fixture": request.node.name, "budget": {
                "context_budget_tokens": case.config.llm.context_budget_tokens,
                "max_tokens": case.config.llm.max_tokens,
                "safety_margin": case.config.llm.context_safety_margin,
            }, "requests": case.provider.requests, "request_mappings": projections,
                "request_estimates": estimates, "provider_kind": "deterministic_test_double",
                "runs": case.runs,
                "history_before": case.before, "history_after": after,
                "history_before_sha256": digest(case.before), "history_after_sha256": digest(after)}
            (output / f"{request.node.name}.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        await case.database.dispose()


def _history(count: int) -> list[dict[str, Any]]:
    return [{"role": "user" if index % 2 == 0 else "assistant", "content": f"history-{index}"}
            for index in range(count)]


def _assert_pairs(messages: list[dict[str, Any]]) -> list[dict[str, list[str]]]:
    pending: list[str] = []
    mapping = []
    for message in messages:
        content = message["content"]
        blocks = content if isinstance(content, list) else []
        uses = [block["id"] for block in blocks if block.get("type") == "tool_use"]
        results = [block["tool_use_id"] for block in blocks if block.get("type") == "tool_result"]
        if pending:
            assert message["role"] == "user"
            assert len(results) == len(set(results)) and set(results) == set(pending)
            mapping.append({"calls": pending, "results": results})
            pending = []
        else:
            assert not results, "orphan tool result reached Provider"
        if uses:
            assert message["role"] == "assistant" and len(uses) == len(set(uses))
            pending = uses
    assert not pending, "tool calls without results reached Provider"
    return mapping


@pytest.mark.parametrize("count", [1, 1999, 2000, 2001, 2002])
async def test_runtime_latest_history_reaches_provider_at_page_boundaries(runtime_case, count):
    await runtime_case.seed(_history(count))
    run = await runtime_case.run()
    assert run.status == "succeeded", run.reason
    sent = runtime_case.provider.requests[0]["messages"]
    assert any(message["content"] == f"history-{count - 1}" for message in sent)
    assert [message["content"] for message in sent[:-1]] == [f"history-{i}" for i in range(count)]
    assert "CURRENT_INPUT" in sent[-1]["content"]
    assert await runtime_case.snapshot() == runtime_case.before
    # Public browsing still exposes the oldest page in chronological order.
    async with runtime_case.database.session() as sql:
        first = await StateRepository(sql).list_messages(runtime_case.session_id)
        remaining = await StateRepository(sql).list_messages(
            runtime_case.session_id, after_sequence=first[-1].sequence,
        )
    assert [row.sequence for row in first] == list(range(min(count + 2, 2000)))
    assert [row.sequence for row in remaining] == list(range(min(count + 2, 2000), count + 2))


@pytest.mark.parametrize("calls", [1, 3])
@pytest.mark.parametrize("start", [1872, 1998], ids=["reverse-page-boundary", "old-2000-boundary"])
async def test_tool_groups_cross_pages_without_losing_id_pairs(runtime_case, calls, start):
    messages = _history(2002)
    ids = [f"specific-tool-{index}" for index in range(calls)]
    messages[start:start + 4] = [
        {"role": "user", "content": "LATEST_CONSTRAINT: keep exact identifiers"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": name, "name": "read_file", "input": {"path": "fixture"}}
            for name in ids]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": name, "content": f"result-{name}"}
            for name in reversed(ids)]},
        {"role": "assistant", "content": "group complete"},
    ]
    await runtime_case.seed(messages)
    assert (await runtime_case.run()).status == "succeeded"
    mapping = _assert_pairs(runtime_case.provider.requests[0]["messages"])
    assert {tuple(pair["calls"]) for pair in mapping} == {tuple(ids)}
    assert await runtime_case.snapshot() == runtime_case.before


async def test_tight_budget_keeps_latest_constraints_and_not_old_large_groups(runtime_case):
    runtime_case.config.llm.context_budget_tokens = 2200
    await runtime_case.seed([
        {"role": "user", "content": "OLD_OMITTED " + "x" * 6000},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "LATEST_CONSTRAINT: answer in Chinese"},
        {"role": "assistant", "content": "understood"},
    ])
    assert (await runtime_case.run()).status == "succeeded"
    sent = runtime_case.provider.requests[0]
    serialized = json.dumps(sent["messages"], ensure_ascii=False)
    assert "LATEST_CONSTRAINT" in serialized and "CURRENT_INPUT" in serialized
    assert "OLD_OMITTED" not in serialized
    assert len(json.dumps({key: sent[key] for key in ("messages", "tools", "system")},
                          ensure_ascii=False).encode()) <= 2200 - 64 - 64
    assert await runtime_case.snapshot() == runtime_case.before


@pytest.mark.parametrize("part", ["latest-group", "current-input", "system", "tool-schema"])
async def test_mandatory_content_over_budget_fails_before_provider(runtime_case, part):
    runtime_case.config.llm.context_budget_tokens = 1600 if part != "tool-schema" else 900
    goal = "CURRENT_INPUT"
    history = _history(2)
    if part == "latest-group":
        history[0]["content"] = "LATEST_REQUIRED " + "界" * 1200
    elif part == "current-input":
        goal += "界" * 1200
    elif part == "system":
        context_path = runtime_case.root / ".tars/context.md"
        context_path.parent.mkdir(parents=True)
        context_path.write_text("WORKSPACE_CONSTRAINT " + "x" * 5000, encoding="utf-8")
    await runtime_case.seed(history)
    run = await runtime_case.run(goal, tool="read_file" if part == "tool-schema" else None)
    assert run.status == "failed"
    assert "context_budget_exceeded" in (run.reason or "")
    assert not runtime_case.provider.requests
    assert await runtime_case.snapshot() == runtime_case.before


async def test_new_large_tool_result_is_not_sent_or_silently_truncated(runtime_case):
    runtime_case.config.llm.context_budget_tokens = 3000
    runtime_case.config.compaction.tool_result_limit = 8
    runtime_case.config.compaction.tool_result_keep = 4
    (runtime_case.root / "large.txt").write_text("x" * 12000, encoding="utf-8")
    runtime_case.provider.responses = [LlmResponse(stop_reason="tool_use", tool_calls=[
        ToolCallBlock(id="oversized-result-id", name="read_file", input={"path": "large.txt"}),
    ])]
    await runtime_case.seed(_history(2))
    run = await runtime_case.run(tool="read_file")
    assert run.status == "failed" and "context_budget_exceeded" in (run.reason or "")
    assert len(runtime_case.provider.requests) == 1
    assert [tool["name"] for tool in runtime_case.provider.requests[0]["tools"]] == ["read_file"]
    async with runtime_case.database.session() as sql:
        audit = await StateRepository(sql).list_run_messages(run.id)
    results = [block for row in audit if isinstance(row.content, list)
               for block in row.content if block.get("type") == "tool_result"]
    assert results and results[0]["tool_use_id"] == "oversized-result-id"
    assert len(results[0]["content"]) > 3000
    assert all(not row.committed for row in audit)
    assert await runtime_case.snapshot() == runtime_case.before
    assert (await runtime_case.run("RECOVER_AFTER_BUDGET_FAILURE")).status == "succeeded"
    assert len(runtime_case.provider.requests) == 2
    assert "oversized-result-id" not in json.dumps(runtime_case.provider.requests[-1])


async def test_runtime_excludes_inactive_uncommitted_and_other_sessions(runtime_case):
    await runtime_case.seed([
        {"role": "user", "content": "INACTIVE_SECRET", "active": False},
        {"role": "assistant", "content": "FAILED_AUDIT", "committed": False},
        {"role": "user", "content": "CANCELLED_AUDIT", "committed": False},
        {"role": "user", "content": "LATEST_FORMAL"},
        {"role": "assistant", "content": "formal reply"},
    ])
    other = await runtime_case.runtime.create_session("chat", workspace_root=runtime_case.root)
    async with runtime_case.database.transaction() as sql:
        await StateRepository(sql).add_message(MessageRecord(
            session_id=other.id, sequence=0, role="user", content="OTHER_SESSION_SECRET",
            committed=True, active=True,
        ))
    assert (await runtime_case.run()).status == "succeeded"
    sent = json.dumps(runtime_case.provider.requests[0], ensure_ascii=False)
    assert "LATEST_FORMAL" in sent
    assert all(value not in sent for value in (
        "INACTIVE_SECRET", "FAILED_AUDIT", "CANCELLED_AUDIT", "OTHER_SESSION_SECRET",
    ))
    assert await runtime_case.snapshot() == runtime_case.before


async def test_manual_compaction_reads_the_full_bounded_history_and_restores(runtime_case):
    await runtime_case.seed(_history(2002))
    runtime_case.provider.responses = [LlmResponse(stop_reason="end_turn", text="SAVED_SUMMARY")]
    await runtime_case.runtime.compact_session(runtime_case.session_id)
    assert "history-2001" in json.dumps(runtime_case.provider.requests[0]["messages"])
    after = await runtime_case.snapshot()
    assert len(after) == 2002 and all(not row["active"] for row in after)
    assert [{k: v for k, v in row.items() if k != "active"} for row in after] == [
        {k: v for k, v in row.items() if k != "active"} for row in runtime_case.before
    ]
    await runtime_case.reopen()
    await runtime_case.runtime.resume_session(runtime_case.session_id)
    assert (await runtime_case.run("AFTER_RESTORE")).status == "succeeded"
    sent = runtime_case.provider.requests[-1]["messages"]
    assert sent[0]["content"] == "SAVED_SUMMARY"
    assert "AFTER_RESTORE" in sent[-1]["content"]
    assert "history-" not in json.dumps(sent)


@pytest.mark.parametrize("source", ["toml", "environment"])
async def test_legacy_truncation_options_parse_but_warn_and_are_not_effective(source, caplog):
    config = TarsConfig()
    if source == "toml":
        _apply_toml(config, {"compaction": {"tool_result_limit": 123, "tool_result_keep": 12}},
                    trusted=True)
    else:
        _apply_env(config, {"TARS_COMPACT_TOOL_LIMIT": "123", "TARS_COMPACT_TOOL_KEEP": "12"})
    assert config.compaction.tool_result_limit == 123 and config.compaction.tool_result_keep == 12
    assert "deprecated" in caplog.text and "not applied" in caplog.text
    snapshot = safe_config_snapshot(config)
    assert "tool_result_limit" not in snapshot["compaction"]
    assert "tool_result_keep" not in snapshot["compaction"]
    assert snapshot["ignored_config"]["compaction.tool_result_limit"] == 123


def test_default_example_does_not_recommend_unused_truncation_options():
    example = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
    assert "TARS_COMPACT_TOOL_LIMIT=" not in example
    assert "TARS_COMPACT_TOOL_KEEP=" not in example


@pytest.mark.parametrize("mismatch", ["wrong-id", "duplicate-id", "missing-result"])
async def test_equal_counts_or_old_results_cannot_replace_exact_tool_id_pairs(runtime_case, mismatch):
    results = ["call-a", "call-b"]
    if mismatch == "wrong-id":
        results = ["call-a", "call-from-another-step"]
    elif mismatch == "duplicate-id":
        results = ["call-a", "call-a"]
    else:
        results = ["call-a"]
    await runtime_case.seed([
        {"role": "user", "content": "LATEST_REQUIRED"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": name, "name": "read_file", "input": {}}
            for name in ["call-a", "call-b"]]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": name, "content": "synthetic"}
            for name in results]},
    ])
    run = await runtime_case.run()
    assert run.status == "failed" and "context_history_invalid" in (run.reason or "")
    assert not runtime_case.provider.requests
    assert await runtime_case.snapshot() == runtime_case.before


async def test_budget_keeps_global_workspace_notes_and_current_input(runtime_case):
    runtime_case.config.llm.context_budget_tokens = 2400
    home = Path(os.environ["TARS_HOME"])
    assert home.is_relative_to(runtime_case.root)
    (home / "context.md").write_text("GLOBAL: approvals remain required", encoding="utf-8")
    project = runtime_case.root / ".tars/context.md"
    project.parent.mkdir(parents=True)
    project.write_text("WORKSPACE: never write outside this workspace", encoding="utf-8")
    runtime_case.runtime._artifact_store.append_note(runtime_case.session_id, "NOTES: use UTC", "test")
    await runtime_case.seed([
        {"role": "user", "content": "OMIT_OLD " + "x" * 9000},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "LATEST: answer in Chinese"},
        {"role": "assistant", "content": "ack"},
    ])
    assert (await runtime_case.run()).status == "succeeded"
    sent = runtime_case.provider.requests[0]
    assert all(value in sent["system"] for value in ("GLOBAL:", "WORKSPACE:", "NOTES:"))
    assert "LATEST:" in json.dumps(sent["messages"]) and "CURRENT_INPUT" in sent["messages"][-1]["content"]
    assert not sent["tools"]
    assert await runtime_case.snapshot() == runtime_case.before


async def test_automatic_compaction_retains_current_instruction_and_history_audit(runtime_case):
    runtime_case.config.compaction.auto_threshold = 0.8
    (runtime_case.root / "small.txt").write_text("small fixture", encoding="utf-8")
    await runtime_case.seed(_history(4))
    runtime_case.provider.responses = [
        LlmResponse(stop_reason="tool_use", tool_calls=[
            ToolCallBlock(id="read-before-summary", name="read_file", input={"path": "small.txt"}),
        ], usage=UsageStats(100, 20, context_pct=0.9)),
        LlmResponse(stop_reason="end_turn", text="AUTO_SUMMARY"),
        LlmResponse(stop_reason="end_turn", text="finished after summary"),
    ]
    assert (await runtime_case.run("EXACT_CURRENT_CONSTRAINT", tool="read_file")).status == "succeeded"
    assert len(runtime_case.provider.requests) == 3
    sent = runtime_case.provider.requests[-1]["messages"]
    assert sent[0]["content"] == "AUTO_SUMMARY"
    assert sent[-1]["content"] == "EXACT_CURRENT_CONSTRAINT"
    _assert_pairs(sent)
    after = await runtime_case.snapshot()
    assert len(after) == len(runtime_case.before) and all(not row["active"] for row in after)
    assert [row["content"] for row in after] == [row["content"] for row in runtime_case.before]
    await runtime_case.reopen()
    assert (await runtime_case.run("AFTER_REOPEN")).status == "succeeded"
    assert runtime_case.provider.requests[-1]["messages"][0]["content"] == "AUTO_SUMMARY"


async def test_partial_history_never_marks_unseen_records_as_compacted(runtime_case):
    runtime_case.config.compaction.auto_threshold = 0.8
    runtime_case.config.llm.context_budget_tokens = 2400
    (runtime_case.root / "small.txt").write_text("ok", encoding="utf-8")
    await runtime_case.seed([
        {"role": "user", "content": "UNSEEN_OLD " + "x" * 10000},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "LATEST_MUST_STAY"},
        {"role": "assistant", "content": "ack"},
    ])
    runtime_case.provider.responses = [
        LlmResponse(stop_reason="tool_use", tool_calls=[
            ToolCallBlock(id="partial-history-read", name="read_file", input={"path": "small.txt"}),
        ], usage=UsageStats(100, 20, context_pct=0.9)),
        LlmResponse(stop_reason="end_turn", text="done"),
    ]
    assert (await runtime_case.run(tool="read_file")).status == "succeeded"
    assert len(runtime_case.provider.requests) == 2
    assert all(item["run_id"] != "compact" for item in runtime_case.provider.requests)
    assert await runtime_case.snapshot() == runtime_case.before


async def test_manual_compaction_over_budget_keeps_database_and_never_calls_provider(runtime_case):
    runtime_case.config.llm.context_budget_tokens = 1800
    await runtime_case.seed([
        {"role": "user", "content": "must not summarize a partial prefix " + "x" * 9000},
        {"role": "assistant", "content": "original reply"},
    ])
    with pytest.raises(HandlerError, match="context_budget_exceeded"):
        await runtime_case.runtime.compact_session(runtime_case.session_id)
    assert not runtime_case.provider.requests
    assert await runtime_case.snapshot() == runtime_case.before


async def test_ignored_legacy_limit_does_not_replace_effective_builtin_output_limit(runtime_case):
    runtime_case.config.compaction.tool_result_limit = 8
    runtime_case.config.compaction.tool_result_keep = 4
    (runtime_case.root / "large.txt").write_text("x" * 100_000, encoding="utf-8")
    runtime_case.provider.responses = [LlmResponse(stop_reason="tool_use", tool_calls=[
        ToolCallBlock(id="bounded-read", name="read_file", input={"path": "large.txt"}),
    ])]
    await runtime_case.seed(_history(2))
    assert (await runtime_case.run(tool="read_file")).status == "succeeded"
    results = runtime_case.provider.requests[1]["messages"][-1]["content"]
    assert results[0]["tool_use_id"] == "bounded-read"
    assert 8 < len(results[0]["content"]) < 100_000
    assert "truncated" in results[0]["content"]
    _assert_pairs(runtime_case.provider.requests[1]["messages"])
    assert await runtime_case.snapshot() == runtime_case.before


def test_context_budget_configuration_is_explicit_and_endpoint_window_is_unverified():
    config = TarsConfig()
    _apply_toml(config, {"llm": {"max_tokens": 128, "context_budget_tokens": 5000,
                                  "context_safety_margin": 256}}, trusted=True)
    _validate_config(config)
    snapshot = safe_config_snapshot(config)
    assert snapshot["llm"]["context_budget_tokens"] == 5000
    assert snapshot["context_budget"]["endpoint_window_verified"] is False
    _apply_env(config, {"TARS_LLM_CONTEXT_BUDGET_TOKENS": "6000",
                        "TARS_LLM_CONTEXT_SAFETY_MARGIN": "512"})
    _validate_config(config)
    assert config.llm.context_budget_tokens == 6000 and config.llm.context_safety_margin == 512


@pytest.mark.parametrize("budget,margin", [(0, 10), (100, -1), (100, 50), (True, 0)])
def test_invalid_or_fully_reserved_context_budget_is_rejected(budget, margin):
    config = TarsConfig()
    config.llm.max_tokens = 64
    with pytest.raises(SystemExit, match="Config error"):
        _apply_toml(config, {"llm": {"context_budget_tokens": budget,
                                      "context_safety_margin": margin}}, trusted=True)
        _validate_config(config)


async def test_recent_loading_stops_after_a_bounded_page_when_budget_is_tight(runtime_case):
    runtime_case.config.llm.context_budget_tokens = 1800
    await runtime_case.seed(_history(2002))
    pages = []

    def observe_query(connection, cursor, statement, parameters, context, many):
        if "ORDER BY messages.sequence DESC" in statement:
            pages.append((statement, parameters))

    engine = runtime_case.database.engine.sync_engine
    event.listen(engine, "before_cursor_execute", observe_query)
    try:
        assert (await runtime_case.run()).status == "succeeded"
    finally:
        event.remove(engine, "before_cursor_execute", observe_query)
    assert len(pages) == 1
    assert "LIMIT" in pages[0][0] and pages[0][1][-2] == 128
    sent = runtime_case.provider.requests[0]["messages"]
    assert len(sent) < 128
    assert any(message["content"] == "history-2001" for message in sent)
    assert await runtime_case.snapshot() == runtime_case.before
