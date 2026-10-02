from __future__ import annotations

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from scripts import probe_deepseek as probe

from tars_agent.core.bus.events import LlmTokenEvent
from tars_agent.core.config import TarsConfig
from tars_agent.core.llm.types import LlmResponse
from tars_agent.core.persistence.request_budget import (
    ModelRequestBudgetExceeded,
    RequestLedger,
)


def ledger_file(tmp_path: Path, initial: int = 0) -> Path:
    path = tmp_path / "original.sqlite3"
    ledger = RequestLedger(path, limit=None)
    ledger.counts()
    for _ in range(initial):
        ledger.reserve()
    return path


def test_durable_allowance_is_not_renewed_by_new_output_or_restart(tmp_path: Path):
    ledger = ledger_file(tmp_path, initial=5)
    saved = tmp_path / "access.json"
    with probe.ProbeBudget(saved, ledger, None) as budget:
        assert budget.summary()["absolute_cap"] == 17
        bound = budget.ledger_type()(ledger)
        assert bound.reserve() == 6
        assert bound.reserve() == 7
    with probe.ProbeBudget(saved, ledger, None) as budget:
        assert budget.summary()["remaining"] == 10
        for _ in range(10):
            budget.ledger_type()(ledger).reserve()
        with pytest.raises(ModelRequestBudgetExceeded):
            budget.ledger_type()(ledger).reserve()
    with probe.ProbeBudget(saved, ledger, None) as budget:
        with pytest.raises(ModelRequestBudgetExceeded):
            budget.require_remaining()
    assert probe.read_count(ledger) == 17


def test_smaller_user_cumulative_limit_wins(tmp_path: Path):
    ledger = ledger_file(tmp_path, 5)
    with probe.ProbeBudget(tmp_path / "budget.json", ledger, 6) as budget:
        bound = budget.ledger_type()(ledger, limit=1000)
        bound.reserve()
        with pytest.raises(ModelRequestBudgetExceeded):
            bound.reserve()
        assert budget.summary()["absolute_cap"] == 17


def test_lock_rejects_concurrent_probe_and_releases_after_failure(tmp_path: Path):
    ledger = ledger_file(tmp_path)
    path = tmp_path / "budget.json"
    with probe.ProbeBudget(path, ledger, None):
        with pytest.raises(RuntimeError, match="exclusive lock"):
            with probe.ProbeBudget(path, ledger, None):
                pytest.fail("second probe acquired the same OS lock")
    with probe.ProbeBudget(path, ledger, None) as budget:
        assert budget.lock.acquired


def test_threaded_http_reservations_cannot_exceed_cap(tmp_path: Path):
    ledger = ledger_file(tmp_path)
    with probe.ProbeBudget(tmp_path / "budget.json", ledger, None) as budget:
        bound = budget.ledger_type()(ledger)

        def reserve():
            try:
                return bound.reserve()
            except ModelRequestBudgetExceeded:
                return None

        with ThreadPoolExecutor(max_workers=6) as pool:
            values = list(pool.map(lambda _: reserve(), range(24)))
        assert sum(value is not None for value in values) == 12
        assert probe.read_count(ledger) == 12


@pytest.mark.parametrize("damage", ["missing_record", "missing_ledger", "replaced_ledger", "regressed", "changed_cap"])
def test_restart_refuses_ledger_reset_or_corrupt_budget(tmp_path: Path, damage: str):
    ledger = ledger_file(tmp_path)
    path = tmp_path / "budget.json"
    with probe.ProbeBudget(path, ledger, None) as budget:
        budget.ledger_type()(ledger).reserve()
    if damage == "missing_record":
        path.unlink()
    elif damage == "missing_ledger":
        ledger.unlink()
    elif damage == "replaced_ledger":
        replacement = tmp_path / "replacement"
        replacement.mkdir()
        ledger_file(replacement, 1).replace(ledger)
    elif damage == "regressed":
        with sqlite3.connect(ledger) as connection:
            connection.execute("DELETE FROM requests")
    else:
        record = json.loads(path.read_text())
        record["absolute_cap"] += 1
        probe.write_json(path, record)
    with pytest.raises((ValueError, FileNotFoundError)):
        with probe.ProbeBudget(path, ledger, None):
            pytest.fail("damaged original evidence must fail closed")


def test_no_execute_does_not_read_credentials_create_output_or_call_provider(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("non-executing preview must not read credentials or call model")

    monkeypatch.setattr(probe, "load_deepseek_config", forbidden)
    monkeypatch.setattr(probe.AnthropicProvider, "from_config", forbidden)
    output = tmp_path / "no-output"
    assert probe.main(["--output", str(output), "--private-env", "nonexistent"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "not_executed"
    assert not output.exists()


@pytest.mark.parametrize("lightweight", [False, True])
def test_cli_uses_only_deepseek_loader_and_new_fixed_budget(tmp_path, monkeypatch, lightweight):
    ledger = ledger_file(tmp_path, 20)
    config = configuration(ledger)
    calls = []

    def loader(**kwargs):
        calls.append(kwargs)
        return config

    async def fake_execute(output, passed_config, budget, *, tool_check):
        assert passed_config is not config
        assert passed_config.llm.max_tokens == (256 if lightweight else config.llm.max_tokens)
        assert passed_config.llm.attempts == (1 if lightweight else 2)
        assert tool_check == ("read" if lightweight else "read-write-read")
        assert budget.path == tmp_path / "deepseek-access-budget.json"
        assert budget.summary()["profile"] == probe.DEEPSEEK_PROFILE
        return {"status": "passed", "profile": probe.DEEPSEEK_PROFILE}

    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "BUDGET_PATH", tmp_path / "deepseek-access-budget.json")
    monkeypatch.setattr(probe, "load_deepseek_config", loader)
    monkeypatch.setattr(probe, "execute", fake_execute)
    output = tmp_path / "build/internship/new-output"
    options = ["--tool-check", "read", "--max-output-tokens", "256", "--attempts", "1"] if lightweight else []
    assert probe.main(["--execute", "--output", str(output), *options]) == 0
    assert calls == [{"private_env": None}]
    assert config.llm.max_tokens == 8192 and config.llm.attempts == 2
    record = json.loads((tmp_path / "deepseek-access-budget.json").read_text())
    assert record["profile"] == probe.DEEPSEEK_PROFILE
    assert record["start_count"] == 20 and record["absolute_cap"] == 32
    assert "profile_history" not in record and "migrated_from_version" not in record
    assert probe.read_count(ledger) == 20


def test_new_allowance_preserves_historical_evidence_without_loading_it(tmp_path):
    ledger = ledger_file(tmp_path, 20)
    # These intentionally invalid JSON files prove historical reports are opaque:
    # the current single-profile probe has no parser or migration for them.
    historical = [tmp_path / "archived-budget-one.json", tmp_path / "archived-budget-two.json"]
    for path in historical:
        path.write_bytes(b"opaque historical evidence")
    path = tmp_path / "deepseek-access-budget.json"
    with probe.ProbeBudget(path, ledger, None) as budget:
        assert budget.record["start_count"] == 20 and budget.limit == 32
        budget.ledger_type()(ledger).reserve()
    with probe.ProbeBudget(path, ledger, None) as budget:
        assert budget.summary()["remaining"] == 11
    assert all(item.read_bytes() == b"opaque historical evidence" for item in historical)


def test_foreign_profile_budget_is_rejected_without_migration(tmp_path):
    ledger = ledger_file(tmp_path, 3)
    path = tmp_path / "deepseek-access-budget.json"
    record = {"version": 1, "profile": "archived-other-provider",
              "ledger": probe.ledger_identity(ledger), "start_count": 3,
              "absolute_cap": 15, "high_water_count": 3}
    probe.write_json(path, record)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        with probe.ProbeBudget(path, ledger, None):
            pytest.fail("foreign profile must not migrate to a new allowance")
    assert path.read_bytes() == before and probe.read_count(ledger) == 3


def test_deepseek_manifest_has_fixed_official_model_and_label():
    suite, expected = probe.synthetic_suite(max_output_tokens=2048, attempts=1)
    assert suite.suite_id == "deepseek-access-tools"
    assert suite.name == "DeepSeek synthetic file tool loop"
    assert suite.model_config_ref.reference == "deepseek-flash-official-anthropic-v1"
    assert suite.model_config_ref.model == "deepseek-flash"
    assert expected["marker"] not in suite.tasks[0].goal


def events_database(tmp_path: Path, expected: dict, *, tool_check="read-write-read") -> Path:
    path = tmp_path / "state.db"
    content = json.dumps(expected, ensure_ascii=False)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE runs(id TEXT,parent_run_id TEXT,status TEXT)")
        connection.execute("INSERT INTO runs VALUES ('parent',NULL,'succeeded')")
        connection.execute("CREATE TABLE events(cursor INTEGER,event_type TEXT,payload TEXT)")
        cursor = 0
        sequence = [("read_file", probe.READ_FIXTURE)] if tool_check == "read" else [
            ("read_file", "source.json"), ("write_file", "result.json"), ("read_file", "result.json"),
        ]
        for index, (name, filename) in enumerate(sequence):
            params = {"path": filename}
            if name == "write_file":
                params["content"] = content
            for event_type in ("tool.call_started", "tool.execution_started", "tool.call_finished"):
                cursor += 1
                event = {"run_id": "parent", "tool_use_id": f"id{index}", "tool_name": name,
                         "params": params, "backend": "workspace_sandbox", "output": content}
                connection.execute("INSERT INTO events VALUES (?,?,?)", (cursor, event_type, json.dumps(event)))
        if tool_check == "read":
            for event_type in ("llm.model_selected", "llm.usage"):
                cursor += 1
                connection.execute("INSERT INTO events VALUES (?,?,?)", (cursor, event_type, json.dumps({"run_id": "parent"})))
    return path


def test_suite_does_not_give_marker_or_chinese_answer_in_goal():
    suite, expected = probe.synthetic_suite()
    task = suite.tasks[0]
    assert expected["marker"] not in task.goal and expected["text"] not in task.goal
    assert task.timeout_s == 600
    assert task.tool_whitelist == ["read_file", "write_file"]
    assert task.metadata["protected_paths"] == ["source.json"]
    assert task.grader.kind == "json_equals"


def test_persisted_matching_sandbox_events_prove_loop(tmp_path):
    expected = {"marker": "fixture-only", "text": "中文"}
    path = events_database(tmp_path, expected)
    proof = probe.audit_tool_events(path, "parent", expected)
    assert [item["path"] for item in proof["sequence"]] == ["source.json", "result.json", "result.json"]


@pytest.mark.parametrize("prefix,accepted", [("./", True), ("././", True), ("../", False), ("/", False), ("C:/", False), ("nested/../", False)])
def test_only_leading_current_directory_is_normalized(tmp_path, prefix, accepted):
    expected = {"marker": "fixture-only", "text": "中文"}
    path = events_database(tmp_path, expected)
    with sqlite3.connect(path) as connection:
        event = json.loads(connection.execute("SELECT payload FROM events WHERE cursor=1").fetchone()[0])
        event["params"]["path"] = prefix + "source.json"
        connection.execute("UPDATE events SET payload=? WHERE cursor=1", (json.dumps(event),))
    if accepted:
        assert probe.audit_tool_events(path, "parent", expected)["parent_succeeded"] is True
    else:
        with pytest.raises(ValueError):
            probe.audit_tool_events(path, "parent", expected)


@pytest.mark.parametrize("failure", [None, "cleanup_unknown", "cleanup_return_only", "residue", "parent_failed", "missing_marker", "changed_source"])
@pytest.mark.parametrize("tool_check", ["read-write-read", "read"])
def test_attempt_needs_cleanup_identity_success_and_preserved_fixture(tmp_path, failure, tool_check):
    expected = {"marker": "fixture-only", "text": "中文"}
    state = tmp_path / "state"
    state.mkdir()
    events_database(state, expected, tool_check=tool_check)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    filenames = (probe.READ_FIXTURE,) if tool_check == "read" else ("source.json", "result.json")
    for name in filenames:
        (workspace / name).write_text(json.dumps(expected), encoding="utf-8")
    attempt = probe.TaskAttemptResult(
        task_id="synthetic", repetition=1, status="passed", run_terminal_status="success",
        started_at="now", finished_at="now", latency_ms=1, output=expected["marker"] + expected["text"],
        collateral_damage=False,
        cleanup={"runtime_cleanup_completed": True, "workspace_removed": True},
        evaluation={"tree_terminal": True, "evidence_directory": str(tmp_path), "parent_run_id": "parent",
                    "runtime_cleanup_confirmation": {"confirmed": True, "source": "docker_instance_inventory",
                                                     "scope_id": "instance", "remaining_resource_ids": []}},
    )
    if failure == "cleanup_unknown":
        attempt.evaluation["runtime_cleanup_confirmation"]["confirmed"] = None
    elif failure == "cleanup_return_only":
        attempt.evaluation["runtime_cleanup_confirmation"]["source"] = "cleanup_returned"
    elif failure == "residue":
        attempt.evaluation["runtime_cleanup_confirmation"]["remaining_resource_ids"] = ["container"]
    elif failure == "parent_failed":
        attempt.run_terminal_status = "failed"
    elif failure == "missing_marker":
        attempt.output = "I did it"
    elif failure == "changed_source":
        (workspace / filenames[0]).write_text('{}', encoding="utf-8")
    if failure:
        with pytest.raises(ValueError):
            probe.validate_attempt(attempt, expected, tool_check=tool_check)
    else:
        result = probe.validate_attempt(attempt, expected, tool_check=tool_check)
        assert result["status"] == "passed"
        assert result["write_verified"] is (tool_check == "read-write-read")


@pytest.mark.parametrize("forgery", ["missing_execution", "failed_tool", "wrong_content", "wrong_path", "write_tool", "no_followup", "followup_before_read"])
def test_read_mode_requires_real_read_and_later_model_response(tmp_path, forgery):
    expected = {"marker": "unpredictable-fixture", "text": "中文结果"}
    path = events_database(tmp_path, expected, tool_check="read")
    with sqlite3.connect(path) as connection:
        if forgery == "missing_execution":
            connection.execute("DELETE FROM events WHERE cursor=2")
        elif forgery == "failed_tool":
            connection.execute("UPDATE events SET event_type='tool.call_failed' WHERE cursor=3")
        elif forgery == "no_followup":
            connection.execute("DELETE FROM events WHERE event_type LIKE 'llm.%'")
        elif forgery == "followup_before_read":
            connection.execute("UPDATE events SET cursor=cursor-8 WHERE event_type LIKE 'llm.%'")
        else:
            cursor = 3 if forgery == "wrong_content" else 1
            event = json.loads(connection.execute("SELECT payload FROM events WHERE cursor=?", (cursor,)).fetchone()[0])
            if forgery == "wrong_content":
                event["output"] = '{}'
            elif forgery == "wrong_path":
                event["params"]["path"] = "source.json"
            else:
                event["tool_name"] = "write_file"
            connection.execute("UPDATE events SET payload=? WHERE cursor=?", (json.dumps(event), cursor))
    with pytest.raises(ValueError):
        probe.audit_tool_events(path, "parent", expected, tool_check="read")


def test_read_mode_final_answer_must_contain_chinese_fixture_value(tmp_path):
    expected = {"marker": "unpredictable-fixture", "text": "中文结果"}
    attempt = probe.TaskAttemptResult(
        task_id="read", repetition=1, status="passed", run_terminal_status="success",
        started_at="now", finished_at="now", latency_ms=1, output=expected["marker"],
        collateral_damage=False, cleanup={"runtime_cleanup_completed": True, "workspace_removed": True},
        evaluation={"tree_terminal": True, "runtime_cleanup_confirmation": {
            "confirmed": True, "source": "docker_instance_inventory", "scope_id": "instance", "remaining_resource_ids": [],
        }},
    )
    with pytest.raises(ValueError):
        probe.validate_attempt(attempt, expected, tool_check="read")


async def test_read_mode_parameters_saved_in_manifest_and_summary_without_api(tmp_path, monkeypatch):
    from types import SimpleNamespace

    ledger = ledger_file(tmp_path)
    config = configuration(ledger)
    config.llm.max_tokens = 256
    config.llm.attempts = 1
    output = tmp_path / "output"
    output.mkdir()

    async def empty_suite(*args, **kwargs):
        return SimpleNamespace(attempts=[])

    monkeypatch.setattr(probe, "run_eval_suite", empty_suite)
    with probe.ProbeBudget(tmp_path / "budget.json", ledger, None) as budget:
        result = await probe.execute(output, config, budget, tool_check="read")
    suite = json.loads((output / "synthetic-suite.json").read_text(encoding="utf-8"))
    task = suite["tasks"][0]
    expected = json.loads(task["fixture_files"][probe.READ_FIXTURE])
    assert result["tool_check"] == "read"
    assert result["probe_options"] == task["metadata"]["probe_options"] == {
        "tool_check": "read", "max_output_tokens": 256, "attempts": 1, "max_steps": 8, "timeout_s": 600,
    }
    assert task["tool_whitelist"] == ["read_file"]
    assert task["metadata"]["protected_paths"] == [probe.READ_FIXTURE]
    assert expected["marker"] not in task["goal"] and expected["text"] not in task["goal"]
    assert "write_file" not in task["goal"]
    assert result["status"] == "failed" and probe.read_count(ledger) == 0


@pytest.mark.parametrize("options", [["--max-output-tokens", "0"], ["--max-output-tokens", "-1"], ["--attempts", "0"], ["--attempts", "3"], ["--tool-check", "unapproved"]])
def test_invalid_probe_only_controls_are_rejected_before_any_request(options):
    with pytest.raises(SystemExit) as failure:
        probe.main(options)
    assert failure.value.code == 2


@pytest.mark.parametrize("forgery", ["wrong_path", "parallel", "wrong_output", "missing_execution", "failed_tool", "failed_parent", "unexpected_child", "mismatched_id"])
def test_tool_claims_cannot_replace_ordered_success_events(tmp_path, forgery):
    expected = {"marker": "fixture-only", "text": "中文"}
    path = events_database(tmp_path, expected)
    with sqlite3.connect(path) as connection:
        if forgery == "failed_parent":
            connection.execute("UPDATE runs SET status='failed'")
        elif forgery == "unexpected_child":
            connection.execute("INSERT INTO runs VALUES ('child','parent','succeeded')")
        elif forgery == "missing_execution":
            connection.execute("DELETE FROM events WHERE cursor=2")
        elif forgery == "failed_tool":
            connection.execute("UPDATE events SET event_type='tool.call_failed' WHERE cursor=6")
        elif forgery == "parallel":
            connection.execute("UPDATE events SET cursor=-1 WHERE cursor=4")
        else:
            cursor = 1 if forgery == "wrong_path" else 3
            event = json.loads(connection.execute("SELECT payload FROM events WHERE cursor=?", (cursor,)).fetchone()[0])
            if forgery == "wrong_path":
                event["params"]["path"] = "elsewhere.json"
            elif forgery == "wrong_output":
                event["output"] = '{"marker":"model made this up"}'
            else:
                event["tool_use_id"] = "invented"
            connection.execute("UPDATE events SET payload=? WHERE cursor=?", (json.dumps(event), cursor))
    with pytest.raises(ValueError):
        probe.audit_tool_events(path, "parent", expected)


class TokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        events = [
            {"type": "message_start", "message": {"id": "synthetic", "type": "message", "role": "assistant", "model": probe.DEEPSEEK_MODEL,
                "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "1"}},
        ]
        for event in events:
            yield ("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode()
        await asyncio.sleep(30)
        pytest.fail("first-token cancellation must close the stream before completion")


def configuration(ledger: Path) -> TarsConfig:
    config = TarsConfig()
    config.llm.default_model = probe.DEEPSEEK_MODEL
    config.llm.expected_model = probe.DEEPSEEK_MODEL
    config.llm.api_key = "synthetic-private-key"
    config.llm.base_url = "https://example.invalid/anthropic"
    config.llm.request_budget_path = ledger
    config.llm.retry_delay_s = .001
    return config


async def test_actual_sdk_token_cancellation_closes_client_without_retry(tmp_path, monkeypatch):
    ledger = ledger_file(tmp_path)
    received = []

    def handler(request):
        received.append(json.loads(request.content))
        return httpx.Response(200, stream=TokenStream(), headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(probe.provider_module.httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handler))
    with probe.ProbeBudget(tmp_path / "budget.json", ledger, None) as budget:
        with patch.object(probe.provider_module, "RequestLedger", budget.ledger_type()):
            result = await probe.cancellation_probe(configuration(ledger), budget)
    assert result["status"] == "passed"
    assert result["client_closed"] and result["cancelled_error_observed"]
    assert result["request_count"] == 1 and result["actual_model"] is None
    assert len(received) == 1 and "tools" not in received[0]


@pytest.mark.parametrize("status,expected_requests", [(401, 1), (429, 2)])
async def test_failure_before_token_does_not_claim_cancellation(tmp_path, monkeypatch, status, expected_requests):
    ledger = ledger_file(tmp_path)
    monkeypatch.setattr(probe.provider_module.httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(
        lambda request: httpx.Response(status, json={"error": {"type": "api_error", "message": "synthetic-private-key"}})))
    with probe.ProbeBudget(tmp_path / "budget.json", ledger, None) as budget:
        with patch.object(probe.provider_module, "RequestLedger", budget.ledger_type()):
            result = await probe.cancellation_probe(configuration(ledger), budget)
    assert result["status"] == "unverified"
    assert result["request_count"] == expected_requests
    assert result["client_closed"] is True
    assert "synthetic-private-key" not in json.dumps(result)


async def test_application_retry_cannot_send_thirteenth_http_attempt(tmp_path, monkeypatch):
    ledger = ledger_file(tmp_path)
    sent = []

    def handler(request):
        sent.append(1)
        return httpx.Response(429, json={"error": {"type": "rate_limit_error", "message": "busy"}})

    monkeypatch.setattr(probe.provider_module.httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handler))
    with probe.ProbeBudget(tmp_path / "budget.json", ledger, None) as budget:
        for _ in range(11):
            budget.ledger_type()(ledger).reserve()
        with patch.object(probe.provider_module, "RequestLedger", budget.ledger_type()):
            result = await probe.cancellation_probe(configuration(ledger), budget)
    assert result["status"] == "unverified"
    assert len(sent) == 1 and probe.read_count(ledger) == 12


async def test_token_but_ignored_cancel_does_not_pass(tmp_path, monkeypatch):
    ledger = ledger_file(tmp_path)

    class IgnoresCancel:
        def __init__(self):
            self._client = self
        def is_closed(self):
            return True
        async def close(self):
            pass
        async def chat(self, messages, tools, bus, run_id, **kwargs):
            RequestLedger(ledger).reserve()
            await bus.publish(LlmTokenEvent(run_id=run_id, token="synthetic", ts="now"))
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass
            return LlmResponse(stop_reason="end_turn")

    monkeypatch.setattr(probe.AnthropicProvider, "from_config", lambda config: IgnoresCancel())
    with probe.ProbeBudget(tmp_path / "budget.json", ledger, None) as budget:
        result = await probe.cancellation_probe(configuration(ledger), budget)
    assert result["status"] == "failed" and not result["cancelled_error_observed"]


async def test_failed_tool_phase_does_not_start_cancellation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    ledger = ledger_file(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    async def failed_suite(*args, **kwargs):
        return SimpleNamespace(attempts=[])

    async def forbidden_cancel(*args, **kwargs):
        pytest.fail("cancellation must follow a proven tool loop")

    monkeypatch.setattr(probe, "run_eval_suite", failed_suite)
    monkeypatch.setattr(probe, "cancellation_probe", forbidden_cancel)
    with probe.ProbeBudget(tmp_path / "budget.json", ledger, None) as budget:
        result = await probe.execute(output, configuration(ledger), budget)
    assert result["status"] == "failed"
    assert result["cancellation_phase"]["status"] == "not_run"
    assert (output / "probe-summary.json").is_file()
    assert probe.read_count(ledger) == 0


@pytest.mark.parametrize("wrapped", [False, True])
async def test_provider_failure_metadata_redacts_key_and_preserves_exception(tmp_path, monkeypatch, wrapped):
    from types import SimpleNamespace

    import anthropic

    ledger = ledger_file(tmp_path)
    config = configuration(ledger)
    output = tmp_path / "output"
    output.mkdir()
    response = httpx.Response(500, request=httpx.Request("POST", "https://example.invalid"),
                              headers={"authorization": "secret-header-do-not-save"})
    http_error = anthropic.InternalServerError(
        "sensitive-body-do-not-save " + config.llm.api_key, response=response,
        body={"error": {"type": "api_error", "code": config.llm.api_key,
                        "message": "request failed " + config.llm.api_key},
              "private_details": "sensitive-body-do-not-save"},
    )
    raised = RuntimeError("outer " + config.llm.api_key) if wrapped else http_error
    if wrapped:
        raised.__cause__ = http_error

    async def failing_chat(*args, **kwargs):
        raise raised

    async def catching_suite(*args, **kwargs):
        # Simulate RuntimeService retaining its generic llm_error outcome while
        # proving the observation wrapper did not replace/suppress the exception.
        with pytest.raises(type(raised)) as captured:
            await probe.AnthropicProvider.chat(object())
        assert captured.value is raised
        return SimpleNamespace(attempts=[])

    monkeypatch.setattr(probe.AnthropicProvider, "chat", failing_chat)
    monkeypatch.setattr(probe, "run_eval_suite", catching_suite)
    with probe.ProbeBudget(tmp_path / "budget.json", ledger, None) as budget:
        result = await probe.execute(output, config, budget)
    saved = (output / "probe-summary.json").read_text(encoding="utf-8")
    assert all(secret not in saved for secret in (
        config.llm.api_key, "secret-header-do-not-save", "sensitive-body-do-not-save",
    ))
    error = result["provider_errors"][0]
    assert error == {
        "type": "RuntimeError" if wrapped else "InternalServerError", "http_status": 500,
        "service_error_type": "api_error", "service_error_code": "[REDACTED]",
        "service_error_message": "request failed [REDACTED]", "retry_after": None,
        "cause_types": ["InternalServerError"] if wrapped else [],
    }
    assert result["completed_response_observations"] == []
    assert probe.read_count(ledger) == 0


def test_provider_diagnostics_do_not_serialize_nested_or_freeform_service_values():
    error = RuntimeError("not logged")
    error.body = {"error": {"type": {"headers": "private"}, "code": "a full response body"}}
    error.__cause__ = ConnectionError("private network details")
    result = probe.safe_provider_error(error, "key")
    assert result["http_status"] is None
    assert result["service_error_type"] is None and result["service_error_code"] is None
    assert result["cause_types"] == ["ConnectionError"]


@pytest.mark.parametrize("actual", [None, "proxy/deepseek-flash-alias", "synthetic-private-key", "unexpected freeform response"])
def test_mismatch_diagnostics_preserve_unknown_or_alias_and_redact_key(actual):
    error = RuntimeError("fixed mismatch message")
    error.expected_model = probe.DEEPSEEK_MODEL
    error.actual_model = actual
    result = probe.safe_provider_error(error, "synthetic-private-key")
    assert result["expected_model"] == probe.DEEPSEEK_MODEL
    expected = "[REDACTED]" if actual == "synthetic-private-key" else None if actual == "unexpected freeform response" else actual
    assert result["actual_model"] == expected
    assert "synthetic-private-key" not in json.dumps(result)


def test_model_identity_fields_are_absent_without_mismatch_attributes():
    result = probe.safe_provider_error(RuntimeError("unrelated"), "key")
    assert "expected_model" not in result and "actual_model" not in result


def test_service_message_and_retry_after_redact_before_truncation():
    from types import SimpleNamespace

    known = "known-secret-token"
    unknown_sk = "sk-synthetic-other-secret"
    unknown_dotted = "a1" * 16 + ".synthetic_other_secret"
    message = f"RPM exceeded {known} {unknown_sk} {unknown_dotted}\n\x00\u200b " + "x" * 600
    error = RuntimeError("do not serialize this exception or headers")
    error.body = {"error": {"message": message}, "private": "whole-body-secret"}
    error.response = SimpleNamespace(headers={
        "retry-after": f"60\r\n\x00 {known} {unknown_sk} {unknown_dotted}" + "y" * 600,
        "authorization": "whole-header-secret",
    })
    result = probe.safe_provider_error(error, known)
    assert len(result["service_error_message"]) == 500
    assert len(result["retry_after"]) <= 500
    assert result["service_error_message"].startswith("RPM exceeded [REDACTED] [REDACTED] [REDACTED]")
    serialized = json.dumps(result, ensure_ascii=False)
    for value in (known, unknown_sk, unknown_dotted, "whole-body-secret", "whole-header-secret"):
        assert value not in serialized
    assert all(ord(char) >= 32 and char != "\u200b" for char in result["service_error_message"])
    assert all(ord(char) >= 32 for char in result["retry_after"])


@pytest.mark.parametrize("body,expected", [
    ({"error": {"message": "tokens per minute exhausted"}}, "tokens per minute exhausted"),
    ({"message": "daily quota exhausted"}, "daily quota exhausted"),
    ({"error": {"message": None}, "message": "top-level fallback"}, "top-level fallback"),
    ({"error": {"message": None}}, None),
    ({"error": {"message": {"nested": "must-not-be-serialized"}}}, None),
    ({}, None),
    (None, None),
])
def test_service_message_fallback_and_missing_values(body, expected):
    error = RuntimeError("do not use generic exception as service message")
    error.body = body
    result = probe.safe_provider_error(error, "known-key")
    assert result["service_error_message"] == expected
    assert result["retry_after"] is None


@pytest.mark.parametrize("value", [None, "120", "Wed, 21 Oct 2015 07:28:00 GMT"])
def test_retry_after_retains_only_the_selected_safe_header(value):
    from types import SimpleNamespace

    error = RuntimeError("synthetic")
    error.response = SimpleNamespace(headers={"retry-after": value, "x-secret": "not-copied"})
    result = probe.safe_provider_error(error, "known-key")
    assert result["retry_after"] == value
    assert "not-copied" not in json.dumps(result)
