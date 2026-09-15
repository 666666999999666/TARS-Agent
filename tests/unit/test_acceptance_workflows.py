from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.acceptance_workflows import (
    AcceptanceLedger,
    ApprovalPolicy,
    EnvironmentBlocked,
    acceptance_ledger_type,
    counts,
    effective_request_limit,
    fixed_configuration,
)

from tars_agent.core.persistence.request_budget import REQUEST_LIMIT, ModelRequestBudgetExceeded


def ledger_fixture(path: Path, real: int) -> None:
    with sqlite3.connect(path) as sql:
        sql.execute("CREATE TABLE requests (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, reserved_at TEXT NOT NULL)")
        sql.executemany("INSERT INTO requests(kind,reserved_at) VALUES ('real','fixture')", [()] * real)


def test_acceptance_gate_uses_existing_rows_and_blocks_request_76(tmp_path: Path) -> None:
    path = tmp_path / "budget.sqlite3"
    ledger_fixture(path, 74)
    ledger = AcceptanceLedger(path)
    assert ledger.reserve() == 75
    with pytest.raises(ModelRequestBudgetExceeded):
        ledger.reserve()
    assert counts(path) == {"real": 75, "probe": 0}
    assert ledger.reserve("probe") == 1
    assert counts(path) == {"real": 75, "probe": 1}
    assert REQUEST_LIMIT == 100


def test_acceptance_gate_never_recreates_missing_ledger(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        AcceptanceLedger(path).reserve()
    assert not path.exists()


def test_acceptance_rejects_changed_home_before_reading_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TARS_HOME", str(tmp_path / "replacement-home"))
    with pytest.raises(EnvironmentBlocked, match="不能为验收换HOME"):
        fixed_configuration()


def test_approval_is_owned_parameter_scoped_and_never_allows_host(tmp_path: Path) -> None:
    policy = ApprovalPolicy("owned", tmp_path, writable={"allowed.txt"}, denied={"denied.txt"},
                            commands={"python safe.py"})
    event = {"session_id": "owned", "request_kind": "tool", "tool_name": "write_file", "params": {"path": "allowed.txt"}}
    assert policy.decide(event) == "allow_once"
    assert policy.decide({**event, "session_id": "foreign"}) is None
    assert policy.decide({**event, "request_kind": "host_fallback"}) == "deny_once"
    assert policy.decide({**event, "params": {"path": "denied.txt"}}) == "deny_once"
    assert policy.decide({**event, "params": {"path": "../outside.txt"}}) == "deny_once"
    assert policy.decide({**event, "tool_name": "spawn_agent"}) == "deny_once"
    bash = {**event, "tool_name": "bash", "params": {"command": "python safe.py", "timeout": 30}}
    assert policy.decide(bash) == "allow_once"
    assert policy.decide({**bash, "params": {"command": "python unsafe.py"}}) == "deny_once"
    assert policy.decide({**bash, "params": {"command": "python safe.py", "timeout": 3600}}) == "deny_once"


def test_plan_mode_creates_no_runtime_home_or_request(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[2]
    home = tmp_path / "must-not-exist"
    state_home = tmp_path / "must-not-create-state"
    output_root = tmp_path / "must-not-create-output"
    env = os.environ.copy()
    env["TARS_HOME"] = str(home)
    result = subprocess.run([sys.executable, str(project / "scripts/acceptance_workflows.py"),
                             "--state-home", str(state_home), "--output-root", str(output_root)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mode"] == "plan_only_no_requests"
    assert not home.exists()
    assert not state_home.exists()
    assert not output_root.exists()


def test_daemon_uses_isolated_state_but_preserves_original_request_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import acceptance_workflows as workflows

    import tars_agent.core.app as app_module
    import tars_agent.core.llm.provider as provider_module
    from tars_agent.core.config import TarsConfig
    from tars_agent.core.paths import tars_home

    user_home = tmp_path / "user"
    original_home = user_home / ".tars-baseline"
    original_home.mkdir(parents=True)
    original_state = original_home / "state.db"
    original_state.write_bytes(b"existing user session database must remain untouched")
    ledger_path = original_home / "acceptance" / "request-budget.sqlite3"
    ledger_path.parent.mkdir()
    ledger_fixture(ledger_path, 7)
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setenv("TARS_HOME", str(original_home))
    config = TarsConfig()
    config.llm.request_budget_path = ledger_path
    config.llm.default_model = "unchanged-model"
    monkeypatch.setattr(workflows, "fixed_configuration", lambda: (config, original_home, ledger_path))
    # Record these bindings so the in-process child test restores its temporary overrides.
    monkeypatch.setattr(app_module, "get_config", app_module.get_config)
    monkeypatch.setattr(provider_module, "RequestLedger", provider_module.RequestLedger)
    state_home = tmp_path / "isolated-state"
    controller = workflows.Workflows(False, state_home=state_home, output_root=tmp_path / "evidence")

    class StopBeforeDocker(RuntimeError):
        pass

    async def stop_before_docker(_sandbox: object) -> None:
        snapshot = app_module.get_config()
        assert tars_home() == state_home
        assert snapshot.llm.request_budget_path == ledger_path
        assert snapshot.llm.default_model == "unchanged-model"
        assert snapshot.logging.file == str(state_home / "logs" / "core.log")
        assert snapshot.trace.file == str(state_home / "traces" / "daemon.jsonl")
        raise StopBeforeDocker()

    # Run the real Core bootstrap and SQLite migration, then stop before Docker or a model call.
    monkeypatch.setattr(app_module, "initialize_runtime_router", stop_before_docker)
    with pytest.raises(StopBeforeDocker):
        workflows.daemon_child(state_home=controller.home)

    assert controller.home == state_home
    assert controller.ledger == ledger_path
    assert (state_home / "state.db").is_file()
    with sqlite3.connect(state_home / "state.db") as sql:
        assert sql.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
    assert original_state.read_bytes() == b"existing user session database must remain untouched"
    assert counts(ledger_path) == {"real": 7, "probe": 0}
    assert not (state_home / "acceptance" / "request-budget.sqlite3").exists()


def test_isolated_state_must_not_overlap_original_home(tmp_path: Path) -> None:
    from scripts.acceptance_workflows import isolated_state_home

    original_home = tmp_path / "user-home"
    for candidate in (original_home, original_home / "test-state", tmp_path):
        with pytest.raises(EnvironmentBlocked, match="独立"):
            isolated_state_home(candidate, original_home)


def _completed_workflow_fixture(tmp_path, monkeypatch, run_status, tools):
    from scripts import acceptance_workflows as workflows

    controller = workflows.Workflows(False, output_root=tmp_path / "evidence")

    async def rpc(method, params, timeout=20):
        assert params == {"run_id": "owned-run"}
        if method == "run.get":
            return {"run_id": "owned-run", "status": run_status, "result": {"text": "fixture"}}
        assert method == "run.metrics"
        return {}

    monkeypatch.setattr(controller, "rpc", rpc)
    monkeypatch.setattr(controller, "snapshot", lambda _session, _run: {
        "database_run": {"id": "owned-run", "status": run_status}, "tools": tools,
        "events": [{"event_type": "run.finished", "payload": {"run_id": "owned-run"}}],
    })
    monkeypatch.setattr(workflows, "counts", lambda _path: {"real": 0, "probe": 0})
    return controller


@pytest.mark.parametrize("run_status", ["succeeded", "failed", "cancelled", "interrupted"])
@pytest.mark.parametrize("tool_status", ["queued", "running"])
async def test_terminal_run_rejects_unfinished_owned_tool_records(
    tmp_path, monkeypatch, run_status, tool_status,
) -> None:
    from scripts.acceptance_workflows import WorkflowFailure

    controller = _completed_workflow_fixture(tmp_path, monkeypatch, run_status, [{
        "id": "owned-tool", "run_id": "owned-run", "status": tool_status,
    }])
    with pytest.raises(WorkflowFailure, match="queued/running"):
        await controller.completed("owned-session", "owned-run", expected=run_status)
    saved = json.loads((controller.directory / "runs" / "owned-run.json").read_text(encoding="utf-8"))
    assert saved["terminal_tool_check"] == {
        "run_id": "owned-run", "run_status": run_status,
        "passed": False, "unfinished_tool_ids": ["owned-tool"],
    }


async def test_terminal_tool_check_ignores_another_runs_active_record(tmp_path, monkeypatch) -> None:
    controller = _completed_workflow_fixture(tmp_path, monkeypatch, "cancelled", [{
        "id": "foreign-tool", "run_id": "foreign-run", "status": "running",
        "tool_name": "mcp_fixture",
    }])
    evidence = await controller.completed("owned-session", "owned-run", expected="cancelled")
    assert evidence["terminal_tool_check"]["passed"] is True


def test_unlimited_acceptance_keeps_counting_above_both_historical_limits(tmp_path: Path) -> None:
    path = tmp_path / "existing.sqlite3"
    ledger_fixture(path, 74)
    ledger_type = acceptance_ledger_type(None)
    ledger = ledger_type(path, limit=None)
    assert ledger.limit is None
    for expected in range(75, 103):
        assert ledger.reserve() == expected
    assert counts(path) == {"real": 102, "probe": 0}
    with sqlite3.connect(path) as sql:
        assert sql.execute("SELECT count(*) FROM requests WHERE reserved_at='fixture'").fetchone()[0] == 74


@pytest.mark.parametrize(("script_limit", "production_limit", "expected"), [
    (None, 80, 80), (90, None, 90), (90, 85, 85), (75, 100, 75),
])
def test_acceptance_uses_stricter_configured_limit(
    tmp_path: Path, script_limit: int | None, production_limit: int | None, expected: int,
) -> None:
    path = tmp_path / "existing.sqlite3"
    ledger_fixture(path, expected - 1)
    ledger = acceptance_ledger_type(script_limit)(path, limit=production_limit)
    assert effective_request_limit(script_limit, production_limit) == expected
    assert ledger.reserve() == expected
    with pytest.raises(ModelRequestBudgetExceeded):
        ledger.reserve()
    assert counts(path)["real"] == expected


@pytest.mark.parametrize(("argument", "expected"), [("unlimited", None), ("UNLIMITED", None), ("120", 120)])
def test_real_cli_parses_request_limit_without_running_workflows(
    tmp_path: Path, argument: str, expected: int | None,
) -> None:
    project = Path(__file__).resolve().parents[2]
    home = tmp_path / "must-not-exist"
    env = {**os.environ, "TARS_HOME": str(home)}
    result = subprocess.run(
        [sys.executable, str(project / "scripts/acceptance_workflows.py"), "--request-limit", argument],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=10,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["script_request_limit"] == expected
    assert report["mode"] == "plan_only_no_requests"
    assert not home.exists()


@pytest.mark.parametrize("argument", ["0", "-1", "none", "1.5"])
def test_real_cli_rejects_invalid_request_limit_without_model_calls(tmp_path: Path, argument: str) -> None:
    project = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(project / "scripts/acceptance_workflows.py"), "--request-limit", argument],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=10,
    )
    assert result.returncode == 2
    assert "request-limit" in result.stderr
