from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_direct_cli_launcher_imports_harness_before_rejecting_existing_output(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[2]
    existing = tmp_path / "existing"
    existing.mkdir()
    result = subprocess.run(
        [sys.executable, str(project / "scripts/run_internship_evals.py"), "cli",
         "--output", str(existing), "--execute"],
        cwd=tmp_path, env={**os.environ, "TARS_HOME": str(tmp_path / "not-a-profile"),
                          "PYTHONUTF8": "1"},
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode != 0
    assert "CLI evidence directory must be new" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert list(existing.iterdir()) == []


def test_cli_launcher_dry_run_does_not_create_output(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[2]
    output = tmp_path / "not-created"
    result = subprocess.run(
        [sys.executable, str(project / "scripts/run_internship_evals.py"), "cli",
         "--output", str(output)], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode == 0
    assert '"rounds": 3' in result.stdout
    assert not output.exists()


@pytest.mark.parametrize("dataset", ["train", "dev"])
def test_deepseek_paired_plan_only_without_credentials(tmp_path, monkeypatch, capsys, dataset):
    import json

    from scripts import run_internship_evals as launcher

    def forbidden():
        raise AssertionError("plan-only must not read credentials")

    monkeypatch.setattr(launcher, "get_config", forbidden)
    monkeypatch.setattr(launcher, "load_deepseek_config", forbidden)
    output = tmp_path / "not-created"
    monkeypatch.setattr(sys, "argv", ["launcher", f"deepseek-{dataset}-paired", "--output", str(output)])
    assert launcher.main() == 0
    suite = json.loads(capsys.readouterr().out)
    spec = suite["appworld"]
    assert spec["task_ids"] == (list(launcher.TRAIN_TASK_IDS) if dataset == "train" else None)
    assert spec["paired_comparison"] is True
    assert (spec["dataset"], spec["workers"], spec["max_steps"], spec["task_timeout_s"], spec["infrastructure_retries"]) == (dataset, 1, 60, 900, 1)
    assert not output.exists()


async def test_deepseek_launcher_preserves_old_evidence_before_loading_credentials(tmp_path, monkeypatch):
    from scripts import run_internship_evals as launcher

    output = tmp_path / "old-evidence"
    output.mkdir()
    marker = output / "original"
    marker.write_text("keep")
    monkeypatch.setattr(launcher, "load_deepseek_config", lambda: pytest.fail("must reject first"))
    with pytest.raises(ValueError, match="already exists"):
        await launcher.execute("deepseek-dev-paired", output, False)
    assert marker.read_text() == "keep" and list(output.iterdir()) == [marker]


async def test_deepseek_launcher_uses_only_dedicated_profile_and_original_ledger(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import run_internship_evals as launcher

    from tars_agent.core.eval.appworld import load_deepseek_config
    from tars_agent.core.persistence.cost_budget import CostLedger
    from tars_agent.core.persistence.request_budget import RequestLedger

    ledger = RequestLedger(tmp_path / "original.sqlite3")
    ledger.reserve()
    cost = CostLedger.initialize(tmp_path / "cost.sqlite3", request_budget_path=ledger.path)
    config = load_deepseek_config(private_env=tmp_path / "missing.env", environment={
        "DEEPSEEK_API_KEY": "unit-placeholder", "DEEPSEEK_REQUEST_BUDGET_PATH": str(ledger.path),
        "DEEPSEEK_COST_BUDGET_PATH": str(cost.path),
    })

    async def run(path, output, selected, *, repository_root):
        assert selected is config and selected.agent.max_steps == 60
        assert selected.llm.request_budget_path == ledger.path
        assert selected.llm.default_model == selected.llm.expected_model == "deepseek-flash"
        return SimpleNamespace(summary=SimpleNamespace(model_dump=lambda **_: {}),
                               benchmark={"complete": True}, adapter="appworld")

    monkeypatch.setattr(launcher, "get_config", lambda: pytest.fail("ordinary credentials forbidden"))
    monkeypatch.setattr(launcher, "load_deepseek_config", lambda: config)
    monkeypatch.setattr(launcher, "run_eval_suite", run)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *args, **kwargs: "")
    root = tmp_path / "batch"
    assert await launcher.execute("deepseek-train-paired", root / "train", False, root) == 0
    assert config.llm.request_limit == 33121
    assert ledger.counts()["real"] == 1
