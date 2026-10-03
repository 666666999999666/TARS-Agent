from __future__ import annotations

import json
from pathlib import Path

import pytest

from tars_agent.core.eval import appworld
from tars_agent.core.persistence.request_budget import RequestLedger


def environment_for(tmp_path: Path) -> dict[str, str]:
    ledger = RequestLedger(tmp_path / "original.sqlite3", limit=None)
    ledger.reserve()
    return {"DEEPSEEK_API_KEY": "unit-deepseek-placeholder",
            "DEEPSEEK_REQUEST_BUDGET_PATH": str(ledger.path)}


def test_deepseek_loader_pins_official_model_and_preserves_original_ledger(tmp_path):
    environment = environment_for(tmp_path)
    environment.update(TARS_LLM_API_KEY="unrelated-placeholder", ANTHROPIC_API_KEY="unrelated-placeholder",
                       TARS_LLM_BASE_URL="https://unrelated.invalid")
    ledger = Path(environment["DEEPSEEK_REQUEST_BUDGET_PATH"])
    before = ledger.read_bytes()
    config = appworld.load_deepseek_config(private_env=tmp_path / "missing.env", environment=environment)
    assert config.llm.default_model == config.llm.expected_model == "deepseek-flash"
    assert config.llm.base_url == "https://api.deepseek.com/anthropic"
    assert config.llm.api_key == "unit-deepseek-placeholder" and config.llm.anthropic_api_key == ""
    assert (config.llm.attempts, config.llm.retry_delay_s) == (2, 1)
    assert config.llm.context_budget_tokens == 131072 and config.compaction.auto_threshold == 0
    assert config.llm.request_budget_path == ledger and config.llm.request_limit is None
    assert ledger.read_bytes() == before
    binding = appworld.model_protocol_binding(config)
    assert binding["provider_profile"] == "deepseek-flash-official-anthropic-v1"
    assert binding["model_fallback"] == "not_requested"
    assert "placeholder" not in json.dumps(binding)


def test_deepseek_private_file_and_trusted_environment_use_same_names(tmp_path):
    environment = environment_for(tmp_path)
    private = tmp_path / "dedicated.env"
    private.write_text("DEEPSEEK_API_KEY=private-placeholder\n"
                       f"DEEPSEEK_REQUEST_BUDGET_PATH={environment['DEEPSEEK_REQUEST_BUDGET_PATH']}\n"
                       "DEEPSEEK_REQUEST_LIMIT=50\n", encoding="utf-8")
    config = appworld.load_deepseek_config(private_env=private, environment={})
    assert config.llm.api_key == "private-placeholder" and config.llm.request_limit == 50
    overridden = appworld.load_deepseek_config(private_env=private, environment=environment)
    assert overridden.llm.api_key == environment["DEEPSEEK_API_KEY"]
    with pytest.raises(ValueError, match="dedicated"):
        appworld.load_deepseek_config(private_env=private, environment={"DEEPSEEK_API_KEY": ""})


def test_deepseek_does_not_fall_back_to_ordinary_key(tmp_path):
    environment = environment_for(tmp_path)
    environment.pop("DEEPSEEK_API_KEY")
    environment.update(TARS_LLM_API_KEY="unrelated-placeholder", ANTHROPIC_API_KEY="unrelated-placeholder")
    with pytest.raises(ValueError, match="dedicated"):
        appworld.load_deepseek_config(private_env=tmp_path / "missing.env", environment=environment)


@pytest.mark.parametrize("limit", ["", "0", "-1", "unlimited", "1.5"])
def test_deepseek_rejects_invalid_explicit_cumulative_cap(tmp_path, limit):
    environment = environment_for(tmp_path)
    environment["DEEPSEEK_REQUEST_LIMIT"] = limit
    with pytest.raises(ValueError, match="positive cumulative"):
        appworld.load_deepseek_config(private_env=tmp_path / "missing.env", environment=environment)


@pytest.mark.parametrize("change", ["model", "ledger", "cap", "guard", "endpoint", "retry"])
def test_deepseek_worker_cannot_replace_its_pinned_config(tmp_path, change):
    environment = environment_for(tmp_path)
    environment["DEEPSEEK_REQUEST_LIMIT"] = "50"
    config = appworld.load_deepseek_config(private_env=tmp_path / "missing.env", environment=environment)
    job = {"profile": appworld.DEEPSEEK_PROFILE, "model": appworld.DEEPSEEK_MODEL,
           "request_limit": 50, "request_budget_path": str(config.llm.request_budget_path), "max_steps": 20}
    if change == "model":
        job["model"] = "another-model"
    elif change == "ledger":
        job["request_budget_path"] = str(tmp_path / "replacement.sqlite3")
    elif change == "cap":
        job["request_limit"] = 51
    elif change == "guard":
        config.llm.expected_model = ""
    elif change == "endpoint":
        config.llm.base_url = "https://unrelated.invalid"
    else:
        config.llm.retry_delay_s = 30
    with pytest.raises(ValueError):
        appworld.worker_config(config, job)
    assert not (tmp_path / "replacement.sqlite3").exists()
