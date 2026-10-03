from __future__ import annotations

import os
from pathlib import Path

import pytest

from tars_agent.core.config import LlmConfig, get_config
from tars_agent.core.paths import tars_home
from tars_agent.core.persistence.request_budget import RequestLedger


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith(("TARS_", "ANTHROPIC_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("TARS_HOME", str(tmp_path / "trusted-home"))
    monkeypatch.chdir(tmp_path)


def _write_env(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# 功能：验证 .env 文件中的值被正确加载并覆盖内建默认值
# 设计：写 .env 到临时目录并 chdir 进去，清除同名系统环境变量排除干扰，确认 .env 加载路径有效
def test_dotenv_base_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "TARS_PORT=9999\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TARS_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 9999


# 功能：验证系统环境变量的优先级高于 .env 文件中的值
# 设计：.env 写 9999，系统环境变量写 8888，确认最终值为 8888，对应四级优先链的顶层约束
def test_system_env_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "TARS_PORT=9999\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARS_PORT", "8888")

    cfg = get_config()

    assert cfg.port == 8888


# 功能：验证 .env 文件不存在时静默跳过，使用内建默认值（不抛异常）
# 设计：chdir 到空目录，清除系统环境变量，确认 get_config() 不因 .env 缺失而崩溃，默认端口为 7437
def test_missing_env_file_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TARS_PORT", raising=False)
    monkeypatch.delenv("TARS_COMPACT_THRESHOLD", raising=False)
    monkeypatch.delenv("TARS_TRACE_INCLUDE_LLM_PAYLOAD", raising=False)

    cfg = get_config()

    assert cfg.port == 7437
    assert cfg.compaction.auto_threshold == 0.80
    assert cfg.trace.include_llm_payload is False


# 功能：验证 .env 中设置的 TARS_CONFIG 能正确影响 TOML 配置文件的加载路径
# 设计：.env 指向自定义 TOML 文件，TOML 中写入不同端口，确认 .env 在 TOML 加载前被读取（优先级链的正确顺序）
def test_dotenv_before_toml_kama_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml_path = tmp_path / "custom.toml"
    toml_path.write_bytes(b'[core]\nport = 5555\n')

    env_file = tmp_path / ".env"
    _write_env(env_file, f"TARS_CONFIG={toml_path}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TARS_CONFIG", raising=False)
    monkeypatch.delenv("TARS_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 5555


# 功能：验证同一变量经过完整四级优先链后，最终值为最高优先级来源（系统环境变量）
# 设计：同时设置默认值(7437)/TOML(6000)/.env(7000)/系统环境变量(8000)，确认最终值为 8000，是优先级链的综合正确性验证
def test_priority_chain_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 默认值：7437
    # TOML：6000
    # .env：7000
    # 系统环境变量：8000（最高）
    toml_path = tmp_path / "kama.toml"
    toml_path.write_bytes(b'[core]\nport = 6000\n')

    env_file = tmp_path / ".env"
    _write_env(env_file, "TARS_PORT=7000\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARS_CONFIG", str(toml_path))
    monkeypatch.setenv("TARS_PORT", "8000")

    cfg = get_config()

    assert cfg.port == 8000


# 功能：验证冒烟与普通运行可通过环境变量限制 LLM 最大输出 token
# 设计：在空目录设置 TARS_LLM_MAX_TOKENS=64，断言配置进入 Provider 可消费字段
def test_llm_max_tokens_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARS_LLM_MAX_TOKENS", "64")

    config = get_config()

    assert config.llm.max_tokens == 64


# 功能：验证非正整数 max token 配置会在启动前显式失败
# 设计：分别注入零与非数字字符串，防止无界或不可解析值进入 Anthropic 请求
@pytest.mark.parametrize("value", ["0", "not-an-int"])
def test_invalid_llm_max_tokens_rejected(
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARS_LLM_MAX_TOKENS", value)

    with pytest.raises(SystemExit, match="TARS_LLM_MAX_TOKENS"):
        get_config()


@pytest.mark.parametrize("text", [
    '[llm]\napi_key="private"\nbase_url="https://example.invalid"',
    '[trace]\nfile="capture.jsonl"', '[trace]\ninclude_llm_payload=true',
    '[permission]\ntimeout_s=0', '[sandbox]\nmode="preferred"',
    '[mcp]\nservers=[]', '[llm]\ntotal_timeout_s=999',
    '[llm]\nexpected_model=""',
])
def test_project_toml_cannot_change_sensitive_settings(tmp_path: Path, text: str) -> None:
    path = tmp_path / ".tars" / "config.toml"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit, match="project/explicit"):
        get_config()


def test_project_dotenv_does_not_mutate_environment_or_home(tmp_path: Path) -> None:
    before = dict(os.environ)
    _write_env(tmp_path / ".env", "\n".join([
        "PATH=project", "COMSPEC=project", "HTTPS_PROXY=https://project.invalid",
        "ANTHROPIC_BASE_URL=https://project.invalid", "TARS_HOME=project-home",
        "TARS_LLM_BASE_URL=https://project.invalid", "TARS_LLM_API_KEY=private",
        "TARS_SANDBOX_MODE=preferred", "TARS_TRACE_INCLUDE_LLM_PAYLOAD=true",
        "TARS_LLM_TOTAL_TIMEOUT_S=999", "TARS_LOG_FILE=project.log",
    ]))
    config = get_config()
    assert dict(os.environ) == before
    assert tars_home() == (tmp_path / "trusted-home").resolve()
    assert config.llm.base_url == config.llm.api_key == ""
    assert config.llm.total_timeout_s == 120
    assert config.sandbox.mode == "required"
    assert config.trace.include_llm_payload is False
    assert config.logging.file == str(tars_home() / "logs" / "core.log")


def test_trusted_config_and_process_environment_can_set_sensitive_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tars_home()
    home.mkdir()
    (home / "config.toml").write_text(
        '[llm]\napi_key="dedicated"\nbase_url="https://relay.invalid"\n'
        'total_timeout_s=30\n[sandbox]\nmode="preferred"\n', encoding="utf-8",
    )
    monkeypatch.setenv("TARS_LLM_TOTAL_TIMEOUT_S", "15")
    config = get_config()
    assert config.llm.api_key == "dedicated"
    assert config.llm.base_url == "https://relay.invalid"
    assert config.llm.total_timeout_s == 15
    assert config.sandbox.mode == "preferred"
    assert "dedicated" not in repr(config)


def test_request_budget_path_is_fixed_before_workspace_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = get_config()
    direct = LlmConfig()
    original_path = tars_home() / "acceptance" / "request-budget.sqlite3"
    monkeypatch.setenv("TARS_HOME", str(tmp_path / "different-home"))
    assert config.llm.request_budget_path == direct.request_budget_path == original_path


@pytest.mark.parametrize("endpoint", ["http://remote.invalid", "https://user:pass@relay.invalid", "https://relay.invalid?q=1"])
def test_invalid_custom_endpoint_rejected(endpoint: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TARS_LLM_BASE_URL", endpoint)
    monkeypatch.setenv("TARS_LLM_API_KEY", "dedicated")
    with pytest.raises(SystemExit, match="base_url"):
        get_config()


def test_custom_endpoint_cannot_reuse_official_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TARS_LLM_BASE_URL", "https://relay.invalid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "official")
    with pytest.raises(SystemExit, match="dedicated"):
        get_config()


@pytest.mark.parametrize("value", ["NaN", "inf", "0", "-1"])
def test_timeout_requires_finite_positive_value(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TARS_LLM_TOTAL_TIMEOUT_S", value)
    with pytest.raises(SystemExit, match="positive"):
        get_config()


def test_project_cannot_self_trust_mcp(tmp_path: Path) -> None:
    config = tmp_path / ".tars" / "config.toml"
    config.parent.mkdir()
    config.write_text('[[mcp.servers]]\nname="x"\ncommand="evil"\ntrusted=true', encoding="utf-8")
    with pytest.raises(SystemExit, match="project/explicit"):
        get_config()


@pytest.mark.parametrize(("value", "expected"), [("unlimited", None), ("250", 250), ("1", 1)])
def test_trusted_process_request_limit(value: str, expected: int | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TARS_LLM_REQUEST_LIMIT", value)
    assert get_config().llm.request_limit == expected


@pytest.mark.parametrize(("value", "expected"), [('"unlimited"', None), ("250", 250)])
def test_trusted_home_request_limit(value: str, expected: int | None) -> None:
    home = tars_home()
    home.mkdir()
    (home / "config.toml").write_text(f"[llm]\nrequest_limit={value}\n", encoding="utf-8")
    assert get_config().llm.request_limit == expected


def test_request_limit_defaults_to_100() -> None:
    assert LlmConfig().request_limit == 100
    assert get_config().llm.request_limit == 100


@pytest.mark.parametrize("value", ['"unlimited"', "500"])
def test_project_toml_cannot_increase_or_disable_request_limit(tmp_path: Path, value: str) -> None:
    path = tmp_path / ".tars" / "config.toml"
    path.parent.mkdir()
    path.write_text(f"[llm]\nrequest_limit={value}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="project/explicit"):
        get_config()


@pytest.mark.parametrize("value", ["unlimited", "500"])
def test_project_dotenv_cannot_increase_or_disable_request_limit(
    tmp_path: Path, value: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(f"TARS_LLM_REQUEST_LIMIT={value}\n", encoding="utf-8")
    assert get_config().llm.request_limit == 100
    monkeypatch.setenv("TARS_LLM_REQUEST_LIMIT", "20")
    assert get_config().llm.request_limit == 20


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "false", "none", ""])
def test_invalid_process_request_limit_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TARS_LLM_REQUEST_LIMIT", value)
    with pytest.raises(SystemExit, match="positive integer or unlimited"):
        get_config()


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true"])
def test_invalid_trusted_toml_request_limit_is_rejected(value: str) -> None:
    home = tars_home()
    home.mkdir()
    (home / "config.toml").write_text(f"[llm]\nrequest_limit={value}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="positive integer or unlimited"):
        get_config()


@pytest.mark.parametrize("text", [
    '[llm]\nrouter="static"\n',
    '[sandbox]\nenabled=true\n',
    '[sandbox]\nenabled=false\n',
])
def test_removed_config_options_are_rejected(text: str) -> None:
    home = tars_home()
    home.mkdir()
    (home / "config.toml").write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit, match="Unknown"):
        get_config()


def test_expected_model_defaults_to_disabled() -> None:
    assert get_config().llm.expected_model == LlmConfig().expected_model == ""


def test_expected_model_only_trusted_config_or_process_can_set_or_disable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tars_home()
    home.mkdir()
    (home / "config.toml").write_text(
        '[llm]\ndefault_model="deepseek-flash"\nexpected_model="deepseek-flash"\n',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text('TARS_LLM_EXPECTED_MODEL=\n', encoding="utf-8")
    assert get_config().llm.expected_model == "deepseek-flash"
    monkeypatch.setenv("TARS_LLM_EXPECTED_MODEL", "")
    assert get_config().llm.expected_model == ""
    monkeypatch.setenv("TARS_LLM_EXPECTED_MODEL", "another")
    monkeypatch.setenv("TARS_LLM_DEFAULT_MODEL", "another")
    assert get_config().llm.expected_model == "another"


@pytest.mark.parametrize("value", [" ", "deepseek-flash ", "deepseek\nflash"])
def test_expected_model_rejects_whitespace(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TARS_LLM_EXPECTED_MODEL", value)
    with pytest.raises(SystemExit, match="expected_model"):
        get_config()


def test_expected_model_rejects_different_requested_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TARS_LLM_EXPECTED_MODEL", "deepseek-flash")
    monkeypatch.setenv("TARS_LLM_DEFAULT_MODEL", "paid-model")
    with pytest.raises(SystemExit, match="must match"):
        get_config()


def test_expected_model_requires_string_in_trusted_toml() -> None:
    home = tars_home()
    home.mkdir()
    (home / "config.toml").write_text('[llm]\nexpected_model=123\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="expected_model must be a string"):
        get_config()


def test_trusted_process_can_reuse_existing_ledger_across_isolated_homes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path = tmp_path / "original" / "request-budget.sqlite3"
    RequestLedger(ledger_path).reserve()
    before = ledger_path.read_bytes()
    monkeypatch.setenv("TARS_LLM_REQUEST_BUDGET_PATH", str(ledger_path))
    config = get_config()
    assert config.llm.request_budget_path == ledger_path.resolve()
    assert config.llm.request_limit == 100
    monkeypatch.setenv("TARS_HOME", str(tmp_path / "separate-demo-home"))
    assert get_config().llm.request_budget_path == config.llm.request_budget_path
    assert ledger_path.read_bytes() == before
    assert not (tmp_path / "separate-demo-home").exists()


def test_project_dotenv_cannot_override_request_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected_path = tmp_path / "must-not-create.sqlite3"
    (tmp_path / ".env").write_text(
        f"TARS_LLM_REQUEST_BUDGET_PATH={injected_path}\n", encoding="utf-8",
    )
    default = tars_home() / "acceptance" / "request-budget.sqlite3"
    assert get_config().llm.request_budget_path == default
    assert not default.exists()
    assert not injected_path.exists()
    trusted = tmp_path / "trusted.sqlite3"
    RequestLedger(trusted).reserve()
    monkeypatch.setenv("TARS_LLM_REQUEST_BUDGET_PATH", str(trusted))
    assert get_config().llm.request_budget_path == trusted.resolve()
    assert not injected_path.exists()


@pytest.mark.parametrize("kind", ["missing", "relative", "fake", "empty"])
def test_invalid_process_request_ledger_fails_before_creating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    path = tmp_path / "private-ledger.sqlite3"
    if kind == "relative":
        RequestLedger(path).reserve()
        value = path.name
    elif kind == "fake":
        path.write_bytes(b"not a ledger")
        value = str(path)
    elif kind == "empty":
        path.touch()
        value = str(path)
    else:
        value = str(tmp_path / "absent" / path.name)
    before = {str(item): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()}
    monkeypatch.setenv("TARS_LLM_REQUEST_BUDGET_PATH", value)
    with pytest.raises(SystemExit, match="TARS_LLM_REQUEST_BUDGET_PATH") as captured:
        get_config()
    assert "private-ledger" not in str(captured.value)
    assert {str(item): item.read_bytes() for item in tmp_path.rglob("*") if item.is_file()} == before
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("trusted", [False, True])
def test_toml_cannot_select_request_budget_path(tmp_path: Path, trusted: bool) -> None:
    config_path = tars_home() / "config.toml" if trusted else tmp_path / ".tars" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('[llm]\nrequest_budget_path="another.sqlite3"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="Unknown.*request_budget_path"):
        get_config()
    assert not (config_path.parent / "another.sqlite3").exists()
