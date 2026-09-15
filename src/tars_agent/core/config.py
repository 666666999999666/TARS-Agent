from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from tars_agent.core.paths import tars_home

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7437
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FORMAT = "text"
_DEFAULT_MAX_STEPS = 20
_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_SANDBOX_IMAGE = "tars-agent-sandbox:0.8.0"


@dataclass
class LoggingConfig:
    level: str = _DEFAULT_LOG_LEVEL
    file: str = field(default_factory=lambda: str(tars_home() / "logs" / "core.log"))
    format: str = _DEFAULT_LOG_FORMAT  # "text" | "json"


@dataclass
class AgentConfig:
    max_steps: int = _DEFAULT_MAX_STEPS


@dataclass
class LlmConfig:
    default_model: str = _DEFAULT_MODEL
    max_tokens: int = 8192
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    anthropic_api_key: str = field(default="", repr=False)
    total_timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 60.0
    write_timeout_s: float = 10.0
    pool_timeout_s: float = 10.0
    attempts: int = 2
    request_limit: int | None = 100
    retry_delay_s: float = 1.0
    request_budget_path: Path = field(
        default_factory=lambda: tars_home() / "acceptance" / "request-budget.sqlite3",
        repr=False,
    )


@dataclass
class TraceConfig:
    enabled: bool = True
    file: str = field(default_factory=lambda: str(tars_home() / "traces" / "daemon.jsonl"))
    include_llm_payload: bool = False  # 默认只保留元数据，显式开启才记录模型载荷


@dataclass
class PermissionConfig:
    timeout_s: float = 60.0  # 审批超时秒数；0 表示不超时


@dataclass
class SandboxConfig:
    mode: str = "required"
    docker_binary: str = "docker"
    image: str = _DEFAULT_SANDBOX_IMAGE
    memory: str = "512m"
    memory_swap: str = "512m"
    cpus: float = 1.0
    pids_limit: int = 128
    nofile_limit: int = 1024
    output_limit_bytes: int = 64 * 1024


@dataclass
class CompactionConfig:
    # Trigger automatic compaction at 80% context usage; set to 0 to disable.
    auto_threshold: float = 0.80
    tool_result_limit: int = 8_000  # tool_result 截断触发字符数
    tool_result_keep: int = 4_000   # 截断后保留的前缀字符数


@dataclass
class McpServerConfig:
    name: str
    transport: str = "stdio"
    trusted: bool = False
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    connect_timeout_s: float = 10.0
    tool_timeout_s: float = 120.0


@dataclass
class McpConfig:
    servers: list[McpServerConfig] = field(default_factory=list)


@dataclass
class TarsConfig:
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    permission: PermissionConfig = field(default_factory=PermissionConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    mcp: McpConfig = field(default_factory=McpConfig)


def get_config() -> TarsConfig:
    """Load values without giving project files authority over process environment."""
    config = TarsConfig()
    home = tars_home()
    config.logging.file = str(home / "logs" / "core.log")
    config.trace.file = str(home / "traces" / "daemon.jsonl")
    # Capture once: eval workspaces and later cwd/HOME changes cannot reset the ledger.
    config.llm.request_budget_path = home / "acceptance" / "request-budget.sqlite3"
    dotenv_env = {
        key: value for key, value in dotenv_values(".env", interpolate=False).items()
        if isinstance(value, str)
    }
    explicit = os.environ.get("TARS_CONFIG") or dotenv_env.get("TARS_CONFIG")
    global_path = (home / "config.toml").resolve()
    config_paths = (
        [Path(explicit).expanduser()] if explicit
        else [global_path, Path(".tars/config.toml")]
    )
    for config_path in config_paths:
        if config_path.exists():
            try:
                with config_path.open("rb") as handle:
                    data = tomllib.load(handle)
            except tomllib.TOMLDecodeError as exc:
                raise SystemExit(f"Config parse error ({config_path}): {exc}") from exc
            _apply_toml(config, data, trusted=config_path.resolve() == global_path)
    _apply_env(config, dotenv_env, trusted=False)
    _apply_env(config, os.environ, trusted=True)
    _validate_config(config)
    return config


# 将已解析的 TOML 根表写入 config；未知小节或类型错误时退出进程
def _apply_toml(
    config: TarsConfig, data: dict[str, Any], *, trusted: bool = False
) -> None:
    sensitive = {
        "logging": {"file"},
        "trace": {"enabled", "file", "include_llm_payload"},
        "permission": {"timeout_s"},
        "llm": {"api_key", "base_url", "total_timeout_s", "connect_timeout_s",
                "read_timeout_s", "write_timeout_s", "pool_timeout_s",
                "attempts", "retry_delay_s", "request_limit"},
    }
    if not trusted:
        for section in ("sandbox", "mcp"):
            if section in data:
                raise SystemExit(
                    f"Config error: project/explicit config may not define [{section}]; "
                    "use TARS_HOME/config.toml or trusted process environment"
                )
        for section, keys in sensitive.items():
            value = data.get(section)
            if isinstance(value, dict) and keys.intersection(value):
                raise SystemExit(
                    f"Config error: project/explicit config may not define sensitive [{section}] "
                    "settings; use TARS_HOME/config.toml or trusted process environment"
                )

    known_sections = {
        "core",
        "logging",
        "agent",
        "llm",
        "trace",
        "permission",
        "sandbox",
        "compaction",
        "mcp",
    }
    unknown = set(data.keys()) - known_sections
    if unknown:
        raise SystemExit(f"Unknown top-level config keys: {', '.join(sorted(unknown))}")

    if "core" in data:
        core = data["core"]
        if not isinstance(core, dict):
            raise SystemExit("Config error: [core] must be a table")
        unknown_core: set[str] = set(core.keys()) - {"host", "port"}
        if unknown_core:
            raise SystemExit(f"Unknown [core] keys: {', '.join(sorted(unknown_core))}")
        if "host" in core:
            val = core["host"]
            if not isinstance(val, str):
                raise SystemExit("Config error: core.host must be a string")
            config.host = val
        if "port" in core:
            val = core["port"]
            if not isinstance(val, int):
                raise SystemExit("Config error: core.port must be an integer")
            config.port = val

    if "logging" in data:
        log = data["logging"]
        if not isinstance(log, dict):
            raise SystemExit("Config error: [logging] must be a table")
        unknown_log: set[str] = set(log.keys()) - {"level", "file", "format"}
        if unknown_log:
            raise SystemExit(f"Unknown [logging] keys: {', '.join(sorted(unknown_log))}")
        for key in ("level", "file", "format"):
            if key in log:
                val = log[key]
                if not isinstance(val, str):
                    raise SystemExit(f"Config error: logging.{key} must be a string")
                setattr(config.logging, key, val)

    if "agent" in data:
        agent = data["agent"]
        if not isinstance(agent, dict):
            raise SystemExit("Config error: [agent] must be a table")
        unknown_agent: set[str] = set(agent.keys()) - {"max_steps"}
        if unknown_agent:
            raise SystemExit(f"Unknown [agent] keys: {', '.join(sorted(unknown_agent))}")
        if "max_steps" in agent:
            val = agent["max_steps"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit("Config error: agent.max_steps must be a positive integer")
            config.agent.max_steps = val

    if "llm" in data:
        llm = data["llm"]
        if not isinstance(llm, dict):
            raise SystemExit("Config error: [llm] must be a table")
        unknown_llm: set[str] = set(llm.keys()) - {
            "default_model", "max_tokens", "api_key", "base_url",
            "total_timeout_s", "connect_timeout_s", "read_timeout_s", "write_timeout_s",
            "pool_timeout_s", "attempts", "retry_delay_s", "request_limit",
        }
        if unknown_llm:
            raise SystemExit(f"Unknown [llm] keys: {', '.join(sorted(unknown_llm))}")
        for key in ("api_key", "base_url"):
            if key in llm:
                if not isinstance(llm[key], str):
                    raise SystemExit(f"Config error: llm.{key} must be a string")
                setattr(config.llm, key, llm[key])
        for key in ("total_timeout_s", "connect_timeout_s", "read_timeout_s",
                    "write_timeout_s", "pool_timeout_s", "retry_delay_s"):
            if key in llm:
                setattr(config.llm, key, _positive_number(llm[key], f"llm.{key}"))
        if "request_limit" in llm:
            config.llm.request_limit = _parse_request_limit(
                llm["request_limit"], "llm.request_limit"
            )
        if "attempts" in llm:
            value = llm["attempts"]
            if type(value) is not int or value not in (1, 2):
                raise SystemExit("Config error: llm.attempts must be 1 or 2")
            config.llm.attempts = value
        if "default_model" in llm:
            val = llm["default_model"]
            if not isinstance(val, str):
                raise SystemExit("Config error: llm.default_model must be a string")
            config.llm.default_model = val
        if "max_tokens" in llm:
            val = llm["max_tokens"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit("Config error: llm.max_tokens must be a positive integer")
            config.llm.max_tokens = val

    if "trace" in data:
        trace = data["trace"]
        if not isinstance(trace, dict):
            raise SystemExit("Config error: [trace] must be a table")
        unknown_trace: set[str] = set(trace.keys()) - {"enabled", "file", "include_llm_payload"}
        if unknown_trace:
            raise SystemExit(f"Unknown [trace] keys: {', '.join(sorted(unknown_trace))}")
        if "enabled" in trace:
            val = trace["enabled"]
            if not isinstance(val, bool):
                raise SystemExit("Config error: trace.enabled must be a boolean")
            config.trace.enabled = val
        if "file" in trace:
            val = trace["file"]
            if not isinstance(val, str):
                raise SystemExit("Config error: trace.file must be a string")
            config.trace.file = val
        if "include_llm_payload" in trace:
            val = trace["include_llm_payload"]
            if not isinstance(val, bool):
                raise SystemExit("Config error: trace.include_llm_payload must be a boolean")
            config.trace.include_llm_payload = val

    if "permission" in data:
        perm = data["permission"]
        if not isinstance(perm, dict):
            raise SystemExit("Config error: [permission] must be a table")
        unknown_perm: set[str] = set(perm.keys()) - {"timeout_s"}
        if unknown_perm:
            raise SystemExit(f"Unknown [permission] keys: {', '.join(sorted(unknown_perm))}")
        if "timeout_s" in perm:
            val = perm["timeout_s"]
            if not isinstance(val, (int, float)) or val < 0:
                raise SystemExit("Config error: permission.timeout_s must be a non-negative number")
            config.permission.timeout_s = float(val)

    if "sandbox" in data:
        sandbox = data["sandbox"]
        if not isinstance(sandbox, dict):
            raise SystemExit("Config error: [sandbox] must be a table")
        known_sandbox_keys = {
            "mode",
            "docker_binary",
            "image",
            "memory",
            "memory_swap",
            "cpus",
            "pids_limit",
            "nofile_limit",
            "output_limit_bytes",
        }
        unknown_sandbox = set(sandbox.keys()) - known_sandbox_keys
        if unknown_sandbox:
            raise SystemExit(
                f"Unknown [sandbox] keys: {', '.join(sorted(unknown_sandbox))}"
            )
        if "mode" in sandbox:
            value = sandbox["mode"]
            if value not in ("required", "preferred"):
                raise SystemExit("Config error: sandbox.mode must be required or preferred")
            config.sandbox.mode = value
        for key in ("docker_binary", "image", "memory", "memory_swap"):
            if key in sandbox:
                value = sandbox[key]
                if not isinstance(value, str) or not value:
                    raise SystemExit(f"Config error: sandbox.{key} must be a non-empty string")
                setattr(config.sandbox, key, value)
        if "cpus" in sandbox:
            value = sandbox["cpus"]
            if not isinstance(value, (int, float)) or value <= 0:
                raise SystemExit("Config error: sandbox.cpus must be positive")
            config.sandbox.cpus = float(value)
        for key in ("pids_limit", "nofile_limit", "output_limit_bytes"):
            if key in sandbox:
                value = sandbox[key]
                if not isinstance(value, int) or value <= 0:
                    raise SystemExit(f"Config error: sandbox.{key} must be a positive integer")
                setattr(config.sandbox, key, value)

    if "compaction" in data:
        comp = data["compaction"]
        if not isinstance(comp, dict):
            raise SystemExit("Config error: [compaction] must be a table")
        known_compaction_keys = {
            "auto_threshold",
            "tool_result_limit",
            "tool_result_keep",
        }
        unknown_comp: set[str] = set(comp.keys()) - known_compaction_keys
        if unknown_comp:
            raise SystemExit(f"Unknown [compaction] keys: {', '.join(sorted(unknown_comp))}")
        if "auto_threshold" in comp:
            val = comp["auto_threshold"]
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                raise SystemExit("Config error: compaction.auto_threshold must be between 0 and 1")
            config.compaction.auto_threshold = float(val)
        if "tool_result_limit" in comp:
            val = comp["tool_result_limit"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit(
                    "Config error: compaction.tool_result_limit must be a positive integer"
                )
            config.compaction.tool_result_limit = val
        if "tool_result_keep" in comp:
            val = comp["tool_result_keep"]
            if not isinstance(val, int) or val <= 0:
                raise SystemExit(
                    "Config error: compaction.tool_result_keep must be a positive integer"
                )
            config.compaction.tool_result_keep = val

    if "mcp" in data:
        mcp = data["mcp"]
        if not isinstance(mcp, dict):
            raise SystemExit("Config error: [mcp] must be a table")
        unknown_mcp: set[str] = set(mcp.keys()) - {"servers"}
        if unknown_mcp:
            raise SystemExit(f"Unknown [mcp] keys: {', '.join(sorted(unknown_mcp))}")
        servers_raw = mcp.get("servers", [])
        if not isinstance(servers_raw, list):
            raise SystemExit("Config error: mcp.servers must be an array of tables")
        for i, srv in enumerate(servers_raw):
            if not isinstance(srv, dict):
                raise SystemExit(f"Config error: mcp.servers[{i}] must be a table")
            name = srv.get("name")
            if not isinstance(name, str) or not name:
                raise SystemExit(f"Config error: mcp.servers[{i}].name must be a non-empty string")
            transport = srv.get("transport", "stdio")
            if transport not in ("stdio", "streamable_http"):
                raise SystemExit(
                    "Config error: "
                    f"mcp.servers[{i}].transport must be 'stdio' or 'streamable_http'"
                )
            s = McpServerConfig(name=name, transport=transport)
            known_server_keys = {
                "name",
                "transport",
                "trusted",
                "command",
                "args",
                "env",
                "cwd",
                "url",
                "headers",
                "connect_timeout_s",
                "tool_timeout_s",
            }
            unknown_server = set(srv.keys()) - known_server_keys
            if unknown_server:
                raise SystemExit(
                    f"Unknown mcp.servers[{i}] keys: {', '.join(sorted(unknown_server))}"
                )
            if "trusted" in srv:
                val = srv["trusted"]
                if not isinstance(val, bool):
                    raise SystemExit(
                        f"Config error: mcp.servers[{i}].trusted must be a boolean"
                    )
                s.trusted = val
            if "command" in srv:
                val = srv["command"]
                if not isinstance(val, str):
                    raise SystemExit(f"Config error: mcp.servers[{i}].command must be a string")
                s.command = val
            if "args" in srv:
                val = srv["args"]
                if not isinstance(val, list):
                    raise SystemExit(f"Config error: mcp.servers[{i}].args must be an array")
                s.args = [str(a) for a in val]
            if "env" in srv:
                val = srv["env"]
                if not isinstance(val, dict):
                    raise SystemExit(f"Config error: mcp.servers[{i}].env must be a table")
                s.env = {str(k): str(v) for k, v in val.items()}
            if "cwd" in srv:
                val = srv["cwd"]
                if not isinstance(val, str):
                    raise SystemExit(f"Config error: mcp.servers[{i}].cwd must be a string")
                s.cwd = val
            if "url" in srv:
                val = srv["url"]
                if not isinstance(val, str):
                    raise SystemExit(f"Config error: mcp.servers[{i}].url must be a string")
                s.url = val
            if "headers" in srv:
                val = srv["headers"]
                if not isinstance(val, dict):
                    raise SystemExit(f"Config error: mcp.servers[{i}].headers must be a table")
                s.headers = {str(k): str(v) for k, v in val.items()}
            for key in ("connect_timeout_s", "tool_timeout_s"):
                if key in srv:
                    val = srv[key]
                    if not isinstance(val, (int, float)) or val <= 0:
                        raise SystemExit(
                            f"Config error: mcp.servers[{i}].{key} must be positive"
                        )
                    setattr(s, key, float(val))
            if not s.trusted:
                raise SystemExit(f"Config error: mcp.servers[{i}] must set trusted=true")
            if s.transport == "stdio" and not s.command:
                raise SystemExit(f"Config error: mcp.servers[{i}] stdio requires command")
            if s.transport == "streamable_http" and not s.url:
                raise SystemExit(
                    f"Config error: mcp.servers[{i}] streamable_http requires url"
                )
            if s.transport == "stdio":
                from pathlib import PureWindowsPath
                command = Path(s.command)
                if not command.is_absolute() and (
                    command.name != s.command or PureWindowsPath(s.command).name != s.command
                ):
                    raise SystemExit("Config error: MCP command must be absolute or a bare name")
            config.mcp.servers.append(s)


# 用 TARS_* 环境变量覆盖 config 中对应字段（若变量已设置）
def _apply_env(
    config: TarsConfig, environ: Mapping[str, str] | None = None, *, trusted: bool = True
) -> None:
    source = os.environ if environ is None else environ
    if not trusted:
        allowed = {
            "TARS_HOST", "TARS_PORT", "TARS_LOG_LEVEL", "TARS_LOG_FORMAT",
            "TARS_MAX_STEPS", "TARS_LLM_DEFAULT_MODEL", "TARS_LLM_MAX_TOKENS",
            "TARS_COMPACT_THRESHOLD", "TARS_COMPACT_TOOL_LIMIT", "TARS_COMPACT_TOOL_KEEP",
            "ANTHROPIC_API_KEY",
        }
        source = {key: value for key, value in source.items() if key in allowed}
    for key, env_name in (
        ("api_key", "TARS_LLM_API_KEY"), ("base_url", "TARS_LLM_BASE_URL"),
        ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    ):
        if env_name in source:
            setattr(config.llm, key, source[env_name])
    for key in ("total_timeout_s", "connect_timeout_s", "read_timeout_s",
                "write_timeout_s", "pool_timeout_s", "retry_delay_s"):
        env_name = f"TARS_LLM_{key.upper()}"
        if env_name in source:
            setattr(config.llm, key, _positive_number(source[env_name], env_name))
    request_limit = source.get("TARS_LLM_REQUEST_LIMIT")
    if request_limit is not None:
        config.llm.request_limit = _parse_request_limit(request_limit, "TARS_LLM_REQUEST_LIMIT")
    attempts = source.get("TARS_LLM_ATTEMPTS")
    if attempts is not None:
        if attempts not in ("1", "2"):
            raise SystemExit("Config error: TARS_LLM_ATTEMPTS must be 1 or 2")
        config.llm.attempts = int(attempts)
    mode = source.get("TARS_SANDBOX_MODE")
    if mode is not None:
        if mode not in ("required", "preferred"):
            raise SystemExit("Config error: TARS_SANDBOX_MODE must be required or preferred")
        config.sandbox.mode = mode

    host = source.get("TARS_HOST")
    if host is not None:
        config.host = host

    port_str = source.get("TARS_PORT")
    if port_str is not None:
        try:
            config.port = int(port_str)
        except ValueError:
            raise SystemExit(f"Config error: TARS_PORT must be an integer, got: {port_str!r}")

    log_level = source.get("TARS_LOG_LEVEL")
    if log_level is not None:
        config.logging.level = log_level

    log_file = source.get("TARS_LOG_FILE")
    if log_file is not None:
        config.logging.file = log_file

    log_format = source.get("TARS_LOG_FORMAT")
    if log_format is not None:
        config.logging.format = log_format

    max_steps_str = source.get("TARS_MAX_STEPS")
    if max_steps_str is not None:
        try:
            val = int(max_steps_str)
            if val <= 0:
                raise SystemExit(
                    "Config error: TARS_MAX_STEPS must be a positive integer,"
                    f" got: {max_steps_str!r}"
                )
            config.agent.max_steps = val
        except ValueError:
            raise SystemExit(
                f"Config error: TARS_MAX_STEPS must be an integer, got: {max_steps_str!r}"
            )

    default_model = source.get("TARS_LLM_DEFAULT_MODEL")
    if default_model is not None:
        config.llm.default_model = default_model

    max_tokens = source.get("TARS_LLM_MAX_TOKENS")
    if max_tokens is not None:
        try:
            max_tokens_value = int(max_tokens)
        except ValueError as exc:
            raise SystemExit("Config error: TARS_LLM_MAX_TOKENS must be an integer") from exc
        if max_tokens_value <= 0:
            raise SystemExit("Config error: TARS_LLM_MAX_TOKENS must be positive")
        config.llm.max_tokens = max_tokens_value

    trace_enabled = source.get("TARS_TRACE_ENABLED")
    if trace_enabled is not None:
        config.trace.enabled = _boolean(trace_enabled, "TARS_TRACE_ENABLED")

    trace_file = source.get("TARS_TRACE_FILE")
    if trace_file is not None:
        config.trace.file = trace_file

    trace_payload = source.get("TARS_TRACE_INCLUDE_LLM_PAYLOAD")
    if trace_payload is not None:
        config.trace.include_llm_payload = _boolean(trace_payload, "TARS_TRACE_INCLUDE_LLM_PAYLOAD")

    perm_timeout = source.get("TARS_PERMISSION_TIMEOUT_S")
    if perm_timeout is not None:
        try:
            perm_timeout_val = float(perm_timeout)
            if perm_timeout_val < 0:
                raise SystemExit(
                    f"Config error: TARS_PERMISSION_TIMEOUT_S must be >= 0, got: {perm_timeout!r}"
                )
            config.permission.timeout_s = perm_timeout_val
        except ValueError:
            raise SystemExit(
                f"Config error: TARS_PERMISSION_TIMEOUT_S must be a number, got: {perm_timeout!r}"
            )

    compact_threshold = source.get("TARS_COMPACT_THRESHOLD")
    if compact_threshold is not None:
        try:
            compact_threshold_val = float(compact_threshold)
            if not (0.0 <= compact_threshold_val <= 1.0):
                raise SystemExit(
                    "Config error: TARS_COMPACT_THRESHOLD must be between 0 and 1, "
                    f"got: {compact_threshold!r}"
                )
            config.compaction.auto_threshold = compact_threshold_val
        except ValueError:
            raise SystemExit(
                f"Config error: TARS_COMPACT_THRESHOLD must be a number, got: {compact_threshold!r}"
            )

    compact_tool_limit = source.get("TARS_COMPACT_TOOL_LIMIT")
    if compact_tool_limit is not None:
        try:
            compact_tool_limit_val = int(compact_tool_limit)
            if compact_tool_limit_val <= 0:
                raise SystemExit(
                    "Config error: TARS_COMPACT_TOOL_LIMIT must be a positive integer, "
                    f"got: {compact_tool_limit!r}"
                )
            config.compaction.tool_result_limit = compact_tool_limit_val
        except ValueError:
            raise SystemExit(
                "Config error: TARS_COMPACT_TOOL_LIMIT must be an integer, "
                f"got: {compact_tool_limit!r}"
            )

    compact_tool_keep = source.get("TARS_COMPACT_TOOL_KEEP")
    if compact_tool_keep is not None:
        try:
            compact_tool_keep_val = int(compact_tool_keep)
            if compact_tool_keep_val <= 0:
                raise SystemExit(
                    "Config error: TARS_COMPACT_TOOL_KEEP must be a positive integer, "
                    f"got: {compact_tool_keep!r}"
                )
            config.compaction.tool_result_keep = compact_tool_keep_val
        except ValueError:
            raise SystemExit(
                "Config error: TARS_COMPACT_TOOL_KEEP must be an integer, "
                f"got: {compact_tool_keep!r}"
            )

    sandbox_image = source.get("TARS_SANDBOX_IMAGE")
    if sandbox_image is not None:
        if not sandbox_image:
            raise SystemExit("Config error: TARS_SANDBOX_IMAGE must not be empty")
        config.sandbox.image = sandbox_image

    docker_binary = source.get("TARS_DOCKER_BINARY")
    if docker_binary is not None:
        if not docker_binary:
            raise SystemExit("Config error: TARS_DOCKER_BINARY must not be empty")
        config.sandbox.docker_binary = docker_binary


def _parse_request_limit(value: object, label: str) -> int | None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "unlimited":
            return None
        if not normalized.isascii() or not normalized.isdecimal():
            raise SystemExit(f"Config error: {label} must be a positive integer or unlimited")
        value = int(normalized)
    if type(value) is not int or value <= 0:
        raise SystemExit(f"Config error: {label} must be a positive integer or unlimited")
    return value


def _positive_number(value: object, label: str) -> float:
    try:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"Config error: {label} must be a positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise SystemExit(f"Config error: {label} must be a finite positive number")
    return result


def _boolean(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"Config error: {label} must be a boolean")


def _validate_config(config: TarsConfig) -> None:
    from urllib.parse import urlsplit

    if config.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("Config error: core.host must be a loopback host")
    if isinstance(config.port, bool) or not 1 <= config.port <= 65535:
        raise SystemExit("Config error: core.port must be between 1 and 65535")
    if not math.isfinite(config.permission.timeout_s) or config.permission.timeout_s < 0:
        raise SystemExit("Config error: permission.timeout_s must be finite and nonnegative")
    if config.llm.base_url:
        if not config.llm.api_key:
            raise SystemExit(
                "Config error: custom llm.base_url requires dedicated TARS_LLM_API_KEY"
            )
        try:
            parsed = urlsplit(config.llm.base_url)
            _ = parsed.port
            valid = bool(parsed.hostname) and not parsed.username and not parsed.password
            valid = valid and not parsed.query and not parsed.fragment
            valid = valid and (parsed.scheme == "https" or (
                parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
            ))
            valid = valid and not any(c.isspace() for c in config.llm.base_url)
        except ValueError:
            valid = False
        if not valid:
            raise SystemExit(
                "Config error: llm.base_url must be HTTPS or loopback HTTP without credentials"
            )
