"""Local W04 preparation gate; never starts Core, a model, or Docker commands."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from tars_agent.core.artifacts import ArtifactStore
from tars_agent.core.config import get_config
from tars_agent.core.events.bus import EventBus
from tars_agent.core.runner import AgentRunner
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.skills.loader import SkillLoader
from tars_agent.core.tools.runtime.factory import build_runtime_router
from tests.unit.test_runner import ScriptedProvider

DEMO_TOOLS = frozenset({"read_file", "list_dir", "write_file", "bash"})


class DemoPreparationError(ValueError):
    pass


def check_demo(workspace: Path, home: Path) -> dict[str, object]:
    workspace, home = workspace.resolve(strict=True), home.resolve(strict=True)
    if workspace.is_relative_to(home) or home.is_relative_to(workspace):
        raise DemoPreparationError("test HOME and workspace must be separate, non-nested directories")
    local = workspace / ".tars/skills/demo.md"
    if not local.is_file() or local.is_symlink():
        raise DemoPreparationError("local /demo preset is missing or is a symlink; stop preparation")
    if not (home / "config.toml").is_file():
        raise DemoPreparationError("isolated HOME/config.toml is missing")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("TARS_", "KAMA_", "ANTHROPIC_", "OPENAI_"))}
    environment.update(TARS_HOME=str(home), TARS_CONFIG=str(home / "config.toml"))
    previous = Path.cwd()
    with patch.dict(os.environ, environment, clear=True):
        try:
            os.chdir(workspace)
            try:
                config = get_config()
            except SystemExit as exc:
                raise DemoPreparationError("invalid isolated demonstration configuration") from exc
            if config.sandbox.mode != "required":
                raise DemoPreparationError("demonstration requires Docker required mode")
            if config.mcp.servers:
                raise DemoPreparationError("external MCP servers must not be configured")
            if (config.llm.api_key or config.llm.anthropic_api_key or config.llm.base_url
                    or config.llm.default_model != "w04-scripted-no-network"):
                raise DemoPreparationError("preparation requires the credential-free test provider configuration")
            if config.compaction.auto_threshold != 0 or config.trace.include_llm_payload:
                raise DemoPreparationError("automatic compaction and full model payload tracing must be disabled")
            skill = SkillLoader(workspace_root=workspace).resolve("demo")
            if (skill is None or skill.name != "demo" or len(skill.allowed_tools) != 4
                    or set(skill.allowed_tools) != DEMO_TOOLS
                    or skill.system_prompt_template.strip() != "$ARGUMENTS"):
                raise DemoPreparationError("invalid /demo preset; exact four-tool whitelist is required")
            # Resolve the same production binding, without constructing a service or a database.
            service = object.__new__(RuntimeService)
            goal = "W04 preparation sample"
            effective, options = service._resolve_skill(f"/demo {goal}", workspace_root=workspace)
            allowed = options.get("tool_whitelist")
            if effective != goal or not isinstance(allowed, list) or set(allowed) != DEMO_TOOLS:
                raise DemoPreparationError("Runtime did not bind the /demo whitelist")
            router = build_runtime_router(config.sandbox)
            # Do NOT call preflight/cleanup: their timeout and cleanup branches can kill/rm.
            provider = ScriptedProvider()
            runner = AgentRunner(config, bus=EventBus(), tool_runtime=router, provider=provider)
            registry = runner._build_registry(
                run_id="w04-preparation", session_id="w04-preparation", workspace_root=workspace,
                artifact_store=ArtifactStore(home / "preparation-artifacts"), provider=provider,
                tool_whitelist=allowed,
            )
            names = [str(schema["name"]) for schema in registry.tool_schemas()]
            if len(names) != 4 or set(names) != DEMO_TOOLS or router._allow_host_fallback:
                raise DemoPreparationError("effective registry or backend policy is not the expected demo scope")
            return {"workspace": str(workspace), "home": str(home), "sandbox_mode": config.sandbox.mode,
                    "runtime_backend_type": type(router._sandbox).__name__, "host_fallback": False,
                    "mcp_servers": len(config.mcp.servers), "effective_tools": names,
                    "spawn_registered": "spawn_agent" in names, "model_calls": len(provider.inputs),
                    "docker_commands_executed": 0,
                    "preset_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
                    "status": "prepared_only_no_core_or_container_created"}
        finally:
            os.chdir(previous)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check_demo(args.workspace, args.home)
    except (OSError, DemoPreparationError) as exc:
        parser.exit(1, f"W04 preparation stopped: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
