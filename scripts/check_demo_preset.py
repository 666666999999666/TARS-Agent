"""Offline preflight only: resolve /demo and build its actual registry without a model call."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.unit.test_runner import ScriptedProvider  # noqa: E402

from tars_agent.core.artifacts import ArtifactStore  # noqa: E402
from tars_agent.core.config import get_config  # noqa: E402
from tars_agent.core.events.bus import EventBus  # noqa: E402
from tars_agent.core.runner import AgentRunner  # noqa: E402
from tars_agent.core.runtime.service import RuntimeService  # noqa: E402
from tars_agent.core.tools.runtime.factory import build_runtime_router  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    evidence = args.evidence.resolve()
    if evidence.is_relative_to(workspace):
        parser.error("evidence must be outside the model workspace")
    local = workspace / ".tars/skills/demo.md"
    expected = (ROOT / "docs/w04/demo.md").read_bytes()
    if not local.is_file() or local.is_symlink() or local.read_bytes() != expected:
        parser.error("local /demo is missing or changed; stop, do not submit ordinary input")
    try:
        config = get_config()
    except SystemExit:
        parser.error("local configuration is invalid; no credentials are printed")
    if config.sandbox.mode != "required" or config.mcp.servers:
        parser.error("Docker required and zero external MCP servers are mandatory")
    effective, options = object.__new__(RuntimeService)._resolve_skill("/demo W05 preflight", workspace_root=workspace)
    allowed = options.get("tool_whitelist")
    tools = {"read_file", "list_dir", "write_file", "bash"}
    if effective != "W05 preflight" or not isinstance(allowed, list) or len(allowed) != 4 or set(allowed) != tools:
        parser.error("Runtime did not bind the exact four-tool whitelist")
    provider = ScriptedProvider()
    router = build_runtime_router(config.sandbox)
    registry = AgentRunner(config, bus=EventBus(), tool_runtime=router, provider=provider)._build_registry(
        run_id="w05-preflight", session_id="w05-preflight", workspace_root=workspace,
        artifact_store=ArtifactStore(evidence / "preflight-artifacts"), provider=provider,
        tool_whitelist=allowed,
    )
    names = [schema["name"] for schema in registry.tool_schemas()]
    if len(names) != 4 or set(names) != tools or router._allow_host_fallback:
        parser.error("effective registry/backend mismatch; stop")
    print(json.dumps({"prepared": True, "workspace": str(workspace), "tools": names,
                      "sandbox": "required", "host_fallback": False, "model_requests": 0,
                      "preset_sha256": hashlib.sha256(expected).hexdigest()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
