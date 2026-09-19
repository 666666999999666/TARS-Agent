"""Real Core with the existing ScriptedProvider and test-only observation guards."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

from tars_agent.core.app import CoreApp
from tars_agent.core.config import get_config
from tars_agent.core.llm.types import LlmResponse, ToolCallBlock
from tars_agent.core.tools.runtime.docker import DockerRuntime
from tests.unit.test_runner import ScriptedProvider
from tests.w04_preparation import DEMO_TOOLS, check_demo
from tests.w04_support import (
    EXECUTION,
    ROOT,
    append,
    digest,
    process_identity,
    register_container,
    save,
    scope_path,
    verify_container,
    verify_process,
)

SCOPE = scope_path(os.environ["W04_SCOPE"])
EVIDENCE = Path(os.environ["W04_EVIDENCE"]).resolve()
IMAGE = json.loads((EXECUTION / "image-result.json").read_text())["selected"]
PRESET_HASH = digest(ROOT / "docs/w04/demo.md")


class ObservedDockerRuntime(DockerRuntime):
    """Keep the production operation; veto any cleanup whose identity is unconfirmed."""

    def __init__(self, config):
        super().__init__(config)
        self.owned = {}
        self.exec_identities = {}
        self.evidence = EVIDENCE / "docker-operations.jsonl"
        save(EVIDENCE / "runtime-identity.json", {"instance": self._instance_id,
             "image": IMAGE, "workspace": str(SCOPE / "workspace"), "core_pid": os.getpid()})

    async def _command(self, args, *, timeout_s=None):
        if args[1] == "rm":
            targets = [arg for arg in args[2:] if not arg.startswith("-")]
            for cid in targets:
                if cid not in self.owned:
                    raise RuntimeError("W04 guard refuses cleanup of an unregistered resource")
                await asyncio.to_thread(verify_container, self.owned[cid], self.evidence, allow_absent=True)
            append(self.evidence, {"phase": "production_removal", "targets": targets})
        result = await super()._command(args, timeout_s=timeout_s)
        append(self.evidence, {"phase": "production_command", "args": args, "exit_code": result[0],
                               "stderr": result[2]})
        if args[1] == "run" and result[0] == 0:
            run = next(arg.split("=", 1)[1] for arg in args if arg.startswith("com.tars-agent.run="))
            cid = result[1].strip()
            value = await asyncio.to_thread(register_container, cid, workspace=SCOPE / "workspace",
                instance=self._instance_id, run=run, image=IMAGE["id"], log=self.evidence)
            self.owned[cid] = value
            save(EVIDENCE / "containers" / f"{cid}.json", value)
        return result

    async def _exec(self, container, request):
        original = request.on_started

        async def observed(backend, cid):
            await asyncio.to_thread(verify_container, self.owned[cid], self.evidence)
            process, _run = self._active[request.invocation_id]
            identity = await asyncio.to_thread(process_identity, process.pid)
            if cid not in identity["CommandLine"] or "/opt/tars/worker.py" not in identity["CommandLine"]:
                raise RuntimeError("docker exec process does not match the owned invocation")
            self.exec_identities[request.invocation_id] = identity
            append(self.evidence, {"phase": "exec_identity", "invocation": request.invocation_id,
                                  "identity": identity, "container_id": cid})
            if original is not None:
                await original(backend, cid)

        return await super()._exec(container, replace(request, on_started=observed))

    async def cancel(self, invocation_id):
        active = self._active.get(invocation_id)
        if active is not None:
            process, run = active
            if process.returncode is None:
                identity = self.exec_identities.get(invocation_id)
                if identity is None:
                    raise RuntimeError("no startup identity for docker exec; stop")
                await asyncio.to_thread(verify_process, identity)
            container = self._containers.get(run)
            if container is not None:
                await asyncio.to_thread(verify_container, self.owned[container.id], self.evidence)
            append(self.evidence, {"phase": "production_cancel", "invocation": invocation_id})
        return await super().cancel(invocation_id)


class GuardedScriptedProvider(ScriptedProvider):
    """Fixed responses only, with the live whitelist checked before every response."""

    @classmethod
    def from_config(cls, config):
        if config.api_key or config.anthropic_api_key or config.base_url:
            raise RuntimeError("real provider configuration is forbidden in W04")
        return cls()

    async def close(self):
        pass

    async def chat(self, messages, tool_schemas, bus, run_id, **kwargs):
        prepared = check_demo(SCOPE / "workspace", SCOPE / "home")
        live = get_config()
        names = [schema["name"] for schema in tool_schemas]
        if (prepared["preset_sha256"] != PRESET_HASH or len(names) != 4
                or set(names) != DEMO_TOOLS or live.sandbox.mode != "required"
                or live.sandbox.image != IMAGE["tag"] or live.mcp.servers
                or live.compaction.auto_threshold != 0):
            append(EVIDENCE / "provider.jsonl", {"guard": "failed", "run_id": run_id, "tools": names})
            raise RuntimeError("W04 live preset/backend/whitelist check failed; no tool response")
        if not self.inputs:
            goal = next(message["content"] for message in reversed(messages)
                        if message["role"] == "user" and isinstance(message["content"], str))
            plan_file = (SCOPE / "home" / os.environ["W04_PLAN_FILE"]).resolve(strict=True)
            if not plan_file.is_relative_to((SCOPE / "home").resolve()):
                raise RuntimeError("test plan is outside the authorized HOME")
            plans = json.loads(plan_file.read_text(encoding="utf-8"))
            if goal not in plans:
                raise RuntimeError("test provider has no authorized plan for this exact input")
            plan = plans[goal]
            calls = [ToolCallBlock(id=call["id"], name=call["name"], input=call["input"])
                     for call in plan]
            self.responses = ([LlmResponse(stop_reason="tool_use", tool_calls=calls)] if calls else [])
            self.responses.append(LlmResponse(stop_reason="end_turn", text="W04 scripted response only"))
        append(EVIDENCE / "provider.jsonl", {"guard": "passed", "run_id": run_id,
            "preset_sha256": PRESET_HASH, "tools": names, "messages": messages,
            "provider": "existing tests.unit.test_runner.ScriptedProvider", "real_model": False})
        return await super().chat(messages, tool_schemas, bus, run_id, **kwargs)


def main():
    import tars_agent.core.app as app_module
    import tars_agent.core.runner as runner_module
    import tars_agent.core.tools.runtime.factory as factory

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    check_demo(SCOPE / "workspace", SCOPE / "home")
    source = {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / "src").rglob("*.py"))}
    save(EVIDENCE / "core-source.json", {"pid": os.getpid(), "app_file": app_module.__file__,
         "runner_file": runner_module.__file__, "source_files": source,
         "test_overrides": ["ScriptedProvider injection", "Docker identity observation/veto only"],
         "production_files_modified": False})
    app_module.AnthropicProvider = GuardedScriptedProvider
    runner_module.AnthropicProvider = GuardedScriptedProvider
    factory.DockerRuntime = ObservedDockerRuntime
    asyncio.run(CoreApp().run())


if __name__ == "__main__":
    main()
