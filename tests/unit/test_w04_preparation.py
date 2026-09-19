from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tars_agent.cli.client import TerminalClient
from tars_agent.core.bus.events import PermissionRequestedEvent, RunFinishedEvent
from tars_agent.core.config import TarsConfig, get_config
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import LlmResponse, ToolCallBlock
from tars_agent.core.permissions.manager import PermissionManager
from tars_agent.core.persistence import Database
from tars_agent.core.runner import AgentRunner
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.tools.builtin.write_file import WriteFileTool
from tars_agent.core.tools.invocation import invoke_tool
from tars_agent.core.tools.registry import ToolRegistry
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter
from tars_agent.core.tools.runtime.docker import DockerRuntime
from tests.unit.test_cli_client import FakeInput
from tests.unit.test_runner import ScriptedProvider
from tests.w04_preparation import DEMO_TOOLS, DemoPreparationError, check_demo

ROOT = Path(__file__).resolve().parents[2]


def _directories(tmp_path: Path, preset: bytes | None = b"template") -> tuple[Path, Path]:
    workspace, home = tmp_path / "workspace", tmp_path / "home"
    (workspace / ".tars/skills").mkdir(parents=True)
    home.mkdir()
    (home / "config.toml").write_bytes((ROOT / "docs/w04/config.toml").read_bytes())
    if preset is not None:
        (workspace / ".tars/skills/demo.md").write_bytes(
            (ROOT / "docs/w04/demo.md").read_bytes() if preset == b"template" else preset,
        )
    return workspace, home


def test_demo_gate_checks_real_registry_without_docker_commands(tmp_path, monkeypatch):
    workspace, home = _directories(tmp_path)

    async def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not run Docker, preflight or cleanup")

    for name in ("_command", "preflight", "cleanup", "cleanup_run", "execute"):
        monkeypatch.setattr(DockerRuntime, name, forbidden)
    result = check_demo(workspace, home)
    assert result["sandbox_mode"] == "required"
    assert result["runtime_backend_type"] == "DockerRuntime"
    assert set(result["effective_tools"]) == DEMO_TOOLS
    assert result["host_fallback"] is False and result["mcp_servers"] == 0
    assert result["spawn_registered"] is False and result["model_calls"] == 0


@pytest.mark.parametrize("preset", [None, b"not a valid demo", b"\xff\xfe",
    b"---\nname: demo\nallowed_tools: [read_file, list_dir, write_file, bash]\n---\n$ARGUMENTS\n"])
def test_missing_or_invalid_local_demo_stops_before_registry(tmp_path, monkeypatch, preset):
    workspace, home = _directories(tmp_path, preset)
    # A valid global fallback must not rescue a missing/invalid local demonstration preset.
    (home / "skills").mkdir()
    (home / "skills/demo.md").write_bytes((ROOT / "docs/w04/demo.md").read_bytes())

    def forbidden(*args, **kwargs):
        raise AssertionError("the preparation gate must stop before building a registry")

    monkeypatch.setattr(AgentRunner, "_build_registry", forbidden)
    with pytest.raises(DemoPreparationError):
        check_demo(workspace, home)


def test_preferred_mode_is_rejected_by_demo_gate(tmp_path):
    workspace, home = _directories(tmp_path)
    config = home / "config.toml"
    config.write_text(config.read_text().replace('mode = "required"', 'mode = "preferred"'))
    with pytest.raises(DemoPreparationError, match="required mode"):
        check_demo(workspace, home)


async def test_demo_whitelist_reaches_test_provider_through_runtime(tmp_path, monkeypatch):
    workspace, home = _directories(tmp_path)
    check_demo(workspace, home)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("TARS_HOME", str(home))
    monkeypatch.setenv("TARS_CONFIG", str(home / "config.toml"))
    config = get_config()
    database = Database(home / "state.db")
    await database.create_schema()
    provider = ScriptedProvider(LlmResponse(stop_reason="end_turn", text="preparation only"))
    fake = FakeRuntime()
    bus = EventBus()
    finished = asyncio.Event()

    async def observe(event):
        if isinstance(event, RunFinishedEvent):
            finished.set()

    bus.subscribe(observe)
    runtime = RuntimeService(database, lambda: AgentRunner(
        config, bus=bus, provider=provider,
        tool_runtime=RuntimeRouter(fake, allow_host_fallback=False),
    ), bus, artifacts_root=home / "artifacts", llm_config=config.llm)
    waiter = asyncio.create_task(finished.wait())
    try:
        session = await runtime.create_session("chat", workspace_root=workspace)
        submitted = await runtime.submit_message(session.id, "/demo List the sample files")
        _done, pending = await asyncio.wait([waiter], timeout=5)
        assert not pending, "Runtime must finish before test cleanup"
        assert (await runtime.get_run(submitted.run_id)).status == "succeeded"
        assert set(tool["name"] for tool in provider.schemas) == DEMO_TOOLS
        assert provider.inputs[0][0][-1]["content"] == "List the sample files"
        assert config.sandbox.mode == "required" and not config.mcp.servers
        assert not fake.requests  # Registry verification only; no tool operation was requested.
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        try:
            await runtime.shutdown()  # Only in-process test tasks; no Docker or OS process.
        finally:
            await database.dispose()


async def test_full_cli_parameters_and_denial_do_not_reuse_changed_parameter_approval(tmp_path, capsys):
    prefix = "共同前缀-" * 500
    samples = [{"path": "result.txt", "content": prefix + tail,
                "metadata": {"nested": ["中文", {"value": "visible"}]}}
               for tail in ("TAIL_DENY_A", "TAIL_ALLOW_B", "TAIL_DENY_C")]
    decisions = ["deny_once", "allow_session", "deny_once"]
    manager = PermissionManager()
    fake = FakeRuntime()  # Only write_file below; never bash, Docker, subprocess or host fallback.
    registry = ToolRegistry()
    registry.register(WriteFileTool(RuntimeRouter(fake, allow_host_fallback=False)))
    bus = EventBus()
    requests = []
    displayed = []

    async def approve(event):
        if not isinstance(event, PermissionRequestedEvent):
            return
        index = len(requests)
        requests.append(event)
        assert len(fake.requests) == (0 if index < 2 else 1)
        if index < 2:
            assert not (tmp_path / "result.txt").exists()
        client = TerminalClient(TarsConfig(), interactive=True, reader=FakeInput())
        client._session_id = "w04-session"
        client._replaying = False  # Approval is shown after the initial event replay completes.
        client._permissions[event.request_id] = event.model_dump(mode="json")
        client._show_input()
        rendered = capsys.readouterr().err
        parameters = next(line for line in rendered.splitlines() if line.startswith("{"))
        assert json.loads(parameters) == samples[index]
        assert samples[index]["content"] in rendered
        displayed.append(rendered)
        assert manager.respond(event.request_id, "w04-session", decisions[index])

    bus.subscribe(approve)
    results = []
    for index, params in enumerate(samples):
        results.append(await invoke_tool(registry, ToolCallBlock(
            id=f"write-{index}", name="write_file", input=params,
        ), bus, "w04-unit-run", permission_manager=manager,
            session_id="w04-session", workspace_root=tmp_path))
    assert [result.is_error for result in results] == [True, False, True]
    assert len(fake.requests) == 1 and fake.requests[0].params == samples[1]
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == samples[1]["content"]
    assert len({request.request_id for request in requests}) == 3
    assert len({request.parameter_digest for request in requests}) == 3
    assert len({request.param_preview for request in requests}) == 1
    assert all("y=allow once" in display for display in displayed)
    print("W04_APPROVAL_EVIDENCE " + json.dumps({
        "requests": [request.model_dump(mode="json") for request in requests],
        "displayed_before_response": displayed, "decisions": decisions,
        "result_errors": [result.is_error for result in results],
        "executed_tool_ids": [request.tool_use_id for request in fake.requests],
        "written_content": (tmp_path / "result.txt").read_text(encoding="utf-8"),
    }, ensure_ascii=False))
