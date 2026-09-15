from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import LlmResponse, UsageStats
from tars_agent.core.persistence import Database, RunRecord, SessionRecord, StateRepository
from tars_agent.core.subagent.registry import BackgroundTaskRegistry
from tars_agent.core.subagent.tool import AgentResultTool, SpawnAgentTool
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter


@pytest.fixture
async def registry(tmp_path: Path) -> AsyncIterator[BackgroundTaskRegistry]:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    async with database.transaction() as session:
        repository = StateRepository(session)
        await repository.add_session(SessionRecord(
            id="sess-test", status="running", workspace_root=str(tmp_path),
            active_run_id="parent-run-01",
        ))
        await repository.add_run(RunRecord(
            id="parent-run-01", session_id="sess-test", kind="chat", status="running",
        ))
    registry = BackgroundTaskRegistry(database, EventBus())
    try:
        yield registry
    finally:
        await registry.shutdown()
        await database.dispose()


async def _stored_children(registry: BackgroundTaskRegistry) -> list[RunRecord]:
    async with registry._database.session() as session:
        return list((await session.scalars(select(RunRecord).where(
            RunRecord.kind == "subagent",
        ))).all())


def _make_provider(result_text: str = "child done") -> Any:
    provider = AsyncMock()
    provider.chat = AsyncMock(
        return_value=LlmResponse(
            stop_reason="end_turn",
            tool_calls=[],
            text=result_text,
            usage=UsageStats(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                context_pct=0.01,
            ),
        )
    )
    return provider


def _make_tool(
    tmp_path: Path,
    registry: BackgroundTaskRegistry,
    provider: Any = None,
    depth: int = 0,
) -> tuple[SpawnAgentTool, BackgroundTaskRegistry, EventBus]:
    bus = registry._bus
    tool = SpawnAgentTool(
        provider=provider or _make_provider(),
        parent_bus=bus,
        parent_run_id="parent-run-01",
        permission_manager=None,
        max_steps=5,
        task_registry=registry,
        runs_dir=tmp_path,
        session_id="sess-test",
        depth=depth,
        tool_runtime=RuntimeRouter(FakeRuntime(), allow_host_fallback=False),
    )
    return tool, registry, bus


# 功能：前台模式下 spawn_agent 应阻塞直到子 agent 完成并返回其结果
# 设计：使用返回 end_turn 的 mock provider，验证 tool_result.content 包含 provider 返回的文字
@pytest.mark.asyncio
async def test_foreground_returns_result(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    tool, _, _ = _make_tool(tmp_path, registry, _make_provider("analysis complete"))
    result = await tool.invoke({
        "description": "分析代码",
        "prompt": "分析 src/ 目录",
    })
    assert not result.is_error
    assert "analysis complete" in result.content


# 功能：后台模式应立即返回含 run_id 的消息，不阻塞等待子 agent
# 设计：run_in_background=true 后验证返回消息含 "run_id=" 并且任务注册表已有对应条目
@pytest.mark.asyncio
async def test_background_returns_run_id(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    tool, registry, _ = _make_tool(tmp_path, registry)
    result = await tool.invoke({
        "description": "后台任务",
        "prompt": "做点事",
        "run_in_background": True,
    })
    assert not result.is_error
    assert "run_id=" in result.content
    # extract run_id from message
    run_id = result.content.split("run_id=")[1].split(".")[0]
    assert registry.get(run_id) is not None


# 功能：后台任务未完成时 agent_result 应返回 "still running"
# 设计：用 Event 阻塞 provider.chat，在未等待任务完成时查询 agent_result
@pytest.mark.asyncio
async def test_agent_result_pending(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    event = asyncio.Event()

    async def slow_chat(*args: Any, **kwargs: Any) -> LlmResponse:
        await event.wait()
        return LlmResponse(
            stop_reason="end_turn",
            tool_calls=[],
            text="done",
            usage=UsageStats(0, 0, 0, 0, 0.0),
        )

    provider = MagicMock()
    provider.chat = slow_chat

    tool, registry, _ = _make_tool(tmp_path, registry, provider)
    spawn_result = await tool.invoke({
        "description": "slow task",
        "prompt": "do something slow",
        "run_in_background": True,
    })
    run_id = spawn_result.content.split("run_id=")[1].split(".")[0]

    result_tool = AgentResultTool(registry)
    result = await result_tool.invoke({"run_id": run_id})
    assert result.content == "still running"
    assert not result.is_error

    event.set()
    await asyncio.sleep(0.05)


# 功能：后台任务完成后 agent_result 应返回子 agent 的最终文本
# 设计：等待后台任务 task 完成后调用 agent_result，断言返回内容与 provider 结果一致
@pytest.mark.asyncio
async def test_agent_result_done(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    tool, registry, _ = _make_tool(tmp_path, registry, _make_provider("final answer"))
    spawn_result = await tool.invoke({
        "description": "bg task",
        "prompt": "do it",
        "run_in_background": True,
    })
    run_id = spawn_result.content.split("run_id=")[1].split(".")[0]

    entry = registry.get(run_id)
    assert entry is not None
    task, _ = entry
    await asyncio.wait_for(task, timeout=5.0)

    result_tool = AgentResultTool(registry)
    result = await result_tool.invoke({"run_id": run_id})
    assert not result.is_error
    assert "final answer" in result.content


# 功能：depth=2 时调用 spawn_agent 应返回 is_error=True（嵌套限制）
# 设计：构造 depth=2 的工具，断言 invoke 直接返回错误而不调用 provider
@pytest.mark.asyncio
async def test_nesting_limit(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    provider = _make_provider()
    tool, _, _ = _make_tool(tmp_path, registry, provider, depth=2)
    result = await tool.invoke({
        "description": "nested",
        "prompt": "do nested work",
    })
    assert result.is_error
    assert "nesting limit" in result.content
    provider.chat.assert_not_called()


# 功能：agent_result 查询不存在的 run_id 应返回 is_error=True
# 设计：空 registry 中查询随机 run_id，验证错误消息含 "Unknown"
@pytest.mark.asyncio
async def test_agent_result_unknown_run_id(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    tool = AgentResultTool(registry)
    result = await tool.invoke({"run_id": "nonexistent-id"})
    assert result.is_error
    assert "Unknown" in result.content


# 功能：SubagentStartedEvent 应在前台 spawn 时发布到父 bus
# 设计：订阅父 bus 收集所有事件，断言 subagent.started 出现，且 parent_run_id 和 description 正确
@pytest.mark.asyncio
async def test_foreground_publishes_started_event(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    from tars_agent.core.bus.events import SubagentStartedEvent

    tool, _, bus = _make_tool(tmp_path, registry)
    events: list[Any] = []

    async def _collect(e: Any) -> None:
        events.append(e)

    bus.subscribe(_collect)

    await tool.invoke({
        "description": "test task",
        "prompt": "test prompt",
    })
    started = [e for e in events if isinstance(e, SubagentStartedEvent)]
    assert len(started) == 1
    assert started[0].parent_run_id == "parent-run-01"
    assert started[0].description == "test task"


async def test_unknown_profile_never_starts_model(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    tool, registry, _ = _make_tool(tmp_path, registry)
    result = await tool.invoke({"description": "x", "prompt": "x", "subagent_type": "missing-profile"})
    assert result.is_error and "Unknown" in result.content
    assert await _stored_children(registry) == []
    tool._provider.chat.assert_not_awaited()


def test_empty_profile_and_parent_whitelist_cannot_expand_tools(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    from tars_agent.core.agents.loader import AgentProfile
    tool, _, bus = _make_tool(tmp_path, registry)
    empty = AgentProfile("empty", "", "", allowed_tools=[])
    assert tool._build_child_registry(bus, "child", empty).tool_schemas() == []
    tool._parent_allowed_tools = {"read_file"}
    broad = AgentProfile("all", "", "", allow_all_tools=True)
    assert [schema["name"] for schema in tool._build_child_registry(bus, "child", broad).tool_schemas()] == ["read_file"]


async def test_profile_model_reaches_provider(tmp_path: Path, registry: BackgroundTaskRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    import tars_agent.core.subagent.tool as module
    from tars_agent.core.agents.loader import AgentProfile
    provider = _make_provider("base")
    chosen = _make_provider("chosen")
    provider.with_model = MagicMock(return_value=chosen)
    monkeypatch.setattr(module._profile_loader, "load", lambda *args, **kwargs: AgentProfile(
        "alternate", "", "", model="alternate-model"))
    tool, _, _ = _make_tool(tmp_path, registry, provider)
    result = await tool.invoke({"description": "x", "prompt": "x", "subagent_type": "alternate"})
    assert result.content == "chosen"
    provider.with_model.assert_called_once_with("alternate-model")
    provider.chat.assert_not_awaited()
    chosen.chat.assert_awaited_once()


async def test_cancelled_parent_cannot_spawn(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    tool, registry, _ = _make_tool(tmp_path, registry)
    registry.block_descendants({"parent-run-01"})
    result = await tool.invoke({"description": "x", "prompt": "x"})
    assert result.is_error and "cancellation" in result.content
    assert await _stored_children(registry) == []
    tool._provider.chat.assert_not_awaited()


async def test_repeated_cancel_does_not_interrupt_child_cleanup(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    from tars_agent.core.context import ExecutionContext
    await registry.create_run(run_id="child", session_id="sess-test", parent_run_id="parent-run-01",
                              description="x", prompt="x", depth=1, background=True)
    started = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    async def work():
        await registry.mark_running("child")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            finished.set()
    child = asyncio.create_task(work())
    registry.register("child", child, ExecutionContext(run_id="child", goal="x", max_steps=1),
                      session_id="sess-test", parent_run_id="parent-run-01")
    await started.wait()
    first = asyncio.create_task(registry.cancel("child", session_id="sess-test"))
    await cleaning.wait()
    second = asyncio.create_task(registry.cancel("child", session_id="sess-test"))
    await asyncio.sleep(0)
    release.set()
    assert all(await asyncio.gather(first, second))
    assert finished.is_set()
    assert (await registry.get_snapshot("child")).status == "cancelled"


async def test_omitted_profile_passes_no_tool_schemas_to_model(tmp_path: Path, registry: BackgroundTaskRegistry) -> None:
    provider = _make_provider("reasoning only")
    tool, _, _ = _make_tool(tmp_path, registry, provider)
    result = await tool.invoke({"description": "x", "prompt": "reason without tools"})
    assert not result.is_error
    assert provider.chat.call_args.kwargs["tool_schemas"] == []


def test_orchestrate_profiles_get_only_the_parent_permitted_workspace_tools(
    tmp_path: Path, registry: BackgroundTaskRegistry, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.agents.loader import AgentProfileLoader
    from tars_agent.core.skills.loader import SkillLoader

    monkeypatch.setenv("TARS_HOME", str(tmp_path / "trusted-home"))
    skill = SkillLoader(workspace_root=tmp_path).resolve("orchestrate")
    assert skill is not None
    profiles = AgentProfileLoader()
    executor = profiles.load("executor", workspace_root=tmp_path)
    reviewer = profiles.load("reviewer", workspace_root=tmp_path)
    assert executor is not None and reviewer is not None
    tool, _, bus = _make_tool(tmp_path, registry)
    tool._parent_allowed_tools = set(skill.allowed_tools)

    executor_tools = {schema["name"] for schema in tool._build_child_registry(
        bus, "executor", executor,
    ).tool_schemas()}
    assert {"read_file", "list_dir", "write_file", "bash"} <= executor_tools
    reviewer_tools = {schema["name"] for schema in tool._build_child_registry(
        bus, "reviewer", reviewer,
    ).tool_schemas()}
    assert reviewer_tools == {"read_file", "list_dir"}

    tool._parent_allowed_tools -= {"write_file", "bash"}
    limited_tools = {schema["name"] for schema in tool._build_child_registry(
        bus, "limited-executor", executor,
    ).tool_schemas()}
    assert {"read_file", "list_dir"} <= limited_tools
    assert not {"write_file", "bash"} & limited_tools
