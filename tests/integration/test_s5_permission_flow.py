"""
S5 permission flow integration tests.

No daemon subprocess needed — uses AgentRunner in-process with a mock LLM
provider, the real PermissionManager and a FakeRuntime; no real model or shell is used.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from tars_agent.core.artifacts import ArtifactStore
from tars_agent.core.config import TarsConfig
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import LlmResponse, ToolCallBlock
from tars_agent.core.permissions.manager import PermissionManager
from tars_agent.core.runner import AgentRunner
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter

# ── stub providers ────────────────────────────────────────────────────────────


class _SingleBashProvider:
    """Step 1: bash tool call. Step 2: end_turn."""

    def __init__(self, command: str = "echo hello") -> None:
        self._command = command
        self._step = 0

    async def chat(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        self._step += 1
        if self._step == 1:
            tc = ToolCallBlock(id="tc1", name="bash", input={"command": self._command})
            return LlmResponse(stop_reason="tool_use", tool_calls=[tc])
        return LlmResponse(stop_reason="end_turn", text="done")


class _TwoBashProvider:
    """Step 1+2: two separate bash calls. Step 3: end_turn."""

    def __init__(self, second_command: str = "echo second") -> None:
        self._step = 0
        self._second_command = second_command

    async def chat(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        self._step += 1
        if self._step == 1:
            tc = ToolCallBlock(id="tc1", name="bash", input={"command": "echo first"})
            return LlmResponse(stop_reason="tool_use", tool_calls=[tc])
        if self._step == 2:
            tc = ToolCallBlock(id="tc2", name="bash", input={"command": self._second_command})
            return LlmResponse(stop_reason="tool_use", tool_calls=[tc])
        return LlmResponse(stop_reason="end_turn", text="done")


# ── helper ────────────────────────────────────────────────────────────────────


def _runner(
    provider: object,
    bus: EventBus,
    manager: PermissionManager,
    tmp_path: Path,
    max_steps: int = 10,
) -> AgentRunner:
    config = TarsConfig()
    config.agent.max_steps = max_steps
    return AgentRunner(
        config,
        bus=bus,
        provider=provider,  # type: ignore[arg-type]
        permission_manager=manager,
        tool_runtime=RuntimeRouter(FakeRuntime()),
    )


def _inputs(tmp_path: Path) -> dict:
    return {
        "run_id": "permission-run",
        "session_id": "permission-session",
        "workspace_root": tmp_path,
        "history": [{"role": "user", "content": "Run the requested command"}],
        "artifact_store": ArtifactStore(tmp_path / "artifacts" / "sessions"),
        "session_notes": "",
    }


# ── tests ─────────────────────────────────────────────────────────────────────


# 功能：验证 allow_once 决策后工具正常执行并写入 tool.call_finished 事件
# 设计：在 permission.requested 事件到达时同步调用 manager.respond("allow_once")；
#       Future 在同一 event-loop turn 内解决，工具随后执行；断言 tool.call_finished 存在且 tool.call_failed 不存在
async def test_permission_allow_once_tool_executes(tmp_path: Path) -> None:
    manager = PermissionManager()
    bus = EventBus()
    event_types: list[str] = []

    async def collect(e: BaseModel) -> None:
        t = getattr(e, "type", "")
        event_types.append(t)
        if t == "permission.requested":
            manager.respond(
                getattr(e, "request_id", ""),
                getattr(e, "session_id", ""),
                "allow_once",
            )

    bus.subscribe(collect)
    outcome = await _runner(_SingleBashProvider(), bus, manager, tmp_path).run_and_capture(
        "run bash", **_inputs(tmp_path)
    )

    assert "permission.requested" in event_types
    assert "tool.call_finished" in event_types
    assert "tool.call_failed" not in event_types
    assert outcome.status == "success"


# 功能：验证 deny_once 决策后工具不执行，事件流中出现 permission_denied 错误
# 设计：在 permission.requested 时 respond("deny_once")；断言 tool.call_failed 的 error_class 为
#       "permission_denied"，且 tool.call_finished 不出现，确认工具从未被调用
async def test_permission_deny_once_tool_not_executed(tmp_path: Path) -> None:
    manager = PermissionManager()
    bus = EventBus()
    event_types: list[str] = []
    failed_events: list[BaseModel] = []

    async def collect(e: BaseModel) -> None:
        t = getattr(e, "type", "")
        event_types.append(t)
        if t == "permission.requested":
            manager.respond(
                getattr(e, "request_id", ""),
                getattr(e, "session_id", ""),
                "deny_once",
            )
        if t == "tool.call_failed":
            failed_events.append(e)

    bus.subscribe(collect)
    await _runner(_SingleBashProvider(), bus, manager, tmp_path).run_and_capture("run bash", **_inputs(tmp_path))

    assert "permission.requested" in event_types
    assert "tool.call_failed" in event_types
    assert "tool.call_finished" not in event_types
    assert getattr(failed_events[0], "error_class", None) == "permission_denied"


# 相同参数复用本会话授权；参数变化必须重新审批。
@pytest.mark.parametrize(("second_command", "expected_requests"), [
    ("echo first", 1), ("echo second", 2),
])
async def test_allow_session_cache_is_scoped_to_identical_parameters(
    tmp_path: Path, second_command: str, expected_requests: int,
) -> None:
    manager = PermissionManager()
    bus = EventBus()
    perm_requested_count = 0

    async def collect(e: BaseModel) -> None:
        nonlocal perm_requested_count
        if getattr(e, "type", "") == "permission.requested":
            perm_requested_count += 1
            manager.respond(
                getattr(e, "request_id", ""),
                getattr(e, "session_id", ""),
                "allow_session",
            )

    bus.subscribe(collect)
    outcome = await _runner(_TwoBashProvider(second_command), bus, manager, tmp_path).run_and_capture(
        "run two bash commands", **_inputs(tmp_path)
    )

    assert perm_requested_count == expected_requests
    assert outcome.status == "success"
