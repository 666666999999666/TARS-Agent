from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tars_agent.core.app import CoreApp
from tars_agent.core.bus.commands import (
    AgentRunCommand,
    CoreShutdownCommand,
    CoreShutdownResult,
    PingCommand,
    PongResult,
)
from tars_agent.core.bus.events import CoreStartedEvent
from tars_agent.core.persistence import CURRENT_SCHEMA_REVISION


# 功能：验证 PingCommand 序列化后再反序列化，client 和 type 字段完整保留
# 设计：JSON 往返测试确认 wire 协议的序列化正确性，type 字段是 discriminated union 的判别键
def test_ping_command_roundtrip() -> None:
    cmd = PingCommand(client="cli/0.0.1")
    cmd2 = PingCommand.model_validate_json(cmd.model_dump_json())
    assert cmd2.client == "cli/0.0.1"
    assert cmd2.type == "core.ping"


async def test_agent_run_handler_preserves_requested_workspace(tmp_path: Path) -> None:
    class _FakeRuntime:
        def __init__(self) -> None:
            self.workspace_root: Path | None = None

        async def create_session(
            self,
            mode: str,
            *,
            title: str,
            workspace_root: Path | None,
        ) -> SimpleNamespace:
            assert mode == "one_shot"
            assert title == "inspect workspace"
            self.workspace_root = workspace_root
            return SimpleNamespace(id="sess-1")

        async def submit_message(
            self, session_id: str, goal: str
        ) -> SimpleNamespace:
            assert session_id == "sess-1"
            assert goal == "inspect workspace"
            return SimpleNamespace(run_id="run-1")

    command = AgentRunCommand(
        goal="inspect workspace",
        workspace_root=str(tmp_path),
    )
    app = CoreApp()
    runtime = _FakeRuntime()
    app._runtime = runtime  # type: ignore[assignment]

    result = await app._agent_run_handler(command.model_dump())

    assert result.run_id == "run-1"
    assert runtime.workspace_root == tmp_path


# 功能：验证 PingCommand 的 type 字段默认值为 "core.ping"
# 设计：Literal 默认值测试，type 是 Command union 的判别键，必须与 union 定义完全一致，否则反序列化时会路由到错误类型
def test_ping_command_default_type() -> None:
    cmd = PingCommand(client="x")
    assert cmd.type == "core.ping"


# 功能：验证缺少必填 client 字段时 pydantic 校验失败
# 设计：传入空 dict 触发校验，确认 client 是必填字段，防止 daemon 收到不完整的 ping 命令进入 handler
def test_ping_command_missing_client_raises() -> None:
    with pytest.raises(ValidationError):
        PingCommand.model_validate({})


# 功能：验证 PongResult 序列化往返后所有字段完整保留
# 设计：与 PingCommand 对称，测试命令-响应对的两端序列化，确认 int 和 str 字段类型在往返中不变
def test_pong_result_roundtrip() -> None:
    pong = PongResult(
        server_version="0.0.1",
        uptime_ms=42,
        received_at="2026-05-11T00:00:00Z",
        schema_revision=CURRENT_SCHEMA_REVISION,
    )
    pong2 = PongResult.model_validate(pong.model_dump())
    assert pong2.server_version == "0.0.1"
    assert pong2.uptime_ms == 42
    assert pong2.protocol_version == 2
    assert pong2.event_schema_versions == [1]


# 功能：验证 core.shutdown 命令与响应可稳定进行协议序列化
# 设计：检查固定 type 与默认 ok，防止 Core 注册的方法缺少对应 wire model
def test_core_shutdown_models_roundtrip() -> None:
    command = CoreShutdownCommand.model_validate_json(
        CoreShutdownCommand(token="test-control-token").model_dump_json()
    )
    result = CoreShutdownResult.model_validate_json(CoreShutdownResult().model_dump_json())
    assert command.type == "core.shutdown"
    assert command.token == "test-control-token"
    assert result.ok is True


# 功能：验证 CoreStartedEvent 序列化往返后 listen_addr 和 type 字段正确保留
# 设计：CoreStartedEvent 是 daemon 启动通知，往返测试确认 type 的 Literal 约束在反序列化后保持（不被字段名覆盖）
def test_core_started_event_roundtrip() -> None:
    evt = CoreStartedEvent(listen_addr="127.0.0.1:7437", version="0.0.1")
    evt2 = CoreStartedEvent.model_validate_json(evt.model_dump_json())
    assert evt2.listen_addr == "127.0.0.1:7437"
    assert evt2.type == "core.started"


def test_core_shutdown_rejects_missing_control_token() -> None:
    with pytest.raises(ValidationError):
        CoreShutdownCommand.model_validate({})
