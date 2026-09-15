from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import os
import secrets
import signal
import time
from collections.abc import Awaitable, Callable
from datetime import UTC
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

import tars_agent
from tars_agent.core.bus.commands import (
    AgentRunCommand,
    AgentRunResult,
    CoreShutdownCommand,
    CoreShutdownResult,
    EventSubscribeCommand,
    EventSubscribeResult,
    McpServerInfo,
    McpStatusCommand,
    McpStatusResult,
    PermissionRespondCommand,
    PermissionRespondResult,
    PongResult,
    RunCancelCommand,
    RunGetCommand,
    RunInfo,
    RunMetricsCommand,
    SessionCloseCommand,
    SessionCloseResult,
    SessionCompactCommand,
    SessionCompactResult,
    SessionCreateCommand,
    SessionCreateResult,
    SessionGetCommand,
    SessionGetHistoryCommand,
    SessionGetHistoryResult,
    SessionGetResult,
    SessionInfo,
    SessionListCommand,
    SessionListResult,
    SessionResumeCommand,
    SessionResumeResult,
    SessionRetryCommand,
    SessionRetryResult,
    SessionSendMessageCommand,
    SessionSendMessageResult,
)
from tars_agent.core.bus.envelope import HandlerError
from tars_agent.core.config import TarsConfig, get_config
from tars_agent.core.control import (
    CORE_LAUNCH_ID_ENV,
    DaemonControl,
    control_file_for,
    remove_control_file,
    write_control_file,
)
from tars_agent.core.events.bus import EventBus
from tars_agent.core.events.durable import DurableEventHub
from tars_agent.core.llm.provider import AnthropicProvider
from tars_agent.core.logging_setup import setup_logging
from tars_agent.core.mcp.server import McpServerManager
from tars_agent.core.observability.metrics import RunMetricsProjection
from tars_agent.core.paths import tars_home
from tars_agent.core.permissions.manager import PermissionManager
from tars_agent.core.permissions.storage import load_policy_file
from tars_agent.core.persistence import (
    CURRENT_SCHEMA_REVISION,
    EVENT_SCHEMA_VERSION,
    Database,
    bootstrap_state,
)
from tars_agent.core.processes import finish_cleanup
from tars_agent.core.runner import AgentRunner
from tars_agent.core.runtime import RunSnapshot, RuntimeService, SessionSnapshot
from tars_agent.core.subagent.registry import BackgroundTaskRegistry
from tars_agent.core.tools.runtime import RuntimeRouter, initialize_runtime_router
from tars_agent.core.trace.record import TraceRecord
from tars_agent.core.trace.writer import TraceWriter
from tars_agent.core.transport.ipc_broadcaster import IpcEventBroadcaster
from tars_agent.core.transport.socket_server import SocketServer, get_connection_sender

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.datetime.now(UTC).isoformat()


def _config_log_summary(config: TarsConfig) -> dict[str, Any]:
    """Return an allowlisted summary that can never contain configured secrets."""
    return {
        "host": config.host,
        "port": config.port,
        "log_level": config.logging.level,
        "model": config.llm.default_model,
        "max_steps": config.agent.max_steps,
        "trace_enabled": config.trace.enabled,
        "sandbox": {
            "image": config.sandbox.image,
            "network": "none",
        },
        "mcp_servers": [
            {
                "name": server.name,
                "transport": server.transport,
                "trusted": server.trusted,
            }
            for server in config.mcp.servers
        ],
    }


class CoreApp:
    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._bus = EventBus()
        self._broadcaster: IpcEventBroadcaster | None = None
        self._event_hub: DurableEventHub | None = None
        self._trace: TraceWriter | None = None
        self._config: TarsConfig | None = None
        self._database: Database | None = None
        self._runtime: RuntimeService | None = None
        self._permission_manager: PermissionManager | None = None
        self._mcp_manager: McpServerManager | None = None
        self._tool_runtime: RuntimeRouter | None = None
        self._subagent_registry: BackgroundTaskRegistry | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._control_token = secrets.token_urlsafe(32)
        self._launch_id = os.environ.get(CORE_LAUNCH_ID_ENV) or secrets.token_urlsafe(32)

    # 请求 Core 有序退出；响应写回后事件循环进入统一清理流程
    async def _shutdown_handler(self, params: dict[str, Any]) -> CoreShutdownResult:
        command = CoreShutdownCommand.model_validate(params)
        if not secrets.compare_digest(
            command.token.encode("utf-8"), self._control_token.encode("utf-8"),
        ):
            raise HandlerError(-32001, "invalid daemon control token")
        if self._shutdown_event is None:
            raise RuntimeError("shutdown lifecycle is not initialized")
        return CoreShutdownResult()

    # 处理 core.ping 请求，返回服务版本、运行时长和接收时间
    async def _ping_handler(self, params: dict[str, Any]) -> PongResult:
        client = params.get("client", "unknown")
        logger.debug("ping from %s", client)
        return PongResult(
            server_version=tars_agent.__version__,
            uptime_ms=int((time.monotonic() - self._start_time) * 1000),
            received_at=datetime.datetime.now(datetime.UTC).isoformat(),
            launch_id=self._launch_id,
            schema_revision=CURRENT_SCHEMA_REVISION,
            event_schema_versions=[EVENT_SCHEMA_VERSION],
        )

    # 将 EventBus 事件写入 trace（作为 EventBus 订阅者）
    async def _trace_event_handler(self, event: BaseModel) -> None:
        assert self._trace is not None
        event_dict = event.model_dump()
        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE",
                layer="event",
                kind="event",
                run_id=event_dict.get("run_id"),
                data=event_dict,
            )
        )

    # 启动一次 agent run：异步创建 AgentRunner 并立即返回 run_id
    async def _agent_run_handler(self, params: dict[str, Any]) -> AgentRunResult:
        assert self._runtime is not None
        cmd = AgentRunCommand.model_validate(params)
        workspace_root = Path(cmd.workspace_root) if cmd.workspace_root is not None else None
        session = await self._runtime.create_session(
            mode="one_shot",
            title=cmd.goal[:40],
            workspace_root=workspace_root,
        )
        submitted = await self._runtime.submit_message(session.id, cmd.goal)
        return AgentRunResult(run_id=submitted.run_id)

    # 创建 chat 或 one_shot session，并返回 session_id
    async def _session_create_handler(self, params: dict[str, Any]) -> SessionCreateResult:
        assert self._runtime is not None
        cmd = SessionCreateCommand.model_validate(params)
        workspace_root = Path(cmd.workspace_root) if cmd.workspace_root is not None else None
        session = await self._runtime.create_session(
            mode=cmd.mode,
            title=cmd.title,
            workspace_root=workspace_root,
        )
        assert session.workspace_root is not None
        return SessionCreateResult(
            session_id=session.id,
            status=session.status,  # type: ignore[arg-type]
            workspace_root=session.workspace_root,
        )

    # 分页返回持久化 Session 摘要，不触发任何 Runtime 状态变更
    async def _session_list_handler(self, params: dict[str, Any]) -> SessionListResult:
        assert self._runtime is not None
        cmd = SessionListCommand.model_validate(params)
        sessions = await self._runtime.list_sessions(
            status=cmd.status,
            limit=cmd.limit,
            offset=cmd.offset,
        )
        return SessionListResult(sessions=[self._session_info(item) for item in sessions])

    # 恢复既有 Session，并返回活动 Run 与该 Session 的最新事件 cursor
    async def _session_resume_handler(self, params: dict[str, Any]) -> SessionResumeResult:
        assert self._runtime is not None
        cmd = SessionResumeCommand.model_validate(params)
        session = await self._runtime.resume_session(cmd.session_id)
        assert self._event_hub is not None
        await self._event_hub.flush()
        active_run = None
        if session.active_run_id is not None:
            active_run = self._run_info(await self._runtime.get_run(session.active_run_id))
        latest_cursor = await self._runtime.latest_event_cursor(session.id)
        return SessionResumeResult(
            session=self._session_info(session),
            active_run=active_run,
            latest_cursor=latest_cursor,
        )

    async def _session_get_handler(self, params: dict[str, Any]) -> SessionGetResult:
        """Return a Session snapshot without publishing a resume event or mutating state."""
        assert self._runtime is not None
        cmd = SessionGetCommand.model_validate(params)
        session, latest_run = await self._runtime.get_session_with_latest_run(
            cmd.session_id
        )
        if self._event_hub is not None:
            await self._event_hub.flush()
        latest_cursor = await self._runtime.latest_event_cursor(session.id)
        return SessionGetResult(
            session=self._session_info(session),
            latest_run=self._run_info(latest_run) if latest_run is not None else None,
            latest_cursor=latest_cursor,
        )

    # 创建持久化 Turn 与 queued Run 后立即返回，不等待 LLM 执行完成
    async def _session_send_handler(self, params: dict[str, Any]) -> SessionSendMessageResult:
        assert self._runtime is not None
        cmd = SessionSendMessageCommand.model_validate(params)
        submitted = await self._runtime.submit_message(
            cmd.session_id,
            cmd.content,
            client_message_id=cmd.client_message_id,
        )
        return SessionSendMessageResult(
            run_id=submitted.run_id,
            status=submitted.status,  # type: ignore[arg-type]
            deduplicated=submitted.deduplicated,
        )

    # 返回 session 的完整 Anthropic messages 历史
    async def _session_history_handler(self, params: dict[str, Any]) -> SessionGetHistoryResult:
        assert self._runtime is not None
        cmd = SessionGetHistoryCommand.model_validate(params)
        messages = await self._runtime.get_history(
            cmd.session_id,
            after_sequence=cmd.after_sequence,
            limit=cmd.limit,
        )
        return SessionGetHistoryResult(messages=messages)

    # 接收客户端权限审批响应，resolve 对应挂起的 Future
    async def _permission_respond_handler(self, params: dict[str, Any]) -> PermissionRespondResult:
        cmd = PermissionRespondCommand.model_validate(params)
        logger.info(
            "permission.respond received request_id=%s decision=%s",
            cmd.request_id,
            cmd.decision,
        )
        if self._permission_manager is None:
            logger.error("permission.respond: PermissionManager not initialized")
            return PermissionRespondResult()
        return PermissionRespondResult(
            ok=self._permission_manager.respond(
                cmd.request_id,
                cmd.session_id,
                cmd.decision,
            )
        )

    async def _mcp_status_handler(self, params: dict[str, Any]) -> McpStatusResult:
        McpStatusCommand.model_validate(params)
        statuses = self._mcp_manager.statuses() if self._mcp_manager is not None else []
        return McpStatusResult(
            servers=[
                McpServerInfo(
                    name=status.name,
                    transport=status.transport,
                    status=status.status,
                    tool_count=status.tool_count,
                    protocol_version=status.protocol_version,
                    error=status.error,
                )
                for status in statuses
            ]
        )

    # 压缩 Session 活动上下文并事务性提交摘要，失败不修改原上下文
    async def _session_compact_handler(self, params: dict[str, Any]) -> SessionCompactResult:
        assert self._runtime is not None
        cmd = SessionCompactCommand.model_validate(params)
        result = await self._runtime.compact_session(cmd.session_id, focus=cmd.focus)
        return SessionCompactResult(
            summary_tokens=result.summary_tokens,
            saved_tokens=max(
                0,
                result.original_token_estimate - result.summary_tokens,
            ),
        )

    # 关闭 session 并返回 closed 状态
    async def _session_close_handler(self, params: dict[str, Any]) -> SessionCloseResult:
        assert self._runtime is not None
        cmd = SessionCloseCommand.model_validate(params)
        await self._runtime.close_session(cmd.session_id)
        return SessionCloseResult(status="closed")

    # 为失败或中断 Run 创建新的显式 attempt，不覆盖原记录
    async def _session_retry_handler(self, params: dict[str, Any]) -> SessionRetryResult:
        assert self._runtime is not None
        cmd = SessionRetryCommand.model_validate(params)
        submitted = await self._runtime.retry_run(
            cmd.run_id,
            confirm_side_effects=cmd.confirm_side_effects,
        )
        return SessionRetryResult(
            run_id=submitted.run_id,
            status=submitted.status,  # type: ignore[arg-type]
        )

    # 查询一个持久化 Run 的状态、attempt、失败原因与副作用标记
    async def _run_get_handler(self, params: dict[str, Any]) -> RunInfo:
        assert self._runtime is not None
        cmd = RunGetCommand.model_validate(params)
        return self._run_info(await self._runtime.get_run(cmd.run_id))

    async def _run_metrics_handler(self, params: dict[str, Any]) -> dict[str, Any]:
        assert self._database is not None
        command = RunMetricsCommand.model_validate(params)
        metrics = await RunMetricsProjection(self._database).project(command.run_id)
        if metrics is None:
            raise HandlerError(-32030, "run not found")
        return dict(TypeAdapter(type(metrics)).dump_python(metrics, mode="json"))

    # 取消受 RunSupervisor 管理的 Run 并等待状态确定性落库
    async def _run_cancel_handler(self, params: dict[str, Any]) -> RunInfo:
        assert self._runtime is not None
        cmd = RunCancelCommand.model_validate(params)
        return self._run_info(await self._runtime.cancel_run(cmd.run_id))

    # 将内部 Run 快照映射为 Wire Protocol V2 响应模型
    @staticmethod
    def _run_info(snapshot: RunSnapshot) -> RunInfo:
        return RunInfo(
            run_id=snapshot.id,
            session_id=snapshot.session_id,
            turn_id=snapshot.turn_id,
            parent_run_id=snapshot.parent_run_id,
            retry_of_run_id=snapshot.retry_of_run_id,
            kind=snapshot.kind,
            attempt=snapshot.attempt,
            status=snapshot.status,  # type: ignore[arg-type]
            reason=snapshot.reason,
            side_effects_started=snapshot.side_effects_started,
            result=snapshot.result,
            created_at=snapshot.created_at,
            started_at=snapshot.started_at,
            finished_at=snapshot.finished_at,
        )

    # 将 Runtime Session 快照映射为稳定的 Wire Protocol 响应模型
    @staticmethod
    def _session_info(snapshot: SessionSnapshot) -> SessionInfo:
        return SessionInfo(
            session_id=snapshot.id,
            mode=snapshot.mode,
            status=snapshot.status,  # type: ignore[arg-type]
            title=snapshot.title,
            workspace_root=snapshot.workspace_root,
            active_run_id=snapshot.active_run_id,
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
            closed_at=snapshot.closed_at,
        )

    # 从 SQLite cursor 建立回放与实时无缝衔接的事件订阅
    async def _subscribe_handler(self, params: dict[str, Any]) -> EventSubscribeResult:
        cmd = EventSubscribeCommand.model_validate(params)
        if cmd.session_id is not None and cmd.run_id is not None:
            raise HandlerError(-32602, "session_id and run_id are mutually exclusive")
        sender = get_connection_sender()
        assert self._broadcaster is not None
        subscription_id, replayed_count, high_water, truncated = (
            await self._broadcaster.subscribe(
                sender,
                cmd.topics,
                session_id=cmd.session_id,
                run_id=cmd.run_id,
                after_cursor=cmd.after_cursor,
            )
        )
        return EventSubscribeResult(
            subscription_id=subscription_id,
            replayed_count=replayed_count,
            high_water_cursor=high_water,
            replay_truncated=truncated,
        )

    async def _cleanup_resources(
        self, resources: list[tuple[str, Callable[[], Awaitable[None]]]],
    ) -> None:
        failures: list[str] = []
        for name, close in reversed(resources):
            try:
                await close()
            except Exception:
                failures.append(name)
                logger.exception("daemon cleanup failed resource=%s", name)
        if failures:
            raise RuntimeError("daemon cleanup failed: " + ", ".join(failures))

    async def run(self) -> None:
        self._start_time = time.monotonic()
        self._config = get_config()
        config = self._config
        setup_logging(config)
        home = tars_home()
        resources: list[tuple[str, Callable[[], Awaitable[None]]]] = []
        installed_signals: list[signal.Signals] = []
        loop = asyncio.get_running_loop()
        control_path = control_file_for(config.port)
        try:
            if config.trace.enabled:
                self._trace = TraceWriter(Path(config.trace.file).expanduser())
                resources.append(("trace", self._trace.stop))
                await self._trace.start()
                self._bus.subscribe(self._trace_event_handler)

            policy_file = home / "policy.toml"
            self._permission_manager = PermissionManager(
                policy_file=policy_file, timeout_s=config.permission.timeout_s,
            )
            logger.info("permission manager: timeout_s=%.1f persistent=%d entries",
                        config.permission.timeout_s, len(load_policy_file(policy_file)))
            bootstrap = await bootstrap_state(home / "state.db", backup_dir=home / "backups")
            self._database = bootstrap.database
            resources.append(("database", self._database.dispose))
            self._event_hub = DurableEventHub(self._database)
            resources.append(("events", self._event_hub.stop))
            await self._event_hub.start()
            self._bus.subscribe(self._event_hub.handle)
            self._broadcaster = IpcEventBroadcaster(self._event_hub, trace=self._trace)
            resources.append(("broadcaster", self._broadcaster.stop))
            self._mcp_manager = McpServerManager()
            resources.append(("mcp", self._mcp_manager.stop_all))
            if config.mcp.servers:
                await self._mcp_manager.start_all(config.mcp.servers)
            tool_runtime = await initialize_runtime_router(config.sandbox)
            self._tool_runtime = tool_runtime
            resources.append(("tool runtime", self._tool_runtime.cleanup))
            self._subagent_registry = BackgroundTaskRegistry(self._database, self._bus)
            resources.append(("subagents", self._subagent_registry.shutdown))
            self._runtime = RuntimeService(
                self._database,
                runner_factory=lambda: AgentRunner(
                    config, bus=self._bus, trace=self._trace,
                    permission_manager=self._permission_manager,
                    mcp_manager=self._mcp_manager, tool_runtime=tool_runtime,
                    task_registry=self._subagent_registry,
                ),
                bus=self._bus, artifacts_root=home / "artifacts",
                subagent_registry=self._subagent_registry,
                tool_runtime=self._tool_runtime,
                compaction_provider_factory=lambda: AnthropicProvider.from_config(config.llm),
            )
            resources.append(("runtime", self._runtime.shutdown))
            recovered = await self._runtime.recover_interrupted()
            if recovered:
                logger.warning("marked %d unfinished run(s) as interrupted", recovered)
            self._shutdown_event = asyncio.Event()
            shutdown_event = self._shutdown_event
            server = SocketServer(config.host, config.port, self._broadcaster, trace=self._trace)
            resources.append(("server", server.stop))
            server.register("core.ping", self._ping_handler)
            server.register("core.shutdown", self._shutdown_handler,
                            after_response=shutdown_event.set)
            server.register("agent.run", self._agent_run_handler)
            server.register("event.subscribe", self._subscribe_handler)
            server.register("session.create", self._session_create_handler)
            server.register("session.list", self._session_list_handler)
            server.register("session.resume", self._session_resume_handler)
            server.register("session.get", self._session_get_handler)
            server.register("session.send_message", self._session_send_handler)
            server.register("session.get_history", self._session_history_handler)
            server.register("session.close", self._session_close_handler)
            server.register("session.retry", self._session_retry_handler)
            server.register("run.get", self._run_get_handler)
            server.register("run.cancel", self._run_cancel_handler)
            server.register("run.metrics", self._run_metrics_handler)
            server.register("permission.respond", self._permission_respond_handler)
            server.register("mcp.status", self._mcp_status_handler)
            server.register("session.compact", self._session_compact_handler)
            addr = await server.start()
            write_control_file(DaemonControl(
                pid=os.getpid(), host=config.host, port=config.port,
                token=self._control_token, launch_id=self._launch_id,
            ), control_path)
            logger.info("tars-core %s listening addr=%s", tars_agent.__version__, addr)
            logger.info("config: %s", _config_log_summary(config))
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, shutdown_event.set)
                except (NotImplementedError, RuntimeError):
                    logger.debug("async signal handler unavailable for %s", sig.name)
                else:
                    installed_signals.append(sig)
            await shutdown_event.wait()
        finally:
            for sig in installed_signals:
                loop.remove_signal_handler(sig)
            try:
                await finish_cleanup(
                    asyncio.create_task(self._cleanup_resources(resources)),
                    failure_message="daemon cleanup failed during cancellation",
                )
            finally:
                remove_control_file(self._control_token, control_path)
                self._database = None
                self._runtime = None
                self._event_hub = None
                self._broadcaster = None
                self._tool_runtime = None
                self._subagent_registry = None
                self._shutdown_event = None


def run() -> None:
    parser = argparse.ArgumentParser(prog="tars-core", description="TARS-Agent core daemon")
    parser.add_argument("--version", action="version", version=tars_agent.__version__)
    parser.parse_args()
    asyncio.run(CoreApp().run())
