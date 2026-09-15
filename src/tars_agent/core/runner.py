from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tars_agent.core.artifacts import ArtifactStore
from tars_agent.core.bus.events import RunStartedEvent
from tars_agent.core.compact.compactor import CompactionResult, Compactor
from tars_agent.core.config import TarsConfig
from tars_agent.core.context import ExecutionContext
from tars_agent.core.events.bus import EventBus
from tars_agent.core.execution import cleanup_run, execute_loop, mark_cancelled
from tars_agent.core.llm.base import LLMProvider
from tars_agent.core.llm.provider import AnthropicProvider
from tars_agent.core.loop import AgentLoop
from tars_agent.core.mcp.server import McpServerManager
from tars_agent.core.memory.loader import load_context_file
from tars_agent.core.paths import tars_home
from tars_agent.core.permissions.manager import PermissionManager
from tars_agent.core.subagent.registry import BackgroundTaskRegistry
from tars_agent.core.subagent.tool import AgentCancelTool, AgentResultTool, SpawnAgentTool
from tars_agent.core.task.manager import TaskManager
from tars_agent.core.tools.assembly import build_base_registry
from tars_agent.core.tools.builtin import NoteSaveTool
from tars_agent.core.tools.registry import ToolRegistry
from tars_agent.core.tools.runtime import RuntimeRouter
from tars_agent.core.trace.provider import TracingProvider
from tars_agent.core.trace.writer import TraceWriter

log = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    status: str
    result: str
    reason: str | None
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    active_context: list[dict[str, Any]] | None = None
    compactions: list[CompactionResult] = field(default_factory=list)


class AgentRunner:
    """Prepare one execution and return its result; RuntimeService owns persistence."""

    def __init__(
        self,
        config: TarsConfig,
        *,
        bus: EventBus,
        tool_runtime: RuntimeRouter,
        provider: LLMProvider | None = None,
        trace: TraceWriter | None = None,
        permission_manager: PermissionManager | None = None,
        mcp_manager: McpServerManager | None = None,
        task_registry: BackgroundTaskRegistry | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._tool_runtime = tool_runtime
        self._provider = provider
        self._trace = trace
        self._permission_manager = permission_manager
        self._mcp_manager = mcp_manager
        self._task_registry = task_registry

    def _build_registry(
        self,
        *,
        run_id: str,
        session_id: str,
        workspace_root: Path,
        artifact_store: ArtifactStore,
        provider: LLMProvider,
        tool_whitelist: list[str] | None,
    ) -> ToolRegistry:
        allowed = set(tool_whitelist) if tool_whitelist is not None else None
        task_manager = TaskManager(artifact_store.run_dir(session_id, run_id) / ".tasks")
        registry = build_base_registry(
            self._tool_runtime, task_manager, allowed_tools=allowed,
        )

        def permits(name: str) -> bool:
            return allowed is None or name in allowed

        if permits("note_save"):
            registry.register(NoteSaveTool(artifact_store, session_id, run_id))
        if self._task_registry is not None:
            if permits("spawn_agent"):
                registry.register(SpawnAgentTool(
                    provider=provider,
                    parent_bus=self._bus,
                    parent_run_id=run_id,
                    permission_manager=self._permission_manager,
                    max_steps=self._config.agent.max_steps,
                    task_registry=self._task_registry,
                    runs_dir=artifact_store.runs_dir(session_id),
                    session_id=session_id,
                    tool_runtime=self._tool_runtime,
                    workspace_root=workspace_root,
                    parent_allowed_tools=allowed,
                    depth=0,
                ))
            if permits("agent_result"):
                registry.register(AgentResultTool(self._task_registry, session_id=session_id))
            if permits("agent_cancel"):
                registry.register(AgentCancelTool(self._task_registry, session_id=session_id))
        if self._mcp_manager is not None:
            for tool in self._mcp_manager.get_tools():
                if permits(tool.name):
                    registry.register(tool)
        return registry

    async def _release_provider(self, run_id: str, provider: LLMProvider) -> bool:
        close = getattr(provider, "close", None)
        if not callable(close):
            return False
        release = (
            self._task_registry.release_when_idle(run_id, close)
            if self._task_registry is not None else close()
        )
        operation = asyncio.create_task(release)
        cancelled = False
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                cancelled = True
        operation.result()
        return cancelled

    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str,
        session_id: str,
        workspace_root: Path,
        history: list[dict[str, Any]],
        artifact_store: ArtifactStore,
        session_notes: str,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
    ) -> RunOutcome:
        root = workspace_root.expanduser().resolve(strict=True)
        context = ExecutionContext(
            run_id=run_id,
            goal=goal,
            max_steps=self._config.agent.max_steps,
            prefill_messages=history,
            session_notes=session_notes,
            global_context=load_context_file(tars_home() / "context.md"),
            project_context=load_context_file(root / ".tars/context.md"),
            system_prompt_override=system_prompt_override,
        )
        owned_provider: LLMProvider | None = None
        try:
            await self._bus.publish(RunStartedEvent(
                run_id=run_id, goal=goal, ts=datetime.now(UTC).isoformat(),
            ))
            try:
                provider = self._provider or AnthropicProvider.from_config(self._config.llm)
            except SystemExit as exc:
                raise RuntimeError(str(exc)) from exc
            if self._provider is None:
                owned_provider = provider
            if self._trace is not None:
                provider = TracingProvider(
                    provider, self._trace,
                    include_payload=self._config.trace.include_llm_payload,
                )
            registry = self._build_registry(
                run_id=run_id,
                session_id=session_id,
                workspace_root=root,
                artifact_store=artifact_store,
                provider=provider,
                tool_whitelist=tool_whitelist,
            )
            loop = AgentLoop(
                provider, registry, self._bus,
                permission_manager=self._permission_manager,
                compactor=Compactor(
                    self._bus, artifact_store.session_dir(session_id), session_id,
                ),
                compact_threshold=self._config.compaction.auto_threshold,
                session_id=session_id,
                workspace_root=root,
                defer_compaction_publish=True,
            )
            await execute_loop(loop, context)
        except asyncio.CancelledError:
            mark_cancelled(context)
        except Exception:
            log.exception("agent preparation failed run_id=%s", run_id)
            context.mark_failed("llm_error")
        finally:
            cleanup_error: BaseException | None = None
            try:
                if await cleanup_run(self._tool_runtime, run_id):
                    mark_cancelled(context)
            except BaseException as exc:
                cleanup_error = exc
            try:
                if owned_provider is not None:
                    if await self._release_provider(run_id, owned_provider):
                        mark_cancelled(context)
            except BaseException:
                if cleanup_error is None:
                    raise
                log.exception("provider release failed after tool cleanup error run_id=%s", run_id)
            if cleanup_error is not None:
                raise cleanup_error
        return RunOutcome(
            status=context.status,
            result=context.result,
            reason=context.reason,
            messages=context.audit_messages,
            steps=context.step,
            active_context=(list(context.messages) if context.compactions else None),
            compactions=list(context.compactions),
        )
