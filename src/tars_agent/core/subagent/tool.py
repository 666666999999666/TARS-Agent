from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from tars_agent.core.agents.loader import AgentProfile, AgentProfileLoader
from tars_agent.core.bus.events import SubagentStartedEvent
from tars_agent.core.compact.budget import ContextBudget
from tars_agent.core.config import SandboxConfig
from tars_agent.core.context import ExecutionContext
from tars_agent.core.events.bus import EventBus
from tars_agent.core.execution import cleanup_run, execute_loop
from tars_agent.core.loop import AgentLoop
from tars_agent.core.processes import finish_cleanup
from tars_agent.core.runs import new_run_id
from tars_agent.core.subagent.registry import BackgroundTaskRegistry
from tars_agent.core.tools.assembly import build_base_registry
from tars_agent.core.tools.base import BaseTool, ToolResult
from tars_agent.core.tools.registry import ToolRegistry
from tars_agent.core.tools.runtime import (
    RuntimeCleanupPending,
    RuntimeRouter,
    build_runtime_router,
)

if TYPE_CHECKING:
    from tars_agent.core.llm.base import LLMProvider
    from tars_agent.core.permissions.manager import PermissionManager

_profile_loader = AgentProfileLoader()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SpawnAgentParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    description: str
    prompt: str
    run_in_background: bool = False
    subagent_type: str = ""


# 在隔离的冷启动上下文中派生子 agent，支持前台阻塞和后台并行两种模式
class SpawnAgentTool(BaseTool):
    name = "spawn_agent"
    description = (
        "Spawn an isolated sub-agent to handle a self-contained sub-task. "
        "The sub-agent starts with a clean context containing only the provided prompt — "
        "it does not inherit the current conversation history. "
        "Use run_in_background=true to run in parallel; retrieve result later with agent_result."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "3-5 word task description shown in progress display",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Complete task description including all context the sub-agent needs. "
                    "The sub-agent cannot see the parent conversation, so be explicit."
                ),
            },
            "run_in_background": {
                "type": "boolean",
                "description": "When true, returns immediately with a run_id; use agent_result to poll.",  # noqa: E501
            },
            "subagent_type": {
                "type": "string",
                "description": "Agent role profile (planner/executor/reviewer). Leave empty for default.",  # noqa: E501
            },
        },
        "required": ["description", "prompt"],
    }
    params_model = SpawnAgentParams

    # 构造 SpawnAgentTool；depth=0 表示根 agent，最大允许嵌套深度为 2
    def __init__(
        self,
        provider: LLMProvider,
        parent_bus: EventBus,
        parent_run_id: str,
        permission_manager: PermissionManager | None,
        max_steps: int,
        task_registry: BackgroundTaskRegistry,
        runs_dir: Path,
        session_id: str,
        tool_runtime: RuntimeRouter | None = None,
        workspace_root: Path | None = None,
        parent_allowed_tools: set[str] | None = None,
        depth: int = 0,
        context_budget: ContextBudget | None = None,
    ) -> None:
        self._provider = provider
        self._parent_bus = parent_bus
        self._parent_run_id = parent_run_id
        self._permission_manager = permission_manager
        self._max_steps = max_steps
        self._task_registry = task_registry
        self._runs_dir = runs_dir
        self._session_id = session_id
        self._tool_runtime = tool_runtime or build_runtime_router(SandboxConfig())
        self._runtime_preflight_required = tool_runtime is None
        self._parent_allowed_tools = parent_allowed_tools
        self._workspace_root = (workspace_root or Path.cwd()).expanduser().resolve(strict=True)
        self._depth = depth
        self._context_budget = context_budget or ContextBudget()

    # 派生子 agent，前台时阻塞直到完成并返回结果，后台时立即返回 run_id
    async def invoke(
        self,
        params: dict[str, object],
        *,
        context: object | None = None,
    ) -> ToolResult:
        p = SpawnAgentParams.model_validate(params)

        if self._depth >= 2:
            return ToolResult(
                content="Subagent nesting limit (2) reached; cannot spawn further subagents.",
                is_error=True,
                error_type="runtime_error",
            )

        try:
            self._task_registry.assert_can_spawn(self._parent_run_id)
            if self._runtime_preflight_required:
                status = await self._tool_runtime.preflight()
                if not status.available:
                    raise RuntimeError(f"sandbox.mode=required preflight failed: {status.reason}")
                self._runtime_preflight_required = False
            profile: AgentProfile | None = None
            if p.subagent_type:
                profile = _profile_loader.load(p.subagent_type, workspace_root=self._workspace_root)
                if profile is None:
                    raise ValueError(f"Unknown subagent profile: {p.subagent_type}")
            child_provider = self._provider
            if profile is not None and profile.model:
                with_model = getattr(self._provider, "with_model", None)
                if not callable(with_model):
                    raise ValueError("provider does not support profile model override")
                child_provider = with_model(profile.model)
        except (ValueError, RuntimeError) as exc:
            return ToolResult(content=str(exc), is_error=True, error_type="runtime_error")

        child_run_id = new_run_id()
        child_context = ExecutionContext(
            run_id=child_run_id,
            goal=p.prompt,
            max_steps=self._max_steps,
            system_prompt_override=profile.system_prompt if profile else None,
        )

        child_bus = EventBus()

        # 将子 bus 所有事件桥接到父 bus，TUI 据此渲染嵌套进度
        async def _bridge(event: BaseModel) -> None:
            await self._parent_bus.publish(event)

        child_bus.subscribe(_bridge)

        child_registry = self._build_child_registry(
            child_bus, child_run_id, profile, provider=child_provider
        )
        child_loop = AgentLoop(
            child_provider,
            child_registry,
            child_bus,
            permission_manager=self._permission_manager,
            session_id=self._session_id,
            workspace_root=self._workspace_root,
            context_budget=self._context_budget,
        )

        task: asyncio.Task[None] | None = None
        created = False
        try:
            await self._task_registry.create_run(
                run_id=child_run_id,
                session_id=self._session_id,
                parent_run_id=self._parent_run_id,
                description=p.description,
                prompt=p.prompt,
                depth=self._depth + 1,
                background=p.run_in_background,
            )
            created = True
            self._task_registry.assert_can_spawn(self._parent_run_id)
            child_run_path = self._runs_dir / child_run_id
            await self._task_registry.start_recording(
                child_run_id, child_run_path / "events.jsonl",
            )
            await self._parent_bus.publish(
                SubagentStartedEvent(
                    run_id=child_run_id,
                    parent_run_id=self._parent_run_id,
                    description=p.description,
                    ts=_now(),
                )
            )
            self._task_registry.assert_can_spawn(self._parent_run_id)
            task = asyncio.create_task(
                self._run_child(
                    child_loop, child_context, child_run_id,
                ),
                name=f"tars-subagent:{child_run_id}",
            )
            self._task_registry.register(
                child_run_id, task, child_context,
                session_id=self._session_id, parent_run_id=self._parent_run_id,
            )
            if p.run_in_background:
                return ToolResult(
                    content=(
                        f"Subagent started in background. run_id={child_run_id}. "
                        f"Use agent_result(run_id='{child_run_id}') to retrieve result."
                    )
                )
            await asyncio.shield(task)
        except BaseException as exc:
            if task is not None and not task.done():
                if not task.cancelling():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if not created and await self._task_registry.get_snapshot(child_run_id) is None:
                raise
            if self._task_registry.cleanup_is_pending(child_run_id):
                raise RuntimeCleanupPending(child_run_id) from exc
            await self._task_registry.finish(
                child_run_id,
                status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                reason="cancelled" if isinstance(exc, asyncio.CancelledError) else "spawn_failed",
            )
            raise

        snapshot = await self._task_registry.get_snapshot(child_run_id, session_id=self._session_id)
        assert snapshot is not None
        if snapshot.status == "succeeded":
            return ToolResult(
                content=snapshot.result or "Subagent completed with no text output."
            )
        return ToolResult(
            content=(
                snapshot.result
                or f"Subagent failed (status={snapshot.status}, reason={snapshot.reason})"
            ),
            is_error=True,
            error_type="runtime_error",
        )

    # 运行子循环；只有资源清理结束，registry 才提交终态并关闭事件文件。
    async def _run_child(
        self,
        loop: AgentLoop,
        context: ExecutionContext,
        run_id: str,
    ) -> None:
        try:
            await self._task_registry.mark_running(run_id)
            await execute_loop(loop, context)
        except asyncio.CancelledError:
            context.status = "cancelled"
            context.reason = "cancelled"
            raise
        except Exception:
            context.status = "failed"
            context.reason = "runtime_error"
            raise
        finally:
            await finish_cleanup(
                asyncio.create_task(self._finalize_child(context, run_id)),
                failure_message="subagent cleanup failed during cancellation",
            )

    async def _finalize_child(self, context: ExecutionContext, run_id: str) -> None:
        pending = False
        try:
            cancelled = await cleanup_run(self._tool_runtime, run_id)
            if cancelled:
                context.status = "cancelled"
                context.reason = "cancelled"
            self._task_registry.complete_cleanup(run_id)
        except RuntimeCleanupPending:
            pending = True
            self._task_registry.mark_cleanup_pending(run_id)
            raise
        except Exception:
            if context.status == "success":
                context.status = "failed"
                context.reason = "cleanup_failed"
            raise
        finally:
            if not pending:
                await self._task_registry.finish(
                    run_id,
                    status=context.status,
                    result=context.result,
                    reason=context.reason,
                )

    # 构造子 registry；基于角色配置过滤工具，深度允许时注册嵌套 SpawnAgentTool
    def _build_child_registry(
        self,
        child_bus: EventBus,
        child_run_id: str,
        profile: AgentProfile | None,
        *,
        provider: LLMProvider | None = None,
    ) -> ToolRegistry:
        from tars_agent.core.task.manager import TaskManager

        allowed: set[str] | None = (
            None if profile is not None and profile.allow_all_tools
            else set(profile.allowed_tools) if profile is not None else set()
        )
        if self._parent_allowed_tools is not None:
            allowed = (set(self._parent_allowed_tools) if allowed is None
                       else allowed & self._parent_allowed_tools)

        def _allowed(name: str) -> bool:
            return allowed is None or name in allowed

        child_task_manager = TaskManager(self._runs_dir / child_run_id / ".tasks")
        registry = build_base_registry(
            self._tool_runtime, child_task_manager, allowed_tools=allowed,
        )

        if self._depth < 1:
            nested = SpawnAgentTool(
                provider=provider or self._provider,
                parent_bus=child_bus,
                parent_run_id=child_run_id,
                permission_manager=self._permission_manager,
                max_steps=self._max_steps,
                task_registry=self._task_registry,
                runs_dir=self._runs_dir,
                session_id=self._session_id,
                tool_runtime=self._tool_runtime,
                workspace_root=self._workspace_root,
                parent_allowed_tools=allowed,
                depth=self._depth + 1,
                context_budget=self._context_budget,
            )
            if _allowed("spawn_agent"):
                registry.register(nested)
            if _allowed("agent_result"):
                registry.register(
                    AgentResultTool(self._task_registry, session_id=self._session_id)
                )
            if _allowed("agent_cancel"):
                registry.register(
                    AgentCancelTool(self._task_registry, session_id=self._session_id)
                )

        return registry


class AgentResultParams(BaseModel):
    run_id: str


# 查询后台 subagent 的执行状态和最终结果
class AgentResultTool(BaseTool):
    name = "agent_result"
    description = (
        "Retrieve the result of a background sub-agent previously started with spawn_agent. "
        "Returns 'still running' if the sub-agent has not yet completed."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run_id returned by spawn_agent(run_in_background=true)",
            },
        },
        "required": ["run_id"],
    }
    params_model = AgentResultParams

    # 初始化，持有共享的后台任务注册表
    def __init__(
        self,
        task_registry: BackgroundTaskRegistry,
        *,
        session_id: str | None = None,
    ) -> None:
        self._task_registry = task_registry
        self._session_id = session_id

    # 查询指定 run_id 的后台任务状态，返回结果或错误
    async def invoke(
        self,
        params: dict[str, object],
        *,
        context: object | None = None,
    ) -> ToolResult:
        p = AgentResultParams.model_validate(params)
        snapshot = await self._task_registry.get_snapshot(
            p.run_id,
            session_id=self._session_id,
        )
        if snapshot is None:
            return ToolResult(
                content=f"Unknown subagent run_id: {p.run_id}.",
                is_error=True,
                error_type="runtime_error",
            )
        if snapshot.status in {"queued", "running"}:
            return ToolResult(content="still running")
        if snapshot.status == "cancelled":
            return ToolResult(
                content="Subagent was cancelled.", is_error=True, error_type="runtime_error"
            )
        if snapshot.status != "succeeded":
            return ToolResult(
                content=(
                    snapshot.result
                    or f"Subagent failed (status={snapshot.status}, reason={snapshot.reason})"
                ),
                is_error=True,
                error_type="runtime_error",
            )
        return ToolResult(
            content=snapshot.result or "Subagent completed with no text result."
        )


class AgentCancelParams(BaseModel):
    run_id: str


class AgentCancelTool(BaseTool):
    name = "agent_cancel"
    description = "Cancel an active background sub-agent owned by the current Session."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"run_id": {"type": "string"}},
        "required": ["run_id"],
    }
    params_model = AgentCancelParams

    def __init__(
        self,
        task_registry: BackgroundTaskRegistry,
        *,
        session_id: str | None = None,
    ) -> None:
        self._task_registry = task_registry
        self._session_id = session_id

    async def invoke(
        self,
        params: dict[str, object],
        *,
        context: object | None = None,
    ) -> ToolResult:
        parsed = AgentCancelParams.model_validate(params)
        snapshot = await self._task_registry.get_snapshot(
            parsed.run_id,
            session_id=self._session_id,
        )
        if snapshot is None:
            return ToolResult(
                content=f"Unknown subagent run_id: {parsed.run_id}.",
                is_error=True,
                error_type="runtime_error",
            )
        if snapshot.status not in {"queued", "running"}:
            return ToolResult(
                content=f"Subagent is already {snapshot.status}.",
                is_error=True,
                error_type="runtime_error",
            )
        cancelled = await self._task_registry.cancel(
            parsed.run_id,
            session_id=self._session_id,
        )
        if not cancelled:
            return ToolResult(
                content="Subagent is no longer active.",
                is_error=True,
                error_type="runtime_error",
            )
        return ToolResult(content=f"Cancelled subagent {parsed.run_id}.")
