from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from tars_agent.core.tools.base import BaseTool, ToolResult
from tars_agent.core.tools.runtime.models import ToolCallContext, ToolExecutionRequest
from tars_agent.core.tools.runtime.router import RuntimeRouter


class WorkspaceRuntimeTool(BaseTool):
    execution_profile = "workspace_sandbox"
    params_model: ClassVar[type[BaseModel]]

    def __init__(self, runtime: RuntimeRouter) -> None:
        self._runtime = runtime

    async def invoke(
        self,
        params: dict[str, object],
        *,
        context: ToolCallContext | None = None,
    ) -> ToolResult:
        if context is None:
            return ToolResult(
                content="workspace tool invocation is missing ToolCallContext",
                is_error=True,
                error_type="sandbox_policy_denied",
                retryable=False,
            )
        timeout_value = params.get("timeout", 120.0)
        timeout_s = float(timeout_value) if isinstance(timeout_value, (int, float)) else 120.0
        request = ToolExecutionRequest(
            invocation_id=f"{context.run_id}:{context.tool_use_id}",
            run_id=context.run_id,
            session_id=context.session_id,
            tool_use_id=context.tool_use_id,
            tool_name=self.name,
            params=params,
            workspace_root=context.workspace_root,
            timeout_s=timeout_s,
            network_mode=context.network_mode,
            on_started=context.on_execution_started,
        )
        result = await self._runtime.execute(
            request,
            authorize_host_fallback=context.authorize_host_fallback,
        )
        return ToolResult(
            content=result.content,
            is_error=result.is_error,
            error_type=result.error_type,
            retryable=result.retryable,
            backend=result.backend,
            started=result.started,
            container_id=result.container_id,
        )


__all__ = ["WorkspaceRuntimeTool"]
