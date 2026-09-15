from __future__ import annotations

from typing import Protocol

from tars_agent.core.tools.runtime.models import (
    RuntimeStatus,
    ToolExecutionRequest,
    ToolExecutionResult,
)


class ToolRuntime(Protocol):
    async def preflight(self) -> RuntimeStatus: ...

    async def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult: ...

    async def cancel(self, invocation_id: str) -> None: ...

    async def cleanup_run(self, run_id: str) -> None: ...

    async def cleanup(self) -> None: ...


__all__ = ["ToolRuntime"]
