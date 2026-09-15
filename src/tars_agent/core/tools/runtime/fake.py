from __future__ import annotations

from tars_agent.core.tools.runtime.models import (
    RuntimeStatus,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from tars_agent.sandbox.worker import execute_payload


class FakeRuntime:
    """Deterministic test runtime; never selected by production configuration."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.requests: list[ToolExecutionRequest] = []
        self.cleaned_runs: list[str] = []
        self.cancelled: list[str] = []

    async def preflight(self) -> RuntimeStatus:
        return RuntimeStatus(
            available=self.available,
            backend="workspace_sandbox",
            reason=None if self.available else "sandbox_unavailable",
        )

    async def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        if not self.available:
            return ToolExecutionResult(
                content="fake sandbox unavailable",
                backend="workspace_sandbox",
                started=False,
                is_error=True,
                error_type="sandbox_unavailable",
            )
        self.requests.append(request)
        if request.on_started is not None:
            await request.on_started("workspace_sandbox", "fake-container")
        raw = await execute_payload(
            request.worker_payload(),
            request.workspace_root,
            sandboxed=False,
        )
        exit_code_value = raw.get("exit_code")
        elapsed_value = raw.get("elapsed_ms")
        return ToolExecutionResult(
            content=str(raw.get("content", "")),
            backend="workspace_sandbox",
            started=True,
            is_error=bool(raw.get("is_error", False)),
            error_type=(
                str(raw["error_type"])
                if raw.get("error_type") is not None
                else None
            ),
            retryable=False,
            exit_code=exit_code_value if isinstance(exit_code_value, int) else None,
            stdout=str(raw.get("stdout", "")),
            stderr=str(raw.get("stderr", "")),
            truncated=bool(raw.get("truncated", False)),
            elapsed_ms=elapsed_value if isinstance(elapsed_value, int) else 0,
            container_id="fake-container",
        )

    async def cancel(self, invocation_id: str) -> None:
        self.cancelled.append(invocation_id)

    async def cleanup_run(self, run_id: str) -> None:
        self.cleaned_runs.append(run_id)

    async def cleanup(self) -> None:
        return None


__all__ = ["FakeRuntime"]
