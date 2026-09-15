from __future__ import annotations

import asyncio
from collections.abc import Iterable

from tars_agent.core.config import SandboxConfig
from tars_agent.core.tools.runtime.models import (
    RuntimeStatus,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from tars_agent.sandbox.worker import execute_payload


class HostRuntime:
    """Explicit, approval-gated host fallback for workspace tools."""

    def __init__(self, output_limit_bytes: int | None = None) -> None:
        self._output_limit_bytes = (
            output_limit_bytes
            if output_limit_bytes is not None
            else SandboxConfig().output_limit_bytes
        )
        self._active: dict[str, tuple[asyncio.Task[object], str]] = {}

    async def preflight(self) -> RuntimeStatus:
        return RuntimeStatus(available=True, backend="host")

    async def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("host tool execution requires an asyncio task")
        self._active[request.invocation_id] = (task, request.run_id)
        try:
            if request.on_started is not None:
                await request.on_started("host", None)
            result = await execute_payload(
                request.worker_payload(output_limit_bytes=self._output_limit_bytes),
                request.workspace_root,
                sandboxed=False,
            )
        finally:
            self._active.pop(request.invocation_id, None)
        return ToolExecutionResult(
            content=str(result.get("content", "")),
            backend="host",
            started=True,
            is_error=bool(result.get("is_error", False)),
            error_type=_optional_string(result.get("error_type")),
            retryable=bool(result.get("retryable", False)),
            exit_code=_optional_int(result.get("exit_code")),
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            truncated=bool(result.get("truncated", False)),
            elapsed_ms=_int_or_zero(result.get("elapsed_ms")),
        )

    async def cancel(self, invocation_id: str) -> None:
        active = self._active.get(invocation_id)
        if active is None:
            return
        task, _ = active
        if task is asyncio.current_task():
            return
        if not task.cancelling():
            task.cancel()
        await asyncio.shield(asyncio.gather(task, return_exceptions=True))

    async def cleanup_run(self, run_id: str) -> None:
        await self._cancel_matching(
            invocation_id
            for invocation_id, (_, active_run_id) in tuple(self._active.items())
            if active_run_id == run_id
        )

    async def cleanup(self) -> None:
        await self._cancel_matching(tuple(self._active))

    async def _cancel_matching(self, invocation_ids: Iterable[str]) -> None:
        for invocation_id in invocation_ids:
            await self.cancel(invocation_id)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) else 0


__all__ = ["HostRuntime"]
