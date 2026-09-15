from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from tars_agent.core.tools.runtime.host import HostRuntime
from tars_agent.core.tools.runtime.models import (
    RuntimeStatus,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from tars_agent.core.tools.runtime.protocol import ToolRuntime

_FALLBACKABLE = frozenset(
    {
        "sandbox_disabled",
        "sandbox_unavailable",
        "sandbox_image_missing",
        "sandbox_start_failed",
    }
)


@dataclass(frozen=True, slots=True)
class HostFallbackRequest:
    tool_name: str
    params: dict[str, object]
    reason: str
    parameter_digest: str
    platform_shell: str
    warning: str = (
        "Docker isolation is unavailable. This one invocation will run directly "
        "on the host with host filesystem and network visibility."
    )


HostFallbackAuthorizer = Callable[[HostFallbackRequest], Awaitable[bool]]


class RuntimeRouter:
    def __init__(
        self,
        sandbox: ToolRuntime,
        host: HostRuntime | None = None,
        *,
        allow_host_fallback: bool = True,
    ) -> None:
        self._sandbox = sandbox
        output_limit = getattr(sandbox, "output_limit_bytes", None)
        self._host = host or HostRuntime(output_limit_bytes=output_limit)
        self._allow_host_fallback = allow_host_fallback

    async def preflight(self) -> RuntimeStatus:
        return await self._sandbox.preflight()

    async def execute(
        self,
        request: ToolExecutionRequest,
        *,
        authorize_host_fallback: HostFallbackAuthorizer | None = None,
    ) -> ToolExecutionResult:
        sandbox_result = await self._sandbox.execute(request)
        if (
            not sandbox_result.is_error
            or sandbox_result.started
            or sandbox_result.error_type not in _FALLBACKABLE
        ):
            return sandbox_result
        if not self._allow_host_fallback:
            return sandbox_result
        if authorize_host_fallback is None:
            return sandbox_result
        digest = _parameter_digest(request.params)
        approved = await authorize_host_fallback(
            HostFallbackRequest(
                tool_name=request.tool_name,
                params=request.params,
                reason=sandbox_result.error_type or "sandbox_unavailable",
                parameter_digest=digest,
                platform_shell=_host_executor(request.tool_name),
            )
        )
        if not approved:
            return ToolExecutionResult(
                content="Host fallback was denied or expired; no host process was started.",
                backend="workspace_sandbox",
                started=False,
                is_error=True,
                error_type="host_fallback_denied",
                retryable=False,
            )
        if not hmac.compare_digest(digest, _parameter_digest(request.params)):
            return ToolExecutionResult(
                content=(
                    "Host fallback parameters changed after approval; execution was blocked."
                ),
                backend="workspace_sandbox",
                started=False,
                is_error=True,
                error_type="sandbox_policy_denied",
                retryable=False,
            )
        return await self._host.execute(request)

    async def cancel(self, invocation_id: str) -> None:
        try:
            await self._sandbox.cancel(invocation_id)
        finally:
            await self._host.cancel(invocation_id)

    async def cleanup_run(self, run_id: str) -> None:
        try:
            await self._sandbox.cleanup_run(run_id)
        finally:
            await self._host.cleanup_run(run_id)

    async def cleanup(self) -> None:
        try:
            await self._sandbox.cleanup()
        finally:
            await self._host.cleanup()

    def pending_cleanup_run_ids(self) -> tuple[str, ...]:
        pending = getattr(self._sandbox, "pending_cleanup_run_ids", None)
        if not callable(pending):
            return ()
        return tuple(run_id for run_id in pending() if isinstance(run_id, str))

    @property
    def sandbox(self) -> ToolRuntime:
        return self._sandbox


def _parameter_digest(params: dict[str, object]) -> str:
    canonical = json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _host_executor(tool_name: str) -> str:
    if tool_name != "bash":
        return "Python workspace worker (no shell)"
    if os.name == "nt":
        return os.environ.get("COMSPEC", "cmd.exe")
    return "/bin/sh"


__all__ = ["HostFallbackAuthorizer", "HostFallbackRequest", "RuntimeRouter"]
