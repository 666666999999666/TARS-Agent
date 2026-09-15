from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from tars_agent.core.bus.events import (
    PermissionDeniedEvent,
    PermissionGrantedEvent,
    PermissionRequestedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
    ToolExecutionStartedEvent,
)
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import ToolCallBlock
from tars_agent.core.tools.base import ToolResult
from tars_agent.core.tools.errors import RateLimitedError
from tars_agent.core.tools.registry import ToolRegistry
from tars_agent.core.tools.runtime.models import ToolBackend, ToolCallContext
from tars_agent.core.tools.runtime.router import HostFallbackRequest

if TYPE_CHECKING:
    from tars_agent.core.permissions.manager import PermissionManager

_DEFAULT_TIMEOUT: float = 120.0
_MAX_RETRIES: int = 2
_RETRY_BASE_S: float = 2.0  # backoff base; tests can monkeypatch to 0


def _now() -> str:
    return datetime.now(UTC).isoformat()


# 发布 ToolCallFailedEvent 并返回对应 ToolResult
async def _fail(
    bus: EventBus,
    run_id: str,
    tool_call: ToolCallBlock,
    error_class: str,
    error_message: str,
    elapsed_ms: int,
    *,
    attempt: int = 1,
    retryable: bool = False,
) -> ToolResult:
    await bus.publish(
        ToolCallFailedEvent(
            run_id=run_id,
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            error_class=error_class,
            error_message=error_message,
            elapsed_ms=elapsed_ms,
            attempt=attempt,
            retryable=retryable,
            ts=_now(),
        )
    )
    return ToolResult(
        content=error_message,
        is_error=True,
        error_type=error_class,
        retryable=retryable,
    )


# 校验参数、检查权限、限时调用工具、发布进度事件，失败时指数退避重试，返回 ToolResult（不抛异常）
async def invoke_tool(
    registry: ToolRegistry,
    tool_call: ToolCallBlock,
    bus: EventBus,
    run_id: str,
    timeout: float = _DEFAULT_TIMEOUT,
    *,
    permission_manager: PermissionManager | None = None,
    session_id: str = "",
    workspace_root: Path | None = None,
) -> ToolResult:
    t0 = time.monotonic()

    await bus.publish(
        ToolCallStartedEvent(
            run_id=run_id,
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            params=dict(tool_call.input),
            ts=_now(),
        )
    )

    def elapsed() -> int:
        return int((time.monotonic() - t0) * 1000)

    tool = registry.get(tool_call.name)
    if tool is None:
        return await _fail(
            bus, run_id, tool_call,
            "runtime_error", f"unknown tool: {tool_call.name}", elapsed(),
        )

    if tool.params_model is not None:
        try:
            tool.params_model.model_validate(dict(tool_call.input))
        except ValidationError as exc:
            return await _fail(
                bus, run_id, tool_call,
                "schema_error", str(exc), elapsed(),
            )

    if permission_manager is not None:
        async def _emit_permission(raw: dict[str, Any]) -> None:
            await bus.publish(PermissionRequestedEvent(**raw, run_id=run_id))

        allowed, decision, request_id = await permission_manager.check_and_wait(
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            params=dict(tool_call.input),
            session_id=session_id,
            event_emitter=_emit_permission,
            workspace_root=workspace_root,
        )
        if allowed:
            if decision not in ("auto_allow",):
                await bus.publish(
                    PermissionGrantedEvent(
                        run_id=run_id,
                        request_id=request_id,
                        tool_use_id=tool_call.id,
                        decision=decision,
                        ts=_now(),
                    )
                )
        else:
            if decision != "auto_deny":
                await bus.publish(
                    PermissionDeniedEvent(
                        run_id=run_id,
                        request_id=request_id,
                        tool_use_id=tool_call.id,
                        decision=decision,
                        ts=_now(),
                    )
                )
            return await _fail(
                bus, run_id, tool_call,
                "permission_denied",
                "Permission was not granted (denied or expired). The tool was not executed. "
                "Use a permitted alternative or obtain fresh approval before retrying.",
                elapsed(),
            )

    for attempt in range(1, _MAX_RETRIES + 2):
        error_class: str | None = None
        error_message: str | None = None
        retryable = False

        try:
            started = False

            async def _execution_started(
                backend: ToolBackend,
                container_id: str | None,
            ) -> None:
                nonlocal started
                if started:
                    return
                started = True
                await bus.publish(
                    ToolExecutionStartedEvent(
                        run_id=run_id,
                        tool_use_id=tool_call.id,
                        tool_name=tool_call.name,
                        backend=backend,
                        container_id=container_id,
                        attempt=attempt,
                        ts=_now(),
                    )
                )

            async def _authorize_host_fallback(request: HostFallbackRequest) -> bool:
                if permission_manager is None:
                    return False

                async def emit(raw: dict[str, Any]) -> None:
                    await bus.publish(PermissionRequestedEvent(**raw, run_id=run_id))

                approved, fallback_decision, fallback_request_id = (
                    await permission_manager.request_host_fallback(
                        tool_use_id=tool_call.id,
                        request=request,
                        session_id=session_id,
                        event_emitter=emit,
                    )
                )
                event_type = PermissionGrantedEvent if approved else PermissionDeniedEvent
                await bus.publish(
                    event_type(
                        run_id=run_id,
                        request_id=fallback_request_id,
                        tool_use_id=tool_call.id,
                        decision=fallback_decision,
                        ts=_now(),
                    )
                )
                return approved

            root = (workspace_root or Path.cwd()).expanduser().resolve(strict=True)
            tool_context = ToolCallContext(
                run_id=run_id,
                session_id=session_id,
                tool_use_id=tool_call.id,
                workspace_root=root,
                on_execution_started=_execution_started,
                authorize_host_fallback=_authorize_host_fallback,
            )
            if tool.execution_profile != "workspace_sandbox":
                await _execution_started(
                    "external" if tool.execution_profile == "external" else "in_process",
                    None,
                )
            invocation = (
                tool.invoke(dict(tool_call.input), context=tool_context)
                if tool.execution_profile == "workspace_sandbox"
                else tool.invoke(dict(tool_call.input))
            )
            result = await asyncio.wait_for(invocation, timeout=timeout)
            ms = elapsed()

            if result.is_error:
                error_class = result.error_type or "runtime_error"
                error_message = result.content
                retryable = result.retryable is True
            else:
                await bus.publish(
                    ToolCallFinishedEvent(
                        run_id=run_id,
                        tool_use_id=tool_call.id,
                        tool_name=tool_call.name,
                        elapsed_ms=ms,
                        output=result.content,
                        retryable=result.retryable is True,
                        ts=_now(),
                    )
                )
                return result

        except RateLimitedError as exc:
            error_class = "rate_limited"
            error_message = str(exc)
            retryable = True
        except TimeoutError:
            return await _fail(
                bus, run_id, tool_call,
                "timeout", f"tool timed out after {timeout}s", elapsed(),
                attempt=attempt,
            )
        except Exception as exc:
            error_class = "runtime_error"
            error_message = str(exc)

        assert error_class is not None and error_message is not None
        ms = elapsed()

        if retryable and attempt <= _MAX_RETRIES:
            await bus.publish(
                ToolCallFailedEvent(
                    run_id=run_id,
                    tool_use_id=tool_call.id,
                    tool_name=tool_call.name,
                    error_class=error_class,
                    error_message=error_message,
                    elapsed_ms=ms,
                    attempt=attempt,
                    retryable=True,
                    ts=_now(),
                )
            )
            await asyncio.sleep(_RETRY_BASE_S * (2 ** (attempt - 1)))
            continue

        return await _fail(
            bus, run_id, tool_call,
            error_class, error_message, ms,
            attempt=attempt,
            retryable=retryable,
        )

    # unreachable, but keeps mypy happy
    return ToolResult(content="internal error", is_error=True, error_type="runtime_error")
