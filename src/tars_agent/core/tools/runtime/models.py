from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

ToolExecutionProfile = Literal["in_process", "workspace_sandbox", "external"]
ToolBackend = Literal["in_process", "workspace_sandbox", "host", "external"]
ExecutionStartedCallback = Callable[[ToolBackend, str | None], Awaitable[None]]
if TYPE_CHECKING:
    from tars_agent.core.tools.runtime.router import HostFallbackAuthorizer


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    run_id: str
    session_id: str
    tool_use_id: str
    workspace_root: Path
    network_mode: Literal["disabled"] = "disabled"
    on_execution_started: ExecutionStartedCallback | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    authorize_host_fallback: HostFallbackAuthorizer | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        root = self.workspace_root.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("workspace_root must be an existing directory")
        object.__setattr__(self, "workspace_root", root)


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    available: bool
    backend: ToolBackend
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolExecutionRequest:
    invocation_id: str
    run_id: str
    session_id: str
    tool_use_id: str
    tool_name: str
    params: dict[str, object]
    workspace_root: Path
    timeout_s: float
    network_mode: Literal["disabled"] = "disabled"
    on_started: ExecutionStartedCallback | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def worker_payload(
        self,
        *,
        output_limit_bytes: int | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "invocation_id": self.invocation_id,
            "tool_name": self.tool_name,
            "params": self.params,
            "timeout_s": self.timeout_s,
        }
        if output_limit_bytes is not None:
            payload["output_limit_bytes"] = output_limit_bytes
        return payload


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    content: str
    backend: ToolBackend
    started: bool
    is_error: bool = False
    error_type: str | None = None
    retryable: bool = False
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False
    elapsed_ms: int = 0
    container_id: str | None = None


WORKSPACE_TOOLS = frozenset({"bash", "read_file", "write_file", "list_dir"})
MUTATING_WORKSPACE_TOOLS = frozenset({"bash", "write_file"})


__all__ = [
    "ExecutionStartedCallback",
    "MUTATING_WORKSPACE_TOOLS",
    "RuntimeStatus",
    "ToolBackend",
    "ToolCallContext",
    "ToolExecutionProfile",
    "ToolExecutionRequest",
    "ToolExecutionResult",
    "WORKSPACE_TOOLS",
]


class RuntimeCleanupPending(RuntimeError):
    """Owned execution resources still exist or could not be verified absent."""
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"runtime cleanup is not yet confirmed for run {run_id}")
