from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any, Literal

from tars_agent.core.permissions.policy import (
    DEFAULT_POLICIES,
    PermissionDecision,
    ToolPolicy,
    matches_outside_cwd,
    param_preview,
)
from tars_agent.core.permissions.storage import load_policy_file, save_policy_file
from tars_agent.core.tools.runtime.router import HostFallbackRequest

logger = logging.getLogger(__name__)

PermissionRequestKind = Literal["tool", "host_fallback"]
PermissionEmitter = Callable[[dict[str, Any]], Awaitable[None]]


def _now() -> str:
    return datetime.datetime.now(UTC).isoformat()


def _request_id() -> str:
    return f"perm-{uuid.uuid4().hex}"


@dataclass(slots=True)
class _PendingRequest:
    future: asyncio.Future[str]
    request_id: str
    request_kind: PermissionRequestKind
    session_id: str
    tool_name: str
    allowed_decisions: frozenset[str]
    parameter_digest: str


class PermissionManager:
    """Evaluate action policy and own session-scoped, typed approval requests."""

    def __init__(
        self,
        policies: dict[str, ToolPolicy] | None = None,
        *,
        policy_file: Path | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self._policies = policies or dict(DEFAULT_POLICIES)
        self._pending: dict[str, _PendingRequest] = {}
        self._session_decisions: dict[tuple[str, str, str], str] = {}
        self._policy_file = policy_file
        loaded = load_policy_file(policy_file) if policy_file is not None else {}
        # V1.1 never migrates a legacy persistent allow into a host-capable grant.
        self._persistent_denies = {
            tool: decision for tool, decision in loaded.items() if decision == "deny"
        }
        self._legacy_allows = sorted(
            tool for tool, decision in loaded.items() if decision == "allow"
        )
        if self._legacy_allows and policy_file is not None:
            report = policy_file.with_name("policy-migration-v1.1.json")
            report.write_text(
                json.dumps(
                    {
                        "ignored_legacy_allows": self._legacy_allows,
                        "reason": (
                            "V1.1 grants are session scoped and never authorize "
                            "host fallback"
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            save_policy_file(self._persistent_denies, policy_file)
        self._timeout_s = timeout_s

    def evaluate(self, tool_name: str, params: dict[str, Any]) -> PermissionDecision:
        from tars_agent.core.permissions.policy import evaluate

        return evaluate(tool_name, params, self._policies.get(tool_name))

    async def check_and_wait(
        self,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        event_emitter: PermissionEmitter,
        *,
        workspace_root: Path | None = None,
    ) -> tuple[bool, str, str]:
        if workspace_root is not None and tool_name in {"read_file", "write_file", "list_dir"}:
            from tars_agent.sandbox.worker import SandboxPolicyError, resolve_workspace_path
            try:
                resolve_workspace_path(workspace_root, str(params.get("path", ".")), write=True)
            except (SandboxPolicyError, OSError, ValueError):
                return False, "auto_deny", ""
        command = str(params.get("command", "")) if tool_name == "bash" else ""
        policy = self._policies.get(tool_name)
        if command and policy and any(
            re.search(pattern, command) for pattern in policy.deny_patterns
        ):
            return False, "auto_deny", ""
        if self._persistent_denies.get(tool_name) == "deny":
            return False, "auto_deny", ""
        if policy is not None and policy.default == PermissionDecision.DENY:
            return False, "auto_deny", ""
        digest = hashlib.sha256(
            json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        forced_ask = bool(command and matches_outside_cwd(command))
        if not forced_ask:
            cached = self._session_decisions.get((session_id, tool_name, digest))
            if cached is not None:
                return cached == "allow", f"auto_{cached}", ""
            if self._persistent_denies.get(tool_name) == "deny":
                return False, "auto_deny", ""
            if command and policy and any(
                re.search(pattern, command) for pattern in policy.allow_patterns
            ):
                return True, "auto_allow", ""
            if policy is not None and policy.default == PermissionDecision.ALLOW:
                return True, "auto_allow", ""
            if policy is not None and policy.default == PermissionDecision.DENY:
                return False, "auto_deny", ""
        return await self._request(
            request_kind="tool",
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            params=params,
            session_id=session_id,
            backend=(
                "in_process"
                if tool_name not in {"bash", "read_file", "write_file", "list_dir"}
                else "workspace_sandbox"
            ),
            risk="high" if tool_name in {"bash", "write_file"} else "medium",
            reason="tool policy requires approval",
            allowed_decisions=("allow_once", "allow_session", "deny_once", "deny_session"),
            event_emitter=event_emitter,
        )

    async def request_host_fallback(
        self,
        *,
        tool_use_id: str,
        request: HostFallbackRequest,
        session_id: str,
        event_emitter: PermissionEmitter,
    ) -> tuple[bool, str, str]:
        return await self._request(
            request_kind="host_fallback",
            tool_use_id=tool_use_id,
            tool_name=request.tool_name,
            params=request.params,
            session_id=session_id,
            backend="host",
            risk="critical",
            reason=request.reason,
            allowed_decisions=("allow_host_once", "deny_once"),
            parameter_digest=request.parameter_digest,
            platform_shell=request.platform_shell,
            warning=request.warning,
            event_emitter=event_emitter,
        )

    async def _request(
        self,
        *,
        request_kind: PermissionRequestKind,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        backend: str,
        risk: str,
        reason: str,
        allowed_decisions: tuple[str, ...],
        event_emitter: PermissionEmitter,
        parameter_digest: str | None = None,
        platform_shell: str | None = None,
        warning: str | None = None,
    ) -> tuple[bool, str, str]:
        request_id = _request_id()
        digest = parameter_digest or hashlib.sha256(
            json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        pending = _PendingRequest(
            future=future,
            request_id=request_id,
            request_kind=request_kind,
            session_id=session_id,
            tool_name=tool_name,
            allowed_decisions=frozenset(allowed_decisions),
            parameter_digest=digest,
        )
        self._pending[request_id] = pending
        try:
            await event_emitter(
                {
                    "type": "permission.requested",
                    "request_id": request_id,
                    "request_kind": request_kind,
                    "tool_use_id": tool_use_id,
                    "tool_name": tool_name,
                    "params": params,
                    "param_preview": param_preview(tool_name, params),
                    "session_id": session_id,
                    "backend": backend,
                    "risk": risk,
                    "reason": reason,
                    "allowed_decisions": list(allowed_decisions),
                    "parameter_digest": digest,
                    "platform_shell": platform_shell,
                    "warning": warning,
                    "ts": _now(),
                }
            )
            raw = (
                await asyncio.wait_for(future, timeout=self._timeout_s)
                if self._timeout_s > 0
                else await future
            )
        except TimeoutError:
            return False, "timeout", request_id
        finally:
            # A run cancellation must not leave an unreachable approval request
            # behind.  `respond()` and timeout already remove it; this is the
            # cancellation-safe, idempotent final guard.
            self._pending.pop(request_id, None)
        if request_kind == "host_fallback":
            return raw == "allow_host_once", raw, request_id
        if raw == "allow_session":
            self._session_decisions[(session_id, tool_name, digest)] = "allow"
        elif raw == "deny_session":
            self._session_decisions[(session_id, tool_name, digest)] = "deny"
        return raw in {"allow_once", "allow_session"}, raw, request_id

    def respond(self, request_id: str, session_id: str, decision: str) -> bool:
        pending = self._pending.get(request_id)
        if pending is None or pending.session_id != session_id:
            return False
        if decision not in pending.allowed_decisions:
            return False
        self._pending.pop(request_id, None)
        if not pending.future.done():
            pending.future.set_result(decision)
        return True

    def cancel_session(self, session_id: str, reason: str = "session_closed") -> None:
        del reason
        request_ids = [
            request_id
            for request_id, pending in self._pending.items()
            if pending.session_id == session_id
        ]
        for request_id in request_ids:
            pending = self._pending.pop(request_id)
            if not pending.future.done():
                pending.future.set_result("deny_once")


__all__ = ["PermissionEmitter", "PermissionManager", "PermissionRequestKind"]
