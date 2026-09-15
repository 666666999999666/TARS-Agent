from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from tars_agent.core.permissions.manager import PermissionManager
from tars_agent.core.tools.runtime.router import HostFallbackRequest


async def _request_and_respond(
    manager: PermissionManager,
    *,
    decision: str,
    session_id: str = "s1",
) -> tuple[bool, str, str, list[dict[str, Any]]]:
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)
        assert manager.respond(event["request_id"], session_id, decision)

    allowed, resolved, request_id = await manager.check_and_wait(
        "tool-1",
        "bash",
        {"command": "echo hi"},
        session_id,
        emit,
    )
    return allowed, resolved, request_id, emitted


async def test_tool_permission_uses_typed_request_id() -> None:
    manager = PermissionManager()
    allowed, decision, request_id, emitted = await _request_and_respond(
        manager,
        decision="allow_once",
    )
    assert allowed
    assert decision == "allow_once"
    assert request_id.startswith("perm-")
    assert emitted[0]["request_kind"] == "tool"
    assert emitted[0]["allowed_decisions"] == [
        "allow_once",
        "allow_session",
        "deny_once",
        "deny_session",
    ]


async def test_allow_session_does_not_cross_session() -> None:
    manager = PermissionManager()
    await _request_and_respond(manager, decision="allow_session", session_id="s1")
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)
        manager.respond(event["request_id"], "s2", "deny_once")

    allowed, _, _ = await manager.check_and_wait(
        "tool-2", "bash", {"command": "echo"}, "s2", emit
    )
    assert not allowed
    assert emitted


async def test_wrong_session_or_invalid_decision_does_not_resolve() -> None:
    manager = PermissionManager(timeout_s=0.01)

    async def emit(event: dict[str, Any]) -> None:
        assert not manager.respond(event["request_id"], "other", "allow_once")
        assert not manager.respond(event["request_id"], "s1", "allow_host_once")

    allowed, decision, _ = await manager.check_and_wait(
        "tool-1", "bash", {"command": "echo"}, "s1", emit
    )
    assert not allowed
    assert decision == "timeout"


async def test_host_fallback_only_accepts_single_host_grant() -> None:
    manager = PermissionManager()
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)
        assert event["allowed_decisions"] == ["allow_host_once", "deny_once"]
        assert manager.respond(event["request_id"], "s1", "allow_host_once")

    allowed, decision, _ = await manager.request_host_fallback(
        tool_use_id="tool-1",
        request=HostFallbackRequest(
            tool_name="bash",
            params={"command": "echo"},
            reason="sandbox_unavailable",
            parameter_digest="digest",
            platform_shell="PowerShell",
        ),
        session_id="s1",
        event_emitter=emit,
    )
    assert allowed
    assert decision == "allow_host_once"
    assert emitted[0]["request_kind"] == "host_fallback"
    assert emitted[0]["parameter_digest"] == "digest"


async def test_host_grant_is_consumed_once() -> None:
    manager = PermissionManager(timeout_s=0.01)
    first_request_id = ""

    async def allow_first(event: dict[str, Any]) -> None:
        nonlocal first_request_id
        first_request_id = event["request_id"]
        manager.respond(first_request_id, "s1", "allow_host_once")

    request = HostFallbackRequest(
        "bash", {"command": "echo"}, "sandbox_unavailable", "digest", "shell"
    )
    first, _, _ = await manager.request_host_fallback(
        tool_use_id="t1", request=request, session_id="s1", event_emitter=allow_first
    )
    assert first

    async def do_not_answer(event: dict[str, Any]) -> None:
        assert event["request_id"] != first_request_id

    second, decision, _ = await manager.request_host_fallback(
        tool_use_id="t2", request=request, session_id="s1", event_emitter=do_not_answer
    )
    assert not second
    assert decision == "timeout"


async def test_cancel_session_resolves_pending() -> None:
    manager = PermissionManager(timeout_s=1)
    emitted = asyncio.Event()

    async def emit(event: dict[str, Any]) -> None:
        emitted.set()

    task = asyncio.create_task(
        manager.check_and_wait("tool", "bash", {"command": "echo"}, "s1", emit)
    )
    await emitted.wait()
    manager.cancel_session("s1")
    allowed, decision, _ = await task
    assert not allowed
    assert decision == "deny_once"


async def test_cancelled_wait_removes_pending_request() -> None:
    manager = PermissionManager(timeout_s=0)
    emitted = asyncio.Event()
    request_id = ""

    async def emit(event: dict[str, Any]) -> None:
        nonlocal request_id
        request_id = event["request_id"]
        emitted.set()

    task = asyncio.create_task(
        manager.check_and_wait("tool", "bash", {"command": "echo"}, "s1", emit)
    )
    await emitted.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert request_id
    assert not manager.respond(request_id, "s1", "allow_once")


def test_legacy_allow_is_removed_but_deny_is_retained(tmp_path: Path) -> None:
    policy = tmp_path / "policy.toml"
    policy.write_text('[always]\nbash = "allow"\nwrite_file = "deny"\n', encoding="utf-8")
    PermissionManager(policy_file=policy)
    assert 'bash = "allow"' not in policy.read_text(encoding="utf-8")
    assert 'write_file = "deny"' in policy.read_text(encoding="utf-8")
    report = tmp_path / "policy-migration-v1.1.json"
    assert "bash" in report.read_text(encoding="utf-8")


async def test_session_grant_requires_identical_parameters() -> None:
    manager = PermissionManager()
    await _request_and_respond(manager, decision="allow_session")
    events = []

    async def deny(event):
        events.append(event)
        manager.respond(event["request_id"], "s1", "deny_once")

    same, _, _ = await manager.check_and_wait("t2", "bash", {"command": "echo hi"}, "s1", deny)
    assert same and not events
    changed, _, _ = await manager.check_and_wait("t3", "bash", {"command": "echo changed"}, "s1", deny)
    assert not changed and len(events) == 1


async def test_emitter_failure_removes_pending_approval() -> None:
    manager = PermissionManager()

    async def failed(event):
        raise RuntimeError("transport failed")

    import pytest
    with pytest.raises(RuntimeError, match="transport failed"):
        await manager.check_and_wait("t", "bash", {"command": "echo hi"}, "s", failed)
    assert manager._pending == {}


async def test_policy_deny_precedes_identical_cached_grant() -> None:
    from tars_agent.core.permissions.policy import PermissionDecision, ToolPolicy
    manager = PermissionManager()
    await _request_and_respond(manager, decision="allow_session")
    manager._policies["bash"] = ToolPolicy(default=PermissionDecision.DENY)
    async def unexpected(event):
        raise AssertionError("denied tool must not ask")
    allowed, decision, _ = await manager.check_and_wait("t", "bash", {"command": "echo hi"}, "s1", unexpected)
    assert not allowed and decision == "auto_deny"


async def test_workspace_escape_is_rejected_before_cached_grant(tmp_path: Path) -> None:
    manager = PermissionManager()
    async def unexpected(event):
        raise AssertionError("escaping path must not request approval")
    allowed, decision, _ = await manager.check_and_wait(
        "t", "read_file", {"path": "../outside.txt"}, "s", unexpected, workspace_root=tmp_path)
    assert not allowed and decision == "auto_deny"
