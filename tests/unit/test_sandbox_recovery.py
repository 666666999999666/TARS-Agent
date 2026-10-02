from __future__ import annotations

import asyncio
import copy
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tars_agent.core.config import SandboxConfig
from tars_agent.core.control import CoreHomeLock
from tars_agent.core.tools.runtime import DockerRuntime
from tars_agent.core.tools.runtime.models import (
    RuntimeCleanupPending,
    ToolExecutionRequest,
    ToolExecutionResult,
)
from tars_agent.core.tools.runtime.recovery import (
    SandboxRecoveryError,
    SandboxResourceRecord,
    SandboxResourceStore,
    mount_identity,
    recover_core_sandboxes,
)


class DockerReplies:
    """In-memory Docker command responses; never launches a subprocess."""

    def __init__(self) -> None:
        self.engine = "engine-A"
        self.image = "sha256:" + "f" * 64
        self.containers: dict[str, dict[str, Any]] = {}
        self.calls: list[list[str]] = []
        self.fail: str | None = None
        self.keep_after_rm = False
        self.start_mode = "normal"
        self._next_id = 1

    async def __call__(
        self, args: list[str], *, timeout_s: float | None = None,
    ) -> tuple[int, str, str]:
        del timeout_s
        self.calls.append(args)
        verb = args[1]
        if self.fail == verb:
            return 1, "", "injected Docker failure"
        if verb == "info":
            return 0, self.engine if args[-1] == "{{.ID}}" else "test-version", ""
        if verb == "image":
            return 0, json.dumps([{"Id": self.image}]), ""
        if verb == "ps":
            filter_ = args[-1]
            ids = list(self.containers)
            if filter_.startswith("id="):
                ids = [cid for cid in ids if cid == filter_[3:]]
            elif filter_.startswith("label="):
                key, value = filter_[6:].split("=", 1)
                ids = [cid for cid in ids
                       if self.containers[cid]["Config"]["Labels"].get(key) == value]
            return 0, "\n".join(ids), ""
        if verb == "inspect":
            target = args[-1]
            value = next((item for cid, item in self.containers.items()
                          if cid == target or item["Name"] == "/" + target), None)
            if value is None:
                return 1, "", "Error: No such container"
            return 0, json.dumps([value]), ""
        if verb == "run":
            if self.start_mode == "absent":
                return 124, "", "timeout"
            labels = dict(args[index + 1].split("=", 1)
                          for index, argument in enumerate(args[:-1]) if argument == "--label")
            mount = args[args.index("--mount") + 1]
            workspace = mount.split("src=", 1)[1].rsplit(",dst=", 1)[0]
            container_id = f"{self._next_id:064x}"
            self._next_id += 1
            self.containers[container_id] = {
                "Id": container_id, "Name": "/" + args[args.index("--name") + 1],
                "Created": "2026-09-29T12:00:00.123456789Z", "Image": args[-1],
                "Config": {"Labels": labels},
                "Mounts": [{"Type": "bind", "Source": workspace,
                            "Destination": "/workspace", "RW": True}],
                "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True},
            }
            if self.start_mode == "cancel":
                raise asyncio.CancelledError
            if self.start_mode == "lost_reply":
                return 124, "", "timeout"
            return 0, container_id, ""
        if verb == "rm":
            assert len(args) == 4 and len(args[-1]) == 64
            if not self.keep_after_rm:
                self.containers.pop(args[-1], None)
            return 0, "", ""
        raise AssertionError(f"unexpected Docker command: {args}")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SandboxResourceStore]:
    lock = CoreHomeLock(tmp_path / "home")
    lock.acquire()
    try:
        yield SandboxResourceStore(lock)
    finally:
        lock.release()


def request(tmp_path: Path, run_id: str = "run-1") -> ToolExecutionRequest:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return ToolExecutionRequest(
        invocation_id=run_id + ":tool", run_id=run_id, session_id="session-1",
        tool_use_id="tool-1", tool_name="read_file", params={"path": "input.txt"},
        workspace_root=workspace.resolve(), timeout_s=1.0,
    )


async def managed_runtime(
    store: SandboxResourceStore, docker: DockerReplies, monkeypatch: pytest.MonkeyPatch,
) -> DockerRuntime:
    config = SandboxConfig()
    owner = await recover_core_sandboxes(config, store, launch_id="launch-A", command=docker)
    runtime = DockerRuntime(config, owner=owner)
    monkeypatch.setattr(runtime, "_command", docker)
    return runtime


async def make_container(
    store: SandboxResourceStore, docker: DockerReplies, monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> tuple[DockerRuntime, SandboxResourceRecord]:
    runtime = await managed_runtime(store, docker, monkeypatch)
    container, result = await runtime._ensure_container(request(workspace))
    assert container is not None and not result.is_error
    return runtime, store.load()[0]


def test_home_lock_is_exclusive_released_and_not_inherited(tmp_path: Path) -> None:
    first = CoreHomeLock(tmp_path / "home")
    second = CoreHomeLock(tmp_path / "home")
    different = CoreHomeLock(tmp_path / "different")
    first.acquire()
    try:
        assert first._handle is not None
        assert not os.get_inheritable(first._handle.fileno())
        with pytest.raises(RuntimeError, match="HOME is in use"):
            second.acquire()
        different.acquire()
        different.release()
    finally:
        first.release()
    second.acquire()
    second.release()
    assert (tmp_path / "home" / "control" / "core.lock").is_file()


def test_store_cannot_be_used_without_home_lock(tmp_path: Path) -> None:
    lock = CoreHomeLock(tmp_path)
    with pytest.raises(SandboxRecoveryError, match="home_lock_not_held"):
        SandboxResourceStore(lock)
    lock.acquire()
    store = SandboxResourceStore(lock)
    lock.release()
    with pytest.raises(SandboxRecoveryError, match="home_lock_not_held"):
        store.load()


@pytest.mark.parametrize("invalid", [b"{", b'{"version":99}', b"[]"])
def test_corrupt_records_fail_closed(store: SandboxResourceStore, invalid: bytes) -> None:
    store.directory.mkdir(parents=True)
    (store.directory / ("a" * 32 + ".json")).write_bytes(invalid)
    with pytest.raises(SandboxRecoveryError, match="invalid_or_missing_resource_record"):
        store.load()


async def test_intent_is_persisted_before_create_and_confirmation_before_exec(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    runtime = await managed_runtime(store, docker, monkeypatch)

    async def command(args: list[str], *, timeout_s: float | None = None) -> tuple[int, str, str]:
        if args[1] == "run":
            records = store.load()
            assert len(records) == 1 and records[0].phase == "intent"
            assert args[-1] == docker.image
        return await docker(args, timeout_s=timeout_s)

    async def execute(_container: Any, _request: ToolExecutionRequest) -> ToolExecutionResult:
        record = store.load()[0]
        assert record.phase == "confirmed" and record.container_id in docker.containers
        return ToolExecutionResult("read", "workspace_sandbox", True)

    monkeypatch.setattr(runtime, "_command", command)
    monkeypatch.setattr(runtime, "_exec", execute)
    result = await runtime.execute(request(tmp_path))
    assert result.started and not result.is_error
    await runtime.cleanup_run("run-1")
    assert store.load() == [] and docker.containers == {}


async def test_record_write_failure_never_starts_container(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    runtime = await managed_runtime(store, docker, monkeypatch)

    def fail_replace(_source: Any, _target: Any) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(os, "replace", fail_replace)
    container, error = await runtime._ensure_container(request(tmp_path))
    assert container is None and error.error_type == "sandbox_recovery_required"
    assert not any(args[1] == "run" for args in docker.calls)


async def test_restart_removes_old_owner_but_preserves_other_home_and_ephemeral(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    _, record = await make_container(store, docker, monkeypatch, tmp_path)
    old = copy.deepcopy(docker.containers[record.container_id])
    foreign = copy.deepcopy(old)
    foreign["Id"] = "b" * 64
    foreign["Name"] = "/foreign"
    foreign["Config"]["Labels"]["com.tars-agent.home"] = "c" * 64
    ephemeral = copy.deepcopy(old)
    ephemeral["Id"] = "e" * 64
    ephemeral["Name"] = "/ephemeral"
    ephemeral["Config"]["Labels"] = {
        "com.tars-agent.sandbox": "true", "com.tars-agent.purpose": "ephemeral",
    }
    docker.containers[foreign["Id"]] = foreign
    docker.containers[ephemeral["Id"]] = ephemeral
    before = copy.deepcopy(docker.containers)
    await recover_core_sandboxes(SandboxConfig(), store, launch_id="launch-new", command=docker)
    assert set(docker.containers) == {foreign["Id"], ephemeral["Id"]}
    assert all(value == before[cid] for cid, value in docker.containers.items())
    assert store.load() == []
    await recover_core_sandboxes(SandboxConfig(), store, launch_id="launch-next", command=docker)
    assert len([args for args in docker.calls if args[1] == "rm"]) == 1


async def test_lost_create_reply_is_recovered_from_persisted_intent(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    runtime, confirmed = await make_container(store, docker, monkeypatch, tmp_path)
    intent = SandboxResourceRecord.model_validate({
        **confirmed.model_dump(), "phase": "intent", "container_id": None, "created": None,
    })
    store.save(intent)
    del runtime
    await recover_core_sandboxes(SandboxConfig(), store, launch_id="next", command=docker)
    assert not docker.containers and not store.load()


async def test_absent_intent_does_not_become_success_or_allow_a_second_creation(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    runtime = await managed_runtime(store, docker, monkeypatch)
    docker.start_mode = "absent"
    _, error = await runtime._ensure_container(request(tmp_path))
    assert error.error_type == "sandbox_recovery_required"
    _, second_error = await runtime._ensure_container(request(tmp_path))
    assert second_error.error_type == "sandbox_recovery_required"
    assert len([args for args in docker.calls if args[1] == "run"]) == 1
    with pytest.raises(SandboxRecoveryError, match="creation_outcome_unknown"):
        await recover_core_sandboxes(SandboxConfig(), store, launch_id="next", command=docker)
    assert store.load()[0].phase == "intent"


async def test_absent_confirmed_record_is_safely_retired_on_restart(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    await make_container(store, docker, monkeypatch, tmp_path)
    docker.containers.clear()  # Crash after Docker removal, before retiring the record.
    await recover_core_sandboxes(SandboxConfig(), store, launch_id="next", command=docker)
    assert store.load() == []
    assert not any(args[1] == "rm" for args in docker.calls)


@pytest.mark.parametrize("field", ["Id", "Created", "Image", "Name", "mount", "home", "launch", "run", "resource"])
async def test_identity_mismatch_never_deletes(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    docker = DockerReplies()
    _, record = await make_container(store, docker, monkeypatch, tmp_path)
    container = docker.containers[record.container_id]
    if field == "mount":
        container["Mounts"][0]["Source"] = str(tmp_path / "different")
    elif field in {"home", "launch", "run", "resource"}:
        container["Config"]["Labels"]["com.tars-agent." + field] = "wrong"
    else:
        container[field] = "wrong"
    with pytest.raises(SandboxRecoveryError):
        await recover_core_sandboxes(SandboxConfig(), store, launch_id="next", command=docker)
    assert not any(args[1] == "rm" for args in docker.calls)
    assert store.load()


@pytest.mark.parametrize("kind", ["legacy", "unrecorded"])
async def test_unknown_containers_block_before_any_removal(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    docker = DockerReplies()
    _, record = await make_container(store, docker, monkeypatch, tmp_path)
    unknown = copy.deepcopy(docker.containers[record.container_id])
    unknown["Id"] = "b" * 64
    if kind == "legacy":
        unknown["Config"]["Labels"].pop("com.tars-agent.purpose")
    else:
        unknown["Config"]["Labels"]["com.tars-agent.resource"] = "b" * 32
    docker.containers[unknown["Id"]] = unknown
    with pytest.raises(SandboxRecoveryError):
        await recover_core_sandboxes(SandboxConfig(), store, launch_id="next", command=docker)
    assert not any(args[1] == "rm" for args in docker.calls)
    assert len(docker.containers) == 2


async def test_changed_engine_blocks_before_deletion(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    await make_container(store, docker, monkeypatch, tmp_path)
    docker.engine = "other-engine"
    with pytest.raises(SandboxRecoveryError, match="record_engine_mismatch"):
        await recover_core_sandboxes(SandboxConfig(), store, launch_id="next", command=docker)
    assert not any(args[1] == "rm" for args in docker.calls)


@pytest.mark.parametrize("failure", ["info", "inspect", "ps", "rm", "still_present"])
async def test_query_and_cleanup_failures_preserve_records(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    docker = DockerReplies()
    await make_container(store, docker, monkeypatch, tmp_path)
    if failure == "still_present":
        docker.keep_after_rm = True
    else:
        docker.fail = failure
    with pytest.raises(SandboxRecoveryError):
        await recover_core_sandboxes(SandboxConfig(), store, launch_id="next", command=docker)
    assert store.load()
    docker.fail = None
    docker.keep_after_rm = False
    await recover_core_sandboxes(SandboxConfig(), store, launch_id="next", command=docker)
    assert store.load() == []


async def test_failed_cleanup_can_retry_after_identity_was_persisted(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    runtime, record = await make_container(store, docker, monkeypatch, tmp_path)
    docker.fail = "rm"
    with pytest.raises(RuntimeCleanupPending):
        await runtime.cleanup_run("run-1")
    assert runtime.pending_cleanup_run_ids() == ("run-1",)
    docker.fail = None
    await runtime.cleanup()
    assert not runtime.pending_cleanup_run_ids() and store.load() == []
    assert record.container_id not in docker.containers


async def test_cancelled_start_reconciles_only_its_record(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    runtime = await managed_runtime(store, docker, monkeypatch)
    docker.start_mode = "cancel"
    with pytest.raises(asyncio.CancelledError):
        await runtime._ensure_container(request(tmp_path))
    assert store.load() == [] and not docker.containers


def test_mount_identity_preserves_linux_case_and_normalizes_desktop_drive() -> None:
    assert mount_identity(r"C:\Users\Person\Work") == mount_identity("/run/desktop/mnt/host/c/Users/Person/Work")
    assert mount_identity("/work/ABC") != mount_identity("/work/abc")


def test_ephemeral_construction_does_not_create_home_records(tmp_path: Path) -> None:
    tool_request = request(tmp_path)
    before = set(tmp_path.rglob("*"))
    runtime = DockerRuntime(SandboxConfig())
    args = runtime.build_container_args(tool_request, "tars-ephemeral")
    assert "com.tars-agent.purpose=ephemeral" in args
    assert runtime._owner is None and not runtime._managed_records
    assert set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("direction", ["parent", "equal", "child"])
async def test_workspace_cannot_expose_core_identity_records(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direction: str,
) -> None:
    from dataclasses import replace

    docker = DockerReplies()
    runtime = await managed_runtime(store, docker, monkeypatch)
    root = {"parent": store.lock.home.parent, "equal": store.lock.home,
            "child": store.lock.home / "workspace"}[direction]
    root.mkdir(exist_ok=True)
    _, error = await runtime._ensure_container(replace(request(tmp_path), workspace_root=root))
    assert error.error_type == "sandbox_recovery_required"
    assert "workspace_overlaps_core_home" in error.content
    assert not any(args[1] == "run" for args in docker.calls)


async def test_confirmation_write_failure_prevents_first_tool_execution(
    store: SandboxResourceStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    runtime = await managed_runtime(store, docker, monkeypatch)
    original_save = store.save

    def save(record: SandboxResourceRecord) -> None:
        if record.phase == "confirmed":
            raise SandboxRecoveryError("resource_record_write_failed", record.resource_id)
        original_save(record)

    async def unexpected_exec(*_args: Any) -> ToolExecutionResult:
        raise AssertionError("cannot execute before confirmation is durable")

    monkeypatch.setattr(store, "save", save)
    monkeypatch.setattr(runtime, "_exec", unexpected_exec)
    result = await runtime.execute(request(tmp_path))
    assert result.is_error and result.started is False
    assert store.load()[0].phase == "intent"
    assert docker.containers
    monkeypatch.setattr(store, "save", original_save)
    await runtime.cleanup()
    assert store.load() == [] and docker.containers == {}


async def test_inventory_error_keeps_exit_code_and_redacted_diagnostics(
    store: SandboxResourceStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = DockerReplies()
    monkeypatch.setenv("RECOVERY_DIAGNOSTIC_SECRET", "private-diagnostic-value")

    async def command(args: list[str], *, timeout_s: float | None = None) -> tuple[int, str, str]:
        if args[1] == "ps":
            assert timeout_s == 15.0
            return 124, "", (
                "command timed out after 15s; private-diagnostic-value "
                "https://private-user:private-password@example.invalid token=private-token"
            )
        return await docker(args, timeout_s=timeout_s)

    with pytest.raises(SandboxRecoveryError) as caught:
        await recover_core_sandboxes(SandboxConfig(), store, launch_id="next", command=command)
    assert caught.value.reason == "docker_inventory_failed"
    assert caught.value.exit_code == 124 and caught.value.operation == "inventory"
    assert "timed out after 15s" in str(caught.value)
    assert "[REDACTED]" in str(caught.value)
    assert "private-" not in str(caught.value)
    assert not any(args[1] == "rm" for args in docker.calls)
