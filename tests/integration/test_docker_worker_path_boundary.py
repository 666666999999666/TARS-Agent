"""Opt-in worker boundary proof, below the unchanged production permission layer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from tars_agent.core.config import SandboxConfig
from tars_agent.core.control import CoreHomeLock
from tars_agent.core.tools.runtime import DockerRuntime
from tars_agent.core.tools.runtime.models import ToolExecutionRequest
from tars_agent.core.tools.runtime.recovery import (
    SandboxResourceStore,
    recover_core_sandboxes,
    validate_container,
)

pytestmark = pytest.mark.docker


def _save(root: Path, evidence: dict[str, Any]) -> None:
    (root / "results.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8",
    )


@pytest.mark.timeout(240)
async def test_real_worker_rejects_parent_path_without_leaking_or_mutating_sentinel() -> None:
    if os.environ.get("RUN_CORE_RECOVERY_DOCKER") != "1":
        pytest.skip("explicit RUN_CORE_RECOVERY_DOCKER=1 is required")
    image = os.environ.get("CORE_RECOVERY_IMAGE", "")
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", image), "pin CORE_RECOVERY_IMAGE to a full image ID"
    requested_root = os.environ.get("CORE_RECOVERY_ROOT")
    assert requested_root, "set a fresh CORE_RECOVERY_ROOT beneath build/core-recovery"
    project = Path(__file__).resolve().parents[2]
    allowed = (project / "build" / "core-recovery").resolve()
    root = Path(requested_root).resolve()
    assert allowed.is_relative_to(project.resolve())
    assert root != allowed and root.is_relative_to(allowed) and not root.exists()
    root.mkdir(parents=True)
    workspace, home = root / "workspace", root / "home"
    workspace.mkdir()
    home.mkdir()
    assert not workspace.is_relative_to(home) and not home.is_relative_to(workspace)
    inside_value = "inside-boundary-probe\n"
    outside_value = "outside-synthetic-" + secrets.token_hex(32)
    (workspace / "inside.txt").write_bytes(inside_value.encode("utf-8"))
    outside = root / "outside.txt"
    outside.write_text(outside_value, encoding="utf-8")
    before = hashlib.sha256(outside.read_bytes()).hexdigest()
    evidence: dict[str, Any] = {
        "test_layer": "direct_runtime_docker_worker", "root": str(root), "image": image,
        "model_calls": 0, "passed": False, "errors": [], "cleanup_calls": 0,
        "outside_sha256_before": before,
    }
    _save(root, evidence)
    lock = CoreHomeLock(home)
    runtime: DockerRuntime | None = None
    cleanup_complete = False
    try:
        lock.acquire()
        store = SandboxResourceStore(lock)
        config = SandboxConfig(mode="required", image=image)
        owner = await recover_core_sandboxes(
            config, store, launch_id="worker-boundary-" + secrets.token_hex(12),
            command=DockerRuntime._command,
        )
        runtime = DockerRuntime(config, owner=owner)
        run_id = "path-boundary-" + secrets.token_hex(6)
        evidence["owner"] = {
            "home_id": store.home_id, "engine_id": owner.engine_id,
            "launch_id": owner.launch_id, "instance_id": runtime._instance_id,
        }
        _save(root, evidence)

        def request(tool_use_id: str, path: str) -> ToolExecutionRequest:
            return ToolExecutionRequest(
                invocation_id=f"{run_id}:{tool_use_id}", run_id=run_id,
                session_id="path-boundary-session", tool_use_id=tool_use_id,
                tool_name="read_file", params={"path": path}, workspace_root=workspace,
                timeout_s=10,
            )

        inside = await runtime.execute(request("inside", "inside.txt"))
        evidence["normal_read"] = {
            "started": inside.started, "is_error": inside.is_error,
            "error_type": inside.error_type, "container_id": inside.container_id,
            "expected_sha256": hashlib.sha256(inside_value.encode()).hexdigest(),
            "returned_sha256": hashlib.sha256(inside.content.encode()).hexdigest(),
        }
        _save(root, evidence)
        records = store.load()
        assert len(records) == 1
        record = records[0]
        assert record.phase == "confirmed" and record.container_id == inside.container_id
        inspected = await runtime.inspect_run(run_id)
        assert inspected is not None
        validate_container(record, inspected)
        evidence["identity"] = record.model_dump()
        evidence["owner_process_pid"] = os.getpid()
        evidence["inspected_identity"] = {
            key: inspected[key] for key in ("Id", "Name", "Image", "Created", "Mounts")
        }
        evidence["inspected_identity"]["labels"] = inspected["Config"]["Labels"]
        evidence["normal_read_succeeded"] = (
            not inside.is_error and inside.started and inside.content == inside_value
        )
        _save(root, evidence)
        if not evidence["normal_read_succeeded"]:
            raise AssertionError("normal workspace read did not execute successfully")

        rejected = await runtime.execute(request("outside", "../outside.txt"))
        returned = "\n".join((rejected.content, rejected.stdout, rejected.stderr))
        not_leaked = outside_value not in returned
        after = hashlib.sha256(outside.read_bytes()).hexdigest()
        evidence.update(
            outside_sha256_after=after, outside_unchanged=after == before,
            outside_content_not_returned=not_leaked,
            worker_rejection={
                "started": rejected.started, "backend": rejected.backend,
                "is_error": rejected.is_error, "error_type": rejected.error_type,
                "container_id": rejected.container_id,
                "returned_content_sha256": hashlib.sha256(returned.encode()).hexdigest(),
            },
        )
        _save(root, evidence)
        if not (rejected.started and rejected.backend == "workspace_sandbox" and rejected.is_error
                and rejected.error_type == "sandbox_policy_denied"
                and rejected.container_id == record.container_id):
            raise AssertionError("the real worker did not return the expected path-policy denial")
        if after != before or not not_leaked:
            raise AssertionError("the external synthetic sentinel changed or its content was returned")

        evidence["cleanup_calls"] += 1
        await runtime.cleanup()
        confirmation = await runtime.confirm_cleanup()
        evidence["cleanup_confirmation"] = asdict(confirmation)
        assert confirmation.confirmed is True and confirmation.scope_id == record.instance_id
        assert store.load() == [] and not runtime.pending_cleanup_run_ids()
        assert record.container_id is not None
        code, stdout, _ = await runtime._command(
            [config.docker_binary, "ps", "--all", "--quiet", "--no-trunc", "--filter",
             f"id={record.container_id}"], timeout_s=15,
        )
        assert code == 0 and not stdout.strip(), "full container ID absence was not confirmed"
        evidence["full_container_id_absent"] = True
        cleanup_complete = True
        evidence["passed"] = True
    except BaseException as exc:
        evidence["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if runtime is not None and not cleanup_complete:
                evidence["cleanup_calls"] += 1
                try:
                    await runtime.cleanup()
                    confirmation = await runtime.confirm_cleanup()
                    evidence["cleanup_after_failure"] = asdict(confirmation)
                    if confirmation.confirmed is not True:
                        raise RuntimeError("production cleanup remains unconfirmed")
                except Exception as exc:
                    evidence["errors"].append(f"cleanup: {type(exc).__name__}: {exc}")
                evidence["passed"] = False
        finally:
            lock.release()
            _save(root, evidence)
