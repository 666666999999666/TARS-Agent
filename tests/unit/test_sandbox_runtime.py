from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tars_agent.core.config import SandboxConfig
from tars_agent.core.tools.runtime import DockerRuntime, FakeRuntime, HostRuntime, RuntimeRouter
from tars_agent.core.tools.runtime.models import ToolExecutionRequest
from tars_agent.sandbox.worker import execute_payload, resolve_workspace_path


def _request(tmp_path: Path, *, tool_name: str = "read_file") -> ToolExecutionRequest:
    return ToolExecutionRequest(
        invocation_id="run:tool",
        run_id="run-1",
        session_id="session-1",
        tool_use_id="tool-1",
        tool_name=tool_name,
        params={"path": "file.txt"},
        workspace_root=tmp_path.resolve(),
        timeout_s=10,
    )


def test_docker_args_apply_security_controls(tmp_path: Path) -> None:
    runtime = DockerRuntime(SandboxConfig())
    args = runtime.build_container_args(_request(tmp_path), "tars-test")
    joined = " ".join(args)
    assert "--network none" in joined
    assert "--read-only" in args
    assert "--cap-drop ALL" in joined
    assert "no-new-privileges:true" in args
    assert "--pids-limit 128" in joined
    assert "--memory 512m" in joined
    assert "--memory-swap 512m" in joined
    assert "--cpus 1.0" in joined
    assert "nofile=1024:1024" in args
    assert "--init" in args
    assert "dst=/workspace" in joined
    assert "docker.sock" not in joined
    cleared_env = {
        args[index + 1]
        for index, value in enumerate(args[:-1])
        if value == "--env"
    }
    assert cleared_env == {
        "ANTHROPIC_API_KEY=",
        "SSH_AUTH_SOCK=",
        "HTTP_PROXY=",
        "HTTPS_PROXY=",
        "ALL_PROXY=",
        "NO_PROXY=",
        "FTP_PROXY=",
        "http_proxy=",
        "https_proxy=",
        "all_proxy=",
        "no_proxy=",
        "ftp_proxy=",
    }


@pytest.mark.parametrize(
    ("run_code", "run_stdout"),
    ((124, ""), (1, ""), (0, "")),
)
async def test_uncertain_container_late_creation_is_reconciled_by_run_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_code: int,
    run_stdout: str,
) -> None:
    from tars_agent.core.tools.runtime.models import RuntimeStatus

    runtime = DockerRuntime(SandboxConfig())
    commands: list[list[str]] = []
    inspect_count = 0

    async def available():  # type: ignore[no-untyped-def]
        return RuntimeStatus(True, "workspace_sandbox")

    async def command(args, *, timeout_s=None):  # type: ignore[no-untyped-def]
        nonlocal inspect_count
        del timeout_s
        commands.append(args)
        if args[1] == "run":
            return run_code, run_stdout, "run failed"
        if args[1] == "inspect":
            inspect_count += 1
            if inspect_count == 1:
                return 1, "", "Error: No such object"
            return 0, runtime._instance_id, ""  # type: ignore[attr-defined]
        if args[1:3] == ["rm", "--force"]:
            return 0, "", ""
        raise AssertionError(f"unexpected Docker command: {args}")

    monkeypatch.setattr(runtime, "preflight", available)
    monkeypatch.setattr(runtime, "_command", command)
    result = await runtime.execute(_request(tmp_path))

    assert result.is_error
    assert result.started is False
    run_args = next(args for args in commands if args[1] == "run")
    container_name = run_args[run_args.index("--name") + 1]
    assert container_name.startswith(f"tars-{runtime._instance_id}-")  # type: ignore[attr-defined]
    assert container_name in runtime._uncertain_container_names  # type: ignore[attr-defined]
    assert not any(args[1:3] == ["rm", "--force"] for args in commands)

    # The daemon creates the labelled container after the first compensating
    # lookup. Run finalization must retry the exact name and remove it.
    await runtime.cleanup_run("run-1")

    remove_args = next(args for args in commands if args[1:3] == ["rm", "--force"])
    assert remove_args[-1] == container_name
    assert container_name not in runtime._uncertain_container_names  # type: ignore[attr-defined]


async def test_cancelled_container_start_finishes_compensating_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.tools.runtime.models import RuntimeStatus

    runtime = DockerRuntime(SandboxConfig())
    commands: list[list[str]] = []

    async def available():  # type: ignore[no-untyped-def]
        return RuntimeStatus(True, "workspace_sandbox")

    async def command(args, *, timeout_s=None):  # type: ignore[no-untyped-def]
        del timeout_s
        commands.append(args)
        if args[1] == "run":
            raise asyncio.CancelledError
        if args[1] == "inspect":
            return 0, runtime._instance_id, ""  # type: ignore[attr-defined]
        if args[1:3] == ["rm", "--force"]:
            return 0, "", ""
        raise AssertionError(f"unexpected Docker command: {args}")

    monkeypatch.setattr(runtime, "preflight", available)
    monkeypatch.setattr(runtime, "_command", command)

    with pytest.raises(asyncio.CancelledError):
        await runtime.execute(_request(tmp_path))

    run_args = next(args for args in commands if args[1] == "run")
    remove_args = next(args for args in commands if args[1:3] == ["rm", "--force"])
    assert remove_args[-1] == run_args[run_args.index("--name") + 1]


async def test_uncertain_cleanup_never_removes_foreign_same_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.tools.runtime.models import RuntimeStatus

    runtime = DockerRuntime(SandboxConfig())
    commands: list[list[str]] = []

    async def available():  # type: ignore[no-untyped-def]
        return RuntimeStatus(True, "workspace_sandbox")

    async def command(args, *, timeout_s=None):  # type: ignore[no-untyped-def]
        del timeout_s
        commands.append(args)
        if args[1] == "run":
            return 1, "", "name conflict"
        if args[1] == "inspect":
            return 0, "different-runtime-instance", ""
        raise AssertionError(f"unexpected Docker command: {args}")

    monkeypatch.setattr(runtime, "preflight", available)
    monkeypatch.setattr(runtime, "_command", command)

    result = await runtime.execute(_request(tmp_path))

    assert result.is_error
    assert not any(args[1:3] == ["rm", "--force"] for args in commands)


def test_resolve_rejects_absolute_and_parent_escape(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text("ok")
    assert resolve_workspace_path(tmp_path, "safe.txt") == tmp_path / "safe.txt"
    with pytest.raises(PermissionError):
        resolve_workspace_path(tmp_path, "../outside.txt")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    with pytest.raises(PermissionError):
        resolve_workspace_path(tmp_path, str(outside))


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlink unsupported")
def test_resolve_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-symlink-target.txt"
    outside.write_text("secret")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not permitted")
    with pytest.raises(PermissionError):
        resolve_workspace_path(tmp_path, "link.txt")


async def test_worker_never_inherits_api_key_into_bash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sentinel-secret")
    result = await execute_payload(
        {
            "tool_name": "bash",
            "params": {
                "command": subprocess.list2cmdline(
                    [
                        sys.executable,
                        "-c",
                        "import os; print(os.getenv('ANTHROPIC_API_KEY'))",
                    ]
                )
            },
            "timeout_s": 10,
        },
        tmp_path,
        sandboxed=False,
    )
    assert result["is_error"] is False
    assert "sentinel-secret" not in str(result["content"])
    assert "None" in str(result["content"])


async def test_worker_applies_configured_output_limit_to_bash_and_read(
    tmp_path: Path,
) -> None:
    command = subprocess.list2cmdline(
        [sys.executable, "-c", "print('x' * 10000)"]
    )
    bash = await execute_payload(
        {
            "tool_name": "bash",
            "params": {"command": command},
            "timeout_s": 10,
            "output_limit_bytes": 128,
        },
        tmp_path,
        sandboxed=False,
    )
    assert bash["truncated"] is True
    assert len(str(bash["stdout"]).encode()) <= 128
    assert len(str(bash["content"]).encode()) <= 160

    (tmp_path / "large.txt").write_text("y" * 1000)
    read = await execute_payload(
        {
            "tool_name": "read_file",
            "params": {"path": "large.txt"},
            "output_limit_bytes": 64,
        },
        tmp_path,
        sandboxed=False,
    )
    assert read["truncated"] is True
    assert str(read["content"]).startswith("y" * 64)


async def test_cancelled_host_bash_kills_child_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    killed = asyncio.Event()

    class HangingStream:
        async def read(self, _size: int) -> bytes:
            started.set()
            await killed.wait()
            return b""

    class HangingProcess:
        returncode: int | None = None
        stdout = HangingStream()
        stderr = HangingStream()

        async def wait(self) -> int:
            await killed.wait()
            return self.returncode or 0

        def kill(self) -> None:
            self.returncode = -9
            killed.set()

    async def create(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return HangingProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", create)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    task = asyncio.create_task(
        execute_payload(
            {"tool_name": "bash", "params": {"command": "hang"}, "timeout_s": 60},
            tmp_path,
            sandboxed=False,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert killed.is_set()


async def test_router_cancel_stops_active_host_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    killed = asyncio.Event()

    class HangingStream:
        async def read(self, _size: int) -> bytes:
            started.set()
            await killed.wait()
            return b""

    class HangingProcess:
        returncode: int | None = None
        stdout = HangingStream()
        stderr = HangingStream()

        async def wait(self) -> int:
            await killed.wait()
            return self.returncode or 0

        def kill(self) -> None:
            self.returncode = -9
            killed.set()

    async def create(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return HangingProcess()

    async def approve(_request):  # type: ignore[no-untyped-def]
        return True

    monkeypatch.setattr(asyncio, "create_subprocess_shell", create)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    router = RuntimeRouter(FakeRuntime(available=False), HostRuntime())
    request = ToolExecutionRequest(
        invocation_id="host-active",
        run_id="run-host",
        session_id="session-1",
        tool_use_id="tool-host",
        tool_name="bash",
        params={"command": "hang"},
        workspace_root=tmp_path.resolve(),
        timeout_s=60,
    )
    execution = asyncio.create_task(
        router.execute(request, authorize_host_fallback=approve)
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    await router.cancel("host-active")

    assert killed.is_set()
    assert execution.cancelled()


class _CountingHost(HostRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def execute(self, request: ToolExecutionRequest):  # type: ignore[no-untyped-def]
        self.calls += 1
        return await super().execute(request)


async def test_unavailable_sandbox_denial_starts_no_host_process(tmp_path: Path) -> None:
    host = _CountingHost()
    router = RuntimeRouter(FakeRuntime(available=False), host)

    async def deny(_request):  # type: ignore[no-untyped-def]
        return False

    result = await router.execute(_request(tmp_path), authorize_host_fallback=deny)
    assert result.error_type == "host_fallback_denied"
    assert result.started is False
    assert host.calls == 0


async def test_host_once_executes_exactly_one_invocation(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("hello")
    host = _CountingHost()
    router = RuntimeRouter(FakeRuntime(available=False), host)
    decisions = iter((True, False))

    async def approve_once(_request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)
        return next(decisions)

    first = await router.execute(_request(tmp_path), authorize_host_fallback=approve_once)
    second = await router.execute(_request(tmp_path), authorize_host_fallback=approve_once)
    assert not first.is_error
    assert second.error_type == "host_fallback_denied"
    assert host.calls == 1


async def test_host_fallback_rejects_parameters_mutated_after_approval(
    tmp_path: Path,
) -> None:
    host = _CountingHost()
    router = RuntimeRouter(FakeRuntime(available=False), host)
    request = _request(tmp_path)

    async def approve_then_mutate(_request):  # type: ignore[no-untyped-def]
        request.params["path"] = "changed-after-approval.txt"
        return True

    result = await router.execute(
        request,
        authorize_host_fallback=approve_then_mutate,
    )

    assert result.error_type == "sandbox_policy_denied"
    assert result.started is False
    assert host.calls == 0


async def test_host_fallback_discloses_full_params_real_shell_and_warning(
    tmp_path: Path,
) -> None:
    router = RuntimeRouter(FakeRuntime(available=False), _CountingHost())
    request = _request(tmp_path, tool_name="bash")
    request.params.clear()
    request.params["command"] = "echo a-command-that-must-not-be-truncated"
    captured = None

    async def deny(fallback_request):  # type: ignore[no-untyped-def]
        nonlocal captured
        captured = fallback_request
        return False

    await router.execute(request, authorize_host_fallback=deny)
    assert captured is not None
    assert captured.params == request.params
    assert captured.platform_shell != "host default shell"
    assert "network visibility" in captured.warning


async def test_started_sandbox_failure_never_falls_back(tmp_path: Path) -> None:
    class StartedFailure(FakeRuntime):
        async def execute(self, request: ToolExecutionRequest):  # type: ignore[no-untyped-def]
            from tars_agent.core.tools.runtime.models import ToolExecutionResult

            return ToolExecutionResult(
                "lost",
                "workspace_sandbox",
                True,
                is_error=True,
                error_type="sandbox_lost",
            )

    host = _CountingHost()
    router = RuntimeRouter(StartedFailure(), host)

    async def approve(_request):  # type: ignore[no-untyped-def]
        return True

    result = await router.execute(_request(tmp_path), authorize_host_fallback=approve)
    assert result.error_type == "sandbox_lost"
    assert host.calls == 0


async def test_docker_command_timeout_kills_hung_cli(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class HungProcess:
        returncode: int | None = None
        killed = False
        communicates = 0

        async def communicate(self):  # type: ignore[no-untyped-def]
            self.communicates += 1
            if self.communicates == 1:
                await asyncio.Future()
            self.returncode = -9
            return b"", b""

        def kill(self) -> None:
            self.killed = True

    process = HungProcess()

    async def create(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    code, stdout, stderr = await DockerRuntime._command(
        ["docker", "info"],
        timeout_s=0.01,
    )
    assert code == 124
    assert stdout == ""
    assert "timed out" in stderr
    assert process.killed


async def test_worker_oom_is_non_retryable_and_discards_run_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OomProcess:
        returncode = 137

        async def communicate(self, _payload):  # type: ignore[no-untyped-def]
            return b"", b""

    async def create(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return OomProcess()

    commands: list[list[str]] = []

    async def command(args, *, timeout_s=None):  # type: ignore[no-untyped-def]
        del timeout_s
        commands.append(args)
        if "inspect" in args:
            return 0, '{"OOMKilled":true}', ""
        return 0, "", ""

    runtime = DockerRuntime(SandboxConfig())
    container = SimpleNamespace(
        id="container-1",
        name="tars-test",
        run_id="run-1",
        workspace_root=tmp_path,
        lock=asyncio.Lock(),
    )
    runtime._containers["run-1"] = container  # type: ignore[attr-defined]
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(runtime, "_command", command)

    result = await runtime._exec(container, _request(tmp_path))  # type: ignore[arg-type]

    assert result.error_type == "sandbox_oom"
    assert result.started is True
    assert result.retryable is False
    assert "run-1" not in runtime._containers  # type: ignore[attr-defined]
    assert any(command_args[1:3] == ["rm", "--force"] for command_args in commands)


async def test_failed_container_removal_is_retried_by_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = DockerRuntime(SandboxConfig())
    runtime._containers["run-1"] = SimpleNamespace(  # type: ignore[attr-defined]
        id="container-1",
        name="tars-test",
        run_id="run-1",
        workspace_root=tmp_path,
        lock=asyncio.Lock(),
    )
    attempts = 0

    async def command(_args, *, timeout_s=None):  # type: ignore[no-untyped-def]
        nonlocal attempts
        del timeout_s
        attempts += 1
        return (1, "", "daemon temporarily unavailable") if attempts == 1 else (0, "", "")

    monkeypatch.setattr(runtime, "_command", command)
    from tars_agent.core.tools.runtime import RuntimeCleanupPending
    with pytest.raises(RuntimeCleanupPending):
        await runtime.cleanup_run("run-1")
    assert runtime.pending_cleanup_run_ids() == ("run-1",)
    assert "run-1" in runtime._containers  # type: ignore[attr-defined]
    await runtime.cleanup()
    assert "run-1" not in runtime._containers  # type: ignore[attr-defined]


async def test_cleanup_reaps_only_current_runtime_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = DockerRuntime(SandboxConfig())
    runtime._container_start_attempted = True  # type: ignore[attr-defined]
    commands: list[list[str]] = []

    async def command(args, *, timeout_s=None):  # type: ignore[no-untyped-def]
        del timeout_s
        commands.append(args)
        if args[1] == "ps":
            return 0, "owned-1\nowned-2\n", ""
        if args[1:3] == ["rm", "--force"]:
            return 0, "", ""
        raise AssertionError(f"unexpected Docker command: {args}")

    monkeypatch.setattr(runtime, "_command", command)
    await runtime.cleanup()

    list_args = commands[0]
    assert list_args[-1] == (
        f"label=com.tars-agent.instance={runtime._instance_id}"  # type: ignore[attr-defined]
    )
    assert commands[1] == ["docker", "rm", "--force", "owned-1", "owned-2"]


async def test_workspace_mutation_lock_serializes_same_workspace_only(tmp_path: Path) -> None:
    class ConcurrentDocker(DockerRuntime):
        def __init__(self) -> None:
            super().__init__(SandboxConfig())
            self.active: dict[Path, int] = {}
            self.max_active: dict[Path, int] = {}
            self.global_active = 0
            self.global_max = 0

        async def preflight(self):  # type: ignore[no-untyped-def]
            from tars_agent.core.tools.runtime.models import RuntimeStatus

            return RuntimeStatus(True, "workspace_sandbox")

        async def _ensure_container(self, request):  # type: ignore[no-untyped-def]
            container = SimpleNamespace(
                id=request.run_id,
                name=request.run_id,
                run_id=request.run_id,
                workspace_root=request.workspace_root,
                lock=asyncio.Lock(),
            )
            from tars_agent.core.tools.runtime.models import ToolExecutionResult

            return container, ToolExecutionResult("", "workspace_sandbox", False)

        async def _exec(self, container, request):  # type: ignore[no-untyped-def]
            del container
            root = request.workspace_root
            self.active[root] = self.active.get(root, 0) + 1
            self.max_active[root] = max(self.max_active.get(root, 0), self.active[root])
            self.global_active += 1
            self.global_max = max(self.global_max, self.global_active)
            await asyncio.sleep(0.03)
            self.active[root] -= 1
            self.global_active -= 1
            from tars_agent.core.tools.runtime.models import ToolExecutionResult

            return ToolExecutionResult("ok", "workspace_sandbox", True)

    other = tmp_path / "other"
    other.mkdir()
    runtime = ConcurrentDocker()

    def mutation(run_id: str, root: Path) -> ToolExecutionRequest:
        request = _request(root, tool_name="write_file")
        return ToolExecutionRequest(
            invocation_id=run_id,
            run_id=run_id,
            session_id=request.session_id,
            tool_use_id=run_id,
            tool_name=request.tool_name,
            params=request.params,
            workspace_root=root.resolve(),
            timeout_s=10,
        )

    await asyncio.gather(
        runtime.execute(mutation("same-1", tmp_path)),
        runtime.execute(mutation("same-2", tmp_path)),
        runtime.execute(mutation("other", other)),
    )
    assert runtime.max_active[tmp_path.resolve()] == 1
    assert runtime.max_active[other.resolve()] == 1
    assert runtime.global_max == 2


async def test_required_sandbox_disables_host_fallback(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock
    host = AsyncMock()
    authorize = AsyncMock(return_value=True)
    router = RuntimeRouter(FakeRuntime(available=False), host, allow_host_fallback=False)
    result = await router.execute(_request(tmp_path), authorize_host_fallback=authorize)
    assert result.is_error and not result.started
    host.execute.assert_not_awaited()
    authorize.assert_not_awaited()


async def test_failed_preflight_cleans_unreturned_runtime() -> None:
    from unittest.mock import AsyncMock, patch

    from tars_agent.core.tools.runtime import RuntimeStatus, initialize_runtime_router
    runtime = AsyncMock()
    runtime.preflight.return_value = RuntimeStatus(available=False, backend="workspace_sandbox", reason="offline")
    with patch("tars_agent.core.tools.runtime.factory.build_runtime_router", return_value=runtime):
        with pytest.raises(RuntimeError, match="required preflight failed"):
            await initialize_runtime_router(SandboxConfig())
    runtime.cleanup.assert_awaited_once()


async def test_sandbox_cleanup_failure_still_cleans_host() -> None:
    from unittest.mock import AsyncMock
    sandbox = AsyncMock()
    sandbox.cleanup_run.side_effect = RuntimeError("cleanup failed")
    host = AsyncMock()
    router = RuntimeRouter(sandbox, host)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await router.cleanup_run("run")
    host.cleanup_run.assert_awaited_once_with("run")


@pytest.mark.parametrize("cancelled", [False, True])
async def test_docker_started_callback_failure_reaps_cli_and_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool,
) -> None:
    from dataclasses import replace
    from unittest.mock import AsyncMock
    class Process:
        returncode = None
        killed = False
        async def communicate(self, *args):
            assert self.killed
            self.returncode = -9
            return b"", b""
        def kill(self):
            self.killed = True
    process = Process()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    runtime = DockerRuntime(SandboxConfig())
    cleanup = AsyncMock()
    monkeypatch.setattr(runtime, "cleanup_run", cleanup)
    async def failed(backend, container_id):
        if cancelled:
            raise asyncio.CancelledError
        raise RuntimeError("event sink failed")
    request = replace(_request(tmp_path), on_started=failed)
    container = SimpleNamespace(id="owned-container")
    with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
        await runtime._exec(container, request)
    assert process.killed and process.returncode == -9
    assert runtime._active == {}
    cleanup.assert_awaited_once_with(request.run_id)


async def test_failed_container_removal_is_explicitly_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tars_agent.core.tools.runtime import RuntimeCleanupPending
    runtime = DockerRuntime(SandboxConfig())
    runtime._containers["run-1"] = SimpleNamespace(id="owned", run_id="run-1")
    async def failed(args, *, timeout_s):
        assert timeout_s == 3.0
        return 1, "", "daemon temporarily unavailable"
    monkeypatch.setattr(runtime, "_command", failed)
    router = RuntimeRouter(runtime)
    with pytest.raises(RuntimeCleanupPending) as caught:
        await router.cleanup_run("run-1")
    assert caught.value.run_id == "run-1"
    assert router.pending_cleanup_run_ids() == ("run-1",)


@pytest.mark.parametrize("message", [
    "Error: No such object: owned", "error: no such object: owned",
    "Error response from daemon: No such container: owned",
    "error response from daemon: no such container: owned",
])
def test_docker_missing_container_error_accepts_real_cli_casing(message: str) -> None:
    from tars_agent.core.tools.runtime.docker import _is_missing_container
    assert _is_missing_container(message)


def test_unavailable_daemon_is_not_evidence_of_container_absence() -> None:
    from tars_agent.core.tools.runtime.docker import _is_missing_container
    assert not _is_missing_container("Cannot connect to the Docker daemon")
