from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tars_agent.core.config import SandboxConfig
from tars_agent.core.tools.runtime import DockerRuntime
from tars_agent.core.tools.runtime.models import ToolExecutionRequest

pytestmark = pytest.mark.docker
_TEST_INSTANCE_IDS: set[str] = set()
_TEST_CONTAINER_IDS: dict[str, set[str]] = {}


def _runtime(config: SandboxConfig) -> DockerRuntime:
    runtime = DockerRuntime(config)
    _TEST_INSTANCE_IDS.add(runtime._instance_id)
    _TEST_CONTAINER_IDS[runtime._instance_id] = set()
    live = Path.cwd() / "build/qa/docker-real-live-instances.jsonl"
    live.parent.mkdir(parents=True, exist_ok=True)
    with live.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"instance_id": runtime._instance_id, "pytest_pid": os.getpid()}) + "\n")
    return runtime


@pytest.fixture(scope="module", autouse=True)
def owned_instance_evidence():
    yield
    observations = []
    for instance_id in sorted(_TEST_INSTANCE_IDS):
        remaining = _sandbox_container_ids(instance_id)
        observed = sorted(_TEST_CONTAINER_IDS[instance_id])
        still_inspectable = [container_id for container_id in observed if _docker_inspect(container_id) is not None]
        observations.append({"instance_id": instance_id, "observed_container_ids": observed,
                             "remaining_container_ids": remaining, "still_inspectable": still_inspectable})
        if remaining:
            _remove_containers(set(remaining), instance_id)
    evidence = Path.cwd() / "build/qa/docker-real-test-instances.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")
    assert not any(item["remaining_container_ids"] or item["still_inspectable"] for item in observations)



def _docker_ready(image: str) -> bool:
    try:
        info = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if info.returncode:
        return False
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return inspect.returncode == 0


def _docker_inspect(container_id: str) -> dict[str, object] | None:
    result = subprocess.run(
        ["docker", "inspect", container_id],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode:
        return None
    records = json.loads(result.stdout)
    return records[0] if records else None


def _sandbox_container_ids(instance_id: str) -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--filter",
         f"label=com.tars-agent.instance={instance_id}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return [line for line in result.stdout.splitlines() if line]


def _remove_containers(container_ids: set[str], instance_id: str) -> None:
    if not container_ids:
        return
    owned = set(_sandbox_container_ids(instance_id))
    if not container_ids <= owned:
        raise AssertionError("refusing to clean a container outside this test instance")
    subprocess.run(
        ["docker", "rm", "--force", *sorted(container_ids)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


async def test_real_docker_workspace_isolation_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    (docker_config / "config.json").write_text(
        json.dumps(
            {
                "proxies": {
                    "default": {
                        "httpProxy": "http://proxy-user:proxy-pass@proxy.invalid:3128",
                        "httpsProxy": "http://secure-user:secure-pass@proxy.invalid:3129",
                        "allProxy": "socks5://all-user:all-pass@proxy.invalid:1080",
                        "noProxy": "private.invalid",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCKER_CONFIG", str(docker_config))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-anthropic-secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/host/ssh-agent.sock")
    image = os.environ.get("TARS_SANDBOX_IMAGE", "tars-agent-sandbox:0.8.0")
    if not _docker_ready(image):
        if os.environ.get("TARS_REQUIRE_DOCKER_E2E") == "1":
            pytest.fail("required Docker engine or sandbox image is unavailable")
        pytest.skip("Docker engine or sandbox image unavailable")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside-sentinel.txt"
    outside.write_text("HOST-SECRET")
    runtime = _runtime(SandboxConfig(image=image))
    instance_id = runtime._instance_id
    assert not _sandbox_container_ids(instance_id)
    container_id = ""

    def request(tool_name: str, params: dict[str, object], tool_id: str) -> ToolExecutionRequest:
        return ToolExecutionRequest(
            invocation_id=f"run-1:{tool_id}",
            run_id="run-1",
            session_id="session-1",
            tool_use_id=tool_id,
            tool_name=tool_name,
            params=params,
            workspace_root=workspace.resolve(),
            timeout_s=10,
        )

    try:
        write = await runtime.execute(request("write_file", {"path": "ok.txt", "content": "ok"}, "1"))
        assert not write.is_error
        assert write.container_id is not None
        _TEST_CONTAINER_IDS[instance_id].add(write.container_id)
        assert (workspace / "ok.txt").read_text() == "ok"
        source_worker = Path(__file__).resolve().parents[2] / "src/tars_agent/sandbox/worker.py"
        expected_worker_hash = hashlib.sha256(source_worker.read_bytes()).hexdigest()
        fingerprint = await runtime.execute(request(
            "bash", {"command": "sha256sum /opt/tars/worker.py"}, "worker-fingerprint"
        ))
        assert not fingerprint.is_error and expected_worker_hash in fingerprint.content

        # Workspace tools expose relative paths; use Linux paths inside the
        # actual container when checking the second, container-side boundary.
        escape = await runtime.execute(request("read_file", {"path": "../outside-sentinel.txt"}, "2"))
        assert escape.error_type == "sandbox_policy_denied"
        absolute_escape = await runtime.execute(request(
            "read_file", {"path": "/outside-sentinel.txt"}, "absolute-escape"
        ))
        assert absolute_escape.error_type == "sandbox_policy_denied"
        assert outside.read_text() == "HOST-SECRET"

        # Create the Linux link in the actual container; Windows host symlink
        # privileges must not silently remove this security scenario.
        link = await runtime.execute(request(
            "bash", {"command": "ln -s /outside-sentinel.txt /workspace/outside-link"}, "link"
        ))
        assert not link.is_error
        symlink_escape = await runtime.execute(
            request("read_file", {"path": "outside-link"}, "symlink")
        )
        assert symlink_escape.error_type == "sandbox_policy_denied"

        bash_escape = await runtime.execute(
            request("bash", {"command": "test ! -e /outside-sentinel.txt"}, "bash-escape")
        )
        assert not bash_escape.is_error

        secret = await runtime.execute(
            request(
                "bash",
                {
                    "command": (
                        "for name in ANTHROPIC_API_KEY SSH_AUTH_SOCK "
                        "HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY FTP_PROXY "
                        "http_proxy https_proxy all_proxy no_proxy ftp_proxy; do "
                        "value=\"$(printenv \"$name\" || true)\"; "
                        "test -z \"$value\" || { echo \"$name leaked\"; exit 42; }; "
                        "done"
                    )
                },
                "3",
            )
        )
        assert not secret.is_error
        assert "leaked" not in secret.content

        inspected = await runtime.inspect_run("run-1")
        assert inspected is not None
        container_id = str(inspected["Id"])
        host_config = inspected["HostConfig"]
        assert host_config["NetworkMode"] == "none"
        assert host_config["ReadonlyRootfs"] is True
        assert host_config["CapDrop"] == ["ALL"]
        assert "no-new-privileges:true" in host_config["SecurityOpt"]
        assert host_config["Memory"] == 512 * 1024 * 1024
        assert host_config["MemorySwap"] == 512 * 1024 * 1024
        assert host_config["PidsLimit"] == 128
        assert host_config["NanoCpus"] == 1_000_000_000
        assert host_config["Ulimits"] == [
            {"Name": "nofile", "Hard": 1024, "Soft": 1024}
        ]
        assert host_config["Tmpfs"].keys() == {"/home/tars", "/tmp"}
        configured_env = {
            name: value
            for item in inspected["Config"]["Env"]
            if "=" in item
            for name, value in [item.split("=", 1)]
        }
        for name in (
            "ANTHROPIC_API_KEY",
            "SSH_AUTH_SOCK",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "FTP_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "ftp_proxy",
        ):
            assert configured_env.get(name) == ""
        mounts = inspected["Mounts"]
        assert len(mounts) == 1
        assert mounts[0]["Destination"] == "/workspace"
        assert mounts[0]["RW"] is True
    finally:
        try:
            await runtime.cleanup()
        finally:
            leaked_ids = set(_sandbox_container_ids(instance_id))
            if leaked_ids:
                _remove_containers(leaked_ids, instance_id)

    await asyncio.sleep(0)
    assert container_id
    assert _docker_inspect(container_id) is None
    assert not leaked_ids, f"test instance left containers before emergency cleanup: {leaked_ids}"
    assert not _sandbox_container_ids(instance_id)


async def test_real_docker_chinese_workspace_round_trip(tmp_path: Path) -> None:
    workspace = tmp_path / "中文 工作区"
    workspace.mkdir()
    runtime = _runtime(SandboxConfig())
    def request(name, params, tool_id):
        return ToolExecutionRequest(
            invocation_id=f"chinese:{tool_id}", run_id="chinese", session_id="chinese",
            tool_use_id=tool_id, tool_name=name, params=params,
            workspace_root=workspace.resolve(), timeout_s=10,
        )
    try:
        written = await runtime.execute(request("write_file", {"path": "报告.txt", "content": "中文验收✓"}, "write"))
        assert not written.is_error
        assert written.container_id is not None
        _TEST_CONTAINER_IDS[runtime._instance_id].add(written.container_id)
        read = await runtime.execute(request("read_file", {"path": "报告.txt"}, "read"))
        assert not read.is_error and read.content == "中文验收✓"
        bash = await runtime.execute(request("bash", {"command": "printf '中文输出'"}, "bash"))
        assert not bash.is_error and bash.content == "中文输出"
        assert (workspace / "报告.txt").read_text(encoding="utf-8") == "中文验收✓"
    finally:
        await runtime.cleanup()
    assert not _sandbox_container_ids(runtime._instance_id)


async def test_real_required_missing_image_fails_before_container_start(tmp_path: Path, monkeypatch) -> None:
    import uuid

    from tars_agent.core.tools.runtime import RuntimeRouter, factory, initialize_runtime_router
    config = SandboxConfig(image=f"tars-agent-absent:{uuid.uuid4().hex}", mode="required")
    runtime = _runtime(config)
    router = RuntimeRouter(runtime, allow_host_fallback=False)
    monkeypatch.setattr(factory, "build_runtime_router", lambda _: router)
    with pytest.raises(RuntimeError, match="required preflight failed: sandbox_image_missing"):
        await initialize_runtime_router(config)
    assert not runtime._container_start_attempted
    assert not _sandbox_container_ids(runtime._instance_id)


async def test_real_preferred_needs_independent_host_approval(tmp_path: Path) -> None:
    import uuid

    from tars_agent.core.bus.events import PermissionRequestedEvent
    from tars_agent.core.events.bus import EventBus
    from tars_agent.core.llm.types import ToolCallBlock
    from tars_agent.core.permissions.manager import PermissionManager
    from tars_agent.core.tools.builtin.write_file import WriteFileTool
    from tars_agent.core.tools.invocation import invoke_tool
    from tars_agent.core.tools.registry import ToolRegistry
    from tars_agent.core.tools.runtime import RuntimeRouter
    runtime = _runtime(SandboxConfig(image=f"tars-agent-absent:{uuid.uuid4().hex}", mode="preferred"))
    router = RuntimeRouter(runtime, allow_host_fallback=True)
    registry = ToolRegistry()
    registry.register(WriteFileTool(router))
    permissions = PermissionManager(timeout_s=0.05)
    bus = EventBus()
    host_decisions = iter(["deny_once", "allow_host_once", None])
    requests = []
    async def respond(event):
        if not isinstance(event, PermissionRequestedEvent):
            return
        requests.append(event.request_kind)
        decision = next(host_decisions) if event.request_kind == "host_fallback" else "allow_session"
        if decision is not None:
            assert permissions.respond(event.request_id, event.session_id, decision)
    bus.subscribe(respond)
    async def invoke(index):
        return await invoke_tool(
            registry, ToolCallBlock(id=str(index), name="write_file", input={"path": "批准.txt", "content": "仅此一次"}),
            bus, run_id="preferred", session_id="preferred", permission_manager=permissions,
            workspace_root=tmp_path,
        )
    try:
        denied = await invoke(1)
        assert denied.is_error and not (tmp_path / "批准.txt").exists()
        approved = await invoke(2)
        assert not approved.is_error and approved.backend == "host"
        assert (tmp_path / "批准.txt").read_text(encoding="utf-8") == "仅此一次"
        expired = await invoke(3)
        assert expired.is_error and expired.error_type == "host_fallback_denied"
        assert requests == ["tool", "host_fallback", "host_fallback", "host_fallback"]
        assert not runtime._container_start_attempted
    finally:
        await router.cleanup()


@pytest.mark.parametrize("parent_exits", [False, True])
async def test_real_docker_timeout_cleans_descendants(tmp_path: Path, parent_exits: bool) -> None:
    runtime = _runtime(SandboxConfig())
    command = "(sleep 1; printf orphan > late.txt) & " + ("exit 0" if parent_exits else "wait")
    request = ToolExecutionRequest(
        invocation_id="timeout:tool", run_id="timeout", session_id="timeout", tool_use_id="tool",
        tool_name="bash", params={"command": command, "timeout": 0.2},
        workspace_root=tmp_path.resolve(), timeout_s=0.2,
    )
    container_id = None
    try:
        result = await runtime.execute(request)
        assert result.is_error and result.error_type == "timeout"
        container_id = result.container_id
        assert container_id is not None
        _TEST_CONTAINER_IDS[runtime._instance_id].add(container_id)
        await asyncio.sleep(1.2)
        assert not (tmp_path / "late.txt").exists()
    finally:
        await runtime.cleanup()
    assert container_id is not None and _docker_inspect(container_id) is None
    assert not _sandbox_container_ids(runtime._instance_id)


async def test_real_docker_cancellation_confirms_container_removed(tmp_path: Path) -> None:
    runtime = _runtime(SandboxConfig())
    request = ToolExecutionRequest(
        invocation_id="cancel:tool", run_id="cancel", session_id="cancel", tool_use_id="tool",
        tool_name="bash", params={"command": "printf started > started.txt; (sleep 2; printf orphan > late.txt) & wait"},
        workspace_root=tmp_path.resolve(), timeout_s=20,
    )
    task = asyncio.create_task(runtime.execute(request))
    container_id = None
    try:
        async with asyncio.timeout(15):
            while not (tmp_path / "started.txt").exists():
                if task.done():
                    raise AssertionError(f"tool exited before marker: {task.result()}")
                await asyncio.sleep(0.02)
        inspected = await runtime.inspect_run("cancel")
        assert inspected is not None
        container_id = str(inspected["Id"])
        _TEST_CONTAINER_IDS[runtime._instance_id].add(container_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _docker_inspect(container_id) is None
        await asyncio.sleep(2.2)
        assert not (tmp_path / "late.txt").exists()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await runtime.cleanup()
    assert container_id is not None and not _sandbox_container_ids(runtime._instance_id)
