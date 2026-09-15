from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tars_agent.core.config import SandboxConfig
from tars_agent.core.tools.runtime.models import (
    MUTATING_WORKSPACE_TOOLS,
    RuntimeCleanupPending,
    RuntimeStatus,
    ToolExecutionRequest,
    ToolExecutionResult,
)

log = logging.getLogger(__name__)

_CLEARED_CONTAINER_ENV = (
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
)

_UNCERTAIN_RUN_CLEANUP_GRACE_S = 1.0
_UNCERTAIN_RUNTIME_CLEANUP_GRACE_S = 3.0
_UNCERTAIN_CONTAINER_POLL_S = 0.1


@dataclass(slots=True)
class _Container:
    id: str
    name: str
    run_id: str
    workspace_root: Path
    lock: asyncio.Lock


class DockerRuntime:
    """Run-scoped, network-disabled Docker runtime for workspace tools."""

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._instance_id = uuid.uuid4().hex[:12]
        self._containers: dict[str, _Container] = {}
        self._container_guard = asyncio.Lock()
        self._workspace_locks: dict[Path, asyncio.Lock] = {}
        self._active: dict[str, tuple[asyncio.subprocess.Process, str]] = {}
        self._uncertain_container_names: set[str] = set()
        self._uncertain_run_ids: dict[str, str] = {}
        self._container_start_attempted = False
        self._preflight_cache: tuple[float, RuntimeStatus] | None = None

    @property
    def output_limit_bytes(self) -> int:
        return self._config.output_limit_bytes

    async def preflight(self) -> RuntimeStatus:
        now = time.monotonic()
        if self._preflight_cache is not None and now - self._preflight_cache[0] < 2.0:
            return self._preflight_cache[1]
        docker = await self._command(
            [self._config.docker_binary, "info", "--format", "{{.ServerVersion}}"],
            timeout_s=15.0,
        )
        if docker[0] != 0:
            status = RuntimeStatus(
                available=False,
                backend="workspace_sandbox",
                reason="sandbox_unavailable",
                details={"stderr": _redact(docker[2])},
            )
        else:
            image = await self._command(
                [self._config.docker_binary, "image", "inspect", self._config.image],
                timeout_s=15.0,
            )
            if image[0] != 0:
                status = RuntimeStatus(
                    available=False,
                    backend="workspace_sandbox",
                    reason="sandbox_image_missing",
                    details={"image": self._config.image},
                )
            else:
                status = RuntimeStatus(
                    available=True,
                    backend="workspace_sandbox",
                    details={
                        "server_version": docker[1].strip(),
                        "image": self._config.image,
                    },
                )
        self._preflight_cache = (now, status)
        return status

    def build_container_args(self, request: ToolExecutionRequest, name: str) -> list[str]:
        mount = f"type=bind,src={request.workspace_root},dst=/workspace"
        args = [
            self._config.docker_binary,
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--label",
            "com.tars-agent.sandbox=true",
            "--label",
            f"com.tars-agent.instance={self._instance_id}",
            "--label",
            f"com.tars-agent.run={request.run_id}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(self._config.pids_limit),
            "--memory",
            self._config.memory,
            "--memory-swap",
            self._config.memory_swap,
            "--cpus",
            str(self._config.cpus),
            "--ulimit",
            f"nofile={self._config.nofile_limit}:{self._config.nofile_limit}",
            "--init",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",  # nosec B108: isolated container tmpfs mount, not a host file.
            "--tmpfs",
            "/home/tars:rw,noexec,nosuid,nodev,size=16m",
            "--mount",
            mount,
            "--workdir",
            "/workspace",
        ]
        for name in _CLEARED_CONTAINER_ENV:
            # Docker CLI proxy settings in ~/.docker/config.json are injected into
            # new containers automatically.  Explicit empty values override those
            # settings and also prevent image-level credentials from becoming visible.
            args.extend(["--env", f"{name}="])
        user = _host_user()
        if user is not None:
            args.extend(["--user", user])
        args.append(self._config.image)
        return args

    async def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        status = await self.preflight()
        if not status.available:
            return ToolExecutionResult(
                content=status.reason or "Docker sandbox unavailable",
                backend="workspace_sandbox",
                started=False,
                is_error=True,
                error_type=status.reason or "sandbox_unavailable",
                retryable=False,
            )
        container, error = await self._ensure_container(request)
        if container is None:
            return error
        workspace_lock = self._workspace_locks.setdefault(
            request.workspace_root,
            asyncio.Lock(),
        )
        mutation_lock = (
            workspace_lock
            if request.tool_name in MUTATING_WORKSPACE_TOOLS
            else _NullAsyncLock()
        )
        async with mutation_lock, container.lock:
            return await self._exec(container, request)

    async def _ensure_container(
        self,
        request: ToolExecutionRequest,
    ) -> tuple[_Container | None, ToolExecutionResult]:
        async with self._container_guard:
            existing = self._containers.get(request.run_id)
            if existing is not None:
                if existing.workspace_root != request.workspace_root:
                    return None, ToolExecutionResult(
                        content="run workspace changed after sandbox creation",
                        backend="workspace_sandbox",
                        started=False,
                        is_error=True,
                        error_type="sandbox_policy_denied",
                    )
                return existing, ToolExecutionResult("", "workspace_sandbox", False)
            name = _container_name(request.run_id, self._instance_id)
            self._container_start_attempted = True
            # Until ``docker run`` returns a usable id, its daemon-side outcome is
            # ambiguous. Keep a tombstone even when the first lookup says that the
            # name does not exist: a killed CLI may have already submitted a request
            # that the daemon completes later.
            self._uncertain_container_names.add(name)
            self._uncertain_run_ids[name] = request.run_id
            try:
                code, stdout, stderr = await self._command(
                    self.build_container_args(request, name),
                    timeout_s=30.0,
                )
            except asyncio.CancelledError:
                await _complete_cleanup_before_cancelling(
                    self._cleanup_uncertain_container(name)
                )
                raise
            if code != 0:
                await self._cleanup_uncertain_container(name)
                return None, ToolExecutionResult(
                    content=_redact(stderr or stdout or "failed to create sandbox container"),
                    backend="workspace_sandbox",
                    started=False,
                    is_error=True,
                    error_type="sandbox_start_failed",
                    retryable=False,
                )
            container_id = stdout.strip()
            if not container_id:
                await self._cleanup_uncertain_container(name)
                return None, ToolExecutionResult(
                    content="Docker returned no container id",
                    backend="workspace_sandbox",
                    started=False,
                    is_error=True,
                    error_type="sandbox_start_failed",
                )
            self._uncertain_container_names.discard(name)
            container = _Container(
                id=container_id,
                name=name,
                run_id=request.run_id,
                workspace_root=request.workspace_root,
                lock=asyncio.Lock(),
            )
            self._containers[request.run_id] = container
            return container, ToolExecutionResult("", "workspace_sandbox", False)

    async def _exec(
        self,
        container: _Container,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:
        payload = json.dumps(
            request.worker_payload(output_limit_bytes=self._config.output_limit_bytes),
            ensure_ascii=False,
        ).encode() + b"\n"
        args = [
            self._config.docker_binary,
            "exec",
            "--interactive",
            container.id,
            "python",
            "/opt/tars/worker.py",
            "--workspace",
            "/workspace",
        ]
        started_at = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            await self.cleanup_run(request.run_id)
            return ToolExecutionResult(
                content=str(exc),
                backend="workspace_sandbox",
                started=False,
                is_error=True,
                error_type="sandbox_lost",
            )
        self._active[request.invocation_id] = (process, request.run_id)
        try:
            if request.on_started is not None:
                await request.on_started("workspace_sandbox", container.id)
            stdout_raw, stderr_raw = await asyncio.wait_for(
                process.communicate(payload),
                timeout=request.timeout_s + 5.0,
            )
        except TimeoutError:
            await self.cancel(request.invocation_id)
            return ToolExecutionResult(
                content=f"sandbox tool timed out after {request.timeout_s:g}s",
                backend="workspace_sandbox",
                started=True,
                is_error=True,
                error_type="timeout",
                retryable=False,
                elapsed_ms=int((time.monotonic() - started_at) * 1000),
                container_id=container.id,
            )
        except (asyncio.CancelledError, Exception):
            await _complete_cleanup_before_cancelling(self.cancel(request.invocation_id))
            raise
        finally:
            self._active.pop(request.invocation_id, None)
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        stderr = stderr_raw.decode("utf-8", errors="replace")
        if process.returncode != 0:
            error_type = await self._container_failure_type(
                container,
                process.returncode,
            )
            await self.cleanup_run(request.run_id)
            return ToolExecutionResult(
                content=(
                    "sandbox exceeded its memory limit"
                    if error_type == "sandbox_oom"
                    else _redact(stderr or "sandbox worker exited unexpectedly")
                ),
                backend="workspace_sandbox",
                started=True,
                is_error=True,
                error_type=error_type,
                retryable=False,
                exit_code=process.returncode,
                stderr=_redact(stderr),
                elapsed_ms=elapsed_ms,
                container_id=container.id,
            )
        try:
            raw = json.loads(stdout_raw)
            if not isinstance(raw, dict):
                raise ValueError("worker response must be an object")
        except (json.JSONDecodeError, ValueError) as exc:
            await self.cleanup_run(request.run_id)
            return ToolExecutionResult(
                content=f"invalid sandbox worker response: {exc}",
                backend="workspace_sandbox",
                started=True,
                is_error=True,
                error_type="sandbox_lost",
                retryable=False,
                elapsed_ms=elapsed_ms,
                container_id=container.id,
            )
        return ToolExecutionResult(
            content=str(raw.get("content", "")),
            backend="workspace_sandbox",
            started=True,
            is_error=bool(raw.get("is_error", False)),
            error_type=_optional_string(raw.get("error_type")),
            retryable=False,
            exit_code=_optional_int(raw.get("exit_code")),
            stdout=str(raw.get("stdout", "")),
            stderr=str(raw.get("stderr", "")),
            truncated=bool(raw.get("truncated", False)),
            elapsed_ms=int(raw.get("elapsed_ms", elapsed_ms)),
            container_id=container.id,
        )

    async def _container_failure_type(
        self,
        container: _Container,
        returncode: int | None,
    ) -> str:
        code, stdout, _ = await self._command(
            [
                self._config.docker_binary,
                "inspect",
                "--format",
                "{{json .State}}",
                container.id,
            ],
            timeout_s=10.0,
        )
        if code == 0:
            try:
                state = json.loads(stdout)
            except json.JSONDecodeError:
                state = None
            if isinstance(state, dict) and state.get("OOMKilled") is True:
                return "sandbox_oom"
        # Docker exec conventionally reports a SIGKILL/OOM as 137.  Inspect is
        # authoritative when available; the exit code preserves the useful
        # failure semantic after an --rm container has already disappeared.
        if returncode in {137, -9}:
            return "sandbox_oom"
        return "sandbox_lost"

    async def cancel(self, invocation_id: str) -> None:
        active = self._active.get(invocation_id)
        if active is None:
            return
        process, run_id = active
        if process.returncode is None:
            process.kill()
            await process.communicate()
        await self.cleanup_run(run_id)

    async def cleanup_run(self, run_id: str) -> None:
        async with self._container_guard:
            container = self._containers.get(run_id)
            if container is not None:
                code, _, stderr = await self._command(
                    [self._config.docker_binary, "rm", "--force", container.id],
                    timeout_s=3.0,
                )
                if code == 0 or _is_missing_container(stderr):
                    self._containers.pop(run_id, None)
                else:
                    # Retain the mapping so daemon shutdown can retry cleanup.  Dropping
                    # it here would turn a transient Docker failure into an orphan that
                    # this runtime can no longer address.
                    log.warning(
                        "failed to remove sandbox container id=%s error=%s",
                        container.id,
                        stderr,
                    )

            uncertain_name = _container_name(run_id, self._instance_id)
            if uncertain_name in self._uncertain_container_names:
                await self._cleanup_uncertain_container(
                    uncertain_name,
                    grace_s=_UNCERTAIN_RUN_CLEANUP_GRACE_S,
                )
            if run_id in self._containers or uncertain_name in self._uncertain_container_names:
                raise RuntimeCleanupPending(run_id)

    def pending_cleanup_run_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._containers) | {
            self._uncertain_run_ids[name] for name in self._uncertain_container_names
            if name in self._uncertain_run_ids
        }))

    async def _cleanup_uncertain_container(
        self,
        name: str,
        *,
        grace_s: float = 0.0,
    ) -> None:
        """Reconcile an ambiguous ``docker run`` without deleting foreign containers."""
        self._uncertain_container_names.add(name)
        deadline = asyncio.get_running_loop().time() + max(0.0, grace_s)
        while True:
            code, stdout, stderr = await self._command(
                [
                    self._config.docker_binary,
                    "inspect",
                    "--format",
                    '{{ index .Config.Labels "com.tars-agent.instance" }}',
                    name,
                ],
                timeout_s=3.0,
            )
            if code == 0:
                owner = stdout.strip()
                if owner != self._instance_id:
                    log.error(
                        "refusing to remove foreign container name=%s owner=%s",
                        name,
                        owner or "<unlabelled>",
                    )
                    return
                remove_code, _, remove_stderr = await self._command(
                    [self._config.docker_binary, "rm", "--force", name],
                    timeout_s=3.0,
                )
                if remove_code == 0 or _is_missing_container(remove_stderr):
                    # Inspect proved that the delayed request created our labelled
                    # container. Once it is removed (or concurrently disappears),
                    # this exact run request cannot create it a second time.
                    self._uncertain_container_names.discard(name)
                    return
                log.warning(
                    "failed to remove uncertain sandbox container name=%s error=%s",
                    name,
                    remove_stderr,
                )
                return

            if not _is_missing_container(stderr):
                log.warning(
                    "failed to inspect uncertain sandbox container name=%s error=%s",
                    name,
                    stderr,
                )
                return
            if asyncio.get_running_loop().time() >= deadline:
                # Absence is not conclusive immediately after killing the Docker
                # CLI. Retain the tombstone so run and runtime cleanup can retry.
                return
            await asyncio.sleep(_UNCERTAIN_CONTAINER_POLL_S)

    async def _reap_current_instance(self) -> None:
        """Remove only containers labelled for this exact runtime instance."""
        code, stdout, stderr = await self._command(
            [
                self._config.docker_binary,
                "ps",
                "--all",
                "--quiet",
                "--filter",
                f"label=com.tars-agent.instance={self._instance_id}",
            ],
            timeout_s=3.0,
        )
        if code != 0:
            log.warning("failed to list sandbox instance containers error=%s", stderr)
            raise RuntimeCleanupPending("<runtime>")
        container_ids = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not container_ids:
            return
        remove_code, _, remove_stderr = await self._command(
            [self._config.docker_binary, "rm", "--force", *container_ids],
            timeout_s=3.0,
        )
        if remove_code != 0 and not _is_missing_container(remove_stderr):
            log.warning(
                "failed to reap sandbox instance containers ids=%s error=%s",
                ",".join(container_ids),
                remove_stderr,
            )
            raise RuntimeCleanupPending("<runtime>")
        removed = set(container_ids)
        for run_id, container in tuple(self._containers.items()):
            if container.id in removed:
                self._containers.pop(run_id, None)

    async def cleanup(self) -> None:
        # One finite sweep; daemon shutdown must surface unconfirmed ownership.
        failures: list[RuntimeCleanupPending] = []
        for invocation_id in tuple(self._active):
            try:
                await self.cancel(invocation_id)
            except RuntimeCleanupPending as exc:
                failures.append(exc)
        for run_id in tuple(self._containers):
            try:
                await self.cleanup_run(run_id)
            except RuntimeCleanupPending as exc:
                failures.append(exc)
        for name in tuple(self._uncertain_container_names):
            await self._cleanup_uncertain_container(
                name, grace_s=_UNCERTAIN_RUNTIME_CLEANUP_GRACE_S,
            )
        if self._container_start_attempted:
            await self._reap_current_instance()
        pending = self.pending_cleanup_run_ids()
        if pending or self._uncertain_container_names:
            raise RuntimeCleanupPending(pending[0] if pending else "<runtime>")
        if failures and self._active:
            raise failures[0]

    async def inspect_run(self, run_id: str) -> dict[str, Any] | None:
        container = self._containers.get(run_id)
        if container is None:
            return None
        code, stdout, _ = await self._command(
            [self._config.docker_binary, "inspect", container.id],
            timeout_s=10.0,
        )
        if code != 0:
            return None
        parsed = json.loads(stdout)
        return parsed[0] if isinstance(parsed, list) and parsed else None

    @staticmethod
    async def _command(
        args: list[str],
        *,
        timeout_s: float | None = None,
    ) -> tuple[int, str, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return 127, "", str(exc)
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_s,
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            return 124, "", f"command timed out after {timeout_s:g}s"
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.communicate()
            raise
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )


class _NullAsyncLock:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


def _container_name(run_id: str, instance_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:20]
    return f"tars-{instance_id}-{digest}"


def _is_missing_container(stderr: str) -> bool:
    normalized = stderr.casefold()
    return "no such container" in normalized or "no such object" in normalized


async def _complete_cleanup_before_cancelling(coro: Any) -> None:
    """Finish compensating cleanup before propagating task cancellation."""
    operation = asyncio.create_task(coro)
    try:
        await asyncio.shield(operation)
    except asyncio.CancelledError:
        await operation
        raise


def _host_user() -> str | None:
    if os.name == "nt":
        return None
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return None
    return f"{getuid()}:{getgid()}"


def _redact(value: str) -> str:
    return value[:4096]


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


__all__ = ["DockerRuntime"]
