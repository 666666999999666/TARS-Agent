from __future__ import annotations

import argparse
import asyncio
import json
import locale
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Any

_DEFAULT_OUTPUT_BYTES = 64 * 1024
_MAX_WRITE_BYTES = 1024 * 1024
_MAX_LIST_ENTRIES = 200
_MAX_LIST_DEPTH = 4
# Container environment fallback only; this does not create a temporary file.
_CONTAINER_HOME_FALLBACK = "/tmp/tars-home"  # nosec B108


class SandboxPolicyError(PermissionError):
    pass


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_workspace_path(
    root: Path,
    raw_path: str,
    *,
    write: bool = False,
) -> Path:
    """Resolve a path and reject absolute/traversal/symlink workspace escapes."""
    workspace = root.resolve(strict=True)
    # Device namespaces and reserved components are unsafe even when parsed on POSIX.
    normalized = raw_path.replace("\\", "/")
    if (normalized.startswith(("//", "/??/")) or "\x00" in raw_path
            or PureWindowsPath(raw_path).is_reserved()):
        raise SandboxPolicyError("device and network paths are not allowed")
    for component in normalized.split("/"):
        if re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", component):
            raise SandboxPolicyError("reserved device names are not allowed")
        if ":" in component and not re.fullmatch(r"[A-Za-z]:", component):
            raise SandboxPolicyError("alternate data streams are not allowed")
    supplied = Path(raw_path)
    candidate = supplied if supplied.is_absolute() else workspace / supplied
    lexical = candidate.resolve(strict=False)
    if not _inside(workspace, lexical):
        raise SandboxPolicyError(f"path escapes workspace: {raw_path}")
    resolved = candidate.resolve(strict=not write)
    if not _inside(workspace, resolved):
        raise SandboxPolicyError(f"path escapes workspace: {raw_path}")
    return resolved


def _result(
    content: str,
    *,
    is_error: bool = False,
    error_type: str | None = None,
    retryable: bool = False,
    exit_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
    truncated: bool = False,
) -> dict[str, object]:
    return {
        "content": content,
        "is_error": is_error,
        "error_type": error_type,
        "retryable": retryable,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": truncated,
    }


async def _bash(
    params: dict[str, object],
    workspace: Path,
    timeout_s: float,
    *,
    sandboxed: bool,
    output_limit_bytes: int,
) -> dict[str, object]:
    command = str(params.get("command", ""))
    requested_timeout = params.get("timeout", timeout_s)
    timeout = min(
        float(requested_timeout) if isinstance(requested_timeout, (int, float)) else timeout_s,
        timeout_s,
    )
    if timeout <= 0:
        return _result("invalid timeout", is_error=True, error_type="schema_error")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "HOME": os.environ.get("HOME", _CONTAINER_HOME_FALLBACK),
    }
    if os.name == "nt":
        for name in ("SystemRoot", "ComSpec", "PATHEXT", "TEMP", "TMP"):
            value = os.environ.get(name)
            if value is not None:
                env[name] = value
    env["PYTHONIOENCODING"] = "utf-8"
    if sandboxed:
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    elif os.name == "nt":
        # The helper owns a kill-on-close Job before it starts the shell.
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", str(Path(__file__).resolve()), "--host-shell", command,
            cwd=workspace, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    output_tasks = (
        asyncio.create_task(_read_bounded(proc.stdout, output_limit_bytes)),
        asyncio.create_task(_read_bounded(proc.stderr, output_limit_bytes)),
    )
    try:
        async with asyncio.timeout(timeout):
            (stdout_raw, stdout_truncated), (stderr_raw, stderr_truncated) = (
                await asyncio.gather(*output_tasks)
            )
            await proc.wait()
    except TimeoutError:
        await _terminate_process(proc)
        await asyncio.gather(*output_tasks, return_exceptions=True)
        return _result(
            f"[timeout after {timeout:g}s]",
            is_error=True,
            error_type="timeout",
            retryable=False,
        )
    except asyncio.CancelledError:
        await _terminate_process(proc)
        await asyncio.gather(*output_tasks, return_exceptions=True)
        raise
    combined = stdout_raw + stderr_raw
    truncated = stdout_truncated or stderr_truncated or len(combined) > output_limit_bytes
    visible = _decode_output(combined[:output_limit_bytes], sandboxed=sandboxed)
    if truncated:
        visible += "\n[truncated]"
    stdout_visible = stdout_raw[:output_limit_bytes]
    stderr_budget = max(output_limit_bytes - len(stdout_visible), 0)
    stdout = _decode_output(stdout_visible, sandboxed=sandboxed)
    stderr = _decode_output(stderr_raw[:stderr_budget], sandboxed=sandboxed)
    exit_code = proc.returncode or 0
    if exit_code:
        return _result(
            f"[exit {exit_code}]\n{visible}",
            is_error=True,
            error_type="runtime_error",
            retryable=False,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=truncated,
        )
    return _result(
        visible or "[no output]",
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        truncated=truncated,
    )


async def _read_bounded(
    stream: asyncio.StreamReader | None,
    limit: int,
) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    captured = bytearray()
    total = 0
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        total += len(chunk)
        remaining = limit - len(captured)
        if remaining > 0:
            captured.extend(chunk[:remaining])
    return bytes(captured), total > limit


def _decode_output(raw: bytes, *, sandboxed: bool) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        encoding = (locale.getpreferredencoding(False)
                    if os.name == "nt" and not sandboxed else "utf-8")
        return raw.decode(encoding, errors="replace")


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    # A POSIX shell may have exited while descendants still own its pipes.
    if os.name != "nt" and isinstance(getattr(proc, "pid", None), int):
        try:
            kill_group = getattr(os, "killpg")
            kill_group(proc.pid, getattr(signal, "SIGKILL"))
        except ProcessLookupError:
            pass
    elif proc.returncode is None:
        # Closing the helper also closes its Job and kills all descendants.
        proc.kill()
    try:
        async with asyncio.timeout(5.0):
            await proc.wait()
    except TimeoutError:
        if proc.returncode is None:
            proc.kill()
        raise RuntimeError("owned shell process did not terminate within 5 seconds")


def _windows_host_shell(command: str) -> int:
    """Start a Windows shell only after this helper belongs to an owned Job."""
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
    ]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        kernel.CloseHandle(job)
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        kernel.CloseHandle(job)
        raise ctypes.WinError(ctypes.get_last_error())
    # The Job handle intentionally stays open until helper process exit; Python
    # descendants do not inherit it, so no child can prolong the ownership scope.
    # Deliberate shell tool: permission policy and worker isolation precede this call.
    return subprocess.call(command, shell=True)  # nosec B602


def _check_regular_file_links(details: os.stat_result) -> None:
    if not stat.S_ISREG(details.st_mode):
        raise SandboxPolicyError("file operation requires a regular file")
    if details.st_nlink > 1:
        raise SandboxPolicyError("files with multiple hard links are not allowed")


def _open_regular_file(path: Path, *, write: bool = False) -> int:
    # Check both the named entry and the opened inode. Never use O_TRUNC before
    # the descriptor check: an entry can change after path validation.
    try:
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        if not write:
            raise
    else:
        _check_regular_file_links(details)
    flags = os.O_WRONLY | os.O_CREAT if write else os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o666)
    try:
        _check_regular_file_links(os.fstat(descriptor))
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_file(
    params: dict[str, object],
    workspace: Path,
    output_limit_bytes: int,
) -> dict[str, object]:
    raw_path = str(params.get("path", ""))
    path = resolve_workspace_path(workspace, raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {raw_path}")
    with os.fdopen(_open_regular_file(path), "rb") as handle:
        raw = handle.read(output_limit_bytes + 1)
    truncated = len(raw) > output_limit_bytes
    content = raw[:output_limit_bytes].decode("utf-8", errors="replace")
    if truncated:
        content += "\n[truncated]"
    return _result(content, truncated=truncated)


def _write_file(params: dict[str, object], workspace: Path) -> dict[str, object]:
    raw_path = str(params.get("path", ""))
    content = str(params.get("content", ""))
    encoded = content.encode("utf-8")
    if len(encoded) > _MAX_WRITE_BYTES:
        return _result(
            f"content too large: {len(encoded)} bytes (limit 1 MB)",
            is_error=True,
            error_type="sandbox_policy_denied",
            retryable=False,
        )
    resolve_workspace_path(workspace, raw_path, write=True)
    candidate = Path(raw_path)
    parent_raw = str(candidate.parent) if str(candidate.parent) else "."
    workspace_resolved = workspace.resolve(strict=True)
    parent_candidate = Path(parent_raw)
    if not parent_candidate.is_absolute():
        parent_candidate = workspace_resolved / parent_candidate
    parent = parent_candidate.resolve(strict=False)
    if not _inside(workspace_resolved, parent):
        raise SandboxPolicyError(f"path escapes workspace: {raw_path}")
    parent.mkdir(parents=True, exist_ok=True)
    path = resolve_workspace_path(workspace, raw_path, write=True)
    with os.fdopen(_open_regular_file(path, write=True), "w", encoding="utf-8") as handle:
        handle.truncate(0)
        handle.write(content)
    return _result(f"wrote {len(encoded)} bytes to {raw_path}")


def _list_dir(params: dict[str, object], workspace: Path) -> dict[str, object]:
    raw_path = str(params.get("path", "."))
    max_depth_value = params.get("max_depth", 2)
    max_depth = int(max_depth_value) if isinstance(max_depth_value, int) else 2
    if not 1 <= max_depth <= _MAX_LIST_DEPTH:
        return _result("invalid max_depth", is_error=True, error_type="schema_error")
    root = resolve_workspace_path(workspace, raw_path)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {raw_path}")
    lines = [f"{raw_path}/"]
    count = 0

    def walk(directory: Path, depth: int, prefix: str) -> None:
        nonlocal count
        entries = sorted(directory.iterdir(), key=lambda entry: (entry.is_file(), entry.name))
        for index, entry in enumerate(entries):
            if count >= _MAX_LIST_ENTRIES:
                lines.append(f"{prefix}... (truncated)")
                return
            resolved = entry.resolve(strict=True)
            if not _inside(workspace.resolve(strict=True), resolved):
                raise SandboxPolicyError(f"symlink escapes workspace: {entry}")
            last = index == len(entries) - 1
            connector = "└── " if last else "├── "
            is_dir = entry.is_dir()
            lines.append(f"{prefix}{connector}{entry.name}{'/' if is_dir else ''}")
            count += 1
            if is_dir and not entry.is_symlink() and depth < max_depth:
                walk(entry, depth + 1, prefix + ("    " if last else "│   "))

    walk(root, 1, "")
    return _result("\n".join(lines))


async def execute_payload(
    payload: dict[str, Any],
    workspace_root: Path,
    *,
    sandboxed: bool,
) -> dict[str, object]:
    started = time.monotonic()
    tool_name = str(payload.get("tool_name", ""))
    params = payload.get("params", {})
    if not isinstance(params, dict):
        return _result("params must be an object", is_error=True, error_type="schema_error")
    timeout_s = float(payload.get("timeout_s", 120.0))
    output_limit_value = payload.get("output_limit_bytes", _DEFAULT_OUTPUT_BYTES)
    output_limit_bytes = (
        output_limit_value
        if isinstance(output_limit_value, int) and output_limit_value > 0
        else _DEFAULT_OUTPUT_BYTES
    )
    try:
        if tool_name == "bash":
            result = await _bash(
                params,
                workspace_root,
                timeout_s,
                sandboxed=sandboxed,
                output_limit_bytes=output_limit_bytes,
            )
        elif tool_name == "read_file":
            result = _read_file(params, workspace_root, output_limit_bytes)
        elif tool_name == "write_file":
            result = _write_file(params, workspace_root)
        elif tool_name == "list_dir":
            result = _list_dir(params, workspace_root)
        else:
            result = _result(
                f"unsupported workspace tool: {tool_name}",
                is_error=True,
                error_type="sandbox_policy_denied",
            )
    except SandboxPolicyError as exc:
        result = _result(
            str(exc),
            is_error=True,
            error_type="sandbox_policy_denied",
            retryable=False,
        )
    except FileNotFoundError as exc:
        result = _result(str(exc), is_error=True, error_type="not_found", retryable=False)
    except (OSError, ValueError) as exc:
        result = _result(str(exc), is_error=True, error_type="runtime_error", retryable=False)
    result["elapsed_ms"] = int((time.monotonic() - started) * 1000)
    return result


async def _serve(workspace: Path) -> int:
    raw = await asyncio.to_thread(sys.stdin.buffer.readline)
    if not raw:
        return 2
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("request must be a JSON object")
        response = await execute_payload(payload, workspace, sandboxed=True)
    except (json.JSONDecodeError, ValueError) as exc:
        response = _result(str(exc), is_error=True, error_type="schema_error")
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-shell")
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    args = parser.parse_args()
    if args.host_shell is not None:
        if os.name != "nt":
            raise SystemExit("host-shell helper is Windows-only")
        raise SystemExit(_windows_host_shell(args.host_shell))
    raise SystemExit(asyncio.run(_serve(args.workspace)))


if __name__ == "__main__":
    main()


__all__ = ["SandboxPolicyError", "execute_payload", "resolve_workspace_path"]
