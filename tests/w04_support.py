"""Test-only identity checks and evidence for the explicitly authorized W04 run."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "build/w04/20260917-003750"
ISOLATED = Path(os.environ.get("W04_ISOLATED_DIR", str(PREP / "isolated"))).resolve()
EXECUTION = Path(os.environ.get("W04_EXECUTION_DIR", str(PREP / "execution-20260917-103341"))).resolve()
if not ISOLATED.is_relative_to(PREP / "isolated") or not EXECUTION.is_relative_to(PREP):
    raise ValueError("W04 evidence or isolation directory escapes the authorized root")
SCOPES = {"docker-lifecycle", "f05-observe/A", "f05-observe/B", "f05-reclaim/A", "f05-reclaim/B"}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": time.time(), **data}, ensure_ascii=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scope_path(name):
    if name not in SCOPES:
        raise ValueError("scope is outside this W04 authorization")
    path = (ISOLATED / name).resolve(strict=True)
    if not path.is_relative_to(ISOLATED.resolve()):
        raise ValueError("scope escapes the authorized root")
    return path


def command(args, log, *, timeout=40):
    """Bound the wait; never kill on timeout or treat a failed query as absence."""
    started = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    append(log, {"phase": "command_started", "args": args, "pid": proc.pid})
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        append(log, {"phase": "query_timeout_stop", "args": args, "pid": proc.pid,
                     "forced_cleanup": False})
        raise
    result = {"args": args, "pid": proc.pid, "started": started, "exit_code": proc.returncode,
              "stdout": out.decode("utf-8", errors="replace"),
              "stderr": err.decode("utf-8", errors="replace")}
    recorded = dict(result)
    if args[0] == "docker" and "inspect" in args and "--format" not in args:
        recorded["stdout"] = "[inspect payload retained only through selected identity fields]"
    append(log, recorded)
    return result


def docker(args, log, *, timeout=40):
    return command(["docker", *args], log, timeout=timeout)


def checked(result):
    if result["exit_code"] != 0:
        raise RuntimeError(f"command failed: {result['args']!r}: {result['stderr']}")
    return result["stdout"]


def process_identity(pid):
    # Numeric PID only; no untrusted command text enters PowerShell.
    code = ("[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); "
            f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}' | "
            "Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine,"
            "@{n='Created';e={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress")
    result = command(["powershell.exe", "-NoProfile", "-Command", code],
                     EXECUTION / "process-queries.jsonl")
    raw = checked(result).strip()
    if not raw:
        raise RuntimeError(f"process {pid} absent; not an identity match")
    data = json.loads(raw)
    if not data.get("ExecutablePath") or not data.get("CommandLine") or not data.get("Created"):
        raise RuntimeError("process identity incomplete; stop")
    return data


def verify_process(expected):
    actual = process_identity(expected["ProcessId"])
    for key in ("ProcessId", "Created", "ExecutablePath", "CommandLine"):
        if actual[key] != expected[key]:
            raise RuntimeError(f"process identity mismatch: {key}")
    append(EXECUTION / "identity-checks.jsonl", {"kind": "process", "identity": actual})
    return actual


def kill_verified_core(expected):
    """Terminate the verified Core handle only, never its parent or a process-name group."""
    import ctypes
    from ctypes import wintypes

    verify_process(expected)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000 | 0x0001 | 0x00100000, False, expected["ProcessId"])
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        values = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            raise ctypes.WinError(ctypes.get_last_error())
        created = ((values[0].dwHighDateTime << 32) | values[0].dwLowDateTime) / 10_000_000 - 11644473600
        expected_time = datetime.fromisoformat(expected["Created"].replace("Z", "+00:00")).timestamp()
        if abs(created - expected_time) > 0.000002:
            raise RuntimeError("opened process handle has a different creation time; no kill")
        if not kernel.TerminateProcess(handle, 99):
            raise ctypes.WinError(ctypes.get_last_error())
        if kernel.WaitForSingleObject(handle, 10000) != 0:
            raise RuntimeError("verified Core did not exit within 10 seconds")
    finally:
        kernel.CloseHandle(handle)


def container_ids(log):
    return checked(docker(["ps", "--all", "--quiet", "--no-trunc"], log)).splitlines()


def inspect_container(cid, log):
    if len(cid) != 64 or any(char not in "0123456789abcdef" for char in cid):
        raise ValueError("a full container ID is required")
    if cid not in container_ids(log):
        return None  # Only a successful full listing establishes absence.
    value = json.loads(checked(docker(["inspect", cid], log)))[0]
    # Do not persist environment variables or Docker proxy configuration.
    return {"id": value["Id"], "name": value["Name"], "image": value["Image"],
            "created": value["Created"], "started": value["State"]["StartedAt"],
            "state": value["State"], "labels": value["Config"]["Labels"],
            "mounts": value["Mounts"], "network": value["HostConfig"]["NetworkMode"],
            "read_only": value["HostConfig"]["ReadonlyRootfs"]}


def mount_path(value):
    # Docker Desktop may normalize a Windows bind source into /run/desktop/mnt/host/c/...
    text = str(value).replace("\\", "/").rstrip("/").lower()
    prefix = "/run/desktop/mnt/host/"
    if text.startswith(prefix):
        text = text[len(prefix)] + ":" + text[len(prefix) + 1:]
    return text


def verify_container(expected, log, *, allow_absent=False):
    current = inspect_container(expected["id"], log)
    if current is None:
        if allow_absent:
            return None
        raise RuntimeError("container already absent; no operation authorized by this check")
    for key in ("id", "image", "created", "labels", "mounts"):
        if current[key] != expected[key]:
            raise RuntimeError(f"container identity mismatch: {key}")
    append(log, {"phase": "container_identity_verified", "container": current})
    return current


def register_container(cid, *, workspace, instance, run, image, log):
    value = inspect_container(cid, log)
    if value is None:
        raise RuntimeError("new container was not found")
    if value["image"] != image or value["labels"].get("com.tars-agent.instance") != instance:
        raise RuntimeError("image/instance mismatch")
    if value["labels"].get("com.tars-agent.run") != run or value["labels"].get("com.tars-agent.sandbox") != "true":
        raise RuntimeError("run/sandbox label mismatch")
    if len(value["mounts"]) != 1:
        raise RuntimeError("expected exactly one authorized workspace bind mount")
    mount = value["mounts"][0]
    if mount["Type"] != "bind" or mount["Destination"] != "/workspace" or mount_path(mount["Source"]) != mount_path(workspace):
        raise RuntimeError("workspace mount mismatch")
    if value["network"] != "none" or not value["read_only"]:
        raise RuntimeError("sandbox controls changed")
    return value


def remove_exact(expected, log, *, reason):
    current = verify_container(expected, log, allow_absent=True)
    if current is None:
        return
    append(log, {"phase": "manual_removal", "reason": reason, "id": expected["id"]})
    checked(docker(["rm", "--force", expected["id"]], log))
    if inspect_container(expected["id"], log) is not None:
        raise RuntimeError("container remains after removal")


def clean_environment(home, port, image):
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE",
               "APPDATA", "LOCALAPPDATA", "COMSPEC"}
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(TARS_HOME=str(home), TARS_CONFIG=str(home / "config.toml"), TARS_PORT=str(port),
               TARS_SANDBOX_IMAGE=image, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONUNBUFFERED="1", W04_ISOLATED_DIR=str(ISOLATED), W04_EXECUTION_DIR=str(EXECUTION))
    return env
