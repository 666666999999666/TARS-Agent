from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import BinaryIO

from tars_agent.core.paths import tars_home

CONTROL_FILE = tars_home() / "control" / "tars-core-7437.json"
CORE_LAUNCH_ID_ENV = "TARS_CORE_LAUNCH_ID"


class CoreHomeLock:
    """Hold one OS lock for the entire lifetime of a Core using this state root."""

    def __init__(self, home: Path) -> None:
        self.home = home.expanduser().resolve()
        self._handle: BinaryIO | None = None

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def acquire(self) -> None:
        if self.acquired:
            raise RuntimeError("Core HOME lock is already held by this owner")
        path = self.home / "control" / "core.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        try:
            os.set_inheritable(handle.fileno(), False)
            # Windows byte-range locks require a byte; never truncate or unlink
            # the file, which would allow another process to lock a different inode.
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("Core HOME is in use or its exclusive lock is unavailable") from exc
        except BaseException:
            handle.close()
            raise
        self._handle = handle

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def control_file_for(port: int) -> Path:
    return tars_home() / "control" / f"tars-core-{port}.json"


@dataclass(frozen=True)
class DaemonControl:
    pid: int
    host: str
    port: int
    token: str = field(repr=False)
    launch_id: str | None = None


def write_control_file(
    control: DaemonControl,
    path: Path = CONTROL_FILE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{control.pid}.tmp")
    try:
        temp_path.write_text(
            json.dumps(asdict(control), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            temp_path.chmod(0o600)
        except OSError:
            pass
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_control_file(path: Path = CONTROL_FILE) -> DaemonControl | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("control file must contain a JSON object")
        launch_id = raw.get("launch_id")
        if launch_id is not None and not isinstance(launch_id, str):
            raise TypeError("launch_id must be a string or null")
        return DaemonControl(
            pid=int(raw["pid"]),
            host=str(raw["host"]),
            port=int(raw["port"]),
            token=str(raw["token"]),
            launch_id=launch_id,
        )
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def remove_control_file(
    expected_token: str,
    path: Path = CONTROL_FILE,
) -> None:
    current = read_control_file(path)
    if current is not None and current.token == expected_token:
        path.unlink(missing_ok=True)
