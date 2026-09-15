from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from tars_agent.core.paths import tars_home

CONTROL_FILE = tars_home() / "control" / "tars-core-7437.json"
CORE_LAUNCH_ID_ENV = "TARS_CORE_LAUNCH_ID"


def control_file_for(port: int) -> Path:
    return tars_home() / "control" / f"tars-core-{port}.json"


@dataclass(frozen=True)
class DaemonControl:
    pid: int
    host: str
    port: int
    token: str
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
