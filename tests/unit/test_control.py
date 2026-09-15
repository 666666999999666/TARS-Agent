from __future__ import annotations

import json
from pathlib import Path

from tars_agent.core.control import (
    DaemonControl,
    read_control_file,
    remove_control_file,
    write_control_file,
)


def test_control_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    expected = DaemonControl(
        pid=123,
        host="127.0.0.1",
        port=55123,
        token="secret",
        launch_id="launch-123",
    )

    write_control_file(expected, path)

    assert read_control_file(path) == expected


def test_control_file_is_only_removed_by_owner_token(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    expected = DaemonControl(pid=123, host="127.0.0.1", port=55123, token="owner")
    write_control_file(expected, path)

    remove_control_file("other", path)
    assert path.exists()

    remove_control_file("owner", path)
    assert not path.exists()


def test_invalid_control_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text("not-json", encoding="utf-8")

    assert read_control_file(path) is None

    path.write_text("[]", encoding="utf-8")
    assert read_control_file(path) is None

    path.write_text(
        json.dumps(
            {
                "pid": 123,
                "host": "127.0.0.1",
                "port": 55123,
                "token": "token",
                "launch_id": 123,
            }
        ),
        encoding="utf-8",
    )
    assert read_control_file(path) is None


def test_control_file_without_launch_id_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(
        json.dumps(
            {"pid": 123, "host": "127.0.0.1", "port": 55123, "token": "legacy"}
        ),
        encoding="utf-8",
    )

    assert read_control_file(path) == DaemonControl(
        pid=123,
        host="127.0.0.1",
        port=55123,
        token="legacy",
        launch_id=None,
    )
