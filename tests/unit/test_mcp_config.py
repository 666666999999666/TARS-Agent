from __future__ import annotations

from pathlib import Path

import pytest

from tars_agent.core.config import get_config


def _load_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
):
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARS_HOME", str(tmp_path))
    monkeypatch.setenv("TARS_CONFIG", str(path))
    return get_config()


def test_stdio_and_streamable_http_config_are_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config(
        tmp_path,
        monkeypatch,
        """
[[mcp.servers]]
name = "local"
transport = "stdio"
trusted = true
command = "python"
args = ["server.py"]
env = { EXPLICIT = "yes" }

[[mcp.servers]]
name = "remote"
transport = "streamable_http"
trusted = true
url = "http://127.0.0.1:8000/mcp"
headers = { Authorization = "${MCP_TOKEN}" }
connect_timeout_s = 3
tool_timeout_s = 9
""",
    )
    assert [server.transport for server in config.mcp.servers] == [
        "stdio",
        "streamable_http",
    ]
    assert config.mcp.servers[0].env == {"EXPLICIT": "yes"}
    assert config.mcp.servers[1].headers == {"Authorization": "${MCP_TOKEN}"}
    assert config.mcp.servers[1].tool_timeout_s == 9


def test_legacy_tcp_transport_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit, match="stdio.*streamable_http"):
        _load_config(
            tmp_path,
            monkeypatch,
            """
[[mcp.servers]]
name = "legacy"
transport = "tcp"
trusted = true
host = "127.0.0.1"
port = 9999
""",
        )


def test_mcp_server_requires_explicit_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit, match="trusted=true"):
        _load_config(
            tmp_path,
            monkeypatch,
            """
[[mcp.servers]]
name = "local"
transport = "stdio"
command = "python"
""",
        )
