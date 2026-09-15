from __future__ import annotations

from tars_agent.core.app import _config_log_summary
from tars_agent.core.config import McpServerConfig, TarsConfig


def test_config_log_summary_never_contains_mcp_secrets() -> None:
    config = TarsConfig()
    config.mcp.servers = [
        McpServerConfig(
            name="private",
            transport="streamable_http",
            trusted=True,
            url="https://user:URL-SECRET@example.test/mcp",
            headers={"Authorization": "Bearer HEADER-SECRET"},
            env={"MCP_TOKEN": "ENV-SECRET"},
        )
    ]

    rendered = repr(_config_log_summary(config))

    assert "private" in rendered
    assert "streamable_http" in rendered
    assert "URL-SECRET" not in rendered
    assert "HEADER-SECRET" not in rendered
    assert "ENV-SECRET" not in rendered
