from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from tars_agent.core.config import McpServerConfig
from tars_agent.core.control import read_control_file
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import ToolCallBlock
from tars_agent.core.mcp.client import McpClient
from tars_agent.core.mcp.tool import McpTool
from tars_agent.core.tools.invocation import invoke_tool
from tars_agent.core.tools.registry import ToolRegistry


def _unused_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_port(port: int) -> None:
    for _ in range(100):
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.05)
            continue
        writer.close()
        await writer.wait_closed()
        del reader
        return
    raise AssertionError(f"MCP HTTP test server did not listen on port {port}")


async def test_official_sdk_stdio_discovery_and_call() -> None:
    server = Path(__file__).with_name("mcp_stdio_server.py")
    client = McpClient(
        McpServerConfig(
            name="stdio",
            transport="stdio",
            trusted=True,
            command=sys.executable,
            args=[str(server)],
            connect_timeout_s=10,
            tool_timeout_s=10,
        )
    )
    try:
        await client.connect()
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == {"add", "count_side_effect", "get_count"}
        result = await client.call_tool("add", {"a": 2, "b": 3})
        assert not result.is_error
        assert '"sum": 5' in result.content
        assert client.protocol_version is not None
    finally:
        await client.close()


async def test_official_sdk_side_effect_failure_and_timeout_execute_once() -> None:
    server = Path(__file__).with_name("mcp_stdio_server.py")
    client = McpClient(
        McpServerConfig(
            name="counter",
            transport="stdio",
            trusted=True,
            command=sys.executable,
            args=[str(server)],
            connect_timeout_s=10,
            tool_timeout_s=0.2,
        )
    )
    try:
        await client.connect()
        definitions = {tool.name: tool for tool in await client.list_tools()}
        registry = ToolRegistry()
        registry.register(McpTool(client, "counter", definitions["count_side_effect"]))

        for index, mode in enumerate(("fail", "timeout"), start=1):
            result = await invoke_tool(
                registry,
                ToolCallBlock(
                    id=f"tool-{index}",
                    name="counter__count_side_effect",
                    input={"mode": mode},
                ),
                EventBus(),
                run_id=f"run-{index}",
                timeout=2,
            )
            assert result.is_error
            assert result.retryable is False

            count = await client.call_tool("get_count", {})
            assert f'"calls": {index}' in count.content
    finally:
        await client.close()


async def test_official_sdk_streamable_http_discovery_and_call() -> None:
    server = Path(__file__).with_name("mcp_http_server.py")
    port = _unused_loopback_port()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(server),
        str(port),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    client = McpClient(
        McpServerConfig(
            name="http",
            transport="streamable_http",
            trusted=True,
            url=f"http://127.0.0.1:{port}/mcp",
            connect_timeout_s=10,
            tool_timeout_s=10,
        )
    )
    try:
        await _wait_for_port(port)
        await client.connect()
        tools = await client.list_tools()
        assert [tool.name for tool in tools] == ["multiply"]
        result = await client.call_tool("multiply", {"a": 6, "b": 7})
        assert not result.is_error
        assert '"product": 42' in result.content
    finally:
        await client.close()
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.communicate(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.communicate()


async def test_core_mcp_status_reports_live_and_failed_stdio_servers(
    tmp_path: Path,
) -> None:
    server = Path(__file__).with_name("mcp_stdio_server.py")
    port = _unused_loopback_port()
    runtime_home = tmp_path / "tars-home"
    runtime_home.mkdir()
    config_path = runtime_home / "config.toml"
    config_path.write_text(
        "\n".join(
            (
                "[[mcp.servers]]",
                'name = "live"',
                'transport = "stdio"',
                "trusted = true",
                f"command = {json.dumps(sys.executable)}",
                f"args = [{json.dumps(str(server))}]",
                "connect_timeout_s = 5",
                "tool_timeout_s = 5",
                "",
                "[[mcp.servers]]",
                'name = "broken"',
                'transport = "stdio"',
                "trusted = true",
                f"command = {json.dumps(str(tmp_path / 'missing-mcp-server'))}",
                "connect_timeout_s = 1",
                "tool_timeout_s = 1",
            )
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.update(
        {
            "TARS_CONFIG": str(config_path),
            "TARS_PORT": str(port),
            "TARS_HOME": str(runtime_home),
            "TARS_LOG_FILE": "",
            "TARS_LOG_LEVEL": "WARNING",
            "TARS_TRACE_ENABLED": "false",
            "TARS_SANDBOX_MODE": "preferred",
        }
    )
    captured_path = runtime_home / "core-test-output.log"
    captured = captured_path.open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "tars_agent.core"],
        env=env,
        stdout=captured,
        stderr=subprocess.STDOUT,
    )
    writer: asyncio.StreamWriter | None = None
    try:
        await _wait_for_port(port)
        if process.poll() is not None:
            raise AssertionError(
                "Core exited before mcp.status\n"
                + captured_path.read_text(encoding="utf-8", errors="replace")
            )
        reader, writer = await asyncio.open_connection("127.0.0.1", port)

        async def call(method: str, request_id: str) -> dict[str, object]:
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": {},
            }
            if method == "core.shutdown":
                control = read_control_file(
                    Path(env["TARS_HOME"]) / "control" / f"tars-core-{port}.json",
                )
                assert control is not None
                request["params"] = {"token": control.token}
            assert writer is not None
            writer.write((json.dumps(request) + "\n").encode())
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            response = json.loads(line)
            assert response["id"] == request_id
            assert "error" not in response
            return response["result"]

        result = await call("mcp.status", "mcp-status")
        statuses = {item["name"]: item for item in result["servers"]}
        assert statuses["live"]["status"] == "connected"
        assert statuses["live"]["transport"] == "stdio"
        assert statuses["live"]["tool_count"] == 3
        assert statuses["live"]["protocol_version"]
        assert statuses["broken"]["status"] == "failed"
        assert statuses["broken"]["tool_count"] == 0
        assert statuses["broken"]["error"]

        assert await call("core.shutdown", "shutdown") == {"ok": True}
        try:
            await asyncio.to_thread(process.wait, 15)
        except subprocess.TimeoutExpired as exc:
            raise AssertionError(captured_path.read_text(encoding="utf-8", errors="replace")) from exc
        assert process.returncode == 0, captured_path.read_text(encoding="utf-8", errors="replace")
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=1)
            except TimeoutError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 2)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait, 2)
        captured.close()
