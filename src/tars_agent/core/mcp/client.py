from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

from tars_agent.core.config import McpServerConfig

if TYPE_CHECKING:
    import httpx2
    from mcp import Client


class McpServerUnavailableError(Exception):
    pass


class McpToolError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class McpToolDef:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class McpCallResult:
    content: str
    is_error: bool


class McpClient:
    """Lifecycle wrapper around the official MCP Python SDK v2 client."""

    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._client: Client | None = None
        self._http_client: httpx2.AsyncClient | None = None
        self.health_status = "disconnected"
        self.last_error: str | None = None
        self._owner: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        self._close_error: Exception | None = None

    async def connect(self) -> None:
        if self._owner is not None or self._client is not None:
            raise RuntimeError("MCP client is already connected")
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._stop = asyncio.Event()
        self._close_error = None
        self._owner = asyncio.create_task(
            self._own_connection(ready, self._stop), name=f"tars-mcp:{self._config.name}"
        )
        try:
            await asyncio.shield(ready)
        except BaseException:
            self._stop.set()
            await asyncio.shield(self._owner)
            self._owner = None
            # Retrieve a concurrently raised handshake error after caller cancellation.
            if ready.done() and not ready.cancelled():
                ready.exception()
            raise

    async def _own_connection(
        self, ready: asyncio.Future[None], stop: asyncio.Event
    ) -> None:
        """AnyIO transport scopes must enter and exit in the same persistent task."""
        failed = False
        try:
            await self._connect_in_owner()
            ready.set_result(None)
            await stop.wait()
        except BaseException as exc:
            failed = True
            if not ready.done():
                ready.set_exception(McpServerUnavailableError(str(exc)))
            else:
                self._record_failure("MCP transport stopped unexpectedly")
        finally:
            try:
                await self._close_in_owner()
            except Exception as exc:
                self._close_error = exc
                failed = True
            if failed and not stop.is_set():
                self.health_status = "failed"
                self.last_error = "MCP transport failed; restart the server to reconnect"

    async def _connect_in_owner(self) -> None:
        if self._client is not None:
            raise RuntimeError("MCP client is already connected")
        client: Client | None = None
        try:
            if not self._config.trusted:
                raise ValueError("MCP server requires trusted configuration")
            import httpx2
            from mcp import Client
            from mcp.client.stdio import StdioServerParameters, stdio_client
            from mcp.client.streamable_http import streamable_http_client

            if self._config.transport == "stdio":
                if not self._config.command:
                    raise ValueError("stdio transport requires command")
                stdio_env = _stdio_environment(self._config.env)
                executable = _resolve_stdio_command(
                    self._config.command, environ=stdio_env,
                    cwd=Path.cwd(),
                )
                parameters = StdioServerParameters(
                    command=executable,
                    args=list(self._config.args),
                    env=stdio_env,
                    cwd=self._config.cwd or None,
                )
                client = Client(
                    stdio_client(parameters),
                    read_timeout_seconds=self._config.tool_timeout_s,
                )
            elif self._config.transport == "streamable_http":
                if not self._config.url:
                    raise ValueError("streamable_http transport requires url")
                headers = _expand_headers(self._config.headers)
                self._http_client = httpx2.AsyncClient(
                    headers=headers,
                    timeout=self._config.connect_timeout_s,
                    trust_env=False,
                    follow_redirects=False,
                )
                transport = streamable_http_client(
                    self._config.url,
                    http_client=self._http_client,
                )
                client = Client(
                    transport,
                    read_timeout_seconds=self._config.tool_timeout_s,
                )
            else:
                raise ValueError(f"unsupported MCP transport: {self._config.transport}")
            # Keep context-manager entry and exit in this task.  ``wait_for`` creates
            # a child Task, which can violate AnyIO cancel-scope ownership during a
            # partially-entered stdio transport cleanup.
            async with asyncio.timeout(self._config.connect_timeout_s):
                await client.__aenter__()
            self._client = client
            self.health_status = "connected"
            self.last_error = None
        except (Exception, asyncio.CancelledError) as exc:
            self.health_status = "failed"
            self.last_error = "MCP connection failed"
            # ``Client.__aenter__`` may already have started a stdio process or
            # HTTP transport before discovery fails or the outer timeout fires.
            # Keep no half-open transport: unwind the context even though the
            # client was never promoted to ``self._client``.
            if client is not None:
                try:
                    async with asyncio.timeout(self._config.connect_timeout_s):
                        await client.__aexit__(type(exc), exc, exc.__traceback__)
                except Exception:
                    pass
            if self._http_client is not None:
                await self._http_client.aclose()
                self._http_client = None
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise McpServerUnavailableError(str(exc)) from exc

    async def list_tools(self) -> list[McpToolDef]:
        client = self._require_client()
        try:
            response = await asyncio.wait_for(
                client.list_tools(),
                timeout=self._config.tool_timeout_s,
            )
        except asyncio.CancelledError:
            self._record_failure("MCP discovery cancelled locally")
            raise
        except Exception as exc:
            self._record_failure("MCP discovery failed")
            raise McpServerUnavailableError(str(exc)) from exc
        return [
            McpToolDef(
                name=tool.name,
                description=tool.description or tool.title or "",
                input_schema=dict(tool.input_schema),
            )
            for tool in response.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpCallResult:
        from mcp import types

        client = self._require_client()
        try:
            response = await asyncio.wait_for(
                client.call_tool(
                    name,
                    arguments,
                    read_timeout_seconds=self._config.tool_timeout_s,
                ),
                timeout=self._config.tool_timeout_s,
            )
        except asyncio.CancelledError:
            self._record_failure("MCP call cancelled locally; external effects may continue")
            raise
        except TimeoutError as exc:
            self._record_failure("MCP tool call timed out")
            raise McpServerUnavailableError("MCP tool call timed out") from exc
        except Exception as exc:
            self._record_failure("MCP tool call failed")
            raise McpServerUnavailableError(str(exc)) from exc
        text_parts: list[str] = []
        non_text: list[str] = []
        for block in response.content:
            if isinstance(block, types.TextContent):
                text_parts.append(block.text)
            elif isinstance(block, types.ImageContent):
                non_text.append("[MCP image content omitted]")
            elif isinstance(block, types.AudioContent):
                non_text.append("[MCP audio content omitted]")
            else:
                non_text.append(f"[MCP {block.type} content omitted]")
        if not text_parts and response.structured_content is not None:
            text_parts.append(
                json.dumps(response.structured_content, ensure_ascii=False, sort_keys=True)
            )
        text_parts.extend(non_text)
        content = "\n".join(text_parts) or "[MCP tool returned no text content]"
        return McpCallResult(content=content, is_error=response.is_error)

    def _record_failure(self, message: str) -> None:
        self.health_status = "degraded"
        self.last_error = message

    async def close(self) -> None:
        if self._owner is None:
            await self._close_in_owner()
            return
        owner = self._owner
        assert self._stop is not None
        self._stop.set()
        await asyncio.shield(owner)
        self._owner = None
        if self._close_error is not None:
            raise McpServerUnavailableError("MCP transport cleanup failed") from self._close_error

    async def _close_in_owner(self) -> None:
        self.health_status = "stopped"
        client, self._client = self._client, None
        try:
            if client is not None:
                try:
                    async with asyncio.timeout(self._config.connect_timeout_s):
                        await client.__aexit__(None, None, None)
                except TimeoutError as exc:
                    raise McpServerUnavailableError(
                        "MCP client close timed out"
                    ) from exc
        finally:
            if self._http_client is not None:
                await self._http_client.aclose()
                self._http_client = None

    @property
    def protocol_version(self) -> str | None:
        if self._client is None:
            return None
        value = self._client.protocol_version
        return str(value) if value is not None else None

    def _require_client(self) -> Client:
        if self._client is None:
            raise McpServerUnavailableError("MCP client is not connected")
        return self._client


def _stdio_environment(configured: dict[str, str]) -> dict[str, str]:
    allowed_names = (
        "PATH",
        "PATHEXT",
        "SystemRoot",
        "ComSpec",
        "WINDIR",
        "LANG",
        "LC_ALL",
        "TMP",
        "TEMP",
    )
    result = {
        name: value
        for name in allowed_names
        if (value := os.environ.get(name)) is not None
    }
    result.update(configured)
    return result


_IS_WINDOWS = os.name == "nt"
_WINDOWS_DEFAULT_PATHEXT = ".COM;.EXE;.BAT;.CMD;.VBS;.JS;.WS;.MSC"


def _env_value(environ: Mapping[str, str], name: str) -> str | None:
    if not _IS_WINDOWS:
        return environ.get(name)
    expected = name.casefold()
    return next(
        (value for key, value in environ.items() if key.casefold() == expected),
        None,
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _candidate_names(command: str, environ: Mapping[str, str]) -> list[str]:
    if not _IS_WINDOWS:
        return [command]

    pathext_source = _env_value(environ, "PATHEXT") or _WINDOWS_DEFAULT_PATHEXT
    extensions: list[str] = []
    seen: set[str] = set()
    for raw_extension in pathext_source.split(";"):
        extension = raw_extension.strip().rstrip(".")
        if not extension:
            continue
        if not extension.startswith("."):
            extension = "." + extension
        identity = extension.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        extensions.append(extension)

    names = [command + extension for extension in extensions]
    if any(command.casefold().endswith(extension.casefold()) for extension in extensions):
        names.insert(0, command)
    return names


def _resolve_stdio_command(
    command: str,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> str:
    """Resolve an MCP executable without consulting the project directory."""

    if not command or command != command.strip() or "\x00" in command:
        raise ValueError("MCP stdio command must be a non-empty executable name")

    command_path = Path(command)
    if command_path.is_absolute():
        candidate = command_path.resolve(strict=False)
        if not candidate.is_file():
            raise FileNotFoundError(f"stdio command was not found: {command}")
        return str(candidate)

    # PureWindowsPath also rejects drive-relative and backslash-containing commands
    # when this validation is exercised on a non-Windows test host.
    if command_path.name != command or PureWindowsPath(command).name != command:
        raise ValueError(
            "MCP stdio command must be an absolute path or a bare executable name"
        )

    source = os.environ if environ is None else environ
    path_value = _env_value(source, "PATH")
    if not path_value:
        raise FileNotFoundError(
            f"MCP stdio command not found in trusted PATH: {command!r}"
        )

    current = Path.cwd() if cwd is None else cwd
    current_lexical = Path(os.path.abspath(current))
    current_resolved = current.resolve(strict=True)
    path_separator = ";" if _IS_WINDOWS else os.pathsep
    access_mode = os.F_OK if _IS_WINDOWS else os.F_OK | os.X_OK

    seen_directories: set[str] = set()
    for raw_directory in path_value.split(path_separator):
        if not raw_directory:
            continue
        directory = Path(raw_directory)
        if not directory.is_absolute():
            continue
        lexical_directory = Path(os.path.abspath(directory))
        try:
            resolved_directory = directory.resolve(strict=True)
        except OSError:
            continue
        if not resolved_directory.is_dir():
            continue
        if _is_within(lexical_directory, current_lexical) or _is_within(
            resolved_directory, current_resolved
        ):
            continue

        identity = os.path.normcase(str(resolved_directory))
        if identity in seen_directories:
            continue
        seen_directories.add(identity)

        for candidate_name in _candidate_names(command, source):
            candidate = resolved_directory / candidate_name
            if not candidate.is_file() or not os.access(candidate, access_mode):
                continue
            try:
                resolved_candidate = candidate.resolve(strict=True)
            except OSError:
                continue
            if _is_within(resolved_candidate, current_resolved):
                continue
            return str(resolved_candidate)

    raise FileNotFoundError(
        f"MCP stdio command not found in trusted PATH: {command!r}"
    )




_HEADER_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_headers(headers: dict[str, str]) -> dict[str, str]:
    expanded: dict[str, str] = {}
    for name, raw_value in headers.items():
        def replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            value = os.environ.get(env_name)
            if value is None:
                raise ValueError(f"missing environment variable for MCP header: {env_name}")
            return value

        value = _HEADER_ENV_PATTERN.sub(replace, raw_value)
        if "${" in value:
            raise ValueError("invalid MCP header environment interpolation")
        expanded[name] = value
    return expanded


__all__ = [
    "McpCallResult",
    "McpClient",
    "McpServerUnavailableError",
    "McpToolDef",
    "McpToolError",
]
