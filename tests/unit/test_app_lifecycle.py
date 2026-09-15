from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from tars_agent.core import app as app_module
from tars_agent.core.app import CoreApp
from tars_agent.core.bus.envelope import HandlerError
from tars_agent.core.config import TarsConfig


async def test_shutdown_requires_matching_local_control_token() -> None:
    app = CoreApp()
    app._shutdown_event = asyncio.Event()
    with pytest.raises(HandlerError) as caught:
        await app._shutdown_handler({"token": "wrong"})
    assert caught.value.code == -32001
    assert not app._shutdown_event.is_set()
    assert (await app._shutdown_handler({"token": app._control_token})).ok
    assert not app._shutdown_event.is_set()  # SocketServer sets it only after sending the reply.


async def test_cleanup_attempts_all_resources_in_reverse_order() -> None:
    order: list[str] = []
    async def first() -> None:
        order.append("database")
    async def second() -> None:
        order.append("runtime")
        raise OSError("cleanup failure")
    async def third() -> None:
        order.append("server")
    with pytest.raises(RuntimeError, match="runtime"):
        await CoreApp()._cleanup_resources([("database", first), ("runtime", second), ("server", third)])
    assert order == ["server", "runtime", "database"]


@pytest.mark.parametrize("failure", ["trace", "events", "mcp", "tools", "server"])
async def test_startup_failure_closes_every_acquired_resource(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    acquired: list[str] = []
    closed: list[str] = []
    class Resource:
        def __init__(self, name: str) -> None:
            self.name = name
            acquired.append(name)
        async def start(self) -> str:
            if self.name == failure:
                raise RuntimeError("injected startup failure")
            return "127.0.0.1:1234"
        async def start_all(self, _servers: Any) -> None:
            await self.start()
        async def stop(self) -> None:
            closed.append(self.name)
        async def handle(self, _event: Any) -> None:
            pass
        async def recover_interrupted(self) -> int:
            return 0
        def register(self, *_args: Any, **_kwargs: Any) -> None:
            pass
        dispose = stop
        stop_all = stop
        cleanup = stop
        shutdown = stop
    config = TarsConfig()
    config.trace.enabled = True
    config.mcp.servers = [SimpleNamespace()]  # type: ignore[list-item]
    monkeypatch.setattr(app_module, "get_config", lambda: config)
    monkeypatch.setattr(app_module, "setup_logging", lambda _config: None)
    monkeypatch.setattr(app_module, "TraceWriter", lambda _path: Resource("trace"))
    monkeypatch.setattr(app_module, "PermissionManager", lambda **_kw: object())
    monkeypatch.setattr(app_module, "load_policy_file", lambda _path: {})
    async def bootstrap(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(database=Resource("database"))
    monkeypatch.setattr(app_module, "bootstrap_state", bootstrap)
    monkeypatch.setattr(app_module, "DurableEventHub", lambda _db: Resource("events"))
    monkeypatch.setattr(app_module, "IpcEventBroadcaster", lambda *_a, **_kw: Resource("broadcaster"))
    monkeypatch.setattr(app_module, "McpServerManager", lambda: Resource("mcp"))
    async def initialize(_config: Any) -> Any:
        if failure == "tools":
            raise RuntimeError("injected startup failure")
        return Resource("tools")
    monkeypatch.setattr(app_module, "initialize_runtime_router", initialize)
    monkeypatch.setattr(app_module, "BackgroundTaskRegistry", lambda _db, _bus: Resource("subagents"))
    monkeypatch.setattr(app_module, "RuntimeService", lambda *_a, **_kw: Resource("runtime"))
    monkeypatch.setattr(app_module, "SocketServer", lambda *_a, **_kw: Resource("server"))
    with pytest.raises(RuntimeError, match="injected startup failure"):
        await CoreApp().run()
    assert closed == list(reversed(acquired))


async def test_invalid_shutdown_payload_does_not_echo_control_token() -> None:
    from pydantic import ValidationError

    from tars_agent.core.bus.commands import CoreShutdownCommand
    from tars_agent.core.transport.socket_server import _validation_details
    with pytest.raises(ValidationError) as caught:
        CoreShutdownCommand.model_validate({"token": {"secret": "DO-NOT-LOG"}})
    assert "DO-NOT-LOG" not in repr(_validation_details(caught.value))
