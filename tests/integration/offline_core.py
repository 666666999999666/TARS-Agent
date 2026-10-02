"""Explicit test entrypoint: real Core/SQLite/IPC, no Docker execution or recovery."""

from __future__ import annotations

import asyncio
from typing import Any

from tars_agent.core.app import CoreApp
from tars_agent.core.config import SandboxConfig
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter
from tars_agent.core.tools.runtime.recovery import SandboxResourceStore


def install_offline_runtime() -> None:
    import tars_agent.core.app as app_module

    async def recover(
        _config: SandboxConfig, store: SandboxResourceStore, **_kwargs: Any,
    ) -> None:
        # These tests never create Docker resources; don't pretend to recover one.
        assert store.load() == [], "offline Core cannot own sandbox resources"

    async def initialize(_config: SandboxConfig, **_kwargs: Any) -> RuntimeRouter:
        return RuntimeRouter(FakeRuntime(available=False), allow_host_fallback=False)

    app_module.recover_core_sandboxes = recover  # type: ignore[assignment]
    app_module.initialize_runtime_router = initialize


def main() -> None:
    install_offline_runtime()
    asyncio.run(CoreApp().run())


if __name__ == "__main__":
    main()
