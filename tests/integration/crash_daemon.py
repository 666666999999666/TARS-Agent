"""Test-only Core process whose provider pauses after one durable side effect."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Self

from tars_agent.core.app import CoreApp
from tars_agent.core.llm.types import LlmResponse, ToolCallBlock


class CrashWindowProvider:
    """Save one note, then stay inside the next model call until the process dies."""

    def __init__(self, model: str, *, max_tokens: int = 8192, **kwargs: Any) -> None:
        del model, max_tokens, kwargs
        self._calls = 0

    @classmethod
    def from_config(cls, config: Any) -> Self:
        return cls(config.default_model, max_tokens=config.max_tokens)

    async def close(self) -> None:
        pass

    async def chat(self, **kwargs: Any) -> LlmResponse:
        del kwargs
        self._calls += 1
        if self._calls == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="crash-note",
                        name="note_save",
                        input={"content": "written exactly once before crash"},
                    )
                ],
            )
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class CrashSubagentProvider:
    """Start a background child and keep both parent and child calls in flight."""

    def __init__(self, model: str, *, max_tokens: int = 8192, **kwargs: Any) -> None:
        del model, max_tokens, kwargs
        self._root_run_id: str | None = None

    @classmethod
    def from_config(cls, config: Any) -> Self:
        return cls(config.default_model, max_tokens=config.max_tokens)

    async def close(self) -> None:
        pass

    async def chat(self, **kwargs: Any) -> LlmResponse:
        run_id = str(kwargs["run_id"])
        marker = os.environ.get("TARS_CRASH_MARKER")
        if marker:
            with Path(marker).open("a", encoding="utf-8") as handle:
                handle.write(f"{run_id}\n")
        if self._root_run_id is None:
            self._root_run_id = run_id
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="spawn-crash-child",
                        name="spawn_agent",
                        input={
                            "description": "crash recovery child",
                            "prompt": "remain active until the daemon is terminated",
                            "run_in_background": True,
                        },
                    )
                ],
            )
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def main() -> None:
    # AgentRunner resolves this module global lazily at execution time.  Keeping
    # the patch inside a dedicated subprocess avoids any test-process leakage.
    import tars_agent.core.app as app_module
    import tars_agent.core.runner as runner_module

    provider = (
        CrashSubagentProvider
        if os.environ.get("TARS_CRASH_SCENARIO") == "subagent"
        else CrashWindowProvider
    )
    app_module.AnthropicProvider = provider  # type: ignore[assignment]
    runner_module.AnthropicProvider = provider  # type: ignore[assignment]
    asyncio.run(CoreApp().run())


if __name__ == "__main__":
    main()
