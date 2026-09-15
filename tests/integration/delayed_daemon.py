"""Test-only Core process whose provider completes after a short delay."""

from __future__ import annotations

import asyncio
from typing import Any, Self

from tars_agent.core.app import CoreApp
from tars_agent.core.llm.types import LlmResponse


class DelayedProvider:
    def __init__(self, model: str, *, max_tokens: int = 8192, **kwargs: Any) -> None:
        del model, max_tokens, kwargs

    @classmethod
    def from_config(cls, config: Any) -> Self:
        return cls(config.default_model, max_tokens=config.max_tokens)

    async def close(self) -> None:
        pass

    async def chat(self, **kwargs: Any) -> LlmResponse:
        del kwargs
        await asyncio.sleep(0.6)
        return LlmResponse(
            stop_reason="end_turn",
            text="completed after disconnect",
        )


def main() -> None:
    import tars_agent.core.app as app_module
    import tars_agent.core.runner as runner_module

    app_module.AnthropicProvider = DelayedProvider  # type: ignore[attr-defined]
    runner_module.AnthropicProvider = DelayedProvider  # type: ignore[attr-defined]
    asyncio.run(CoreApp().run())


if __name__ == "__main__":
    main()
