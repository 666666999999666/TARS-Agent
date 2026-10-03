"""Real Core and Docker lifecycle with a fixed, credential-free test provider."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Self

from tars_agent.core.app import CoreApp
from tars_agent.core.llm.types import LlmResponse, ToolCallBlock


class RecoveryProbeProvider:
    def __init__(self) -> None:
        self.calls = 0

    @classmethod
    def from_config(cls, config: Any) -> Self:
        if config.api_key or config.anthropic_api_key or config.base_url:
            raise RuntimeError("real provider configuration is forbidden in the recovery probe")
        return cls()

    async def close(self) -> None:
        return None

    async def chat(self, **kwargs: Any) -> LlmResponse:
        self.calls += 1
        evidence = Path(os.environ["RECOVERY_PROBE_CALLS"])
        with evidence.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"run_id": kwargs["run_id"], "call": self.calls}) + "\n")
        if self.calls == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[ToolCallBlock(
                    id="recovery-probe", name="bash",
                    input={"command": "python probe.py", "timeout": 120},
                )],
            )
        return LlmResponse(stop_reason="end_turn", text="fixed probe finished")


def main() -> None:
    import tars_agent.core.app as app_module
    import tars_agent.core.runner as runner_module

    # Deliberately leave production recovery and DockerRuntime untouched.
    app_module.AnthropicProvider = RecoveryProbeProvider  # type: ignore[assignment]
    runner_module.AnthropicProvider = RecoveryProbeProvider  # type: ignore[assignment]
    asyncio.run(CoreApp().run())


if __name__ == "__main__":
    main()
