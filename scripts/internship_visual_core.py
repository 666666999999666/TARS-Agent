"""Real Core/TUI fixture with a deterministic provider; sends no model requests."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    os.environ["TARS_HOME"] = str(args.home.resolve())
    from tars_agent.core import app as app_module
    from tars_agent.core.config import TarsConfig
    from tars_agent.core.llm.types import LlmResponse, ToolCallBlock
    from tars_agent.core.runner import AgentRunner

    config = TarsConfig()
    config.port = args.port
    config.permission.timeout_s = 600
    config.trace.enabled = False
    config.compaction.auto_threshold = 0
    app_module.get_config = lambda: config

    class VisualProvider:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, tool_schemas, bus, run_id, **kwargs):
            self.calls += 1
            if self.calls == 1:
                content = "\n".join(
                    f"参数第 {i:03d} 行：中文审批检查 [red] 保持原样，不解释为样式。"
                    for i in range(1, 81)
                ) + "\nTAIL_MARKER_审批参数结束"
                return LlmResponse(stop_reason="tool_use", tool_calls=[
                    ToolCallBlock("visual-write", "write_file", {
                        "path": "visual-only.txt", "content": content,
                    }),
                ])
            return LlmResponse(stop_reason="end_turn", text="确定性界面检查结束。")

    def make_runner(actual_config, **kwargs):
        return AgentRunner(actual_config, provider=VisualProvider(), **kwargs)

    app_module.AgentRunner = make_runner
    asyncio.run(app_module.CoreApp().run())


if __name__ == "__main__":
    main()
