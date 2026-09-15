from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from pydantic import BaseModel

from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.provider import AnthropicProvider

pytestmark = pytest.mark.integration


# 功能：可信 CI 仅用一次文本流验证真实 Anthropic token、usage 与成功终结
# 设计：空工具、64 token、45 秒外层超时，模型只从仓库变量读取，不执行开放式 Agent 任务
async def test_real_anthropic_bounded_text_stream() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("TARS_SMOKE_MODEL")
    if not api_key or not model:
        pytest.skip("trusted Anthropic smoke credentials/model unavailable")

    events: list[BaseModel] = []
    bus = EventBus()

    async def collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(collect)
    provider = AnthropicProvider(model=model, max_tokens=64)
    response = await asyncio.wait_for(
        provider.chat(
            messages=[{"role": "user", "content": "Reply with exactly: smoke-ok"}],
            tool_schemas=[],
            bus=bus,
            run_id="trusted-smoke",
            step=1,
            system="Return the requested short text only.",
        ),
        timeout=45,
    )

    event_types = [getattr(event, "type", "") for event in events]
    assert response.stop_reason == "end_turn"
    assert response.text.strip()
    assert "llm.token" in event_types
    assert "llm.usage" in event_types
    assert response.usage is not None
    usage: dict[str, Any] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    assert usage["input_tokens"] > 0
    assert 0 < usage["output_tokens"] <= 64
