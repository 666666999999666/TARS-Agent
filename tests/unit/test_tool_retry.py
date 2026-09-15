from __future__ import annotations

import pytest

import tars_agent.core.tools.invocation as inv_mod
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import ToolCallBlock
from tars_agent.core.tools.base import BaseTool, ToolResult
from tars_agent.core.tools.errors import RateLimitedError
from tars_agent.core.tools.invocation import invoke_tool
from tars_agent.core.tools.registry import ToolRegistry

# --- stub tools --------------------------------------------------------------

class _FailNTimes(BaseTool):
    """Fails with runtime_error for the first n calls, then succeeds."""
    name = "fail_n"
    description = "Fails n times then succeeds"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    def __init__(
        self,
        n: int,
        *,
        error_type: str = "runtime_error",
        retryable: bool = True,
    ) -> None:
        self._remaining = n
        self._error_type = error_type
        self._retryable = retryable

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        if self._remaining > 0:
            self._remaining -= 1
            return ToolResult(
                content="transient error",
                is_error=True,
                error_type=self._error_type,
                retryable=self._retryable,
            )
        return ToolResult(content="ok")


class _RateLimitedNTimes(BaseTool):
    """Raises RateLimitedError for the first n calls, then succeeds."""
    name = "rate_n"
    description = "Rate-limits n times then succeeds"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, n: int) -> None:
        self._remaining = n

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        if self._remaining > 0:
            self._remaining -= 1
            raise RateLimitedError("429 Too Many Requests")
        return ToolResult(content="ok")


class _AlwaysFails(BaseTool):
    name = "always_fail"
    description = "Always fails"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, error_type: str = "runtime_error", *, retryable: bool = False) -> None:
        self._error_type = error_type
        self._retryable = retryable

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(
            content="permanent error",
            is_error=True,
            error_type=self._error_type,
            retryable=self._retryable,
        )


# --- helper ------------------------------------------------------------------

def _call(name: str) -> ToolCallBlock:
    return ToolCallBlock(id="t1", name=name, input={})


async def _run(tool: BaseTool, *, monkeypatch: pytest.MonkeyPatch) -> tuple[ToolResult, list]:
    monkeypatch.setattr(inv_mod, "_RETRY_BASE_S", 0.0)
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    events: list = []

    async def _collect(e: object) -> None:
        events.append(e)

    bus.subscribe(_collect)
    result = await invoke_tool(registry, _call(tool.name), bus, run_id="r")
    return result, events


# --- tests -------------------------------------------------------------------


# 功能：验证仅显式 retryable 的 runtime_error 才会自动重试并成功
# 设计：测试桩明确返回 retryable=True，避免把错误类型本身误当成幂等声明
async def test_runtime_error_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(_FailNTimes(1), monkeypatch=monkeypatch)
    assert not result.is_error
    assert result.content == "ok"
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 1
    assert failed_events[0].attempt == 1  # type: ignore[attr-defined]
    assert failed_events[0].retryable is True  # type: ignore[attr-defined]


# 功能：验证 rate_limited 异常触发重试，重试成功后返回正常结果
# 设计：_RateLimitedNTimes(1) 抛 RateLimitedError 一次，第二次成功；检查 error_class 字段
async def test_rate_limited_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(_RateLimitedNTimes(1), monkeypatch=monkeypatch)
    assert not result.is_error
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 1
    assert failed_events[0].error_class == "rate_limited"  # type: ignore[attr-defined]


# 功能：验证显式 retryable 的 runtime_error 耗尽两次重试后记录完整 attempt
# 设计：测试桩明确声明可重试，断言三次执行与失败事件一一对应
async def test_runtime_error_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(
        _AlwaysFails("runtime_error", retryable=True),
        monkeypatch=monkeypatch,
    )
    assert result.is_error
    assert result.error_type == "runtime_error"
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 3
    attempts = [e.attempt for e in failed_events]  # type: ignore[attr-defined]
    assert attempts == [1, 2, 3]


# 功能：验证未声明 retryable 的 runtime_error 默认只执行一次
# 设计：使用永久失败工具统计 execution_started 与 failed 事件，锁定副作用工具不透明重放语义
async def test_runtime_error_defaults_to_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(_AlwaysFails("runtime_error"), monkeypatch=monkeypatch)
    assert result.is_error
    started = [e for e in events if e.type == "tool.execution_started"]  # type: ignore[attr-defined]
    failed = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(started) == 1
    assert len(failed) == 1
    assert failed[0].retryable is False  # type: ignore[attr-defined]
    assert result.retryable is False


# 功能：验证 rate_limited 耗尽重试后最终返回失败，error_class 为 rate_limited
# 设计：_RateLimitedNTimes(10) 始终抛异常，断言 3 个 failed 事件且 error_class 统一
async def test_rate_limited_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(_RateLimitedNTimes(10), monkeypatch=monkeypatch)
    assert result.is_error
    assert result.error_type == "rate_limited"
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 3
    assert all(e.error_class == "rate_limited" for e in failed_events)  # type: ignore[attr-defined]


# 功能：验证 schema_error 不触发重试，直接失败
# 设计：_AlwaysFails("schema_error") 首次即 schema 错误，断言只发一次 failed 事件且 attempt=1
async def test_schema_error_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    result, events = await _run(_AlwaysFails("schema_error"), monkeypatch=monkeypatch)
    assert result.is_error
    assert result.error_type == "schema_error"
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 1
    assert failed_events[0].attempt == 1  # type: ignore[attr-defined]


# 功能：验证 timeout_error 不触发重试，直接失败
# 设计：SlowTool 配合极短超时触发 TimeoutError；断言只发一次 failed 事件，不重试（重试会再次超时）
async def test_timeout_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    class _SlowTool(BaseTool):
        name = "slow"
        description = "sleeps"
        input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

        async def invoke(self, params: dict[str, object]) -> ToolResult:
            await asyncio.sleep(60)
            return ToolResult(content="done")

    monkeypatch.setattr(inv_mod, "_RETRY_BASE_S", 0.0)
    registry = ToolRegistry()
    registry.register(_SlowTool())
    bus = EventBus()
    events: list = []

    async def _collect(e: object) -> None:
        events.append(e)

    bus.subscribe(_collect)
    result = await invoke_tool(registry, _call("slow"), bus, run_id="r", timeout=0.05)

    assert result.is_error
    assert result.error_type == "timeout"
    failed_events = [e for e in events if e.type == "tool.call_failed"]  # type: ignore[attr-defined]
    assert len(failed_events) == 1


# 功能：验证成功后的 tool.call_failed 事件中 error_class 字段存在且取值合法
# 设计：_FailNTimes(2) 两次失败后成功，检查所有 failed 事件的 error_class 均在合法枚举内
async def test_failed_event_has_valid_error_class(monkeypatch: pytest.MonkeyPatch) -> None:
    valid_classes = {"runtime_error", "timeout", "schema_error", "permission_denied", "rate_limited"}
    result, events = await _run(_FailNTimes(2), monkeypatch=monkeypatch)
    assert not result.is_error
    for e in events:
        if e.type == "tool.call_failed":  # type: ignore[attr-defined]
            assert e.error_class in valid_classes  # type: ignore[attr-defined]
