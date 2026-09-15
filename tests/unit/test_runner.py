from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from tars_agent.core.artifacts import ArtifactStore
from tars_agent.core.config import TarsConfig
from tars_agent.core.events.bus import EventBus
from tars_agent.core.llm.types import LlmResponse, ToolCallBlock
from tars_agent.core.runner import AgentRunner
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeCleanupPending, RuntimeRouter


class ScriptedProvider:
    def __init__(self, *responses: LlmResponse) -> None:
        self.responses = list(responses)
        self.inputs: list[tuple[list[dict[str, object]], str | None]] = []
        self.schemas: list[dict[str, object]] = []

    async def chat(self, messages, tool_schemas, bus, run_id, **kwargs) -> LlmResponse:
        self.inputs.append(([dict(message) for message in messages], kwargs.get("system")))
        self.schemas = list(tool_schemas)
        return self.responses.pop(0)


def runner_inputs(tmp_path: Path, *, history=None, notes="", **kwargs):
    return {
        "run_id": "run-1",
        "session_id": "sess-1",
        "workspace_root": tmp_path,
        "history": history if history is not None else [{"role": "user", "content": "goal"}],
        "artifact_store": ArtifactStore(tmp_path / "artifacts" / "sessions"),
        "session_notes": notes,
        **kwargs,
    }


def make_runner(provider=None, *, config=None, bus=None, runtime=None):
    return AgentRunner(
        config or TarsConfig(),
        provider=provider,
        bus=bus or EventBus(),
        tool_runtime=runtime or RuntimeRouter(FakeRuntime(), allow_host_fallback=False),
    )


async def test_runner_returns_result_without_publishing_or_saving_terminal_state(tmp_path: Path):
    events: list[BaseModel] = []
    bus = EventBus()

    async def collect(event):
        events.append(event)

    bus.subscribe(collect)
    runner = make_runner(ScriptedProvider(LlmResponse(stop_reason="end_turn", text="done")), bus=bus)
    outcome = await runner.run_and_capture("goal", **runner_inputs(tmp_path))

    assert outcome.status == "success"
    assert outcome.result == "done"
    assert outcome.steps == 1
    assert [event.type for event in events] == ["run.started", "step.started", "step.finished"]
    assert all(event.run_id == "run-1" for event in events)
    assert not list(tmp_path.rglob("events.jsonl"))
    assert not list(tmp_path.rglob("thread.jsonl"))
    assert not list(tmp_path.rglob("meta.json"))


async def test_only_explicit_history_and_notes_enter_model_context(tmp_path: Path):
    inputs = runner_inputs(tmp_path, history=[
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "goal"},
    ], notes="本次指定的笔记")
    store = inputs["artifact_store"]
    store.append_note("sess-1", "文件内的旧笔记", "old-run")
    legacy = store.session_dir("sess-1") / "thread.jsonl"
    legacy.write_text('{"role":"user","content":"旧文件消息"}\n', encoding="utf-8")
    provider = ScriptedProvider(LlmResponse(stop_reason="end_turn", text="done"))

    outcome = await make_runner(provider).run_and_capture("goal", **inputs)

    messages, system = provider.inputs[0]
    assert messages == inputs["history"]
    assert len(inputs["history"]) == 3
    assert "本次指定的笔记" in system
    assert "文件内的旧笔记" not in system
    assert "旧文件消息" in legacy.read_text(encoding="utf-8")
    assert outcome.messages == [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}]


async def test_note_save_executes_and_returns_tool_result_to_model(tmp_path: Path):
    provider = ScriptedProvider(
        LlmResponse(stop_reason="tool_use", tool_calls=[
            ToolCallBlock(id="note-1", name="note_save", input={"content": "使用 Python 3.12"}),
        ]),
        LlmResponse(stop_reason="end_turn", text="noted"),
    )
    inputs = runner_inputs(tmp_path)
    outcome = await make_runner(provider).run_and_capture("goal", **inputs)

    assert outcome.status == "success"
    assert "使用 Python 3.12" in inputs["artifact_store"].read_notes("sess-1")
    assert provider.inputs[1][0][-1]["content"] == [
        {"type": "tool_result", "tool_use_id": "note-1", "content": "saved"},
    ]


async def test_empty_whitelist_exposes_no_tools(tmp_path: Path):
    provider = ScriptedProvider(LlmResponse(stop_reason="end_turn", text="done"))
    await make_runner(provider).run_and_capture("goal", **runner_inputs(tmp_path, tool_whitelist=[]))
    assert provider.schemas == []


async def test_max_steps_returns_failure_and_cleans_runtime(tmp_path: Path):
    config = TarsConfig()
    config.agent.max_steps = 2
    provider = ScriptedProvider(*[
        LlmResponse(stop_reason="tool_use", tool_calls=[
            ToolCallBlock(id=f"tool-{index}", name="unknown_tool", input={}),
        ]) for index in range(2)
    ])
    runtime = AsyncMock()
    outcome = await make_runner(provider, config=config, runtime=runtime).run_and_capture(
        "goal", **runner_inputs(tmp_path),
    )
    assert outcome.status == "failed"
    assert outcome.reason == "exceeded_max_steps"
    assert len(provider.inputs) == 2
    runtime.cleanup_run.assert_awaited_once_with("run-1")


async def test_provider_configuration_failure_is_a_failed_outcome(tmp_path: Path):
    runtime = AsyncMock()
    with patch("tars_agent.core.runner.AnthropicProvider.from_config", side_effect=SystemExit("missing key")):
        outcome = await make_runner(runtime=runtime).run_and_capture("goal", **runner_inputs(tmp_path))
    assert outcome.status == "failed"
    assert outcome.reason == "llm_error"
    runtime.cleanup_run.assert_awaited_once_with("run-1")


async def test_cancellation_returns_outcome_after_resource_cleanup(tmp_path: Path):
    started = asyncio.Event()

    class WaitingProvider:
        async def chat(self, *args, **kwargs):
            started.set()
            await asyncio.Event().wait()

    runtime = AsyncMock()
    task = asyncio.create_task(make_runner(WaitingProvider(), runtime=runtime).run_and_capture(
        "goal", **runner_inputs(tmp_path),
    ))
    await started.wait()
    task.cancel()
    outcome = await task
    assert outcome.status == "cancelled"
    assert outcome.reason == "cancelled"
    runtime.cleanup_run.assert_awaited_once_with("run-1")


async def test_repeated_cancel_cannot_hide_pending_resource_cleanup(tmp_path: Path):
    cleaning = asyncio.Event()
    release = asyncio.Event()

    async def pending_cleanup(run_id):
        cleaning.set()
        await release.wait()
        raise RuntimeCleanupPending(run_id)

    runtime = AsyncMock()
    runtime.cleanup_run.side_effect = pending_cleanup
    provider = ScriptedProvider(LlmResponse(stop_reason="end_turn", text="done"))
    task = asyncio.create_task(make_runner(provider, runtime=runtime).run_and_capture(
        "goal", **runner_inputs(tmp_path),
    ))
    await cleaning.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(RuntimeCleanupPending):
        await task
    runtime.cleanup_run.assert_awaited_once_with("run-1")


async def test_owned_provider_closes_after_tool_cleanup(tmp_path: Path):
    order = []
    runtime = AsyncMock()
    provider = ScriptedProvider(LlmResponse(stop_reason="end_turn", text="done"))

    async def clean(run_id):
        order.append("tools")

    async def close():
        order.append("provider")

    runtime.cleanup_run.side_effect = clean
    provider.close = close
    with patch("tars_agent.core.runner.AnthropicProvider.from_config", return_value=provider):
        outcome = await make_runner(runtime=runtime).run_and_capture("goal", **runner_inputs(tmp_path))
    assert outcome.status == "success"
    assert order == ["tools", "provider"]

async def test_cancellation_during_provider_close_waits_for_release(tmp_path: Path):
    closing = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    provider = ScriptedProvider(LlmResponse(stop_reason="end_turn", text="done"))

    async def close():
        closing.set()
        await release.wait()
        closed.set()

    provider.close = close
    with patch("tars_agent.core.runner.AnthropicProvider.from_config", return_value=provider):
        operation = asyncio.create_task(make_runner().run_and_capture("goal", **runner_inputs(tmp_path)))
        await closing.wait()
        operation.cancel()
        await asyncio.sleep(0)
        still_waiting = not operation.done()
        release.set()
        result = (await asyncio.gather(operation, return_exceptions=True))[0]
    assert still_waiting, "cancellation interrupted provider.close before it released the connection"
    assert closed.is_set()
    assert result.status == "cancelled"

async def test_provider_close_failure_does_not_hide_pending_tool_cleanup(tmp_path: Path):
    provider = ScriptedProvider(LlmResponse(stop_reason="end_turn", text="done"))
    runtime = AsyncMock()
    runtime.cleanup_run.side_effect = RuntimeCleanupPending("run-1")

    async def close():
        raise OSError("provider close failed")

    provider.close = close
    with patch("tars_agent.core.runner.AnthropicProvider.from_config", return_value=provider):
        with pytest.raises(RuntimeCleanupPending):
            await make_runner(runtime=runtime).run_and_capture("goal", **runner_inputs(tmp_path))
