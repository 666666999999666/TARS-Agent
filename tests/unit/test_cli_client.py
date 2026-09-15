from __future__ import annotations

import asyncio

import pytest

from tars_agent.cli.client import TerminalClient
from tests.cli_core_stub import CliCoreStub


class FakeInput:
    def __init__(self) -> None:
        self.lines: asyncio.Queue[str | None] = asyncio.Queue()
        self.reading = asyncio.Event()
        self.closed = False

    async def read(self) -> str | None:
        self.reading.set()
        return await self.lines.get()

    def feed(self, line: str | None) -> None:
        self.lines.put_nowait(line)

    def close(self) -> None:
        self.closed = True


async def result_of(task: asyncio.Task[int]) -> int:
    return await asyncio.wait_for(task, timeout=5)


async def wait_notice(capsys, marker: str) -> str:
    observed = ""
    async with asyncio.timeout(5):
        while marker not in observed:
            observed += capsys.readouterr().err
            if marker not in observed:
                await asyncio.sleep(0.01)
    return observed


def assert_client_only(core: CliCoreStub) -> None:
    methods = [name for name, _ in core.calls]
    assert "session.close" not in methods
    assert "core.shutdown" not in methods
    assert "agent.run" not in methods


async def test_goal_catches_completion_before_submission_response(capsys):
    async with CliCoreStub() as core:
        code = await asyncio.wait_for(TerminalClient(core.config(), interactive=False).run(goal="read"), 5)
        methods = [method for method, _ in core.calls]
        assert methods.index("session.create") < methods.index("event.subscribe") < methods.index("session.send_message")
        assert "run.get" in methods and "run.metrics" in methods
        assert next(params for method, params in core.calls if method == "session.create")["mode"] == "one_shot"
        assert next(params for method, params in core.calls if method == "event.subscribe")["topics"] == ["*"]
        submission = next(params for method, params in core.calls if method == "session.send_message")
        assert submission["content"] == "read"
        assert submission["client_message_id"]
        assert_client_only(core)
    assert code == 0
    assert "wire-final" in capsys.readouterr().out


@pytest.mark.parametrize("emit_failure", [False, True])
async def test_goal_recovered_tool_failure_still_has_nonzero_exit(emit_failure, capsys):
    async def submit(core, run_id):
        core.tool_failures[run_id] = 0 if emit_failure else 1
        core.tool_successes[run_id] = 1 if emit_failure else 0
        if emit_failure:
            await core.emit({"type": "tool.call_failed", "run_id": run_id,
                             "tool_use_id": "read-1", "tool_name": "read_file",
                             "error_class": "runtime_error", "error_message": "missing test file",
                             "elapsed_ms": 0, "attempt": 1, "retryable": False})
        await core.finish(run_id, text="recovered-result")

    async with CliCoreStub(submit) as core:
        code = await asyncio.wait_for(TerminalClient(core.config(), interactive=False).run(goal="repair"), 5)
    assert code == 1
    assert "recovered-result" in capsys.readouterr().out


async def test_failed_main_run_is_not_success():
    async def submit(core, run_id):
        core.runs[run_id]["reason"] = "llm_error"
        await core.finish(run_id, status="failed", text="")

    async with CliCoreStub(submit) as core:
        assert await asyncio.wait_for(TerminalClient(core.config(), interactive=False).run(goal="fail"), 5) == 1


async def test_concurrent_main_and_child_permissions_are_answered_separately(capsys):
    ready = asyncio.Event()

    async def submit(core, run_id):
        await core.permission(run_id, "approval-main")
        await core.child(run_id, "child-1")
        await core.permission("child-1", "approval-child")
        ready.set()

    async def answer(core, params):
        if len(core.approval_answers) == 2:
            await core.finish("child-1")
            await core.finish(core.main_run_id)

    reader = FakeInput()
    async with CliCoreStub(submit) as core:
        core.on_permission = answer
        client = TerminalClient(core.config(), interactive=True, reader=reader)
        task = asyncio.create_task(client.run(goal="two approvals"))
        await asyncio.wait_for(ready.wait(), 5)
        await wait_notice(capsys, "[approval: approval-main]")
        reader.feed("y")
        await core.wait_command("permission.respond")
        await wait_notice(capsys, "[approval: approval-child]")
        reader.feed("y")
        assert await result_of(task) == 0
        assert {(row["request_id"], row["decision"]) for row in core.approval_answers} == {
            ("approval-main", "allow_once"), ("approval-child", "allow_once"),
        }
        assert all(row["session_id"] == core.session_id for row in core.approval_answers)
        assert reader.closed


async def test_host_fallback_uses_the_separate_host_authorization(capsys):
    async def submit(core, run_id):
        await core.permission(run_id, "host-approval", host=True)

    async def answer(core, params):
        await core.finish(core.main_run_id)

    reader = FakeInput()
    async with CliCoreStub(submit) as core:
        core.on_permission = answer
        task = asyncio.create_task(TerminalClient(core.config(), interactive=True, reader=reader).run(goal="host"))
        await core.wait_command("session.send_message")
        await wait_notice(capsys, "[approval: host-approval]")
        reader.feed("y")
        assert await result_of(task) == 0
        assert core.approval_answers[0]["decision"] == "allow_host_once"


async def test_noninteractive_goal_denies_approval_instead_of_hanging():
    async def submit(core, run_id):
        await core.permission(run_id, "no-terminal")

    async def answer(core, params):
        core.tool_failures[core.main_run_id] = 1
        await core.finish(core.main_run_id, text="tool refused")

    async with CliCoreStub(submit) as core:
        core.on_permission = answer
        assert await asyncio.wait_for(TerminalClient(core.config(), interactive=False).run(goal="write"), 5) == 1
        assert [row["decision"] for row in core.approval_answers] == ["deny_once"]
        assert_client_only(core)


async def test_chat_interrupt_cancels_current_run_and_accepts_next_turn(capsys):
    async def submit(core, run_id):
        if run_id == "run-2":
            await core.finish(run_id, text="second-result")

    reader = FakeInput()
    async with CliCoreStub(submit) as core:
        client = TerminalClient(core.config(), interactive=True, reader=reader)
        task = asyncio.create_task(client.run())
        reader.feed("first")
        await core.wait_command("session.send_message")
        client.interrupt()
        await core.wait_command("run.cancel")
        await wait_notice(capsys, "[result: run-1]")
        reader.feed("second")
        await core.wait_command("session.send_message", 2)
        await wait_notice(capsys, "[result: run-2]")
        reader.feed(None)
        assert await result_of(task) == 0
        assert core.runs["run-1"]["status"] == "cancelled"
        assert core.runs["run-2"]["status"] == "succeeded"
        assert [params["content"] for name, params in core.calls if name == "session.send_message"] == ["first", "second"]
        assert_client_only(core)
        assert reader.closed


async def test_idle_chat_interrupt_exits_130_without_cancelling_session():
    reader = FakeInput()
    async with CliCoreStub() as core:
        client = TerminalClient(core.config(), interactive=True, reader=reader)
        task = asyncio.create_task(client.run())
        await core.wait_command("event.subscribe")
        client.interrupt()
        assert await result_of(task) == 130
        assert not any(method == "run.cancel" for method, _ in core.calls)
        assert_client_only(core)
        assert reader.closed


async def test_chat_eof_during_run_detaches_without_ending_the_run():
    async def submit(core, run_id):
        pass

    reader = FakeInput()
    async with CliCoreStub(submit) as core:
        task = asyncio.create_task(TerminalClient(core.config(), interactive=True, reader=reader).run())
        reader.feed("keep working")
        await core.wait_command("session.send_message")
        reader.feed(None)
        assert await result_of(task) == 0
        assert core.runs["run-1"]["status"] == "running"
        assert not any(method == "run.cancel" for method, _ in core.calls)
        assert_client_only(core)
        assert reader.closed


async def test_goal_interrupt_returns_130():
    async def submit(core, run_id):
        pass

    async with CliCoreStub(submit) as core:
        client = TerminalClient(core.config(), interactive=False)
        task = asyncio.create_task(client.run(goal="wait"))
        await core.wait_command("session.send_message")
        client.interrupt()
        assert await result_of(task) == 130
        assert core.runs["run-1"]["status"] == "cancelled"
        assert_client_only(core)


async def test_child_completion_does_not_finish_the_main_wait():
    ready = asyncio.Event()

    async def submit(core, run_id):
        await core.child(run_id, "child-1")
        await core.finish("child-1")
        ready.set()

    async with CliCoreStub(submit) as core:
        task = asyncio.create_task(TerminalClient(core.config(), interactive=False).run(goal="parent"))
        await asyncio.wait_for(ready.wait(), 5)
        done, _ = await asyncio.wait({task}, timeout=0.05)
        assert not done
        await core.finish("run-1")
        assert await result_of(task) == 0


async def test_goal_reports_background_child_without_shutdown(capsys):
    async def submit(core, run_id):
        await core.child(run_id, "child-background")
        await core.finish(run_id)

    async with CliCoreStub(submit) as core:
        assert await asyncio.wait_for(TerminalClient(core.config(), interactive=False).run(goal="background"), 5) == 0
        assert core.runs["child-background"]["status"] == "running"
        assert_client_only(core)
    assert "child-background" in capsys.readouterr().err


async def test_lost_submission_response_does_not_resubmit_the_goal():
    async def submit(core, run_id):
        core.writers[-1].close()

    async with CliCoreStub(submit) as core:
        assert await asyncio.wait_for(TerminalClient(core.config(), interactive=False).run(goal="one side effect"), 5) == 1
        assert sum(method == "session.send_message" for method, _ in core.calls) == 1
        assert len(core.runs) == 1
        assert_client_only(core)


async def test_resume_does_not_treat_resolved_old_permission_as_new_input(capsys):
    reader = FakeInput()
    async with CliCoreStub() as core:
        old = core._run_info("run-old")
        old.update(status="succeeded", result={"text": "old answer", "steps": 1})
        core.runs["run-old"] = old
        core.main_run_id = "run-old"
        core.replay = [
            {"type": "permission.requested", "run_id": "run-old", "request_id": "old-request",
             "session_id": core.session_id, "tool_use_id": "old-tool", "tool_name": "write_file",
             "params": {"path": "old.txt"}, "param_preview": "old.txt", "request_kind": "tool",
             "allowed_decisions": ["allow_once", "deny_once"], "backend": "workspace_sandbox", "risk": "high", "reason": "approval"},
            {"type": "permission.granted", "run_id": "run-old", "request_id": "old-request", "tool_use_id": "old-tool", "decision": "allow_once"},
            {"type": "run.finished", "run_id": "run-old", "status": "success", "reason": None, "steps": 1},
        ]
        task = asyncio.create_task(TerminalClient(core.config(), interactive=True, reader=reader).run(resume_session_id=core.session_id))
        await core.wait_command("event.subscribe")
        old_output = await wait_notice(capsys, "> ")
        assert "[approval: old-request]" not in old_output
        reader.feed("fresh task")
        await core.wait_command("session.send_message")
        await core.wait_command("run.metrics")
        reader.feed(None)
        assert await result_of(task) == 0
        assert not core.approval_answers
        assert not any(method == "session.create" for method, _ in core.calls)
        assert next(params for method, params in core.calls if method == "session.send_message")["content"] == "fresh task"

async def test_recovered_child_tool_failure_is_counted_for_the_goal():
    async def submit(core, run_id):
        await core.child(run_id, "child-recovered")
        await core.emit({"type": "tool.call_failed", "run_id": "child-recovered",
                         "tool_use_id": "child-tool", "tool_name": "read_file",
                         "error_class": "runtime_error", "error_message": "first read failed",
                         "elapsed_ms": 0, "attempt": 1, "retryable": False})
        await core.finish("child-recovered", text="child repaired the task")
        await core.finish(run_id, text="main succeeded")

    async with CliCoreStub(submit) as core:
        assert await asyncio.wait_for(TerminalClient(core.config(), interactive=False).run(goal="delegate"), 5) == 1

async def test_replay_limit_reconnect_keeps_cursor_and_resolves_old_approval(capsys):
    async def submit(core, run_id):
        pass

    async def subscribe(core, params):
        number = sum(name == "event.subscribe" for name, _ in core.calls)
        return {"subscription_id": f"sub-{number}", "replayed_count": 0 if number == 1 else 2,
                "high_water_cursor": 0 if number == 1 else 3, "replay_truncated": False}

    reader = FakeInput()
    async with CliCoreStub(submit) as core:
        core.on_subscribe = subscribe
        task = asyncio.create_task(TerminalClient(core.config(), interactive=True, reader=reader).run(goal="paged replay"))
        await core.wait_command("run.get")
        progress = {"type": "tool.call_started", "run_id": "run-1", "tool_use_id": "read-once",
                    "tool_name": "read_file", "params": {"path": "sample.txt"}}
        await core.emit(progress, cursor=1)
        await core.overflow(reason="replay_limit", last_cursor=1)
        await core.wait_command("event.subscribe", 2)
        await core.wait_command("run.get", 2)
        await core.emit(progress, cursor=1)  # The overlapping page repeats an already seen row.
        await core.emit({"type": "permission.requested", "run_id": "run-1", "request_id": "historical",
                         "session_id": core.session_id, "tool_use_id": "old-write", "tool_name": "write_file",
                         "params": {"path": "old.txt"}, "param_preview": "old.txt", "request_kind": "tool",
                         "allowed_decisions": ["allow_once", "deny_once"], "backend": "workspace_sandbox",
                         "risk": "high", "reason": "approval"}, cursor=2)
        await core.emit({"type": "permission.granted", "run_id": "run-1", "request_id": "historical",
                         "tool_use_id": "old-write", "decision": "allow_once"}, cursor=3)
        await core.finish("run-1")
        assert await result_of(task) == 0
        subscriptions = [params for name, params in core.calls if name == "event.subscribe"]
        assert subscriptions[1]["after_cursor"] == 1
        assert sum(name == "session.send_message" for name, _ in core.calls) == 1
        assert sum(name == "session.create" for name, _ in core.calls) == 1
        assert not core.approval_answers
        assert_client_only(core)
    output = capsys.readouterr()
    assert output.err.count("read_file") == 1
    assert "[approval: historical]" not in output.err
    assert output.out.count("wire-final") == 1

async def test_replay_before_subscribe_response_does_not_block_new_approval(capsys):
    async def submit(core, run_id):
        pass

    async def subscribe(core, params):
        number = sum(name == "event.subscribe" for name, _ in core.calls)
        if number == 1:
            return {"subscription_id": "first", "replayed_count": 0,
                    "high_water_cursor": 0, "replay_truncated": False}
        await core.emit({"type": "permission.requested", "run_id": "run-1", "request_id": "old",
                         "session_id": core.session_id, "tool_use_id": "old-tool", "tool_name": "write_file",
                         "params": {"path": "old.txt"}, "request_kind": "tool",
                         "allowed_decisions": ["allow_once", "deny_once"]}, cursor=2)
        await core.emit({"type": "permission.granted", "run_id": "run-1", "request_id": "old",
                         "tool_use_id": "old-tool", "decision": "allow_once"}, cursor=3)
        # Core's event pump can deliver replay before the subscription RPC reply.
        await asyncio.sleep(0.05)
        return {"subscription_id": "second", "replayed_count": 2,
                "high_water_cursor": 3, "replay_truncated": False}

    async def answer(core, params):
        await core.finish(core.main_run_id)

    reader = FakeInput()
    async with CliCoreStub(submit) as core:
        core.on_subscribe = subscribe
        core.on_permission = answer
        client = TerminalClient(core.config(), interactive=True, reader=reader)
        task = asyncio.create_task(client.run(goal="replay before reply"))
        try:
            await core.wait_command("run.get")
            await core.emit({"type": "step.started", "run_id": "run-1", "step": 1}, cursor=1)
            await core.overflow(reason="replay_limit", last_cursor=1)
            await core.wait_command("event.subscribe", 2)
            await core.wait_command("run.get", 2)
            await core.permission("run-1", "new-live")
            try:
                observed = await wait_notice(capsys, "[approval: new-live]")
            except TimeoutError:
                pytest.fail("new live approval was blocked after replay arrived before the subscription response")
            assert "[approval: old]" not in observed
            reader.feed("y")
            assert await result_of(task) == 0
            assert [row["request_id"] for row in core.approval_answers] == ["new-live"]
            assert sum(name == "session.send_message" for name, _ in core.calls) == 1
        finally:
            if not task.done():
                client.interrupt()
                await result_of(task)

async def test_goal_eof_cannot_override_an_explicit_interrupt():
    async def submit(core, run_id):
        pass

    async def slow_cancel(core, run_id):
        await asyncio.sleep(0.05)

    reader = FakeInput()
    async with CliCoreStub(submit) as core:
        core.on_cancel = slow_cancel
        client = TerminalClient(core.config(), interactive=True, reader=reader)
        task = asyncio.create_task(client.run(goal="cancel before input closes"))
        await core.wait_command("run.get")
        client.interrupt()
        reader.feed(None)
        assert await result_of(task) == 130
        assert sum(method == "run.cancel" for method, _ in core.calls) == 1
        assert core.runs["run-1"]["status"] == "cancelled"
