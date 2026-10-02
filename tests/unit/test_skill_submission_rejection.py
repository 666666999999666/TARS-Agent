from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from tars_agent.core.bus.envelope import HandlerError
from tars_agent.core.events.bus import EventBus
from tars_agent.core.persistence import Database, MessageRecord, RunRecord, TurnRecord
from tars_agent.core.runner import RunOutcome
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.skills.loader import SkillLoader


@pytest.mark.parametrize("failure", [
    "missing", "empty_name", "invalid_name", "invalid_utf8", "oversize", "directory",
    "invalid_declared_name",
])
async def test_rejected_skill_creates_no_run_turn_message_or_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    monkeypatch.setenv("TARS_HOME", str(tmp_path / "home"))
    local = tmp_path / ".tars" / "skills"
    local.mkdir(parents=True)
    command = "/review target.txt"
    if failure == "missing":
        command = "/unknown-skill target.txt"
    elif failure == "empty_name":
        command = "/"
    elif failure == "invalid_name":
        command = "/../private target.txt"
    elif failure == "invalid_utf8":
        (local / "review.md").write_bytes(b"\xff")
    elif failure == "oversize":
        (local / "review.md").write_bytes(b"x" * (64 * 1024 + 1))
    elif failure == "directory":
        (local / "review.md").mkdir()
    else:
        (local / "review.md").write_text("---\nname: ../private\n---\nprompt", encoding="utf-8")

    calls: list[str] = []

    class Runner:
        async def run_and_capture(self, goal: str, **kwargs: Any) -> RunOutcome:
            calls.append(goal)
            return RunOutcome(status="success", result="done", reason=None)

    database = Database(tmp_path / "state.db")
    await database.create_schema()
    bus = EventBus()
    events: list[str] = []

    async def observe(event: Any) -> None:
        events.append(event.type)

    bus.subscribe(observe)
    runtime = RuntimeService(
        database, Runner, bus, artifacts_root=tmp_path / "artifacts",  # type: ignore[arg-type]
    )
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        with pytest.raises(HandlerError) as caught:
            await runtime.submit_message(session.id, command, client_message_id="correctable")
        assert caught.value.code == -32602
        assert "本次任务未启动" in str(caught.value)
        assert calls == []
        assert not any(kind in events for kind in ("run.started", "skill.invoked", "session.message_received"))
        async with database.session() as sql:
            for model in (RunRecord, TurnRecord, MessageRecord):
                assert await sql.scalar(select(func.count()).select_from(model)) == 0
        current = await runtime.get_session(session.id)
        assert current.status == session.status
        assert current.active_run_id is None
        # A rejected attempt did not consume the client identity or strand the session.
        result = await runtime.submit_message(session.id, "ordinary input", client_message_id="correctable")
        await runtime.supervisor.wait(result.run_id)
        assert calls == ["ordinary input"]
        assert (await runtime.get_run(result.run_id)).status == "succeeded"
    finally:
        await runtime.shutdown()
        await database.dispose()


def test_invalid_local_skill_does_not_fall_back_to_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TARS_HOME", str(tmp_path / "home"))
    local = tmp_path / ".tars" / "skills"
    local.mkdir(parents=True)
    (local / "review.md").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="cannot load skill"):
        SkillLoader(workspace_root=tmp_path).resolve("review")


def test_plain_text_skill_still_resolves_with_empty_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TARS_HOME", str(tmp_path / "home"))
    local = tmp_path / ".tars" / "skills"
    local.mkdir(parents=True)
    (local / "plain.md").write_text("请回答：$ARGUMENTS", encoding="utf-8")
    loader = SkillLoader(workspace_root=tmp_path)
    skill = loader.resolve("plain")
    assert skill is not None
    assert loader.render_prompt(skill, "hello") == "请回答：hello"
    assert skill.allowed_tools == []


async def test_cli_skill_rejection_keeps_nonzero_exit_and_does_not_resubmit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from tars_agent.cli.client import TerminalClient
    from tars_agent.core.transport.socket_client import IpcError, SocketClient
    from tests.cli_core_stub import CliCoreStub

    send = SocketClient.send_command
    submissions: list[dict[str, Any]] = []

    async def reject_skill(self: SocketClient, method: str, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if method == "session.send_message":
            submissions.append(params)
            raise IpcError(-32602, "未找到 Skill /missing；本次任务未启动")
        return await send(self, method, params, **kwargs)

    monkeypatch.setattr(SocketClient, "send_command", reject_skill)
    async with CliCoreStub() as core:
        code = await asyncio.wait_for(
            TerminalClient(core.config(), interactive=False).run(goal="/missing"), 5,
        )
        assert not core.runs
    assert code == 1
    assert len(submissions) == 1
    assert "本次任务未启动" in capsys.readouterr().err
