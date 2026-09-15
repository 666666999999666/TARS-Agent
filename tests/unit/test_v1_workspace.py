from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tars_agent.core.bus.envelope import HandlerError
from tars_agent.core.events.bus import EventBus
from tars_agent.core.persistence import Database
from tars_agent.core.runner import RunOutcome
from tars_agent.core.runtime.service import RuntimeService
from tars_agent.core.skills.loader import SkillLoader


class CapturingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run_and_capture(self, goal: str, **kwargs: Any) -> RunOutcome:
        self.calls.append((goal, kwargs))
        return RunOutcome(status="success", result="done", reason=None)


def write_skill(root: Path, label: str) -> None:
    skills = root / ".tars" / "skills"
    skills.mkdir(parents=True)
    (skills / "inspect.md").write_text(
        "---\nname: inspect\ndescription: inspect workspace\n"
        "allowed_tools:\n  - read_file\n---\n" + label + " $ARGUMENTS",
        encoding="utf-8",
    )


async def test_skill_uses_session_workspace_instead_of_daemon_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, workspace = tmp_path / "daemon", tmp_path / "workspace"
    write_skill(daemon, "wrong-project")
    write_skill(workspace, "session-project")
    monkeypatch.chdir(daemon)
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    runner = CapturingRunner()
    runtime = RuntimeService(
        database, lambda: runner, EventBus(),  # type: ignore[arg-type,return-value]
        artifacts_root=tmp_path / "artifacts",
    )
    try:
        session = await runtime.create_session("chat", workspace_root=workspace)
        submitted = await runtime.submit_message(session.id, "/inspect input.txt")
        await runtime.supervisor.wait(submitted.run_id)
        assert runner.calls[0][0] == "session-project input.txt"
        assert runner.calls[0][1]["workspace_root"] == workspace.resolve()
        assert runner.calls[0][1]["tool_whitelist"] == ["read_file"]
    finally:
        await runtime.shutdown()
        await database.dispose()


@pytest.mark.parametrize("content", ["/", "/   "])
async def test_empty_skill_command_is_rejected_without_starting_a_run(
    tmp_path: Path, content: str,
) -> None:
    database = Database(tmp_path / "state.db")
    await database.create_schema()
    runner = CapturingRunner()
    runtime = RuntimeService(
        database, lambda: runner, EventBus(),  # type: ignore[arg-type,return-value]
        artifacts_root=tmp_path / "artifacts",
    )
    try:
        session = await runtime.create_session("chat", workspace_root=tmp_path)
        with pytest.raises(HandlerError, match="skill name"):
            await runtime.submit_message(session.id, content)
        assert runner.calls == []
        assert (await runtime.get_session(session.id)).status == "ready"
    finally:
        await runtime.shutdown()
        await database.dispose()


def test_skill_listing_and_resolution_share_explicit_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, workspace = tmp_path / "daemon", tmp_path / "workspace"
    write_skill(daemon, "wrong-project")
    write_skill(workspace, "session-project")
    monkeypatch.chdir(daemon)
    loader = SkillLoader(workspace_root=workspace)
    assert "inspect" in loader.list_all()
    skill = loader.resolve("inspect")
    assert skill is not None and skill.system_prompt_template == "session-project $ARGUMENTS"
    listed = next(skill for skill in loader.list_all_skills() if skill.name == "inspect")
    assert listed.system_prompt_template == skill.system_prompt_template
