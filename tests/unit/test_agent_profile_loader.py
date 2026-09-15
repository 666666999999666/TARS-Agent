from __future__ import annotations

from pathlib import Path

import pytest

from tars_agent.core.agents.loader import AgentProfileLoader


# 功能：内建 planner 角色配置应能被 AgentProfileLoader 加载
# 设计：直接调用 load("planner")，验证关键字段非空
def test_builtin_planner_found() -> None:
    loader = AgentProfileLoader()
    profile = loader.load("planner")
    assert profile is not None
    assert profile.name == "planner"
    assert profile.system_prompt != ""
    assert "read_file" in profile.allowed_tools or len(profile.allowed_tools) > 0


def test_builtin_reviewer_only_has_read_tools() -> None:
    profile = AgentProfileLoader().load("reviewer")
    assert profile is not None
    assert profile.allowed_tools == ["read_file", "list_dir"]


# 功能：内建三种角色均可加载
# 设计：参数化测试所有内建角色名
@pytest.mark.parametrize("role", ["planner", "executor", "reviewer"])
def test_all_builtin_roles_found(role: str) -> None:
    loader = AgentProfileLoader()
    profile = loader.load(role)
    assert profile is not None, f"builtin role '{role}' not found"
    assert profile.allowed_tools  # 每个内建角色都有 allowed_tools


# 功能：未知角色名应返回 None
# 设计：查找不存在的角色，断言返回 None 而非抛异常
def test_unknown_role_returns_none() -> None:
    loader = AgentProfileLoader()
    result = loader.load("nonexistent_role_xyz")
    assert result is None


# 功能：TOML 角色配置文件应被正确解析
# 设计：写入临时 TOML 文件，通过 _parse 解析并验证实际生效的字段
def test_toml_parsed(tmp_path: Path) -> None:
    content = """\
[agent]
description = "测试角色"
system_prompt = "你是测试助手。"
allowed_tools = ["read_file", "bash"]
"""
    p = tmp_path / "tester.toml"
    p.write_text(content, encoding="utf-8")
    loader = AgentProfileLoader()
    profile = loader._parse(p, "tester")
    assert profile.name == "tester"
    assert profile.description == "测试角色"
    assert profile.system_prompt == "你是测试助手。"
    assert "read_file" in profile.allowed_tools
    assert "bash" in profile.allowed_tools
    assert profile.model == ""


# 功能：项目本地角色配置应覆盖内建同名配置
# 设计：在 .kama/agents/ 中写入同名 TOML，monkeypatch cwd，断言加载到本地版本
def test_project_overrides_builtin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_agents = tmp_path / ".tars" / "agents"
    local_agents.mkdir(parents=True)
    (local_agents / "planner.toml").write_text(
        '[agent]\ndescription = "local planner"\nsystem_prompt = "local prompt"\n'
        'allowed_tools = ["list_dir"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    loader = AgentProfileLoader()
    profile = loader.load("planner")
    assert profile is not None
    assert profile.description == "local planner"
    assert "list_dir" in profile.allowed_tools


@pytest.mark.parametrize("name", ["../executor", "/executor", "a/b", "..", "NUL.txt", "NUL", "CON"])
def test_profile_name_cannot_escape_resource_directory(name: str) -> None:
    with pytest.raises(ValueError):
        AgentProfileLoader().load(name)


def test_empty_profile_does_not_grant_all_tools(tmp_path: Path) -> None:
    path = tmp_path / "empty.toml"
    path.write_text('[agent]\nallowed_tools = []\nmodel = "alternate"\n', encoding="utf-8")
    profile = AgentProfileLoader()._parse(path, "empty")
    assert profile.allowed_tools == []
    assert profile.allow_all_tools is False
    assert profile.model == "alternate"


def test_profile_read_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "huge.toml"
    path.write_text("#" * 70_000, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        AgentProfileLoader()._parse(path, "huge")
