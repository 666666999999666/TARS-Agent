from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

from tars_agent.core.paths import tars_home


def validate_resource_name(name: str, label: str) -> None:
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name)
            or PureWindowsPath(name).is_reserved()):
        raise ValueError(f"invalid {label}")


@dataclass
class AgentProfile:
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    allow_all_tools: bool = False
    model: str = ""


# 按三级优先级（项目本地 > 用户全局 > 内建）查找并解析角色配置
class AgentProfileLoader:
    _BUILTIN_DIR = Path(__file__).parent / "builtin"

    # 查找指定角色配置；未找到返回 None
    def load(self, name: str, *, workspace_root: Path | None = None) -> AgentProfile | None:
        validate_resource_name(name, "agent name")
        roots = [(workspace_root or Path.cwd()), tars_home(), self._BUILTIN_DIR]
        for root, path in zip(roots, self._search_paths(name, workspace_root=workspace_root)):
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"agent profile escapes resource root: {name!r}")
            if path.exists():
                try:
                    return self._parse(path, name)
                except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid agent profile {name!r}: {exc}") from exc
        return None

    # 返回 [项目本地, 用户全局, 内建] 路径；load() 返回第一个存在的，项目本地优先级最高
    def _search_paths(self, name: str, *, workspace_root: Path | None = None) -> list[Path]:
        builtin = self._BUILTIN_DIR / f"{name}.toml"
        global_ = tars_home() / "agents" / f"{name}.toml"
        local = (workspace_root or Path.cwd()) / ".tars/agents" / f"{name}.toml"
        return [local, global_, builtin]

    # 解析 TOML 角色配置文件
    def _parse(self, path: Path, name: str) -> AgentProfile:
        with path.open("rb") as handle:
            raw = handle.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ValueError("agent profile exceeds 64 KiB")
        data = tomllib.loads(raw.decode("utf-8"))
        agent = data.get("agent")
        if not isinstance(agent, dict):
            raise ValueError("[agent] must be a table")

        description = agent.get("description", "")
        system_prompt = agent.get("system_prompt", "")
        model = agent.get("model", "")
        allowed_tools = agent.get("allowed_tools", [])
        allow_all_tools = agent.get("allow_all_tools", False)
        if not all(isinstance(value, str) for value in (description, system_prompt, model)):
            raise ValueError("description, system_prompt, and model must be strings")
        if not isinstance(allowed_tools, list) or not all(
            isinstance(tool, str) and tool for tool in allowed_tools
        ):
            raise ValueError("allowed_tools must be an array of non-empty strings")
        if not isinstance(allow_all_tools, bool):
            raise ValueError("allow_all_tools must be a boolean")
        if allow_all_tools and allowed_tools:
            raise ValueError("allow_all_tools and allowed_tools are mutually exclusive")

        return AgentProfile(
            name=name,
            description=description,
            system_prompt=system_prompt.strip(),
            allowed_tools=allowed_tools,
            allow_all_tools=allow_all_tools,
            model=model,
        )
