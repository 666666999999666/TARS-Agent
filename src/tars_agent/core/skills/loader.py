from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from tars_agent.core.agents.loader import validate_resource_name
from tars_agent.core.paths import tars_home


@dataclass
class Skill:
    name: str
    description: str
    system_prompt_template: str
    allowed_tools: list[str] = field(default_factory=list)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# 解析 Markdown skill 文件，提取 frontmatter 和正文 system prompt
def _parse_skill_file(path: Path) -> Skill:
    with path.open("rb") as handle:
        raw = handle.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        raise ValueError("skill exceeds 64 KiB")
    text = raw.decode("utf-8")
    name = path.stem
    description = ""
    allowed_tools: list[str] = []
    body = text

    m = _FRONTMATTER_RE.match(text)
    if m:
        front = m.group(1)
        body = text[m.end():]
        lines = front.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith("name:"):
                name = stripped[len("name:"):].strip().strip('"').strip("'")
            elif stripped.startswith("description:"):
                val = stripped[len("description:"):].strip().strip('"').strip("'")
                # YAML 块标量：> (折叠) 或 | (保留换行)，后续缩进行是内容
                if val in (">", "|"):
                    fold = val == ">"
                    parts: list[str] = []
                    i += 1
                    while i < len(lines) and (
                        lines[i].startswith(" ") or lines[i].startswith("\t")
                    ):
                        parts.append(lines[i].strip())
                        i += 1
                    description = (" ".join(parts) if fold else "\n".join(parts)).strip()
                    continue
                else:
                    description = val
            elif stripped.startswith("allowed_tools:"):
                pass
            elif stripped.startswith("- "):
                allowed_tools.append(stripped[2:].strip())
            i += 1

    validate_resource_name(name, "skill name")
    return Skill(
        name=name,
        description=description,
        system_prompt_template=body.strip(),
        allowed_tools=allowed_tools,
    )


# 按三级优先级（项目本地 > 用户全局 > 内建）查找并解析 skill
class SkillLoader:
    _BUILTIN_DIR = Path(__file__).parent / "builtin"

    def __init__(self, *, workspace_root: Path | None = None) -> None:
        # The caller chooses the project once; a daemon's cwd is not a session's project.
        self._project_dir = (workspace_root or Path.cwd()).expanduser().resolve() / ".tars/skills"

    def _roots(self) -> tuple[Path, Path, Path]:
        return self._project_dir, tars_home() / "skills", self._BUILTIN_DIR

    # 按优先级查找 skill 文件；未找到返回 None
    def resolve(self, name: str) -> Skill | None:
        validate_resource_name(name, "skill name")
        for path in self._search_paths(name):
            if not self._within_roots(path):
                raise ValueError("skill path escapes its resource root")
            if path.exists():
                try:
                    return _parse_skill_file(path)
                except Exception:
                    return None
        return None

    def _within_roots(self, path: Path) -> bool:
        absolute = path.absolute()
        for root in self._roots():
            expected = root.absolute()
            if absolute.is_relative_to(expected):
                return (root.resolve() == expected and not path.is_symlink()
                        and path.resolve().is_relative_to(expected))
        return False

    # 返回候选路径列表，同时支持扁平文件（name.md）和目录式（name/SKILL.md）两种格式
    def _search_paths(self, name: str) -> list[Path]:
        paths: list[Path] = []
        for d in self._roots():
            paths.append(d / f"{name}.md")
            paths.append(d / name / "SKILL.md")
        return paths

    # 列出所有可用 skill 名称（内建 + 用户全局 + 项目本地，去重后以项目本地覆盖为准）
    def list_all(self) -> list[str]:
        seen: dict[str, None] = {}
        for d in reversed(self._roots()):
            if d.exists():
                for f in sorted(d.glob("*.md")):
                    if self._within_roots(f):
                        seen[f.stem] = None
                for f in sorted(d.glob("*/SKILL.md")):
                    if self._within_roots(f):
                        seen[f.parent.name] = None
        return list(seen)

    # 列出所有可用 Skill 对象（含描述），项目本地覆盖同名内建
    def list_all_skills(self) -> list[Skill]:
        seen: dict[str, Skill] = {}
        for d in reversed(self._roots()):
            if d.exists():
                for f in sorted(d.glob("*.md")):
                    try:
                        if not self._within_roots(f):
                            continue
                        skill = _parse_skill_file(f)
                        seen[skill.name] = skill
                    except Exception:
                        pass
                for f in sorted(d.glob("*/SKILL.md")):
                    try:
                        if not self._within_roots(f):
                            continue
                        skill = _parse_skill_file(f)
                        seen[skill.name] = skill
                    except Exception:
                        pass
        return list(seen.values())

    # 将 $ARGUMENTS 替换为用户传入的参数字符串
    def render_prompt(self, skill: Skill, arguments: str) -> str:
        return skill.system_prompt_template.replace("$ARGUMENTS", arguments)
