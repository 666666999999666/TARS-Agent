from __future__ import annotations

from tars_agent.core.task.manager import TaskManager
from tars_agent.core.tools.base import BaseTool
from tars_agent.core.tools.builtin import (
    BashTool,
    ListDirTool,
    ReadFileTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    WriteFileTool,
)
from tars_agent.core.tools.registry import ToolRegistry
from tars_agent.core.tools.runtime import RuntimeRouter


def build_base_registry(
    runtime: RuntimeRouter,
    task_manager: TaskManager,
    *,
    allowed_tools: set[str] | None = None,
) -> ToolRegistry:
    """Build the same workspace and checklist tools for main and child runs."""
    tools: list[BaseTool] = [
        ReadFileTool(runtime),
        BashTool(runtime),
        WriteFileTool(runtime),
        ListDirTool(runtime),
        TaskCreateTool(task_manager),
        TaskUpdateTool(task_manager),
        TaskListTool(task_manager),
        TaskGetTool(task_manager),
    ]
    registry = ToolRegistry()
    for tool in tools:
        if allowed_tools is None or tool.name in allowed_tools:
            registry.register(tool)
    return registry
