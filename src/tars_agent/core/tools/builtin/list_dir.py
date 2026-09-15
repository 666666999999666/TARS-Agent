from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tars_agent.core.tools.runtime.router import RuntimeRouter
from tars_agent.core.tools.runtime.workspace import WorkspaceRuntimeTool

_MAX_DEPTH = 4


class ListDirParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str = "."
    max_depth: int = Field(default=2, ge=1, le=_MAX_DEPTH)


class ListDirTool(WorkspaceRuntimeTool):
    params_model = ListDirParams
    name = "list_dir"
    description = (
        "List up to 200 workspace entries as a tree. Paths and symlinks may not "
        f"escape the workspace; maximum depth is {_MAX_DEPTH}."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative directory path."},
            "max_depth": {
                "type": "integer",
                "description": f"Maximum recursion depth (default 2, max {_MAX_DEPTH}).",
            },
        },
        "required": [],
    }

    def __init__(self, runtime: RuntimeRouter) -> None:
        super().__init__(runtime)
