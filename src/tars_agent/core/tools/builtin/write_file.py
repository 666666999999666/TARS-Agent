from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from tars_agent.core.tools.runtime.router import RuntimeRouter
from tars_agent.core.tools.runtime.workspace import WorkspaceRuntimeTool


class WriteFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    content: str


class WriteFileTool(WorkspaceRuntimeTool):
    params_model = WriteFileParams
    name = "write_file"
    description = (
        "Write up to 1 MiB of UTF-8 text inside the workspace sandbox. Paths and "
        "symlinks may not escape the workspace."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "content": {"type": "string", "description": "Text content to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, runtime: RuntimeRouter) -> None:
        super().__init__(runtime)
