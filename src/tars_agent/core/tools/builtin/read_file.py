from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from tars_agent.core.tools.runtime.router import RuntimeRouter
from tars_agent.core.tools.runtime.workspace import WorkspaceRuntimeTool


class ReadFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str


class ReadFileTool(WorkspaceRuntimeTool):
    params_model = ReadFileParams
    name = "read_file"
    description = (
        "Read a UTF-8 text file inside the workspace sandbox. Paths and symlinks may "
        "not escape the workspace; files larger than 512 KiB are truncated."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."}
        },
        "required": ["path"],
    }

    def __init__(self, runtime: RuntimeRouter) -> None:
        super().__init__(runtime)
