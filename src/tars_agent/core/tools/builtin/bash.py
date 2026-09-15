from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tars_agent.core.tools.runtime.router import RuntimeRouter
from tars_agent.core.tools.runtime.workspace import WorkspaceRuntimeTool

_DEFAULT_TIMEOUT = 60


class BashParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    command: str
    timeout: int = Field(default=_DEFAULT_TIMEOUT, ge=1, le=120)


class BashTool(WorkspaceRuntimeTool):
    params_model = BashParams
    name = "bash"
    description = (
        "Execute a non-interactive POSIX shell command inside the workspace sandbox. "
        "The sandbox has no network and output is truncated at 64 KiB."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {
                "type": "integer",
                "description": f"Maximum seconds to wait (default {_DEFAULT_TIMEOUT}, max 120).",
            },
        },
        "required": ["command"],
    }

    def __init__(self, runtime: RuntimeRouter) -> None:
        super().__init__(runtime)
