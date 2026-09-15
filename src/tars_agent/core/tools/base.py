from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from tars_agent.core.tools.runtime.models import (
        ToolBackend,
        ToolCallContext,
        ToolExecutionProfile,
    )


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    # "runtime_error" | "timeout" | "schema_error" | "permission_denied"
    error_type: str | None = None
    retryable: bool | None = None
    backend: ToolBackend | None = None
    started: bool | None = None
    container_id: str | None = None


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict[str, object]
    params_model: ClassVar[type[BaseModel] | None] = None
    execution_profile: ClassVar[ToolExecutionProfile] = "in_process"

    # 执行工具调用，返回结果或错误
    @abstractmethod
    async def invoke(
        self,
        params: dict[str, object],
        *,
        context: ToolCallContext | None = None,
    ) -> ToolResult: ...
