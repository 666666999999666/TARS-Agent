from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field

from tars_agent.core.persistence.models import SessionMode

DurableSessionStatus = Literal["ready", "running", "closed"]
RunStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
]


class PingCommand(BaseModel):
    type: Literal["core.ping"] = "core.ping"
    client: str


class PongResult(BaseModel):
    server_version: str
    uptime_ms: int
    received_at: str  # ISO 8601
    protocol_version: Literal[2] = 2
    launch_id: str | None = None
    schema_revision: str
    event_schema_versions: list[int] = Field(default_factory=lambda: [1])


class CoreShutdownCommand(BaseModel):
    type: Literal["core.shutdown"] = "core.shutdown"
    token: str = Field(min_length=1)


class CoreShutdownResult(BaseModel):
    ok: bool = True


class AgentRunCommand(BaseModel):
    type: Literal["agent.run"] = "agent.run"
    goal: str
    workspace_root: str | None = None


class AgentRunResult(BaseModel):
    run_id: str


class EventSubscribeCommand(BaseModel):
    type: Literal["event.subscribe"] = "event.subscribe"
    topics: list[str] = Field(default_factory=lambda: ["*"])
    session_id: str | None = None
    run_id: str | None = None
    after_cursor: int = Field(default=0, ge=0)


class EventSubscribeResult(BaseModel):
    subscription_id: str
    replayed_count: int = 0
    high_water_cursor: int = Field(ge=0)
    replay_truncated: bool = False


class SessionCreateCommand(BaseModel):
    type: Literal["session.create"] = "session.create"
    mode: SessionMode = "chat"
    title: str = ""
    workspace_root: str | None = None


class SessionCreateResult(BaseModel):
    session_id: str
    status: DurableSessionStatus
    workspace_root: str


class SessionInfo(BaseModel):
    session_id: str
    mode: str
    status: DurableSessionStatus
    title: str
    workspace_root: str | None
    active_run_id: str | None
    created_at: str
    updated_at: str
    closed_at: str | None


class SessionListCommand(BaseModel):
    type: Literal["session.list"] = "session.list"
    status: DurableSessionStatus | None = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class SessionListResult(BaseModel):
    sessions: list[SessionInfo]


class SessionResumeCommand(BaseModel):
    type: Literal["session.resume"] = "session.resume"
    session_id: str


class SessionGetCommand(BaseModel):
    type: Literal["session.get"] = "session.get"
    session_id: str


class SessionSendMessageCommand(BaseModel):
    type: Literal["session.send_message"] = "session.send_message"
    session_id: str
    content: str
    client_message_id: str | None = None


class SessionSendMessageResult(BaseModel):
    run_id: str
    status: RunStatus
    deduplicated: bool = False


class SessionGetHistoryCommand(BaseModel):
    type: Literal["session.get_history"] = "session.get_history"
    session_id: str
    after_sequence: int | None = Field(default=None, ge=-1)
    limit: int = Field(default=2_000, ge=1, le=2_000)


class SessionGetHistoryResult(BaseModel):
    messages: list[dict[str, Any]]


class SessionCloseCommand(BaseModel):
    type: Literal["session.close"] = "session.close"
    session_id: str


class SessionCloseResult(BaseModel):
    status: DurableSessionStatus


class SessionRetryCommand(BaseModel):
    type: Literal["session.retry"] = "session.retry"
    run_id: str
    confirm_side_effects: bool = False


class SessionRetryResult(BaseModel):
    run_id: str
    status: RunStatus


class RunGetCommand(BaseModel):
    type: Literal["run.get"] = "run.get"
    run_id: str


class RunInfo(BaseModel):
    run_id: str
    session_id: str
    turn_id: str | None
    parent_run_id: str | None
    retry_of_run_id: str | None
    kind: str
    attempt: int
    status: RunStatus
    reason: str | None
    side_effects_started: bool
    result: dict[str, Any] | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class SessionResumeResult(BaseModel):
    session: SessionInfo
    active_run: RunInfo | None = None
    latest_cursor: int = Field(ge=0)


class SessionGetResult(BaseModel):
    session: SessionInfo
    latest_run: RunInfo | None = None
    latest_cursor: int = Field(ge=0)


class RunMetricsCommand(BaseModel):
    type: Literal["run.metrics"] = "run.metrics"
    run_id: str


class RunCancelCommand(BaseModel):
    type: Literal["run.cancel"] = "run.cancel"
    run_id: str


class PermissionRespondCommand(BaseModel):
    type: Literal["permission.respond"] = "permission.respond"
    request_id: str
    session_id: str
    decision: Literal[
        "allow_once",
        "allow_session",
        "deny_once",
        "deny_session",
        "allow_host_once",
    ]


class PermissionRespondResult(BaseModel):
    ok: bool = True


class McpStatusCommand(BaseModel):
    type: Literal["mcp.status"] = "mcp.status"


class McpServerInfo(BaseModel):
    name: str
    transport: str
    status: str
    tool_count: int = Field(ge=0)
    protocol_version: str | None = None
    error: str | None = None


class McpStatusResult(BaseModel):
    servers: list[McpServerInfo]


class SessionCompactCommand(BaseModel):
    type: Literal["session.compact"] = "session.compact"
    session_id: str
    focus: str = ""


class SessionCompactResult(BaseModel):
    summary_tokens: int
    saved_tokens: int


# 根据 type 字段决定命令类型的判别联合
Command = Annotated[
    PingCommand
    | CoreShutdownCommand
    | AgentRunCommand
    | EventSubscribeCommand
    | SessionCreateCommand
    | SessionListCommand
    | SessionResumeCommand
    | SessionGetCommand
    | SessionSendMessageCommand
    | SessionGetHistoryCommand
    | SessionCloseCommand
    | SessionRetryCommand
    | RunGetCommand
    | RunCancelCommand
    | RunMetricsCommand
    | PermissionRespondCommand
    | McpStatusCommand
    | SessionCompactCommand,
    Discriminator("type"),
]
