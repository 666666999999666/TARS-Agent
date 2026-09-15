"""Real IPC acceptance workflows. Without --execute this only prints a plan."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import shlex
import socket
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tars_agent.core.control import (
    CORE_LAUNCH_ID_ENV,
    DaemonControl,
    read_control_file,
)
from tars_agent.core.persistence.request_budget import (
    REQUEST_LIMIT,
    ModelRequestBudgetExceeded,
    RequestKind,
    RequestLedger,
)
from tars_agent.core.processes import subprocess_group_kwargs, terminate_popen_process_tree
from tars_agent.core.transport.socket_client import SocketClient

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT / "build" / "v1-simplify" / "real-workflows"
DEFAULT_SCRIPT_LIMIT = 75
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
RUN_TIMEOUT = 180.0


def now() -> str:
    return datetime.now(UTC).isoformat()


def json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextmanager
def database(path: Path, *, write: bool = False) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path.resolve().as_uri() + ("?mode=rw" if write else "?mode=ro"),
                                 uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def counts(path: Path) -> dict[str, int]:
    with database(path) as sql:
        result = {"real": 0, "probe": 0}
        result.update({str(row[0]): int(row[1]) for row in sql.execute(
            "SELECT kind, count(*) FROM requests GROUP BY kind"
        )})
        return result


def parse_request_limit(value: str) -> int | None:
    if value.strip().lower() == "unlimited":
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("request-limit必须是正整数或unlimited") from exc
    if limit <= 0:
        raise argparse.ArgumentTypeError("request-limit必须是正整数或unlimited")
    return limit


def effective_request_limit(script_limit: int | None, production_limit: int | None) -> int | None:
    finite = [value for value in (script_limit, production_limit) if value is not None]
    return min(finite) if finite else None


def limit_argument(limit: int | None) -> str:
    return "unlimited" if limit is None else str(limit)


class AcceptanceLedger(RequestLedger):
    """Script-only send gate; same ledger, no recreated tables, no refunded requests."""

    def __init__(
        self, path: Path, *, limit: int | None = REQUEST_LIMIT,
        script_limit: int | None = DEFAULT_SCRIPT_LIMIT,
    ) -> None:
        super().__init__(path, limit=effective_request_limit(script_limit, limit))

    def reserve(self, kind: RequestKind = "real") -> int:
        with database(self.path, write=True) as sql:
            sql.execute("BEGIN IMMEDIATE")
            used = int(sql.execute("SELECT count(*) FROM requests WHERE kind=?", (kind,)).fetchone()[0])
            if kind == "real" and self.limit is not None and used >= self.limit:
                raise ModelRequestBudgetExceeded(
                    f"Acceptance limit {self.limit} reached; request not sent"
                )
            sql.execute("INSERT INTO requests(kind, reserved_at) VALUES (?, ?)", (kind, now()))
            sql.commit()
            return used + 1


def acceptance_ledger_type(script_limit: int | None) -> type[AcceptanceLedger]:
    class ConfiguredAcceptanceLedger(AcceptanceLedger):
        def __init__(self, path: Path, *, limit: int | None = REQUEST_LIMIT) -> None:
            super().__init__(path, limit=limit, script_limit=script_limit)
    return ConfiguredAcceptanceLedger


class WorkflowFailure(RuntimeError):
    pass


class BudgetStop(WorkflowFailure):
    pass


class EnvironmentBlocked(WorkflowFailure):
    pass


def fixed_configuration() -> tuple[Any, Path, Path]:
    from tars_agent.core.config import get_config
    from tars_agent.core.paths import tars_home
    home = (Path.home() / ".tars-baseline").resolve()
    if tars_home() != home:
        raise EnvironmentBlocked("TARS_HOME必须仍指向原来的~/.tars-baseline，不能为验收换HOME")
    explicit = os.environ.get("TARS_CONFIG")
    if explicit and Path(explicit).expanduser().resolve() != home / "config.toml":
        raise EnvironmentBlocked("验收只读取固定HOME/config.toml")
    try:
        config = get_config()
    except SystemExit as exc:
        raise EnvironmentBlocked("受信配置校验失败，请在本机检查；报告不显示凭证") from exc
    ledger = home / "acceptance" / "request-budget.sqlite3"
    if config.llm.request_budget_path.resolve() != ledger or not ledger.is_file():
        raise EnvironmentBlocked("必须复用已有账本；缺失时拒绝创建或重置")
    if config.sandbox.mode != "required":
        raise EnvironmentBlocked("真实工具验收必须使用sandbox.mode=required")
    if config.llm.default_model != "deepseek-flash":
        raise EnvironmentBlocked("本轮约定模型为deepseek-flash")
    if config.llm.base_url.rstrip("/") != "https://api.deepseek.com/anthropic":
        raise EnvironmentBlocked("本轮约定为DeepSeek官方Anthropic兼容端点")
    if config.mcp.servers:
        raise EnvironmentBlocked("本脚本仅验收本地工具，不启动额外MCP服务")
    return config, home, ledger


def isolated_state_home(path: Path, config_home: Path) -> Path:
    resolved = path.expanduser().resolve()
    source = config_home.resolve()
    if resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise EnvironmentBlocked("验收状态与证据目录必须独立于原模型配置目录")
    return resolved


@dataclass
class ApprovalPolicy:
    session_id: str
    workspace: Path
    writable: set[str] = field(default_factory=set)
    denied: set[str] = field(default_factory=set)
    commands: set[str] = field(default_factory=set)

    def decide(self, event: dict[str, Any]) -> str | None:
        if event.get("session_id") != self.session_id:
            return None
        if event.get("request_kind", "tool") != "tool":
            return "deny_once"
        params = event.get("params", {})
        tool = event.get("tool_name")
        if tool == "bash":
            timeout = params.get("timeout", 30)
            allowed = (params.get("command") in self.commands and isinstance(timeout, (int, float))
                       and not isinstance(timeout, bool) and 0 < timeout <= 30)
            return "allow_once" if allowed else "deny_once"
        if tool == "note_save":
            return "allow_once"
        if tool not in {"read_file", "list_dir", "write_file"}:
            return "deny_once"
        try:
            raw = Path(str(params.get("path", ".")))
            path = (raw if raw.is_absolute() else self.workspace / raw).resolve()
            relative = path.relative_to(self.workspace.resolve()).as_posix()
        except (ValueError, OSError):
            return "deny_once"
        if tool == "write_file" and (relative in self.denied or relative not in self.writable):
            return "deny_once"
        return "allow_once"


class Workflows:
    def __init__(
        self, retain: bool, request_limit: int | None = DEFAULT_SCRIPT_LIMIT,
        *, state_home: Path | None = None, output_root: Path = DEFAULT_OUTPUT_ROOT,
    ) -> None:
        self.request_limit = request_limit
        self.effective_limit: int | None = request_limit
        self.config_home = (Path.home() / ".tars-baseline").resolve()
        output_root = isolated_state_home(output_root, self.config_home)
        self.directory = output_root / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(3)
        )
        self.home = isolated_state_home(state_home or self.directory / "state", self.config_home)
        if self.home.exists():
            raise EnvironmentBlocked("state-home必须是新目录，拒绝复用已有会话或数据")
        self.directory.mkdir(parents=True, exist_ok=False)
        if self.home != self.directory:
            self.home.mkdir(parents=True, exist_ok=False)
        self.retain = retain
        self.ledger = self.config_home / "acceptance" / "request-budget.sqlite3"
        self.client: SocketClient | None = None
        self.reader: asyncio.Task[None] | None = None
        self.approval_tasks: set[asyncio.Task[None]] = set()
        self.answered: set[str] = set()
        self.policies: dict[str, ApprovalPolicy] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.cursors: dict[str, int] = {}
        self.decisions: list[dict[str, Any]] = []
        self.active_runs: dict[str, str] = {}
        self.process: subprocess.Popen[bytes] | None = None
        self.control: DaemonControl | None = None
        self.port = 0
        self.docker_binary = "docker"
        self.report: dict[str, Any] = {
            "kind": "real_ipc_workflows", "started_at": now(), "status": "incomplete",
            "controller_pid": os.getpid(), "artifact_root": str(self.directory),
            "home": str(self.home), "state_home": str(self.home),
            "config_home": str(self.config_home), "ledger": str(self.ledger),
            "state_isolation": "SQLite会话、artifacts、control、日志和trace使用独立state-home；仅模型配置和请求账本复用原目录",
            "script_request_limit": request_limit, "request_limits_resolved": False,
            "production_default_request_limit": REQUEST_LIMIT, "processes": [], "cases": [],
            "all_nine_complete": False,
            "ui_boundary": "真实Core/Provider和IPC流程；脚本自动提交与审批，不替代用户操作Textual TUI的验收",
            "harness_control": "保持生产Core与Provider；脚本只绑定隔离状态目录、自动审批白名单及额外发送限制，原账本持续计数",
        }
        self.checkpoint()

    def checkpoint(self) -> None:
        json_file(self.directory / "manifest.json", self.report)
        for session_id, records in self.events.items():
            path = self.directory / "events" / f"{session_id}.jsonl"
            path.parent.mkdir(exist_ok=True)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                            encoding="utf-8")
        json_file(self.directory / "approvals.json", self.decisions)

    def configure_limit(self, production_limit: int | None) -> None:
        self.effective_limit = effective_request_limit(self.request_limit, production_limit)
        self.report.update(production_request_limit=production_limit,
                           effective_request_limit=self.effective_limit,
                           request_limits_resolved=True)

    def limit_reached(self) -> bool:
        return (self.effective_limit is not None
                and counts(self.ledger)["real"] >= self.effective_limit)

    def capacity(self) -> None:
        if self.limit_reached():
            raise BudgetStop(f"总计已达到有效限制{self.effective_limit}次，不再开始新的模型操作")

    async def rpc(self, method: str, params: dict[str, Any], timeout: float = 20) -> dict[str, Any]:
        if self.client is None:
            raise WorkflowFailure("IPC客户端未连接")
        return await self.client.send_command(method, params, timeout_s=timeout)

    async def connect(self) -> None:
        self.client = SocketClient("127.0.0.1", self.port)
        await self.client.connect()
        self.client.on_event_envelope(self.on_envelope)
        self.reader = asyncio.create_task(self.client.run_event_loop())

    async def disconnect(self) -> None:
        if self.client is not None:
            await self.client.close()
        if self.reader is not None:
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.approval_tasks:
            await asyncio.gather(*tuple(self.approval_tasks), return_exceptions=True)
        self.client = None
        self.reader = None

    async def on_envelope(self, envelope: dict[str, Any]) -> None:
        if envelope.get("kind") != "event":
            self.report.setdefault("stream_issues", []).append(envelope)
            return
        session_id = str(envelope.get("session_id", ""))
        if session_id not in self.policies:
            return
        cursor = int(envelope["cursor"])
        if cursor <= self.cursors.get(session_id, 0):
            return
        self.cursors[session_id] = cursor
        self.events[session_id].append(envelope)
        event = envelope["event"]
        if event.get("type") == "permission.requested":
            operation = asyncio.create_task(self.answer_permission(session_id, event))
            self.approval_tasks.add(operation)
            operation.add_done_callback(self.approval_tasks.discard)

    async def answer_permission(self, session_id: str, event: dict[str, Any]) -> None:
        request_id = str(event["request_id"])
        if request_id in self.answered:
            return
        decision = self.policies[session_id].decide(event)
        if decision is None:
            return
        self.answered.add(request_id)
        record = {"session_id": session_id, "run_id": event.get("run_id"),
                  "request_id": request_id, "request_kind": event.get("request_kind", "tool"),
                  "tool_name": event.get("tool_name"), "decision": decision, "at": now()}
        self.decisions.append(record)
        try:
            response = await self.rpc("permission.respond", {
                "session_id": session_id, "request_id": request_id, "decision": decision,
            })
            record["accepted"] = response.get("ok")
        except Exception as exc:
            record["error_type"] = type(exc).__name__

    async def start(self, *, guarded: bool = True) -> None:
        config, self.config_home, self.ledger = fixed_configuration()
        self.docker_binary = config.sandbox.docker_binary
        self.configure_limit(config.llm.request_limit)
        if guarded:
            self.capacity()
        for path in (self.home / "control").glob("tars-core-*.json"):
            existing = read_control_file(path)
            if existing is None:
                continue
            try:
                _reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", existing.port), 0.3,
                )
            except (OSError, TimeoutError):
                continue
            writer.close()
            await writer.wait_closed()
            raise EnvironmentBlocked("验收state-home已有活动daemon；脚本不会接管")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = int(listener.getsockname()[1])
        launch_id = secrets.token_urlsafe(24)
        env = os.environ.copy()
        env.update({"TARS_HOME": str(self.config_home),
                    "TARS_CONFIG": str(self.config_home / "config.toml"),
                    "TARS_HOST": "127.0.0.1", "TARS_PORT": str(self.port), CORE_LAUNCH_ID_ENV: launch_id})
        if guarded:
            env["TARS_MAX_STEPS"] = "8"
        log_path = self.directory / f"daemon-{len(self.report['processes']) + 1}.log"
        flags = subprocess_group_kwargs()
        if os.name == "nt":
            flags["creationflags"] = flags.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        command = [sys.executable, str(Path(__file__).resolve()), "--daemon-child",
                   "--state-home", str(self.home),
                   "--request-limit", limit_argument(self.request_limit)]
        if not guarded:
            command.append("--unguarded")
        with log_path.open("ab") as log:
            self.process = subprocess.Popen(command, cwd=self.directory, env=env,
                                             stdout=log, stderr=log, **flags)
        process_record = {"launcher_pid": self.process.pid, "port": self.port,
                          "launch_id": launch_id, "log": str(log_path), "started_at": now(),
                          "acceptance_send_guard": guarded,
                          "production_request_limit": config.llm.request_limit,
                          "effective_request_limit": (self.effective_limit if guarded
                                                      else config.llm.request_limit)}
        self.report["processes"].append(process_record)
        self.checkpoint()
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise EnvironmentBlocked("daemon启动失败；查看专属日志，不能把required改为preferred")
            candidate = read_control_file(self.home / "control" / f"tars-core-{self.port}.json")
            if candidate is not None and candidate.launch_id == launch_id:
                await self.connect()
                pong = await self.rpc("core.ping", {"client": "acceptance-workflows"})
                if pong.get("launch_id") != launch_id:
                    raise EnvironmentBlocked("控制文件与真实RPC启动归属不一致")
                self.control = candidate
                process_record.update(daemon_pid=candidate.pid, ready_at=now())
                self.report.update(requested_model=config.llm.default_model,
                                   endpoint=config.llm.base_url, image=config.sandbox.image)
                self.checkpoint()
                return
            await asyncio.sleep(0.1)
        raise EnvironmentBlocked("daemon未在45秒内真正就绪")

    async def stop(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if self.control is not None and process.poll() is None:
                if self.client is None:
                    await self.connect()
                await self.rpc("core.shutdown", {"token": self.control.token}, 8)
        finally:
            await self.disconnect()
            try:
                await asyncio.to_thread(process.wait, 35)
            except subprocess.TimeoutExpired:
                await asyncio.to_thread(terminate_popen_process_tree, process, grace_s=2.0)
                self.report.setdefault("cleanup_issues", []).append("优雅关闭超时，仅回收本次进程树")
            code = process.poll()
            self.report["processes"][-1].update(stopped_at=now(), exit_code=code)
            if code != 0:
                self.report.setdefault("cleanup_issues", []).append("本次daemon未正常退出")
            self.process = None
            self.control = None
            self.checkpoint()

    async def session(self, workspace: Path, title: str) -> str:
        response = await self.rpc("session.create", {
            "mode": "chat", "workspace_root": str(workspace), "title": title,
        })
        session_id = str(response["session_id"])
        if not session_id.startswith("sess-") or not session_id[5:].isalnum():
            raise WorkflowFailure("服务返回了无效会话标识")
        self.policies[session_id] = ApprovalPolicy(session_id, workspace)
        self.events[session_id] = []
        self.cursors[session_id] = 0
        await self.rpc("event.subscribe", {"session_id": session_id, "after_cursor": 0})
        return session_id

    def session_run_count(self, session_id: str) -> int:
        with database(self.home / "state.db") as sql:
            row = sql.execute("SELECT count(*) FROM runs WHERE session_id=?", (session_id,)).fetchone()
            return int(row[0])

    async def resume(self, session_id: str) -> dict[str, Any]:
        before = self.session_run_count(session_id)
        result = await self.rpc("session.resume", {"session_id": session_id})
        await self.rpc("event.subscribe", {
            "session_id": session_id, "after_cursor": self.cursors[session_id],
        })
        if result["session"]["session_id"] != session_id or self.session_run_count(session_id) != before:
            raise WorkflowFailure("恢复创建了新会话或重复任务")
        return result

    def snapshot(self, session_id: str, run_id: str) -> dict[str, Any]:
        if session_id not in self.policies:
            raise WorkflowFailure("拒绝查询不属于本脚本的会话")
        with database(self.home / "state.db") as sql:
            run = sql.execute("SELECT * FROM runs WHERE id=? AND session_id=?", (run_id, session_id)).fetchone()
            if run is None:
                raise WorkflowFailure("RPC任务未在所属会话SQLite中找到")
            events = [dict(row) for row in sql.execute(
                "SELECT cursor,event_type,payload FROM events WHERE session_id=? ORDER BY cursor", (session_id,),
            )]
            tools = [dict(row) for row in sql.execute("SELECT * FROM tool_invocations WHERE run_id=?", (run_id,))]
            messages = [dict(row) for row in sql.execute(
                "SELECT role,committed,active FROM messages WHERE run_id=?", (run_id,),
            )]
        for event in events:
            event["payload"] = json.loads(event["payload"])
        for tool in tools:
            tool["parameters"] = json.loads(tool["parameters"])
        return {"database_run": dict(run), "events": events, "tools": tools, "messages": messages}

    async def submit(self, session_id: str, prompt: str) -> str:
        self.capacity()
        message_id = "acceptance-" + secrets.token_hex(12)
        submission = {"session_id": session_id, "client_message_id": message_id, "at": now()}
        self.report.setdefault("submissions", []).append(submission)
        self.checkpoint()
        try:
            response = await self.rpc("session.send_message", {
                "session_id": session_id, "content": prompt, "client_message_id": message_id,
            })
        except Exception:
            # Discover an ambiguous response in SQLite; never resend the last input.
            with database(self.home / "state.db") as sql:
                row = sql.execute("SELECT runs.id FROM runs JOIN turns ON runs.turn_id=turns.id "
                                  "WHERE turns.session_id=? AND turns.client_message_id=?",
                                  (session_id, message_id)).fetchone()
                if row:
                    self.active_runs[session_id] = str(row[0])
                    submission["run_id"] = str(row[0])
            raise
        run_id = str(response["run_id"])
        submission["run_id"] = run_id
        self.active_runs[session_id] = run_id
        return run_id

    async def completed(self, session_id: str, run_id: str, *, expected: str = "succeeded") -> dict[str, Any]:
        deadline = time.monotonic() + RUN_TIMEOUT
        while time.monotonic() < deadline:
            response = await self.rpc("run.get", {"run_id": run_id})
            if response["status"] in TERMINAL:
                evidence = self.snapshot(session_id, run_id)
                finished = [row for row in evidence["events"] if row["event_type"] == "run.finished"
                            and row["payload"].get("run_id") == run_id]
                if finished:
                    break
            await asyncio.sleep(0.1)
        else:
            await self.rpc("run.cancel", {"run_id": run_id}, 16)
            raise WorkflowFailure("任务或持久完成事件超过180秒，已请求取消")
        self.active_runs.pop(session_id, None)
        evidence.update(rpc_run=response, metrics=await self.rpc("run.metrics", {"run_id": run_id}),
                        budget_after=counts(self.ledger))
        pending_tool_ids = [str(tool["id"]) for tool in evidence["tools"]
                            if tool["run_id"] == run_id and tool["status"] in {"queued", "running"}]
        evidence["terminal_tool_check"] = {
            "run_id": run_id, "run_status": response["status"],
            "passed": not pending_tool_ids, "unfinished_tool_ids": pending_tool_ids,
        }
        json_file(self.directory / "runs" / f"{run_id}.json", evidence)
        self.checkpoint()
        if pending_tool_ids:
            raise WorkflowFailure("终态Run仍有queued/running工具记录：" + ", ".join(pending_tool_ids))
        if response["status"] != expected:
            if self.limit_reached():
                raise BudgetStop(f"有效限制{self.effective_limit}次已达到，未完成目标保留未验证")
            raise WorkflowFailure(f"任务终态为{response['status']}，要求{expected}；见任务证据")
        if evidence["database_run"]["status"] != response["status"]:
            raise WorkflowFailure("RPC与SQLite终态不一致")
        if any(tool["tool_name"] in {"read_file", "write_file", "list_dir", "bash"}
               and tool["backend"] != "workspace_sandbox" and tool["started_at"]
               for tool in evidence["tools"]):
            raise WorkflowFailure("发现宿主回退执行，不能计作隔离通过")
        text = (response.get("result") or {}).get("text", "")
        print(json.dumps({"session_id": session_id, "run_id": run_id,
                          "status": response["status"], "text": text}, ensure_ascii=False))
        return evidence

    async def task(self, session_id: str, prompt: str) -> dict[str, Any]:
        return await self.completed(session_id, await self.submit(session_id, prompt))

    @staticmethod
    def marked_file(path: Path, marker: str) -> dict[str, Any]:
        with path.open("rb") as stream:
            content = stream.read(65_537)
        if len(content) > 65_536 or marker not in content.decode("utf-8"):
            raise WorkflowFailure("实际文件未通过随机标记核验")
        return {"path": str(path), "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "marker_matched": True}

    @staticmethod
    def used(evidence: dict[str, Any], tool: str, path: str | None = None) -> bool:
        return any(row["tool_name"] == tool and row["status"] == "succeeded"
                   and (path is None or str(row["parameters"].get("path", "")).replace("\\", "/").removeprefix("./") == path) for row in evidence["tools"])

    async def file_case(self, workspace: Path, marker: str, case: dict[str, Any]) -> None:
        session_id = await self.session(workspace, case["name"])
        case["session_id"] = session_id
        self.policies[session_id].writable.add("报告.md")
        evidence = await self.task(session_id,
            "只使用list_dir/read_file/write_file/note_save。先列出当前目录，读取源数据.txt中的随机标记，"
            "将标记原样写入报告.md，再调用read_file读回报告.md核对，最后用note_save保存该标记。"
            "不能只描述计划；最后用中文说明实际完成步骤。所有路径均相对当前工作区。")
        for tool, path in [("list_dir", None), ("read_file", "源数据.txt"), ("write_file", "报告.md"),
                           ("read_file", "报告.md"), ("note_save", None)]:
            if not self.used(evidence, tool, path):
                raise WorkflowFailure(f"缺少实际工具成功证据：{tool}/{path}")
        case["artifacts"] = [self.marked_file(workspace / "报告.md", marker),
            self.marked_file(self.home / "artifacts" / "sessions" / session_id / "notes.md", marker)]

    async def permission_case(self, workspace: Path, marker: str, case: dict[str, Any]) -> None:
        session_id = await self.session(workspace, case["name"])
        case["session_id"] = session_id
        policy = self.policies[session_id]
        policy.denied.add("被拒绝.txt")
        policy.writable.add("替代结果.txt")
        rejected = await self.task(session_id,
            "先read_file读取源数据.txt，尝试write_file把其中标记写入被拒绝.txt。若审批被拒绝，"
            "请明确说明拒绝原因，然后结束；不要换路径或绕过审批。只用read_file/write_file。")
        denial = [row for row in rejected["tools"] if row["tool_name"] == "write_file"
                  and row["error_class"] == "permission_denied"]
        text = str((rejected["rpc_run"].get("result") or {}).get("text", ""))
        if not denial or (workspace / "被拒绝.txt").exists() or not any(
            word in text.lower() for word in ("拒绝", "权限", "denied")
        ):
            raise WorkflowFailure("缺少拒绝事件、无副作用或明确拒绝说明")
        allowed = await self.task(session_id,
            "现在发出新的合法请求：将源数据.txt的标记写入替代结果.txt，并读回核验；"
            "本次write_file将单独审批。不要再尝试被拒绝.txt，只用read_file/write_file。")
        grants = [row for row in self.decisions if row.get("run_id") == allowed["rpc_run"]["run_id"]
                  and row["decision"] == "allow_once" and row.get("accepted")]
        if not grants or not self.used(allowed, "write_file", "替代结果.txt"):
            raise WorkflowFailure("替代操作缺少本次独立授权及实际执行证据")
        case["artifacts"] = [self.marked_file(workspace / "替代结果.txt", marker)]
        case["denied_requests"] = [row["request_id"] for row in self.decisions
                                   if row["session_id"] == session_id and row["decision"] == "deny_once"]

    async def session_case(self, workspace: Path, marker: str, case: dict[str, Any]) -> None:
        session_id = await self.session(workspace, case["name"])
        case["session_id"] = session_id
        policy = self.policies[session_id]
        policy.writable.update({"退出后回忆.txt", "重启后回忆.txt", "停止后继续.txt"})
        await self.task(session_id, "用read_file读取源数据.txt，记住其中项目代号并用note_save保存。"
                        "之后会在压缩、退出和重启后提问。只用read_file/note_save。")
        cursor = self.cursors[session_id]
        await self.disconnect()
        await self.connect()
        case["client_resume"] = await self.resume(session_id)
        case["cursor_before_resume"] = cursor
        recalled = await self.task(session_id,
            "仅根据本会话记忆，把刚才的项目代号写入退出后回忆.txt。不要再次读取源数据。")
        if self.used(recalled, "read_file", "源数据.txt"):
            raise WorkflowFailure("退出恢复后的回答重新读取了源文件，不能作为会话记忆证据")
        self.capacity()
        with database(self.home / "state.db") as sql:
            before = int(sql.execute("SELECT count(*) FROM compactions WHERE session_id=?", (session_id,)).fetchone()[0])
        try:
            case["compact_rpc"] = await self.rpc("session.compact", {"session_id": session_id,
                "focus": "保留项目代号的精确文本，以及当前任务和已完成文件"}, 150)
        except Exception:
            if self.limit_reached():
                raise BudgetStop("压缩期间达到有效请求限制；原始历史状态以SQLite为准") from None
            raise
        with database(self.home / "state.db") as sql:
            summaries = [dict(row) for row in sql.execute("SELECT * FROM compactions WHERE session_id=?", (session_id,))]
        if len(summaries) != before + 1 or not summaries[-1]["summary"]:
            raise WorkflowFailure("手动压缩未持久化新的有效摘要")
        json_file(self.directory / "compactions" / f"{session_id}.json", summaries)
        await self.stop()
        await self.start()
        case["daemon_resume"] = await self.resume(session_id)
        recalled = await self.task(session_id,
            "依据本会话记忆，把项目代号写入重启后回忆.txt。不要再次读取源数据。")
        if self.used(recalled, "read_file", "源数据.txt"):
            raise WorkflowFailure("daemon重启后重新读取源文件，不能作为恢复记忆证据")
        start_marker, forbidden = "取消已开始.txt", "取消后禁止出现.txt"
        code = (f"from pathlib import Path; import time; Path({start_marker!r}).write_text({marker!r}, encoding='utf-8'); "
                f"time.sleep(8); Path({forbidden!r}).write_text('should-not-exist', encoding='utf-8')")
        command = "python -c " + shlex.quote(code)
        policy.commands.add(command)
        cancel_id = await self.submit(session_id,
            "现在只调用一次bash，command必须逐字使用下面整行，timeout=30；等待完成，不调用其他工具：\n" + command)
        deadline = time.monotonic() + RUN_TIMEOUT
        while not (workspace / start_marker).exists():
            if time.monotonic() >= deadline:
                raise WorkflowFailure("未观察到容器工具开始标记，无法验证运行中停止")
            status = await self.rpc("run.get", {"run_id": cancel_id})
            if status["status"] in TERMINAL:
                raise WorkflowFailure("取消目标在出现开始标记前结束")
            await asyncio.sleep(0.05)
        observed_at = time.monotonic()
        case["cancel_rpc"] = await self.rpc("run.cancel", {"run_id": cancel_id}, 16)
        cancelled = await self.completed(session_id, cancel_id, expected="cancelled")
        if any(row["committed"] for row in cancelled["messages"]):
            raise WorkflowFailure("取消任务的消息被写入成功上下文")
        await asyncio.sleep(max(0.0, 9 - (time.monotonic() - observed_at)))
        if (workspace / forbidden).exists():
            raise WorkflowFailure("取消后延迟子进程仍产生了文件副作用")
        await self.task(session_id, "继续新的合法任务：依据记忆，把项目代号写入停止后继续.txt，随后结束。")
        case["artifacts"] = [self.marked_file(workspace / name, marker) for name in
                             ("退出后回忆.txt", "重启后回忆.txt", "停止后继续.txt", start_marker)]
        case["cancelled_run_id"] = cancel_id
        case["delayed_side_effect_absent"] = True

    async def verify_owned_containers(self) -> None:
        remaining: list[str] = []
        run_ids = {row["run_id"] for row in self.report.get("submissions", []) if "run_id" in row}
        for run_id in sorted(run_ids):
            result = await asyncio.to_thread(subprocess.run, [
                self.docker_binary, "ps", "--all", "--quiet", "--filter",
                f"label=com.tars-agent.run={run_id}",
            ], capture_output=True, text=True, encoding="utf-8", timeout=5)
            if result.returncode:
                raise WorkflowFailure("所属容器清理检查未确认；不把docker CLI退出当工具已停止")
            remaining.extend(result.stdout.split())
        self.report["remaining_owned_containers"] = remaining
        if remaining:
            raise WorkflowFailure("本次运行仍有所属容器残留")

    async def close_owned_sessions(self) -> None:
        for session_id in self.policies:
            try:
                await self.rpc("session.close", {"session_id": session_id}, 20)
            except Exception as exc:
                self.report.setdefault("cleanup_issues", []).append({
                    "session_id": session_id, "error_type": type(exc).__name__,
                    "detail": "会话关闭未确认，daemon关闭仍会尝试收尾",
                })

    async def execute(self, selection: str) -> int:
        selected = [kind for kind in ("file", "permission", "session") if selection in {"all", kind}]
        self.report["planned_cases"] = [f"{kind}-{number}" for kind in selected for number in range(1, 4)]
        try:
            config, self.config_home, self.ledger = fixed_configuration()
            self.configure_limit(config.llm.request_limit)
            self.report["budget_before"] = counts(self.ledger)
            self.report["source_head"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True,
            ).strip()
            self.report["source_dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=PROJECT, text=True,
            ).strip())
            self.report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            from tars_agent.core.eval.provenance import compute_tree_digest
            self.report["source_tree_sha256"] = compute_tree_digest(
                PROJECT, exclude=(self.directory, self.home),
            )
            del config
            await self.start()
            methods = {"file": self.file_case, "permission": self.permission_case, "session": self.session_case}
            for kind, method in methods.items():
                if selection not in {"all", kind}:
                    continue
                previous_failed = False
                for number in range(1, 4):
                    case: dict[str, Any] = {"name": f"{kind}-{number}", "kind": kind, "round": number,
                                            "status": "not_run", "started_at": now()}
                    self.report["cases"].append(case)
                    if previous_failed:
                        case.update(status="not_run_previous_failure", finished_at=now())
                        continue
                    self.capacity()
                    workspace = self.directory / "中文工作区" / case["name"]
                    workspace.mkdir(parents=True)
                    marker = "TARS-" + secrets.token_hex(16)
                    (workspace / "源数据.txt").write_text(marker + "\n", encoding="utf-8")
                    case.update(workspace=str(workspace), marker=marker, budget_before=counts(self.ledger))
                    try:
                        await method(workspace, marker, case)
                        case["status"] = "passed"
                    except BudgetStop:
                        case["status"] = "not_completed_budget"
                        raise
                    except Exception as exc:
                        case.update(status="failed", error_type=type(exc).__name__,
                                    error=str(exc) if isinstance(exc, WorkflowFailure) else "见本次daemon和任务证据")
                        previous_failed = True
                    finally:
                        session_id = case.get("session_id")
                        case["run_ids"] = [row["run_id"] for row in self.report.get("submissions", [])
                                           if row["session_id"] == session_id and "run_id" in row]
                        case.update(finished_at=now(), budget_after=counts(self.ledger))
                        self.checkpoint()
            cases = self.report["cases"]
            self.report["status"] = "passed" if all(case["status"] == "passed" for case in cases) else "failed"
            self.report["all_nine_complete"] = len(cases) == 9 and self.report["status"] == "passed"
        except BudgetStop as exc:
            self.report.update(status="blocked_budget", error=str(exc))
        except (EnvironmentBlocked, SystemExit) as exc:
            self.report.update(status="environment_blocked",
                               error=str(exc) if isinstance(exc, EnvironmentBlocked) else "配置未就绪")
        except Exception as exc:
            self.report.update(status="failed", error_type=type(exc).__name__, error="见本次私有验收证据")
        finally:
            try:
                if self.retain and self.report["status"] == "passed" and self.process is not None:
                    await self.stop()
                    await self.start(guarded=False)
                    self.report.update(retained=True, retained_port=self.port,
                                       retained_session_ids=list(self.policies))
                    await self.disconnect()
                else:
                    if self.client is not None:
                        await self.close_owned_sessions()
                    await self.stop()
                if self.report.get("submissions"):
                    await self.verify_owned_containers()
            except Exception as exc:
                self.report.setdefault("cleanup_issues", []).append(type(exc).__name__)
                if not self.report.get("retained"):
                    try:
                        await self.stop()
                    except Exception:
                        self.report.setdefault("cleanup_issues", []).append("最终服务收尾仍未确认")
            if self.ledger.is_file():
                try:
                    self.report["budget_after"] = counts(self.ledger)
                except sqlite3.Error:
                    self.report["budget_after_unavailable"] = True
            self.report["finished_at"] = now()
            if self.report.get("cleanup_issues") and self.report["status"] == "passed":
                self.report["status"] = "failed_cleanup"
            if self.report.get("stream_issues") and self.report["status"] == "passed":
                self.report["status"] = "failed_stream"
            self.report["all_nine_complete"] = (len(self.report["cases"]) == 9
                                                  and self.report["status"] == "passed")
            self.checkpoint()
        print(json.dumps({"status": self.report["status"], "manifest": str(self.directory / "manifest.json"),
                          "budget_after": self.report.get("budget_after")}, ensure_ascii=False))
        return 0 if self.report["status"] == "passed" else 3 if self.report["status"] == "blocked_budget" else 1


def daemon_child(
    request_limit: int | None = DEFAULT_SCRIPT_LIMIT, *, state_home: Path, guarded: bool = True,
) -> None:
    config, config_home, _ledger = fixed_configuration()
    home = isolated_state_home(state_home, config_home)
    os.environ["TARS_HOME"] = str(home)
    config.logging.file = str(home / "logs" / "core.log")
    config.trace.file = str(home / "traces" / "daemon.jsonl")
    import tars_agent.core.app as app_module
    import tars_agent.core.llm.provider as provider_module
    app_module.get_config = lambda: config
    if guarded:
        provider_module.RequestLedger = acceptance_ledger_type(request_limit)
    asyncio.run(app_module.CoreApp().run())


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="显式开始真实验收，会消耗固定账本额度")
    parser.add_argument("--retain", action="store_true", help="通过后保留普通daemon/会话供用户TUI接续")
    parser.add_argument("--workflow", choices=["all", "file", "permission", "session"], default="all")
    parser.add_argument("--request-limit", type=parse_request_limit, default=DEFAULT_SCRIPT_LIMIT,
                        metavar="N|unlimited", help="本脚本累计请求限制；与受信生产配置取更严格者")
    parser.add_argument("--state-home", type=Path,
                        help="新的验收状态目录；默认使用本轮证据目录下的state，不复用原会话库")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="验收证据根目录，每轮自动创建独立子目录")
    parser.add_argument("--daemon-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--unguarded", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.daemon_child:
        if args.state_home is None:
            parser.error("--daemon-child requires --state-home")
        daemon_child(args.request_limit, state_home=args.state_home, guarded=not args.unguarded)
        return 0
    if not args.execute:
        print(json.dumps({"mode": "plan_only_no_requests", "rounds_per_workflow": 3,
                          "workflows": ["file", "permission", "session"],
                          "script_request_limit": args.request_limit,
                          "production_default_request_limit": REQUEST_LIMIT,
                          "limit_policy": "与受信生产配置取更严格的非空限制；都为unlimited时不限次",
                          "output_root": str(args.output_root.expanduser().resolve()),
                          "state_home": (str(args.state_home.expanduser().resolve())
                                         if args.state_home else "每轮证据目录/state"),
                          "state_isolation": "只复用原模型配置和请求账本，不接管原daemon或原会话库",
                          "next": "镜像及真实隔离验收就绪后，显式加--execute"}, ensure_ascii=False, indent=2))
        return 0
    try:
        workflows = Workflows(args.retain, args.request_limit,
                              state_home=args.state_home, output_root=args.output_root)
    except EnvironmentBlocked as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return asyncio.run(workflows.execute(args.workflow))


if __name__ == "__main__":
    raise SystemExit(main())
