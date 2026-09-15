from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tars_agent.cli.commands.chat import cmd_chat
from tars_agent.cli.commands.core import cmd_core_start, cmd_core_status, cmd_core_stop
from tars_agent.cli.commands.eval import cmd_eval_report, cmd_eval_run
from tars_agent.cli.commands.ping import cmd_ping
from tars_agent.cli.commands.run import cmd_run
from tars_agent.cli.commands.runs import cmd_run_cancel, cmd_run_metrics, cmd_run_status
from tars_agent.cli.commands.sandbox import cmd_sandbox_build, cmd_sandbox_doctor
from tars_agent.cli.commands.sessions import cmd_sessions_list
from tars_agent.cli.commands.trace import cmd_trace
from tars_agent.cli.commands.version import cmd_version
from tars_agent.core.config import get_config
from tars_agent.core.logging_setup import setup_logging


# CLI 主入口：解析命令行参数并分发到对应子命令
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tars", description="TARS-Agent CLI: chat, one-shot tasks and Core management",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ping", help="Ping the core daemon")
    chat_parser = subparsers.add_parser("chat", help="Start a continuous CLI conversation")
    chat_parser.add_argument("--resume", metavar="SESSION_ID", help="Resume an existing session")
    run_parser = subparsers.add_parser("run", help="Run a goal, or inspect/cancel an existing Run")
    run_parser.add_argument("--goal", help="Execute one goal and exit this client")
    run_sub = run_parser.add_subparsers(dest="run_command")
    run_status = run_sub.add_parser("status", help="Show durable Run status")
    run_status.add_argument("run_id")
    run_cancel = run_sub.add_parser("cancel", help="Cancel an active Run")
    run_cancel.add_argument("run_id")
    run_metrics = run_sub.add_parser("metrics", help="Show read-only persisted Run metrics")
    run_metrics.add_argument("run_id")

    sessions_parser = subparsers.add_parser("sessions", help="List durable sessions")
    sessions_sub = sessions_parser.add_subparsers(dest="sessions_command")
    sessions_list = sessions_sub.add_parser("list", help="List recent sessions")
    sessions_list.add_argument("--status", choices=["ready", "running", "closed"])

    core_parser = subparsers.add_parser("core", help="Manage the core daemon")
    core_sub = core_parser.add_subparsers(dest="core_command")
    core_sub.add_parser("start", help="Start the daemon in the background")
    core_sub.add_parser("stop", help="Stop the running daemon")
    core_sub.add_parser("status", help="Show daemon status")

    sandbox_parser = subparsers.add_parser("sandbox", help="Manage the tool sandbox")
    sandbox_sub = sandbox_parser.add_subparsers(dest="sandbox_command")
    sandbox_sub.add_parser("build", help="Build the pinned local sandbox image")
    sandbox_sub.add_parser("doctor", help="Check Docker daemon and sandbox image")

    trace_parser = subparsers.add_parser("trace", help="View system trace log")
    trace_parser.add_argument("run_id", nargs="?", default=None, help="Filter by run ID")
    trace_parser.add_argument("--layer", choices=["ipc", "event", "llm"], help="Filter by layer")
    trace_parser.add_argument("--direction", help="Filter by direction (e.g. CORE→LLM)")
    trace_parser.add_argument("--raw", action="store_true", help="Output raw NDJSON")
    trace_parser.add_argument("--follow", "-f", action="store_true", help="Follow new records")

    eval_parser = subparsers.add_parser("eval", help="Run or report an explicit evaluation suite")
    eval_sub = eval_parser.add_subparsers(dest="eval_command")
    eval_run = eval_sub.add_parser("run")
    eval_run.add_argument("--suite", required=True, type=Path)
    eval_run.add_argument("--output", required=True, type=Path)
    eval_run.add_argument("--model")
    eval_report = eval_sub.add_parser("report")
    eval_report.add_argument("result", type=Path)
    eval_report.add_argument("--format", choices=["json", "md"], default="md")
    eval_report.add_argument("--output", type=Path)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    args = parser.parse_args()

    if args.command == "run":
        if args.goal is not None and (not args.goal.strip() or args.run_command is not None):
            run_parser.error(
                "--goal must be nonempty and cannot be combined with a management action"
            )
        if args.goal is None and args.run_command is None:
            run_parser.error("provide --goal or a management action")

    if args.version:
        cmd_version()
        return

    if args.command == "eval" and args.eval_command == "report":
        cmd_eval_report(args.result, format=args.format, output=args.output)
        return

    config = get_config()
    setup_logging(config)

    if args.command == "ping":
        cmd_ping(config)
    elif args.command == "chat":
        cmd_chat(config, resume_session_id=args.resume)
    elif args.command == "run":
        if args.goal is not None:
            cmd_run(args.goal, config)
        elif args.run_command == "status":
            cmd_run_status(config, args.run_id)
        elif args.run_command == "metrics":
            cmd_run_metrics(config, args.run_id)
        elif args.run_command == "cancel":
            cmd_run_cancel(config, args.run_id)
        else:
            run_parser.print_help()
            sys.exit(1)
    elif args.command == "sessions":
        if args.sessions_command == "list":
            cmd_sessions_list(config, status=args.status)
        else:
            sessions_parser.print_help()
            sys.exit(1)
    elif args.command == "core":
        if args.core_command == "start":
            cmd_core_start(config)
        elif args.core_command == "stop":
            cmd_core_stop(config)
        elif args.core_command == "status":
            cmd_core_status(config)
        else:
            core_parser.print_help()
            sys.exit(1)
    elif args.command == "trace":
        cmd_trace(
            args.run_id,
            config,
            layer=args.layer,
            direction=args.direction,
            raw=args.raw,
            follow=args.follow,
        )
    elif args.command == "eval":
        if args.eval_command == "run":
            cmd_eval_run(config, suite=args.suite, output=args.output, model=args.model)
        elif args.eval_command == "report":
            cmd_eval_report(args.result, format=args.format, output=args.output)
        else:
            eval_parser.print_help()
            sys.exit(1)
    elif args.command == "sandbox":
        if args.sandbox_command == "build":
            cmd_sandbox_build(config)
        elif args.sandbox_command == "doctor":
            cmd_sandbox_doctor(config)
        else:
            sandbox_parser.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
