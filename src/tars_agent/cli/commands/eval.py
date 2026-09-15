from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Literal

from tars_agent.core.config import TarsConfig
from tars_agent.core.eval.report import render_json_report, render_markdown_report
from tars_agent.core.eval.runner import load_artifact_result, run_eval_suite


# 原子写入报告文件，未指定输出路径时由调用方直接打印
def _write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# 执行 Eval 套件并把完整证据 JSON 写入指定文件
def cmd_eval_run(
    config: TarsConfig,
    *,
    suite: Path,
    output: Path,
    model: str | None = None,
) -> None:
    if model:
        config.llm.default_model = model
    try:
        result = asyncio.run(run_eval_suite(suite, output, config))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    summary = result.summary
    print(f"artifact: {output.resolve()}")
    print(f"result: {(output / 'result.json').resolve()}")
    print(
        "attempts: "
        f"{summary.total_attempts} "
        f"passed={summary.passed} failed={summary.failed} "
        f"skipped={summary.skipped} errors={summary.errors}"
    )
    if summary.failed or summary.errors or summary.skipped:
        raise SystemExit(1)


# 校验既有结果 JSON，并按 json 或 Markdown 格式输出报告
def cmd_eval_report(
    result_path: Path,
    *,
    format: Literal["json", "md"],
    output: Path | None = None,
) -> None:
    try:
        result = load_artifact_result(result_path)
        rendered = (
            render_json_report(result) if format == "json" else render_markdown_report(result)
        )
        if output is None:
            print(rendered, end="")
        else:
            _write_report(output, rendered)
            print(f"report: {output.resolve()}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


__all__ = ["cmd_eval_report", "cmd_eval_run"]
