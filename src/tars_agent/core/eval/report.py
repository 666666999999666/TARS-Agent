from __future__ import annotations

from tars_agent.core.eval.models import EvalRunResult


# 将可能包含表格控制符的文本压缩并转义为 Markdown 单元格
def _cell(value: object, *, limit: int = 180) -> str:
    text = "—" if value is None or value == "" else str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("|", "\\|")


# 以稳定字段顺序渲染完整 JSON 结果
def render_json_report(result: EvalRunResult) -> str:
    return result.model_dump_json(indent=2, exclude_none=False) + "\n"


# 渲染适合代码仓库留档的 Markdown 汇总与逐次结果
def render_markdown_report(result: EvalRunResult) -> str:
    if result.adapter == "appworld":
        return _appworld_report(result)
    summary = result.summary
    pass_rate = "—" if summary.pass_rate is None else f"{summary.pass_rate:.1%}"
    dirty = (
        "unknown"
        if result.provenance.git_dirty is None
        else ("yes" if result.provenance.git_dirty else "no")
    )
    lines = [
        f"# Eval Report: {result.suite_name}",
        "",
        f"- Run ID: `{result.run_id}`",
        f"- Suite ID: `{result.suite_id}`",
        f"- Adapter: `{result.adapter}`",
        f"- Started: `{result.started_at}`",
        f"- Finished: `{result.finished_at}`",
        f"- Git SHA: `{result.provenance.git_sha or 'unknown'}`",
        f"- Dirty worktree: `{dirty}`",
        f"- Tree digest: `{result.provenance.tree_digest}`",
        f"- Lock hash: `{result.provenance.lock_hash or 'missing'}`",
        f"- Config hash: `{result.provenance.config_hash}`",
        f"- Model: `{result.provenance.model or 'not recorded'}`",
        f"- Model config ref: `{result.model_config_ref.reference}`",
        f"- Docker image: `{result.provenance.docker_image or 'not configured'}`",
        f"- Docker image digest: `{result.provenance.docker_image_digest or 'unknown'}`",
        "",
        "## Summary",
        "",
        (
            "| Attempts | Passed | Failed | Skipped | Errors | Pass rate | Input tokens | "
            "Output tokens | Estimated cost |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary.total_attempts} | {summary.passed} | {summary.failed} | "
            f"{summary.skipped} | {summary.errors} | {pass_rate} | "
            f"{_cell(summary.total_input_tokens)} | {_cell(summary.total_output_tokens)} | "
            f"{_cell(summary.estimated_cost)} {_cell(summary.currency)} |"
        ),
        "",
        "## Attempts",
        "",
        (
            "| Task | Repeat | Status | Run terminal | Goal | Steps | Score | Latency ms | "
            "Tools | Tool errors/refusals | Sandbox backends | Fallback | Timeout | "
            "Cleanup | Error |"
        ),
        (
            "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | "
            "--- | --- | --- | --- |"
        ),
    ]
    for attempt in result.attempts:
        lines.append(
            f"| {_cell(attempt.source_task_id or attempt.task_id)} | {attempt.repetition} | "
            f"{attempt.status.value} | {_cell(attempt.run_terminal_status)} | "
            f"{_cell(attempt.goal_completed)} | {_cell(attempt.step_count)} | "
            f"{_cell(attempt.score)} | {attempt.latency_ms} | "
            f"{_cell(', '.join(attempt.tools.tool_names))} | "
            f"{_cell(attempt.tools.errors)}/{_cell(attempt.tools.refusals)} | "
            f"{_cell(', '.join(attempt.sandbox.observed_backends))} | "
            f"{_cell(attempt.sandbox.fallback_used)} | "
            f"{_cell(attempt.sandbox.timed_out)} | "
            f"{_cell(attempt.sandbox.cleanup_completed)} | "
            f"{_cell(attempt.error)} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            (
                "Unknown token, tool, sandbox, or cleanup fields remain `null`; "
                "this report does not infer them."
            ),
            "A recorded estimated cost is valid only against the embedded dated pricing snapshot.",
            "",
        ]
    )
    return "\n".join(lines)


def _appworld_report(result: EvalRunResult) -> str:
    benchmark = result.benchmark
    metrics = benchmark.get("official_metrics")
    values = metrics if isinstance(metrics, dict) else {}
    lines = [
        f"# AppWorld: {result.suite_name}", "",
        f"- TARS source digest: `{result.provenance.tree_digest}`",
        f"- Model: `{result.model_config_ref.model}`",
        f"- AppWorld source: `{benchmark.get('source_ref')}`",
        f"- Dataset: `{benchmark.get('dataset')}`",
        f"- Interface: {benchmark.get('interface')}",
        f"- Complete: {benchmark.get('complete')}",
        f"- Coverage: {benchmark.get('completed_tasks')}/{benchmark.get('expected_tasks')}", "",
        "- 官方源列表的场景变体完整："
        f"{_cell(benchmark.get('official_source_scenarios_complete'))}",
        "- 本次选集的场景变体完整："
        f"{_cell(benchmark.get('selected_scenario_variants_complete'))}", "",
        "| Official metric | Percent |", "| --- | ---: |",
        f"| Task Goal Completion | {_cell(values.get('task_goal_completion'))} |",
        f"| Scenario Goal Completion | {_cell(values.get('scenario_goal_completion'))} |", "",
        "These are official state-based scores. TARS runtime status is recorded separately.",
        f"Runtime attempts: {result.summary.total_attempts}; "
        f"normally completed: {result.summary.passed}; failures: {result.summary.failed}; "
        f"infrastructure errors: {result.summary.errors}.",
        "Per-task test reports are not displayed or used for prompt tuning.",
        f"Errors: {_cell(benchmark.get('errors'))}", "",
    ]
    note = benchmark.get("scenario_goal_completion_note")
    if note:
        lines.extend([str(note), ""])
    return "\n".join(lines)


__all__ = ["render_json_report", "render_markdown_report"]
