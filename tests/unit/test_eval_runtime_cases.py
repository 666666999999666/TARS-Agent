from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from tars_agent.core.tools.builtin import BashTool, ReadFileTool, WriteFileTool
from tars_agent.core.tools.runtime import FakeRuntime, RuntimeRouter, ToolCallContext


# 功能：验证 workspace sandbox 正常写入后可从同一授权根读取原内容
# 设计：用 FakeRuntime 执行真实工具协议，断言后端、文件内容和读回结果形成零模型闭环
async def test_runtime_case_normal_read_write(tmp_path: Path) -> None:
    fake = FakeRuntime()
    runtime = RuntimeRouter(fake, allow_host_fallback=False)
    context = ToolCallContext(
        run_id="eval-runtime",
        session_id="eval-session",
        tool_use_id="write-then-read",
        workspace_root=tmp_path,
    )

    written = await WriteFileTool(runtime).invoke(
        {"path": "nested/evidence.txt", "content": "runtime evidence"},
        context=context,
    )
    read = await ReadFileTool(runtime).invoke(
        {"path": "nested/evidence.txt"},
        context=context,
    )

    assert not written.is_error
    assert written.backend == "workspace_sandbox"
    assert not read.is_error
    assert read.content == "runtime evidence"
    assert len(fake.requests) == 2


# 功能：验证 bash 超时后延迟子进程不能继续写 marker，且 run cleanup 被调用
# 设计：启动会在超时后写文件的 Python 子进程，等待越过写入时点并检查 marker 与 FakeRuntime 清理记录
async def test_runtime_case_bash_timeout_cleans_child_process(tmp_path: Path) -> None:
    marker = tmp_path / "late-marker.txt"
    script = (
        "import pathlib,time; "
        "time.sleep(1.5); "
        f"pathlib.Path({str(marker)!r}).write_text('late', encoding='utf-8')"
    )
    command = subprocess.list2cmdline([sys.executable, "-c", script])
    fake = FakeRuntime()
    runtime = RuntimeRouter(fake, allow_host_fallback=False)
    context = ToolCallContext(
        run_id="eval-timeout",
        session_id="eval-session",
        tool_use_id="bash-timeout",
        workspace_root=tmp_path,
    )

    result = await BashTool(runtime).invoke(
        {"command": command, "timeout": 1},
        context=context,
    )
    await runtime.cleanup_run("eval-timeout")
    await asyncio.sleep(0.8)

    assert result.is_error
    assert result.error_type == "timeout"
    assert not marker.exists()
    assert fake.cleaned_runs == ["eval-timeout"]
