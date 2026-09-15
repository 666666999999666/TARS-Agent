from __future__ import annotations

import asyncio

from tars_agent.core.runtime.supervisor import RunSupervisor


# 功能：验证 Supervisor 注册的 Run 完成后会自动移出活动集合
# 设计：用 Event 控制协程完成时点，分别断言运行中和完成后的可观察状态
async def test_supervisor_removes_completed_run() -> None:
    release = asyncio.Event()

    async def run() -> None:
        await release.wait()

    supervisor = RunSupervisor()
    supervisor.start("run-1", run)
    assert supervisor.is_active("run-1")

    release.set()
    await supervisor.wait("run-1")
    await asyncio.sleep(0)

    assert not supervisor.is_active("run-1")
    assert supervisor.active_run_ids() == ()


# 功能：验证 shutdown 会取消并收集全部活动 Run，且不留下悬挂 Task
# 设计：启动两个永不自行结束的协程，在 finally 记录清理证据并检查活动集合归零
async def test_supervisor_shutdown_cancels_and_gathers_all_runs() -> None:
    cleaned: list[str] = []

    async def run(run_id: str) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(run_id)

    supervisor = RunSupervisor()
    supervisor.start("run-1", lambda: run("run-1"))
    supervisor.start("run-2", lambda: run("run-2"))
    await asyncio.sleep(0)

    await supervisor.shutdown()

    assert set(cleaned) == {"run-1", "run-2"}
    assert supervisor.active_run_ids() == ()
