from __future__ import annotations

from tars_agent.core.config import SandboxConfig
from tars_agent.core.tools.runtime.docker import DockerRuntime
from tars_agent.core.tools.runtime.router import RuntimeRouter


# 按 sandbox.mode 构造唯一的 Docker/Host 路由策略。
def build_runtime_router(config: SandboxConfig) -> RuntimeRouter:
    return RuntimeRouter(
        DockerRuntime(config),
        allow_host_fallback=config.mode == "preferred",
    )


# required 模式在返回 Router 前完成沙箱可用性检查并失败关闭。
async def initialize_runtime_router(config: SandboxConfig) -> RuntimeRouter:
    runtime = build_runtime_router(config)
    try:
        if config.mode == "required":
            status = await runtime.preflight()
            if not status.available:
                reason = status.reason or "sandbox_unavailable"
                raise RuntimeError(f"sandbox.mode=required preflight failed: {reason}")
        return runtime
    except BaseException:
        await runtime.cleanup()
        raise


__all__ = ["build_runtime_router", "initialize_runtime_router"]
