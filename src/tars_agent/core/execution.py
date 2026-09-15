from __future__ import annotations

import asyncio
import logging

from tars_agent.core.context import ExecutionContext
from tars_agent.core.loop import AgentLoop
from tars_agent.core.tools.runtime import RuntimeRouter

log = logging.getLogger(__name__)


def mark_cancelled(context: ExecutionContext) -> None:
    context.status = "cancelled"
    context.reason = "cancelled"


async def execute_loop(loop: AgentLoop, context: ExecutionContext) -> None:
    """Run the shared model/tool loop; its owner commits the resulting state."""
    try:
        await loop.run(context)
    except asyncio.CancelledError:
        mark_cancelled(context)
    except Exception:
        log.exception("agent execution failed run_id=%s step=%d", context.run_id, context.step)
        context.mark_failed("runtime_error")


async def cleanup_run(runtime: RuntimeRouter, run_id: str) -> bool:
    """Wait through repeated cancellation, preserving any resource cleanup failure."""
    operation = asyncio.create_task(runtime.cleanup_run(run_id))
    cancelled = False
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            cancelled = True
    # In particular, RuntimeCleanupPending must reach the owner even if another
    # cancellation arrived while cleanup was running.
    operation.result()
    return cancelled
