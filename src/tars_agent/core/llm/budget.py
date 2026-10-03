from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from dataclasses import dataclass

import httpx

from tars_agent.core.persistence.cost_budget import CostBudgetError, CostLedger
from tars_agent.core.persistence.request_budget import (
    ModelRequestBudgetExceeded,
    RequestKind,
    RequestLedger,
)


@dataclass
class CostAttempt:
    run_id: str
    step: int
    attempt: int
    max_tokens: int
    reservation: str | None = None


current_cost_attempt: ContextVar[CostAttempt | None] = ContextVar("cost_attempt", default=None)


class BudgetTransport(httpx.AsyncBaseTransport):
    """Count at the network transport boundary, including application retries."""

    def __init__(
        self, inner: httpx.AsyncBaseTransport, ledger: RequestLedger, *,
        kind: RequestKind = "real", cost_ledger: CostLedger | None = None,
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._kind = kind
        self._cost_ledger = cost_ledger

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._kind == "probe" and request.url.host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Fault probes must target loopback; remote calls use the real budget")
        if self._cost_ledger is None:
            await asyncio.to_thread(self._ledger.reserve, self._kind)
        else:
            scope = current_cost_attempt.get()
            if (scope is None or scope.reservation is not None or self._kind != "real"
                    or request.url.scheme != "https" or request.url.host != "api.deepseek.com"
                    or request.url.path != "/anthropic/v1/messages"):
                raise CostBudgetError("Cost-guarded calls require the frozen DeepSeek endpoint")
            try:
                body = json.loads(request.content)
                valid = (body["model"] == "deepseek-flash"
                         and body["max_tokens"] == scope.max_tokens)
            except (ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                raise CostBudgetError("Request differs from the frozen cost policy")
            await self._reserve_cost(scope)
        return await self._inner.handle_async_request(request)

    async def _reserve_cost(self, scope: CostAttempt) -> None:
        assert self._cost_ledger is not None
        operation = asyncio.create_task(asyncio.to_thread(
            self._cost_ledger.reserve_attempt, self._ledger, max_tokens=scope.max_tokens,
            run_id=scope.run_id, step=scope.step, attempt=scope.attempt,
        ))
        cancelled = False
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                cancelled = True
        scope.reservation = operation.result()
        # Cancellation must not leave a ledger-writing thread behind the worker's
        # final accounting snapshot, and must never proceed to the HTTP send.
        if cancelled:
            raise asyncio.CancelledError()

    async def aclose(self) -> None:
        await self._inner.aclose()


__all__ = ["BudgetTransport", "ModelRequestBudgetExceeded", "RequestKind", "RequestLedger"]
