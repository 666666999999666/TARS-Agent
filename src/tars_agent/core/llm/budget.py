from __future__ import annotations

import asyncio

import httpx

from tars_agent.core.persistence.request_budget import (
    ModelRequestBudgetExceeded,
    RequestKind,
    RequestLedger,
)


class BudgetTransport(httpx.AsyncBaseTransport):
    """Count at the network transport boundary, including application retries."""

    def __init__(
        self, inner: httpx.AsyncBaseTransport, ledger: RequestLedger, *,
        kind: RequestKind = "real",
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._kind = kind

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._kind == "probe" and request.url.host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Fault probes must target loopback; remote calls use the real budget")
        await asyncio.to_thread(self._ledger.reserve, self._kind)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


__all__ = ["BudgetTransport", "ModelRequestBudgetExceeded", "RequestKind", "RequestLedger"]
