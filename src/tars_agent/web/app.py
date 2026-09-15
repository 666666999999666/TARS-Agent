from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from tars_agent.web.auth import LocalWebAuth
from tars_agent.web.core import CoreReader

COOKIE_NAME = "tars_web_session"
KEEPALIVE_S = 15.0
STATIC_DIR = Path(__file__).with_name("static")


class BootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=1, max_length=256)


def create_app(
    core: CoreReader,
    *,
    auth: LocalWebAuth,
    allowed_host: str = "127.0.0.1",
) -> FastAPI:
    local_auth = auth

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(
        title="TARS-Agent Web",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.auth = local_auth
    app.state.core = core

    @app.middleware("http")
    async def local_security(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        host = (request.url.hostname or "").lower()
        allowed_hosts = {allowed_host.lower()}
        if allowed_host == "127.0.0.1":
            allowed_hosts.add("localhost")
        if host not in allowed_hosts:
            return JSONResponse({"detail": "invalid Host"}, status_code=400)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            expected = f"http://{request.headers.get('host')}"
            if origin != expected:
                return JSONResponse({"detail": "invalid Origin"}, status_code=403)
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        )
        return response

    def require_auth(
        session_token: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None,
    ) -> None:
        if not local_auth.authenticated(session_token):
            raise HTTPException(status_code=401, detail="bootstrap required")

    @app.post("/api/bootstrap", status_code=204)
    async def bootstrap(payload: BootstrapRequest, response: Response) -> None:
        session_token = local_auth.exchange(payload.token)
        if session_token is None:
            raise HTTPException(status_code=401, detail="invalid or expired bootstrap")
        response.set_cookie(
            COOKIE_NAME,
            session_token,
            httponly=True,
            secure=False,
            samesite="strict",
            path="/",
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        try:
            pong = await asyncio.wait_for(core.ping(), timeout=2.0)
        except Exception:
            return {"status": "degraded", "core": "disconnected"}
        return {
            "status": "ok",
            "core": "connected",
            "protocol_version": pong.get("protocol_version"),
        }

    @app.get("/api/sessions", dependencies=[Depends(require_auth)])
    async def sessions(
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        return {"sessions": await core.list_sessions(limit=limit, offset=offset)}

    @app.get("/api/sessions/{session_id}", dependencies=[Depends(require_auth)])
    async def session_detail(
        session_id: str,
    ) -> dict[str, Any]:
        return await core.get_session(session_id)

    @app.get("/api/runs/{run_id}", dependencies=[Depends(require_auth)])
    async def run_detail(
        run_id: str,
    ) -> dict[str, Any]:
        return await core.get_run(run_id)

    @app.get("/api/events", dependencies=[Depends(require_auth)])
    async def events(
        request: Request,
        session_id: str,
        after_cursor: Annotated[int | None, Query(ge=0)] = None,
    ) -> StreamingResponse:
        header_cursor = request.headers.get("last-event-id")
        cursor = after_cursor or 0
        if header_cursor:
            try:
                cursor = max(cursor, int(header_cursor))
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid Last-Event-ID") from None

        async def stream() -> AsyncIterator[str]:
            iterator = core.stream_events(session_id, after_cursor=cursor).__aiter__()

            async def next_envelope() -> dict[str, Any]:
                return await iterator.__anext__()

            next_item: asyncio.Task[dict[str, Any]] = asyncio.create_task(next_envelope())
            try:
                while True:
                    done, _ = await asyncio.wait({next_item}, timeout=KEEPALIVE_S)
                    if not done:
                        # Keep the pending anext task alive. Cancelling it would close
                        # an async generator and silently destroy the Core subscription.
                        yield ": keepalive\n\n"
                        continue
                    try:
                        envelope = next_item.result()
                    except StopAsyncIteration:
                        return
                    next_item = asyncio.create_task(next_envelope())
                    payload = json.dumps(
                        envelope,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    event_id = envelope.get("cursor") or envelope.get("last_cursor")
                    event_name = str(envelope.get("kind", "event"))
                    id_line = f"id: {event_id}\n" if isinstance(event_id, int) else ""
                    yield f"{id_line}event: {event_name}\ndata: {payload}\n\n"
            finally:
                if not next_item.done():
                    next_item.cancel()
                await asyncio.gather(next_item, return_exceptions=True)
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    await close()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/")
    @app.get("/sessions/{session_id}")
    @app.get("/runs/{run_id}")
    async def web_shell() -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=503, detail="Web assets have not been built")
        return FileResponse(index)

    assets = STATIC_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    return app


__all__ = ["COOKIE_NAME", "KEEPALIVE_S", "create_app"]
