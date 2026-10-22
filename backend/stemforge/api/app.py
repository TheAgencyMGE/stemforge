"""FastAPI application factory: middleware, error handling and routers."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .. import __version__
from ..config import get_settings
from ..logging_setup import configure_logging, request_id_var
from ..security import RateLimiter

log = logging.getLogger("stemforge.api")


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(status_code=status, detail={"code": code, "message": message})


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.monotonic()
        try:
            settings = get_settings()
            path = request.url.path
            if path.startswith("/api/") and not path.startswith("/api/files/") and path not in ("/api/healthz", "/api/readyz"):
                if settings.api_token:
                    auth = request.headers.get("authorization", "")
                    if auth != f"Bearer {settings.api_token}":
                        return _error(401, "unauthorized", "missing or invalid API token", rid)
                if request.method in ("POST", "PUT", "PATCH", "DELETE"):
                    cost = 5.0 if request.method == "POST" and path.rstrip("/") in ("/api/projects", "/api/projects/import") else 1.0
                    if not self.limiter.allow(_client_ip(request), cost):
                        return _error(429, "rate_limited", "too many requests; slow down and retry shortly", rid)
            try:
                response = await call_next(request)
            except Exception:  # noqa: BLE001 - keep the request id on unexpected errors
                log.exception("unhandled error", extra={"path": path})
                return _error(500, "internal_error", "unexpected server error; see logs with this request id", rid)
            response.headers["x-request-id"] = rid
            response.headers["x-content-type-options"] = "nosniff"
            response.headers["referrer-policy"] = "no-referrer"
            log.info("request", extra={"method": request.method, "path": path, "status": response.status_code,
                                       "ms": round((time.monotonic() - t0) * 1000, 1)})
            return response
        finally:
            request_id_var.reset(token)


def _error(status: int, code: str, message: str, rid: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
                                                               "request_id": rid or request_id_var.get()}})


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    if settings.env == "production" and not settings.signing_secret and settings.storage_backend == "local":
        raise RuntimeError("STEMFORGE_SIGNING_SECRET must be set in production")
    from ..migrate import upgrade

    upgrade()

    app = FastAPI(title="Stemforge API", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(RequestContextMiddleware, limiter=RateLimiter(settings.rate_limit_per_minute))
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"],
        expose_headers=["content-disposition", "x-request-id"],
    )

    @app.exception_handler(HTTPException)
    async def _http_exc(_req: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            return _error(exc.status_code, exc.detail["code"], exc.detail["message"])
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation(_req: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        return _error(422, "invalid_request", f"{loc}: {first.get('msg', 'invalid input')}".strip(": "))

    @app.exception_handler(Exception)
    async def _unhandled(_req: Request, exc: Exception):
        log.exception("unhandled error")
        return _error(500, "internal_error", "unexpected server error; see logs with this request id")

    from . import routes

    app.include_router(routes.router, prefix="/api")
    return app
