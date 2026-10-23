"""FastAPI application entry point.

Creates the app, configures middleware, and registers all route modules.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
import os

from dotenv import load_dotenv

from dta import __version__

load_dotenv()

from .logging_config import configure_logging  # noqa: E402  # must run after load_dotenv

configure_logging()

# Imports below must follow load_dotenv() and configure_logging() so submodules see the
# populated environment and root logger config when initialised at import time.
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from .jobs import JobQueue, get_job_queue  # noqa: E402
from .model_routes import router as model_router  # noqa: E402
from .routes.auth import router as auth_router  # noqa: E402
from .routes.chat import router as chat_router  # noqa: E402
from .routes.files import router as files_router  # noqa: E402
from .routes.gee import router as gee_router  # noqa: E402
from .routes.health import router as health_router  # noqa: E402
from .routes.jobs import router as jobs_router  # noqa: E402
from .routes.session import router as session_router  # noqa: E402
from .routes.settings import router as settings_router  # noqa: E402
from .routes.tiles import router as tiles_router  # noqa: E402
from .schemas import ErrorCode, ErrorDetail, ErrorResponse  # noqa: E402
from .sessions import (  # noqa: E402
    GuestQuotaError,
    SessionMiddleware,
    analysis_counter,
    guest_session_hours,
    log_startup_warnings,
)
from .uploads import cleanup_guest_uploads  # noqa: E402
from .utils import UPLOAD_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# Seconds between runs of the periodic cleanup (expired guest jobs, outputs and uploads).
CLEANUP_INTERVAL_S = 3600.0


async def run_cleanup(queue: JobQueue) -> None:
    """Remove expired guest jobs with their outputs, expired guest uploads, and idle quota counters."""
    hours = guest_session_hours()
    await queue.cleanup_old_jobs(guest_hours=hours)
    await asyncio.to_thread(cleanup_guest_uploads, UPLOAD_DIR, hours)
    analysis_counter.prune()


async def _cleanup_periodically(queue: JobQueue) -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_S)
        try:
            await run_cleanup(queue)
        except Exception:
            logger.exception("Periodic cleanup failed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log_startup_warnings()
    queue = get_job_queue()
    await queue.start()
    cleanup = asyncio.create_task(_cleanup_periodically(queue))
    try:
        yield
    finally:
        cleanup.cancel()
        await asyncio.gather(cleanup, return_exceptions=True)
        await queue.stop()


app = FastAPI(
    title="DT4LC API",
    version=__version__,
    description=(
        "REST API for the Digital Twin for Land Cover (DT4LC): async geospatial jobs, "
        "LLM-powered chat and planning, file uploads, map tiles, Google Earth Engine data, "
        "and ML model management."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Health, capabilities, registry, and metrics"},
        {"name": "chat", "description": "COE planning, execution, and streaming chat"},
        {"name": "jobs", "description": "Async analysis job queue"},
        {"name": "session", "description": "Session cookie, role and guest quota"},
        {"name": "auth", "description": "GitHub sign-in and sign-out"},
        {"name": "files", "description": "GeoTIFF upload, listing, and download"},
        {"name": "tiles", "description": "XYZ map tiles from GeoTIFF files"},
        {"name": "gee", "description": "Google Earth Engine data fetch and layer export"},
        {"name": "models", "description": "ML model download and cache management"},
        {"name": "settings", "description": "Operator settings (404 for everyone else)"},
    ],
)

# Added before CORS, so CORS stays outermost and answers preflights without a session.
app.add_middleware(SessionMiddleware)

cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ERROR_CODES = {
    400: ErrorCode.BAD_REQUEST,
    403: ErrorCode.UNAUTHORIZED,
    404: ErrorCode.NOT_FOUND,
    413: ErrorCode.FILE_TOO_LARGE,
    422: ErrorCode.VALIDATION_ERROR,
}


@app.exception_handler(StarletteHTTPException)  # type: ignore[misc]
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Return HTTP errors as ErrorResponse: {"ok": false, "error": {"code", "message", "details"}}."""
    default = ErrorCode.INTERNAL_ERROR if exc.status_code >= 500 else ErrorCode.BAD_REQUEST
    code = _ERROR_CODES.get(exc.status_code, default)
    body = ErrorResponse(error=ErrorDetail(code=code, message=str(exc.detail)))
    return JSONResponse(body.model_dump(mode="json"), status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(GuestQuotaError)  # type: ignore[misc]
async def guest_quota_handler(request: Request, exc: GuestQuotaError) -> JSONResponse:
    """429 ``GUEST_QUOTA`` with ``details: {limit, used, resets_at}`` and ``Retry-After``."""
    body = ErrorResponse(error=ErrorDetail(code="GUEST_QUOTA", message=str(exc), details=exc.details))
    return JSONResponse(body.model_dump(mode="json"), status_code=429, headers={"Retry-After": str(exc.retry_after_s)})


app.include_router(health_router)
app.include_router(session_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(jobs_router)
app.include_router(files_router)
app.include_router(tiles_router)
app.include_router(gee_router)
app.include_router(model_router)
app.include_router(settings_router)
