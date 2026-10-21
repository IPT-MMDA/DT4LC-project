"""FastAPI application entry point.

Creates the app, configures middleware, and registers all route modules.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv

load_dotenv()

from .logging_config import configure_logging  # noqa: E402  # must run after load_dotenv

configure_logging()

# Imports below must follow load_dotenv() and configure_logging() so submodules see the
# populated environment and root logger config when initialised at import time.
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

from .jobs import get_job_queue  # noqa: E402
from .model_routes import router as model_router  # noqa: E402
from .routes.chat import router as chat_router  # noqa: E402
from .routes.files import router as files_router  # noqa: E402
from .routes.gee import router as gee_router  # noqa: E402
from .routes.health import router as health_router  # noqa: E402
from .routes.jobs import router as jobs_router  # noqa: E402
from .routes.tiles import router as tiles_router  # noqa: E402
from .schemas import ErrorCode, ErrorDetail, ErrorResponse  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    queue = get_job_queue()
    await queue.start()
    try:
        yield
    finally:
        await queue.stop()


app = FastAPI(
    title="DT4LC API",
    version="1.0.0",
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
        {"name": "files", "description": "GeoTIFF upload, listing, and download"},
        {"name": "tiles", "description": "XYZ map tiles from GeoTIFF files"},
        {"name": "gee", "description": "Google Earth Engine data fetch and layer export"},
        {"name": "models", "description": "ML model download and cache management"},
    ],
)

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
}


@app.exception_handler(StarletteHTTPException)  # type: ignore[misc]
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Return HTTP errors as ErrorResponse: {"ok": false, "error": {"code", "message", "details"}}."""
    default = ErrorCode.INTERNAL_ERROR if exc.status_code >= 500 else ErrorCode.BAD_REQUEST
    code = _ERROR_CODES.get(exc.status_code, default)
    body = ErrorResponse(error=ErrorDetail(code=code, message=str(exc.detail)))
    return JSONResponse(body.model_dump(mode="json"), status_code=exc.status_code, headers=exc.headers)


app.include_router(health_router)
app.include_router(chat_router)
app.include_router(jobs_router)
app.include_router(files_router)
app.include_router(tiles_router)
app.include_router(gee_router)
app.include_router(model_router)
