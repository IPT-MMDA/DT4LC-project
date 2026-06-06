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
from fastapi.exceptions import RequestValidationError  # noqa: E402
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


app = FastAPI(title="DT4LC API", version="1.0.0", lifespan=lifespan)

cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(jobs_router)
app.include_router(files_router)
app.include_router(tiles_router)
app.include_router(gee_router)
app.include_router(model_router)


# ---------------------------------------------------------------------------
# Global exception handlers – return the standardised error envelope
# ---------------------------------------------------------------------------

_STATUS_CODE_TO_ERROR_CODE: dict[int, ErrorCode] = {
    400: ErrorCode.BAD_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    409: ErrorCode.CONFLICT,
    422: ErrorCode.VALIDATION_ERROR,
}


def _build_error_response(status_code: int, message: str, details: dict | None = None) -> ErrorResponse:
    code = _STATUS_CODE_TO_ERROR_CODE.get(status_code, ErrorCode.INTERNAL_ERROR)
    return ErrorResponse(error=ErrorDetail(code=code, message=message, details=details))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    response = _build_error_response(
        422,
        "Request validation failed",
        details={"errors": exc.errors()},
    )
    return JSONResponse(status_code=422, content=response.model_dump())


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    response = _build_error_response(
        exc.status_code,
        exc.detail if isinstance(exc.detail, str) else "An error occurred",
    )
    return JSONResponse(status_code=exc.status_code, content=response.model_dump())
