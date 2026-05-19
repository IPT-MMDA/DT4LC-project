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
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .jobs import get_job_queue  # noqa: E402
from .model_routes import router as model_router  # noqa: E402
from .routes.chat import router as chat_router  # noqa: E402
from .routes.files import router as files_router  # noqa: E402
from .routes.gee import router as gee_router  # noqa: E402
from .routes.health import router as health_router  # noqa: E402
from .routes.jobs import router as jobs_router  # noqa: E402
from .routes.tiles import router as tiles_router  # noqa: E402


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
    description="API for Digital Twin for Land Cover (DT4LC)",
    version="1.0.0",
    lifespan=lifespan,
)

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
