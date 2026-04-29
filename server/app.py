"""FastAPI application entry point.

Creates the app, configures middleware, and registers all route modules.
"""

from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv

load_dotenv()

from .logging_config import configure_logging

configure_logging()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .jobs import get_job_queue
from .model_routes import router as model_router
from .routes.chat import router as chat_router
from .routes.files import router as files_router
from .routes.gee import router as gee_router
from .routes.health import router as health_router
from .routes.jobs import router as jobs_router
from .routes.tiles import router as tiles_router


@asynccontextmanager
async def lifespan(app: FastAPI):
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
