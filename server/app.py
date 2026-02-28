"""FastAPI application entry point.

Creates the app, configures middleware, and registers all route modules.
"""

import logging
import os

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model_routes import router as model_router
from .routes.chat import router as chat_router
from .routes.files import router as files_router
from .routes.gee import router as gee_router
from .routes.health import router as health_router
from .routes.jobs import router as jobs_router
from .routes.tiles import router as tiles_router

app = FastAPI(title="DT4LC API", version="1.0.0")

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
