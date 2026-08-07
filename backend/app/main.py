"""FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload
from within backend/ (see backend/README.md).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import agents, board_configs, experiments, metrics, races, replays

app = FastAPI(
    title="RL Minesweeper Lab API",
    description="Serves RL experiment artifacts (rl/results/) to the frontend as a read-only REST API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(agents.router)
app.include_router(board_configs.router)
app.include_router(experiments.router)
app.include_router(metrics.router)
app.include_router(replays.router)
app.include_router(races.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    """Basic liveness check -- does not touch the filesystem."""
    return {"status": "ok"}
