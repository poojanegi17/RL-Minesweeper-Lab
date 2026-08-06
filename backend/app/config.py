"""Backend configuration.

All paths and cross-origin settings live here so nothing else in the app
hardcodes a filesystem layout or an allowed frontend origin.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> repo root -> rl/results_public
#
# `rl/results/` (the full local training output -- checkpoints, CSVs, every
# raw dump) is gitignored and never deployed. `rl/results_public/` is the
# deployment-safe subset that actually ships: experiment summaries, the
# per-episode metric histories the frontend's charts read, and replay JSON
# files -- no model checkpoints (.pt/.pth), no CSVs (results_loader.py never
# reads them; see its module docstring), nothing else. Same on-disk shape as
# `rl/results/` (ablation-style experiment directories + loose top-level
# files + a `replays/` folder), so `ResultsLoader`/`ReplayLoader` don't need
# to know or care which one they're pointed at.
#
# Override with `MINESWEEPER_RESULTS_DIR` (e.g. `MINESWEEPER_RESULTS_DIR=../rl/results_public`
# when running the backend from a different working directory, or an absolute
# path in a deployment) -- this default only has to work for "just cloned the
# repo and ran the backend from backend/", which it does since it's resolved
# from this file's own location, not the process's current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "rl" / "results_public"

# Every current experiment script (`dqn_experiment.py`, `ppo_experiment.py`,
# `evaluate_agents.py`) hardcodes these three values with no CLI override, so
# unlike per-run fields (seed, timestamps), this is a safe structural
# constant rather than a guess about any specific run -- see backend/README.md
# ("Known data gaps") for why this isn't treated the same as seed, which
# *is* overridable per-run and therefore never assumed.
BENCHMARK_BOARD_ROWS = 5
BENCHMARK_BOARD_COLS = 5
BENCHMARK_BOARD_MINES = 5


class Settings(BaseSettings):
    """Runtime configuration, overridable via environment variables (prefix `MINESWEEPER_`)."""

    model_config = SettingsConfigDict(env_prefix="MINESWEEPER_")

    results_dir: Path = _DEFAULT_RESULTS_DIR
    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
