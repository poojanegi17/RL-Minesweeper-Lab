# RL Minesweeper Lab — Backend API

A read-only FastAPI layer that serves the RL pipeline's existing experiment artifacts to the frontend as clean REST endpoints. No database, no new state — every response is derived live from JSON files an RL training/evaluation script already wrote.

```
frontend (React, :5173)
        |
        | HTTP (CORS-allowed for localhost:5173)
        v
FastAPI backend (:8000)
        |
        | reads only -- never writes
        v
rl/results_public/*.json, rl/results_public/*/*.json   (default -- see "Which results directory?" below)
```

## Architecture

```
backend/
├── app/
│   ├── main.py              # FastAPI app, CORS, router wiring, /health
│   ├── config.py            # results_dir (default rl/results_public/), CORS origins, board-size constants
│   ├── schemas/              # Pydantic response models (no filesystem logic)
│   │   ├── agent.py
│   │   ├── experiment.py
│   │   ├── metrics.py
│   │   └── replay.py          # Every field declared explicitly -- see "Replay artifacts" below
│   ├── routes/                # Thin HTTP handlers -- no file reading here
│   │   ├── agents.py          # GET /api/agents (static catalog)
│   │   ├── experiments.py     # GET /api/experiments, /{id}, /{id}/ablation
│   │   ├── metrics.py         # GET /api/experiments/{id}/metrics, /api/leaderboard
│   │   └── replays.py         # GET /api/replays, /{id}
│   └── services/
│       ├── results_loader.py  # Reads rl/results/*.json (experiments)
│       └── replay_loader.py   # Reads rl/results/replays/*.json (episode replays)
├── tests/
├── requirements.txt
└── README.md
```

**Every filesystem read goes through `services/results_loader.py`.** Routes never open a file themselves — they call `ResultsLoader` (injected via FastAPI's `Depends`, which is also how tests substitute a temp directory instead of the real `rl/results/`) and translate its `ExperimentRecord` objects into Pydantic response models.

## How the backend reads RL artifacts

Two shapes of artifact exist on disk, and the loader handles both:

- **Ablation-style experiment directories** — `exp_A_baseline/`, `ppo_exp_C_shaped/`, etc. Each holds one `{agent}_history_{episodes}.json` + `.csv` + `_summary.json` triplet, plus a `checkpoints_{episodes}/` directory of `.pt` files. Checkpoint *filenames* (not contents — the `.pt` files themselves are never opened) are reported in `ExperimentDetail.artifacts`.
- **Loose top-level files** — `dqn_history_25000.json` (+ a summary), or `dqn_evaluate_agents_history.json` (no summary at all — `evaluate_agents.py` never writes one).

Discovery (`ResultsLoader._discover`) globs one level deep for `*_summary.json` files and pairs each with its matching history file by filename stem; loose history files with no summary are still surfaced as experiments, just with `has_summary: false` and no evaluation metrics. **CSV files are never read** — they're a byte-for-byte export of the same in-memory history `training/history_export.py` also writes to JSON, so reading JSON is strictly equivalent and better-typed.

### Which results directory? (`rl/results/` vs. `rl/results_public/`)

`rl/results/` is the full local training output — every raw dump, CSV, and model checkpoint (`.pt`/`.pth`) an experiment script ever wrote. It's `.gitignore`d and large (100+ MB), so it never ships to a deployment.

`rl/results_public/` is the deployment-safe subset actually served by default: experiment summaries and the per-episode metric histories the frontend's charts read, plus `replays/`. No checkpoints, no CSVs (the loader never reads either — see above). Same on-disk shape as `rl/results/` (ablation-style experiment directories + loose top-level files + `replays/`), so `ResultsLoader`/`ReplayLoader` don't need to know or care which one they're pointed at — this is purely a matter of *which files got copied in*, not a code difference.

`app/config.py`'s `Settings.results_dir` defaults to `rl/results_public/`, resolved from `config.py`'s own file location so it works right after cloning regardless of the launch directory. Override it with `MINESWEEPER_RESULTS_DIR` — e.g. to point a local backend at your full `rl/results/` instead:

```bash
MINESWEEPER_RESULTS_DIR=../rl/results python -m uvicorn app.main:app --reload
```

or an absolute path in a deployment (e.g. `MINESWEEPER_RESULTS_DIR=/app/rl/results_public`). One consequence of the smaller artifact set worth knowing: `ExperimentDetail.artifacts` (`history_csv`, `checkpoint_dir`, `best_checkpoint_file`, `final_checkpoint_file`) correctly reports `false`/`null` for files that were intentionally excluded — the same graceful state these fields already report for any other experiment missing that particular artifact, and not currently rendered by any frontend page. `best_checkpoint` (the small metadata object — episode, win rate, timestamp — used in the Research pipeline's "Best configuration" card) is unaffected either way, since it's read from the summary JSON itself, never from the `.pt` file.

**Resilience.** `GET /api/experiments` skips (and logs) any individual experiment whose summary JSON fails to parse, rather than failing the whole list — one bad file shouldn't take down the listing. `GET /api/experiments/{id}` and `.../metrics`, which are about one specific artifact the caller asked for, instead return a clear `422` naming the problem. An entirely missing or empty `rl/results/` directory is not an error anywhere — it's treated as zero experiments.

**Schema drift.** `dqn_experiment.py` gained `network_size`, `checkpoint_every`, `used_checkpoint`, and `best_checkpoint_metadata` after its earliest runs — so some summary files on disk (`dqn_history_5000_summary.json`) simply don't have those keys. `split_summary_fields()` buckets whatever *is* present into `hyperparameters` / `training_configuration` / `evaluation_metrics` by an allow-list, with anything unrecognized falling through to `hyperparameters` — so an older or a future summary shape both work without a code change.

### Known data gaps (read before trusting a field)

- **Seed is `null` unless the summary literally contains a `"seed"` key.** No current experiment script writes one (`--seed` is an overridable CLI flag on both `dqn_experiment.py` and `ppo_experiment.py`, so the default can't be safely assumed after the fact) — this is a real gap in the RL scripts, not a backend bug.
- **`board` (`"5x5"`) and `mines` (`5`) are a hardcoded constant** (`app/config.py`), not read per-experiment — every current experiment script hardcodes `ROWS=COLS=5, NUM_MINES=5` with no CLI override, so this is a safe structural fact today, but it will silently mislabel any future experiment run on a different board size until the RL scripts are changed to record it themselves.
- **`timestamp` is filesystem mtime**, not an authored field — it reflects when the file was last written, which is a reasonable proxy for "when the run finished" but isn't literally recorded by the training scripts.
- **`status` is always `"completed"`** — these scripts only ever write their output file after a full run finishes, so there's no artifact-based way to represent "running" or "failed."
- **`GET /api/leaderboard` includes Random, CSP, and Q-Learning using the project README's last-recorded figures**, not a live read — those three agents write no experiment artifacts at all (Q-Learning trains in-memory in `evaluate_agents.py` without ever calling `history_export`; CSP/Random aren't "trained" at all). Every leaderboard row is tagged `"source": "experiment_artifact"` or `"source": "static_reference"` so a consumer can tell which is which — never presented as if both were freshly measured.

### Derived metadata: mechanical, not authored

`title`, `description`, and `techniques` fields don't exist as literal keys in any artifact — there's no human-written experiment name or narrative summary anywhere under `rl/results/`. Rather than leave them out or fabricate prose, they're computed deterministically from fields that *are* present:

- `title` / `description` — composed from `agent`, `episodes`, `board`, `mines` (`derive_title`/`derive_description` in `results_loader.py`), e.g. `"DQN - 25,000 episodes"`. A run that's a member of a multi-run family instead gets a variant-derived title (`derive_run_title`/`humanize_variant`) built from its own id's variant suffix — e.g. `"DQN - LR Decay"` for `exp_C_lr_decay` — so siblings that all trained for the same episode count don't render an identical, indistinguishable title.
- `techniques` — a rule table (`_TECHNIQUE_RULES`) mapping real config flags to short labels: `lr_schedule` present → `"LR decay"`, `used_checkpoint` starting with `"best"` → `"Best-checkpoint deployment"`, `reward_mode == "shaped"` → `"Reward shaping"`, etc. Every label traces back to a value actually in the summary JSON. A family's `techniques` is the union (order-preserving, deduplicated) across all its runs.

Two fields are genuinely new information, not present before this milestone: `metrics_available` (existence-based, same convention as `has_summary`) and `artifacts` (an `ArtifactManifest` reporting which files were found — history JSON/CSV, summary, and checkpoint `.pt` filenames, resolved via the `checkpoints_{episodes}` naming convention `dqn_experiment.py`/`ppo_experiment.py` use; `evaluate_agents.py`'s loose output uses unrelated hardcoded checkpoint directory names with no derivable link back to the history file, so those correctly resolve to `null` rather than being guessed).

### Experiment grouping (`GET /api/experiments`, `GET /api/experiments/{id}`)

Raw artifact discovery produces one entry per file (or per ablation-style directory), which used to mean the frontend saw duplicate-looking rows like `exp_A_baseline`...`exp_E_combined` as five separate, nearly-identical "experiments" instead of one comparison. `results_loader.group_experiments` fixes this by reusing the same `<prefix>_<LETTER>_<variant>` id pattern the ablation endpoint already parses (`parse_ablation_id`): any 2+ runs sharing a group prefix are bucketed into one `ExperimentGroup`; everything else (including a lone id that happens to match the pattern but has no siblings) is presented standalone, under its own id.

`GET /api/experiments` returns one `ExperimentSummary` row per group — `run_count`, `episodes_range` (`[min, max]` across the group's runs), a `metrics_summary` (best/avg win rate across runs that actually recorded one), and a `runs: RunBrief[]` list of every member. A standalone run is presented the same shape with `run_count: 1`.

`GET /api/experiments/{id}` tries `{id}` as a family id first (returning the same grouped `ExperimentSummary`, so a family's full `runs` list is one request away); if no family matches, it falls back to the pre-grouping behavior — treating `{id}` as an individual run id and returning full `ExperimentDetail` — so every run id that worked before this feature still resolves exactly the same way. The one addition is `ExperimentDetail.family_id`: which family (if any) this run belongs to, so the frontend can link from a run's detail page back to its family.

Artifacts with no recognizable `dqn`/`ppo` naming (agent `"Unknown"`) still get listed rather than dropped — grouping never hides an artifact it can't classify, it just can't offer techniques/algorithm info for it beyond `"Unknown"`.

### Ablation grouping (`GET /api/experiments/{id}/ablation`)

Groups experiments by parsing ids against a generic `<prefix>_<LETTER>_<variant>` pattern (`parse_ablation_id`) — `exp_A_baseline` → group `exp`, `ppo_exp_C_shaped` → group `ppo_exp`. This is not specific to any one experiment family: the same logic clusters the DQN ablation (`exp_A_baseline` … `exp_E_combined`) and the PPO ablation (`ppo_exp_A_baseline` … `ppo_exp_D_shaped_checkpoint`) without either being hardcoded. An id that doesn't match the pattern, or has no siblings, returns `{"group": null, "members": []}` — a normal outcome, not an error.

### Replay artifacts (`rl/results/replays/`)

A separate, simpler service (`services/replay_loader.py`) from `ResultsLoader`, since replay discovery has none of the summary/history-pairing complexity: one flat directory of self-contained `{agent}_episode_{n}.json` files, written by `rl/evaluation/generate_replays.py` (never by this backend). `ReplayLoader` mirrors `ResultsLoader`'s resilience contract exactly — a malformed replay is skipped from `GET /api/replays`'s listing (logged, not fatal) but raises a `422` if fetched directly by id; a missing/empty `replays/` directory yields `[]`, not an error.

**Security: `schemas/replay.py`'s models declare every field explicitly (no `extra="allow"`).** This is deliberate, not incidental — FastAPI's `response_model` only serializes declared fields, so even a hypothetical malformed or hand-edited replay file containing an undeclared key (e.g. mine positions, which `rl/evaluation/replay.py` never writes to begin with) would be silently dropped on the way out, never reaching the frontend. See the root README's [Replay Visualization](../README.md#replay-visualization) section for the full picture, including why mine positions are never written to a replay file at all rather than being kept in a "debug" section and trusted not to leak.

## Running locally

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

(`python -m uvicorn` rather than the bare `uvicorn` command, since pip-installed console scripts aren't always on `PATH` depending on your Python setup.)

The API is then at `http://localhost:8000`, interactive docs at `http://localhost:8000/docs`, serving from `rl/results_public/` by default. CORS is pre-configured for `http://localhost:5173` (the frontend's Vite dev server); override with `MINESWEEPER_CORS_ORIGINS` or point at a different results directory with `MINESWEEPER_RESULTS_DIR` (both read by `app/config.py` via `pydantic-settings`, prefix `MINESWEEPER_`) — see "Which results directory?" above.

### Tests

```bash
cd backend
pytest
```

Every test builds its own small fixture results directory (`tests/conftest.py`), including a `replays/` subfolder, rather than depending on the real, ever-changing `rl/results/` — including deliberately malformed summary and replay files, to exercise the error paths.

## API endpoints

All endpoints are read-only (`GET`).

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check. Doesn't touch the filesystem. |
| `GET /api/agents` | Static catalog of all 5 agents (name, type, description, whether it has experiment artifacts). |
| `GET /api/experiments` | Every experiment *family* (2+ related runs) or standalone run discoverable under `rl/results/`, grouped (see below). Never errors — empty/missing directory yields `[]`. |
| `GET /api/experiments/{id}` | `{id}` as a family id returns the grouped summary (with its full `runs` list); otherwise `{id}` is treated as an individual run id and full detail is returned instead: algorithm, architecture, description, hyperparameters, training configuration, evaluation metrics, best-checkpoint metadata, artifact manifest. `404` if unknown, `422` if the artifact is malformed. |
| `GET /api/experiments/{id}/metrics` | Full per-episode training history, row-oriented (chart-ready for Recharts) plus a `series` field listing which metric keys this agent type tracks (DQN and PPO differ). |
| `GET /api/experiments/{id}/ablation` | Sibling experiments sharing `{id}`'s ablation family (see below), for comparison. `404`/`422` match `/{id}`'s behavior; an id with no family returns `{"group": null, "members": []}`. |
| `GET /api/leaderboard` | All 5 agents ranked by best known win rate, each tagged with where its numbers came from (see "Known data gaps"). |
| `GET /api/replays` | Every replay discoverable under `rl/results/replays/` (id, agent, experiment_id, won, step count). Never errors — empty/missing directory yields `[]`. |
| `GET /api/replays/{id}` | Full step-by-step timeline: board state, action, reward, done, and agent-specific reasoning per step. `404` if unknown, `422` if malformed or missing required fields. |

### Example: `GET /api/replays/dqn_episode_1`

```json
{
  "id": "dqn_episode_1",
  "agent": "DQN",
  "experiment_id": null,
  "board_size": "5x5",
  "mines": 5,
  "seed": 42,
  "episode_number": 1,
  "generated_at": "2026-08-05T09:20:41+00:00",
  "initial_board": [[-1,-1,-1,-1,-1], "...4 more rows..."],
  "timeline": [
    {
      "step": 1,
      "board_state": [[-1,-1,-1,-1,-1], "...4 more rows, one cell revealed..."],
      "action": { "row": 3, "col": 2 },
      "reward": 1.0,
      "done": false,
      "reasoning": { "q_value": 5.69 }
    }
  ],
  "won": false,
  "total_reward": -4.0,
  "steps": 7
}
```

### Example: `GET /api/experiments/exp` (a family)

```json
{
  "id": "exp",
  "title": "DQN - 5 runs",
  "agent": "DQN",
  "algorithm": "Double DQN",
  "description": "5 DQN runs on a 5x5 board (5 mines), comparing: Baseline, Checkpoint, LR Decay, Small Net, Combined.",
  "techniques": ["Double DQN", "Best-checkpoint deployment", "LR decay", "Reduced network capacity"],
  "board": "5x5",
  "mines": 5,
  "episodes_range": [25000, 25000],
  "run_count": 5,
  "metrics_summary": { "best_run_id": "exp_C_lr_decay", "best_win_rate": 0.035, "avg_win_rate": 0.023, "runs_with_metrics": 5 },
  "runs": [
    { "id": "exp_A_baseline", "title": "DQN - Baseline", "variant": "baseline", "episodes": 25000, "timestamp": "2026-08-04T16:52:05+00:00", "win_rate": 0.02, "avg_reward": -6.72, "metrics_available": true },
    { "id": "exp_C_lr_decay", "title": "DQN - LR Decay", "variant": "lr_decay", "episodes": 25000, "timestamp": "2026-08-04T16:52:30+00:00", "win_rate": 0.035, "avg_reward": -5.91, "metrics_available": true }
  ]
}
```

### Example: `GET /api/experiments/exp_E_combined` (an individual run)

```json
{
  "id": "exp_E_combined",
  "agent": "DQN",
  "episodes": 25000,
  "board": "5x5",
  "mines": 5,
  "seed": null,
  "timestamp": "2026-08-05T04:28:13+00:00",
  "status": "completed",
  "has_summary": true,
  "algorithm": "Double DQN",
  "title": "DQN - Combined",
  "metrics_available": true,
  "techniques": ["Double DQN", "LR decay", "Best-checkpoint deployment"],
  "family_id": "exp",
  "architecture": "CNN Q-network over an 11-channel board encoding, with experience replay and a target network.",
  "description": "Double DQN trained for 25,000 episodes on a 5x5 board (5 mines).",
  "hyperparameters": { "lr": 0.0001, "lr_schedule": [[0, 0.0001], [10000, 0.00005], [20000, 0.00001]], "batch_size": 64, "target_update_every": 25, "network_size": "default" },
  "training_configuration": { "episodes": 25000, "checkpoint_every": 2500, "used_checkpoint": "best_model.pt", "train_seconds": 572.3 },
  "evaluation_metrics": { "win_rate": 0.02, "avg_reward": -6.0, "avg_episode_length": 4.6, "failures": 196, "eval_episodes": 200 },
  "best_checkpoint": { "episode": 22500, "win_rate": 0.06, "average_reward": -4.96, "timestamp": "2026-08-05T04:27:21+00:00" },
  "artifacts": {
    "history_json": true, "history_csv": true, "summary_json": true,
    "checkpoint_dir": "checkpoints_25000", "best_checkpoint_file": "best_model.pt", "final_checkpoint_file": "final_model.pt"
  }
}
```

## Constraints honored

Per the brief: no database, no modification to `frontend/` or any RL algorithm code, no existing RL files moved — everything above is read live from the configured results directory (`rl/results_public/` by default; see "Which results directory?") on every request, nothing is cached, cloned, or re-saved by the backend.
