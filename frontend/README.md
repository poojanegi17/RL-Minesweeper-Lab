# RL Minesweeper Lab — Frontend

React + TypeScript + Vite UI for exploring and comparing the project's RL agents, backed by the FastAPI service in `../backend/`.

```
frontend (React, :5173)
        |  fetch, via src/api/*
        v
backend (FastAPI, :8000)
```

## Architecture

```
src/
├── api/                  # Backend API client -- the only place `fetch` is called
│   ├── client.ts           # Base URL (VITE_API_URL) + ApiError + apiGet<T>
│   ├── agents.ts            # GET /api/agents
│   ├── experiments.ts       # GET /api/experiments[/{id}], GET /api/experiments/{id}/ablation
│   ├── metrics.ts           # GET /api/experiments/{id}/metrics, GET /api/leaderboard
│   └── replays.ts           # GET /api/replays[/{id}]
├── types/                 # TypeScript interfaces mirroring backend/app/schemas/*.py exactly
├── hooks/
│   └── useApiQuery.ts       # loading/error/retry data-fetching hook (no React Query dependency)
├── lib/
│   ├── agentAdapters.ts     # Backend Agent/LeaderboardEntry -> the UI's richer `Agent` shape
│   ├── metricSeriesConfig.ts # Which chart(s) to render per agent (DQN vs. PPO log different metrics)
│   ├── downsample.ts        # Evenly-spaced downsampling for 1000s-of-episode training histories
│   └── cn.ts                 # className helper (pre-existing)
├── data/                  # Shared UI types (`Agent`, `AgentKind`, ...) + the validated agent color
│                             palette (`AGENT_HEX`/`AGENT_STYLES`) -- no mock data; every page is API-driven
├── components/
│   ├── agent/                # AgentCard, AgentStatusBadge, agentIcons
│   ├── experiment/            # ExperimentCard, HyperparameterTable, AblationTable -- shared by
│   │                             AgentDetail and the Experiment Explorer, not duplicated per page
│   ├── replay/                 # ReplayBoard, ReplayControls, ReplayTimeline, ReplayInfo
│   ├── charts/                # PlaceholderChart (bar), MetricLineChart (single series, line),
│   │                             ExperimentMetricsChart (per-agent grid of MetricLineChart)
│   └── ui/                    # Button, Card, Tabs, Select, Badge, Skeleton, EmptyState, ApiErrorState
└── pages/                 # Home, Agents, Compare, AgentDetail, Experiments, ExperimentDetail, Replay, About
```

**Data flow for a connected page:** page component calls `useApiQuery(() => someApiFunction(...), deps)` → `src/api/*` calls `apiGet<T>` → `src/api/client.ts` reads `VITE_API_URL`, fetches, and throws a typed `ApiError` on failure (network or non-2xx) → the hook exposes `{ data, status: "loading" | "success" | "error", error, retry }` → the page renders a `Skeleton`, `ApiErrorState`, `EmptyState`, or the real content based on `status`.

### Why an agent can have "no data" on the detail page

The backend's 5-agent catalog (`GET /api/agents`) is static project metadata; only DQN and PPO have actual training runs under `rl/results/` (Random, CSP, and Q-Learning are never run through an artifact-writing script). `AgentDetail` handles this directly: it always shows the catalog description, and shows empty states (not an error) for Architecture/Hyperparameters/Training/Metrics when an agent has no experiments. See `services/results_loader.py`'s module docstring in the backend for the full explanation of this gap.

### Picking which experiment to show

An agent can have many experiments (see `rl/results/`'s `exp_*`/`ppo_exp_*` directories). Rather than re-deriving "the best one" on the frontend, `AgentDetail` reuses `GET /api/leaderboard`'s `experiment_id` field — the backend already computes the best-win-rate experiment per agent for the leaderboard, so this is the same "best" every other part of the app agrees on.

### Home page

Fully API-driven: hero stats, an agent overview grid, and a "recent experiments" list are all computed from `GET /api/agents` + `GET /api/leaderboard` + `GET /api/experiments`, with one addition — `GET /api/agents` has no `architecture` field (only `ExperimentDetail` does), so `Home.tsx`'s composed fetcher additionally calls `GET /api/experiments/{id}` for whichever agents have a leaderboard `experiment_id` (DQN, PPO today) to show a short architecture snippet per card. Agents without a recorded experiment simply omit that line rather than showing a placeholder. "Recent experiments" reuses the shared `ExperimentCard` component (see below), sorted by timestamp, capped at 6.

### Experiment Explorer (`/experiments`, `/experiments/:experimentId`)

Two more fully API-driven routes, letting any of the 14+ discovered experiments be browsed individually rather than just the one best-win-rate run per agent `AgentDetail` shows:

- **`Experiments.tsx`** (`/experiments`) fetches `GET /api/experiments` once; the Agent/Algorithm filter dropdowns are populated from the distinct values actually present in the response (never a hardcoded option list), so a future third agent shows up automatically.
- **`ExperimentDetail.tsx`** (`/experiments/:experimentId`) composes `GET /api/experiments/{id}` + `.../metrics` + `.../ablation` in one `Promise.all`. A `404` on the detail call is treated as "not found" (renders `EmptyState`, matching `AgentDetail`'s not-found convention) rather than an error — but a `422` (malformed artifact) or network failure still surfaces via `ApiErrorState`, since those genuinely are failures. See `fetchExperimentDetail`'s try/catch for the exact distinction.

**Shared components**, used by both `AgentDetail` and `ExperimentDetail` (extracted during this milestone rather than duplicated): `ExperimentMetricsChart` (the agent-aware grid of training charts), `HyperparameterTable` (a generic key/value renderer used for hyperparameters, training configuration, *and* best-checkpoint metadata — all three are the same `Record<string, SummaryValue>` shape). `ExperimentCard` is shared between `Experiments` and `Home`'s "recent experiments" section. `AblationTable` is new and generic: it renders whatever `ablation.members` the backend returns (see `backend/README.md`'s "Ablation grouping" section for how that's computed without hardcoding any specific experiment family), and shows its own empty state when an experiment has no ablation siblings.

### Replay Viewer (`/replay`)

Watches any of CSP/DQN/PPO/Random play an episode move-by-move. `Replay.tsx` fetches `GET /api/replays` once for the Agent/Episode selectors (populated from real data, same pattern as the Experiment Explorer's filters), then fetches the selected replay's full timeline via `GET /api/replays/{id}` in a second, dependent `useApiQuery`. Playback is local component state (`stepIndex`, `isPlaying`, `speed`) driven by a `setTimeout` loop, not a backend concern.

`components/replay/`:
- **`ReplayBoard`** — the one real, data-driven board renderer in the app. Reuses `BoardIllustration`'s exported `NUMBER_COLORS` palette rather than inventing a second color scheme; `BoardIllustration` itself stays exactly as it was (a fixed decorative board with no props for real data), so nothing about it changed. See its own docstring for exactly how the "mine hit" highlight is derived safely from `action` + `done`/`won` rather than from anything the board array itself reveals — matching the security requirements in the root README's [Replay Visualization](../README.md#replay-visualization) section.
- **`ReplayControls`** — Previous/Play-Pause/Next + 1x/2x/4x speed.
- **`ReplayTimeline`** — "Step N / Total" plus a scrubber.
- **`ReplayInfo`** — result/steps/reward, plus the current step's agent-specific reasoning (`ReplayStep.reasoning`, shaped differently per agent). Falls back to the literal text **"No explanation metadata recorded"** when `reasoning` is `null` (Random always; any agent on a step where nothing was recorded) — never fabricated, and an unrecognized-but-present reasoning shape is shown as raw JSON rather than silently dropped or guessed at.

## Local development

```bash
npm install
npm run dev
```

Needs the backend running at the URL in `.env`'s `VITE_API_URL` (defaults to `http://localhost:8000` — see `../backend/README.md`). CORS is pre-configured on the backend for `http://localhost:5173`.

```bash
npm run build   # tsc -b && vite build
npm run lint     # oxlint
```

### Tests

No test runner (vitest/jest) is configured in this project, so none was added as part of API integration — see the root README's [Frontend Integration](../README.md#frontend-integration) section for how the integration was verified instead (type-checking, a production build, and replaying every page's fetch sequence against the live backend).

## Environment variables

| Variable | Default (`.env`) | Purpose |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Base URL for every `src/api/*` call. Never hardcoded in a component. |
