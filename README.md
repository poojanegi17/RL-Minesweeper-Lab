# RL Minesweeper Lab

**An interactive reinforcement learning laboratory where multiple agents learn, fail, evolve, and get compared on Minesweeper.**

RL Minesweeper Lab pits five independent decision-making approaches — a random baseline, a
logical constraint solver, tabular Q-Learning, a Deep Q-Network, and Proximal Policy
Optimization — against the same game, on the same board, evaluated the same way. It's built
with a custom Gymnasium-compatible Minesweeper environment, a FastAPI backend that serves real
training artifacts, and a React frontend for exploring, replaying, and comparing how each agent
actually thinks.

This project isn't only about training agents to win. It's about **understanding why different
algorithms succeed or fail** on the same problem — and showing that reasoning, not just a
leaderboard number.

**Stack:** React · TypeScript · Tailwind CSS · FastAPI · PyTorch · Gymnasium

🔗 **Live Demo:** [rl-minesweeper-lab.vercel.app](https://rl-minesweeper-lab.vercel.app/) &nbsp;·&nbsp; 📦 **Repository:** [github.com/poojanegi17/RL-Minesweeper-Lab](https://github.com/poojanegi17/RL-Minesweeper-Lab)

```
 Human ──▶ Environment ──▶ Observation ──▶ Agent ──▶ Action ──▶ Reward ──▶ Learning
                                              ▲                              │
                                              └──────────────────────────────┘
```

---

## 📖 Project Overview — Why Minesweeper?

Minesweeper looks like a puzzle, but underneath it's a compact, controllable testbed for the
exact problems that make sequential decision-making under uncertainty hard:

- **Partial observability** — an agent never sees where the mines are, only revealed numbers
  and hidden cells, exactly like a human player.
- **Sequential decisions** — every reveal changes what's known and narrows future choices; it's
  a chain of decisions, not a single classification.
- **Uncertainty** — multiple mine layouts are often consistent with what's currently visible,
  so the agent has to act under real ambiguity, not just noise.
- **Delayed rewards** — a move's consequences (win or loss) can be many steps away from the
  move that mattered.
- **Exploration vs. exploitation** — a learning agent has to risk unknown cells to discover
  what works, while still trying to survive.

Small board, fast episodes, and a well-defined win/loss condition — but every hard problem in
RL shows up in miniature. That combination is why it's the environment here, not because it's
a novel game to solve.

---

## 🖥️ Interactive Demo Features

The frontend isn't a static writeup of results — it's a working lab.

**Play Minesweeper yourself.** The exact same environment every agent trains on is playable in
the browser, so you can feel the difficulty before seeing how any agent handles it.

**Watch AI solve it.** A replay viewer steps through a real recorded episode move by move —
board state, the action taken, and the reward received at each step, for any of Random, CSP,
DQN, or PPO.

**Compare agent minds.** The same board, five different reasoning processes: CSP's logical
constraint deductions, Q-Learning and DQN's Q-value estimates, PPO's action probabilities, and
Random's absence of any of the above — shown side by side, not just described.

**Research journey.** Experiments aren't listed in a table and left there. Each algorithm is
presented as an evolving story: why it was introduced, what was tested, what limitation showed
up, and what that motivated next — see below.

---

## 🧬 Research Journey — Algorithm Evolution

```
Random ──▶ CSP ──▶ Q-Learning ──▶ DQN ──▶ PPO
```

Five algorithms exist because each one answers a question the previous one couldn't.

### Random Agent

**Purpose:** a floor, not a contender. Picks uniformly among hidden cells, with no state, no
memory, and no learning. Every other agent in this project is measured against it.

### CSP Solver

**Why introduced:** before reaching for machine learning at all, test how far pure logical
deduction gets on a board that has real logical structure to exploit.

- Builds a constraint from every revealed number ("exactly N of these hidden neighbors are
  mines")
- Applies deduction rules to find cells that are provably safe or provably mines
- Falls back to the lowest estimated mine probability when nothing is provable

**CSP is the strongest agent in this project (45.5% win rate)** — not because it's more
sophisticated than the learned agents, but because Minesweeper genuinely contains solvable
logical structure that deduction can exploit directly, without needing to learn it from
experience first.

### Q-Learning

**Why introduced:** test whether a value-based method that *learns* from reward, rather than
reasoning from rules, can compete — without the added complexity of a neural network yet.

- Learns a Q-value for every (board state, action) pair from experience
- Epsilon-greedy exploration, with the standard Bellman update

**Limitation:** the table is keyed by the exact board pattern, so nothing generalizes between
similar states. On this project's 5×5 benchmark board, the reachable state space is large
enough that 20,000 training episodes visit almost every state exactly once — the agent ends up
statistically indistinguishable from Random. (On a smaller 4×4/2-mine board, where states
repeat, the same code reaches ~74% — confirming the algorithm itself works; it's the
exact-match representation that doesn't scale.)

### DQN

**Why introduced:** replace the table with a function approximator that can generalize to board
patterns it hasn't exactly seen before.

**Architecture:** an 11-channel one-hot board encoding (hidden mask, revealed mask, one-hot
revealed count 0–8) → a small CNN → one Q-value per cell, trained as **Double DQN** with
experience replay and a target network.

DQN wasn't trained once and reported — a dedicated stability investigation ran four
single-variable experiments plus a combined configuration, all from the same seed:

| Experiment | Problem addressed | Result |
|---|---|---|
| Baseline (Double DQN) | — | 2.0% win rate; loss spikes unpredictably late in training |
| + Best-checkpoint deployment | A good policy found mid-run being silently overwritten later | 3.0% — deploys whichever checkpoint actually scored best |
| + Learning-rate decay | Loss spikes and TD-error instability late in training | **3.5%**, and loss stayed tightly bounded (vs. spiking 300× higher in the baseline) |
| Smaller network | Capacity vs. stability tradeoff | 1.0% — more stable loss, but not enough capacity to learn a competitive policy |
| Combined (LR decay + checkpointing) | Do the two fixes compound? | 2.0% — both help individually, but on this run the combination didn't beat LR decay alone (see repo history for the full analysis) |

**Takeaway:** learning-rate decay was the more fundamental fix (it changes *why* loss diverges);
checkpoint selection is a low-cost safety net that can't make training more stable but also
can't hurt. Even the best DQN configuration remains far behind CSP on this board.

### PPO

**Why introduced:** explore a fundamentally different approach — learning a policy directly,
via actor-critic policy optimization, instead of learning Q-values and deriving a policy from
them.

**Architecture:** the same 11-channel encoding feeds a shared CNN trunk, branching into an
**actor head** (one action logit per hidden cell) and a **critic head** (a scalar state-value
estimate), trained with Generalized Advantage Estimation and a clipped-surrogate objective.

| Experiment | What changed | Win rate |
|---|---|---|
| Baseline | 6,000 episodes, default reward | 1.0% |
| Longer training | 25,000 episodes, same reward | 0.5% — worse, despite 4× more training |
| Reward shaping | Denser reward for larger cascades | **1.5%** — best result, corroborated by entropy and explained-variance trends, not just the win-rate number |
| Reward shaping + checkpointing | Best checkpoint deployed | 1.5% — tied the final weights; no peak-then-degrade shape for checkpointing to rescue here |

**Honest finding:** reward shaping is a real, validated improvement — but a modest one. PPO's
best configuration (1.5%) still sits far below DQN's best (3.5%) and nowhere near CSP (45.5%).
The critic's explained variance tops out under 11%, meaning sparse rewards and hard credit
assignment remain only partially addressed, not solved.

---

## 🔬 Experiment Methodology

Every experiment in this project followed the same loop:

```
Hypothesis ──▶ Implementation change ──▶ Training run ──▶ Evaluation ──▶ Learning
```

One variable changed at a time from a fixed baseline, so results are attributable rather than
"we tried some things and it got better."

**Metrics tracked per training run:**

| DQN | PPO |
|---|---|
| Win rate | Win rate |
| Reward | — |
| Loss | Policy loss, value loss |
| TD-error (mean / max) | Explained variance |
| Q-value statistics | Entropy |
| Gradient norm | — |

---

## 🏁 Results

Same 5×5, 5-mine board, evaluated identically (200 greedy episodes) for every agent:

| Agent | Win Rate | Notes |
|---|---|---|
| Random | 0.5% | Baseline floor |
| **CSP** | **45.5%** | Strongest by far — exploits real logical structure in the game |
| Q-Learning | 0.5% | Doesn't generalize past the exact states it's visited |
| DQN | 1.0% | Best checkpoint, stabilized with learning-rate decay |
| PPO | 0.5% | Shaped reward + best checkpoint |

In dedicated longer-training experiments (25,000 episodes instead of this table's matched
6,000-episode budget), DQN's best configuration reached **3.5%** and PPO's best reached
**1.5%** — both meaningfully better than their matched-budget numbers, and both still far below
CSP.

**No RL agent in this project currently outperforms the CSP solver.** That's reported plainly,
not hedged — the goal here was never only maximum win rate, it was understanding *why* each
approach behaves the way it does on a board with strong exploitable logical structure.

---

## 🏗️ System Architecture

```
        React Frontend
              │
              ▼
       FastAPI Backend
              │
              ▼
     Experiment Results
              │
              ▼
   RL Agents + Replay Data
```

- **Frontend** — interactive visualization dashboard: agent explainer pages, the Research
  Journey pipeline, and the replay viewer. Every page is driven by live API responses; nothing
  is mocked.
- **Backend** — a read-only FastAPI layer that serves experiment metadata, training metrics,
  and replay timelines straight from the artifacts training runs wrote to disk. It never
  trains or writes anything.
- **RL** — the Gymnasium environment, all five agents, and the evaluation pipeline that
  produces every result this app displays.

---

## ⚙️ Technical Implementation

### RL Environment

- Gymnasium-compatible `reset`/`step` interface
- Configurable board size and mine count
- Seeded episodes for reproducible runs
- 11-channel one-hot observation encoding (hidden mask, revealed mask, one-hot revealed count)
- Two reward modes: `default` and `shaped` (denser signal for multi-cell cascade reveals)

### Agent Implementations

| Agent | Approach | Purpose |
|---|---|---|
| Random | Uniform random action | Performance floor |
| CSP | Constraint propagation + probability fallback | Deterministic logical reasoning baseline |
| Q-Learning | Tabular value learning | Learn from reward without a neural network |
| DQN | CNN Q-value approximation, Double DQN, experience replay | Generalize across board patterns |
| PPO | Actor-critic, GAE, clipped-surrogate objective | Learn a policy directly instead of Q-values |

### Experiment Infrastructure

- Checkpointing (best-scoring and final weights, both saved) for DQN and PPO
- Dedicated evaluation scripts, separate from training
- Per-episode metrics collection (reward, loss, win rate, and the agent-specific diagnostics
  above)
- Ablation-style experiment families — every DQN/PPO variant isolates exactly one change from
  a shared baseline, so results are comparable

### Visualization Infrastructure

- Deterministic replay generation for real recorded episodes
- Per-agent decision reasoning surfaced in the replay viewer (Q-values, action probabilities,
  CSP's constraint deductions) — replay files never record mine locations, so nothing about
  the visualization can leak hidden state
- Interactive dashboards for agent comparison and the Research Journey
- Real training curves, not illustrative placeholders

---

## 📊 Project Scale

| Metric | Value |
|---|---|
| Agents implemented | 5 |
| Experiments performed | 14 |
| Total training episodes | ~298,000 |
| Recorded replay episodes | 40 |
| Automated tests | 214 (126 RL + 88 backend) |

---

## 📁 Repository Structure

```
RL Minesweeper Lab/
├── frontend/               # React + TypeScript UI
│   └── src/
│       ├── api/              # Backend API client
│       ├── components/       # agent/, research/, replay/, charts/, about/, ui/
│       ├── pages/             # Home, Agents, AgentDetail, Research, Replay, About
│       └── lib/                # Adapters between API data and the UI
│
├── backend/                # FastAPI read-only serving layer over rl/results_public/
│   └── app/
│       ├── routes/            # agents / experiments / metrics / leaderboard / replays
│       └── services/          # results_loader.py, replay_loader.py
│
└── rl/                     # Python RL environment and agents
    ├── environment/           # Game engine + Gymnasium wrapper
    ├── agents/                 # random, csp, q_learning, dqn, ppo
    ├── models/                  # DQN / PPO networks
    ├── evaluation/              # Training scripts, comparisons, replay generation
    ├── results_public/          # Deployment-safe artifact subset (committed) -- summaries, chart metrics, replays
    ├── results/                 # Full local training output (gitignored) -- checkpoints, CSVs, raw dumps
    └── tests/                   # Pytest suite
```

---

## ▶️ How to Run Locally

**Backend**

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`, backend at `http://localhost:8000` (interactive docs
at `/docs`). CORS and the frontend's `VITE_API_URL` are pre-configured for each other.

**RL environment / experiments**

```bash
cd rl
pip install -r requirements.txt

pytest                                  # run the test suite
python -m evaluation.evaluate_agents    # compare all five agents
```

---

## 🚀 Deployment

**This project is deployed** — frontend on Vercel ([rl-minesweeper-lab.vercel.app](https://rl-minesweeper-lab.vercel.app/)), backend on Render ([rl-minesweeper-lab.onrender.com](https://rl-minesweeper-lab.onrender.com)). The steps below reproduce that setup.

**Frontend (Vercel or any static host)**

```bash
cd frontend
npm run build     # outputs frontend/dist -- a static bundle, tsc -b && vite build
```

Set `VITE_API_URL` as a project environment variable pointing at the deployed backend's base
URL (e.g. `https://your-backend.example.com`) — never commit a production URL into `.env`.
`frontend/.env.example` documents the variable; copy it to `.env` for local development.

**Backend (any ASGI host — Render, Railway, Fly.io, a VM, etc.)**

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

The backend is stateless and read-only, and needs no database. It serves from `rl/results_public/`
by default — a small (~100MB), deployment-safe subset of the full local training output
(`rl/results/`, gitignored: checkpoints, CSVs, raw dumps) containing only what the API actually
reads: experiment summaries, chart-ready metric histories, and replay JSON. It's committed to
the repo, so a normal `git clone` + deploy already has everything the backend needs — no
separate data upload or volume mount required. See backend/README.md's "Which results directory?"
for exactly what's excluded and why.

Two environment variables, both read by `app/config.py` via `pydantic-settings` (prefix `MINESWEEPER_`):

| Variable | Purpose | Example |
|---|---|---|
| `MINESWEEPER_CORS_ORIGINS` | Origins allowed to call the API — defaults to `["http://localhost:5173"]` only | `MINESWEEPER_CORS_ORIGINS=["https://your-frontend.vercel.app"]` (JSON array string) |
| `MINESWEEPER_RESULTS_DIR` | Override the results directory (e.g. to point at your full local `rl/results/` instead) | `MINESWEEPER_RESULTS_DIR=../rl/results` |

**Don't set `MINESWEEPER_RESULTS_DIR` unless you actually need to override it.** An empty/blank
value (e.g. adding the key in your host's dashboard with no value) is *not* the same as leaving
it unset — pydantic-settings treats `""` as a real override, which resolves to the process's
current working directory instead of the built-in `rl/results_public/` default, and every
experiment/replay endpoint silently returns empty. If you ever see `/api/experiments` or
`/api/replays` return `[]` despite `rl/results_public/` being committed, check for exactly this
before anything else.

**Before deploying:** set `MINESWEEPER_CORS_ORIGINS` to the real deployed frontend origin — the
default only allows local dev, so the deployed frontend will get CORS errors until this is set.

---

## 🔭 Future Work

- Larger boards, to test how each approach's limitations scale
- Better exploration strategies for the learned agents
- Self-play or curriculum-style training
- A live, real-time human-vs-agent comparison (distinct from the pre-recorded replay viewer)
- Additional RL algorithms beyond the five implemented here

---

## 🎯 Closing

This project explores not just whether reinforcement learning agents can solve Minesweeper, but
how different learning approaches reason, fail, and improve.
