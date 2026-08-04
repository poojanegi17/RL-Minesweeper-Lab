# RL Minesweeper Lab

A portfolio project exploring reinforcement learning and classical AI approaches to solving Minesweeper — from a custom game engine and Gymnasium environment through logical solvers and (eventually) learned agents, with a React frontend for exploring and comparing results.

## Objective

- Build a custom Minesweeper environment suitable for both classical and learning-based agents
- Compare multiple solving approaches (logical deduction, tabular RL, deep RL) on equal footing
- Visualize agent performance and learning progress
- Eventually ship an interactive web demo where agents can be watched playing live

## Current Status

**Implemented**

- React + TypeScript frontend MVP
- Responsive UI
- Agent catalog
- Agent comparison page
- Agent detail pages
- Static Minesweeper illustration
- Custom Minesweeper game engine
- Gymnasium-compatible RL environment
- Configurable board size and mine count
- Random baseline agent
- CSP (Constraint Satisfaction Problem) solver
- Tabular Q-Learning agent
- Deep Q-Network (DQN) agent with CNN Q-network, experience replay, target network, and a multi-channel state encoding
- Evaluation framework
- Unit tests

**Planned**

- PPO/A2C
- Backend API
- Interactive replay visualization
- Live agent demonstrations

## Project Structure

```
RL Minesweeper Lab/
├── frontend/                  # React + TypeScript UI
│   └── src/
│       ├── app/                # App-level providers (theme, etc.)
│       ├── components/
│       │   ├── agent/          # Agent cards, status badges
│       │   ├── board/          # Minesweeper board illustration
│       │   ├── charts/         # Chart placeholders (Recharts)
│       │   ├── layout/         # Header, footer, page layout
│       │   └── ui/             # Reusable UI primitives (Button, Card, Tabs, ...)
│       ├── data/                # Agent metadata and types
│       └── pages/               # Home, Agents, Compare, AgentDetail, About
│
└── rl/                        # Python RL environment and agents
    ├── environment/
    │   ├── minesweeper.py       # Core game engine
    │   ├── minesweeper_env.py   # Gymnasium environment wrapper
    │   └── utils.py             # Coordinate/array helpers
    ├── agents/
    │   ├── random_agent.py      # Random baseline
    │   ├── csp_solver.py        # CSP logical solver
    │   ├── q_learning_agent.py  # Tabular Q-Learning agent
    │   └── dqn_agent.py         # Double DQN agent + checkpointing + LR scheduling
    ├── models/
    │   └── dqn_network.py       # CNN Q-network (configurable size) + state encoding
    ├── training/
    │   ├── replay_buffer.py     # Experience replay buffer
    │   └── history_export.py    # Training history -> JSON/CSV export
    ├── evaluation/
    │   ├── evaluate_agents.py   # Agent comparison script (uses DQN's best_model.pt)
    │   ├── dqn_experiment.py    # Configurable single DQN training run
    │   ├── compare_experiments.py  # Compare DQN runs by training budget
    │   ├── compare_ablation.py  # Compare DQN stabilization experiments (A/B/C/D)
    │   └── metrics.py           # Episode running and metric aggregation
    └── tests/                   # Pytest suite
```

## Reinforcement Learning Architecture

```
Minesweeper Engine
        ↓
Gymnasium Environment
        ↓
Random Agent
        ↓
CSP Solver
        ↓
Tabular Q-Learning Agent
        ↓
DQN Agent (CNN + Experience Replay + Target Network)
        ↓
Evaluation Framework
        ↓
Future RL Agents (PPO/A2C)
```

- **Minesweeper Engine** (`environment/minesweeper.py`) — Board generation, mine placement, cell reveal/flag logic, and win/loss detection, independent of any RL framework.
- **Gymnasium Environment** (`environment/minesweeper_env.py`) — Wraps the engine in the standard Gymnasium `reset`/`step` API with configurable board size and mine count, so any agent that speaks Gymnasium can play.
- **Random Agent** (`agents/random_agent.py`) — Picks a uniformly random hidden cell each step. Establishes the performance floor.
- **CSP Solver** (`agents/csp_solver.py`) — Builds constraints from revealed numbers ("exactly N of these hidden neighbors are mines"), applies logical deduction to find guaranteed-safe cells and guaranteed mines, and falls back to lowest-probability guessing when no deduction is possible.
- **Tabular Q-Learning Agent** (`agents/q_learning_agent.py`) — Learns a Q-value for each (board pattern, action) pair from experience via epsilon-greedy exploration and the Bellman update, with no built-in Minesweeper logic. Effective on small boards, but a flattened board is used directly as the state key, so it doesn't generalize across the huge state space of larger boards.
- **DQN Agent** (`agents/dqn_agent.py`, `models/dqn_network.py`, `training/replay_buffer.py`) — Replaces the Q-table with a small CNN (configurable capacity) that maps an 11-channel board encoding to one Q-value per cell, trained as Double DQN with experience replay, a target network, best-checkpoint selection, and an optional learning-rate decay schedule. Generalizes across similar board patterns instead of memorizing exact ones — see [Experiments](#experiments) below for what changed and why.
- **Evaluation Framework** (`evaluation/`) — Runs agents over many episodes and reports win rate, average episode length, and failure counts on identical board configurations, so agents can be compared fairly.
- **Future RL Agents** — PPO/A2C will plug into the same environment and evaluation framework once implemented.

## Benchmark Results

| Agent | Type | Win Rate |
|-------|------|----------|
| Random Agent | Baseline | ~0.5% |
| CSP Solver | Logical Solver | ~45.5% |
| Q-Learning Agent | Tabular RL | ~0.5%* |
| DQN Agent | Deep RL (Double DQN, CNN) | ~1.0%* |

*Measured on a 5x5 board with 5 mines over 200 episodes (`rl/evaluation/evaluate_agents.py`), with Q-Learning trained for 20,000 episodes and DQN trained for 6,000 episodes first, evaluated using its best checkpoint (see below). These are current benchmark results and will change as training budgets and algorithms improve.*

\* *Both learned agents are still far behind CSP on this board — see [Experiments](#experiments) for why, and for evidence (on a smaller board, and with a longer DQN training budget) that both are learning correctly rather than being broken.*

## Experiments

This project's two RL milestones so far were both attempts to fix the same underlying problem — a learned agent needs to generalize across board states, not memorize them — approached from two different angles.

**Tabular Q-Learning's limitation: state explosion.** `QLearningAgent` keys a plain dictionary by the exact flattened board (`(-1, -1, 1, 0, ...)`). Every distinct board pattern gets its own independent value, with zero sharing between patterns that only differ by, say, one cell. A 5x5 board has an astronomically large space of reachable visible patterns, so 20,000 training episodes visit only ~29,000 of them — almost all exactly once — leaving the agent statistically indistinguishable from Random (~0.5% win rate) on that board. On a 4x4 board with 2 mines, where the reachable state space is small enough to actually revisit, the same code reaches ~74% win rate after 20,000 episodes, confirming the algorithm itself is correct; it's the *exact-match* state representation that doesn't scale.

**DQN's improvement: function approximation, then a better input for it to approximate over.** Swapping the Q-table for a CNN (`DQNNetwork`) was step one — a network can in principle produce a reasonable estimate for a board pattern it has never exactly seen, by recognizing it's similar to ones it has. But the first version fed the network a single channel of the raw board divided by 8, which only improved 5x5 win rate from 0.5% to 0.5% — no measurable change. The problem: `-1` (hidden) and `0` (revealed, zero neighboring mines) — the two states a Minesweeper player cares most about telling apart — land right next to each other on that numeric scale (`-0.125` vs `0.0`), so the network still had to *infer* a categorical distinction from a barely-there magnitude gap, using generic convolution weights with no structural hint to work with.

Replacing that with an **11-channel one-hot encoding** (channel 0 = hidden mask, channel 1 = revealed mask, channels 2-10 = one-hot revealed count 0-8) makes the distinction explicit instead of learned: a hidden cell lights up exactly channel 0; a revealed `3` lights up channels 1 and 5. A convolution kernel can now read "which category is this cell, and what surrounds it" directly from which channels are active at each position, rather than decoding it from a scalar. This is a better fit for Minesweeper specifically because its clues are categorical (hidden vs. one of nine possible counts), not continuous quantities where nearby numbers should be treated as similar.

The richer input alone wasn't free, though: with the same training settings as before, it initially made the network fit aggressively enough to reintroduce a divergence bug (Q-value loss growing unbounded — see `agents/dqn_agent.py`'s module docstring) that had already been fixed once for the single-channel version. A lower learning rate (`1e-4` vs `5e-4`) and less frequent target-network syncs (every 25 episodes vs every 10) — both standard levers for slowing the pace at which the bootstrap target moves — brought training back to stable, bounded loss. With that fix in place, 5x5 win rate improved from 0.5% to ~1.5% on a 6,000-episode run, and on the 4x4/2-mine sanity board, win rate after 6,000 episodes rose from ~48% (single-channel) to ~56% (multi-channel) with a visibly tighter loss curve.

### DQN stability investigation

Longer training runs told a less flattering story. **Double DQN** (decoupling next-action *selection*, via the online network, from next-action *evaluation*, via the target network — instead of one network's `max` doing both, which systematically overestimates) reduced overestimation bias as intended, but a 50,000-episode run still showed loss recovering to single digits early on and then periodically spiking into the thousands past ~episode 10,000 — and the network's *final* snapshot happened to land during one of those unstable stretches, scoring 0.0% despite in-training windows as high as 2.8%. Longer training had made things worse, not better.

To understand and fix this, four 25,000-episode experiments were run from the same seed, each changing exactly one thing from a Double DQN baseline:

| Experiment | Deployed Win Rate | Best-Checkpoint Win Rate | Avg Reward | Loss Mean (2nd half) | Loss Max (2nd half) |
|---|---|---|---|---|---|
| A: Baseline (final weights) | 2.0% | n/a | -6.72 | 8,685 | 70,741 |
| B: + Best-checkpoint selection | 3.0% | 2.0% | -5.93 | 8,685 (identical training to A) | 70,741 (identical training to A) |
| C: + Learning-rate decay (1e-4 → 5e-5 → 1e-5) | **3.5%** | n/a | -5.91 | **98** | **242** |
| D: Smaller network (8/16 filters vs 16/32) | 1.0% | n/a | -7.25 | 147 | 1,385 |

*("2nd half" = episodes 12,500-25,000, where instability has shown up before. "n/a" means that experiment didn't reload a checkpoint for final evaluation, by design — see below.)*

**Why each one moved the numbers:**

- **B (checkpoint selection) trained under identical conditions to A** — same seed, same hyperparameters, same loss curve. The only difference is *which* weights got evaluated: A's raw final weights, or B's best-scoring checkpoint along the way. B's best checkpoint was found at episode 5,000 — right where A's own in-training win rate peaked (3.2% in the 5,000-10,000 window) before degrading as instability set in later. Checkpoint selection didn't change the training dynamics at all; it just stopped a good policy found mid-run from being silently overwritten by a worse one found later. That alone was worth +1 point of win rate (2.0% → 3.0%).
- **C (LR decay) is the more direct fix.** Loss for A and C is identical for the first 10,000 episodes (same `lr=1e-4`); once C drops to `5e-5` at episode 10,000 and `1e-5` at 20,000, its loss stays contained (max 242 in the second half) while A's explodes 300x higher (max 70,741) over the same stretch. Smaller gradient steps later in training mean a handful of large TD-errors can no longer knock a more-converged network out of a good regime — directly addressing *why* the instability happens, not just insuring against its result. This produced both the best deployed win rate (3.5%) and the tightest loss curve of any experiment.
- **D (smaller network) is more stable than baseline but not competitive.** Its second-half loss (mean 147, max 1,385) is far better contained than A's, so reduced capacity does resist the divergence somewhat — but its win rate (1.0%) is the *worst* of the four, including the unstable baseline. Numerical stability and policy quality turned out to be different axes: the smaller network apparently doesn't have enough capacity to learn a competent policy in the first place, stable or not.

**Lesson learned:** stability and performance need to be measured separately, not assumed to move together. Checkpoint selection is a safety net (it can't make training more stable, only stop a good result from being lost), while LR decay is closer to an actual fix (it changes *why* the divergence happens). The combination wasn't tested in this round — each experiment changed exactly one variable from the baseline for a clean, attributable comparison — but running LR decay *and* checkpoint selection together is the natural next step, since they address different failure modes and nothing here suggests they'd conflict. Also worth flagging: `checkpoint_eval_episodes=50` (used above) is a fairly noisy sample for deciding "is this checkpoint actually better," visible in B's best-checkpoint win rate (2.0%, from a noisy 50-episode sample) not matching its own full 200-episode deployed evaluation (3.0%) — a larger per-checkpoint evaluation budget would make selection more reliable at the cost of more training time spent on evaluation rather than learning.

All experiments remain well behind CSP's exact logical deduction on 5x5 — closing that gap is expected to need combining these fixes, substantially more training, and likely further architectural work, not any single change here.

## Technology Stack

**Frontend**
- React
- TypeScript
- Vite
- Tailwind CSS
- Framer Motion
- Recharts

**RL**
- Python
- Gymnasium
- NumPy
- PyTorch
- Pytest

**Planned**
- Stable-Baselines3

## Installation

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### RL Environment

```bash
cd rl
pip install -r requirements.txt

# Run the test suite
pytest

# Compare Random, CSP, Q-Learning, and DQN agents
python -m evaluation.evaluate_agents
```

## Roadmap

- [x] Custom Minesweeper game engine
- [x] Gymnasium-compatible environment
- [x] Random baseline agent
- [x] CSP logical solver
- [x] Tabular Q-Learning agent
- [x] Deep Q-Network (DQN) agent
- [x] Evaluation framework
- [x] React frontend MVP
- [ ] PPO/A2C agent
- [ ] Backend API connecting frontend to trained agents
- [ ] Interactive replay visualization
- [ ] Live agent demonstrations in the browser
