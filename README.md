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
- Evaluation framework
- Unit tests

**Planned**

- Tabular Q-Learning
- Deep Q-Network (DQN)
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
    │   └── csp_solver.py        # CSP logical solver
    ├── evaluation/
    │   ├── evaluate_agents.py   # Agent comparison script
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
Evaluation Framework
        ↓
Future RL Agents (Q-Learning, DQN, PPO/A2C)
```

- **Minesweeper Engine** (`environment/minesweeper.py`) — Board generation, mine placement, cell reveal/flag logic, and win/loss detection, independent of any RL framework.
- **Gymnasium Environment** (`environment/minesweeper_env.py`) — Wraps the engine in the standard Gymnasium `reset`/`step` API with configurable board size and mine count, so any agent that speaks Gymnasium can play.
- **Random Agent** (`agents/random_agent.py`) — Picks a uniformly random hidden cell each step. Establishes the performance floor.
- **CSP Solver** (`agents/csp_solver.py`) — Builds constraints from revealed numbers ("exactly N of these hidden neighbors are mines"), applies logical deduction to find guaranteed-safe cells and guaranteed mines, and falls back to lowest-probability guessing when no deduction is possible.
- **Evaluation Framework** (`evaluation/`) — Runs agents over many episodes and reports win rate, average episode length, and failure counts on identical board configurations, so agents can be compared fairly.
- **Future RL Agents** — Tabular Q-Learning, DQN, and PPO/A2C will plug into the same environment and evaluation framework once implemented.

## Benchmark Results

| Agent | Type | Win Rate |
|-------|------|----------|
| Random Agent | Baseline | ~0.5% |
| CSP Solver | Logical Solver | ~45.5% |

*Measured on a 5x5 board with 5 mines over 200 episodes (`rl/evaluation/evaluate_agents.py`). These are current benchmark results and will change as more agents are added and algorithms improve.*

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
- Pytest

**Planned**
- PyTorch
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

# Compare Random vs. CSP agents
python -m evaluation.evaluate_agents
```

## Roadmap

- [x] Custom Minesweeper game engine
- [x] Gymnasium-compatible environment
- [x] Random baseline agent
- [x] CSP logical solver
- [x] Evaluation framework
- [x] React frontend MVP
- [ ] Tabular Q-Learning agent
- [ ] Deep Q-Network (DQN) agent
- [ ] PPO/A2C agent
- [ ] Backend API connecting frontend to trained agents
- [ ] Interactive replay visualization
- [ ] Live agent demonstrations in the browser
