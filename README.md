# 🧠 RL Minesweeper Lab

Five agents learning (or deducing) Minesweeper on one shared benchmark, with every number on the
site read live from a real recorded run.

**Live:** [rl-minesweeper-lab.vercel.app](https://rl-minesweeper-lab.vercel.app/) · API on
[Render](https://rl-minesweeper-lab.onrender.com)

---

## 📖 What this is

Minesweeper is a good research board because it is two problems at once. Most of a game is
**deducible** — a revealed `2` with two hidden neighbours proves both are mines. The rest is
**irreducible uncertainty**, where more than one mine layout fits the clues and any move is a bet.

An agent can therefore be measured on two different things: how much of the board it can reason
about, and how well it bets when reasoning runs out. This project compares five approaches on
exactly that, under one evaluation protocol.

The site is a read-only view over committed artifacts. Nothing is simulated in the browser and no
figure is hand-typed into the UI — the API serves the same JSON the training scripts wrote.

---

## 🤖 The five agents

| Agent | How it decides | Learns? |
|---|---|---|
| **Random** | Uniform choice over hidden cells | No — the floor everything else is measured against |
| **CSP** | Constraint propagation over revealed clues; probability estimate when it must guess | No — fixed logic, re-derived every move |
| **Q-Learning** | Tabular value lookup keyed on the exact board pattern | Yes, but only for boards it has seen before |
| **DQN** | Double DQN over an 11-channel encoding, experience replay + target network | Yes, generalizes across similar patterns |
| **PPO** | Actor-critic, GAE, clipped-surrogate objective | Yes, on-policy |

DQN and PPO share the same board encoding and episode budget, so a comparison between them is a
comparison of the algorithms rather than of their inputs.

---

## ⚠️ Two board distributions — read before comparing anything

Every result belongs to one of two environments. They are **different games** and their win rates
must never be subtracted from one another.

| | `first_click_safe` | What it means |
|---|---|---|
| **v1** | `none` | Mines are placed before the first click, so the opening move can lose outright. On a 5×5 board with 5 mines that is a 20% chance of an instant loss no policy could avoid. |
| **v2** | `area` | The 3×3 block around the opening click is guaranteed mine-free — standard desktop Minesweeper behaviour. Every game starts from a cascade. |

A third flag, `guarantee_solvable`, resamples layouts until the board is clearable by deduction
alone. It is **training-only**: evaluation always runs on unfiltered boards, because an agent never
shown a forced guess has learned no way to handle one.

The site has a toggle for this on every table, chart, replay and race. It never mixes the two.

---

## 🏁 Results

The benchmark board: **5×5, 5 mines, 2,000 greedy evaluation episodes, seed 42**. Every agent
faces the identical 2,000 boards.

| Agent | v1 (opening can lose) | v2 (opening safe) |
|---|---|---|
| **CSP** | **43.40%** | 70.35% |
| **DQN** | 38.55% | **77.25%** |
| **Q-Learning** | 1.90% | 71.70% |
| **PPO** | 1.75% | 7.90% |
| **Random** | 0.45% | 1.30% |

Two results here are worth stating plainly.

**A neural network beats the deduction solver.** On protected boards DQN reaches 77.25% against
CSP's 70.35%. CSP is not a heuristic — it is explicit constraint propagation that proves cells safe.

**So does a lookup table.** Q-Learning has no generalization whatsoever; it returns all-zero values
for a board it has not seen and picks at random. It reaches 71.70%, statistically indistinguishable
from CSP (p = 0.35). A protected 5×5 opening simply repeats often enough to memorise — which is
also why it collapses to 1.90% on v1, where the board distribution is far wider.

---

## 📐 Across board sizes and densities

Three board sizes × three mine densities. DQN and PPO are trained once per (level, environment) at
standard density and evaluated at the other two **without retraining**, so density is the only
variable within a row.

**v1 — opening click can lose**

| Board | Mines | Random | Q-Learning | PPO | DQN | CSP |
|---|---|---|---|---|---|---|
| Beginner 5×5 | 3 | 7.30% | 58.10% | 18.55% | 70.35% | **74.45%** |
| | 5 | 0.45% | 1.90% | 1.75% | 38.55% | **43.40%** |
| | 8 | 0.00% | 0.00% | 0.00% | 3.30% | **10.75%** |
| Intermediate 9×9 | 8 | 0.15% | — | 0.20% | 78.90% | **81.45%** |
| | 12 | 0.00% | — | 0.00% | 52.75% | **60.90%** |
| | 18 | 0.00% | — | 0.00% | 8.00% | **23.10%** |
| Expert 16×16 | 30 | 0.00% | — | — | — | **72.65%** |
| | 40 | 0.00% | — | — | — | **51.50%** |
| | 60 | 0.00% | — | — | — | **6.30%** |

**v2 — opening click safe**

| Board | Mines | Random | Q-Learning | PPO | DQN | CSP |
|---|---|---|---|---|---|---|
| Beginner 5×5 | 3 | 10.20% | **91.20%** | 29.85% | 89.40% | 91.10% |
| | 5 | 1.30% | 71.70% | 7.90% | **77.25%** | 70.35% |
| | 8 | 0.15% | 0.60% | 0.95% | **38.90%** | 36.45% |
| Intermediate 9×9 | 8 | 0.10% | — | 0.70% | 97.05% | **98.55%** |
| | 12 | 0.00% | — | 0.00% | 80.15% | **89.95%** |
| | 18 | 0.00% | — | 0.00% | 25.45% | **46.85%** |
| Expert 16×16 | 30 | 0.00% | — | — | — | **95.05%** |
| | 40 | 0.00% | — | — | — | **81.25%** |
| | 60 | 0.00% | — | — | — | **12.75%** |

**Why the dashes are there, not zeros.** Q-Learning is deliberately kept to 5×5 — its table is keyed
on the exact board, and at 9×9 essentially no state ever repeats, so a larger run would measure
nothing. DQN and PPO have no Expert run under the current recipe; those cells are blank rather than
`0.00%` because nothing has been measured, which is a different claim from "measured and lost".

**CSP is the only agent a bigger board helps.** Deductions per game climb steeply — 4.9 at
Beginner, 20.5 at Intermediate, 72.3 at Expert — while forced guesses barely move (2.3 → 1.6 → 1.9).
The deduction-to-guess ratio goes from about 2:1 to 37:1. More board means more structure for the
constraint graph to exploit. That is the exact inverse of what happens to a learned representation.

Density, not size, is what breaks every agent: CSP itself falls to 12.75% at Expert/Dense.

---

## 🔬 The research pipelines

Each learned agent has a per-level pipeline on the Research page, told as a handful of **decisions**
rather than one card per run. Every chapter reads its headline number from the API.

### DQN — Beginner (5×5)

| Chapter | What changed | Result |
|---|---|---|
| **The baseline** | Double DQN, 25,000 episodes, nothing tuned | 11.40% |
| **The ceiling was the budget** | Same configuration, 100,000 episodes | 21.25% |
| **Three fixes that only work together** | `reward_scale` 0.1, `train_every` 4, `epsilon_decay` 0.9997, then the conv head | 37.90% → 38.55% |
| **Changing the game, not the agent** | Trained on no-guess, first-click-safe boards | 77.25% |

Alongside the baseline, four standard levers were each measured against it — learning-rate decay
(12.85%), reward shaping (11.80%), a longer exploration schedule (12.30%) and a deeper network
(6.35%). None clears roughly 13%. That band looked like a ceiling and was not: the same
configuration trained four times longer reaches 21.25%.

The three optimization settings are the largest step in the pipeline, and they **only work
together**. Applied without the reduced replay ratio, reward scaling and slower exploration score
5.95% — worse than changing nothing at all. Their individual contributions sum to 48.85 points
against a bundle worth 16.65.

The architecture change is a null on win rate: 38.55% against 37.90% (p = 0.67). It is kept because
`fully_conv` is 29,089 parameters at *any* board size, which is what makes the next level possible.

The largest single gain came from the board rather than the agent. On a fixed first-click-safe
benchmark the previous chapter's agent scores 54.90% and this one 77.25% — **22.35 points**, with
disjoint confidence intervals, and not one line of the agent changed. Two caveats: the protected
benchmark is an easier game in its own right (worth 16.35 points to an agent that never trained on
one), and the two board settings moved together, so the no-guess curriculum and the safe opening are
not separated.

On its own training distribution — boards where a correct move always exists — that agent reaches
**99.65%**, seven losses in 2,000.

### DQN — Intermediate (9×9)

| Chapter | Result |
|---|---|
| The 5×5 recipe, carried unchanged | 52.75% |
| The same board changes, at 9×9 | 80.15% |

This level exists **because of the conv head**. The old Linear head was board-size-specific, so a
5×5 model could not be built at 9×9 at all. The change that bought no accuracy at Beginner is what
made the transfer possible.

The bigger board is *easier* for this agent — 52.75% against 38.55% at 5×5 — because mine density is
lower (14.8% against 20.0%) and a larger grid leaves more of the board constrained at any moment.
But the advantage over deduction does not survive: at 5×5 this recipe beat CSP at two densities out
of three; at 9×9 it trails CSP at all three. On solvable boards it reaches 92.75%, not the 99.65%
seen at 5×5 — so it is now losing games it had the information to win.

### PPO — Beginner (5×5)

| Chapter | What changed | Result |
|---|---|---|
| **Stable, and barely above Random** | 25,000 episodes; reward shaping and checkpoint selection tried on top | 1.05% |
| **Four times the budget changes nothing** | 100,000 episodes, plus a discount-factor control | 0.90% |
| **The network, and the control it needed** | `fully_conv`, against a matched default-network control | 1.75% |
| **The board is what was holding it back** | First click safe; no-guess and discount controls alongside | 7.90% |

Every change made to the agent came back null. Reward shaping, checkpoint selection, four times the
episodes and the architecture all land inside the noise band. The architecture is null twice: 0.65
points on the original boards (p = 0.11) and 0.05 points once the opening is safe — 7.90% against
7.95%, which is 158 wins against 159.

Only the board moved it: **1.75% → 7.90%**, a 6.15-point gain and the largest effect in PPO's
pipeline.

Two things expected to help made it worse. Restricting training to no-guess boards is a real loss —
10.80% against 13.30% on solvable boards, where the policy that never saw them does better on their
own distribution (p = 0.017). Lowering the discount factor to DQN's 0.9 costs a further 4.85 points.

One place the architecture does measurable work: on no-guess boards the conv head is worth 2.85
points (p = 0.0023), having been worth nothing everywhere else. Why is unresolved — a narrower,
more structured training distribution gives a position-specific Linear head more room to overfit
than a translation-equivariant one, but nothing here tests that.

PPO's best number anywhere is **13.30%**, on boards where a correct move always exists. DQN reaches
99.65% on the same boards.

### PPO — Intermediate (9×9)

| Chapter | Result |
|---|---|
| The 5×5 recipe, carried unchanged | 0.00% (0.20% at sparse) |
| The board change that worked everywhere else | 0.00% (0.70% at sparse) |

Four configurations — both board settings, both networks — all at zero wins in 2,000, and 11 wins
across 400,000 training episodes between them.

The binding constraint is depth. A 9×9 win needs **69 correct reveals**; the median episode ends
after 5 and the deepest ever recorded reached 36. It is not losing winnable games late — it never
reaches a position where a win is in play.

The architecture changes the *failure mode* without changing the outcome. On the default network the
policy collapses to a confident non-winning habit (entropy 0.64). With `fully_conv` it never
collapses at all, plays deeper (139 episodes per 100,000 reach 20+ moves against 29) and produces 11
training wins against 1. None of it converts into an evaluation win.

The comparison with DQN is what this level establishes: same board, same episode budget, same
encoding, comparable gradient updates and now the same network — **52.75% and 80.15% against 0.00%
and 0.00%**.

---

## 🔁 Zero-shot transfer

Because `fully_conv` weights are board-size-independent, a 5×5 checkpoint can be loaded at 9×9 with
no retraining. This is the first thing the conv head made testable.

| Agent | Env | Sparse | Standard | Dense |
|---|---|---|---|---|
| **DQN** transferred from 5×5 | v1 | 53.00% | 26.75% | 2.05% |
| DQN trained at 9×9 | v1 | 78.90% | 52.75% | 8.00% |
| **DQN** transferred from 5×5 | v2 | 78.25% | 40.10% | 3.30% |
| DQN trained at 9×9 | v2 | 97.05% | 80.15% | 25.45% |
| **PPO** transferred from 5×5 | v2 | 1.05% | 0.05% | 0.00% |
| PPO trained at 9×9 | v2 | 0.70% | 0.00% | 0.00% |

DQN keeps roughly **half** its win rate on a board it never trained on; every one of those gaps is
significant. PPO's transferred weights are level with its trained ones (1.05% against 0.70% at
sparse is p = 0.31, not a difference this evidence carries).

The asymmetry is the finding: skipping training costs DQN half its performance and costs PPO
nothing measurable, because there is nothing at 9×9 that PPO manages to learn.

Transfer figures are generalization results, not matched ones, and are labelled as such everywhere
they appear.

---

## 🧩 Environment and encoding

**Board encoding — 11 channels, one-hot.** A hidden mask, a revealed mask, and one channel per
adjacent-mine count 0–8. Minesweeper's clue numbers are categorical, not continuous: a single scalar
channel would place −1 (hidden) numerically next to 0 (revealed, no neighbouring mines), the two
states it matters most to distinguish. There is no flag channel — the environment only exposes a
reveal action.

**Rewards.** `default` is +1 per safe reveal, −10 for a mine, +10 for a win. `shaped` adds 0.2 per
extra cell a cascade opens with a sharper −15/+20. Shaped runs are always *evaluated* under the
default reward so win rates stay comparable.

**`reward_scale`** multiplies the reward before the loss sees it. At ±10 the TD targets sit outside
Huber loss's quadratic regime, where its gradient is constant and large errors stop being penalised
proportionally; scaling to ±1 puts them back inside it.

**Solvability.** `guarantee_solvable` runs a deduction fixpoint (`environment/solvability.py`) and
resamples until the board is clearable without guessing.

---

## 🏗️ Network presets

| Preset | 5×5 | 9×9 | 16×16 | Head |
|---|---|---|---|---|
| `default` | 111,993 | 348,593 | 1,087,968 | Linear |
| `deep` | 130,489 | 367,089 | 1,106,464 | Linear |
| **`fully_conv`** | **29,089** | **29,089** | **29,089** | 1×1 conv |

*(DQN counts. PPO's are the same shapes plus a critic head — 33,442 at every size for `fully_conv`,
112,122 at 5×5 and 348,722 at 9×9 for `default`.)*

The Linear head was never incidental overhead: it is 94% of the `default` network at 5×5 and 99.4%
at 16×16. Replacing it with 1×1 convolutions changes three things at once — the parameter count
stops scaling with board area, deduction rules become translation-equivariant (a `1` with one hidden
neighbour means the same thing everywhere, so it is learned once instead of per position), and one
set of weights runs at any board size.

Two 3×3 convolutions give a **5×5 receptive field** — the whole board at Beginner, 31% at
Intermediate. `fully_conv` deepens the stack to four layers for a 9×9 receptive field.

`default` remains the default so every committed checkpoint still loads.

---

## 📏 Methodology

**2,000 greedy episodes per figure, fixed evaluation seed 42.** On this benchmark a good agent wins
1–40% of games, so the number of *wins* sets the precision, not the number of episodes. At 2,000
episodes a 2% result carries a 95% interval about 1.2 points wide — narrow enough to separate a
genuine 1.5× difference from noise. Evaluation costs seconds against minutes of training, so there
is no reason to economise on it.

**The same 2,000 boards for every agent** at a given cell, drawn from one fixed seed on a fresh
environment no run trained in. Differences between agents are differences in play, not in luck of
the draw.

**Matched compute for DQN vs PPO.** PPO runs 10 epochs over each rollout at batch 32 — roughly
123,000 gradient updates at Beginner and 150,000 at Intermediate, against DQN's 343,000 for the same
100,000 episodes. The same order, rather than the 20× deficit an unmatched configuration produces.

**Significance.** Comparisons quote Fisher exact p-values and 95% Wilson intervals. Where a
difference is not significant, the text says so rather than implying it.

**Seed variance is measured, and it is large.** Two DQN configurations were re-run at seeds 43 and
44 (`results_public/seed_replication/`):

| Configuration | seed 42 | seed 43 | seed 44 | mean | spread |
|---|---|---|---|---|---|
| Masked target | 11.40% | 12.85% | 12.25% | 12.17% | **1.45 pts** |
| Masked target + LR decay | 12.85% | 8.40% | 10.85% | 10.70% | **4.45 pts** |

A 4.45-point spread from the seed alone is **larger than several differences this project reports as
real**, and it is a direct warning about how to read the Beginner tuning arms, which all sit inside
a band of roughly 6 to 13 percent. The large effects — the training budget (+9.85), the three
optimization fixes (+16.65), the board change (+22.35) — are comfortably outside it. The small ones
should be treated as unresolved rather than as rankings.

Every other figure on this page is a single run per configuration.

**One measured caveat on precision.** The 100,000-episode DQN baseline scores 21.25% on its own
2,000 evaluation boards and 19.15% on a different 2,000, so the evaluation set alone moves a figure
by around two points.

---

## 💻 The application

**Frontend** — React + TypeScript + Vite, Tailwind, framer-motion, Recharts.

- **Home** — live leaderboard as a 3D card wheel, a shared-board race, a playable board with an
  agent side-by-side, and the findings this project did not expect
- **Research** — the five-algorithm pipeline and each agent's per-level chapters
- **Agents / Compare / Replay** — per-agent detail, head-to-head comparison, and step-by-step
  episode playback with each agent's own reasoning (Q-values, action probabilities, CSP deductions)

**Backend** — FastAPI, read-only, no database. Routes: `agents`, `experiments`, `metrics`,
`leaderboard`, `replays`, `races`, `board-configs`. Every route reads from `rl/results_public/` on
each request; nothing is cached or re-saved.

**Replays never record mine positions** — not even in metadata. Rather than storing hidden state and
trusting every consumer not to surface it, the data is simply never written down.

**What ships:** 37 experiment runs, 749 replay episodes, 108 shared-board races, and the full
board-size × density grid under both environments. 297 RL tests and 127 backend tests.

**Committed histories are subsampled.** A 100,000-episode run writes a ~40 MB history JSON, and
shipping those at full resolution would put over a gigabyte of artifacts in git — permanently, since
every re-run would add another copy. `rl/evaluation/compact_public_histories.py` keeps 2,000 evenly
spaced rows per history, which is more than the charts render and 98% smaller.

It **subsamples** rather than averages: every row kept is a row the training loop actually wrote,
with its original episode number, and the first and last are always kept. A chart's x-axis still
runs the full length of the run and every plotted point is a real episode. The full-resolution
originals stay in the gitignored `rl/results/`.

```bash
cd rl
python -m evaluation.compact_public_histories --dry-run   # report, change nothing
python -m evaluation.compact_public_histories             # compact in place
```

---

## 📁 Repository structure

```
├── frontend/                 # React + TypeScript + Vite
│   └── src/
│       ├── api/              # Backend API client
│       ├── components/       # landing/, agent/, research/, replay/, compare/, race/, charts/, ui/
│       ├── pages/            # Home, Agents, AgentDetail, Research, Compare, Replay, About
│       └── lib/              # Adapters between API data and the UI
│
├── backend/                  # FastAPI read-only serving layer
│   └── app/
│       ├── routes/           # agents / experiments / metrics / leaderboard / replays / races / board-configs
│       └── services/         # results_loader, replay_loader, race_loader, board_result_loader
│
└── rl/                       # Python RL environment and agents
    ├── environment/          # Game engine, Gymnasium wrapper, solvability fixpoint
    ├── agents/               # random, csp, q_learning, dqn, ppo
    ├── models/               # DQN / PPO networks and presets
    ├── evaluation/           # Training, evaluation, replay/race generation, re-scoring
    ├── results_public/       # Committed, deployment-safe artifacts the API serves
    │   ├── v1/levels/        # Board grid, unprotected opening
    │   └── v2/levels/        # Same grid, first click safe -- NOT comparable to v1
    ├── results/              # Full local output (gitignored) -- checkpoints, CSVs, raw dumps
    ├── analysis/             # Structural analyses and re-scoring reports
    └── tests/                # Pytest suite
```

---

## ▶️ Running locally

**Backend**

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload      # http://localhost:8000, docs at /docs
```

**Frontend**

```bash
cd frontend
npm install
npm run dev                                  # http://localhost:5173
```

**RL experiments**

```bash
cd rl
pip install -r requirements.txt
pytest                                       # the test suite
python -m evaluation.evaluate_agents         # compare all five agents
```

Training a run, and re-measuring the board grid under one environment:

```bash
python -m evaluation.dqn_experiment --episodes 100000 --rows 5 --cols 5 --mines 5 \
    --network-size fully_conv --train-every 4 --reward-scale 0.1 \
    --epsilon-decay 0.9997 --seed 42 --eval-seed 42 \
    --output-dir results/my_run

python -m evaluation.rebaseline_board_configs --agents dqn --levels beginner \
    --first-click-safe area --checkpoint-experiment my_run
```

---

## 🚀 Deployment

Frontend on Vercel, backend on Render. The backend is stateless and read-only, so a normal
`git clone` plus deploy already has everything it needs.

```bash
cd frontend && npm run build     # static bundle in frontend/dist
cd backend  && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set `VITE_API_URL` as a frontend environment variable pointing at the deployed backend — never
commit a production URL. `frontend/.env.example` documents it.

`MINESWEEPER_RESULTS_DIR` overrides which results tree the backend serves (default
`rl/results_public/`); `MINESWEEPER_CORS_ORIGINS` overrides allowed origins.

---

## 🔭 Future work

**1. DQN at Expert (16×16).** The most valuable missing run, and the only board size where no
learned agent has been trained under the current recipe. Everything needed now exists: `fully_conv`
means one set of weights is valid at 16×16, and the Beginner recipe already carried to 9×9 without
retuning. The open question is whether depth defeats it — a 16×16 win needs roughly 216 correct
reveals against 69 at 9×9, and with `gamma = 0.9` a win that far out discounts to almost nothing.
Two things to vary first: the discount factor, and a curriculum that starts from the transferable
9×9 weights rather than from scratch.

```bash
python -m evaluation.dqn_experiment --episodes 100000 --rows 16 --cols 16 --mines 40 \
    --network-size fully_conv --train-every 4 --reward-scale 0.1 \
    --epsilon-decay 0.9997 --first-click-safe area \
    --seed 42 --eval-seed 42 --output-dir results/dqn_expert_A_fully_conv
```

**2. PPO's rollout length.** Unchanged at 256 steps since the 5×5 runs, and it cannot contain a
69-reveal trajectory. This is the one lever never pulled at 9×9, and it should come before any
further board or reward change.

**3. Finish the multi-seed replication.** Two DQN configurations have three seeds each and already
show a spread of up to 4.45 points (see Methodology). Every other configuration — including all four
PPO chapters and both Intermediate levels — is still a single run. Extending the same three-seed
treatment to the headline configurations is what would turn the smaller differences here from
suggestive into settled.

**4. Separating the two board settings.** The 22.35-point gain at Beginner belongs to `first_click_safe`
and `guarantee_solvable` *together*, because they were changed together. One run with only the safe
opening would split them.

**5. Curriculum training.** 5×5 → 9×9 → 16×16 is architecturally possible now that weights transfer.
The zero-shot numbers above are the baseline any curriculum has to beat.

---

## 🎯 Closing

The headline this project started with was that no learned agent beats explicit deduction. That is
no longer true: given a survivable opening click and a spent training budget, DQN passes CSP at 5×5
and a tabular lookup draws level with it.

What decided that was never the algorithm. Across all five agents the two changes worth the most
were spending the training budget honestly and fixing the board the agent was asked to learn from —
both of which look like methodology rather than machine learning.

The results here are largely one careful pass per configuration, and where the seed was varied it
moved things by up to 4.45 points. That is the honest frame: the large effects are real, the small
ones are not yet decided, and the page says so wherever it matters.
