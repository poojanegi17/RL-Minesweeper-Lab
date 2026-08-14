"""Reduce a replay directory to a small, deliberately varied selection.

`generate_replays.py` writes every episode it records, which is the right
behaviour for the generator -- finding a win at Random's 0.15% dense win rate
takes hundreds of episodes. Shipping all of them is not: the site shows one
episode at a time, and a few hundred near-identical two-move losses per board
cell is bulk rather than evidence.

What survives, per directory:

  * every win, up to `--max-wins`, longest first. A win is the episode a
    viewer most wants to see, and the longest one shows the most decision-
    making rather than a lucky one-click cascade.
  * a spread of losses -- shortest, median and longest -- so the selection
    shows how the agent actually fails rather than only its best day. An
    agent that usually dies on move two and occasionally survives twelve is
    misrepresented by either extreme alone.

Episodes are renumbered from 1 so the surviving set reads as a sequence, and
the original episode number is preserved in `source_episode_number` so a
result can still be traced back to the generating run.

Run with:
    python -m evaluation.curate_replays --dry-run results_public/v2/levels/beginner/sparse/replays
    python -m evaluation.curate_replays results_public/v1/levels/**/replays
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load(directory: Path) -> List[Dict[str, Any]]:
    episodes = []
    for path in sorted(directory.glob("*_episode_*.json")):
        try:
            episodes.append({"path": path, "data": json.loads(path.read_text())})
        except (OSError, json.JSONDecodeError):
            continue
    return episodes


def agent_slug(episode: Dict[str, Any]) -> str:
    """The `{slug}_episode_{n}` prefix an episode file is named by, e.g. "dqn"."""
    return str(episode["data"].get("id", episode["path"].stem)).split("_episode_")[0]


def select_per_agent(
    episodes: List[Dict[str, Any]], max_wins: int, max_losses: int, only: Optional[str] = None
) -> List[Dict[str, Any]]:
    """`select` applied within each agent's own episodes rather than across the
    directory.

    A level/density replay directory holds every agent that was recorded there
    (csp, dqn, ppo, random side by side), so selecting globally would rank one
    agent's wins against another's and delete whole agents -- the caps are
    per-agent budgets, not a directory-wide total. The renumbering below already
    assumed per-slug grouping; this makes the selection agree with it.

    `only` restricts the operation to a single agent slug, leaving every other
    agent's files in the directory untouched.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for episode in episodes:
        groups.setdefault(agent_slug(episode), []).append(episode)

    keep: List[Dict[str, Any]] = []
    for slug, group in groups.items():
        # An agent excluded by `only` keeps every one of its episodes.
        keep.extend(group if only is not None and slug != only else select(group, max_wins, max_losses))
    return keep


def select(episodes: List[Dict[str, Any]], max_wins: int, max_losses: int) -> List[Dict[str, Any]]:
    wins = sorted((e for e in episodes if e["data"].get("won")), key=lambda e: -e["data"].get("steps_taken", 0))
    losses = sorted((e for e in episodes if not e["data"].get("won")), key=lambda e: e["data"].get("steps_taken", 0))

    keep = wins[:max_wins]
    if losses:
        # Shortest, median and longest, then fill outward from the median so
        # the set stays representative rather than only extremes.
        picks = [0, len(losses) // 2, len(losses) - 1]
        for offset in range(1, len(losses)):
            if len(set(picks)) >= max_losses:
                break
            picks.extend([len(losses) // 2 - offset, len(losses) // 2 + offset])
        seen, chosen = set(), []
        for i in picks:
            if 0 <= i < len(losses) and i not in seen and len(chosen) < max_losses:
                seen.add(i)
                chosen.append(losses[i])
        keep.extend(chosen)
    return keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim replay directories to a varied selection.")
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--max-wins", type=int, default=3, help="Per agent, not per directory.")
    parser.add_argument("--max-losses", type=int, default=5, help="Per agent, not per directory.")
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Only curate this agent's episodes (the `{slug}_episode_N.json` prefix, e.g. \"dqn\"). "
        "Every other agent's files in the directory are left exactly as they are.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in args.directories:
        if not directory.is_dir():
            print(f"  skip (not a directory): {directory}")
            continue

        episodes = load(directory)
        keep = select_per_agent(episodes, args.max_wins, args.max_losses, args.agent)
        keep_paths = {e["path"] for e in keep}
        wins = sum(1 for e in keep if e["data"].get("won"))

        print(f"  {directory}: {len(episodes)} -> {len(keep)} ({wins} wins, {len(keep) - wins} losses)")
        if args.dry_run:
            continue

        for episode in episodes:
            if episode["path"] not in keep_paths:
                episode["path"].unlink()

        # Renumber the survivors, wins first, so the set reads as a sequence.
        # Numbering restarts per agent: the filename is built from the agent's
        # own slug, so a directory-wide counter would leave each agent after the
        # first starting at someone else's end (csp 1-8, then dqn 9-16) instead
        # of at 1.
        #
        # Two phases: a kept episode's target name is often another kept
        # episode's current name, so writing directly would clobber a file that
        # has not been processed yet and silently lose it.
        renumbered = [e for e in keep if args.agent is None or agent_slug(e) == args.agent]
        for episode in renumbered:
            episode["path"].unlink(missing_ok=True)

        counters: Dict[str, int] = {}
        for episode in renumbered:
            data = episode["data"]
            slug = agent_slug(episode)
            index = counters[slug] = counters.get(slug, 0) + 1
            data["source_episode_number"] = data.get("episode_number")
            data["episode_number"] = index
            data["id"] = f"{slug}_episode_{index}"
            (directory / f"{slug}_episode_{index}.json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
