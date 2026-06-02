"""Resolve tasks/ directory (local repo or Modal mount at /tasks)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def tasks_root() -> Path:
    """Repo ``tasks/`` locally, or Modal mount at ``/tasks`` (see ``modal_common._mount_tasks``)."""
    env = os.environ.get("TASKS_ROOT")
    if env:
        return Path(env)
    modal_mount = Path("/tasks")
    if modal_mount.is_dir():
        return modal_mount
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tasks"
        if candidate.is_dir():
            return candidate
    # Fallback for layouts where repo root is four levels above environments/
    return here.parents[4] / "tasks"


# Default Bonza scenarios (one starter puzzle per scene folder).
BONZA_SCENARIOS: dict[str, str] = {
    "scene1": "games/bonza/scene1/1.txt",
    "scene2": "games/bonza/scene2/1.txt",
}


def bonza_puzzle_path(scenario: str, puzzle: str = "1.txt") -> Path | None:
    """Resolve a scenario id to an on-disk puzzle path, or None if missing."""
    rel = BONZA_SCENARIOS.get(scenario)
    if rel is None:
        rel = f"games/bonza/{scenario}/{puzzle}" if "/" not in scenario else scenario
    path = tasks_root() / rel
    return path if path.is_file() else None


def bonza_end_state_words(scenario: str) -> set[str] | None:
    """Legacy: expected word set from ``end_state.json`` (documentation only)."""
    path = tasks_root() / "games" / "bonza" / scenario / "end_state.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    words = data.get("found_words") or data.get("words") or []
    return {str(w).strip().upper() for w in words if str(w).strip()}


def bonza_success_target(scenario: str) -> dict | None:
    """Eval target for fragment Bonza (from ``end_state.json``)."""
    path = tasks_root() / "games" / "bonza" / scenario / "end_state.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    target: dict = {"fragment_count": int(data.get("fragment_count", 1))}
    labels = data.get("fragment_labels")
    if labels is not None:
        target["fragment_labels"] = [str(x).strip() for x in labels if str(x).strip()]
    return target


def bonza_evaluate_found_words(
    found_words: list[str] | None,
    scenario: str,
) -> dict[str, Any] | None:
    """
    Compare agent-found dictionary words (post-play NLTK H/V scan) to GT.

    Returns None when ``end_state.json`` is missing. Otherwise includes
    matched / missing / extra word lists and ``score`` (1 iff all GT words found).
    """
    expected = bonza_end_state_words(scenario)
    if expected is None:
        return None
    found_set = {str(w).strip().upper() for w in (found_words or []) if str(w).strip()}
    matched = expected & found_set
    missing = expected - found_set
    extra = found_set - expected
    return {
        "expected_words": sorted(expected),
        "found_words": sorted(found_set),
        "matched_words": sorted(matched),
        "missing_words": sorted(missing),
        "extra_words": sorted(extra),
        "match_count": len(matched),
        "expected_count": len(expected),
        "score": 1 if not missing else 0,
        "all_matched": not missing,
    }


def bonza_trial_reward(last_info: dict, scenario: str) -> int | None:
    """1/0 from post-play word match; None if no ``end_state.json``."""
    ev = bonza_evaluate_found_words(last_info.get("valid_english_words"), scenario)
    return ev["score"] if ev is not None else None


def games_tasks(*parts: str) -> Path:
    return tasks_root().joinpath("games", *parts)


def cms_tasks(*parts: str) -> Path:
    return tasks_root().joinpath("cms", *parts)
