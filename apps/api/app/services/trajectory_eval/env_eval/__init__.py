"""Programmatic environment-specific trajectory analysis (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.trajectory_eval.context import TrajectoryContext


def _iter_tool_calls(transcript: dict[str, Any]) -> list[dict]:
    out: list[dict] = []
    for turn in transcript.get("turns") or []:
        if isinstance(turn, dict):
            out.extend(turn.get("tool_calls") or [])
    return out


def run_env_specific_eval(
    run_dir: Path,
    ctx: TrajectoryContext,
    transcript: dict[str, Any],
) -> dict[str, Any] | None:
    from app.services.trajectory_eval.env_eval.cms import analyze_cms_env

    if ctx.env_type.strip().lower() != "cms":
        return None
    tool_calls = _iter_tool_calls(transcript)
    session_id = transcript.get("session_id") or ""
    if not session_id and (run_dir / "summary.json").is_file():
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        session_id = summary.get("session_id") or ""
    return analyze_cms_env(run_dir, session_id, tool_calls)
