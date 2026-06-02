"""Orchestrate trajectory analysis and write REPORT.md to the run directory."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.services.trajectory_eval.context import (
    TrajectoryContext,
    build_trajectory_context,
    build_trajectory_context_from_session,
)
from app.services.trajectory_eval.env_eval import run_env_specific_eval
from app.services.trajectory_eval.llm_eval import evaluate_rubric_batched
from app.services.trajectory_eval.programmatic import analyze_transcript_programmatic
from app.services.trajectory_eval.report import render_report_md

__all__ = [
    "TrajectoryContext",
    "build_trajectory_context",
    "build_trajectory_context_from_session",
    "run_trajectory_analysis",
]


def _trajectory_eval_enabled() -> bool:
    return os.environ.get("TRAJECTORY_EVAL_DISABLE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def load_transcript(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "transcript.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing transcript.json in {run_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


async def run_trajectory_analysis(
    run_dir: Path | str,
    ctx: TrajectoryContext,
    *,
    llm_batch_size: int = 3,
    skip_llm: bool = False,
) -> dict[str, Any]:
    """
    Analyze ``transcript.json`` in ``run_dir`` and write a single ``REPORT.md``
    (programmatic stats + LLM rubric). Does not read reward, summary scores, or GT.
    """
    run_dir = Path(run_dir)
    transcript = load_transcript(run_dir)
    transcript_path = run_dir / "transcript.json"

    programmatic = analyze_transcript_programmatic(transcript, ctx)

    llm_eval: dict[str, Any] | None = None
    llm_error: str | None = None
    if not skip_llm and _trajectory_eval_enabled():
        try:
            llm_eval = await evaluate_rubric_batched(
                transcript,
                ctx,
                batch_size=llm_batch_size,
            )
        except Exception as exc:  # noqa: BLE001
            llm_error = f"{type(exc).__name__}: {exc}"
            llm_eval = {"error": llm_error, "criteria": {}}

    env_specific = run_env_specific_eval(run_dir, ctx, transcript)
    if env_specific and env_specific.get("recall_loss"):
        payload = {
            "substantive": env_specific.get("recall_loss"),
            "all_scoring_fn": env_specific.get("recall_loss_all"),
            "pdf_structure_fn_count": env_specific.get("pdf_structure_fn_count"),
        }
        (run_dir / "recall_attribution.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    report_md = render_report_md(
        ctx=ctx,
        programmatic=programmatic,
        llm_eval=llm_eval,
        run_dir=run_dir,
        transcript_path=transcript_path,
        env_specific=env_specific,
    )
    report_path = run_dir / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")

    return {
        "report_path": str(report_path),
        "programmatic": programmatic,
        "llm_eval": llm_eval,
        "llm_error": llm_error,
        "env_specific": env_specific,
    }
