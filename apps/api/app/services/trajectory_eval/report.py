"""Render REPORT.md from programmatic + LLM evaluation results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.trajectory_eval.context import TrajectoryContext, tool_catalog
from app.services.trajectory_eval.rubrics import rubric_for_env

_EXTRA_TOOL_DESCRIPTIONS: dict[str, str] = {
    "ToolSearch": "Claude SDK tool discovery (loads tool schemas by query).",
}

def _escape_cell(text: str) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def _tool_usage_rows(
    ctx: TrajectoryContext,
    tool_counts: dict[str, int],
) -> list[tuple[str, str, int]]:
    """All catalog tools (0 if unused) plus transcript-only tools."""
    catalog = tool_catalog(ctx)
    rows: list[tuple[str, str, int]] = []
    for name in sorted(catalog.keys()):
        rows.append((name, catalog[name], int(tool_counts.get(name, 0))))
    for name, count in sorted(tool_counts.items(), key=lambda x: (-x[1], x[0])):
        if name in catalog:
            continue
        desc = _EXTRA_TOOL_DESCRIPTIONS.get(name, "Tool used in trajectory (not in env catalog).")
        rows.append((name, desc, int(count)))
    rows.sort(key=lambda r: (-r[2], r[0]))
    return rows


def render_report_md(
    *,
    ctx: TrajectoryContext,
    programmatic: dict[str, Any],
    llm_eval: dict[str, Any] | None,
    run_dir: Path,
    transcript_path: Path,
    env_specific: dict[str, Any] | None = None,
) -> str:
    tools = programmatic.get("tools") or {}
    tool_counts: dict[str, int] = {
        str(k): int(v) for k, v in (tools.get("tool_calls_by_name") or {}).items()
    }

    meta_parts = [ctx.env_type]
    if ctx.extra.get("bonza_scenario"):
        meta_parts.append(f"scenario={ctx.extra['bonza_scenario']}")
    if ctx.persona and ctx.persona != "none":
        meta_parts.append(f"persona={ctx.persona}")

    lines = [
        "# Trajectory evaluation report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Run:** `{run_dir.name}` ({', '.join(meta_parts)})",
        f"**Transcript:** `{transcript_path.name}` · "
        f"{tools.get('total_turns', 0)} turns · "
        f"{tools.get('total_tool_calls', 0)} tool calls",
        "",
        "## Tool usage",
        "",
        "| Tool | Description | Used |",
        "|------|-------------|-----:|",
    ]

    for name, desc, count in _tool_usage_rows(ctx, tool_counts):
        lines.append(
            f"| `{name}` | {_escape_cell(desc)} | {count} |",
        )

    lines.extend([
        "",
        "## Rubric evaluation",
        "",
    ])

    if llm_eval is None:
        lines.append("_LLM rubric skipped (`TRAJECTORY_EVAL_DISABLE` or `skip_llm`)._")
    elif llm_eval.get("error"):
        lines.append(f"_LLM rubric failed: {llm_eval['error']}_")
    else:
        model = llm_eval.get("model") or "unknown"
        batches = llm_eval.get("batches_processed", 0)
        lines.append(f"Model: `{model}` · batches: {batches}")
        lines.append("")
        lines.append("| Criterion | Score | Reasoning |")
        lines.append("|-----------|------:|-----------|")
        id_to_title = {c.id: c.title for c in rubric_for_env(ctx.env_type)}
        for cid, payload in (llm_eval.get("criteria") or {}).items():
            title = id_to_title.get(cid, cid)
            score = payload.get("score")
            score_str = f"{score}/3" if score is not None else "-"
            reasoning = _escape_cell(payload.get("reasoning") or "")
            lines.append(f"| {title} | {score_str} | {reasoning} |")
        if llm_eval.get("note"):
            lines.append("")
            lines.append(f"_{llm_eval['note']}_")

    lines.extend(_render_env_specific_section(env_specific))

    return "\n".join(lines) + "\n"


def _render_env_specific_section(env_specific: dict[str, Any] | None) -> list[str]:
    lines = ["", "## Environment-specific analysis", ""]
    if not env_specific:
        lines.append("_Not available for this environment._")
        return lines

    env_type = env_specific.get("env_type")
    if env_type == "bonza":
        lines.extend(_render_bonza_env(env_specific))
    elif env_type == "cards":
        lines.extend(_render_cards_env(env_specific))
    elif env_type == "cms":
        lines.extend(_render_cms_env(env_specific))
    else:
        lines.append(f"_Unsupported env type: {env_type}_")
    return lines


def _render_bonza_env(data: dict[str, Any]) -> list[str]:
    lines = [
        "| Metric | Value |",
        "|--------|------:|",
        f"| Total moves | {data.get('total_moves', 0)} |",
        f"| Valid moves | {data.get('valid_moves', 0)} |",
        f"| Invalid moves | {data.get('invalid_moves', 0)} |",
        f"| Invalid rate (%) | {data.get('invalid_rate_pct', 0)} |",
    ]
    by_reason = data.get("invalid_by_reason") or {}
    if by_reason:
        lines.extend(["", "**Invalid move reasons**", "", "| Reason | Count |", "|--------|------:|"])
        for reason, count in sorted(by_reason.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| {_escape_cell(reason)} | {count} |")
    return lines


def _render_cards_env(data: dict[str, Any]) -> list[str]:
    lines = [
        "| Metric | Value |",
        "|--------|------:|",
        f"| view_grid calls | {data.get('view_grid_calls', 0)} |",
        f"| Total picks | {data.get('total_picks', 0)} |",
        f"| Valid triplets (sum=15) | {data.get('valid_triplets', 0)} |",
        f"| Invalid picks | {data.get('invalid_picks', 0)} |",
        f"| Invalid: sum ≠ 15 | {data.get('invalid_sum_not_15', 0)} |",
        f"| Invalid: other | {data.get('invalid_other', 0)} |",
        f"| Picks without prior view_grid | {data.get('picks_without_prior_view_grid', 0)} |",
        f"| Grid readable sum ≠ 15 (invalid pick) | {data.get('grid_readable_sum_not_15', 0)} |",
    ]
    if data.get("bonus_picks") is not None:
        lines.append(f"| Picks with layout bonus (+20) | {data.get('bonus_picks', 0)} |")
        lines.append(f"| Picks baseline only (+15) | {data.get('baseline_picks', 0)} |")
    if data.get("total_reward_from_picks") is not None:
        lines.append(f"| Sum of pick rewards | {data.get('total_reward_from_picks')} |")

    picks = data.get("picks") or []
    if picks:
        lines.extend([
            "",
            "**Pick detail**",
            "",
            "| Positions | Reward | Env sum | Valid | Reason |",
            "|-----------|-------:|--------:|:-----:|--------|",
        ])
        for p in picks[:20]:
            pos = ",".join(str(x) for x in (p.get("positions") or []))
            reward = p.get("reward")
            reward_s = int(reward) if isinstance(reward, (int, float)) and float(reward).is_integer() else reward
            lines.append(
                f"| {pos} | {reward_s if reward is not None else '-'} | "
                f"{p.get('picked_sum', '-')} | "
                f"{'yes' if p.get('valid_triplet') else 'no'} | "
                f"{p.get('failure_reason') or '-'} |",
            )

    lines.extend(_cards_interpretation(data))
    return lines


def _cards_interpretation(data: dict[str, Any]) -> list[str]:
    total = int(data.get("total_picks") or 0)
    invalid = int(data.get("invalid_picks") or 0)
    bonus = int(data.get("bonus_picks") or 0)
    baseline = int(data.get("baseline_picks") or 0)
    reward_sum = data.get("total_reward_from_picks")

    lines = ["", "**Interpretation**", ""]
    if total and invalid == 0:
        lines.append(
            f"- **Environment competence:** All {total} picks were valid (sum=15, unique positions). "
            "Fair to conclude the agent understood the action interface, not only the scoring objective."
        )
    if reward_sum is not None and total:
        extra = int(reward_sum) - 15 * total
        lines.append(
            f"- **Score from picks:** {reward_sum} total ({baseline}×+15, {bonus}×layout bonus +5). "
            f"Bonuses contributed **+{extra}** beyond the triplet baseline for this episode length."
        )
    lines.append(
        "- **Completion strategy:** Greedy valid triplets until the deck is empty is sufficient to finish; "
        "pick order among valid triplets does not affect validity, only whether row/col/diag bonuses apply."
    )
    if bonus > 0 and baseline > 0:
        lines.append(
            "- **Planning pattern:** Mixed policy is common — value-only triplets first, then bonus-targeted picks "
            "after `view_grid`. Bonus rules in the system prompt often draw disproportionate reasoning vs their "
            "marginal points (+5 per layout)."
        )
    lines.append(
        "- **Maximum-score strategy:** Global optimum requires lookahead across deck refills (which cards enter, "
        "which geometric triplets become possible). This run shows local bonus hunting, not full-episode optimization; "
        "the agent did not consistently prioritize maximum score over finishing quickly with valid picks."
    )
    return lines


def _render_cms_env(data: dict[str, Any]) -> list[str]:
    if data.get("note"):
        return [f"_{data['note']}_"]

    form = data.get("form_type") or "form"
    tp = int(data.get("true_positive") or 0)
    fp = int(data.get("false_positive") or 0)
    fn_all = int(data.get("false_negative") or 0)
    pdf_internal = int(data.get("pdf_structure_fn_count") or 0)
    recall_loss = data.get("recall_loss") or {}
    fn_sub = int(recall_loss.get("false_negative_total") or data.get("substantive_fn_count") or 0)

    lines = [
        f"**{form}** · F1 **{data.get('score_percent')}%** (P={data.get('precision')} R={data.get('recall')}) · "
        f"TP {tp} · FP {fp} · FN {fn_all}",
    ]

    if fn_sub:
        by_cause = {r.get("cause"): r for r in (recall_loss.get("breakdown") or [])}
        # Agent fault if GT is in session parsed/ cache; data gap if not (transcript not used here).
        agent_fail = (
            int((by_cause.get("retrieval_gap") or {}).get("count") or 0)
            + int((by_cause.get("data_present_agent_failed") or {}).get("count") or 0)
        )
        data_gap = int((by_cause.get("likely_missing_data") or {}).get("count") or 0)
        agent_pct = round(100 * agent_fail / fn_sub) if fn_sub else 0
        data_pct = round(100 * data_gap / fn_sub) if fn_sub else 0
        lines.append(
            f"**Recall failures:** **{agent_pct}%** ({agent_fail}/{fn_sub}) — GT value found in "
            f"**parsed** case text but the field was still wrong or empty (agent). "
            f"**{data_pct}%** ({data_gap}/{fn_sub}) — GT value not in parsed cache (data / evidence gap)."
        )
        if pdf_internal:
            lines.append(f"_({pdf_internal} additional PDF-internal FNs excluded.)_")
    elif pdf_internal:
        lines.append(f"_({pdf_internal} PDF-internal FNs excluded from attribution.)_")

    return lines
