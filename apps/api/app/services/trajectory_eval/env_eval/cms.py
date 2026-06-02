"""Programmatic CMS env analysis: GT scoring + workflow + FN/FP attribution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.environments.cms_env import DocumentIndex
from app.environments.paths import tasks_root
from app.services.form_loader import is_additional_information_field
from app.services.form_scoring import _field_metadata_from_export, parse_filled_export
from app.services.trajectory_eval.env_eval.parsing import parse_tool_result
from app.services.trajectory_eval.env_eval.search_match import (
    agent_retrieved_value,
    value_in_parsed_docs,
)
from app.services.trajectory_eval.programmatic import normalize_tool_name
from app.services.workspace import get_session_workspace

_READ_PATH_RE = re.compile(r'"path"\s*:\s*"([^"]+)"')
_PDF_STRUCTURE_FIELD_RE = re.compile(r"^#(?:area|subform|pageSet)", re.I)

# Human-readable labels for recall-loss causes (shown in REPORT.md).
CAUSE_LABELS: dict[str, str] = {
    "likely_missing_data": "GT value not found in case documents",
    "retrieval_gap": "In case files, but agent never retrieved in transcript",
    "data_present_agent_failed": "Agent retrieved value, but field still wrong/empty",
    "agent_never_retrieved": "Agent never retrieved in transcript (doc presence unknown)",
    "retrieved_but_not_filled": "Retrieved in transcript, but field still wrong/empty",
    "wrong_value_submitted": "Agent submitted a wrong value (false positive)",
}


def _repo_sample_case_dir() -> Path | None:
    """Fallback sample case under ``tasks/cms/sample_case`` (repo or Modal ``/tasks``)."""
    case = tasks_root() / "cms" / "sample_case"
    return case if case.is_dir() else None


def _split_false_negative_fields(
    fn_ids: list[str],
    gt_export: dict[str, Any],
    form_type: str,
) -> tuple[list[str], list[str]]:
    """Split FN ids into user-fillable vs PDF structure/container fields."""
    meta = _field_metadata_from_export(gt_export, form_type) if gt_export else {}
    substantive: list[str] = []
    pdf_internal: list[str] = []
    for fid in fn_ids:
        page = int(meta.get(fid, {}).get("page") or -1) if fid in meta else -1
        if _PDF_STRUCTURE_FIELD_RE.match(fid) or is_additional_information_field(fid, page=page):
            pdf_internal.append(fid)
        else:
            substantive.append(fid)
    return substantive, pdf_internal


def _count_parsed_txt_files(run_dir: Path, session_id: str) -> int | None:
    """Count ``.txt`` files in run or session parsed cache (0 if dir missing)."""
    roots: list[Path] = []
    if (run_dir / "parsed").is_dir():
        roots.append(run_dir / "parsed")
    if session_id:
        parsed = get_session_workspace(session_id, "cms") / "parsed"
        if parsed.is_dir():
            roots.append(parsed)
    for root in roots:
        count = sum(1 for _ in root.rglob("*.txt"))
        if count:
            return count
    return None


def _resolve_case_index(run_dir: Path, session_id: str) -> tuple[DocumentIndex | None, str]:
    """Resolve parsed case index: session workspace, run bundle, or repo sample."""
    candidates: list[tuple[Path, Path | None, str]] = []

    # Prefer run-bundle snapshot (stage 3) over live session workspace.
    if (run_dir / "case").is_dir():
        parsed = run_dir / "parsed" if (run_dir / "parsed").is_dir() else None
        candidates.append((run_dir / "case", parsed, f"run_dir:{run_dir.name}"))

    for sub in ("sample_case", ""):
        case = run_dir / "case" / sub if sub else run_dir / "case"
        if case.is_dir() and any(case.rglob("*")):
            parsed = run_dir / "parsed" if (run_dir / "parsed").is_dir() else None
            candidates.append((case, parsed, f"run_dir:case/{sub or '.'}"))

    if session_id:
        ws = get_session_workspace(session_id, "cms")
        candidates.append((ws / "case", ws / "parsed", f"session:{session_id}"))

    sample = _repo_sample_case_dir()
    if sample is not None:
        candidates.append((sample, None, "repo:tasks/cms/sample_case"))

    for case_dir, parsed_dir, label in candidates:
        if not case_dir.is_dir() or not any(case_dir.rglob("*")):
            continue
        cache = parsed_dir if parsed_dir and parsed_dir.is_dir() else None
        return DocumentIndex(case_dir, parsed_cache_dir=cache), label
    return None, ""


def _summarize_recall_loss(
    fn_rows: list[dict[str, Any]],
    fn_total: int,
    *,
    attribution_mode: str,
) -> dict[str, Any]:
    """Aggregate FN rows into recall-loss buckets with counts and percentages."""
    counts: dict[str, int] = {}
    for row in fn_rows:
        cause = row.get("likely_cause") or "unknown"
        counts[cause] = counts.get(cause, 0) + 1

    attributed = sum(counts.values())
    unattributed = max(0, fn_total - attributed)
    breakdown: list[dict[str, Any]] = []

    for cause, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        pct_fn = round(100 * count / fn_total, 1) if fn_total else 0.0
        breakdown.append({
            "cause": cause,
            "label": CAUSE_LABELS.get(cause, cause),
            "count": count,
            "pct_of_false_negatives": pct_fn,
        })

    if unattributed:
        breakdown.append({
            "cause": "unattributed",
            "label": "Not yet classified (case index unavailable)",
            "count": unattributed,
            "pct_of_false_negatives": round(100 * unattributed / fn_total, 1) if fn_total else 0.0,
        })

    return {
        "attribution_mode": attribution_mode,
        "false_negative_total": fn_total,
        "attributed_count": attributed,
        "unattributed_count": unattributed,
        "breakdown": breakdown,
    }


def _primary_form_scoring(scoring: dict[str, Any]) -> dict[str, Any]:
    """Prefer i140/g28/i907 block from per_form; else first entry."""
    per = scoring.get("per_form") or {}
    for key in ("i140", "g28", "i907"):
        if key in per:
            return per[key]
    if per:
        return next(iter(per.values()))
    return scoring


def _field_rows(
    field_ids: list[str],
    gt_by_form: dict[str, dict[str, str]],
    tool_calls: list[dict],
    index: DocumentIndex | None,
    *,
    kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fid in field_ids:
        gt_val = ""
        for _ft, fields in gt_by_form.items():
            if fid in fields:
                gt_val = fields[fid]
                break

        in_docs = False
        doc_path: str | None = None
        doc_match_mode: str | None = None
        if index and gt_val:
            in_docs, doc_path, doc_match_mode = value_in_parsed_docs(index, gt_val, fid)

        agent_retrieved = False
        retrieve_mode: str | None = None
        if gt_val:
            agent_retrieved, retrieve_mode = agent_retrieved_value(
                tool_calls,
                gt_val,
                fid,
                normalize_tool_name=normalize_tool_name,
            )

        if kind == "fn":
            if index is None:
                cause = (
                    "retrieved_but_not_filled"
                    if agent_retrieved
                    else "agent_never_retrieved"
                )
            elif not in_docs:
                cause = "likely_missing_data"
            elif not agent_retrieved:
                cause = "retrieval_gap"
            else:
                cause = "data_present_agent_failed"
        else:
            cause = "wrong_value_submitted"

        rows.append({
            "field_id": fid,
            "gt_value": gt_val[:80] if gt_val else "",
            "in_parsed_docs": in_docs,
            "doc_path": doc_path,
            "doc_match_mode": doc_match_mode,
            "agent_retrieved": agent_retrieved,
            "retrieve_match_mode": retrieve_mode,
            "likely_cause": cause,
        })
    return rows


def _workflow_from_transcript(tool_calls: list[dict]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    export_attempts = 0
    export_failures = 0
    first_submit_idx: int | None = None
    docs_read: set[str] = set()
    batch_sizes: list[int] = []

    for i, tc in enumerate(tool_calls):
        name = normalize_tool_name(tc.get("name"))
        counts[name] = counts.get(name, 0) + 1

        if name == "submit_form_batch" and first_submit_idx is None:
            first_submit_idx = i

        if name == "read_document":
            inp = tc.get("input") if isinstance(tc.get("input"), dict) else {}
            path = inp.get("path") or inp.get("document_path")
            if path:
                docs_read.add(str(path))

        if name == "next_form_batch":
            inp = tc.get("input") if isinstance(tc.get("input"), dict) else {}
            k = inp.get("k")
            if k is not None:
                try:
                    batch_sizes.append(int(k))
                except (TypeError, ValueError):
                    pass

        if name == "export_filled_pdf":
            export_attempts += 1
            payload = parse_tool_result(tc.get("result"))
            if payload.get("error"):
                export_failures += 1

    prep_calls = 0
    if first_submit_idx is not None:
        for tc in tool_calls[:first_submit_idx]:
            n = normalize_tool_name(tc.get("name"))
            if n in ("search_documents", "read_document", "list_documents", "list_categories"):
                prep_calls += 1

    return {
        "tool_counts": counts,
        "export_filled_pdf_attempts": export_attempts,
        "export_filled_pdf_failures": export_failures,
        "prep_retrieval_calls_before_first_submit": prep_calls,
        "unique_documents_read": len(docs_read),
        "documents_read_sample": sorted(docs_read)[:12],
        "next_form_batch_k_values": batch_sizes,
        "preferred_batch_tools_used": (
            counts.get("next_form_batch", 0) > 0 and counts.get("submit_form_batch", 0) > 0
        ),
        "legacy_single_field_tools": (
            counts.get("next_form_field", 0) + counts.get("submit_form_field", 0)
            + counts.get("fill_form_field", 0)
        ),
    }


def analyze_cms_env(
    run_dir: Path,
    session_id: str,
    tool_calls: list[dict],
) -> dict[str, Any]:
    scoring_path = run_dir / "form_scoring.json"
    filled_path = run_dir / "filled.json"
    gt_path = run_dir / "gt_filled.json"
    summary_path = run_dir / "summary.json"

    if not scoring_path.is_file():
        return {
            "env_type": "cms",
            "note": "No form_scoring.json — run with --gt-json for field-level analysis.",
        }

    scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
    if scoring.get("error"):
        return {"env_type": "cms", "note": f"form_scoring error: {scoring['error']}"}

    form_row = _primary_form_scoring(scoring)
    gt_data = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.is_file() else {}
    gt_by_form = parse_filled_export(gt_data)

    fn_ids = list(form_row.get("false_negative_fields") or [])
    fp_ids = list(form_row.get("false_positive_fields") or [])
    tp_count = int(form_row.get("true_positive") or scoring.get("true_positive") or 0)
    fp_count = int(form_row.get("false_positive") or scoring.get("false_positive") or 0)
    fn_count = int(form_row.get("false_negative") or scoring.get("false_negative") or 0)

    index, case_source = _resolve_case_index(run_dir, session_id)
    case_attribution_available = index is not None

    form_type = next(iter(scoring.get("per_form") or {}), "i140")
    substantive_ids, pdf_internal_ids = _split_false_negative_fields(
        fn_ids, gt_data, form_type,
    )
    if not substantive_ids and fn_ids:
        substantive_ids = fn_ids

    fn_rows_all = _field_rows(fn_ids, gt_by_form, tool_calls, index, kind="fn")
    fn_rows_substantive = _field_rows(substantive_ids, gt_by_form, tool_calls, index, kind="fn")
    fp_rows = _field_rows(fp_ids[:30], gt_by_form, tool_calls, index, kind="fp")

    workflow = _workflow_from_transcript(tool_calls)

    attribution_mode = (
        "case_documents_and_transcript"
        if index is not None
        else "transcript_only"
    )
    recall_loss_all = _summarize_recall_loss(
        fn_rows_all,
        fn_count,
        attribution_mode=attribution_mode,
    )
    recall_loss_substantive = _summarize_recall_loss(
        fn_rows_substantive,
        len(substantive_ids),
        attribution_mode=attribution_mode,
    )

    cause_counts: dict[str, int] = {}
    for row in fn_rows_substantive:
        k = row["likely_cause"]
        cause_counts[k] = cause_counts.get(k, 0) + 1

    fields_submitted: int | None = None
    fields_filled: int | None = None
    fields_total: int | None = None
    form_type: str | None = None

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics") or {}
        fields_submitted = metrics.get("fields_submitted")
        fields_filled = metrics.get("fields_filled")
        fields_total = metrics.get("fields_total")
        form_type = metrics.get("form_type") or summary.get("form_type")

    if filled_path.is_file() and fields_filled is None:
        filled = json.loads(filled_path.read_text(encoding="utf-8"))
        m = filled.get("metrics") or {}
        fields_filled = m.get("fields_filled")
        fields_total = m.get("fields_total")

    return {
        "env_type": "cms",
        "form_type": form_type or "i140",
        "precision": scoring.get("precision"),
        "recall": scoring.get("recall"),
        "f1": scoring.get("f1"),
        "score_percent": scoring.get("score_percent"),
        "true_positive": tp_count,
        "false_positive": fp_count,
        "false_negative": fn_count,
        "gt_filled_fields": form_row.get("gt_filled_fields"),
        "agent_nonempty_fields": form_row.get("agent_nonempty_total"),
        "fields_compared": form_row.get("fields_compared"),
        "fields_submitted": fields_submitted,
        "fields_filled_nonempty": fields_filled,
        "fields_total": fields_total,
        "false_negative_causes": cause_counts,
        "recall_loss": recall_loss_substantive,
        "recall_loss_all": recall_loss_all,
        "pdf_structure_fn_count": len(pdf_internal_ids),
        "substantive_fn_count": len(substantive_ids),
        "false_negative_fields": fn_rows_substantive,
        "false_negative_fields_all": fn_rows_all,
        "false_positive_fields": fp_rows,
        "case_attribution_available": case_attribution_available,
        "case_source": case_source or None,
        "parsed_txt_files": _count_parsed_txt_files(run_dir, session_id),
        **workflow,
    }
