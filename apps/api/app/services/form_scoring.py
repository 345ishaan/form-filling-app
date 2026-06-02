"""
Compare agent ``filled.json`` to a pre-exported GT JSON (same schema as the agent).

Definitions (per-field, after normalization + Levenshtein):

- **TP** — both agent and GT have a value and edit distance ≤ threshold
- **FN** — GT has a value, agent empty
- **FP** — agent has a value and (GT empty OR edit distance > threshold)

Values are compared lowercase with non-alphanumeric characters removed.
Additional Information overflow fields are excluded (same as the agent iterator).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.services.form_loader import (
    build_forms_export,
    extract_form_field_values,
    infer_form_type,
    is_additional_information_field,
)

DEFAULT_LEVENSHTEIN_THRESHOLD = 2


def _gt_form_type_from_path(pdf_path: Path) -> str:
    stem = pdf_path.stem.lower()
    if "i140" in stem or "i-140" in stem:
        return "i140"
    if "g28" in stem or "g-28" in stem:
        return "g28"
    if "i907" in stem or "i-907" in stem:
        return "i907"
    ft, _ = infer_form_type(pdf_path)
    return ft


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def normalize_for_comparison(value: str) -> str:
    """Lowercase and strip non-alphanumeric characters for fuzzy compare."""
    from app.services.form_loader import normalize_acroform_value

    s = normalize_acroform_value(value) or str(value).strip()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def values_match_fuzzy(
    agent_val: str,
    gt_val: str,
    *,
    threshold: int = DEFAULT_LEVENSHTEIN_THRESHOLD,
) -> bool:
    a = normalize_for_comparison(agent_val)
    g = normalize_for_comparison(gt_val)
    if not a or not g:
        return False
    return levenshtein_distance(a, g) <= threshold


def _has_value(value: str) -> bool:
    return bool(normalize_for_comparison(value))


def parse_filled_export(data: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Parse ``filled.json`` or ``gt_*.json`` (``forms`` wrapper or single form block)."""
    if "forms" in data and isinstance(data["forms"], dict):
        forms_section = data["forms"]
    elif data.get("fields") or data.get("values"):
        ft = str(data.get("form_type") or "custom").strip().lower()
        forms_section = {ft: data}
    else:
        forms_section = {
            k: v for k, v in data.items()
            if k not in ("metrics", "pdf_export", "schema_pdf", "values_pdf") and isinstance(v, dict)
        }

    out: dict[str, dict[str, str]] = {}
    for form_type, block in forms_section.items():
        ft = str(form_type).strip().lower()
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("values"), dict):
            out[ft] = {str(k): str(v or "") for k, v in block["values"].items()}
            continue
        values: dict[str, str] = {}
        fields = block.get("fields")
        if isinstance(fields, list):
            for entry in fields:
                if not isinstance(entry, dict):
                    continue
                fid = entry.get("id")
                if fid:
                    values[str(fid)] = str(entry.get("value") or "")
        else:
            for fid, val in block.items():
                if fid in ("title", "field_count", "filled_count", "fields", "form_type", "values"):
                    continue
                values[str(fid)] = str(val or "")
        out[ft] = values
    return out


def _field_metadata_from_export(data: dict[str, Any], form_type: str) -> dict[str, dict]:
    """field_id → {page, ...} from GT/filled export for additional-info filtering."""
    ft = form_type.strip().lower()
    if "forms" in data and isinstance(data["forms"], dict):
        block = data["forms"].get(ft, {})
    elif str(data.get("form_type", "")).lower() == ft:
        block = data
    else:
        block = data.get(ft, {})
    meta: dict[str, dict] = {}
    for entry in block.get("fields") or []:
        if isinstance(entry, dict) and entry.get("id"):
            meta[str(entry["id"])] = entry
    return meta


def _iter_comparison_fields(
    agent: dict[str, str],
    gt: dict[str, str],
    *,
    gt_meta: dict[str, dict] | None = None,
) -> list[str]:
    ids: list[str] = []
    for fid in sorted(set(agent) | set(gt)):
        page = -1
        if gt_meta and fid in gt_meta:
            page = int(gt_meta[fid].get("page") or -1)
        if is_additional_information_field(fid, page=page):
            continue
        ids.append(fid)
    return ids


def compute_precision_recall(
    agent: dict[str, str],
    gt: dict[str, str],
    *,
    threshold: int = DEFAULT_LEVENSHTEIN_THRESHOLD,
    gt_meta: dict[str, dict] | None = None,
) -> dict[str, Any]:
    tp_keys: list[str] = []
    fp_keys: list[str] = []
    fn_keys: list[str] = []

    for fid in _iter_comparison_fields(agent, gt, gt_meta=gt_meta):
        gt_val = gt.get(fid, "")
        agent_val = agent.get(fid, "")
        gt_has = _has_value(gt_val)
        agent_has = _has_value(agent_val)

        if gt_has and agent_has:
            if values_match_fuzzy(agent_val, gt_val, threshold=threshold):
                tp_keys.append(fid)
            else:
                fp_keys.append(fid)
        elif gt_has and not agent_has:
            fn_keys.append(fid)
        elif agent_has and not gt_has:
            fp_keys.append(fid)

    tp = len(tp_keys)
    fp = len(fp_keys)
    fn = len(fn_keys)
    gt_filled = sum(1 for fid in _iter_comparison_fields(agent, gt, gt_meta=gt_meta) if _has_value(gt.get(fid, "")))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "levenshtein_threshold": threshold,
        "gt_filled_fields": gt_filled,
        "agent_nonempty_total": sum(1 for v in agent.values() if _has_value(v)),
        "fields_compared": len(_iter_comparison_fields(agent, gt, gt_meta=gt_meta)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "score_percent": round(f1 * 100, 1),
        "true_positive_fields": sorted(tp_keys),
        "false_positive_fields": sorted(fp_keys),
        "false_negative_fields": sorted(fn_keys),
    }


def compare_filled_to_gt(
    filled: dict[str, dict[str, str]],
    gt_by_form: dict[str, dict[str, str]],
    *,
    threshold: int = DEFAULT_LEVENSHTEIN_THRESHOLD,
    gt_export: dict[str, Any] | None = None,
) -> dict[str, Any]:
    per_form: dict[str, Any] = {}
    tp = fp = fn = 0

    gt_data = gt_export or {}
    all_forms = sorted(set(filled) | set(gt_by_form))
    for form_type in all_forms:
        agent = filled.get(form_type, {})
        gt = gt_by_form.get(form_type, {})
        if not gt and not agent:
            continue
        gt_meta = _field_metadata_from_export(gt_data, form_type) if gt_data else None
        row = compute_precision_recall(
            agent,
            gt,
            threshold=threshold,
            gt_meta=gt_meta,
        )
        per_form[form_type] = row
        tp += row["true_positive"]
        fp += row["false_positive"]
        fn += row["false_negative"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "levenshtein_threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "score_percent": round(f1 * 100, 1),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "per_form": per_form,
    }


def load_filled_json(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON must be a JSON object")
    return parse_filled_export(data)


def export_gt_json_from_pdfs(
    gt_pdf: Path,
    *,
    schema_pdf: Path,
    form_type: str | None = None,
) -> dict[str, Any]:
    """Build a GT JSON artifact using the blank schema + filled GT PDF."""
    block = extract_form_field_values(gt_pdf, schema_pdf=schema_pdf)
    if form_type:
        block["form_type"] = form_type.strip().lower()
    return {
        **build_forms_export(block),
        "schema_pdf": block["schema_pdf"],
        "values_pdf": block["values_pdf"],
    }


def discover_gt_pdfs(gt_dir: Path) -> dict[str, Path]:
    gt_dir = Path(gt_dir)
    found: dict[str, Path] = {}
    for pdf in sorted(gt_dir.glob("form_*gt*.pdf")):
        if pdf.is_file():
            found[_gt_form_type_from_path(pdf)] = pdf
    return found


def score_filled_json_file(
    filled_path: Path,
    *,
    gt_json: Path | None = None,
    gt_pdf: Path | None = None,
    schema_pdf: Path | None = None,
    gt_dir: Path | None = None,
    form_type: str | None = None,
    threshold: int = DEFAULT_LEVENSHTEIN_THRESHOLD,
) -> dict[str, Any]:
    filled = load_filled_json(filled_path)
    gt_export: dict[str, Any]
    gt_by_form: dict[str, dict[str, str]]

    if gt_json is not None:
        gt_export = json.loads(Path(gt_json).read_text(encoding="utf-8"))
        gt_by_form = parse_filled_export(gt_export)
    elif gt_pdf is not None:
        if schema_pdf is None:
            raise ValueError("schema_pdf (blank template) is required with gt_pdf")
        gt_export = export_gt_json_from_pdfs(
            gt_pdf,
            schema_pdf=schema_pdf,
            form_type=form_type,
        )
        gt_by_form = parse_filled_export(gt_export)
    elif gt_dir is not None:
        gt_dir = Path(gt_dir)
        gt_by_form = {}
        gt_export = {"forms": {}}
        json_paths = sorted(gt_dir.glob("form_*.json"))
        json_paths = [p for p in json_paths if "_gt" not in p.stem]
        if json_paths:
            for path in json_paths:
                data = json.loads(path.read_text(encoding="utf-8"))
                parsed = parse_filled_export(data)
                gt_by_form.update(parsed)
                if "forms" in data:
                    gt_export["forms"].update(data["forms"])
        else:
            blank_dir = schema_pdf or (gt_dir / "blank")
            for ft, pdf in discover_gt_pdfs(gt_dir).items():
                schema = _resolve_schema_pdf(ft, blank_dir)
                block = extract_form_field_values(pdf, schema_pdf=schema)
                gt_export["forms"][ft] = block
                gt_by_form[ft] = block["values"]
    else:
        raise ValueError("Provide gt_json, gt_pdf+schema_pdf, or gt_dir+schema")

    metrics = compare_filled_to_gt(
        filled,
        gt_by_form,
        threshold=threshold,
        gt_export=gt_export,
    )
    return {
        "filled_path": str(filled_path),
        "gt_json": str(gt_json) if gt_json else None,
        "forms_in_export": sorted(filled.keys()),
        "forms_in_gt": sorted(gt_by_form.keys()),
        **metrics,
    }


def _resolve_schema_pdf(form_type: str, blank_dir: Path) -> Path:
    blank_dir = Path(blank_dir)
    names = {
        "g28": ["g-28.pdf"],
        "i140": ["i-140.pdf"],
        "i907": ["i-907.pdf"],
    }.get(form_type, [])
    for name in names:
        p = blank_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No blank schema for {form_type!r} in {blank_dir}")
