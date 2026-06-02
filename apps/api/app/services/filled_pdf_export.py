"""
Build filled AcroForm PDFs from a CMS ``filled.json`` export + blank templates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.form_loader import infer_form_type, parse_pdf_form, write_filled_pdf
from app.services.form_scoring import parse_filled_export

_BLANK_BY_FORM_TYPE: dict[str, list[str]] = {
    "g28": ["g-28.pdf", "g28.pdf"],
    "i140": ["i-140.pdf", "i140.pdf"],
    "i907": ["i-907.pdf", "i907.pdf"],
}


def resolve_blank_template(form_type: str, template_dir: Path) -> Path | None:
    """Find a blank USCIS PDF for ``form_type`` under ``template_dir``."""
    template_dir = Path(template_dir)
    if not template_dir.is_dir():
        return None

    ft = form_type.strip().lower()
    for name in _BLANK_BY_FORM_TYPE.get(ft, []) + [f"{ft}.pdf"]:
        candidate = template_dir / name
        if candidate.is_file():
            return candidate

    for pdf in sorted(template_dir.glob("*.pdf")):
        inferred, _ = infer_form_type(pdf)
        if inferred == ft:
            return pdf
    return None


def export_filled_pdfs_from_json(
    filled_path: Path,
    *,
    template_dir: Path,
    out_dir: Path | None = None,
    template_pdf: Path | None = None,
    form_type: str | None = None,
    output_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """
    Read ``filled.json`` and write ``form_filled_<form_type>.pdf`` for each form.

    ``template_pdf`` + ``form_type`` fill a single form; otherwise every form in
    the export is resolved via ``template_dir``.
    """
    filled_path = Path(filled_path)
    data = json.loads(filled_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("filled.json must be a JSON object")

    filled_by_form = parse_filled_export(data)
    if not filled_by_form:
        raise ValueError("No forms found in filled export")

    out_dir = Path(out_dir) if out_dir else filled_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    template_dir = Path(template_dir)

    targets: list[tuple[str, dict[str, str]]] = []
    if template_pdf is not None:
        ft = (form_type or infer_form_type(template_pdf)[0]).strip().lower()
        if ft not in filled_by_form:
            raise ValueError(
                f"filled.json has no form {ft!r}; available: {sorted(filled_by_form)}"
            )
        targets = [(ft, filled_by_form[ft])]
    else:
        targets = sorted(filled_by_form.items())

    exports: list[dict[str, Any]] = []
    errors: list[str] = []

    for ft, values in targets:
        template = (
            Path(template_pdf)
            if template_pdf is not None
            else resolve_blank_template(ft, template_dir)
        )
        if template is None or not template.is_file():
            errors.append(f"{ft}: no blank template in {template_dir}")
            continue

        try:
            parsed = parse_pdf_form(template)
            out_path = Path((output_paths or {}).get(ft) or (out_dir / f"form_filled_{ft}.pdf"))
            meta = write_filled_pdf(template, parsed.fields, values, out_path)
            exports.append({
                "form_type": ft,
                "template_pdf": str(template),
                "workspace_path": f"forms/{out_path.name}",
                **meta,
            })
        except Exception as exc:
            errors.append(f"{ft}: {exc}")

    result: dict[str, Any] = {
        "filled_path": str(filled_path),
        "out_dir": str(out_dir),
        "exports": exports,
    }
    if errors:
        result["errors"] = errors
    if not exports and errors:
        result["error"] = "; ".join(errors)
    return result
