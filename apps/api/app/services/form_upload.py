"""Upload USCIS form PDFs into a CMS session.

The agent — not the API — is responsible for parsing. This module just saves
the bytes to ``workspaces/{sid}/forms/`` so the agent can discover them with
``list_uploaded_forms`` and parse them with ``parse_form``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.services.workspace import (
    commit_workspace_volume_sync,
    get_forms_dir,
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    """Make user-provided filenames safe to write to disk."""
    base = Path(name or "form.pdf").name
    return _SAFE_NAME.sub("_", base) or "form.pdf"


def enrich_forms_with_loaded_state(
    forms: list[dict[str, Any]],
    forms_loaded: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Mark uploads as parsed when matching form_type is loaded in the env."""
    from app.services.form_loader import infer_form_type

    loaded = {fl["form_type"]: fl for fl in (forms_loaded or []) if fl.get("form_type")}
    for entry in forms:
        name = entry.get("filename") or Path(entry.get("pdf_path", "")).name
        form_type, title = infer_form_type(Path(name))
        if form_type in loaded:
            fl = loaded[form_type]
            entry["parsed"] = True
            entry["form_type"] = form_type
            entry["title"] = fl.get("title", title)
            entry["field_count"] = fl.get("field_count")
    return forms


def list_forms_for_session(
    session_id: str,
    *,
    forms_loaded: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """List uploaded PDF forms for a session (used by GET /forms)."""
    forms_dir = get_forms_dir(session_id)
    out: list[dict[str, Any]] = []
    for pdf in sorted(forms_dir.glob("*.pdf")):
        entry: dict[str, Any] = {
            "filename": pdf.name,
            "pdf_path": str(pdf),
            "size_bytes": pdf.stat().st_size,
            "parsed": False,
        }
        out.append(entry)
    return enrich_forms_with_loaded_state(out, forms_loaded)


def save_form_pdf(
    session_id: str,
    *,
    pdf_bytes: bytes,
    filename: str,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """
    Persist an uploaded PDF into ``workspaces/{sid}/forms/``.

    Does NOT parse the PDF and does NOT touch the env. The agent picks up the
    file with ``list_uploaded_forms`` and parses it via ``parse_form``.
    """
    if not pdf_bytes:
        raise ValueError("Empty upload")

    forms_dir = get_forms_dir(session_id)
    safe = _safe_filename(filename)
    if not safe.lower().endswith(".pdf"):
        safe = safe + ".pdf"

    if replace_existing:
        for f in forms_dir.iterdir():
            if f.is_file():
                f.unlink()

    pdf_path = forms_dir / safe
    pdf_path.write_bytes(pdf_bytes)
    commit_workspace_volume_sync()

    return {
        "session_id": session_id,
        "uploaded": {
            "filename": safe,
            "pdf_path": str(pdf_path),
            "workspace_path": f"forms/{safe}",
            "size_bytes": len(pdf_bytes),
        },
        "forms": list_forms_for_session(session_id),
    }


def remove_form_from_session(session_id: str, filename: str) -> None:
    """Delete a previously-uploaded PDF."""
    forms_dir = get_forms_dir(session_id)
    pdf = forms_dir / filename
    pdf.unlink(missing_ok=True)
    commit_workspace_volume_sync()
