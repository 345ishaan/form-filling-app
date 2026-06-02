"""Case document text extraction via LiteParse (https://github.com/run-llama/liteparse)."""

from __future__ import annotations

import os
from pathlib import Path

MAX_DOCUMENT_CHARS = 50_000

# Formats LiteParse handles (PDF, Office, images). Plain text is read directly.
LITEPARSE_SUFFIXES = frozenset({
    ".pdf",
    ".doc",
    ".docx",
    ".docm",
    ".odt",
    ".rtf",
    ".ppt",
    ".pptx",
    ".pptm",
    ".odp",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".ods",
    ".csv",
    ".tsv",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".svg",
})

_PLAIN_TEXT_SUFFIXES = frozenset({".txt", ".md"})

# Images are converted to PDF then parsed; OCR should stay on (LiteParse default).
_IMAGE_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg",
})

# Require LibreOffice on PATH (Modal: libreoffice apt package).
_OFFICE_SUFFIXES = frozenset({
    ".doc", ".docx", ".docm", ".odt", ".rtf",
    ".ppt", ".pptx", ".pptm", ".odp",
    ".xls", ".xlsx", ".xlsm", ".ods", ".csv", ".tsv",
})

_parser: object | None = None


def liteparse_ocr_enabled() -> bool:
    """Set ``CMS_LITEPARSE_OCR=1`` to enable OCR for scanned PDFs (slower)."""
    return os.environ.get("CMS_LITEPARSE_OCR", "").strip().lower() in ("1", "true", "yes")


def _resolve_ocr_enabled(file_path: Path, ocr_enabled: bool | None) -> bool:
    if ocr_enabled is not None:
        return ocr_enabled
    if file_path.suffix.lower() in _IMAGE_SUFFIXES:
        return True
    return liteparse_ocr_enabled()


def _liteparse_kwargs(file_path: Path, ocr_enabled: bool) -> dict:
    """CMS-tuned LiteParse options (see LiteParse README CLI defaults)."""
    return {
        "ocr_enabled": ocr_enabled,
        "ocr_language": "en",
        "dpi": 150,
        "max_pages": 1000,
        "preserve_very_small_text": False,
    }


def _get_parser():
    global _parser
    if _parser is None:
        from liteparse import LiteParse

        _parser = LiteParse()
    return _parser


def cache_path_for(cache_dir: Path, rel_path: str) -> Path:
    """Mirror ``case/rel/foo.pdf`` → ``parsed/rel/foo.txt``."""
    rel = Path(rel_path)
    return cache_dir / rel.with_suffix(".txt")


def parse_case_document(
    file_path: Path,
    *,
    cache_path: Path | None = None,
    ocr_enabled: bool | None = None,
) -> str:
    """
    Extract text from a case evidence file.

    Uses LiteParse for binary/office formats; reads ``.txt``/``.md`` directly.
    Optionally writes through to ``cache_path`` (session ``parsed/`` tree).
    """
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    if cache_path is not None and cache_path.is_file():
        return cache_path.read_text(encoding="utf-8", errors="replace")[:MAX_DOCUMENT_CHARS]

    suffix = file_path.suffix.lower()
    if suffix in _PLAIN_TEXT_SUFFIXES:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    elif suffix in LITEPARSE_SUFFIXES:
        ocr = _resolve_ocr_enabled(file_path, ocr_enabled)
        try:
            result = _get_parser().parse(
                file_path,
                **_liteparse_kwargs(file_path, ocr),
            )
            text = result.text or ""
        except Exception as exc:
            hint = ""
            if suffix in _OFFICE_SUFFIXES:
                hint = " (install LibreOffice for .docx/.xlsx — see LiteParse README)"
            elif suffix in _IMAGE_SUFFIXES:
                hint = " (install ImageMagick for images — see LiteParse README)"
            raise RuntimeError(f"LiteParse failed for {file_path.name}: {exc}{hint}") from exc
    else:
        return f"[Unsupported case document type: {file_path.name}]"

    text = text[:MAX_DOCUMENT_CHARS]
    if cache_path is not None and text:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    return text
