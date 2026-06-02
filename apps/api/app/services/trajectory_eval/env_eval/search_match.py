"""Multi-strategy text matching for CMS doc + transcript search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.services.form_scoring import normalize_for_comparison


@dataclass(frozen=True)
class MatchVariant:
    mode: str
    needle: str


def build_match_variants(
    value: str,
    field_id: str = "",
    *,
    include_field_id_hints: bool = True,
) -> list[MatchVariant]:
    """Build case/normalized/digits/email variants for fuzzy presence checks."""
    raw = (value or "").strip()
    variants: list[MatchVariant] = []
    seen: set[str] = set()

    def add(mode: str, needle: str) -> None:
        n = needle.strip()
        if len(n) < 3:
            return
        key = (mode, n)
        if key in seen:
            return
        seen.add(key)
        variants.append(MatchVariant(mode, n))

    norm = normalize_for_comparison(raw)
    if norm:
        add("normalized", norm)
    if raw:
        add("case_insensitive", raw.lower())
        if raw != raw.lower():
            add("case_sensitive", raw)

    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 7:
        add("digits_only", digits)
        if len(digits) == 10:
            add("phone_us", f"{digits[:3]}-{digits[3:6]}-{digits[6:]}")

    if "@" in raw:
        add("email_full", raw.lower())
        local = raw.split("@", 1)[0].strip()
        if len(local) >= 3:
            add("email_local", local.lower())

    # Multi-word: significant tokens
    for tok in re.findall(r"[A-Za-z]{4,}", raw):
        add("token", tok)
        add("token_lower", tok.lower())

    # Optional field-id token hints (doc search only — too noisy for transcript checks)
    if field_id and include_field_id_hints:
        for part in re.findall(r"[A-Za-z]{4,}", field_id):
            if part.lower() not in ("line", "part", "checkbox", "button"):
                add("field_id_part", part.lower())

    return variants[:12]


def text_matches(text: str, variants: list[MatchVariant]) -> str | None:
    """Return matching mode name if any variant appears in text (most specific first)."""
    if not text:
        return None
    norm_blob = normalize_for_comparison(text)
    lower_blob = text.lower()
    digit_blob = re.sub(r"\D", "", text)

    by_mode: dict[str, list[MatchVariant]] = {}
    for v in variants:
        by_mode.setdefault(v.mode, []).append(v)

    def any_needle(mode: str, predicate: Any) -> bool:
        for v in by_mode.get(mode, []):
            if predicate(v.needle):
                return True
        return False

    # Prefer literal / case-precise hits before normalized fuzzy matches.
    if any_needle("case_sensitive", lambda n: n in text):
        return "case_sensitive"
    if any_needle("case_insensitive", lambda n: n in lower_blob):
        return "case_insensitive"
    if any_needle("normalized", lambda n: n in norm_blob):
        return "normalized"
    if any_needle("digits_only", lambda n: n in digit_blob):
        return "digits_only"
    for mode in by_mode:
        if mode in ("case_sensitive", "case_insensitive", "normalized", "digits_only"):
            continue
        for v in by_mode[mode]:
            if v.needle in lower_blob or v.needle in norm_blob or v.needle in text:
                return mode
    return None


def value_in_parsed_docs(
    index: Any,
    value: str,
    field_id: str = "",
) -> tuple[bool, str | None, str | None]:
    """
    Try multiple search + scan strategies against parsed case documents.

    Returns (found, doc_path, match_mode).
    """
    variants = build_match_variants(value, field_id, include_field_id_hints=False)
    if not variants:
        return False, None, None

    index.ensure_catalog()

    # 1) DocumentIndex keyword search (original query styles)
    queries: list[str] = []
    queries.append(" ".join(v.needle for v in variants[:3] if v.mode in ("normalized", "token_lower")))
    if value.strip():
        queries.append(value.strip())
        queries.append(value.strip().lower())
    for q in queries:
        q = q.strip()
        if not q:
            continue
        for hit in index.search(q, top_k=5):
            snippets = " ".join(hit.get("snippets") or [])
            path = hit.get("path")
            full = ""
            if path:
                try:
                    full = index.document_text(path)
                except Exception:
                    full = snippets
            mode = text_matches(full or snippets, variants)
            if mode:
                return True, path, f"search:{mode}"

    # 2) Full-document scan (handles case-sensitive names search() may miss)
    for rel in index.list_paths():
        try:
            doc_text = index.document_text(rel)
        except Exception:
            continue
        mode = text_matches(doc_text, variants)
        if mode:
            return True, rel, f"scan:{mode}"

    return False, None, None


def agent_retrieved_value(
    tool_calls: list[dict],
    value: str,
    field_id: str,
    *,
    normalize_tool_name: Any,
) -> tuple[bool, str | None]:
    """Check search/read tool results for value presence (value tokens only)."""
    variants = build_match_variants(value, field_id, include_field_id_hints=False)
    if not variants:
        return False, None

    for tc in tool_calls:
        name = normalize_tool_name(tc.get("name"))
        if name not in ("search_documents", "read_document", "list_documents"):
            continue
        result = str(tc.get("result") or "")
        mode = text_matches(result, variants)
        if mode:
            return True, f"transcript:{mode}"
    return False, None
