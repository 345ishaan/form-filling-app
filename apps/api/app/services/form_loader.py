"""
form_loader — Parse a fillable PDF into a normalized field schema for the agent.

Why this exists
---------------
USCIS PDFs (and most AcroForm PDFs) name fields like::

    form1[0].#subform[0].Pt1Line2a_FamilyName[0]y3ifktwjojcfl45jrv

The trailing 12-22 char suffix is a per-PDF random salt, so the field name
differs between a blank template and a filled GT copy of the same form.
We strip the salt to get a stable id (``Pt1Line2a_FamilyName``) and pull
nearby text on the same page as context so the agent knows what to fill.

Public API
----------
- :func:`parse_pdf_form` — read a PDF, return :class:`ParsedForm`.
- :func:`write_filled_pdf` — write filled values into a copy of the source PDF.
- :func:`infer_form_type` — guess form_type from filename (i140, g28, i907, custom).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# AcroForm field name patterns observed in USCIS PDFs.
# Example: form1[0].#subform[3].Pt1Line2a_FamilyName[0]y3ifktwjojcfl45jrv
_SUBFORM_PREFIX = re.compile(r"^form\d+\[\d+\]\.#subform\[\d+\]\.")
_RANDOM_SUFFIX = re.compile(r"([a-z0-9]{12,})$")
_TRAILING_INDEX0 = re.compile(r"\[0\]$")


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize_id(raw: str) -> str:
    """``form1[0].#subform[3].Pt1Line2a_FamilyName[0]y3if…`` → ``Pt1Line2a_FamilyName``."""
    s = _SUBFORM_PREFIX.sub("", raw)
    s = _RANDOM_SUFFIX.sub("", s)
    s = _TRAILING_INDEX0.sub("", s)
    return s.strip()


_HUMANIZE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^Pt(\d+)Line(\d+\w?)_?", re.I), r"Part \1, Line \2 — "),
    (re.compile(r"^Part(\d+)_Item(\d+)_?", re.I), r"Part \1, Item \2 — "),
    (re.compile(r"^Part(\d+)_Line(\d+\w?)_?", re.I), r"Part \1, Line \2 — "),
    (re.compile(r"^P(\d+)_Line(\d+\w?)_?", re.I), r"Part \1, Line \2 — "),
    (re.compile(r"^P(\d+)_Item(\d+)_?", re.I), r"Part \1, Item \2 — "),
]


def _humanize_label(field_id: str) -> str:
    """Best-effort human-readable label from a normalized id."""
    label = field_id
    for pat, repl in _HUMANIZE_PATTERNS:
        new = pat.sub(repl, label)
        if new != label:
            label = new
            break
    label = label.replace("_", " ").strip(" ——-,")
    # Insert spaces between CamelCase tokens: "FamilyName" → "Family Name"
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    return label


# ── Field type classification ────────────────────────────────────────────────

def _classify(field_obj: dict, value: object) -> str:
    """Map PyPDF2 field flags / values to ``text|checkbox|signature|choice``."""
    ft = field_obj.get("/FT")
    if ft == "/Sig":
        return "signature"
    if ft == "/Btn":
        return "checkbox"
    if ft == "/Ch":
        return "choice"
    if isinstance(value, str) and value.startswith("/") and value != "/":
        # /Off, /Yes, /On — checkbox-ish
        return "checkbox"
    return "text"


def _checkbox_options(field_obj: dict) -> list[str]:
    """For Btn fields, list the on-state value(s) advertised by the PDF."""
    options: list[str] = []
    ap = field_obj.get("/AP") or field_obj.get("/Kids", [])
    try:
        if hasattr(ap, "get_object"):
            ap = ap.get_object()
        if isinstance(ap, dict):
            n = ap.get("/N")
            if hasattr(n, "get_object"):
                n = n.get_object()
            if isinstance(n, dict):
                options = [str(k).lstrip("/") for k in n.keys() if str(k) not in ("/Off", "Off")]
    except Exception:
        pass
    return options


# ── Additional Information overflow pages ────────────────────────────────────

_ABOUT_THE_SECTION = re.compile(r"Additional Information About the", re.I)
_CROSS_REF_PART = re.compile(r"space provided in Part \d+", re.I)
_ADDL_SECTION_HEADER = re.compile(r"Part\s+\d+\.\s+Additional Information", re.I)


def _detect_additional_information_pages(pdf_path: Path) -> set[int]:
    """Return 0-indexed pages that are USCIS overflow 'Additional Information' sheets.

    These pages repeat Page/Part/Item Number rows and free-text continuation lines.
    They are intentionally left blank in our GT PDFs and should not be scored or
    offered to the agent. Instruction cross-references (e.g. 'use Part 6') on other
    pages are ignored; 'About the Petitioner' sections are not overflow pages.
    """
    pages: set[int] = set()
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for pidx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if _ABOUT_THE_SECTION.search(text):
                    continue
                match = _ADDL_SECTION_HEADER.search(text)
                if not match:
                    continue
                tail = text[match.end() : match.end() + 400]
                has_overflow_markers = (
                    "Page Number" in tail
                    or "If you need extra space to provide any additional" in text
                )
                if not has_overflow_markers:
                    continue
                if _CROSS_REF_PART.search(text) and "Page Number" not in tail[:120]:
                    continue
                pages.add(pidx)
    except Exception:
        pass
    return pages


def is_additional_information_field(
    field_id: str,
    *,
    page: int = -1,
    overflow_pages: set[int] | None = None,
) -> bool:
    """True when a field belongs to the USCIS overflow Additional Information section."""
    base = field_id.split("#", 1)[0]
    if re.search(r"Pt9Line", base, re.I):
        return True
    if re.search(r"AdditionalInfo", base, re.I):
        return True
    if overflow_pages and page >= 0 and page in overflow_pages:
        return True
    return False


_STRUCTURE_FIELD_RE = re.compile(
    r"^(#(?:area|subform|pageSet)(?:\[|$)|Page\d|form\d|PDF417)",
    re.I,
)


def is_pdf_structure_field(field_id: str) -> bool:
    """AcroForm container/barcode nodes — not user-fillable G-28 fields."""
    return bool(_STRUCTURE_FIELD_RE.match(field_id.split("#", 1)[0]))


def normalize_acroform_value(value: object) -> str:
    """Map AcroForm ``/V`` to agent-facing strings (``Yes``/``No``/plain text)."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if s.startswith("/"):
        low = s.lower()
        if low in ("/off", "/no"):
            return ""
        if low in ("/yes", "/y", "/on", "/1", "/p"):
            return "Yes"
        # Other short on-states (e.g. ``/N`` on some USCIS widgets).
        if len(low) <= 3:
            return "Yes"
        return s.lstrip("/")
    return s


def extract_gt_values_from_pdf(
    pdf_path: Path,
    *,
    schema_pdf: Path | None = None,
) -> dict[str, str]:
    """Normalized field_id → value (legacy helper; prefers schema-based extraction)."""
    block = extract_form_field_values(pdf_path, schema_pdf=schema_pdf)
    return {k: v for k, v in block["values"].items() if str(v).strip()}


def filter_ground_truth_values(pdf_path: Path, values: dict[str, str]) -> dict[str, str]:
    """Drop Additional Information overflow keys from a normalized GT field map."""
    overflow_pages = _detect_additional_information_pages(pdf_path)
    pos_map = _build_position_map(pdf_path)
    id_to_page: dict[str, int] = {}
    for raw_name, pos in pos_map.items():
        nid = _normalize_id(raw_name)
        if nid:
            id_to_page.setdefault(nid, pos[0])

    out: dict[str, str] = {}
    for field_id, value in values.items():
        base = field_id.split("#", 1)[0]
        page = id_to_page.get(field_id, id_to_page.get(base, -1))
        if is_additional_information_field(field_id, page=page, overflow_pages=overflow_pages):
            continue
        out[field_id] = value
    return out


# ── Context extraction ───────────────────────────────────────────────────────

def _build_position_map(pdf_path: Path) -> dict[str, tuple[int, float, float, float, float]]:
    """raw_field_name → (page_idx, x0, top, x1, bottom). Uses pdfplumber annotations."""
    pos: dict[str, tuple[int, float, float, float, float]] = {}
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for pidx, page in enumerate(pdf.pages):
                if not page.annots:
                    continue
                for ann in page.annots:
                    title = ann.get("title") or ""
                    if not title:
                        continue
                    pos.setdefault(
                        title,
                        (
                            pidx,
                            float(ann.get("x0") or 0.0),
                            float(ann.get("top") or 0.0),
                            float(ann.get("x1") or 0.0),
                            float(ann.get("bottom") or 0.0),
                        ),
                    )
    except Exception:
        pass
    return pos


def _word_in_same_column(word_cx: float, widget_cx: float, page_width: float) -> bool:
    """USCIS forms are often two columns — keep label search on the widget's side."""
    mid_x = page_width / 2.0
    return (word_cx < mid_x) == (widget_cx < mid_x)


def _horizontal_overlap(
    wx0: float, wx1: float, box_x0: float, box_x1: float, pad: float
) -> bool:
    return wx1 >= (box_x0 - pad) and wx0 <= (box_x1 + pad)


# ── Hierarchical USCIS form context (Part → block → line item) ─────────────────

_PART_LINE_RE = re.compile(r"^Part\s+(\d+)\.\s*(.*)$", re.I)
_LINE_KEY_RE = re.compile(r"^(\d+\.[a-z]\.)\s*(.*)$", re.I)
_LINE_KEY_FROM_ID = re.compile(r"Line(\d+)([a-z])", re.I)
_PART_FROM_ID = re.compile(r"Pt(\d+)", re.I)
_BLOCK_HEADER_RE = re.compile(
    r"^(Name of|Address of|Contact Information|Signature of|List the|Client'?s? )",
    re.I,
)
_CONTEXT_BOILERPLATE = re.compile(
    r"(additional information|need extra space|space provided in part|form g-28|"
    r"department of homeland|uscis|omb no\.|expires\s|for\s+uscis)",
    re.I,
)


@dataclass
class _PageTextLine:
    text: str
    top: float
    bottom: float
    x0: float
    x1: float
    column: str  # left | right
    kind: str = "other"  # part | block | line_item | other
    part_num: int | None = None
    section_title: str = ""
    line_key: str = ""
    line_label: str = ""


def _is_context_boilerplate(text: str) -> bool:
    return bool(_CONTEXT_BOILERPLATE.search(text))


def _column_for_anchor(x_anchor: float, page_width: float) -> str:
    return "left" if x_anchor < page_width / 2.0 else "right"


def _column_index_for_anchor(x_anchor: float, page_width: float) -> int:
    """0 = left column, 1 = right column (USCIS read order)."""
    return 0 if x_anchor < page_width / 2.0 else 1


def _field_read_order_key(
    field: FormField,
    pos_map: dict[str, tuple],
    page_widths: list[float],
) -> tuple:
    """Sort key: page, then left column top-to-bottom, then right column, then id."""
    pos = pos_map.get(field.raw_name)
    if pos:
        pidx = int(pos[0])
        x0, y0 = float(pos[1]), float(pos[2])
        pw = page_widths[pidx] if 0 <= pidx < len(page_widths) else 612.0
        col = _column_index_for_anchor(x0, pw)
        return (pidx, col, y0, field.id)
    page = field.page if field.page >= 0 else 999
    return (page, 2, 9999.0, field.id)


def _cluster_column_lines(page) -> dict[str, list[_PageTextLine]]:
    """Group page words into left/right column text lines (USCIS two-column layout)."""
    try:
        words = page.extract_words()
    except Exception:
        return {"left": [], "right": []}
    page_w = float(getattr(page, "width", None) or 612.0)
    mid_x = page_w / 2.0
    column_gap = 8.0

    buckets: dict[tuple[str, int], list[dict]] = {}
    for w in words:
        wx0 = float(w.get("x0", 0))
        wx1 = float(w.get("x1", 0))
        if wx1 <= mid_x - column_gap:
            col = "left"
        elif wx0 >= mid_x - column_gap:
            col = "right"
        else:
            col = "left" if (wx0 + wx1) / 2.0 < mid_x else "right"
        bucket_y = int(round(float(w.get("top", 0)) / 2.0) * 2)
        buckets.setdefault((col, bucket_y), []).append(w)

    out: dict[str, list[_PageTextLine]] = {"left": [], "right": []}
    for (col, _y_key), ws in buckets.items():
        ws.sort(key=lambda w: float(w.get("x0", 0)))
        text = " ".join(str(w.get("text") or "") for w in ws).strip()
        if not text or _is_context_boilerplate(text):
            continue
        x0 = min(float(w.get("x0", 0)) for w in ws)
        x1 = max(float(w.get("x1", 0)) for w in ws)
        top = min(float(w.get("top", 0)) for w in ws)
        bottom = max(float(w.get("bottom", 0)) for w in ws)
        out[col].append(_PageTextLine(text=text, top=top, bottom=bottom, x0=x0, x1=x1, column=col))

    for col in out:
        out[col].sort(key=lambda ln: ln.top)
        _classify_page_lines(out[col])
    return out


def _classify_page_lines(lines: list[_PageTextLine]) -> None:
    for ln in lines:
        m_part = _PART_LINE_RE.match(ln.text.strip())
        if m_part:
            ln.kind = "part"
            ln.part_num = int(m_part.group(1))
            ln.section_title = m_part.group(2).strip()
            continue
        m_line = _LINE_KEY_RE.match(ln.text.strip())
        if m_line:
            ln.kind = "line_item"
            ln.line_key = m_line.group(1).lower()
            ln.line_label = f"{m_line.group(1)} {m_line.group(2)}".strip()
            continue
        # Line number only (e.g. "2.a.") — merge with nearby descriptive words.
        solo = re.match(r"^(\d+\.[a-z]\.)$", ln.text.strip(), re.I)
        if solo:
            ln.kind = "line_item"
            ln.line_key = solo.group(1).lower()
            ln.line_label = solo.group(1)
            continue
        if len(ln.text) >= 12 and _BLOCK_HEADER_RE.match(ln.text):
            ln.kind = "block"
            continue
        if len(ln.text) >= 18 and not re.match(r"^\d+\.", ln.text):
            # Subsection titles without a leading line number.
            if not _is_context_boilerplate(ln.text):
                ln.kind = "block"


def _enrich_line_item_labels(lines: list[_PageTextLine]) -> None:
    """Attach '(Last Name)'-style continuations on the next band to '2.a.' rows."""
    for i, ln in enumerate(lines):
        if ln.kind != "line_item" or not ln.line_key:
            continue
        extras: list[str] = []
        for other in lines:
            if other is ln or other.kind == "part":
                continue
            if abs(other.top - ln.top) > 18:
                continue
            if other.top < ln.top - 2:
                continue
            if other.kind == "line_item" and other.line_key:
                continue
            if re.match(r"^\d+\.[a-z]\.", other.text.strip(), re.I):
                continue
            if _is_context_boilerplate(other.text):
                continue
            extras.append(other.text.strip())
        if extras:
            merged = f"{ln.line_label or ln.line_key} {' '.join(extras)}"
            ln.line_label = re.sub(r"\s+", " ", merged).strip()


def _analyze_page_layout(page) -> dict[str, list[_PageTextLine]]:
    layout = _cluster_column_lines(page)
    for col in layout.values():
        _enrich_line_item_labels(col)
    return layout


def _part_num_from_field_id(field_id: str) -> int | None:
    m = _PART_FROM_ID.search(field_id)
    return int(m.group(1)) if m else None


def _line_key_from_field_id(field_id: str) -> str | None:
    m = _LINE_KEY_FROM_ID.search(field_id)
    if not m:
        return None
    letter = (m.group(2) or "").lower()
    return f"{m.group(1)}.{letter}." if letter else f"{m.group(1)}."


def _normalize_section_title(section: str, block: str) -> str:
    """Repair Part headers truncated at the column gutter (e.g. '…Attorney or')."""
    s = re.sub(r"\s+", " ", section).strip()
    if not s:
        return s
    if re.search(r"\bor\s*$", s, re.I) and "accredited representative" in block.lower():
        s = re.sub(r"\s+or\s*$", "", s, flags=re.I)
        return f"{s} or Accredited Representative"
    return s


def _find_line_label(
    lines: list[_PageTextLine],
    line_key: str | None,
    y_mid: float,
) -> str:
    if not line_key:
        return ""
    best = ""
    best_dist = 1e9
    for ln in lines:
        if ln.kind != "line_item" or ln.line_key != line_key:
            continue
        dist = abs((ln.top + ln.bottom) / 2.0 - y_mid)
        if dist < best_dist:
            best_dist = dist
            best = ln.line_label or ln.text
    return best.strip()


def _build_hierarchical_context(
    layout: dict[str, list[_PageTextLine]],
    bbox: tuple[float, float, float, float],
    field_id: str,
    page_width: float,
) -> str:
    """Nested USCIS context: section (Part) + block header + line item, joined with ' + '."""
    x0, y0, x1, y1 = bbox
    y_mid = (y0 + y1) / 2.0
    column = _column_for_anchor(x0, page_width)
    lines = layout.get(column, [])
    if not lines:
        return ""

    want_part = _part_num_from_field_id(field_id)
    line_key = _line_key_from_field_id(field_id)

    section = ""
    for ln in reversed(lines):
        if ln.top >= y_mid + 4:
            continue
        if ln.kind != "part":
            continue
        if want_part is not None and ln.part_num != want_part:
            continue
        section = ln.section_title
        break

    block = ""
    for ln in reversed(lines):
        if ln.top >= y_mid + 2:
            continue
        if ln.kind != "block":
            continue
        block = ln.text.strip()
        break

    line_label = _find_line_label(lines, line_key, y_mid)

    parts: list[str] = []
    if section:
        parts.append(_normalize_section_title(section, block))
    if block:
        parts.append(block)
    if line_label and line_label.lower() not in (block or "").lower():
        parts.append(line_label)

    return " + ".join(p for p in parts if p)


def _context_for_proximity(
    page, bbox: tuple[float, float, float, float], radius: float = 180.0
) -> str:
    """Fallback: nearest words left/above the widget (column-aware)."""
    x0, y0, x1, y1 = bbox
    try:
        words = page.extract_words()
    except Exception:
        return ""
    page_w = float(getattr(page, "width", None) or 612.0)
    column_anchor = x0
    field_w = max(x1 - x0, 1.0)
    label_overlap_x1 = min(x1, x0 + max(100.0, field_w * 0.45))

    row_pad = 5.0
    row_top, row_bottom = y0 - row_pad, y1 + row_pad
    max_label_width = min(220.0, radius)
    max_above_drop = min(80.0, radius * 0.45)
    above_x_pad = 24.0
    min_gap_above = 8.0

    nearby: list[tuple[int, float, str]] = []
    for w in words:
        wx0 = float(w.get("x0", 0))
        wy0 = float(w.get("top", 0))
        wx1 = float(w.get("x1", 0))
        wy1 = float(w.get("bottom", 0))
        wcx = (wx0 + wx1) / 2.0

        if not _word_in_same_column(wcx, column_anchor, page_w):
            continue

        v_overlap = wy1 >= row_top and wy0 <= row_bottom
        is_left_of = (
            v_overlap
            and wx0 <= x0 + 25
            and wx1 >= x0 - max_label_width
        )
        is_above = (
            _horizontal_overlap(wx0, wx1, x0, label_overlap_x1, above_x_pad)
            and wy1 <= y0 - min_gap_above
            and (y0 - wy1) <= max_above_drop
        )
        if is_left_of or is_above:
            distance = (y0 - wy1) if is_above else max(0.0, x0 - wx1)
            nearby.append((0 if is_left_of else 1, distance, w["text"]))

    nearby.sort(key=lambda t: (t[0], t[1]))
    text = " ".join(tok for _, _, tok in nearby[:12])
    return re.sub(r"\s+", " ", text).strip()


def _context_for(
    page,
    bbox: tuple[float, float, float, float],
    field_id: str = "",
    layout: dict[str, list[_PageTextLine]] | None = None,
    page_width: float = 612.0,
) -> str:
    if layout and field_id:
        ctx = _build_hierarchical_context(layout, bbox, field_id, page_width)
        if ctx:
            return ctx
    return _context_for_proximity(page, bbox)


# ── Form metadata ────────────────────────────────────────────────────────────

_FORM_TYPE_BY_PREFIX = {
    "i140": ("i140", "I-140 (Immigrant Petition for Alien Workers)"),
    "i-140": ("i140", "I-140 (Immigrant Petition for Alien Workers)"),
    "g28": ("g28", "G-28 (Notice of Entry of Appearance as Attorney)"),
    "g-28": ("g28", "G-28 (Notice of Entry of Appearance as Attorney)"),
    "i907": ("i907", "I-907 (Request for Premium Processing)"),
    "i-907": ("i907", "I-907 (Request for Premium Processing)"),
}


def infer_form_type(pdf_path: Path) -> tuple[str, str]:
    """Returns (form_type_id, human_title). Falls back to ``custom``."""
    stem = pdf_path.stem.lower()
    for prefix, (fid, title) in _FORM_TYPE_BY_PREFIX.items():
        if prefix in stem:
            return fid, title
    return "custom", pdf_path.stem


# ── Public dataclasses & API ────────────────────────────────────────────────

@dataclass
class FormField:
    id: str               # normalized stable id (e.g. "Pt1Line2a_FamilyName")
    raw_name: str         # original PDF field name (for round-trip / P/R)
    label: str            # human-readable label
    field_type: str       # text | checkbox | signature | choice
    page: int             # 0-indexed page number; -1 if unknown
    options: list[str]    # choices for checkbox / choice fields
    context: str          # text near the widget (label + surrounding instructions)


@dataclass
class ParsedForm:
    form_type: str
    title: str
    pdf_path: str
    page_count: int
    fields: list[FormField]

    def to_dict(self) -> dict:
        return {
            "form_type": self.form_type,
            "title": self.title,
            "pdf_path": self.pdf_path,
            "page_count": self.page_count,
            "fields": [asdict(f) for f in self.fields],
        }


def parse_pdf_form(pdf_path: Path, *, include_signatures: bool = False) -> ParsedForm:
    """
    Parse a fillable PDF into a normalized :class:`ParsedForm`.

    The agent never sees the ugly raw_name; the schema exposes ``id`` + ``label``
    + ``context``. ``raw_name`` is preserved for reconstruction (P/R against GT).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Form PDF not found: {pdf_path}")

    from PyPDF2 import PdfReader

    reader = PdfReader(str(pdf_path))
    raw_fields = reader.get_fields() or {}
    page_count = len(reader.pages)
    pos_map = _build_position_map(pdf_path)

    # Open once for context extraction.
    pages = None
    page_layouts: dict[int, dict[str, list[_PageTextLine]]] = {}
    try:
        import pdfplumber

        pdf_ctx = pdfplumber.open(pdf_path)
        pages = pdf_ctx.pages
        for pidx, page in enumerate(pages):
            page_layouts[pidx] = _analyze_page_layout(page)
    except Exception:
        pdf_ctx = None

    fields: list[FormField] = []
    seen_ids: dict[str, int] = {}
    try:
        for raw_name, field_obj in raw_fields.items():
            value = field_obj.get("/V")
            ftype = _classify(field_obj, value)
            if ftype == "signature" and not include_signatures:
                continue

            fid = _normalize_id(raw_name)
            if not fid or is_pdf_structure_field(fid):
                continue
            # Disambiguate collisions caused by stripping the random suffix.
            if fid in seen_ids:
                seen_ids[fid] += 1
                fid = f"{fid}#{seen_ids[fid]}"
            else:
                seen_ids[fid] = 1

            pos = pos_map.get(raw_name)
            context = ""
            page_idx = -1
            if pos and pages is not None:
                page_idx = pos[0]
                if 0 <= page_idx < len(pages):
                    page = pages[page_idx]
                    page_w = float(getattr(page, "width", None) or 612.0)
                    bbox = (pos[1], pos[2], pos[3], pos[4])
                    layout = page_layouts.get(page_idx)
                    context = _context_for(page, bbox, fid, layout, page_w)

            options = _checkbox_options(field_obj) if ftype in ("checkbox", "choice") else []
            label = _humanize_label(fid)

            fields.append(
                FormField(
                    id=fid,
                    raw_name=raw_name,
                    label=label,
                    field_type=ftype,
                    page=page_idx,
                    options=options,
                    context=context,
                )
            )
    finally:
        if pdf_ctx is not None:
            pdf_ctx.close()

    overflow_pages = _detect_additional_information_pages(pdf_path)
    fields = [
        f
        for f in fields
        if not is_additional_information_field(f.id, page=f.page, overflow_pages=overflow_pages)
    ]

    page_widths = (
        [float(getattr(p, "width", None) or 612.0) for p in pages]
        if pages
        else []
    )
    # Read order: page → left column (top→bottom) → right column → stable id.
    fields.sort(
        key=lambda f: _field_read_order_key(f, pos_map, page_widths),
    )

    form_type, title = infer_form_type(pdf_path)
    return ParsedForm(
        form_type=form_type,
        title=title,
        pdf_path=str(pdf_path),
        page_count=page_count,
        fields=fields,
    )


def filter_fields(
    fields: Iterable[FormField],
    *,
    drop_signatures: bool = True,
    drop_additional_information: bool = True,
    drop_structure: bool = True,
    overflow_pages: set[int] | None = None,
) -> list[FormField]:
    """Default filter for agent-facing iteration."""
    out: list[FormField] = []
    for f in fields:
        if drop_signatures and f.field_type == "signature":
            continue
        if drop_structure and is_pdf_structure_field(f.id):
            continue
        if drop_additional_information and is_additional_information_field(
            f.id, page=f.page, overflow_pages=overflow_pages
        ):
            continue
        out.append(f)
    return out


def _pdf_object(obj: object) -> object | None:
    if obj is None:
        return None
    if hasattr(obj, "get_object"):
        try:
            return obj.get_object()
        except Exception:
            return None
    return obj


def _widget_field_name(annot: object) -> str:
    t = _pdf_object(annot.get("/T") if hasattr(annot, "get") else None)
    return str(t).strip() if t is not None else ""


def _walk_page_widgets(reader) -> list[tuple[str, str]]:
    """Yield ``(field_name, value)`` for every widget annot (incl. nested subforms).

    ``PdfReader.get_fields()`` only returns a shallow subset on USCIS PDFs (~30 of
    ~150 widgets). Page ``/Annots`` traversal is required for Part 3 client fields, etc.
    """
    out: list[tuple[str, str]] = []

    def walk(annot_ref: object) -> None:
        annot = _pdf_object(annot_ref)
        if annot is None or not hasattr(annot, "get"):
            return
        name = _widget_field_name(annot)
        if name:
            out.append((name, normalize_acroform_value(annot.get("/V"))))
        kids = _pdf_object(annot.get("/Kids"))
        if not kids:
            return
        if not isinstance(kids, list):
            kids = [kids]
        for kid in kids:
            walk(kid)

    for page in reader.pages:
        annots = _pdf_object(page.get("/Annots"))
        if not annots:
            continue
        if not isinstance(annots, list):
            annots = [annots]
        for annot_ref in annots:
            walk(annot_ref)
    return out


def _canonical_value_id(field_id: str) -> str:
    """Map duplicate subform copies (``Pt3Line5a_FamilyName[1]``) to schema id."""
    base = field_id.split("#", 1)[0]
    if re.match(r"^Pt\d+", base, re.I) and re.search(r"\[\d+\]$", base):
        return re.sub(r"\[\d+\]$", "", base)
    return base


def _store_field_value(
    by_raw: dict[str, str],
    by_id: dict[str, str],
    raw_name: str,
    val: str,
) -> None:
    by_raw[raw_name] = val
    nid = _normalize_id(raw_name)
    if not nid or is_pdf_structure_field(nid):
        return
    for key in (nid, _canonical_value_id(nid)):
        if str(val).strip() or key not in by_id:
            by_id[key] = val


def _lookup_field_value(
    by_raw: dict[str, str],
    by_id: dict[str, str],
    *,
    raw_name: str,
    field_id: str,
) -> str:
    base_id = field_id.split("#", 1)[0]
    candidates = [raw_name, base_id, _canonical_value_id(base_id)]
    for key in candidates:
        if key in by_raw and str(by_raw[key]).strip():
            return by_raw[key]
    for key in candidates:
        if key in by_id and str(by_id[key]).strip():
            return by_id[key]
    for key in candidates:
        if key in by_raw:
            return by_raw[key]
    return by_id.get(base_id, by_id.get(_canonical_value_id(base_id), ""))


def _read_pdf_field_values(pdf_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return (by_raw_name, by_normalized_id) AcroForm values from a PDF."""
    from PyPDF2 import PdfReader

    by_raw: dict[str, str] = {}
    by_id: dict[str, str] = {}
    reader = PdfReader(str(pdf_path))

    for name, val in _walk_page_widgets(reader):
        _store_field_value(by_raw, by_id, name, val)

    # Merge shallow ``get_fields()`` (some PDFs only expose values there).
    for raw_name, field_obj in (reader.get_fields() or {}).items():
        val = normalize_acroform_value(field_obj.get("/V"))
        _store_field_value(by_raw, by_id, raw_name, val)

    return by_raw, by_id


def count_pdf_acroform_values(pdf_path: Path) -> dict[str, int]:
    """How many widgets and non-empty ``/V`` values exist in the PDF (not the blank schema)."""
    from PyPDF2 import PdfReader

    reader = PdfReader(str(pdf_path))
    seen: set[str] = set()
    non_empty = 0
    for name, val in _walk_page_widgets(reader):
        nid = _normalize_id(name)
        if not nid or is_pdf_structure_field(nid):
            continue
        canon = _canonical_value_id(nid)
        if canon in seen:
            continue
        seen.add(canon)
        if str(val).strip():
            non_empty += 1
    return {"acroform_fields": len(seen), "acroform_non_empty": non_empty}


def extract_form_field_values(
    values_pdf: Path,
    *,
    schema_pdf: Path | None = None,
) -> dict:
    """
    Extract field values using the same schema path as the agent (``parse_pdf_form``
    + additional-information filter). ``schema_pdf`` is usually the blank template;
    ``values_pdf`` is the filled PDF (GT or agent output).
    """
    values_pdf = Path(values_pdf)
    schema_pdf = Path(schema_pdf or values_pdf)
    if not values_pdf.is_file():
        raise FileNotFoundError(f"Values PDF not found: {values_pdf}")
    if not schema_pdf.is_file():
        raise FileNotFoundError(f"Schema PDF not found: {schema_pdf}")

    parsed = parse_pdf_form(schema_pdf)
    overflow_pages = _detect_additional_information_pages(schema_pdf)
    schema_fields = filter_fields(parsed.fields, overflow_pages=overflow_pages)
    by_raw, by_id = _read_pdf_field_values(values_pdf)
    values_pdf_stats = count_pdf_acroform_values(values_pdf)

    entries: list[dict] = []
    values: dict[str, str] = {}
    for field in schema_fields:
        val = _lookup_field_value(
            by_raw, by_id, raw_name=field.raw_name, field_id=field.id,
        )
        values[field.id] = val
        entries.append({
            "id": field.id,
            "raw_name": field.raw_name,
            "label": field.label,
            "field_type": field.field_type,
            "page": field.page,
            "context": field.context,
            "value": val,
        })

    filled_count = sum(1 for v in values.values() if str(v).strip())
    return {
        "form_type": parsed.form_type,
        "title": parsed.title,
        "schema_pdf": str(schema_pdf),
        "values_pdf": str(values_pdf),
        "field_count": len(entries),
        "filled_count": filled_count,
        "values_pdf_acroform_fields": values_pdf_stats["acroform_fields"],
        "values_pdf_acroform_non_empty": values_pdf_stats["acroform_non_empty"],
        "fields": entries,
        "values": values,
    }


def build_forms_export(block: dict) -> dict[str, dict]:
    """Wrap a single form block as ``{"forms": {form_type: block}}``."""
    form_type = str(block.get("form_type") or "custom").strip().lower()
    return {"forms": {form_type: block}}


# ── Filled PDF export ─────────────────────────────────────────────────────────

def _field_from_dict(data: dict) -> FormField:
    return FormField(
        id=str(data.get("id") or ""),
        raw_name=str(data.get("raw_name") or ""),
        label=str(data.get("label") or ""),
        field_type=str(data.get("field_type") or "text"),
        page=int(data.get("page") or -1),
        options=list(data.get("options") or []),
        context=str(data.get("context") or ""),
    )


def _encode_acroform_value(field: FormField, value: str) -> str | None:
    """Map agent-facing values to AcroForm ``/V`` strings (checkboxes use /Yes, /Off)."""
    if not str(value).strip():
        return None
    if field.field_type != "checkbox":
        return str(value)
    v = str(value).strip().lower()
    if v in ("", "no", "false", "0", "off", "/off"):
        return "/Off"
    on = "/Yes"
    for opt in field.options or []:
        o = str(opt).strip()
        if not o or o.lower() in ("off", "/off"):
            continue
        on = o if o.startswith("/") else f"/{o}"
        break
    return on


def write_filled_pdf(
    pdf_path: Path,
    fields: Iterable[FormField] | Iterable[dict],
    filled: dict[str, str],
    out_path: Path,
) -> dict:
    """Write ``filled`` (normalized field ids) into a copy of ``pdf_path``.

    Uses each field's ``raw_name`` (short AcroForm ``/T``) for
    ``PdfWriter.update_page_form_field_values``. Returns metadata including
    ``out_path`` and ``fields_written``.
    """
    pdf_path = Path(pdf_path)
    out_path = Path(out_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Form PDF not found: {pdf_path}")

    specs: list[FormField] = []
    for item in fields:
        specs.append(item if isinstance(item, FormField) else _field_from_dict(item))

    raw_map: dict[str, str] = {}
    for f in specs:
        if not f.raw_name or not f.id:
            continue
        encoded = _encode_acroform_value(f, filled.get(f.id, ""))
        if encoded is None:
            continue
        raw_map[f.raw_name] = encoded

    from PyPDF2 import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    for page in writer.pages:
        writer.update_page_form_field_values(page, raw_map)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        writer.write(fh)

    return {
        "source_pdf": str(pdf_path),
        "out_path": str(out_path),
        "fields_written": len(raw_map),
    }
