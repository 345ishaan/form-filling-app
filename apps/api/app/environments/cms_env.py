"""
CMS — Case management / document processing environment (Gymnasium).

Immigration case: agent processes uploaded case files across evidence
categories, fills uploaded USCIS form(s), and answers questions about the case.

Parsed forms are produced by :mod:`app.services.form_loader` (PDF → normalized
field list with context snippets) and loaded directly via action ``9``.
Ground-truth form PDFs are for offline verification only and are never shipped
into Modal sandboxes.

Action types (set on ``action["action_type"]``):
    0 — search_documents(query)
    1 — read_document(path)
    2 — fill_form_field(form, field, value)
    3 — answer_question(question, answer)
    4 — next_form_field(form?)            (single-field iterator)
    5 — submit_form_field(form?, field?, value)
    6 — get_form_progress()
    7 — next_form_batch(k=5, form?)       (batched iterator, preferred)
    8 — submit_form_batch(values={field_id: value})
    9 — load_parsed_form(parsed_form)     (additive — preserves filled progress)
   10 — export_filled_pdf(form?)          (write form_filled_<type>.pdf under forms/)
"""

import json
from pathlib import Path
from typing import Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Form PDF field extraction (parse_form / GT only — not case documents) ───

def _extract_pdf_fields(path: Path) -> dict[str, str]:
    """Extract form field names and values from a fillable PDF."""
    fields = {}
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                if page.annots:
                    for annot in page.annots:
                        if annot and annot.get("title"):
                            key = annot["title"].strip()
                            val = annot.get("contents", "")
                            if isinstance(val, str):
                                fields[key] = val.strip()
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        pdf_fields = reader.get_fields()
        if pdf_fields:
            for key, f in pdf_fields.items():
                if key not in fields and f.get("/V"):
                    fields[key] = str(f["/V"])
    except Exception:
        pass
    return fields


def _is_ground_truth_pdf(path: Path) -> bool:
    """GT form PDFs must never be indexed or shipped to sandboxes."""
    name = path.name.lower()
    return name.endswith("_gt.pdf") or "_gt." in name and name.endswith(".pdf")


# ── Simple document index ───────────────────────────────────────────────────

class DocumentIndex:
    """Lightweight in-memory document store with keyword search."""

    def __init__(self, doc_dir: Path, parsed_cache_dir: Path | None = None):
        self.docs: dict[str, dict] = {}  # relative_path -> {text, size, category, file_path}
        self.doc_dir = doc_dir
        if parsed_cache_dir is not None:
            self.parsed_cache_dir = parsed_cache_dir
        elif doc_dir.name == "case" and doc_dir.parent.is_dir():
            self.parsed_cache_dir = doc_dir.parent / "parsed"
        else:
            self.parsed_cache_dir = None
        self._catalogued = False

    def ensure_catalog(self) -> None:
        """List case files only — no PDF text extraction (fast reset)."""
        if self._catalogued:
            return
        self._catalogued = True
        if not self.doc_dir.is_dir():
            return
        for f in self.doc_dir.rglob("*"):
            if not f.is_file() or f.suffix.lower() == ".ds_store":
                continue
            if _is_ground_truth_pdf(f):
                continue
            rel = str(f.relative_to(self.doc_dir))
            text: str | None = None
            if f.suffix.lower() in (".txt", ".md"):
                text = f.read_text(errors="replace")[:50_000]
            self.docs[rel] = {
                "path": rel,
                "text": text,
                "file_path": f,
                "category": rel.split("/")[0] if "/" in rel else "",
                "size": f.stat().st_size,
            }

    def _ensure_text(self, rel: str) -> str:
        doc = self.docs.get(rel)
        if not doc:
            return ""
        if doc.get("text") is not None:
            return doc["text"] or ""
        fp: Path = doc["file_path"]
        try:
            try:
                from document_parser import cache_path_for, parse_case_document
            except ImportError:
                from app.environments.document_parser import (
                    cache_path_for,
                    parse_case_document,
                )

            cache_path = (
                cache_path_for(self.parsed_cache_dir, rel)
                if self.parsed_cache_dir is not None
                else None
            )
            doc["text"] = parse_case_document(fp, cache_path=cache_path)
        except Exception as exc:
            doc["text"] = f"[Could not parse {fp.name}: {exc}]"[:50_000]
        return doc["text"] or ""

    def index(self):
        """Full index (legacy); prefer lazy ``ensure_catalog`` + ``_ensure_text``."""
        self.ensure_catalog()
        for rel in list(self.docs.keys()):
            self._ensure_text(rel)

    def list_paths(self) -> list[str]:
        self.ensure_catalog()
        return sorted(self.docs.keys())

    def document_text(self, rel: str) -> str:
        """Load (and cache) full parsed text for a case document path."""
        self.ensure_catalog()
        if rel not in self.docs:
            return ""
        return self._ensure_text(rel)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.ensure_catalog()
        query_terms = query.lower().split()
        results = []
        for rel, doc in self.docs.items():
            text = self._ensure_text(rel).lower()
            if not text:
                continue
            score = sum(text.count(t) for t in query_terms)
            if score > 0:
                full_text = self._ensure_text(rel)
                snippets = []
                for term in query_terms:
                    idx = text.find(term)
                    if idx >= 0:
                        start = max(0, idx - 100)
                        end = min(len(full_text), idx + 150)
                        snippet = full_text[start:end]
                        snippets.append(f"...{snippet}...")
                results.append({
                    "path": rel,
                    "category": doc["category"],
                    "score": score,
                    "snippets": snippets[:3],
                })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def get_document(self, path: str) -> Optional[str]:
        self.ensure_catalog()
        if path not in self.docs:
            return None
        text = self._ensure_text(path)
        return text if text else None

    def list_categories(self) -> list[str]:
        self.ensure_catalog()
        cats = sorted({d["category"] for d in self.docs.values()})
        return cats

    def list_documents(self, category: Optional[str] = None) -> list[dict]:
        self.ensure_catalog()
        results = []
        for rel, doc in self.docs.items():
            if category and doc["category"] != category:
                continue
            results.append({"path": rel, "category": doc["category"], "size": doc["size"]})
        results.sort(key=lambda r: r["path"])
        return results


# ── Gymnasium Environment ────────────────────────────────────────────────────

class DocProcessEnv(gym.Env):
    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        case_dir: str | Path | None = None,
        forms_dir: str | Path | None = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.case_dir = Path(case_dir) if case_dir else self._default_case_dir()
        self.forms_dir = Path(forms_dir) if forms_dir else self._default_forms_dir()
        self.render_mode = render_mode

        self.action_space = spaces.Dict({
            # 0=search 1=read 2=fill 3=answer 4=next_field 5=submit_field
            # 6=form_progress 7=next_form_batch 8=submit_form_batch
            # 9=load_parsed_form 10=export_filled_pdf
            "action_type": spaces.Discrete(11),
            "params": spaces.Text(max_length=20_000),
        })

        self.observation_space = spaces.Dict({
            "task": spaces.Text(max_length=2000),
            "score": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "total_fields": spaces.Box(low=0, high=500, shape=(1,), dtype=np.int32),
            "filled_fields": spaces.Box(low=0, high=500, shape=(1,), dtype=np.int32),
        })

        self.index: DocumentIndex | None = None
        # Form schemas keyed by form_type; values are the parsed-form dict
        # produced by ``form_loader.parse_pdf_form().to_dict()``.
        self.forms: dict[str, dict[str, Any]] = {}
        # Quick lookup: (form_type, field_id) -> field spec dict.
        self._field_index: dict[tuple[str, str], dict[str, Any]] = {}
        self.ground_truth: dict[str, dict[str, str]] = {}  # eval only — never in sandboxes
        self.filled: dict[str, dict[str, str]] = {}  # form_type -> {field_id: value}
        self.score: float = 0.0
        self.questions: list[str] = []
        self.answers: dict[str, str] = {}
        self.current_task = ""
        self._field_queue: list[dict[str, Any]] = []
        self._field_cursor: int = 0
        self.hide_ground_truth_in_responses: bool = False

    def _cms_base(self) -> Path:
        try:
            from paths import cms_tasks
        except ImportError:
            from app.environments.paths import cms_tasks
        return cms_tasks()

    def _default_case_dir(self) -> Path:
        return self._cms_base() / "case"

    def _default_forms_dir(self) -> Path:
        return self._cms_base()

    @staticmethod
    def _dir_has_files(path: Path) -> bool:
        if not path.is_dir():
            return False
        for entry in path.rglob("*"):
            if entry.is_file():
                return True
        return False

    def _resolve_case_dir(self) -> Path:
        """Use the uploaded case directory when present; otherwise keep CMS empty."""
        return self.case_dir

    def _resolve_forms_dir(self) -> Path:
        """Session ``forms/`` when case lives under a workspace; else configured forms_dir."""
        ws_forms = self.case_dir.parent / "forms"
        if ws_forms.is_dir():
            return ws_forms
        if self.forms_dir.is_dir() and self._dir_has_files(self.forms_dir):
            return self.forms_dir
        return ws_forms

    def _resolve_form_pdf(self, form_type: str) -> Path | None:
        """Locate the blank template PDF for a loaded form_type."""
        from app.services.form_loader import infer_form_type

        data = self.forms.get(form_type, {})
        pdf_hint = str(data.get("pdf_path") or "")
        forms_dir = self._resolve_forms_dir()
        candidates: list[Path] = []
        if pdf_hint:
            hint = Path(pdf_hint)
            if hint.is_file():
                candidates.append(hint)
            candidates.append(forms_dir / hint.name)
            if not hint.is_absolute():
                candidates.append(self.case_dir.parent / hint)
                candidates.append(forms_dir / hint)
        if forms_dir.is_dir():
            for pdf in sorted(forms_dir.glob("*.pdf")):
                if pdf.name.startswith("form_filled_"):
                    continue
                ft, _ = infer_form_type(pdf)
                if ft == form_type:
                    candidates.append(pdf)
        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    def _load_parsed_form_data(self, data: dict[str, Any]) -> str:
        """Load parsed-form payload into env state.

        Returns the ``form_type`` of the loaded form. Drops previous
        ``_field_index`` entries for that form_type so re-loading is idempotent.
        """
        form_type = str(data.get("form_type") or "custom")
        fields = data.get("fields") or []
        if not isinstance(fields, list):
            raise ValueError(f"parsed_form.fields must be a list; got {type(fields).__name__}")
        # Drop stale field index entries when re-loading the same form_type.
        self._field_index = {
            k: v for k, v in self._field_index.items() if k[0] != form_type
        }
        self.forms[form_type] = data
        for f in fields:
            fid = str(f.get("id") or "")
            if not fid:
                continue
            self._field_index[(form_type, fid)] = f
        return form_type

    def load_parsed_form(self, parsed_form: dict[str, Any]) -> dict[str, Any]:
        """Additively register a parsed form without resetting filled progress.

        After this:
          - ``self.forms[form_type]`` is set / replaced
          - ``self.filled[form_type]`` is initialised (empty) if not present
          - ``_field_queue`` is rebuilt with all currently-loaded forms; the
            cursor is moved to the first un-filled field across all forms.
        """
        form_type = self._load_parsed_form_data(parsed_form)
        self.filled.setdefault(form_type, {})
        self._build_field_queue()
        # Advance cursor past contiguous prefix of already-submitted fields.
        self._field_cursor = 0
        while self._field_cursor < len(self._field_queue):
            spec = self._field_queue[self._field_cursor]
            if spec["field"] in self.filled.get(spec["form"], {}):
                self._field_cursor += 1
            else:
                break
        data = self.forms[form_type]
        return {
            "form_type": form_type,
            "title": data.get("title", form_type),
            "field_count": len(data.get("fields") or []),
            "forms_loaded": [
                {
                    "form_type": ft,
                    "title": self.forms[ft].get("title", ft),
                    "field_count": len(self.forms[ft].get("fields", []) or []),
                }
                for ft in self._form_types()
            ],
            "cursor": self._field_cursor,
            "total_queue": len(self._field_queue),
        }

    def _form_types(self) -> list[str]:
        """All currently loaded form types (preserves insertion order)."""
        return list(self.forms.keys())

    def _load_ground_truth_from_dir(self, gt_dir: Path) -> None:
        """Offline verification only — never call from Modal sandboxes.

        For each currently-loaded form, look for ``form_<type>_gt.pdf`` (or
        ``form_<TYPE>_gt.pdf``) in ``gt_dir`` and extract raw field values.
        Keys are mapped from raw_name → normalized id so they match what
        the agent fills.
        """
        from app.services.form_loader import extract_form_field_values  # noqa: PLC0415

        self.ground_truth = {}
        blank_dir = gt_dir / "blank"
        for form_type in self._form_types():
            for name in (f"form_{form_type}_gt.pdf", f"form_{form_type.upper()}_gt.pdf"):
                p = gt_dir / name
                if not p.is_file():
                    continue
                schema = self._resolve_gt_schema_pdf(form_type, blank_dir)
                block = (
                    extract_form_field_values(p, schema_pdf=schema)
                    if schema is not None
                    else extract_form_field_values(p)
                )
                self.ground_truth[form_type] = {
                    k: v for k, v in block["values"].items() if str(v).strip()
                }
                break

    def _resolve_gt_schema_pdf(self, form_type: str, blank_dir: Path) -> Path | None:
        names = {
            "g28": ["g-28.pdf"],
            "i140": ["i-140.pdf"],
            "i907": ["i-907.pdf"],
        }.get(form_type, [])
        for name in names:
            p = blank_dir / name
            if p.is_file():
                return p
        return None

    def _has_gt_scoring(self) -> bool:
        return any(self.ground_truth.get(f) for f in self._form_types())

    def _total_field_slots(self) -> int:
        if self._has_gt_scoring():
            return sum(len(self.ground_truth.get(f, {})) for f in self._form_types())
        return sum(len(self.forms.get(f, {}).get("fields", []) or []) for f in self._form_types())

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options and options.get("case_dir"):
            self.case_dir = Path(options["case_dir"])

        case_dir = self._resolve_case_dir()
        try:
            from volume_sync import reload_workspace_volume_if_needed
        except ImportError:
            from app.environments.volume_sync import reload_workspace_volume_if_needed

        reload_workspace_volume_if_needed(case_dir)
        self.index = DocumentIndex(case_dir)
        # Lazy: catalog + PDF parse happen on first search/read, not reset.

        self.forms = {}
        self._field_index = {}
        if options:
            forms = options.get("parsed_forms") or []
            for pf in forms:
                if isinstance(pf, dict):
                    self._load_parsed_form_data(pf)

        self.ground_truth = {}
        if options and options.get("load_ground_truth"):
            gt_dir = Path(options["ground_truth_dir"])
            self._load_ground_truth_from_dir(gt_dir)

        self.filled = {ft: {} for ft in self._form_types()}
        if options and options.get("filled"):
            preserved = options["filled"]
            if isinstance(preserved, dict):
                for ft, vals in preserved.items():
                    if ft in self.filled and isinstance(vals, dict):
                        self.filled[ft] = dict(vals)
        self.score = 0.0
        self.questions = []
        self.answers = {}
        self.current_task = "form_filling"
        self.hide_ground_truth_in_responses = True
        if options and options.get("hide_gt") is False and self._has_gt_scoring():
            self.hide_ground_truth_in_responses = False
        self._build_field_queue()
        self._field_cursor = 0
        while self._field_cursor < len(self._field_queue):
            spec = self._field_queue[self._field_cursor]
            if spec["field"] in self.filled.get(spec["form"], {}):
                self._field_cursor += 1
            else:
                break

        if options and options.get("task") == "qa":
            self.current_task = "qa"
            self.questions = options.get("questions", [
                "What is the beneficiary's full name?",
                "What is the employer's name?",
                "What evidence supports the critical role claim?",
                "List the scholarly articles authored by the beneficiary.",
            ])

        if self.index:
            self.index.ensure_catalog()
        info = {
            "categories": self.index.list_categories() if self.index else [],
            "document_count": len(self.index.docs) if self.index else 0,
            "forms_loaded": [
                {
                    "form_type": ft,
                    "title": self.forms[ft].get("title", ft),
                    "field_count": len(self.forms[ft].get("fields", []) or []),
                }
                for ft in self._form_types()
            ],
            "task": self.current_task,
            "gt_scoring": self._has_gt_scoring(),
            "case_dir": str(case_dir),
        }

        obs = {
            "task": self.current_task,
            "score": np.array([self.score], dtype=np.float32),
            "total_fields": np.array([self._total_field_slots()], dtype=np.int32),
            "filled_fields": np.array([sum(len(v) for v in self.filled.values())], dtype=np.int32),
        }

        return obs, info

    def _build_field_queue(self):
        """Ordered fields: page → left column → right column (from parse_form schema)."""
        self._field_queue = []
        for form_type in self._form_types():
            for f in self.forms[form_type].get("fields", []) or []:
                fid = str(f.get("id") or "")
                if not fid:
                    continue
                self._field_queue.append({
                    "form": form_type,
                    "field": fid,
                    "field_type": str(f.get("field_type") or "text"),
                    "label": str(f.get("label") or fid),
                    "options": list(f.get("options") or []),
                    "page": int(f.get("page", -1)),
                    "context": str(f.get("context") or ""),
                    "entity_hint": self._entity_hint_for_field(fid),
                })

    @staticmethod
    def _entity_hint_for_field(field: str) -> str:
        f = field.lower()
        if "attorney" in f or "g28" in f or "atty" in f or "preparer" in f:
            return "attorney"
        if "petitioner" in f or "employer" in f or "company" in f or "orgname" in f:
            return "petitioner"
        if "beneficiary" in f or "alien" in f or "familyname" in f or "givenname" in f:
            return "beneficiary"
        return "unknown"

    def _case_context_for(self, label: str) -> list[str]:
        """Search case docs for the field's human label (≤2 snippets)."""
        if not self.index or not label:
            return []
        hits = self.index.search(label, top_k=2)
        snippets: list[str] = []
        for h in hits:
            for s in h.get("snippets", [])[:1]:
                snippets.append(f"{h['path']}: {s}")
        return snippets

    def _enrich_field(self, spec: dict) -> dict:
        """Attach iterator metadata + case-document snippets to a queue entry."""
        out = dict(spec)
        out["case_snippets"] = self._case_context_for(spec.get("label") or spec["field"])
        out["index"] = spec.get("index", -1)
        return out

    def _next_field_spec(self, form_filter: str | None = None) -> dict | None:
        ff = (form_filter or "").lower() or None
        while self._field_cursor < len(self._field_queue):
            spec = self._field_queue[self._field_cursor]
            if ff and spec["form"] != ff:
                self._field_cursor += 1
                continue
            enriched = self._enrich_field({**spec, "index": self._field_cursor})
            enriched["remaining"] = len(self._field_queue) - self._field_cursor
            return enriched
        return None

    def _next_form_batch(self, k: int, form_filter: str | None = None) -> dict:
        """Return up to ``k`` next un-submitted fields (does not advance the cursor)."""
        k = max(1, min(int(k or 5), 50))
        ff = (form_filter or "").lower() or None
        batch: list[dict] = []
        cursor = self._field_cursor
        while cursor < len(self._field_queue) and len(batch) < k:
            spec = self._field_queue[cursor]
            if ff and spec["form"] != ff:
                cursor += 1
                continue
            batch.append(self._enrich_field({**spec, "index": cursor}))
            cursor += 1
        return {
            "batch": batch,
            "batch_size": len(batch),
            "cursor": self._field_cursor,
            "remaining": max(0, len(self._field_queue) - self._field_cursor),
            "total": len(self._field_queue),
            "done": cursor >= len(self._field_queue) and not batch,
        }

    def _submit_form_batch(self, values: dict[str, Any], form_filter: str | None = None) -> dict:
        """Fill multiple fields at once. Values are keyed by field_id; if a
        field_id is ambiguous across forms, callers can use ``form_filter`` or
        the qualified form ``"i140/Pt1Line2a_FamilyName"`` form.

        Advances ``_field_cursor`` past any prefix of submitted fields.
        """
        results: list[dict[str, Any]] = []
        ff = (form_filter or "").lower() or None
        for raw_key, value in (values or {}).items():
            if not isinstance(raw_key, str):
                results.append({"field": str(raw_key), "error": "field key must be a string"})
                continue
            if "/" in raw_key:
                form, fid = raw_key.split("/", 1)
                form = form.lower()
            else:
                fid = raw_key
                form = ff or self._infer_form_for_field(fid)
            if not form or (form, fid) not in self._field_index:
                results.append({"field": raw_key, "error": "field not found in loaded forms"})
                continue
            value_s = "" if value is None else str(value)
            self.filled.setdefault(form, {})[fid] = value_s
            fill = self._score_fill(form, fid, value_s)
            results.append({
                "form": form,
                "field": fid,
                "value": value_s,
                "match": fill.get("match"),
                "reward": float(fill.get("reward", 0.0)),
            })

        # Advance cursor past contiguous prefix of submitted fields.
        submitted = {(r["form"], r["field"]) for r in results if "error" not in r}
        while self._field_cursor < len(self._field_queue):
            spec = self._field_queue[self._field_cursor]
            if (spec["form"], spec["field"]) in submitted:
                self._field_cursor += 1
            else:
                break

        filled_total = sum(len(v) for v in self.filled.values())
        return {
            "results": results,
            "submitted": len([r for r in results if "error" not in r]),
            "errors": [r for r in results if "error" in r],
            "cursor": self._field_cursor,
            "remaining": max(0, len(self._field_queue) - self._field_cursor),
            "filled_total": filled_total,
            "total": len(self._field_queue),
        }

    def _infer_form_for_field(self, field_id: str) -> str | None:
        """When a field id is unique across loaded forms, find its form_type."""
        matches = [ft for (ft, fid) in self._field_index.keys() if fid == field_id]
        return matches[0] if len(matches) == 1 else None

    def get_env_info(self) -> dict:
        """Snapshot for /info endpoint (no invalid step)."""
        return {
            "found_words": [],
            "score": self.score,
            "categories": self.index.list_categories() if self.index else [],
            "document_count": len(self.index.docs) if self.index else 0,
            "forms_loaded": [
                {
                    "form_type": ft,
                    "title": self.forms[ft].get("title", ft),
                    "field_count": len(self.forms[ft].get("fields", []) or []),
                }
                for ft in self._form_types()
            ],
            "field_queue_remaining": max(0, len(self._field_queue) - self._field_cursor),
            "gt_scoring": self._has_gt_scoring(),
        }

    def step(self, action: dict):
        action_type = int(action.get("action_type", 0))
        params_str = action.get("params", "{}")

        try:
            params = json.loads(params_str)
        except json.JSONDecodeError:
            params = {"value": params_str}

        reward = 0.0
        info: dict[str, Any] = {"action_type": action_type, "result": None, "error": None}
        terminated = False

        if action_type == 0:  # search
            query = params.get("query", "")
            if query == "__list__":
                info["result"] = {
                    "categories": self.index.list_categories() if self.index else [],
                    "documents": self.index.list_documents(category=params.get("category")) if self.index else [],
                }
                reward = 0.0
            elif self.index:
                results = self.index.search(query)
                info["result"] = results
                reward = 0.1 if results else -0.1
            else:
                info["error"] = "No document index loaded"

        elif action_type == 1:  # read
            path = params.get("path", "")
            if self.index:
                text = self.index.get_document(path)
                if text:
                    info["result"] = {"path": path, "text": text[:5000]}
                    reward = 0.1
                else:
                    info["error"] = f"Document not found: {path}"
                    reward = -0.1
            else:
                info["error"] = "No document index loaded"

        elif action_type == 2:  # fill_form_field
            form = params.get("form", "").lower()
            field = params.get("field", "")
            value = params.get("value", "")

            loaded = self._form_types()
            if form and form not in loaded:
                info["error"] = f"Unknown form: {form}. Loaded: {loaded}"
                reward = -0.5
            elif not field:
                info["error"] = "field name required"
                reward = -0.5
            else:
                if not form:
                    form = self._infer_form_for_field(field) or ""
                if not form:
                    info["error"] = f"Could not infer form for field {field!r}; pass form= explicitly"
                    reward = -0.5
                else:
                    self.filled.setdefault(form, {})[field] = value
                    info["result"] = self._score_fill(form, field, value)
                    reward = float(info["result"].get("reward", 0.0))

        elif action_type == 4:  # next_form_field
            form_filter = params.get("form")
            spec = self._next_field_spec(form_filter)
            if spec is None:
                info["result"] = {"done": True, "message": "All fields processed"}
                reward = 0.0
            else:
                info["result"] = spec
                reward = 0.1

        elif action_type == 5:  # submit_form_field (advances cursor on success)
            form = params.get("form", "").lower()
            field = params.get("field", "")
            value = params.get("value", "")
            current = self._next_field_spec()
            if not field and current:
                field = current["field"]
                form = current["form"]
            if not form and field:
                form = self._infer_form_for_field(field) or ""
            if not form or not field or (form, field) not in self._field_index:
                info["error"] = "form and field required (field must belong to a loaded form)"
                reward = -0.5
            else:
                self.filled.setdefault(form, {})[field] = value
                fill_result = self._score_fill(form, field, value)
                reward = float(fill_result.get("reward", 0.0))
                info["result"] = {
                    "match": fill_result.get("match"),
                    "form": form,
                    "field": field,
                    "value": value,
                }
                if current and current.get("field") == field and current.get("form") == form:
                    self._field_cursor += 1

        elif action_type == 6:  # get_form_progress
            metrics = self._compute_metrics()
            info["result"] = {
                **metrics,
                "iterator_remaining": max(0, len(self._field_queue) - self._field_cursor),
            }
            reward = 0.0

        elif action_type == 7:  # next_form_batch
            k = int(params.get("k", 5))
            form_filter = params.get("form")
            info["result"] = self._next_form_batch(k, form_filter)
            reward = 0.1 if info["result"]["batch_size"] > 0 else 0.0

        elif action_type == 8:  # submit_form_batch
            values = params.get("values") or {}
            form_filter = params.get("form")
            if not isinstance(values, dict):
                info["error"] = "values must be an object {field_id: value}"
                reward = -0.5
            else:
                info["result"] = self._submit_form_batch(values, form_filter)
                reward = sum(float(r.get("reward", 0.0)) for r in info["result"]["results"])

        elif action_type == 9:  # load_parsed_form (additive)
            parsed_form = params.get("parsed_form")
            if not isinstance(parsed_form, dict):
                info["error"] = "parsed_form object required"
                reward = -0.5
            else:
                try:
                    info["result"] = self.load_parsed_form(parsed_form)
                    reward = 0.5
                except ValueError as exc:
                    info["error"] = str(exc)
                    reward = -0.5

        elif action_type == 10:  # export_filled_pdf
            form_filter = (params.get("form") or params.get("form_type") or "").strip().lower()
            try:
                info["result"] = self.export_filled_pdf(form_filter or None)
                reward = 0.5 if info["result"].get("exports") else -0.1
            except Exception as exc:
                info["error"] = str(exc)
                reward = -0.5

        elif action_type == 3:  # answer_question
            question = params.get("question", "")
            answer = params.get("answer", "")
            self.answers[question] = answer
            reward = 0.5  # Always reward attempt; human evaluates quality
            info["result"] = {"question": question, "answer": answer}

        self.score = self._compute_score()
        total_slots = self._total_field_slots()
        total_filled = sum(len(v) for v in self.filled.values())
        if terminated is False and total_slots > 0 and total_filled >= total_slots:
            terminated = True

        obs = {
            "task": self.current_task,
            "score": np.array([self.score], dtype=np.float32),
            "total_fields": np.array([total_slots], dtype=np.int32),
            "filled_fields": np.array([total_filled], dtype=np.int32),
        }
        return obs, reward, terminated, False, info

    def _score_fill(self, form: str, field: str, value: str) -> dict[str, Any]:
        if self._has_gt_scoring():
            gt = self.ground_truth.get(form, {}).get(field, "")
            match = value.lower().strip() == gt.lower().strip()
            payload: dict[str, Any] = {
                "match": match,
                "field": field,
                "value": value,
                "reward": 2.0 if match else -0.5,
            }
            if not match and not self.hide_ground_truth_in_responses:
                payload["ground_truth"] = gt
            return payload
        filled_ok = bool(value.strip())
        return {
            "match": None,
            "field": field,
            "value": value,
            "reward": 0.5 if filled_ok else -0.1,
        }

    def _compute_metrics(self) -> dict[str, Any]:
        """Precision / recall / F1 over filled fields vs. GT (when available).

        Play mode (no GT) returns a simple progress metric instead, so the
        per-form numbers in the UI still tick along as the agent submits.
        """
        per_form: dict[str, dict[str, Any]] = {}
        fields_submitted = sum(len(v) for v in self.filled.values())
        fields_filled = sum(
            1 for fl in self.filled.values() for v in fl.values() if str(v).strip()
        )
        fields_total = self._total_field_slots()

        if self._has_gt_scoring():
            tp = fp = fn = 0
            for form in self._form_types():
                gt = self.ground_truth.get(form, {})
                fl = self.filled.get(form, {})
                scorable = set(gt.keys())
                f_tp = sum(
                    1 for k, v in fl.items()
                    if k in scorable
                    and v.strip()
                    and v.lower().strip() == gt.get(k, "").lower().strip()
                )
                f_filled = sum(1 for k, v in fl.items() if k in scorable and v.strip())
                f_relevant = len(gt)
                f_fp = f_filled - f_tp
                f_fn = f_relevant - f_tp
                tp += f_tp
                fp += f_fp
                fn += f_fn
                p_p = f_tp / f_filled if f_filled else 0.0
                p_r = f_tp / f_relevant if f_relevant else 0.0
                per_form[form] = {
                    "filled": f_filled,
                    "relevant": f_relevant,
                    "correct": f_tp,
                    "precision": round(p_p, 4),
                    "recall": round(p_r, 4),
                }
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            return {
                "mode": "gt",
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "score_percent": round(f1 * 100, 1),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "fields_filled": fields_filled,
                "fields_total": fields_total,
                "per_form": per_form,
            }

        for form in self._form_types():
            fl = self.filled.get(form, {})
            field_count = len(self.forms[form].get("fields", []) or [])
            f_filled = sum(1 for v in fl.values() if v.strip())
            per_form[form] = {
                "filled": f_filled,
                "relevant": field_count,
                "correct": None,
                "precision": None,
                "recall": None,
            }
        pct = round(fields_filled / fields_total * 100, 1) if fields_total else 0.0
        return {
            "mode": "play",
            "precision": None,
            "recall": None,
            "f1": None,
            "score_percent": pct,
            "fields_filled": fields_filled,
            "fields_submitted": fields_submitted,
            "fields_total": fields_total,
            "per_form": per_form,
        }

    def _compute_score(self) -> float:
        return float(self._compute_metrics().get("score_percent") or 0.0)

    def export_form_state(self) -> dict[str, Any]:
        """Snapshot parsed schemas + filled values for case re-upload / remote reset."""
        if not self.forms:
            return {}
        return {
            "parsed_forms": list(self.forms.values()),
            "filled": {ft: dict(vals) for ft, vals in self.filled.items()},
        }

    def export_filled_pdf(self, form_type: str | None = None) -> dict[str, Any]:
        """Write filled AcroForm PDF(s) as ``form_filled_<form_type>.pdf`` under forms/."""
        from app.services.form_loader import write_filled_pdf

        targets = [form_type] if form_type else self._form_types()
        if not targets:
            return {"error": "No forms loaded", "exports": []}

        forms_dir = self._resolve_forms_dir()
        forms_dir.mkdir(parents=True, exist_ok=True)
        exports: list[dict[str, Any]] = []
        errors: list[str] = []

        for ft in targets:
            if ft not in self.forms:
                errors.append(f"Unknown form: {ft}")
                continue
            src = self._resolve_form_pdf(ft)
            if src is None:
                errors.append(f"No template PDF found for form_type={ft!r}")
                continue
            out = forms_dir / f"form_filled_{ft}.pdf"
            fields_meta = self.forms[ft].get("fields", []) or []
            filled = self.filled.get(ft, {})
            try:
                meta = write_filled_pdf(src, fields_meta, filled, out)
                exports.append({
                    "form_type": ft,
                    "path": str(out),
                    "workspace_path": f"forms/{out.name}",
                    **meta,
                })
            except Exception as exc:
                errors.append(f"{ft}: {exc}")

        result: dict[str, Any] = {"exports": exports, "forms_dir": str(forms_dir)}
        if errors:
            result["errors"] = errors
        if not exports and errors:
            result["error"] = "; ".join(errors)
        return result

    @staticmethod
    def _field_export_entry(meta: dict[str, Any], value: str) -> dict[str, Any]:
        """One field row for ``filled.json`` / GT JSON (schema + value)."""
        return {
            "id": meta.get("id") or "",
            "raw_name": meta.get("raw_name") or "",
            "label": meta.get("label") or meta.get("id") or "",
            "field_type": meta.get("field_type") or "text",
            "page": int(meta.get("page") if meta.get("page") is not None else -1),
            "context": str(meta.get("context") or ""),
            "value": value,
        }

    def export_filled(self) -> dict[str, Any]:
        """Snapshot of filled values per form, with schema metadata for P/R."""
        out: dict[str, Any] = {}
        for form_type in self._form_types():
            fields_meta = self.forms[form_type].get("fields", []) or []
            meta_by_id = {
                str(f.get("id") or ""): f for f in fields_meta if f.get("id")
            }
            entries = []
            for fid, val in self.filled.get(form_type, {}).items():
                meta = meta_by_id.get(fid, {"id": fid, "label": fid})
                entries.append(self._field_export_entry(meta, val))
            out[form_type] = {
                "title": self.forms[form_type].get("title", form_type),
                "field_count": len(fields_meta),
                "filled_count": len(entries),
                "fields": entries,
            }
        return out

    def render(self):
        if self.render_mode is None:
            return None

        lines = []
        lines.append("=" * 60)
        lines.append(f"CMS — Case Management  |  Score: {self.score}%")
        lines.append("=" * 60)

        if self.index:
            self.index.ensure_catalog()
            lines.append(f"\nDocuments in case: {len(self.index.docs)} (text loaded on read/search)")

        if not self._form_types():
            lines.append("\nForms: (none loaded — upload a form .pdf to begin)")
            return "\n".join(lines)

        lines.append("\nForms:")
        max_filled_lines = 12
        for form in self._form_types():
            fields_meta = self.forms[form].get("fields", []) or []
            field_ids = [str(f.get("id") or "") for f in fields_meta if f.get("id")]
            filled = self.filled.get(form, {})
            title = self.forms[form].get("title", form)
            filled_ids = [fid for fid in field_ids if str(filled.get(fid, "")).strip()]
            lines.append(
                f"\n  {title} — {len(filled_ids)}/{len(field_ids)} filled"
            )
            if not filled_ids:
                lines.append("    (no values yet)")
                continue
            for fid in filled_ids[:max_filled_lines]:
                val = str(filled.get(fid, ""))
                if self._has_gt_scoring():
                    gt = self.ground_truth.get(form, {}).get(fid, "")
                    status = "✓" if val.lower().strip() == gt.lower().strip() else "✗"
                else:
                    status = "✓"
                lines.append(f"    [{status}] {fid}: {val[:72]}")
            remaining = len(filled_ids) - max_filled_lines
            if remaining > 0:
                lines.append(f"    … and {remaining} more filled field(s)")

        if self.answers:
            lines.append(f"\nQuestions answered: {len(self.answers)}")
            last_q, last_a = next(reversed(self.answers.items()))
            lines.append(f"  Latest: {last_q[:80]}")
            lines.append(f"  → {str(last_a)[:160]}")

        return "\n".join(lines)
