"""Tool JSON schemas for env MCP servers (shared by Games / CMS agents)."""

from __future__ import annotations

# Claude SDK built-in workspace tools (when enabled for an env/mode).
SDK_TOOL_SPECS: list[dict] = [
    {"name": "Read", "description": "Read a file from the agent workspace."},
    {"name": "Write", "description": "Create or overwrite a file in the workspace."},
    {"name": "Edit", "description": "Edit a file by search/replace in the workspace."},
    {"name": "Glob", "description": "Find files by glob pattern in the workspace."},
    {"name": "Grep", "description": "Search file contents by pattern in the workspace."},
    {"name": "Bash", "description": "Run a shell command in the workspace sandbox."},
]


def games_tool_specs(env_type: str, agent_mode: str = "play") -> list[dict]:
    """Env MCP tool specs for games.

    Single-mode design: the agent always has the action tool
    (pick_cards / move_fragment) plus inspection tools. SDK file tools
    (Bash/Read/Write/Glob/Grep) are added separately via build_allowed_tools.
    """
    specs: list[dict] = []
    if env_type == "cards":
        specs.extend([_pick_cards_spec(), _view_grid_spec("4×4 card"), _get_score_spec()])
    elif env_type == "bonza":
        specs.extend([
            _move_fragment_spec(),
            _view_grid_spec("letter"),
            _get_status_spec(),
            _get_valid_words_spec(),
        ])
    return specs


def cms_tool_specs() -> list[dict]:
    return [
        {
            "name": "search_documents",
            "description": "Search parsed case documents. Returns ranked excerpts.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "read_document",
            "description": "Read text of a document by relative path.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "list_documents",
            "description": "List documents in the case, optionally by category.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Optional category filter"},
                },
            },
        },
        {
            "name": "list_categories",
            "description": "List evidence categories in the case.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "answer_question",
            "description": "Record an answer to a case Q&A question.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["question", "answer"],
            },
        },
        {
            "name": "fill_form_field",
            "description": "Fill a single field on a loaded form. Prefer next_form_batch / submit_form_batch when iterating.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "form": {"type": "string"},
                    "field": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["form", "field", "value"],
            },
        },
        {
            "name": "next_form_field",
            "description": "Get the next form field to fill with label, type, options, and context snippets.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "form": {"type": "string", "description": "Optional: limit to one form"},
                },
            },
        },
        {
            "name": "submit_form_field",
            "description": "Submit value for the current (or specified) field from the iterator.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "form": {"type": "string"},
                    "field": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["value"],
            },
        },
        {
            "name": "get_form_progress",
            "description": "Progress across all forms: filled count + precision/recall when GT loaded.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_form_status",
            "description": "Status for one form (text render of filled vs empty fields).",
            "input_schema": {
                "type": "object",
                "properties": {"form": {"type": "string"}},
                "required": ["form"],
            },
        },
        {
            "name": "list_uploaded_forms",
            "description": (
                "List PDFs that the user has dropped into the session ``forms/`` "
                "workspace. Call this first when the user mentions a form."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "parse_form",
            "description": (
                "Parse a fillable PDF in the session workspace into a normalized "
                "field schema (stable ids, labels, field types, page context) and "
                "register it with the env. Fields are then available via "
                "next_form_batch / submit_form_batch. Idempotent — re-parsing the "
                "same form_type replaces its schema and preserves filled progress."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "Workspace-relative (e.g. ``forms/form_i140.pdf``) or absolute path",
                    },
                    "form_type": {
                        "type": "string",
                        "description": "Optional form_type override (default: inferred from filename — i140 / g28 / i907 / custom)",
                    },
                },
                "required": ["pdf_path"],
            },
        },
        {
            "name": "next_form_batch",
            "description": (
                "Return the next K un-submitted fields with id, label, type, page, "
                "nearby PDF context, and case-document snippets. Does NOT advance the "
                "cursor; pair with submit_form_batch. Run parse_form first if no "
                "form is loaded yet."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                    "form": {"type": "string", "description": "Optional: limit to one form_type"},
                },
            },
        },
        {
            "name": "submit_form_batch",
            "description": (
                "Submit values for multiple fields at once. ``values`` is an object "
                "keyed by field id (or qualified ``form_type/field_id``). Cursor "
                "advances past the longest contiguous prefix of submitted fields."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "{field_id: value} (use form_type/field_id to disambiguate)",
                    },
                    "form": {"type": "string", "description": "Optional: default form_type"},
                },
                "required": ["values"],
            },
        },
        {
            "name": "export_filled_pdf",
            "description": (
                "Write filled field values into an AcroForm PDF and save it under the "
                "session ``forms/`` directory as ``form_filled_<form_type>.pdf``. "
                "Call after filling is complete."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "form": {
                        "type": "string",
                        "description": "Optional form_type (default: all loaded forms)",
                    },
                },
            },
        },
    ]


def _pick_cards_spec() -> dict:
    return {
        "name": "pick_cards",
        "description": "Pick 3 cards (positions 0-15) that sum to 15.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pos1": {"type": "integer"},
                "pos2": {"type": "integer"},
                "pos3": {"type": "integer"},
            },
            "required": ["pos1", "pos2", "pos3"],
        },
    }


def _view_grid_spec(kind: str) -> dict:
    return {
        "name": "view_grid",
        "description": f"Show current {kind} grid (ASCII).",
        "input_schema": {"type": "object", "properties": {}},
    }


def _get_score_spec() -> dict:
    return {
        "name": "get_score",
        "description": "Current score and turn info.",
        "input_schema": {"type": "object", "properties": {}},
    }


def _move_fragment_spec() -> dict:
    return {
        "name": "move_fragment",
        "description": (
            "Slide the fragment containing (row, col) in direction (0-7) "
            "for magnitude steps (1-8)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {"type": "integer", "description": "Row 0-8 of a letter in the fragment"},
                "col": {"type": "integer", "description": "Col 0-7 of a letter in the fragment"},
                "direction": {
                    "type": "integer",
                    "description": "0=UP,1=DOWN,2=LEFT,3=RIGHT,4=DIAG_UP_LEFT,5=DIAG_DOWN_RIGHT,6=DIAG_DOWN_LEFT,7=DIAG_UP_RIGHT",
                },
                "magnitude": {"type": "integer", "description": "Steps to slide (1-8)"},
            },
            "required": ["row", "col", "direction", "magnitude"],
        },
    }


def _get_status_spec() -> dict:
    return {
        "name": "get_status",
        "description": "Attempts, fragment count, fragment labels, termination flag.",
        "input_schema": {"type": "object", "properties": {}},
    }


def _get_valid_words_spec() -> dict:
    return {
        "name": "get_valid_words",
        "description": (
            "Scan the current board for horizontal and vertical letter runs; "
            "return valid English words (NLTK) and invalid clusters."
        ),
        "input_schema": {"type": "object", "properties": {}},
    }
