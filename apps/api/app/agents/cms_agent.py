"""
CMS Agent — Case management / document processing (immigration forms).

Env tools via in-process MCP; Bash/Read/Write in session workspace (play mode).
"""

import json

from app.config.settings import get_agent_model
from app.agents.tool_specs import cms_tool_specs
from app.agents.tooling import build_allowed_tools, should_include_sdk_file_tools
from app.services.sandbox_client import SandboxClient
from app.services.workspace import get_session_workspace


def build_system_prompt(agent_mode: str = "play") -> str:
    from app.agents.prompts_loader import load_system_prompt

    return load_system_prompt("cms")


def build_agent_options(
    sandbox_id: str,
    sandbox_client: SandboxClient,
    session_id: str = "",
    agent_mode: str = "play",
    eval_mode: bool = False,
    model: str | None = None,
) -> dict:
    include_sdk = should_include_sdk_file_tools("cms", agent_mode, eval_mode)
    return {
        "system_prompt": build_system_prompt(agent_mode),
        "model": model or get_agent_model(),
        "allowed_tools": build_allowed_tools(
            "cms", agent_mode=agent_mode, include_sdk_file_tools=include_sdk
        ),
        "tool_specs": cms_tool_specs(),
        "cwd": str(get_session_workspace(session_id or sandbox_id, "cms")),
        "permission_mode": "acceptEdits",
        "eval_mode": eval_mode,
    }


async def execute_tool(
    tool_name: str,
    tool_input: dict,
    sandbox_id: str,
    client: SandboxClient,
    eval_mode: bool = False,
    session_id: str = "",
) -> str:
    """Execute a CMS env tool call against the sandbox environment.

    ``session_id`` is required for tools that touch the session workspace
    (e.g. ``parse_form`` resolves PDFs relative to ``workspaces/{sid}/forms/``).
    """
    # GT never lives in sandboxes; do not expose answers over HTTP.
    hide_gt = True

    if tool_name == "parse_form":
        return await _execute_parse_form(tool_input, sandbox_id, client, session_id)

    if tool_name == "list_uploaded_forms":
        return await _execute_list_uploaded_forms(session_id, sandbox_id, client)

    if tool_name == "search_documents":
        result = await client.step(sandbox_id, {
            "action_type": 0,
            "params": json.dumps({"query": tool_input["query"]}),
        })
        info = result.get("info", {})
        results = info.get("result", [])
        if not results:
            return "No matching documents found."
        output = []
        for r in results[:5]:
            output.append(f"\n--- {r['path']} (score: {r.get('score', 0)}) ---")
            for snippet in r.get("snippets", [])[:2]:
                output.append(snippet)
        return "\n".join(output)

    if tool_name == "read_document":
        result = await client.step(sandbox_id, {
            "action_type": 1,
            "params": json.dumps({"path": tool_input["path"]}),
        })
        info = result.get("info", {})
        if info.get("error"):
            return f"Error: {info['error']}"
        doc = info.get("result", {})
        text = doc.get("text", "")
        path = doc.get("path", tool_input["path"])
        return f"--- {path} ---\n{text[:10000]}"

    if tool_name == "list_documents":
        category = tool_input.get("category")
        result = await client.step(sandbox_id, {
            "action_type": 0,
            "params": json.dumps({"query": "__list__", "category": category}),
        })
        info = result.get("info") or {}
        return json.dumps(info.get("result", {}), indent=2)

    if tool_name == "list_categories":
        info = await client.get_info(sandbox_id)
        return json.dumps({"categories": info.get("categories", [])}, indent=2)

    if tool_name == "answer_question":
        result = await client.step(sandbox_id, {
            "action_type": 3,
            "params": json.dumps({
                "question": tool_input["question"],
                "answer": tool_input["answer"],
            }),
        })
        return json.dumps(result.get("info", {}).get("result", {}))

    if tool_name == "fill_form_field":
        result = await client.step(sandbox_id, {
            "action_type": 2,
            "params": json.dumps({
                "form": tool_input["form"],
                "field": tool_input["field"],
                "value": tool_input["value"],
            }),
        })
        return _format_fill_result(result, hide_gt)

    if tool_name == "next_form_field":
        params = {}
        if tool_input.get("form"):
            params["form"] = tool_input["form"]
        result = await client.step(sandbox_id, {
            "action_type": 4,
            "params": json.dumps(params),
        })
        return json.dumps(result.get("info", {}).get("result", {}), indent=2, default=str)

    if tool_name == "submit_form_field":
        result = await client.step(sandbox_id, {
            "action_type": 5,
            "params": json.dumps({
                "form": tool_input.get("form", ""),
                "field": tool_input.get("field", ""),
                "value": tool_input["value"],
            }),
        })
        return _format_fill_result(result, hide_gt)

    if tool_name == "get_form_progress":
        result = await client.step(sandbox_id, {
            "action_type": 6,
            "params": "{}",
        })
        return json.dumps(result.get("info", {}).get("result", {}), indent=2)

    if tool_name == "next_form_batch":
        params: dict = {"k": int(tool_input.get("k", 5))}
        if tool_input.get("form"):
            params["form"] = tool_input["form"]
        result = await client.step(sandbox_id, {
            "action_type": 7,
            "params": json.dumps(params),
        })
        info = result.get("info", {}) or {}
        if info.get("error"):
            return f"Error: {info['error']}"
        payload = info.get("result", {}) or {}
        return json.dumps(payload, indent=2, default=str)

    if tool_name == "submit_form_batch":
        values = tool_input.get("values") or {}
        if not isinstance(values, dict):
            return json.dumps({"error": "values must be an object {field_id: value}"})
        params = {"values": values}
        if tool_input.get("form"):
            params["form"] = tool_input["form"]
        result = await client.step(sandbox_id, {
            "action_type": 8,
            "params": json.dumps(params),
        })
        info = result.get("info", {}) or {}
        if info.get("error"):
            return f"Error: {info['error']}"
        payload = dict(info.get("result", {}) or {})
        if hide_gt:
            for r in payload.get("results", []) or []:
                r.pop("ground_truth", None)
        payload["reward"] = result.get("reward", 0)
        return json.dumps(payload, indent=2, default=str)

    if tool_name == "export_filled_pdf":
        params: dict = {}
        if tool_input.get("form"):
            params["form"] = tool_input["form"]
        result = await client.step(sandbox_id, {
            "action_type": 10,
            "params": json.dumps(params),
        })
        info = result.get("info", {}) or {}
        if info.get("error"):
            return json.dumps({"error": info["error"]})
        return json.dumps(info.get("result", {}) or {}, indent=2, default=str)

    if tool_name == "get_form_status":
        form = (tool_input.get("form") or "").lower()
        render = await client.render(sandbox_id)
        if not render:
            return f"No data for {form}"
        lines = render.split("\n")
        # The render emits "  <title> (X/Y fields filled):" then indented field rows.
        form_section: list[str] = []
        capturing = False
        for line in lines:
            stripped = line.strip().lower()
            if not stripped:
                if capturing:
                    form_section.append(line)
                continue
            if stripped.endswith("fields filled):") and (
                form in stripped or stripped.startswith(form)
            ):
                capturing = True
                form_section.append(line)
                continue
            if capturing and stripped.endswith("fields filled):"):
                break
            if capturing:
                form_section.append(line)
        return "\n".join(form_section).rstrip() if form_section else f"No data for {form}"

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _format_fill_result(result: dict, hide_gt: bool) -> str:
    info = result.get("info", {})
    if info.get("error"):
        return f"Error: {info['error']}"
    data = dict(info.get("result", {}))
    if hide_gt:
        data.pop("ground_truth", None)
    match = data.get("match")
    if match is True:
        data["status"] = "CORRECT"
    elif match is False:
        data["status"] = "INCORRECT"
    else:
        data["status"] = "FILLED" if data.get("value") else "EMPTY"
    data["reward"] = result.get("reward", 0)
    return json.dumps(data)


# ── Agent-invoked form parsing ────────────────────────────────────────────────


def _resolve_workspace_pdf(session_id: str, pdf_path: str) -> "Path | None":
    """Resolve a PDF path against the session workspace if relative."""
    from pathlib import Path

    if not pdf_path:
        return None
    p = Path(pdf_path)
    if not p.is_absolute():
        if not session_id:
            return None
        ws = get_session_workspace(session_id, "cms")
        # Accept ``forms/foo.pdf``, ``foo.pdf``, or even bare ``foo`` (we'll add .pdf)
        candidates = [ws / pdf_path, ws / "forms" / pdf_path]
        if not str(pdf_path).lower().endswith(".pdf"):
            candidates += [ws / f"{pdf_path}.pdf", ws / "forms" / f"{pdf_path}.pdf"]
        for cand in candidates:
            if cand.is_file():
                return cand
        return None
    return p if p.is_file() else None


async def _execute_parse_form(
    tool_input: dict,
    sandbox_id: str,
    client: SandboxClient,
    session_id: str,
) -> str:
    """Parse a workspace PDF and load parsed fields directly into the env."""
    from app.services.form_loader import parse_pdf_form

    pdf_path_in = str(tool_input.get("pdf_path") or "")
    pdf_path = _resolve_workspace_pdf(session_id, pdf_path_in)
    if pdf_path is None:
        return json.dumps({
            "error": f"PDF not found: {pdf_path_in!r}. Drop the file into the workspace "
                     "``forms/`` dir, then call list_uploaded_forms to see available paths.",
        })

    try:
        parsed = parse_pdf_form(pdf_path)
    except Exception as exc:
        return json.dumps({"error": f"Could not parse {pdf_path.name}: {exc}"})

    explicit_type = str(tool_input.get("form_type") or "").strip().lower()
    if explicit_type:
        parsed.form_type = explicit_type

    ws = get_session_workspace(session_id or sandbox_id, "cms")
    parsed_dict = parsed.to_dict()
    try:
        parsed_dict["pdf_path"] = str(pdf_path.relative_to(ws))
    except ValueError:
        parsed_dict["pdf_path"] = f"forms/{pdf_path.name}"

    # Tell the env to additively load it (preserves filled progress).
    step_result = await client.step(sandbox_id, {
        "action_type": 9,
        "params": json.dumps({"parsed_form": parsed_dict}),
    })
    info = step_result.get("info") or {}
    if info.get("error"):
        return json.dumps({"error": f"Env failed to load schema: {info['error']}"})

    payload = info.get("result") or {}
    payload["preview"] = [
        {
            "id": f.id,
            "label": f.label,
            "field_type": f.field_type,
            "page": f.page,
            "context": (f.context[:120] + ("…" if len(f.context) > 120 else "")) if f.context else "",
        }
        for f in parsed.fields[:5]
    ]
    return json.dumps(payload, indent=2, default=str)


async def _execute_list_uploaded_forms(
    session_id: str,
    sandbox_id: str,
    client: SandboxClient,
) -> str:
    """List PDF files in the session workspace ``forms/`` directory."""
    from app.services.form_upload import enrich_forms_with_loaded_state
    from app.services.workspace import get_forms_dir

    if not session_id:
        return json.dumps({"error": "No session_id"})
    forms_dir = get_forms_dir(session_id)
    pdfs: list[dict] = []
    for pdf in sorted(forms_dir.glob("*.pdf")):
        entry = {
            "filename": pdf.name,
            "path": f"forms/{pdf.name}",  # workspace-relative
            "size_kb": round(pdf.stat().st_size / 1024, 1),
            "parsed": False,
        }
        pdfs.append(entry)

    forms_loaded: list[dict] = []
    if sandbox_id:
        try:
            info = await client.get_info(sandbox_id)
            forms_loaded = info.get("forms_loaded") or []
        except Exception:
            pass
    enrich_forms_with_loaded_state(pdfs, forms_loaded)

    return json.dumps({
        "forms_dir": str(forms_dir),
        "uploads": pdfs,
        "hint": (
            "Call parse_form({pdf_path: 'forms/<filename>'}) to parse a PDF that "
            "is not yet parsed, then use next_form_batch."
        ),
    }, indent=2)
