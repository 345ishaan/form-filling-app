"""Upload and ingest CMS case ZIPs into session workspace + env sandbox."""

from __future__ import annotations

from app.config.runtime import is_modal_runtime
from app.services.sandbox_client import SandboxClient
from app.services.workspace import get_case_dir, get_session_workspace, save_case_zip_bytes


def env_case_dir(session_id: str) -> str:
    """Path to case files as seen inside a Modal env sandbox (shared workspace volume)."""
    return f"/vol/workspaces/{session_id}/case"


def reset_options_for_session(session_id: str) -> dict:
    """Reset options pointing the env at uploaded case files."""
    if is_modal_runtime():
        case_dir = env_case_dir(session_id)
    else:
        case_dir = str(get_case_dir(session_id))
    return {"case_dir": case_dir, "hide_gt": True}


async def reset_options_preserving_forms(
    session_id: str,
    sandbox_id: str,
    client: SandboxClient,
) -> dict:
    """Case reset options that keep parsed forms and filled progress."""
    opts = reset_options_for_session(session_id)
    try:
        state = await client.get_form_state(sandbox_id)
    except Exception:
        state = {}
    if state.get("parsed_forms"):
        opts["parsed_forms"] = state["parsed_forms"]
        if state.get("filled"):
            opts["filled"] = state["filled"]
    return opts


async def ingest_case_into_sandbox(
    session_id: str,
    sandbox_id: str,
    client: SandboxClient,
    *,
    zip_bytes: bytes | None = None,
) -> dict:
    """
    Save case zip to the session workspace, then ingest into the env sandbox.

    Remote Modal sandboxes receive the zip via ``/ingest_case`` so extraction and
    ``DocumentIndex`` cataloguing run in the same container (avoids cross-container
    volume reload). The API worker also keeps a copy under ``case/`` for the agent
    SDK cwd and ``parse_form``.
    """
    if zip_bytes is not None:
        meta = await save_case_zip_bytes(session_id, zip_bytes)
    else:
        case_dir = get_case_dir(session_id)
        if not case_dir.is_dir() or not any(case_dir.rglob("*")):
            raise ValueError("No case files found; upload a .zip first")
        meta = {"case_dir": str(case_dir), "file_count": sum(1 for _ in case_dir.rglob("*") if _.is_file())}

    if sandbox_id in client._local_envs:
        reset_opts = await reset_options_preserving_forms(session_id, sandbox_id, client)
        result = await client.reset(sandbox_id, reset_opts)
        return {**meta, "render": result.get("render"), "info": result.get("info", {})}

    zip_path = get_session_workspace(session_id, "cms") / "case.zip"
    if not zip_path.is_file():
        raise ValueError("case.zip missing from workspace")
    result = await client.ingest_case_zip(
        sandbox_id,
        zip_path,
        session_id=session_id,
    )
    info = result.get("info") or {}
    doc_count = info.get("document_count", 0)
    file_count = meta.get("file_count", 0)
    if file_count and not doc_count:
        raise RuntimeError(
            f"Case zip had {file_count} file(s) but env sandbox indexed 0 documents "
            f"(case_dir={info.get('case_dir') or result.get('case_dir')!r})."
        )
    return {**meta, "render": result.get("render"), "info": info}
