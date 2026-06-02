"""
WebSocket router for stateful environment sessions.

Endpoint: /ws/{sim_type}[?session_id=...&persona=...&agent_mode=...]

sim_type: cms

Each WebSocket connection is a persistent session. The server maintains:
- A sandbox running the Gymnasium environment (local or Modal)
- A Claude SDK client for multi-turn agent interaction
- Game state (grid, score, turn) across all messages in the session
- Feedback log for the session

Session resumption: if the frontend passes a `session_id` query param,
the server reconnects to the existing sandbox and resumes the agent.

Client → Server messages:
  {"type": "message", "content": "..."}          — user input to agent
  {"type": "feedback", "rating": 4, "comment": "..."}  — rate agent response
  {"type": "set_persona", "persona": "meticulous"}     — switch persona
  {"type": "ping"}                                     — keep-alive
  {"type": "interrupt"}                                — cancel current agent run

Server → Client messages:
  {"type": "init", "session_id": "...", "env_type": "...", "render": "..."}
  {"type": "state_update", "render": "..."}
  {"type": "text", "text": "..."}
  {"type": "thinking", "thinking": "..."}
  {"type": "tool_use", "name": "...", "id": "...", "input": {...}}
  {"type": "tool_result", "tool_use_id": "...", "content": "..."}
  {"type": "complete", "session_id": "...", "render": "..."}
  {"type": "error", "error": "..."}
  {"type": "pong"}
  {"type": "heartbeat", "timestamp": ...}
"""

import asyncio
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Query

from app.services.case_upload import ingest_case_into_sandbox
from app.services.form_upload import (
    list_forms_for_session,
    remove_form_from_session,
    save_form_pdf,
)
from app.services.sandbox_client import SandboxClient
from app.services.workspace import get_session_workspace
import json
import time

from app.config.runtime import get_sandbox_mode
from app.services.transcript_service import (
    append_event,
    events_path,
    export_harbor_candidate,
    list_transcripts,
    load_transcript,
    save_session_transcript,
    transcript_path,
    update_transcript,
)

router = APIRouter(prefix="/ws", tags=["ws"])

# Global session registry — survives across reconnects for same session_id
_sessions: dict[str, dict] = {}

_VALID_SIM_TYPES = frozenset({"cms"})


def _session_matches_config(session: dict, sim_type: str, persona: str) -> bool:
    return session.get("env_type") == sim_type


async def _drop_session(session_id: str, session: dict) -> None:
    """Terminate sandbox and drop server-side session (parameter change / abandon)."""
    try:
        await session["client"].terminate(session["sandbox_id"])
    except Exception:
        pass
    sdk = session.get("sdk_client")
    if sdk is not None:
        try:
            await sdk.disconnect()
        except Exception:
            pass
    _sessions.pop(session_id, None)


def _normalize_sim_type(sim_type: str) -> str:
    """Map URL slug to env type."""
    key = sim_type.strip().lower().replace("-", "_")
    return "cms" if key == "cms" else key


async def _run_agent_stream(
    ws: WebSocket,
    session: dict,
    user_message: str | None = None,
):
    """Run Claude agent; stream events to the client as they are produced."""
    from app.config.runtime import agent_runs_locally

    session["agent_running"] = True
    try:
        if agent_runs_locally():
            from app.agents.runner import stream_agent_turn

            async for event in stream_agent_turn(session, user_message):
                await ws.send_json(event)
        else:
            from app.services.modal_agent import run_agent_turn_remote

            result = await run_agent_turn_remote(session, user_message)
            for event in result.get("events", []):
                await ws.send_json(event)
    except Exception as e:
        await ws.send_json({
            "type": "error",
            "error": str(e),
            "error_type": "agent_error",
            "recoverable": True,
            "hint": "Ensure modal profile is active and run: cd apps/api && modal deploy modal_app.py",
        })
    finally:
        session["agent_running"] = False
        await _process_message_queue(ws, session)
        save_session_transcript(session, sandbox_mode=get_sandbox_mode())


async def _process_message_queue(ws: WebSocket, session: dict):
    """Process user messages queued during an active agent turn."""
    queue = session.pop("_message_queue", [])
    for item in queue:
        if item.get("type") == "message" and item.get("content", "").strip():
            await _run_agent_stream(ws, session, user_message=item["content"])


@router.websocket("/{sim_type}")
async def environment_ws(
    ws: WebSocket,
    sim_type: str,
    session_id: str | None = Query(None, description="Resume existing session"),
    persona: str = Query("none", description="Agent persona (meticulous|careless|none)"),
    agent_mode: str = Query("play", description="Deprecated; ignored"),
    eval_mode: bool = Query(False, description="Eval: env tools only, hide GT hints"),
):
    """WebSocket endpoint for stateful environment interaction.

    Connects to (or resumes) a session with the selected environment type.
    The WebSocket stays open for the full session — game state, agent
    context, and feedback are preserved across all messages.
    """
    sim_type = _normalize_sim_type(sim_type)

    if sim_type not in _VALID_SIM_TYPES:
        await ws.accept()
        await ws.send_json({
            "type": "error",
            "error": f"Unknown sim_type: {sim_type!r} (expected cms)",
            "hint": "Restart modal serve after pulling latest API code: cd apps/api && modal serve modal_app.py",
        })
        await ws.close()
        return

    await ws.accept()
    await ws.send_json({"type": "status", "message": "Starting environment…"})

    session: dict | None = None
    heartbeat_task: asyncio.Task | None = None
    heartbeat_active = True

    async def heartbeat():
        while heartbeat_active:
            await asyncio.sleep(25)
            if heartbeat_active:
                try:
                    await ws.send_json({
                        "type": "heartbeat",
                        "timestamp": asyncio.get_event_loop().time(),
                    })
                except Exception:
                    break

    heartbeat_task = asyncio.create_task(heartbeat())

    # ── Session resolution ──────────────────────────────────────────────
    existing = _sessions.get(session_id) if session_id else None
    can_resume = (
        existing is not None
        and _session_matches_config(existing, sim_type, persona)
    )

    if can_resume:
        # Resume existing session (same game parameters)
        session = existing
        session["persona"] = persona or session.get("persona", "default")
        session["eval_mode"] = eval_mode
        sandbox_id = session["sandbox_id"]
        client = session["client"]
        is_new = False
    else:
        if existing and session_id:
            await _drop_session(session_id, existing)
        # Create new session (fresh sandbox + agent context)
        session_id = uuid.uuid4().hex[:12]
        client = SandboxClient(mode=get_sandbox_mode())

        sandbox_id = await client.create(
            sim_type,
            session_id=session_id,
        )
        await ws.send_json({"type": "status", "message": "Sandbox ready…"})

        session = {
            "session_id": session_id,
            "env_type": sim_type,
            "sandbox_id": sandbox_id,
            "client": client,
            "persona": persona,
            "agent_mode": "play",
            "eval_mode": eval_mode,
            "sdk_client": None,
            "sdk_connected": False,
            "claude_session_id": None,
            "feedback_log": [],
            "agent_running": False,
            "interrupt_requested": False,
            "_message_queue": [],
            "transcript": [],
            "created_at": time.time(),
            "harbor": {"status": "none", "notes": "", "tags": []},
        }
        get_session_workspace(session_id, sim_type)
        _sessions[session_id] = session
        is_new = True

    try:
        # Send initial state
        reset_options: dict | None = None

        if is_new:
            initial = await client.reset(sandbox_id, reset_options)
            render = initial.get("render", "")
        else:
            render = await client.render(sandbox_id) or ""

        if render and not session.get("initial_render"):
            session["initial_render"] = render

        await ws.send_json({
            "type": "init",
            "session_id": session_id,
            "env_type": sim_type,
            "render": render,
            "resumed": not is_new,
        })

        # ── Message loop ────────────────────────────────────────────────
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "ping":
                await ws.send_json({"type": "pong"})

            elif msg_type == "message":
                content = data.get("content", "")
                if not content.strip():
                    continue

                # If agent is running, queue the message instead of blocking
                if session.get("agent_running"):
                    session.setdefault("_message_queue", []).append(data)
                    await ws.send_json({
                        "type": "text",
                        "text": "(Agent is working — message queued)",
                    })
                    continue

                # Run agent (user message already shown in the client UI)
                await ws.send_json({"type": "status", "message": "Agent working…"})
                agent_task = asyncio.create_task(
                    _run_agent_stream(ws, session, user_message=content)
                )

                # Wait for agent to finish OR an interrupt
                while not agent_task.done():
                    # Check for additional messages (interrupts) while agent runs
                    try:
                        next_data = await asyncio.wait_for(
                            ws.receive_json(), timeout=0.5
                        )
                        if next_data.get("type") == "interrupt":
                            session["interrupt_requested"] = True
                            await ws.send_json({"type": "text", "text": "⚠ Stopping..."})
                        elif next_data.get("type") == "ping":
                            await ws.send_json({"type": "pong"})
                        elif next_data.get("type") == "message":
                            # Queue the message; frontend can resend after agent done
                            session.setdefault("_message_queue", []).append(next_data)
                            await ws.send_json({
                                "type": "text",
                                "text": "(Message received — agent will respond shortly)",
                            })
                        else:
                            session.setdefault("_message_queue", []).append(next_data)
                    except asyncio.TimeoutError:
                        pass  # No message, continue waiting for agent

                # Agent finished — drain queued messages
                await _process_message_queue(ws, session)

            elif msg_type == "feedback":
                fb = {
                    "rating": data.get("rating"),
                    "comment": data.get("comment", ""),
                    "timestamp": time.time(),
                }
                session["feedback_log"].append(fb)
                append_event(session_id, {"type": "feedback", **fb})
                save_session_transcript(session, sandbox_mode=get_sandbox_mode())
                await ws.send_json({
                    "type": "text",
                    "text": f"✓ Feedback recorded: {data.get('rating')}/5",
                })

            elif msg_type == "interrupt":
                if session.get("agent_running"):
                    session["interrupt_requested"] = True
                else:
                    await ws.send_json({"type": "text", "text": "No agent running"})

            elif msg_type == "set_persona":
                session["persona"] = data.get("persona", "default")
                await ws.send_json({
                    "type": "text",
                    "text": f"Persona: {session['persona']}",
                })

    except WebSocketDisconnect:
        if session:
            save_session_transcript(session, sandbox_mode=get_sandbox_mode())
    except Exception as e:
        try:
            await ws.send_json({
                "type": "error",
                "error": str(e) or repr(e),
                "error_type": type(e).__name__,
                "recoverable": True,
            })
        except Exception:
            pass
    finally:
        heartbeat_active = False
        if heartbeat_task is not None:
            heartbeat_task.cancel()
        if session:
            try:
                save_session_transcript(session, sandbox_mode=get_sandbox_mode())
            except Exception:
                pass
        # Do NOT terminate sandbox — session persists for resume
        # Sandbox cleanup happens via idle timeout or explicit /close


# ── CMS case upload ───────────────────────────────────────────────────────────


@router.post("/sessions/{session_id}/case")
async def upload_case_zip(session_id: str, file: UploadFile = File(...)):
    """
    Upload case_sample.zip for a CMS session.

    Extracts into the session workspace and re-points the env sandbox at the case
    (shared Modal volume, or streamed to the sandbox when API runs locally).
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="No active session; connect via WebSocket first")
    if session["env_type"] != "cms":
        raise HTTPException(status_code=400, detail="Case upload is only supported for CMS sessions")

    name = (file.filename or "").lower()
    if not name.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Expected a .zip file")

    data = await file.read()
    try:
        result = await ingest_case_into_sandbox(
            session_id,
            session["sandbox_id"],
            session["client"],
            zip_bytes=data,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    render = result.get("render") or ""
    session["case_uploaded"] = True
    session["case_file_count"] = result.get("file_count", 0)
    return {
        "status": "ok",
        "session_id": session_id,
        "file_count": result.get("file_count", 0),
        "case_dir": result.get("case_dir"),
        "render": render,
        "info": result.get("info", {}),
    }


# ── CMS form upload ──────────────────────────────────────────────────────────


def _require_cms_session(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404, detail="No active session; connect via WebSocket first"
        )
    if session["env_type"] != "cms":
        raise HTTPException(status_code=400, detail="This endpoint is CMS-only")
    return session


@router.post("/sessions/{session_id}/form")
async def upload_form_pdf(
    session_id: str,
    file: UploadFile = File(...),
    replace_existing: bool = Query(
        False,
        description="Drop previously-uploaded forms before adding this one.",
    ),
):
    """
    Upload a fillable USCIS form PDF into the session workspace.

    This endpoint only persists the file. The agent itself parses it via the
    ``parse_form`` MCP tool, which lets it handle one or many forms within
    a single session without API-side coupling.
    """
    session = _require_cms_session(session_id)

    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Expected a .pdf file")

    data = await file.read()
    try:
        result = save_form_pdf(
            session_id,
            pdf_bytes=data,
            filename=file.filename or "form.pdf",
            replace_existing=replace_existing,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    session["forms_uploaded"] = result.get("forms", [])
    return {"status": "ok", **result}


@router.get("/sessions/{session_id}/forms")
async def list_session_forms(session_id: str):
    """List the forms uploaded for a CMS session."""
    session = _require_cms_session(session_id)
    forms_loaded: list = []
    try:
        info = await session["client"].get_info(session["sandbox_id"])
        forms_loaded = info.get("forms_loaded") or []
    except Exception:
        pass
    return {
        "session_id": session_id,
        "forms": list_forms_for_session(session_id, forms_loaded=forms_loaded),
    }


@router.delete("/sessions/{session_id}/forms/{filename}")
async def delete_session_form(session_id: str, filename: str):
    """Remove an uploaded form PDF from the session workspace."""
    _require_cms_session(session_id)
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid form filename")
    remove_form_from_session(session_id, filename)
    return {"status": "ok", "session_id": session_id, "removed": filename}


@router.get("/sessions/{session_id}/filled")
async def get_filled_forms(session_id: str):
    """
    Return the agent's filled values for every loaded form.

    Shape::

        {
          "session_id": ...,
          "metrics": {... output of env._compute_metrics() ...},
          "forms": {
            "i140": {
              "title": "I-140 (Immigrant…)",
              "field_count": 85,
              "filled_count": 42,
              "fields": [
                {"id": "Pt1Line2a_FamilyName", "raw_name": "form1[0]…",
                 "label": "Family Name (…)", "context": "…", "value": "Patel"},
                ...
              ]
            },
            ...
          }
        }

    Use ``raw_name`` to reconstruct the original PDF field id when computing
    precision / recall against an external GT.
    """
    session = _require_cms_session(session_id)
    client: SandboxClient = session["client"]
    sandbox_id: str = session["sandbox_id"]

    env = client._local_envs.get(sandbox_id) if hasattr(client, "_local_envs") else None
    if env is not None and hasattr(env, "export_filled"):
        metrics_step = await client.step(sandbox_id, {"action_type": 6, "params": "{}"})
        metrics = metrics_step.get("info", {}).get("result", {})
        return {
            "session_id": session_id,
            "metrics": metrics,
            "forms": env.export_filled(),
        }

    # Remote (Modal) sandbox: env runs out-of-process. Fall back to a step-driven
    # snapshot using get_form_progress + render. The detailed per-field listing
    # requires a dedicated env-server endpoint (added in env_server.py).
    try:
        resp = await client._remote_call(sandbox_id, "GET", "/filled", None)
        return {"session_id": session_id, **resp}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not export filled forms: {e}") from e


# ── Transcript REST Endpoints ─────────────────────────────────────────────────

from pydantic import BaseModel


class TranscriptPatch(BaseModel):
    harbor_status: str | None = None
    notes: str | None = None
    tags: list[str] | None = None


@router.get("/transcripts")
async def list_transcripts_endpoint(limit: int = 20, env_type: str | None = None):
    """List saved session transcripts, newest first."""
    items = list_transcripts(limit=limit, env_type=env_type)
    return {"transcripts": items, "total": len(items)}


@router.get("/transcripts/{session_id}")
async def get_transcript_endpoint(session_id: str):
    """Retrieve a full session transcript by ID."""
    data = load_transcript(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Transcript not found")
    return data


@router.patch("/transcripts/{session_id}")
async def patch_transcript_endpoint(session_id: str, body: TranscriptPatch):
    """Update harbor metadata (mark candidate, notes, tags)."""
    try:
        patch = body.model_dump(exclude_none=True)
        return update_transcript(session_id, patch)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Transcript not found")


@router.post("/transcripts/{session_id}/harbor")
async def export_harbor_endpoint(session_id: str):
    """Export session to ~/.form-filling-app/harbor-candidates/{session_id}/."""
    try:
        return export_harbor_candidate(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Transcript not found")


@router.delete("/transcripts/{session_id}")
async def delete_transcript_endpoint(session_id: str):
    """Delete a saved transcript."""
    file_path = transcript_path(session_id)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")
    file_path.unlink()
    ev = events_path(session_id)
    if ev.exists():
        ev.unlink()
    return {"deleted": session_id}
