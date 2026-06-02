"""
Modal deployment for form-filling-app (CMS WebSocket API + agent).

  cd apps/api
  uv run python -m modal serve modal_app.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modal_common import MODAL_APP_NAME, WORKSPACE_MOUNT, anthropic_secret, build_api_image, build_env_image

app = modal.App(MODAL_APP_NAME)
env_image = build_env_image()
api_image = build_api_image()


@app.function(
    image=api_image,
    secrets=[anthropic_secret],
    volumes=WORKSPACE_MOUNT,
    cpu=2.0,
    memory=4096,
    timeout=3600,
    min_containers=0,
)
@modal.asgi_app()
def fastapi_asgi():
    sys.path.insert(0, "/root")
    from app.config.anthropic import configure_anthropic_on_startup

    configure_anthropic_on_startup()
    from app.main import app as fastapi_app

    return fastapi_app


@app.function(
    image=api_image,
    secrets=[anthropic_secret],
    volumes=WORKSPACE_MOUNT,
    cpu=2.0,
    memory=4096,
    timeout=3600,
)
def run_agent_turn(payload: dict) -> dict:
    sys.path.insert(0, "/root")
    from app.config.anthropic import configure_anthropic_on_startup
    from app.agents.runner import collect_agent_turn
    from app.services.sandbox_client import SandboxClient
    from app.services.workspace import get_session_workspace

    configure_anthropic_on_startup()

    session = {
        "session_id": payload["session_id"],
        "env_type": "cms",
        "sandbox_id": payload["env_sandbox_url"],
        "client": SandboxClient(mode="modal"),
        "persona": payload.get("persona", "default"),
        "agent_mode": payload.get("agent_mode", "play"),
        "eval_mode": payload.get("eval_mode", False),
        "claude_session_id": payload.get("claude_session_id"),
        "transcript": [],
        "sdk_client": None,
        "sdk_connected": False,
        "interrupt_requested": False,
    }
    get_session_workspace(session["session_id"], "cms")

    events = asyncio.run(collect_agent_turn(session, payload.get("user_message")))

    return {
        "events": events,
        "claude_session_id": session.get("claude_session_id"),
        "final_render": session.get("final_render"),
        "turn_transcript": session.get("_last_turn_transcript"),
    }
