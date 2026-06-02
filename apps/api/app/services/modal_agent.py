"""
Invoke agent turns on Modal (anthropic-api-key secret lives there only).
"""

from __future__ import annotations

import logging
from typing import Any

from app.config.runtime import agent_runs_locally, modal_app_name, modal_function_agent_turn

logger = logging.getLogger(__name__)


def _get_remote_fn():
    import modal

    return modal.Function.from_name(modal_app_name(), modal_function_agent_turn())


async def run_agent_turn_remote(session: dict, user_message: str | None) -> dict[str, Any]:
    """
    Run one agent turn inside Modal. Returns events + updated session fields.
    """
    payload = {
        "session_id": session["session_id"],
        "env_type": session["env_type"],
        "env_sandbox_url": session["sandbox_id"],
        "persona": session.get("persona", "default"),
        "agent_mode": session.get("agent_mode", "play"),
        "eval_mode": session.get("eval_mode", False),
        "user_message": user_message,
        "claude_session_id": session.get("claude_session_id"),
        "transcript_turn_count": len(session.get("transcript", [])),
    }

    fn = _get_remote_fn()
    result = await fn.remote.aio(payload)

    if result.get("claude_session_id"):
        session["claude_session_id"] = result["claude_session_id"]
    if result.get("final_render"):
        session["final_render"] = result["final_render"]
    if result.get("turn_transcript"):
        session.setdefault("transcript", []).append(result["turn_transcript"])

    return result


async def run_agent_turn(session: dict, user_message: str | None) -> list[dict[str, Any]]:
    """
    Run agent turn in-process (Modal API) or delegate to Modal function (local API).
    """
    if agent_runs_locally():
        from app.agents.runner import collect_agent_turn

        session["agent_running"] = True
        return await collect_agent_turn(session, user_message)

    result = await run_agent_turn_remote(session, user_message)
    return result.get("events", [])
