"""
Runtime detection: local orchestrator vs Modal (agent + secrets).

Agent and Anthropic credentials always run on Modal. Local ``uvicorn`` only
proxies WebSocket traffic and manages env sandboxes via Modal.
"""

from __future__ import annotations

import os


def is_modal_runtime() -> bool:
    """True when this process is inside a Modal container (e.g. ``fastapi_asgi``)."""
    if os.environ.get("MODAL_ENVIRONMENT"):
        return True
    if os.environ.get("MODAL_IS_REMOTE") == "1":
        return True
    # Modal worker filesystem marker
    if os.path.exists("/pkg") and os.environ.get("MODAL_TASK_ID"):
        return True
    try:
        import modal

        return not modal.is_local()
    except Exception:
        return False


def get_sandbox_mode() -> str:
    """Env sandboxes always use Modal (override with SANDBOX_MODE=local for debugging)."""
    return os.environ.get("SANDBOX_MODE", "modal").strip().lower()


def agent_runs_locally() -> bool:
    """Agent SDK runs in this process only on Modal."""
    return is_modal_runtime()


def modal_app_name() -> str:
    return os.environ.get("MODAL_APP_NAME", "form-filling-app")


def modal_function_agent_turn() -> str:
    return os.environ.get("MODAL_AGENT_FUNCTION", "run_agent_turn")
