"""
Anthropic / Claude Agent SDK authentication.

Modal: attach ``modal.Secret.from_name("anthropic-api-key")`` to functions;
       keys are injected as environment variables (typically ``ANTHROPIC_API_KEY``).

Local: set the same variable in ``apps/api/.env`` or export it in your shell.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from app.config.settings import ANTHROPIC_ENV_KEYS

logger = logging.getLogger(__name__)

_configured = False


def get_anthropic_api_key() -> str | None:
    """Return API key from environment (Modal secret or .env)."""
    for key in ANTHROPIC_ENV_KEYS:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    return None


def ensure_anthropic_api_key() -> str:
    """
    Require Anthropic API key; normalize into os.environ for Claude CLI/SDK.

    Raises:
        RuntimeError: if no key is configured.
    """
    global _configured
    key = get_anthropic_api_key()
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. For local dev, add it to apps/api/.env. "
            "On Modal, attach secret 'anthropic-api-key' (with key ANTHROPIC_API_KEY) "
            "to the API function, or run: modal profile activate <profile> && modal deploy modal_app.py"
        )

    os.environ["ANTHROPIC_API_KEY"] = key
    os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", key)
    _configured = True
    return key


def is_anthropic_configured() -> bool:
    return bool(get_anthropic_api_key())


def setup_claude_code_config() -> bool:
    """
    Write ~/.claude/settings.json env block so Claude Code CLI picks up the key.
    (Standard Claude Code config initialization.)
    """
    try:
        key = get_anthropic_api_key()
        if not key:
            return False

        claude_dir = Path.home() / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        settings_path = claude_dir / "settings.json"

        env_block = {
            "ANTHROPIC_API_KEY": key,
            "ANTHROPIC_AUTH_TOKEN": key,
        }

        if settings_path.exists():
            try:
                data = json.loads(settings_path.read_text())
            except Exception:
                data = {}
        else:
            data = {}

        data.setdefault("env", {})
        data["env"].update(env_block)
        settings_path.write_text(json.dumps(data, indent=2))
        return True
    except Exception as e:
        logger.warning("Failed to setup Claude Code config: %s", e)
        return False


def get_claude_sdk_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """
    Environment dict for ``ClaudeAgentOptions(env=...)``.

    Call ``ensure_anthropic_api_key()`` before first SDK use.
    """
    key = get_anthropic_api_key() or ""
    base = {
        "ANTHROPIC_API_KEY": key,
        "ANTHROPIC_AUTH_TOKEN": key,
        # Cold-start tolerance for Claude Code subprocess (ms).
        "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": os.environ.get(
            "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "120000"
        ),
    }
    if extra:
        base.update({k: v for k, v in extra.items() if v is not None})
    return base


def configure_anthropic_on_startup() -> None:
    """FastAPI startup: load key and prepare Claude CLI config."""
    if not is_anthropic_configured():
        logger.warning(
            "ANTHROPIC_API_KEY not set — agent WebSocket will fail until configured "
            "(apps/api/.env locally, or Modal secret anthropic-api-key on deploy)."
        )
        return
    ensure_anthropic_api_key()
    setup_claude_code_config()
    logger.info("Anthropic API key configured for Claude Agent SDK")
