"""
Runtime settings — Modal secret names and local .env loading.

Modal secrets (profile-scoped): activate your profile before deploy/run, e.g.
  modal profile activate <your-profile>
  modal deploy modal_app.py

The secret ``anthropic-api-key`` should define at least:
  ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load apps/api/.env for local uvicorn (Modal injects env vars in deployed functions).
_API_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_API_ROOT / ".env", override=False)

# Modal secret names (must exist in the active Modal profile).
MODAL_SECRET_ANTHROPIC = "ant-key"

# Env var names we accept (Modal secret should use ANTHROPIC_API_KEY).
ANTHROPIC_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
)

def get_agent_model() -> str:
    """Single model id for games + CMS Claude Agent SDK sessions."""
    return "claude-sonnet-4-6"