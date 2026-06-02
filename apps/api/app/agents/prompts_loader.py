"""Load system prompts from ``app/prompts/``."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

_PROMPT_FILES = {
    "cms": ("cms_system.md",),
}


def load_system_prompt(env_type: str) -> str:
    """Read prompt file for ``cms``."""
    key = env_type.strip().lower()
    names = _PROMPT_FILES.get(key)
    if not names:
        raise ValueError(f"No system prompt for env_type={env_type!r}")
    for name in names:
        path = _PROMPTS_DIR / name
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"Missing prompt file(s) for {env_type}: {names}")
