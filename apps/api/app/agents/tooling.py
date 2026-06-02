"""Shared tool lists and agent-mode helpers for Games / CMS."""

from __future__ import annotations

# Claude Code built-in tools (executed by SDK in cwd workspace)
SDK_FILE_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]

# Env MCP server name (must match ``create_sdk_mcp_server(name=...)`` in env_mcp.py).
ENV_MCP_SERVER = "env"

# Games env tools (exposed via in-process MCP server "env")
CARDS_ENV_TOOLS = ["pick_cards", "view_grid", "get_score"]
BONZA_ENV_TOOLS = ["move_fragment", "view_grid", "get_status", "get_valid_words"]

# CMS env MCP tools
CMS_ENV_TOOLS = [
    "search_documents",
    "read_document",
    "list_documents",
    "list_categories",
    "answer_question",
    "fill_form_field",
    "get_form_status",
    "next_form_field",
    "submit_form_field",
    "get_form_progress",
    # Form parsing (agent-invoked; replaces the old form_schema.json baking).
    "list_uploaded_forms",
    "parse_form",
    # Batched iterator (preferred for form filling).
    "next_form_batch",
    "submit_form_batch",
    "export_filled_pdf",
]


def mcp_tool_name(bare_name: str) -> str:
    """Fully-qualified Claude SDK tool name for an env MCP tool."""
    return f"mcp__{ENV_MCP_SERVER}__{bare_name}"


def games_env_tool_names(env_type: str, agent_mode: str = "play") -> list[str]:
    """Bare env tool names (as registered in the MCP server).

    Single-mode design: ``agent_mode`` is accepted for back-compat but ignored.
    The agent always gets the action tool plus inspection tools.
    """
    if env_type == "cards":
        return list(CARDS_ENV_TOOLS)
    if env_type == "bonza":
        return list(BONZA_ENV_TOOLS)
    return []


def env_tool_bare_names(env_type: str, agent_mode: str = "play") -> list[str]:
    """Bare env tool names for an env_type (executor lookup uses these)."""
    if env_type in ("cards", "bonza"):
        return games_env_tool_names(env_type, agent_mode)
    if env_type == "cms":
        return list(CMS_ENV_TOOLS)
    return []


def env_tool_qualified_names(env_type: str, agent_mode: str = "play") -> set[str]:
    """Fully-qualified MCP tool names — used to match against SDK ``block.name``."""
    return {mcp_tool_name(n) for n in env_tool_bare_names(env_type, agent_mode)}


def build_allowed_tools(
    env_type: str,
    *,
    agent_mode: str = "play",
    include_sdk_file_tools: bool = False,
) -> list[str]:
    """Compose allowed_tools for ClaudeAgentOptions.

    Env MCP tools are emitted as ``mcp__env__<name>`` so the SDK's allowlist
    matches what the model actually sees. Built-in SDK tools (Bash, Read, …)
    stay bare.
    """
    bare_env = env_tool_bare_names(env_type, agent_mode)
    qualified = [mcp_tool_name(n) for n in bare_env]
    if include_sdk_file_tools:
        qualified = qualified + list(SDK_FILE_TOOLS)
    return qualified


def should_include_sdk_file_tools(env_type: str, agent_mode: str, eval_mode: bool) -> bool:
    """Include Bash/Read/Write/Glob/Grep in the SDK tool list?

    Single-mode design for games: the agent always has the file tools so it
    can write throw-away solver scripts if it wants. Eval mode still strips
    them so the agent must rely on the env MCP tools alone.
    """
    if eval_mode:
        return False
    return env_type in ("cms", "cards", "bonza")
