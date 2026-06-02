"""Environment metadata for trajectory evaluation (no reward / GT)."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.cms_agent import build_system_prompt as build_cms_system_prompt
from app.agents.tool_specs import SDK_TOOL_SPECS, cms_tool_specs
from app.agents.tooling import (
    SDK_FILE_TOOLS,
    build_allowed_tools,
    env_tool_bare_names,
    should_include_sdk_file_tools,
)


@dataclass
class TrajectoryContext:
    env_type: str
    system_prompt: str
    env_tools: list[str]
    sdk_tools: list[str]
    all_tools: list[str]
    tool_descriptions: str
    persona: str = "none"
    agent_mode: str = "play"
    extra: dict = field(default_factory=dict)


def _tool_specs_for_env(
    *,
    agent_mode: str = "play",
    include_sdk: bool = False,
) -> list[dict]:
    specs = list(cms_tool_specs())
    if include_sdk:
        specs = specs + list(SDK_TOOL_SPECS)
    return specs


def tool_catalog(ctx: TrajectoryContext) -> dict[str, str]:
    include_sdk = bool(ctx.sdk_tools)
    catalog: dict[str, str] = {}
    for spec in _tool_specs_for_env(agent_mode=ctx.agent_mode, include_sdk=include_sdk):
        name = str(spec.get("name") or "").strip()
        if name:
            catalog[name] = str(spec.get("description") or "").strip()
    return catalog


def _format_tool_spec(spec: dict) -> str:
    name = spec.get("name", "unknown")
    desc = spec.get("description", "").strip()
    schema = spec.get("input_schema") or {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    params: list[str] = []
    for key, meta in props.items():
        if not isinstance(meta, dict):
            params.append(key)
            continue
        hint = meta.get("description") or meta.get("type", "")
        req = " (required)" if key in required else ""
        params.append(f"{key}{req}: {hint}".strip(": "))
    param_line = f" Parameters: {', '.join(params)}." if params else ""
    return f"- **{name}**: {desc}{param_line}"


def build_tool_descriptions(
    *,
    agent_mode: str = "play",
    include_sdk: bool = False,
) -> str:
    lines: list[str] = ["### Environment tools", ""]
    specs = _tool_specs_for_env(agent_mode=agent_mode, include_sdk=False)
    if specs:
        lines.extend(_format_tool_spec(s) for s in specs)
    else:
        lines.append("_None_")
    if include_sdk:
        lines.extend(["", "### SDK workspace tools", ""])
        lines.extend(_format_tool_spec(s) for s in SDK_TOOL_SPECS)
    return "\n".join(lines)


def build_trajectory_context(
    env_type: str,
    *,
    persona: str = "none",
    agent_mode: str = "play",
    eval_mode: bool = False,
) -> TrajectoryContext:
    env_type = env_type.strip().lower()
    include_sdk = should_include_sdk_file_tools(env_type, agent_mode, eval_mode)
    system_prompt = build_cms_system_prompt(agent_mode=agent_mode)
    env_tools = list(env_tool_bare_names(env_type, agent_mode))
    sdk_tools = list(SDK_FILE_TOOLS) if include_sdk else []
    qualified = build_allowed_tools(
        env_type, agent_mode=agent_mode, include_sdk_file_tools=include_sdk
    )
    tool_descriptions = build_tool_descriptions(
        agent_mode=agent_mode, include_sdk=include_sdk,
    )
    return TrajectoryContext(
        env_type=env_type,
        system_prompt=system_prompt,
        env_tools=env_tools,
        sdk_tools=sdk_tools,
        all_tools=qualified,
        tool_descriptions=tool_descriptions,
        persona=persona,
        agent_mode=agent_mode,
        extra={},
    )


def build_trajectory_context_from_session(
    session: dict,
    *,
    bonza_scenario: str | None = None,
) -> TrajectoryContext:
    return build_trajectory_context(
        str(session.get("env_type") or "cms"),
        persona=str(session.get("persona") or "none"),
        agent_mode=str(session.get("agent_mode") or "play"),
        eval_mode=bool(session.get("eval_mode", False)),
    )
