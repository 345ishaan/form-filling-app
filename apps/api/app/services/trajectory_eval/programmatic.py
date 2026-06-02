"""Programmatic transcript statistics (no LLM)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from app.services.trajectory_eval.context import TrajectoryContext

_MCP_ENV_PREFIX = re.compile(r"^mcp__env__")
_MCP_PREFIX = re.compile(r"^mcp__\w+__")


def normalize_tool_name(name: str) -> str:
    """Bare tool name for grouping (``mcp__env__parse_form`` → ``parse_form``)."""
    s = str(name or "unknown").strip()
    s = _MCP_ENV_PREFIX.sub("", s)
    s = _MCP_PREFIX.sub("", s)
    return s or "unknown"


def _turns_from_transcript(transcript: dict[str, Any]) -> list[dict]:
    turns = transcript.get("turns")
    return turns if isinstance(turns, list) else []


def analyze_tools(
    transcript: dict[str, Any],
    ctx: TrajectoryContext,
) -> dict[str, Any]:
    """Count tool usage and list never-used available tools."""
    turns = _turns_from_transcript(transcript)
    counts_raw: Counter[str] = Counter()
    counts_bare: Counter[str] = Counter()
    env_counts: Counter[str] = Counter()
    sdk_counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    durations: list[int] = []

    for turn in turns:
        for tc in turn.get("tool_calls") or []:
            raw = str(tc.get("name") or "unknown")
            bare = normalize_tool_name(raw)
            counts_raw[raw] += 1
            counts_bare[bare] += 1
            if tc.get("is_env_tool"):
                env_counts[bare] += 1
            else:
                sdk_counts[bare] += 1
            if tc.get("is_error"):
                errors[bare] += 1
            ms = tc.get("duration_ms")
            if isinstance(ms, (int, float)):
                durations.append(int(ms))

    available_env = set(ctx.env_tools)
    available_sdk = set(ctx.sdk_tools)
    used_bare = set(counts_bare.keys())

    unused_env = sorted(available_env - used_bare)
    unused_sdk = sorted(available_sdk - used_bare)

    thinking_chars = sum(
        len(" ".join(turn.get("thinking") or []))
        for turn in turns
    )
    agent_text_chars = sum(
        len(" ".join(turn.get("agent_text") or []))
        for turn in turns
    )

    return {
        "total_turns": len(turns),
        "total_tool_calls": sum(counts_bare.values()),
        "tool_calls_by_name": dict(sorted(counts_bare.items(), key=lambda x: (-x[1], x[0]))),
        "tool_calls_raw": dict(sorted(counts_raw.items(), key=lambda x: (-x[1], x[0]))),
        "env_tool_calls": dict(sorted(env_counts.items(), key=lambda x: (-x[1], x[0]))),
        "sdk_tool_calls": dict(sorted(sdk_counts.items(), key=lambda x: (-x[1], x[0]))),
        "tool_errors_by_name": dict(sorted(errors.items(), key=lambda x: (-x[1], x[0]))),
        "total_tool_errors": sum(errors.values()),
        "unused_env_tools": unused_env,
        "unused_sdk_tools": unused_sdk,
        "never_used_tools": sorted(set(unused_env) | set(unused_sdk)),
        "avg_tool_duration_ms": (
            round(sum(durations) / len(durations), 1) if durations else None
        ),
        "thinking_chars": thinking_chars,
        "agent_text_chars": agent_text_chars,
    }


def analyze_transcript_programmatic(
    transcript: dict[str, Any],
    ctx: TrajectoryContext,
) -> dict[str, Any]:
    tools = analyze_tools(transcript, ctx)
    usage = transcript.get("usage_totals") or {}
    return {
        "session_id": transcript.get("session_id"),
        "env_type": transcript.get("env_type") or ctx.env_type,
        "tools": tools,
        "usage_totals": usage,
    }
