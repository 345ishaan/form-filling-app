"""
Session transcript storage (filesystem-backed).

- Summary JSON: ~/.form-filling-app/sessions/{session_id}.json
- Raw event JSONL: ~/.form-filling-app/sessions/{session_id}.jsonl
- Harbor candidates: ~/.form-filling-app/harbor-candidates/{session_id}/

Uses a turn + tool timeline + export bundle shape without Postgres.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

TRANSCRIPT_DIR = Path.home() / ".form-filling-app" / "sessions"
HARBOR_DIR = Path.home() / ".form-filling-app" / "harbor-candidates"


def events_path(session_id: str) -> Path:
    return TRANSCRIPT_DIR / f"{session_id}.jsonl"


def transcript_path(session_id: str) -> Path:
    return TRANSCRIPT_DIR / f"{session_id}.json"


def append_event(session_id: str, event: dict[str, Any]) -> None:
    """Append one SDK/WS event to the session JSONL log."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), **event}
    with events_path(session_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def build_tool_summary(transcript: list[dict]) -> dict:
    tools = Counter()
    env_tools = Counter()
    sdk_tools = Counter()
    total_duration = 0
    for turn in transcript:
        for tc in turn.get("tool_calls", []):
            name = tc.get("name") or "unknown"
            tools[name] += 1
            if tc.get("is_env_tool"):
                env_tools[name] += 1
            else:
                sdk_tools[name] += 1
            total_duration += tc.get("duration_ms", 0)
    return {
        "total_turns": len(transcript),
        "total_tool_calls": sum(tools.values()),
        "tools_used": dict(tools),
        "env_tools_used": dict(env_tools),
        "sdk_tools_used": dict(sdk_tools),
        "total_tool_duration_ms": total_duration,
    }


def build_tool_timeline(transcript: list[dict]) -> list[dict]:
    """Flat chronological list of tool calls for Review UI."""
    timeline: list[dict] = []
    for turn_idx, turn in enumerate(transcript):
        for tc in turn.get("tool_calls", []):
            timeline.append({
                "turn": turn_idx + 1,
                "name": tc.get("name", "unknown"),
                "tool_use_id": tc.get("tool_use_id"),
                "input": tc.get("input"),
                "result": tc.get("result"),
                "is_error": tc.get("is_error", False),
                "is_env_tool": tc.get("is_env_tool", False),
                "duration_ms": tc.get("duration_ms"),
            })
    return timeline


def session_to_payload(session: dict) -> dict:
    transcript = session.get("transcript", [])
    now = time.time()
    harbor = session.get("harbor") or {
        "status": "none",  # none | candidate | exported
        "notes": "",
        "tags": [],
        "marked_at": None,
        "export_path": None,
    }
    usage_totals = {
        "total_cost_usd": 0.0,
        "duration_ms": 0,
    }
    for turn in transcript:
        u = turn.get("usage") or {}
        if u.get("total_cost_usd"):
            usage_totals["total_cost_usd"] += float(u["total_cost_usd"])
        if u.get("duration_ms"):
            usage_totals["duration_ms"] += int(u["duration_ms"])

    return {
        "session_id": session["session_id"],
        "env_type": session["env_type"],
        "persona": session.get("persona", "default"),
        "agent_mode": session.get("agent_mode", "play"),
        "eval_mode": session.get("eval_mode", False),
        "sandbox_mode": session.get("sandbox_mode", "local"),
        "claude_session_id": session.get("claude_session_id"),
        "initial_render": session.get("initial_render"),
        "final_render": session.get("final_render"),
        "created_at": session.get("created_at", now),
        "updated_at": now,
        "feedback_log": session.get("feedback_log", []),
        "turns": transcript,
        "tool_summary": build_tool_summary(transcript),
        "tool_timeline": build_tool_timeline(transcript),
        "usage_totals": usage_totals,
        "harbor": harbor,
    }


def save_session_transcript(session: dict, *, sandbox_mode: str = "local") -> Path:
    session["sandbox_mode"] = sandbox_mode
    session.setdefault("created_at", time.time())
    payload = session_to_payload(session)
    path = transcript_path(session["session_id"])
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_transcript(session_id: str) -> dict | None:
    path = transcript_path(session_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def update_transcript(session_id: str, patch: dict) -> dict:
    data = load_transcript(session_id)
    if not data:
        raise FileNotFoundError(session_id)
    harbor = data.setdefault("harbor", {})
    if "harbor_status" in patch:
        harbor["status"] = patch["harbor_status"]
        if patch["harbor_status"] == "candidate":
            harbor["marked_at"] = time.time()
    if "notes" in patch:
        harbor["notes"] = patch["notes"]
    if "tags" in patch:
        harbor["tags"] = patch["tags"]
    data["updated_at"] = time.time()
    transcript_path(session_id).write_text(json.dumps(data, indent=2, default=str))
    return data


def list_transcripts(*, limit: int = 20, env_type: str | None = None) -> list[dict]:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    for f in sorted(TRANSCRIPT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            if env_type and data.get("env_type") != env_type:
                continue
            summary = data.get("tool_summary") or {}
            items.append({
                "session_id": data["session_id"],
                "env_type": data.get("env_type"),
                "persona": data.get("persona"),
                "agent_mode": data.get("agent_mode"),
                "turn_count": len(data.get("turns", [])),
                "tool_count": summary.get("total_tool_calls", 0),
                "tools_used": summary.get("tools_used", {}),
                "feedback_count": len(data.get("feedback_log", [])),
                "harbor_status": (data.get("harbor") or {}).get("status", "none"),
                "saved_at": f.stat().st_mtime,
                "updated_at": data.get("updated_at"),
            })
        except Exception:
            continue
        if len(items) >= limit:
            break
    return items


def export_harbor_candidate(session_id: str) -> dict:
    """
    Write a Harbor-oriented bundle for a saved session (stub task scaffold).

    Output: ~/.form-filling-app/harbor-candidates/{session_id}/
      - trajectory.json — full transcript
      - task.toml — minimal Harbor task metadata
      - README.md — human notes + next steps
    """
    data = load_transcript(session_id)
    if not data:
        raise FileNotFoundError(session_id)

    out_dir = HARBOR_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "trajectory.json").write_text(json.dumps(data, indent=2, default=str))

    env_type = data.get("env_type", "unknown")
    task_toml = f'''version = "1.0"

[metadata]
author_name = "form-filling-app"
difficulty = "medium"
category = "agent-sim"
tags = ["{env_type}", "session-export"]

[agent]
timeout_sec = 3600.0

[environment]
# TODO: wire Gymnasium env + MCP from tasks/{env_type}/
'''
    (out_dir / "task.toml").write_text(task_toml)

    harbor = data.setdefault("harbor", {})
    notes = harbor.get("notes", "")
    readme = f"""# Harbor candidate: {session_id}

Environment: **{env_type}**
Persona: {data.get("persona", "default")}
Turns: {len(data.get("turns", []))}
Tool calls: {(data.get("tool_summary") or {}).get("total_tool_calls", 0)}

## Notes
{notes or "(none)"}

## Next steps
1. Add verifier / ground-truth from session failure
2. Copy env Dockerfile from `tasks/{env_type}/`
3. Register in Harbor registry
"""
    (out_dir / "README.md").write_text(readme)

    harbor["status"] = "exported"
    harbor["export_path"] = str(out_dir)
    harbor["exported_at"] = time.time()
    data["harbor"] = harbor
    data["updated_at"] = time.time()
    transcript_path(session_id).write_text(json.dumps(data, indent=2, default=str))

    return {
        "session_id": session_id,
        "export_path": str(out_dir),
        "files": ["trajectory.json", "task.toml", "README.md"],
    }
