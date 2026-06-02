"""Agent turn runner — WebSocket (on Modal) and ``run_agent_turn`` Modal function."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from app.agents.cms_agent import (
    build_agent_options as build_cms_options,
    execute_tool as execute_cms_tool,
)
from app.agents.env_mcp import build_env_mcp_server
from app.agents.tooling import env_tool_qualified_names
from app.config.anthropic import ensure_anthropic_api_key, get_claude_sdk_env
from app.config.settings import get_agent_model
from app.services.sandbox_client import SandboxClient
from app.services.transcript_service import append_event

SDK_MAX_TURNS = 100
SDK_MAX_BUFFER_SIZE = 100_000_000


def env_tool_names(agent_mode: str = "play") -> set[str]:
    return env_tool_qualified_names("cms", agent_mode)


async def ensure_sdk_client(session: dict) -> None:
    if session.get("sdk_connected"):
        return

    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    ensure_anthropic_api_key()

    sandbox_id = session["sandbox_id"]
    client: SandboxClient = session["client"]
    agent_mode = session.get("agent_mode", "play")
    eval_mode = session.get("eval_mode", False)
    session_id = session["session_id"]

    agent_opts = build_cms_options(
        sandbox_id, client,
        session_id=session_id,
        agent_mode=agent_mode,
        eval_mode=eval_mode,
    )

    async def execute_fn(name, inp, sid, cli):
        return await execute_cms_tool(
            name, inp, sid, cli,
            eval_mode=eval_mode,
            session_id=session_id,
        )

    env_mcp = build_env_mcp_server(
        agent_opts["tool_specs"], execute_fn, sandbox_id, client
    )

    options_kwargs: dict = {
        "system_prompt": agent_opts["system_prompt"],
        "allowed_tools": agent_opts["allowed_tools"],
        "mcp_servers": {"env": env_mcp},
        "model": agent_opts.get("model") or get_agent_model(),
        "max_turns": SDK_MAX_TURNS,
        "max_buffer_size": SDK_MAX_BUFFER_SIZE,
        "permission_mode": agent_opts.get("permission_mode", "acceptEdits"),
        "env": get_claude_sdk_env(agent_opts.get("env")),
    }
    if agent_opts.get("cwd"):
        options_kwargs["cwd"] = agent_opts["cwd"]
    if session.get("claude_session_id"):
        options_kwargs["resume"] = session["claude_session_id"]

    sdk_client = ClaudeSDKClient(options=ClaudeAgentOptions(**options_kwargs))
    await sdk_client.connect()
    session["sdk_client"] = sdk_client
    session["sdk_connected"] = True


async def stream_agent_turn(
    session: dict,
    user_message: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    sandbox_id = session["sandbox_id"]
    client: SandboxClient = session["client"]
    env_tools = env_tool_names(session.get("agent_mode", "play"))
    sid = session["session_id"]
    pending_tools: dict[str, dict] = {}

    turn_transcript: dict[str, Any] = {
        "turn": len(session.setdefault("transcript", [])),
        "timestamp": time.time(),
        "user_message": user_message,
        "agent_mode": session.get("agent_mode", "play"),
        "agent_text": [],
        "thinking": [],
        "tool_calls": [],
        "state_snapshots": [],
        "usage": None,
        "error": None,
    }

    if user_message:
        append_event(sid, {"type": "user_message", "content": user_message})

    try:
        await ensure_sdk_client(session)
        sdk_client = session["sdk_client"]

        if user_message:
            await sdk_client.query(user_message)

        async for message in sdk_client.receive_response():
            if session.get("interrupt_requested"):
                try:
                    await sdk_client.interrupt()
                except Exception:
                    pass
                session["interrupt_requested"] = False
                turn_transcript["interrupted"] = True
                yield {"type": "text", "text": "⚠ Interrupted"}
                yield {"type": "interrupted"}
                break

            if isinstance(message, SystemMessage):
                if message.subtype == "init" and isinstance(message.data, dict):
                    csid = message.data.get("session_id")
                    if csid:
                        session["claude_session_id"] = csid
                        turn_transcript["claude_session_id"] = csid

            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        turn_transcript["agent_text"].append(block.text)
                        append_event(sid, {"type": "text", "text": block.text})
                        yield {"type": "text", "text": block.text}
                    elif isinstance(block, ThinkingBlock):
                        turn_transcript["thinking"].append(block.thinking)
                        append_event(sid, {"type": "thinking", "thinking": block.thinking[:2000]})
                        yield {"type": "thinking", "thinking": block.thinking}
                    elif isinstance(block, ToolUseBlock):
                        pending_tools[block.id] = {
                            "name": block.name,
                            "tool_use_id": block.id,
                            "input": block.input,
                            "started_at": time.time(),
                            "is_env_tool": block.name in env_tools,
                        }
                        append_event(sid, {
                            "type": "tool_use",
                            "name": block.name,
                            "id": block.id,
                            "input": block.input,
                        })
                        yield {
                            "type": "tool_use",
                            "name": block.name,
                            "id": block.id,
                            "input": block.input,
                        }

            elif isinstance(message, UserMessage):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        pending = pending_tools.pop(block.tool_use_id, {})
                        tool_name = pending.get("name") or getattr(block, "name", "") or "unknown"
                        started = pending.get("started_at", time.time())
                        duration_ms = int((time.time() - started) * 1000)
                        result_text = str(block.content)[:8000]
                        is_env = pending.get("is_env_tool", tool_name in env_tools)
                        turn_transcript["tool_calls"].append({
                            "name": tool_name,
                            "tool_use_id": block.tool_use_id,
                            "input": pending.get("input"),
                            "result": result_text,
                            "is_error": getattr(block, "is_error", False),
                            "is_env_tool": is_env,
                            "duration_ms": duration_ms,
                        })
                        append_event(sid, {
                            "type": "tool_result",
                            "name": tool_name,
                            "tool_use_id": block.tool_use_id,
                            "is_error": getattr(block, "is_error", False),
                            "duration_ms": duration_ms,
                        })
                        yield {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": str(block.content),
                            "is_error": getattr(block, "is_error", False),
                        }
                        if tool_name in env_tools:
                            render = await client.render(sandbox_id)
                            if render:
                                turn_transcript["state_snapshots"].append(render)
                                yield {"type": "state_update", "render": render}

            elif isinstance(message, ResultMessage):
                turn_transcript["usage"] = {
                    "duration_ms": getattr(message, "duration_ms", None),
                    "total_cost_usd": getattr(message, "total_cost_usd", None),
                    "usage": getattr(message, "usage", None),
                    "num_turns": getattr(message, "num_turns", None),
                }
                render = await client.render(sandbox_id)
                if render:
                    turn_transcript["state_snapshots"].append(render)
                    session["final_render"] = render
                append_event(sid, {
                    "type": "complete",
                    "duration_ms": getattr(message, "duration_ms", None),
                    "total_cost_usd": getattr(message, "total_cost_usd", None),
                })
                yield {
                    "type": "complete",
                    "message": getattr(message, "result", "Done"),
                    "render": render,
                    "session_id": session["session_id"],
                    "duration_ms": getattr(message, "duration_ms", None),
                    "total_cost_usd": getattr(message, "total_cost_usd", None),
                    "usage": getattr(message, "usage", None),
                    "num_turns": getattr(message, "num_turns", None),
                }

    except Exception as e:
        turn_transcript["error"] = str(e)
        yield {
            "type": "error",
            "error": str(e),
            "error_type": "agent_error",
            "recoverable": True,
        }
    finally:
        turn_transcript["duration_ms"] = int((time.time() - turn_transcript["timestamp"]) * 1000)
        session.setdefault("transcript", []).append(turn_transcript)
        session["agent_running"] = False
        session["_last_turn_transcript"] = turn_transcript


async def collect_agent_turn(session: dict, user_message: str | None) -> list[dict]:
    session["agent_running"] = True
    events: list[dict] = []
    async for event in stream_agent_turn(session, user_message):
        events.append(event)
    return events
