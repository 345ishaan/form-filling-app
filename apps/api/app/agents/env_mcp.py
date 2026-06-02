"""In-process MCP server wrapping Gymnasium env tool executors."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from claude_agent_sdk import create_sdk_mcp_server, tool

from app.services.sandbox_client import SandboxClient

ExecuteFn = Callable[[str, dict, str, SandboxClient], Awaitable[str]]


def _tool_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        out["is_error"] = True
    return out


def build_env_mcp_server(
    tool_specs: list[dict],
    execute_fn: ExecuteFn,
    sandbox_id: str,
    client: SandboxClient,
):
    """Build SDK MCP server from OpenAI-style tool spec dicts."""

    sdk_tools = []
    for spec in tool_specs:
        name = spec["name"]
        description = spec["description"]
        schema = spec.get("input_schema", {"type": "object", "properties": {}})

        def _make_handler(tool_name: str):
            async def handler(args: dict) -> dict[str, Any]:
                try:
                    result = await execute_fn(tool_name, args, sandbox_id, client)
                    return _tool_result(str(result))
                except Exception as e:
                    return _tool_result(f"Tool error: {e}", is_error=True)

            return handler

        handler = _make_handler(name)
        sdk_tools.append(tool(name, description, schema)(handler))

    return create_sdk_mcp_server(name="env", version="1.0.0", tools=sdk_tools)
