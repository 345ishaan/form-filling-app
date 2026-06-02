"""Robust parsing of env tool results from transcript strings."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def parse_tool_result(result: str | None) -> dict[str, Any]:
    """Parse tool result JSON; tolerate fences, prefixes, and truncated text."""
    if not result:
        return {}
    text = str(result).strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # List-wrapped MCP payloads
    if text.startswith("["):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict) and item.get("type") == "text":
                        inner = item.get("text")
                        if isinstance(inner, str):
                            return parse_tool_result(inner)
                    if isinstance(item, dict):
                        return item
        except json.JSONDecodeError:
            pass

    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return {"_raw_text": text}


def _coerce_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    return None


def infer_bonza_move(
    payload: dict[str, Any],
    result_text: str,
    *,
    is_error: bool | None = None,
) -> dict[str, Any]:
    """Derive valid_move / error from JSON and plain-text env messages."""
    raw = payload.get("_raw_text") or result_text or ""
    valid = _coerce_bool(payload.get("valid_move"))
    err = str(payload.get("error") or "").strip() or None
    msg = str(payload.get("message") or "").strip()

    if valid is None:
        low = raw.lower()
        if "fragment moved" in low or '"valid_move": true' in low:
            valid = True
        elif any(
            x in low
            for x in (
                "move rejected",
                "action blocked",
                "action failed",
                "invalid",
                '"valid_move": false',
            )
        ):
            valid = False

    if not err:
        for pattern in (
            r"(Action Failed:[^\n]+)",
            r"(Action Blocked:[^\n]+)",
            r"(Move rejected[^\n]*)",
        ):
            m = re.search(pattern, raw, re.I)
            if m:
                err = m.group(1).strip()
                break

    if not err and msg and valid is False:
        err = msg

    if valid is None and is_error:
        valid = False

    if valid is None:
        valid = False if err or is_error else True

    return {"valid": bool(valid), "error": err, "message": msg or None}


def infer_cards_pick(
    payload: dict[str, Any],
    result_text: str,
    tool_input: dict[str, Any] | None,
) -> dict[str, Any]:
    """Derive pick outcome from structured fields, reward, and message text."""
    raw = payload.get("_raw_text") or result_text or ""
    out: dict[str, Any] = {
        "positions": payload.get("positions"),
        "picked_sum": payload.get("picked_sum"),
        "valid_triplet": payload.get("valid_triplet"),
        "failure_reason": payload.get("failure_reason"),
        "reward": payload.get("reward"),
    }

    if tool_input:
        inp = tool_input
        if not out["positions"]:
            out["positions"] = [inp.get("pos1"), inp.get("pos2"), inp.get("pos3")]

    if out["reward"] is None:
        m = re.search(r"Reward:\s*(-?\d+(?:\.\d+)?)", raw, re.I)
        if m:
            out["reward"] = float(m.group(1))

    reward = out.get("reward")
    if out["failure_reason"] is None and reward is not None:
        try:
            r = float(reward)
            if r >= 15:
                out["valid_triplet"] = True
                out["failure_reason"] = None
            elif r == -2:
                out["failure_reason"] = "duplicate_positions"
                out["valid_triplet"] = False
            elif r == -1:
                low = raw.lower()
                if "duplicate" in low or "unique" in low:
                    out["failure_reason"] = "duplicate_positions"
                elif "empty" in low:
                    out["failure_reason"] = "empty_cell"
                elif "range" in low or "out of" in low:
                    out["failure_reason"] = "out_of_range"
                else:
                    out["failure_reason"] = "sum_not_15"
                out["valid_triplet"] = False
        except (TypeError, ValueError):
            pass

    if out["picked_sum"] is None:
        m = re.search(r"sum[:\s]+(\d+)", raw, re.I)
        if m:
            out["picked_sum"] = int(m.group(1))

    fr = out.get("failure_reason")
    if fr == "invalid_pick":
        out["failure_reason"] = "sum_not_15"

    return out
