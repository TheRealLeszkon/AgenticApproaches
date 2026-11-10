"""Robust extraction of a JSON object from free-form VLM text.

VLMs frequently wrap JSON in prose ("Here is the JSON:") or ```json fences, or emit
trailing commentary. This module tries, in order:

1. direct json.loads
2. strip markdown code fences and retry
3. locate the first balanced {...} block (brace-matching, string-aware) and parse it

If everything fails we raise JSONParseError carrying the raw text so callers can save
it for diagnostics and optionally issue a correction prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any


class JSONParseError(ValueError):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _find_balanced_object(text: str) -> str | None:
    """Return the first top-level {...} block, respecting strings/escapes."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        # unbalanced from this start; try the next brace
        start = text.find("{", start + 1)
    return None


def parse_json(text: str) -> dict[str, Any]:
    """Best-effort parse of a single JSON object from ``text``."""
    if text is None:
        raise JSONParseError("empty model output", raw="")

    candidates: list[str] = []
    candidates.append(text.strip())
    stripped = _strip_fences(text).strip()
    if stripped and stripped != text.strip():
        candidates.append(stripped)
    block = _find_balanced_object(stripped) or _find_balanced_object(text)
    if block:
        candidates.append(block)

    last_err: Exception | None = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
            # a bare list/scalar isn't what any stage expects
            last_err = ValueError("parsed JSON was not an object")
        except json.JSONDecodeError as e:
            last_err = e

    raise JSONParseError(f"could not extract JSON object ({last_err})", raw=text)


def correction_prompt(schema_hint: str, raw: str) -> str:
    """Prompt asking the model to re-emit valid JSON only."""
    return (
        "Your previous response could not be parsed as JSON.\n"
        "Return ONLY a single valid JSON object — no prose, no markdown code fences, "
        "no commentary before or after.\n"
        f"It must follow this shape:\n{schema_hint}\n\n"
        "Your previous (unparseable) response was:\n"
        f"{raw[:1500]}"
    )
