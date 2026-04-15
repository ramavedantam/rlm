"""Utility functions for the RLM module."""

from __future__ import annotations

import re


def extract_code_blocks(text: str) -> list[str]:
    """Extract code from ```repl ... ``` blocks in text.

    Returns an empty list if no blocks are found.
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    return [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL)]


def extract_final(text: str) -> str | None:
    """Check if text contains a FINAL(...) call outside of a code block.

    Returns the argument string if found, None otherwise.
    Code-block FINAL() calls are handled by the injected function in REPLEnv.
    """
    # Strip code blocks first so we don't match FINAL() inside them
    stripped = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Find "FINAL(" and then match balanced parens
    idx = stripped.find("FINAL(")
    if idx == -1:
        return None

    start = idx + len("FINAL(")
    depth = 1
    i = start
    while i < len(stripped) and depth > 0:
        if stripped[i] == "(":
            depth += 1
        elif stripped[i] == ")":
            depth -= 1
        i += 1

    if depth == 0:
        content = stripped[start : i - 1].strip()
        # Strip surrounding quotes if present
        if (content.startswith('"') and content.endswith('"')) or \
           (content.startswith("'") and content.endswith("'")):
            content = content[1:-1]
        # Also handle triple-quoted strings
        for q in ['"""', "'''"]:
            if content.startswith(q) and content.endswith(q):
                content = content[3:-3]
                break
        return content
    return None


def truncate(text: str, max_len: int = 100_000) -> str:
    """Truncate text to max_len characters with an indicator."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [truncated, {len(text) - max_len} chars omitted]"
