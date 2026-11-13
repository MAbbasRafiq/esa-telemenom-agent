"""Numeric grounding check for answers, in two passes.

1. Label check: `EW_precision = 0.40` in the answer must equal the value the
   tool printed for that same name (catches recall reported as precision).
2. Value check: any remaining number must appear somewhere in the tool output.

Pure Python (no LLM call), so it adds no latency or cost and is easy to test.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

# 1_234, 1,234, 12.5, -0.3, 73%
_NUMBER_RE = re.compile(r"[-+]?\d[\d,_]*(?:\.\d+)?%?")

# A digit glued to letters is part of an identifier, not a claim: F_0.50, v2.
_IDENTIFIER_RE = re.compile(r"[A-Za-z][_-]?\d$")

_NUMBER_BODY = r"[-+]?\d[\d,_]*(?:\.\d+)?%?"

# "EW_precision=0.68", "EW precision: 0.68", "EW recall is 0.40"
_LABELLED_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_ ]{2,38}?)\s*(?:=|:|\bis\b)\s*[*`]*(" + _NUMBER_BODY + r")",
)

# Markdown table row: "| EW_precision | 0.68 |"
_TABLE_ROW_RE = re.compile(
    r"\|\s*\**([A-Za-z][A-Za-z0-9_ ]{2,38}?)\**\s*\|\s*\**(" + _NUMBER_BODY + r")\**\s*\|",
)

# Names whose value is a setting or a row id, not a reported measurement.
_LABEL_STOPWORDS = frozenset({"beta", "k", "top", "window", "row", "rows"})


def _normalize_label(label: str) -> str:
    """'EW precision' / 'EW_precision' / '**EW Precision**' → 'ewprecision'."""
    return re.sub(r"[^a-z0-9]", "", (label or "").lower())


def labelled_numbers(text: str) -> dict[str, list[tuple[str, float]]]:
    """Map normalized label → [(raw token, value)] found in `text`."""
    out: dict[str, list[tuple[str, float]]] = {}
    for regex in (_LABELLED_RE, _TABLE_ROW_RE):
        for match in regex.finditer(text or ""):
            label = _normalize_label(match.group(1))
            if not label or label in _LABEL_STOPWORDS:
                continue
            token = match.group(2)
            value = _to_float(token)
            if value is None:
                continue
            out.setdefault(label, []).append((token, value))
    return out


def mismatched_labels(answer: str, observations: Sequence[str]) -> list[str]:
    """Labels the answer and the tools both mention with different values.

    Only labels the tool printed are checked, so unknown wording is left to
    the plain value check instead of being flagged twice.
    """
    joined = "\n".join(obs or "" for obs in observations)
    if not answer or not joined.strip():
        return []

    tool_pairs = labelled_numbers(joined)
    if not tool_pairs:
        return []

    flagged: list[str] = []
    seen: set[str] = set()

    for label, pairs in labelled_numbers(answer).items():
        tool_values = [value for _, value in tool_pairs.get(label, [])]
        if not tool_values:
            continue
        for token, value in pairs:
            if _matches(value, _decimals(token), tool_values):
                continue
            key = f"{label}={token}"
            if key in seen:
                continue
            seen.add(key)
            expected = ", ".join(f"{v:g}" for v in sorted(set(tool_values)))
            flagged.append(f"{token} (tool says {expected})")

    return flagged


def _to_float(token: str) -> float | None:
    cleaned = token.rstrip("%").replace(",", "").replace("_", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _decimals(token: str) -> int:
    body = token.rstrip("%")
    if "." not in body:
        return 0
    return len(body.split(".", 1)[1])


def observation_numbers(observations: Iterable[str]) -> list[float]:
    """All numeric values the tools actually produced."""
    values: list[float] = []
    for obs in observations:
        for match in _NUMBER_RE.finditer(obs or ""):
            value = _to_float(match.group())
            if value is not None:
                values.append(value)
    return values


def _matches(value: float, decimals: int, candidates: Sequence[float]) -> bool:
    """True if some tool number rounds to the answer's number at its precision."""
    tolerance = 0.5 * (10 ** -decimals) if decimals else 0.5
    for candidate in candidates:
        if abs(candidate - value) <= tolerance:
            return True
        # Answer may state a ratio as a percentage (0.7312 → "73%").
        if abs(candidate * 100.0 - value) <= tolerance:
            return True
    return False


def unsupported_numbers(answer: str, observations: Sequence[str]) -> list[str]:
    """Numbers in `answer` that no tool observation supports.

    Trivial values (0 and 1) and tokens that appear verbatim in the tool
    output are accepted without numeric matching.
    """
    if not answer:
        return []

    joined = "\n".join(obs or "" for obs in observations)
    if not joined.strip():
        return []

    candidates = observation_numbers([joined])
    flagged: list[str] = []
    seen: set[str] = set()

    for match in _NUMBER_RE.finditer(answer):
        token = match.group()
        if token in seen:
            continue

        value = _to_float(token)
        if value is None:
            continue
        if value in (0.0, 1.0, -1.0):
            continue

        # Verbatim hit covers identifiers, dates and exact quotes.
        if token in joined or token.rstrip("%") in joined:
            continue

        prefix = answer[max(0, match.start() - 2) : match.start()]
        if _IDENTIFIER_RE.search(prefix + token[:1]):
            continue

        if not _matches(value, _decimals(token), candidates):
            seen.add(token)
            flagged.append(token)

    return flagged


def turn_observations(messages: Sequence[Any]) -> list[str]:
    """Tool outputs produced after the last user message."""
    last_human = -1
    for i, msg in enumerate(messages):
        if getattr(msg, "type", None) == "human":
            last_human = i

    relevant = messages[last_human + 1 :] if last_human >= 0 else list(messages)
    out: list[str] = []
    for msg in relevant:
        if getattr(msg, "type", None) != "tool":
            continue
        content = msg.content
        out.append(content if isinstance(content, str) else str(content))
    return out


def last_ai_text(messages: Sequence[Any]) -> str:
    """Final assistant text (no pending tool calls)."""
    for msg in reversed(messages):
        if getattr(msg, "type", None) != "ai":
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = msg.content
        return (content if isinstance(content, str) else str(content)).strip()
    return ""


def check_answer(answer: str, observations: Sequence[str]) -> list[str]:
    """Label mismatches first, then numbers with no support at all."""
    label_flags = mismatched_labels(answer, observations)
    already = {flag.split(" ", 1)[0] for flag in label_flags}
    value_flags = [
        token
        for token in unsupported_numbers(answer, observations)
        if token not in already
    ]
    return label_flags + value_flags


def verify_messages(messages: Sequence[Any]) -> list[str]:
    """Flag numbers in this turn's answer that its tool output does not back."""
    return check_answer(last_ai_text(messages), turn_observations(messages))
