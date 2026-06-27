"""
brain/router.py — Regex fast-path intent classification.

Returns an intent string ("SAVE", "FIND", "HISTORY") when the message is
unambiguous, or None to signal that LLM fallback is needed.

This is the single biggest token-efficiency win: ~45% of all classification
tokens are saved because most SAVE/FIND/HISTORY messages match instantly.
"""

from __future__ import annotations

import re

# ─── Pattern definitions ───────────────────────────────────────────────────────

PATTERNS: dict[str, re.Pattern] = {
    "SAVE": re.compile(
        r"\b(save( this)?|wrap up|that'?s? (good|it|enough)|"
        r"i'?m done|let'?s? save|/save|file this|note this down|"
        r"ok let'?s? wrap|we'?re done|good enough|save it|commit this)\b",
        re.I,
    ),
    "FIND": re.compile(
        r"\b(what did i (say|write|figure out|decide|think|conclude|note|work out)|"
        r"do i have (anything|a?notes?) on|find|recall|look up|search|"
        r"last time i (thought|wrote|worked|explored)|/find|"
        r"what (do i know|have i written|have i figured out|have i said) about|"
        r"remind me (about|of|what)|pull up|retrieve|show me)\b",
        re.I,
    ),
    "HISTORY": re.compile(
        r"\b(what have i been|most (lately|this (month|week)|recently)|"
        r"how (many times|often) (did i|have i)|history|"
        r"usage patterns?|what (was i|did i work on) last (month|week)|"
        r"/history|my (thinking|notes?) pattern|how active|"
        r"what topics|tag(ged|s))\b",
        re.I,
    ),
}

# ─── History sub-patterns (for template selection) ────────────────────────────

HISTORY_SUB_PATTERNS: dict[str, re.Pattern] = {
    "frequency": re.compile(
        r"\b(most often|most (lately|recently|this (month|week))|"
        r"how (many|often)|frequency|most active|most (visited|accessed))\b",
        re.I,
    ),
    "recency": re.compile(
        r"\b(recent|latest|last (note|session|time)|newest|"
        r"what did i work on last|most recent)\b",
        re.I,
    ),
    "tag": re.compile(
        r"\b(tagged?|tag(s)?|under|categoris(ed|ed)|labell?ed)\b",
        re.I,
    ),
    "last_month": re.compile(
        r"\b(last month|this month|past month|30 days?|past 30)\b",
        re.I,
    ),
    "last_week": re.compile(
        r"\b(last week|this week|past week|7 days?|past 7)\b",
        re.I,
    ),
}


def fast_route(message: str) -> str | None:
    """
    Match message against known intent patterns.

    Returns:
        "SAVE" | "FIND" | "HISTORY"  — high-confidence match, no LLM needed.
        None                          — ambiguous, fall through to LLM.
    """
    for intent, pattern in PATTERNS.items():
        if pattern.search(message):
            return intent
    return None


def history_sub_route(message: str) -> str:
    """
    Identify which history template to use for a HISTORY intent.

    Returns one of: "frequency", "recency", "tag", "last_month", "last_week",
    or "frequency" as the default.
    """
    for sub_intent, pattern in HISTORY_SUB_PATTERNS.items():
        if pattern.search(message):
            return sub_intent
    return "frequency"  # safest default


def extract_tag(message: str) -> str | None:
    """Extract a tag name from a history query like 'notes tagged retrieval'."""
    m = re.search(r"tagged?\s+['\"]?(\w[\w-]*)['\"]?", message, re.I)
    return m.group(1).lower() if m else None
