"""
brain/proactive.py — Async background similarity check.

Every N turns during a THINK session, this module spawns a background task
that:
    1. Embeds the last N messages.
    2. Queries ChromaDB parent chunks for cosine similarity.
    3. If any note exceeds the proactive threshold, enqueues it for display.

In the TUI this runs as a real asyncio task.
In Streamlit it is called synchronously every N turns (acceptable at ~30ms).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from brain import config, index

logger = logging.getLogger(__name__)

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


# ─── Synchronous check (Streamlit + TUI both use this) ───────────────────────

def check_proactive(
    recent_messages: list[dict],
    exclude_note_ids: set[str] | None = None,
    threshold: float | None = None,
) -> dict | None:
    """
    Check whether any past note is highly similar to the recent conversation.

    Args:
        recent_messages:   The last N {role, content} dicts from session history.
        exclude_note_ids:  Note IDs already shown (avoid repeating suggestions).
        threshold:         Override the config threshold.

    Returns:
        A dict {note_id, title, summary, age, score} if a match is found, else None.
    """
    if not recent_messages:
        return None

    threshold = threshold if threshold is not None else (
        config.get("proactive_threshold") or config.PROACTIVE_THRESHOLD
    )
    exclude = exclude_note_ids or set()

    # Build context string from last messages
    context = " ".join(
        m["content"] for m in recent_messages[-4:]
        if m.get("role") == "user"
    )
    if not context.strip():
        return None

    embedder  = _get_embedder()
    query_emb = embedder.encode([context])[0].tolist()

    results = index.dense_parent_search(query_emb, top_k=5)

    for r in results:
        note_id = r.get("note_id", "")
        score   = r.get("score", 0.0)
        if score >= threshold and note_id not in exclude:
            note = index.get_note_by_id(note_id)
            if note:
                return {
                    "note_id": note_id,
                    "title":   note.get("title", "Untitled"),
                    "summary": note.get("summary", ""),
                    "age":     _human_age(note.get("created_at", "")),
                    "score":   score,
                }
    return None


# ─── Async wrapper (for TUI asyncio loop) ────────────────────────────────────

async def async_check_proactive(
    recent_messages:  list[dict],
    queue:            asyncio.Queue,
    exclude_note_ids: set[str] | None = None,
) -> None:
    """
    Non-blocking async wrapper — runs the check in a thread pool executor
    so it never blocks the event loop.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: check_proactive(recent_messages, exclude_note_ids),
    )
    if result:
        await queue.put(result)
        logger.debug("Proactive suggestion enqueued: %s", result["title"])


# ─── Helper ───────────────────────────────────────────────────────────────────

def _human_age(iso_str: str) -> str:
    """Convert ISO timestamp to human-readable age string."""
    if not iso_str:
        return "some time ago"
    try:
        dt   = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now  = datetime.now(timezone.utc)
        days = (now - dt).days
        if days == 0:
            return "today"
        elif days == 1:
            return "yesterday"
        elif days < 7:
            return f"{days} days ago"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif days < 365:
            months = days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        else:
            years = days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
    except Exception:
        return "some time ago"
