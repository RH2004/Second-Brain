"""
brain/history.py — MongoDB aggregation analytics + template formatting.

Handles HISTORY intent queries.  For common patterns (frequency, recency,
tag, date window), the result is formatted with a template string — zero LLM
tokens.  Only genuinely complex queries fall through to LLM narration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from brain import index, orchestrator, router as router_mod

logger = logging.getLogger(__name__)

# ─── Template strings ─────────────────────────────────────────────────────────

HISTORY_TEMPLATES: dict[str, str] = {
    "frequency": "📊 You've revisited these topics most often:\n{items}",
    "recency":   "🕐 Your most recent notes:\n{items}",
    "tag":       "🏷️ Notes tagged **'{tag}'**:\n{items}",
    "last_month": "📅 Your activity in the past 30 days:\n{items}",
    "last_week":  "📅 Your activity in the past 7 days:\n{items}",
    "empty":      "You haven't saved any notes yet. Start a thinking session and save it!",
}


# ─── Main entry point ─────────────────────────────────────────────────────────

def surface_patterns(query: str, user_id: str) -> str:
    """
    Dispatch to the right aggregation based on query sub-pattern.
    Returns a formatted string (with or without LLM).
    """
    sub = router_mod.history_sub_route(query)
    tag = router_mod.extract_tag(query)

    if sub == "tag" and tag:
        return _tag_query(tag, user_id)
    elif sub == "recency":
        return _recency_query(user_id)
    elif sub == "last_week":
        return _date_window_query(user_id, days=7, template_key="last_week")
    elif sub == "last_month":
        return _date_window_query(user_id, days=30, template_key="last_month")
    else:
        return _frequency_query(user_id, query)


# ─── Aggregation queries ──────────────────────────────────────────────────────

def _frequency_query(user_id: str, original_query: str) -> str:
    """Most accessed notes (by access_count)."""
    db = index.get_db()
    pipeline = [
        {"$match":  {"user_id": user_id}},
        {"$sort":   {"access_count": -1, "last_accessed": -1}},
        {"$limit":  10},
        {"$project": {"title": 1, "access_count": 1, "tags": 1, "created_at": 1}},
    ]
    results = list(db.notes.aggregate(pipeline))

    if not results:
        return HISTORY_TEMPLATES["empty"]

    # Try to also query access_log for richer frequency data
    log_pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$note_id", "count": {"$sum": 1}, "last": {"$max": "$accessed_at"}}},
        {"$sort":  {"count": -1}},
        {"$limit": 10},
    ]
    log_results = list(db.access_log.aggregate(log_pipeline))
    if log_results:
        # Merge access_log counts with note metadata
        note_ids   = [r["_id"] for r in log_results]
        note_map   = {n["_id"]: n for n in index.get_notes_by_ids(note_ids)}
        items = []
        for i, r in enumerate(log_results, 1):
            note = note_map.get(r["_id"])
            if note:
                items.append(f"{i}. **{note['title']}** — {r['count']} visits")
        if items:
            return HISTORY_TEMPLATES["frequency"].format(items="\n".join(items))

    # Fallback to note access_count field
    items = [
        f"{i}. **{n['title']}** — {n.get('access_count', 0)} visits"
        for i, n in enumerate(results, 1)
    ]
    return HISTORY_TEMPLATES["frequency"].format(items="\n".join(items))


def _recency_query(user_id: str) -> str:
    """Most recently saved notes."""
    db = index.get_db()
    results = list(
        db.notes.find({"user_id": user_id})
        .sort("created_at", -1)
        .limit(10)
    )

    if not results:
        return HISTORY_TEMPLATES["empty"]

    items = [
        f"{i}. **{n['title']}** — saved {n.get('created_at', '')[:10]}"
        + (f" [{', '.join(n.get('tags', [])[:3])}]" if n.get("tags") else "")
        for i, n in enumerate(results, 1)
    ]
    return HISTORY_TEMPLATES["recency"].format(items="\n".join(items))


def _tag_query(tag: str, user_id: str) -> str:
    """Notes matching a given tag."""
    db = index.get_db()
    results = list(
        db.notes.find({"user_id": user_id, "tags": tag})
        .sort("created_at", -1)
        .limit(10)
    )

    if not results:
        return f"No notes found with tag **'{tag}'**."

    items = [
        f"{i}. **{n['title']}** — {n.get('created_at', '')[:10]}"
        for i, n in enumerate(results, 1)
    ]
    return HISTORY_TEMPLATES["tag"].format(tag=tag, items="\n".join(items))


def _date_window_query(user_id: str, days: int, template_key: str) -> str:
    """Notes saved within the last N days, grouped by activity."""
    db    = index.get_db()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    results = list(
        db.notes.find({"user_id": user_id, "created_at": {"$gte": since}})
        .sort("created_at", -1)
    )

    if not results:
        period = f"{days} days"
        return f"No notes saved in the past {period}."

    # Also look at access_log for the same window
    log_results = list(
        db.access_log.aggregate([
            {"$match": {"user_id": user_id, "accessed_at": {"$gte": since}}},
            {"$group": {"_id": "$note_id", "count": {"$sum": 1}}},
            {"$sort":  {"count": -1}},
            {"$limit": 5},
        ])
    )

    saved_items = [
        f"• **{n['title']}** (saved {n['created_at'][:10]})"
        + (f" [{', '.join(n.get('tags', [])[:2])}]" if n.get("tags") else "")
        for n in results
    ]

    output = f"**Saved ({len(results)} note{'s' if len(results) != 1 else ''}):**\n"
    output += "\n".join(saved_items)

    if log_results:
        note_ids = [r["_id"] for r in log_results]
        note_map = {n["_id"]: n for n in index.get_notes_by_ids(note_ids)}
        accessed_items = [
            f"• **{note_map[r['_id']]['title']}** — revisited {r['count']}×"
            for r in log_results
            if r["_id"] in note_map
        ]
        if accessed_items:
            output += "\n\n**Most revisited:**\n" + "\n".join(accessed_items)

    return HISTORY_TEMPLATES[template_key].format(items=output)


# ─── Summary stats (used by Streamlit sidebar) ────────────────────────────────

def get_summary_stats(user_id: str) -> dict:
    """Return quick stats: total notes, sessions, most used tag."""
    db = index.get_db()
    total_notes    = db.notes.count_documents({"user_id": user_id})
    total_sessions = db.sessions.count_documents({"user_id": user_id})

    # Most common tag
    tag_pipeline = [
        {"$match":   {"user_id": user_id}},
        {"$unwind":  "$tags"},
        {"$group":   {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort":    {"count": -1}},
        {"$limit":   1},
    ]
    top_tag_res = list(db.notes.aggregate(tag_pipeline))
    top_tag     = top_tag_res[0]["_id"] if top_tag_res else "—"

    return {
        "total_notes":    total_notes,
        "total_sessions": total_sessions,
        "top_tag":        top_tag,
    }
