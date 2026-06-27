"""
brain/profiles.py — User profile CRUD and personalisation scoring.

User profiles live in MongoDB collection `users`.  Domain weights bias
retrieval ranking toward the user's known expertise areas.

Domain weight updates fire at session end, incrementing weights for tags
found in accessed notes by a small delta and normalising to [0, 1].
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from brain import index

logger = logging.getLogger(__name__)


# ─── Profile CRUD ─────────────────────────────────────────────────────────────

def load_or_create_user(username: str = "default") -> str:
    """
    Load an existing user by username or create a new profile.
    Returns the user_id (str UUID).
    """
    db = index.get_db()
    user = db.users.find_one({"username": username})
    if user:
        return str(user["_id"])

    user_id = str(uuid.uuid4())
    db.users.insert_one({
        "_id":           user_id,
        "username":      username,
        "created_at":    _now(),
        "domain_weights": {},
        "preferred_tags": [],
        "session_count":  0,
        "note_count":     0,
        "last_active":    _now(),
    })
    logger.info("Created new user profile: %s (%s)", username, user_id)
    return user_id


def get_profile(user_id: str) -> dict | None:
    db = index.get_db()
    return db.users.find_one({"_id": user_id})


def increment_session_count(user_id: str) -> None:
    db = index.get_db()
    db.users.update_one(
        {"_id": user_id},
        {"$inc": {"session_count": 1}, "$set": {"last_active": _now()}},
    )


def increment_note_count(user_id: str) -> None:
    db = index.get_db()
    db.users.update_one(
        {"_id": user_id},
        {"$inc": {"note_count": 1}},
    )


def update_preferred_tags(user_id: str, tags: list[str]) -> None:
    """Add tags from a newly saved note to the user's preferred_tags list."""
    db = index.get_db()
    db.users.update_one(
        {"_id": user_id},
        {"$addToSet": {"preferred_tags": {"$each": tags}}},
    )


def update_domain_weights(user_id: str, accessed_note_ids: list[str]) -> None:
    """
    Increment domain weights for tags of accessed notes.
    Called at session end.
    """
    if not accessed_note_ids:
        return

    db = index.get_db()
    for note_id in accessed_note_ids:
        note = db.notes.find_one({"_id": note_id})
        if not note:
            continue
        for tag in note.get("tags", []):
            db.users.update_one(
                {"_id": user_id},
                {"$inc": {f"domain_weights.{tag}": 0.01}},
            )

    # Normalise weights to [0, 1]
    profile = db.users.find_one({"_id": user_id})
    if not profile:
        return
    weights = profile.get("domain_weights", {})
    if weights:
        max_w = max(weights.values()) or 1.0
        normalised = {k: round(v / max_w, 4) for k, v in weights.items()}
        db.users.update_one(
            {"_id": user_id},
            {"$set": {"domain_weights": normalised}},
        )


# ─── Personalisation scoring ───────────────────────────────────────────────────

def personalized_score(
    base_score: float,
    note_tags:  list[str],
    user_profile: dict | None,
) -> float:
    """
    Apply small domain-weight and preferred-tag boosts to a base relevance score.

    Boosts are intentionally small so they nudge ranking without overriding
    genuine relevance.
    """
    if not user_profile:
        return base_score

    domain_weights  = user_profile.get("domain_weights", {})
    preferred_tags  = set(user_profile.get("preferred_tags", []))

    domain_boost = max(
        (domain_weights.get(tag, 0.0) for tag in note_tags),
        default=0.0,
    )
    preferred_boost = sum(
        0.05 for tag in note_tags if tag in preferred_tags
    )
    return base_score + (0.1 * domain_boost) + preferred_boost


# ─── Helper ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
