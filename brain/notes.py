"""
brain/notes.py — Note generation, .md writing, and indexing pipeline.

When SAVE fires:
1. LLM generates a structured NoteResponse from the conversation transcript.
2. The response is written as a Markdown file with YAML frontmatter.
3. MongoDB document is upserted.
4. Parent + child chunks are embedded into ChromaDB.
5. BM25 corpus is updated and re-persisted.
6. Access event is logged.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import frontmatter

from brain import config, index, orchestrator, profiles

logger = logging.getLogger(__name__)

# Lazy-load embedder to avoid startup cost
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedder


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_and_save(
    session,            # SessionManager
    user_id: str,
) -> dict:
    """
    Generate a structured note from the session transcript and persist everything.

    Returns a dict with: note_id, title, file_path, tags.
    """
    transcript = session.get_transcript()
    if not transcript.strip():
        return {"error": "Nothing to save — the session is empty."}

    # ── 1. LLM structured note generation ─────────────────────────────────────
    note_resp = orchestrator.generate_note_content(transcript)

    # ── 2. Build note document ─────────────────────────────────────────────────
    note_id   = str(uuid.uuid4())
    now_iso   = _now()
    slug      = _slugify(note_resp.title)
    date_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filename  = f"{date_str}-{slug}.md"
    file_path = config.NOTES_DIR / filename

    # ── 3. Write Markdown file with YAML frontmatter ──────────────────────────
    body = _build_body(note_resp)
    _write_markdown(file_path, note_id, note_resp, now_iso, body)

    # ── 4. Upsert MongoDB document ────────────────────────────────────────────
    note_doc = {
        "_id":          note_id,
        "user_id":      user_id,
        "title":        note_resp.title,
        "summary":      note_resp.summary,
        "tags":         note_resp.tags,
        "file_path":    str(file_path),
        "created_at":   now_iso,
        "modified_at":  now_iso,
        "session_turns": session.turn_count,
        "access_count": 0,
        "last_accessed": None,
    }
    index.upsert_note(note_doc)

    # ── 5. Embed parent + child chunks into ChromaDB ──────────────────────────
    embedder    = _get_embedder()
    parent_text = f"{note_resp.title}. {note_resp.summary}"
    child_texts, child_sections = _extract_child_chunks(note_resp)

    index.embed_note_chunks(
        note_id=note_id,
        parent_text=parent_text,
        child_texts=child_texts,
        child_sections=child_sections,
        embedder=embedder,
    )

    # ── 6. Update BM25 index ──────────────────────────────────────────────────
    bm25_text = _build_bm25_text(note_resp, body)
    index.add_to_bm25(note_id, bm25_text)

    # ── 7. Update user profile ────────────────────────────────────────────────
    profiles.increment_note_count(user_id)
    profiles.update_preferred_tags(user_id, note_resp.tags)

    # ── 8. Persist session with note_id reference ─────────────────────────────
    session.save_to_db(note_id=note_id)

    logger.info("Note saved: %s → %s", note_resp.title, file_path)
    return {
        "note_id":   note_id,
        "title":     note_resp.title,
        "file_path": str(file_path),
        "tags":      note_resp.tags,
    }


# ─── Markdown helpers ─────────────────────────────────────────────────────────

def _build_body(note_resp) -> str:
    """Build the Markdown body (without frontmatter)."""
    lines = [
        f"# {note_resp.title}",
        "",
        "## Core idea",
        note_resp.core_idea,
        "",
        "## Key insights",
    ]
    for insight in note_resp.key_insights:
        lines.append(f"- {insight}")

    if note_resp.open_questions:
        lines += ["", "## Open questions"]
        for q in note_resp.open_questions:
            lines.append(f"- {q}")

    if note_resp.threads:
        lines += ["", "## Threads to follow"]
        for t in note_resp.threads:
            lines.append(f"- {t}")

    return "\n".join(lines)


def _write_markdown(file_path: Path, note_id: str, note_resp, now_iso: str, body: str) -> None:
    """Write a Markdown file with YAML frontmatter."""
    post = frontmatter.Post(
        content=body,
        id=note_id,
        title=note_resp.title,
        created=now_iso,
        tags=note_resp.tags,
        summary=note_resp.summary,
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(frontmatter.dumps(post))


def load_note_body(note_id: str) -> str:
    """Load the full note body from disk."""
    note_doc = index.get_note_by_id(note_id)
    if not note_doc or not note_doc.get("file_path"):
        return ""
    try:
        with open(note_doc["file_path"], "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        return str(post.content)
    except Exception as exc:
        logger.error("Failed to load note body for %s: %s", note_id, exc)
        return ""


# ─── Chunking ─────────────────────────────────────────────────────────────────

def _extract_child_chunks(note_resp) -> tuple[list[str], list[str]]:
    """Extract individual child chunks (bullets/items) from a NoteResponse."""
    texts    = []
    sections = []

    for insight in note_resp.key_insights:
        texts.append(insight)
        sections.append("Key insights")

    for q in note_resp.open_questions:
        texts.append(q)
        sections.append("Open questions")

    for t in note_resp.threads:
        texts.append(t)
        sections.append("Threads to follow")

    if note_resp.core_idea:
        texts.insert(0, note_resp.core_idea)
        sections.insert(0, "Core idea")

    return texts, sections


def _build_bm25_text(note_resp, body: str) -> str:
    """Concatenate title + summary + full body for BM25 indexing."""
    return f"{note_resp.title} {note_resp.summary} {body}"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:60].rstrip("-")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
