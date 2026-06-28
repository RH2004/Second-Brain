"""
brain/index.py — All storage I/O: MongoDB, ChromaDB, BM25.

This is the single layer that touches disk / databases.  Nothing outside this
module should import pymongo, chromadb, or rank_bm25 directly.

MongoDB fallback: if a real mongod is not running, the module automatically
falls back to mongomock (in-memory), so the app still works on machines that
have not installed MongoDB.
"""

from __future__ import annotations

import os
import pickle
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain import config

logger = logging.getLogger(__name__)

# ─── MongoDB ──────────────────────────────────────────────────────────────────
_mongo_client = None
_db           = None
_using_mock   = False


def _init_mongo():
    global _mongo_client, _db, _using_mock
    if _db is not None:
        return

    try:
        import pymongo
        client = pymongo.MongoClient(
            config.MONGO_URI,
            serverSelectionTimeoutMS=2000,
        )
        client.admin.command("ping")          # fast connection test
        _mongo_client = client
        _using_mock   = False
        logger.info("Connected to real MongoDB at %s", config.MONGO_URI)
    except Exception as exc:
        logger.warning("MongoDB unavailable (%s) — using mongomock fallback", exc)
        import mongomock
        _mongo_client = mongomock.MongoClient()
        _using_mock   = True

    _db = _mongo_client[config.MONGO_DB]
    _create_indexes()


def _create_indexes():
    try:
        _db.notes.create_index([("user_id", 1), ("tags", 1)])
        _db.notes.create_index([("user_id", 1), ("last_accessed", -1)])
        _db.access_log.create_index([("user_id", 1), ("accessed_at", -1)])
        _db.access_log.create_index([("note_id", 1)])
    except Exception:
        pass  # mongomock may not support all index types


def get_db():
    _init_mongo()
    return _db


def is_using_mock() -> bool:
    _init_mongo()
    return _using_mock


# ─── ChromaDB ─────────────────────────────────────────────────────────────────
_chroma_client     = None
_parent_collection = None
_child_collection  = None


def _init_chroma():
    global _chroma_client, _parent_collection, _child_collection
    if _chroma_client is not None:
        return

    import chromadb
    _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    _parent_collection = _chroma_client.get_or_create_collection(
        name="note_parent_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    _child_collection = _chroma_client.get_or_create_collection(
        name="note_child_chunks",
        metadata={"hnsw:space": "cosine"},
    )


def get_parent_collection():
    _init_chroma()
    return _parent_collection


def get_child_collection():
    _init_chroma()
    return _child_collection


# ─── BM25 persistence ─────────────────────────────────────────────────────────
_bm25_obj    = None      # BM25Okapi instance
_bm25_corpus = []        # list of tokenised document lists
_note_ids_bm25: list[str] = []  # parallel list of note IDs


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def load_bm25() -> None:
    """Load persisted BM25 index from disk (called lazily on first FIND)."""
    global _bm25_obj, _bm25_corpus, _note_ids_bm25

    if _bm25_obj is not None:
        return

    path = config.BM25_INDEX_PATH
    if path.exists():
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            _bm25_corpus    = data["corpus"]
            _note_ids_bm25  = data["note_ids"]
            from rank_bm25 import BM25Okapi
            _bm25_obj = BM25Okapi(_bm25_corpus)
            logger.info("Loaded BM25 index (%d docs)", len(_bm25_corpus))
        except Exception as exc:
            logger.warning("Failed to load BM25 index: %s — starting fresh", exc)
            _bm25_corpus   = []
            _note_ids_bm25 = []
            _bm25_obj      = None
    else:
        _bm25_corpus   = []
        _note_ids_bm25 = []
        _bm25_obj      = None


def add_to_bm25(note_id: str, text: str) -> None:
    """Append a note to the BM25 corpus and refit, then persist."""
    global _bm25_obj, _bm25_corpus, _note_ids_bm25

    load_bm25()

    tokens = _tokenize(text)
    _bm25_corpus.append(tokens)
    _note_ids_bm25.append(note_id)

    from rank_bm25 import BM25Okapi
    _bm25_obj = BM25Okapi(_bm25_corpus)

    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"corpus": _bm25_corpus, "note_ids": _note_ids_bm25}, f)

    logger.info("BM25 index updated — %d docs total", len(_bm25_corpus))


def bm25_query(query: str, top_k: int = 10) -> list[dict]:
    """Return top_k note IDs and scores from BM25."""
    load_bm25()

    if _bm25_obj is None or not _note_ids_bm25:
        return []

    tokens = _tokenize(query)
    scores = _bm25_obj.get_scores(tokens)
    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    results = []
    for idx, score in ranked:
        if score >= 0:  # include near-zero scores; rank order still meaningful
            results.append({"note_id": _note_ids_bm25[idx], "score": float(score)})
    return results


# ─── Note CRUD ────────────────────────────────────────────────────────────────

def upsert_note(note: dict) -> str:
    """Insert or update a note document.  Returns the note _id."""
    db = get_db()
    note_id = note.get("_id") or str(uuid.uuid4())
    note["_id"] = note_id
    note.setdefault("access_count", 0)
    note.setdefault("last_accessed", None)

    db.notes.replace_one({"_id": note_id}, note, upsert=True)
    return note_id


def get_note_by_id(note_id: str) -> dict | None:
    db = get_db()
    return db.notes.find_one({"_id": note_id})


def get_recent_notes(limit: int = 5, user_id: str | None = None) -> list[dict]:
    db = get_db()
    filt = {}
    if user_id:
        filt["user_id"] = user_id
    cursor = db.notes.find(filt).sort("created_at", -1).limit(limit)
    return list(cursor)


def get_notes_by_ids(note_ids: list[str]) -> list[dict]:
    db = get_db()
    return list(db.notes.find({"_id": {"$in": note_ids}}))


# ─── Session CRUD ─────────────────────────────────────────────────────────────

def upsert_session(session: dict) -> str:
    db = get_db()
    session_id = session.get("_id") or str(uuid.uuid4())
    session["_id"] = session_id
    db.sessions.replace_one({"_id": session_id}, session, upsert=True)
    return session_id


def get_session(session_id: str) -> dict | None:
    db = get_db()
    return db.sessions.find_one({"_id": session_id})


# ─── Access log ───────────────────────────────────────────────────────────────

def log_access(user_id: str, note_id: str, query: str, intent: str) -> None:
    db = get_db()
    db.access_log.insert_one({
        "user_id":     user_id,
        "note_id":     note_id,
        "accessed_at": _now(),
        "query":       query,
        "intent":      intent,
    })
    # Increment access_count on the note
    db.notes.update_one(
        {"_id": note_id},
        {"$inc": {"access_count": 1}, "$set": {"last_accessed": _now()}},
    )


# ─── ChromaDB note embedding ──────────────────────────────────────────────────

def embed_note_chunks(
    note_id: str,
    parent_text: str,
    child_texts: list[str],
    child_sections: list[str],
    embedder,            # SentenceTransformer instance passed in to avoid circular imports
) -> None:
    """Embed parent and child chunks and store in ChromaDB."""
    parent_col = get_parent_collection()
    child_col  = get_child_collection()

    # Parent chunk
    parent_emb = embedder.encode([parent_text])[0].tolist()
    parent_col.upsert(
        ids=[note_id],
        embeddings=[parent_emb],
        documents=[parent_text],
        metadatas=[{"note_id": note_id}],
    )

    # Child chunks
    if child_texts:
        child_ids  = [f"{note_id}__chunk_{i}" for i in range(len(child_texts))]
        child_embs = embedder.encode(child_texts).tolist()
        child_metas = [
            {"parent_id": note_id, "section": child_sections[i], "chunk_index": i}
            for i in range(len(child_texts))
        ]
        child_col.upsert(
            ids=child_ids,
            embeddings=child_embs,
            documents=child_texts,
            metadatas=child_metas,
        )


def dense_search(query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """Search child chunks; resolve to unique parent note IDs."""
    child_col = get_child_collection()

    if child_col.count() == 0:
        return []

    results = child_col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, child_col.count()),
        include=["metadatas", "distances"],
    )

    seen: set[str] = set()
    ranked: list[dict] = []
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for meta, dist in zip(metadatas, distances):
        meta = meta or {}
        parent_id = meta.get("parent_id", "")
        if parent_id and parent_id not in seen:
            seen.add(parent_id)
            ranked.append({"note_id": parent_id, "score": 1.0 - dist})

    return ranked


def dense_parent_search(query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """Search parent chunks (used for proactive surfacing)."""
    parent_col = get_parent_collection()

    if parent_col.count() == 0:
        return []

    results = parent_col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, parent_col.count()),
        include=["metadatas", "distances", "documents"],
    )

    ranked: list[dict] = []
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]

    for meta, dist, doc in zip(metadatas, distances, documents):
        meta = meta or {}
        ranked.append({
            "note_id": meta.get("note_id", ""),
            "score":   1.0 - dist,
            "text":    doc,
        })
    return ranked


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
