"""
brain/retrieval.py — Full hybrid retrieval pipeline.

Stages:
    1. Dense search  (ChromaDB child chunks → parent resolve)
       + Sparse search (BM25)  ← run in parallel
    2. Reciprocal Rank Fusion (RRF, k=60) → top 5
    3. Cross-encoder reranking → filter by threshold
    4. Personalisation re-weighting
    5. Load full note text for confirmed hits
    6. Log access events
    7. LLM answer synthesis with citations

If no candidates pass the cross-encoder threshold, return an honest "nothing
found" message — no hallucination.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from brain import config, index, notes as notes_mod, orchestrator, profiles, reranker

logger = logging.getLogger(__name__)

# Lazy embedder
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model for retrieval: %s", config.EMBEDDING_MODEL)
        try:
            _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        except Exception as exc:
            logger.warning("Failed loading online model, retrying with local_files_only=True: %s", exc)
            _embedder = SentenceTransformer(config.EMBEDDING_MODEL, local_files_only=True)
    return _embedder


# ─── Main entry point ─────────────────────────────────────────────────────────

def find(
    query:   str,
    user_id: str,
) -> dict:
    """
    Run the full retrieval pipeline.

    Returns dict with:
        found (bool), answer (str), sources (list[dict])
    """
    # ── Stage 1: Dense + Sparse in parallel ───────────────────────────────────
    embedder = _get_embedder()
    query_emb = embedder.encode([query])[0].tolist()

    dense_results  = []
    sparse_results = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_dense  = pool.submit(index.dense_search,  query_emb, config.DENSE_TOP_K)
        fut_sparse = pool.submit(index.bm25_query,    query,     config.SPARSE_TOP_K)
        for fut in as_completed([fut_dense, fut_sparse]):
            if fut is fut_dense:
                dense_results  = fut.result()
            else:
                sparse_results = fut.result()

    logger.debug("Dense: %d, Sparse: %d candidates", len(dense_results), len(sparse_results))

    if not dense_results and not sparse_results:
        return {
            "found":   False,
            "answer":  "I don't have any notes yet. Start a thinking session and save it!",
            "sources": [],
        }

    # ── Stage 2: RRF merge ────────────────────────────────────────────────────
    merged = _rrf_merge(dense_results, sparse_results, top_n=config.RERANK_TOP_N)
    logger.debug("RRF merged: %d candidates", len(merged))

    if not merged:
        return {
            "found":   False,
            "answer":  "No relevant notes found for your query.",
            "sources": [],
        }

    # ── Load note metadata for reranking ──────────────────────────────────────
    note_ids = [r["note_id"] for r in merged]
    note_docs = {n["_id"]: n for n in index.get_notes_by_ids(note_ids)}

    # Attach metadata to merged results
    candidates = []
    for r in merged:
        doc = note_docs.get(r["note_id"])
        if doc:
            c = dict(doc)
            c["rrf_score"] = r["score"]
            candidates.append(c)

    # ── Stage 3: Cross-encoder rerank ─────────────────────────────────────────
    confirmed = reranker.rerank(query, candidates)
    logger.debug("Cross-encoder confirmed: %d notes", len(confirmed))

    if not confirmed:
        return {
            "found":   False,
            "answer":  (
                "I searched my notes but couldn't find anything that closely matches "
                f"your question about \"{query}\". "
                "This topic may not have been saved yet."
            ),
            "sources": [],
        }

    # ── Stage 4: Personalisation re-weighting ─────────────────────────────────
    user_profile = profiles.get_profile(user_id)
    for c in confirmed:
        c["final_score"] = profiles.personalized_score(
            c["rerank_score"], c.get("tags", []), user_profile
        )
    confirmed.sort(key=lambda x: x["final_score"], reverse=True)

    # ── Stage 5: Load full note bodies ────────────────────────────────────────
    for c in confirmed:
        c["body"] = notes_mod.load_note_body(c["_id"])

    # ── Stage 6: Log access events ────────────────────────────────────────────
    for c in confirmed:
        index.log_access(
            user_id=user_id,
            note_id=c["_id"],
            query=query,
            intent="FIND",
        )

    # ── Stage 7: LLM answer synthesis ─────────────────────────────────────────
    answer = orchestrator.synthesise_answer(query, confirmed[:3])

    sources = [
        {
            "note_id":    c["_id"],
            "title":      c["title"],
            "created_at": c.get("created_at", ""),
            "tags":       c.get("tags", []),
            "body":       c.get("body", ""),
            "score":      round(c.get("rerank_score", 0.0), 3),
        }
        for c in confirmed
    ]

    return {
        "found":   True,
        "answer":  answer,
        "sources": sources,
    }


# ─── RRF implementation ───────────────────────────────────────────────────────

def _rrf_merge(
    dense:  list[dict],   # [{note_id, score}]
    sparse: list[dict],   # [{note_id, score}]
    top_n:  int = 5,
    k:      int = None,
) -> list[dict]:
    """Reciprocal Rank Fusion: merges two ranked lists without score normalisation."""
    k = k or config.RRF_K

    scores: dict[str, float] = {}

    for rank, item in enumerate(dense):
        nid = item["note_id"]
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)

    for rank, item in enumerate(sparse):
        nid = item["note_id"]
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"note_id": nid, "score": score} for nid, score in ranked]
