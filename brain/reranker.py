"""
brain/reranker.py — Local CrossEncoder wrapper.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to score query-document pairs.
Fully local, ~30ms per batch, zero API cost, trained for relevance judgment.

The model is loaded lazily on first call to avoid startup latency.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from brain import config

logger = logging.getLogger(__name__)

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder model: %s", config.RERANKER_MODEL)
        _reranker = CrossEncoder(config.RERANKER_MODEL)
        logger.info("Cross-encoder loaded.")
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    threshold: float | None = None,
) -> list[dict]:
    """
    Score each candidate against the query.

    Args:
        query:       The user's search query.
        candidates:  List of note dicts, each with at least 'title' and 'summary'.
        threshold:   Minimum score (0–1) to keep.  Defaults to config.RERANK_THRESHOLD.

    Returns:
        Filtered and sorted list of candidates with 'rerank_score' added.
    """
    if not candidates:
        return []

    threshold = threshold if threshold is not None else config.RERANK_THRESHOLD
    reranker  = _get_reranker()

    pairs  = [(query, f"{c['title']}. {c.get('summary', '')}") for c in candidates]
    scores = reranker.predict(pairs)

    scored = []
    for candidate, score in zip(candidates, scores):
        candidate = dict(candidate)
        candidate["rerank_score"] = float(score)
        if float(score) >= threshold:
            scored.append(candidate)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    logger.debug(
        "Reranked %d candidates → %d confirmed (threshold=%.2f)",
        len(candidates), len(scored), threshold,
    )
    return scored
