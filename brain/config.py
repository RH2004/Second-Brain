"""
brain/config.py — Single source of truth for all settings.

Override any value via environment variables or the Streamlit settings panel.
"""

import os
from pathlib import Path

# ─── Repository root ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()

# ─── Storage paths ────────────────────────────────────────────────────────────
NOTES_DIR       = ROOT / "storage" / "notes"
CHROMA_DIR      = ROOT / "storage" / "chroma"
BM25_INDEX_PATH = ROOT / "storage" / "bm25_index.pkl"

# Ensure directories exist
NOTES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB  = os.getenv("MONGO_DB", "second_brain")

# ─── LLM ──────────────────────────────────────────────────────────────────────
# Swap model by changing this one line (or setting LLM_MODEL env var).
# Supported values:
#   "gemini/gemini-1.5-flash"   ← default, free tier
#   "gpt-4o"
#   "claude-sonnet-4-6"
#   "ollama/llama3"
MODEL = os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash")

# ─── Embedding + reranking ────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ─── Context / token budget ───────────────────────────────────────────────────
TOKEN_BUDGET: int = int(os.getenv("TOKEN_BUDGET", "80000"))

# ─── Retrieval knobs ──────────────────────────────────────────────────────────
DENSE_TOP_K:          int   = 10       # ChromaDB child chunks to fetch
SPARSE_TOP_K:         int   = 10       # BM25 candidates to fetch
RRF_K:                int   = 60       # RRF constant (k in 1/(k+rank))
RERANK_TOP_N:         int   = 5        # candidates fed to cross-encoder
RERANK_THRESHOLD:     float = float(os.getenv("RERANK_THRESHOLD", "-5.0"))   # ms-marco returns raw logits
PROACTIVE_THRESHOLD:  float = float(os.getenv("PROACTIVE_THRESHOLD", "0.78"))
PROACTIVE_EVERY_N:    int   = 4        # trigger proactive check every N turns

# ─── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Second Brain — a thoughtful reasoning partner. "
    "Help the user think through ideas, questions, and problems. "
    "Be concise, incisive, and build on what has been said. "
    "Never invent facts from past notes; only cite what has been retrieved."
)

# ─── Runtime config (mutable by Streamlit settings panel) ────────────────────
# These are module-level variables so the settings panel can update them at
# runtime without restarting the process.
_runtime: dict = {
    "model":               MODEL,
    "token_budget":        TOKEN_BUDGET,
    "proactive_threshold": PROACTIVE_THRESHOLD,
}


def update_config(**kwargs) -> None:
    """Update runtime configuration (called from Streamlit settings panel)."""
    global _runtime
    _runtime.update(kwargs)


def get(key: str):
    """Get a runtime-overridable config value."""
    return _runtime.get(key, globals().get(key.upper()))
