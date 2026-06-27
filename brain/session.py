"""
brain/session.py — Chat history buffer with sliding token window.

Uses tiktoken for accurate token counting.  When the token budget is exceeded,
the oldest turn pair is evicted from the live context and persisted to MongoDB.
No conversation content is ever lost.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import tiktoken

from brain import config, index

logger = logging.getLogger(__name__)

# Use the cl100k_base tokeniser (compatible with GPT-4 and Gemini approximation)
_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = tiktoken.get_encoding("gpt2")
    return _ENCODER


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def _message_tokens(msg: dict) -> int:
    return _count_tokens(msg.get("content", "")) + 4  # role + overhead


class SessionManager:
    """Manages a single user session: history, token budget, persistence."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id:    Optional[str] = None,
    ):
        self.session_id: str = session_id or str(uuid.uuid4())
        self.user_id:    str = user_id    or "default"
        self.history:    list[dict] = []   # {role, content, timestamp}
        self.token_count: int = 0
        self.turn_count:  int = 0
        self._started_at: str = _now()
        self._evicted:    list[dict] = []  # turns removed from live window

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        msg = {"role": "user", "content": content, "timestamp": _now()}
        self._append(msg)

    def add_assistant_message(self, content: str) -> None:
        msg = {"role": "assistant", "content": content, "timestamp": _now()}
        self._append(msg)
        self.turn_count += 1

    def get_messages_for_llm(self) -> list[dict]:
        """Return history in {role, content} format (no timestamp)."""
        return [{"role": m["role"], "content": m["content"]} for m in self.history]

    def get_transcript(self) -> str:
        """Full conversation as a readable string (for note generation)."""
        lines = []
        for m in self.history:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"**{role}:** {m['content']}")
        return "\n\n".join(lines)

    def save_to_db(self, note_id: Optional[str] = None) -> None:
        """Persist the full session (including evicted turns) to MongoDB."""
        session_doc = {
            "_id":        self.session_id,
            "user_id":    self.user_id,
            "started_at": self._started_at,
            "ended_at":   _now(),
            "turns":      self._evicted + self.history,
            "note_id":    note_id,
        }
        index.upsert_session(session_doc)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _append(self, msg: dict) -> None:
        tokens = _message_tokens(msg)
        budget = config.get("token_budget") or config.TOKEN_BUDGET

        # Evict oldest turn pairs until we fit
        while self.token_count + tokens > budget and len(self.history) >= 2:
            oldest_user      = self.history.pop(0)
            oldest_assistant = self.history.pop(0)
            self._evicted.append(oldest_user)
            self._evicted.append(oldest_assistant)
            evicted_tokens = _message_tokens(oldest_user) + _message_tokens(oldest_assistant)
            self.token_count = max(0, self.token_count - evicted_tokens)
            logger.debug("Evicted turn pair — token budget maintained")

        self.history.append(msg)
        self.token_count += tokens


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
