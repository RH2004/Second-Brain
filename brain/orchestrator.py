"""
brain/orchestrator.py — All LLM calls, via litellm + instructor.

This module is the only place in the codebase that makes LLM API calls.
All structured outputs are enforced with instructor + Pydantic schemas.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import instructor
import litellm
from pydantic import BaseModel, Field

from brain import config

logger = logging.getLogger(__name__)

# Silence noisy litellm logs unless debug level requested
litellm.suppress_debug_info = True
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Patch litellm with instructor
_instructor_client = instructor.from_litellm(litellm.completion)


def _model() -> str:
    return config.get("model") or config.MODEL


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class IntentResponse(BaseModel):
    intent:      Literal["THINK", "SAVE", "FIND", "HISTORY"]
    confidence:  float = Field(ge=0.0, le=1.0)
    save_signal: bool  = False
    query_text:  Optional[str] = None
    reply:       str   = ""      # populated only for THINK intent


class NoteResponse(BaseModel):
    title:         str
    summary:       str   = Field(description="One sentence capturing the core idea — the retrieval anchor")
    tags:          list[str]
    core_idea:     str   = Field(description="The central insight worked through in this session")
    key_insights:  list[str]
    open_questions: list[str]
    threads:       list[str] = Field(default_factory=list, description="Related ideas worth exploring")


# ─── Intent classification (LLM fallback) ─────────────────────────────────────

_INTENT_SYSTEM = (
    "You are an intent classifier for a personal knowledge management tool called Second Brain. "
    "Classify the user's message into one of: THINK, SAVE, FIND, HISTORY.\n"
    "THINK: free reasoning, exploring ideas — respond with a helpful reply.\n"
    "SAVE:  user wants to save / file the current session as a note.\n"
    "FIND:  user wants to retrieve / search past notes.\n"
    "HISTORY: user asks about usage patterns, frequency, recency.\n"
    "Be conservative — only classify as SAVE/FIND/HISTORY when clearly intended."
)


def classify_intent(messages: list[dict], user_message: str) -> IntentResponse:
    """LLM fallback for ambiguous intent classification.  Returns IntentResponse."""
    payload = messages + [{"role": "user", "content": user_message}]

    try:
        return _instructor_client.chat.completions.create(
            model=_model(),
            messages=[{"role": "system", "content": _INTENT_SYSTEM}] + payload[-6:],
            response_model=IntentResponse,
            max_tokens=400,
        )
    except Exception as exc:
        logger.error("Intent classification failed: %s", exc)
        return IntentResponse(intent="THINK", confidence=0.5, reply="Let's keep thinking.")


# ─── THINK — free reasoning reply ─────────────────────────────────────────────

def think_reply(messages: list[dict]) -> str:
    """Generate a free-form THINK reply from the full conversation context."""
    try:
        resp = litellm.completion(
            model=_model(),
            messages=[{"role": "system", "content": config.SYSTEM_PROMPT}] + messages,
            max_tokens=800,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error("THINK reply failed: %s", exc)
        return "I'm having trouble connecting to the LLM. Please check your API key."


# ─── SAVE — structured note generation ────────────────────────────────────────

_NOTE_SYSTEM = (
    "You are a structured knowledge extraction engine. "
    "Given a thinking session conversation, generate a concise, retrievable Markdown note. "
    "The summary must be a single sentence that would act as a retrieval anchor. "
    "Tags should be lowercase slug-style (e.g., 'retrieval', 'system-design'). "
    "Key insights should be specific and self-contained — not vague summaries. "
    "Minimum 3 key insights even for short sessions. "
    "The title should be meaningful and specific, not generic."
)


def generate_note_content(transcript: str) -> NoteResponse:
    """Generate a structured note from a conversation transcript."""
    messages = [
        {"role": "system", "content": _NOTE_SYSTEM},
        {"role": "user",   "content": f"Generate a note from this thinking session:\n\n{transcript}"},
    ]
    try:
        return _instructor_client.chat.completions.create(
            model=_model(),
            messages=messages,
            response_model=NoteResponse,
            max_tokens=1200,
        )
    except Exception as exc:
        logger.error("Note generation failed: %s", exc)
        # Return a minimal fallback note
        return NoteResponse(
            title="Session Note",
            summary="A thinking session was recorded.",
            tags=["general"],
            core_idea="Ideas were explored in this session.",
            key_insights=["Session content not fully captured due to an error."],
            open_questions=[],
            threads=[],
        )


# ─── FIND — answer synthesis from retrieved notes ─────────────────────────────

_RAG_SYSTEM = (
    "You are Second Brain. Answer the user's question using ONLY the retrieved note excerpts provided. "
    "Cite the note title in your answer. "
    "If the notes do not contain a relevant answer, say so honestly — never invent. "
    "Be concise and direct."
)


def synthesise_answer(query: str, notes: list[dict]) -> str:
    """Synthesize an answer from retrieved and confirmed notes."""
    notes_text = "\n\n---\n\n".join(
        f"Note: **{n['title']}** (saved {n.get('created_at', '')[:10]})\n\n{n.get('body', '')}"
        for n in notes
    )
    messages = [
        {"role": "system",  "content": _RAG_SYSTEM},
        {"role": "user",    "content": f"Question: {query}\n\n{notes_text}"},
    ]
    try:
        resp = litellm.completion(model=_model(), messages=messages, max_tokens=800)
        return resp.choices[0].message.content or "No answer found."
    except Exception as exc:
        logger.error("Answer synthesis failed: %s", exc)
        return "I couldn't synthesise an answer. Please check your API key."


# ─── HISTORY — complex query narration ────────────────────────────────────────

_HISTORY_SYSTEM = (
    "You are Second Brain's history narrator. "
    "Given a MongoDB result set about the user's thinking patterns, "
    "narrate it in clear, friendly plain English. "
    "Be succinct — 2–4 sentences maximum."
)


def narrate_history(query: str, result_data: str) -> str:
    """Narrate a complex history result via LLM (fallback for non-template queries)."""
    messages = [
        {"role": "system", "content": _HISTORY_SYSTEM},
        {"role": "user",   "content": f"Query: {query}\n\nData:\n{result_data}"},
    ]
    try:
        resp = litellm.completion(model=_model(), messages=messages, max_tokens=300)
        return resp.choices[0].message.content or result_data
    except Exception as exc:
        logger.error("History narration failed: %s", exc)
        return result_data
