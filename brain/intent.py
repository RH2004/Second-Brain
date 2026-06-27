"""
brain/intent.py — Main message dispatch: fast-path → actions; LLM fallback.

This is the central routing layer that every user message passes through.
It returns a response dict with at minimum a 'reply' key.

Response dict shape:
    {
        "intent":  "THINK" | "SAVE" | "FIND" | "HISTORY",
        "reply":   str,                         # text to show user
        "sources": list[dict] | None,           # FIND: source notes
        "note":    dict | None,                 # SAVE: saved note info
        "routed_by": "regex" | "llm",           # for diagnostics
    }
"""

from __future__ import annotations

import logging

from brain import history, notes as notes_mod, orchestrator, retrieval, router

logger = logging.getLogger(__name__)


def route(
    message:  str,
    session,              # SessionManager
    user_id:  str,
) -> dict:
    """
    Route a user message to the appropriate action handler.

    1. Try regex fast-path (zero LLM tokens for clear intents).
    2. If ambiguous, call LLM to classify.
    3. Execute the action and return a response dict.
    """
    # ── Add user message to session history ───────────────────────────────────
    session.add_user_message(message)

    # ── 1. Regex fast-path ────────────────────────────────────────────────────
    intent    = router.fast_route(message)
    routed_by = "regex" if intent else "llm"

    # ── 2. LLM fallback for ambiguous messages ────────────────────────────────
    if intent is None:
        llm_resp = orchestrator.classify_intent(
            session.get_messages_for_llm()[:-1],  # history before current msg
            message,
        )
        intent = llm_resp.intent

        # If LLM already provided a THINK reply, use it directly
        if intent == "THINK" and llm_resp.reply:
            session.add_assistant_message(llm_resp.reply)
            return {
                "intent":    "THINK",
                "reply":     llm_resp.reply,
                "sources":   None,
                "note":      None,
                "routed_by": routed_by,
            }

    logger.info("Intent: %s (via %s) — message: %.60s…", intent, routed_by, message)

    # ── 3. Dispatch to action ─────────────────────────────────────────────────
    if intent == "SAVE":
        return _handle_save(session, user_id, routed_by)

    elif intent == "FIND":
        return _handle_find(message, session, user_id, routed_by)

    elif intent == "HISTORY":
        return _handle_history(message, session, user_id, routed_by)

    else:  # THINK
        return _handle_think(session, routed_by)


# ─── Action handlers ──────────────────────────────────────────────────────────

def _handle_think(session, routed_by: str) -> dict:
    """Generate a free-form THINK reply from full session context."""
    reply = orchestrator.think_reply(session.get_messages_for_llm())
    session.add_assistant_message(reply)
    return {
        "intent":    "THINK",
        "reply":     reply,
        "sources":   None,
        "note":      None,
        "routed_by": routed_by,
    }


def _handle_save(session, user_id: str, routed_by: str) -> dict:
    """Generate and persist a structured note from the session."""
    result = notes_mod.generate_and_save(session, user_id)

    if "error" in result:
        reply = f"⚠️ {result['error']}"
    else:
        tags_str = ", ".join(result.get("tags", []))
        reply = (
            f"✅ **Note saved:** {result['title']}\n\n"
            f"🏷️ Tags: {tags_str}\n"
            f"📄 File: `{result.get('file_path', '')}`"
        )

    session.add_assistant_message(reply)
    return {
        "intent":    "SAVE",
        "reply":     reply,
        "sources":   None,
        "note":      result if "error" not in result else None,
        "routed_by": routed_by,
    }


def _handle_find(message: str, session, user_id: str, routed_by: str) -> dict:
    """Run the full hybrid retrieval pipeline and synthesise an answer."""
    result = retrieval.find(query=message, user_id=user_id)

    if result["found"]:
        sources_text = "\n\n".join(
            f"📄 **{s['title']}** (score: {s['score']:.2f})"
            for s in result["sources"]
        )
        reply = f"{result['answer']}\n\n---\n**Sources:**\n{sources_text}"
    else:
        reply = result["answer"]

    session.add_assistant_message(reply)
    return {
        "intent":    "FIND",
        "reply":     reply,
        "sources":   result.get("sources"),
        "note":      None,
        "routed_by": routed_by,
    }


def _handle_history(message: str, session, user_id: str, routed_by: str) -> dict:
    """Run MongoDB history analytics and format the result."""
    reply = history.surface_patterns(query=message, user_id=user_id)
    session.add_assistant_message(reply)
    return {
        "intent":    "HISTORY",
        "reply":     reply,
        "sources":   None,
        "note":      None,
        "routed_by": routed_by,
    }
