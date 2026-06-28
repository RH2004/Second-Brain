"""
app.py — Second Brain Streamlit Web Application.

Run with:
    streamlit run app.py

This file is the Streamlit entry point.  All core logic lives in brain/.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# ─── Page config (must be the FIRST Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="Second Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# Second Brain\nA local, model-agnostic thinking and retrieval tool.",
    },
)

# ─── Import brain modules AFTER page config ────────────────────────────────────
from brain import config as cfg, history as hist_engine, intent, profiles, proactive
from brain.session import SessionManager

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark gradient background */
  .stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1923 50%, #0a1628 100%);
    color: #e8eaf0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12192e 0%, #0d1520 100%);
    border-right: 1px solid rgba(124, 106, 247, 0.2);
  }
  [data-testid="stSidebar"] * {
    color: #c8ccd8 !important;
  }

  /* Chat messages */
  [data-testid="stChatMessageContent"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: rgba(124, 106, 247, 0.12) !important;
    border-color: rgba(124, 106, 247, 0.3) !important;
  }

  /* Input */
  [data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(124, 106, 247, 0.4) !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    font-family: 'Inter', sans-serif !important;
  }
  [data-testid="stChatInput"] textarea:focus {
    border-color: rgba(124, 106, 247, 0.9) !important;
    box-shadow: 0 0 0 3px rgba(124, 106, 247, 0.15) !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #7c6af7 0%, #5b4de0 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124, 106, 247, 0.4) !important;
  }

  /* Expanders (source cards) */
  [data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
  }
  [data-testid="stExpander"] summary {
    color: #b8bccc !important;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    padding: 12px !important;
  }

  /* Info / success / warning banners */
  .stAlert {
    border-radius: 10px !important;
    border-left-width: 4px !important;
  }

  /* Dividers */
  hr {
    border-color: rgba(255,255,255,0.08) !important;
  }

  /* Tag pills */
  .tag-pill {
    display: inline-block;
    background: rgba(124, 106, 247, 0.2);
    border: 1px solid rgba(124, 106, 247, 0.4);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #a89df7;
    margin: 2px;
  }

  /* Intent badge */
  .intent-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-left: 6px;
  }
  .intent-think  { background: rgba(168, 216, 234, 0.2); color: #a8d8ea; }
  .intent-save   { background: rgba(168, 230, 207, 0.2); color: #a8e6cf; }
  .intent-find   { background: rgba(255, 211, 182, 0.2); color: #ffd3b6; }
  .intent-history{ background: rgba(212, 165, 245, 0.2); color: #d4a5f5; }

  /* Proactive banner */
  .proactive-banner {
    background: linear-gradient(135deg, rgba(124, 106, 247, 0.15) 0%, rgba(91, 77, 224, 0.1) 100%);
    border: 1px solid rgba(124, 106, 247, 0.4);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
    animation: fadeSlideIn 0.4s ease;
  }
  @keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Header */
  .brain-header {
    background: linear-gradient(135deg, rgba(124,106,247,0.15) 0%, rgba(91,77,224,0.08) 100%);
    border: 1px solid rgba(124,106,247,0.25);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 20px;
  }
  .brain-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
  .brain-header p  { margin: 4px 0 0; color: #888; font-size: 0.9rem; }

  /* Scrollbar */
  ::-webkit-scrollbar       { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(124,106,247,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state initialisation ─────────────────────────────────────────────

def _init_state():
    if "session_manager" not in st.session_state:
        user_id = profiles.load_or_create_user(
            st.session_state.get("username", "default")
        )
        st.session_state.user_id         = user_id
        st.session_state.session_manager = SessionManager(user_id=user_id)
        st.session_state.history         = []       # [{role, content, intent, routed_by}]
        st.session_state.proactive_queue = []       # pending proactive suggestions
        st.session_state.shown_note_ids  = set()
        st.session_state.turn_count      = 0
        st.session_state.last_intent     = "THINK"
        profiles.increment_session_count(user_id)


_init_state()

session  = st.session_state.session_manager
user_id  = st.session_state.user_id


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo / title
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 24px;">
      <span style="font-size:2.4rem;">🧠</span>
      <div style="font-size:1.2rem; font-weight:700; color:#a89df7; margin-top:4px;">Second Brain</div>
      <div style="font-size:0.75rem; color:#666; margin-top:2px;">local · private · model-agnostic</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Session stats
    st.markdown("### 📊 Session")
    col1, col2 = st.columns(2)
    col1.metric("Turns", st.session_state.turn_count)
    col2.metric("Tokens", f"{session.token_count:,}")

    st.caption(f"Model: `{cfg.get('model') or cfg.MODEL}`")

    st.divider()

    # Quick actions
    st.markdown("### ⚡ Quick actions")
    col_a, col_b = st.columns(2)
    if col_a.button("💾 Save", use_container_width=True, key="sidebar_save"):
        st.session_state._force_intent = "save"
        st.rerun()
    if col_b.button("📜 History", use_container_width=True, key="sidebar_history"):
        st.session_state._force_intent = "history"
        st.rerun()

    st.divider()

    # Recent notes
    st.markdown("### 📄 Recent notes")
    recent = hist_engine.get_summary_stats(user_id)
    st.caption(f"Total notes: **{recent['total_notes']}**  ·  Top tag: `{recent['top_tag']}`")

    from brain.index import get_recent_notes
    recent_notes = get_recent_notes(limit=6, user_id=user_id)
    if recent_notes:
        for note in recent_notes:
            tag_str = "  ".join(
                f"`{t}`" for t in note.get("tags", [])[:2]
            )
            st.markdown(
                f"**{note['title'][:40]}{'…' if len(note['title']) > 40 else ''}**  \n"
                f"<small style='color:#888'>{note.get('created_at','')[:10]}  {tag_str}</small>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No notes yet. Start thinking and save your first session!")

    st.divider()

    # Settings panel
    with st.expander("⚙️ Settings"):
        new_model = st.selectbox(
            "LLM model",
            ["gemini/gemini-2.5-flash", "gpt-4o", "claude-sonnet-4-6", "ollama/llama3"],
            index=0,
            key="settings_model",
        )
        new_rerank_threshold = st.slider(
            "Reranker threshold (ms-marco logits)", -20.0, 5.0, float(cfg.get("rerank_threshold") or -5.0), 0.5,
            key="settings_rerank_threshold",
            help="ms-marco-MiniLM returns raw logits. -5.0 is a sensible default.",
        )
        new_proactive_threshold = st.slider(
            "Proactive threshold (similarity)", 0.3, 1.0, float(cfg.get("proactive_threshold") or 0.55), 0.01,
            key="settings_proactive_threshold",
            help="Cosine similarity threshold for past note surfacing. 0.55 is a sensible default.",
        )
        new_budget = st.number_input(
            "Token budget", 10_000, 200_000,
            int(cfg.get("token_budget") or cfg.TOKEN_BUDGET), 10_000,
            key="settings_budget",
        )
        if st.button("Apply settings", key="apply_settings"):
            cfg.update_config(
                model=new_model,
                rerank_threshold=new_rerank_threshold,
                proactive_threshold=new_proactive_threshold,
                token_budget=new_budget,
            )
            st.success("Settings updated!")

    # Clear session
    if st.button("🗑️ New session", use_container_width=True, key="clear_btn"):
        session.save_to_db()
        st.session_state.clear()
        st.rerun()


# ─── Main content area ────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="brain-header">
  <h1>🧠 Second Brain</h1>
  <p>Think out loud · save what matters · recall it later</p>
</div>
""", unsafe_allow_html=True)

# ── Handle force-intent from sidebar buttons ───────────────────────────────────
_force = st.session_state.pop("_force_intent", None)
if _force == "save":
    with st.spinner("Generating and saving note…"):
        result = intent.route("let's save this", session, user_id)
    st.session_state.history.append({
        "role":      "user",
        "content":   "💾 Save session",
        "intent":    "SAVE",
        "routed_by": "sidebar",
    })
    st.session_state.history.append({
        "role":      "assistant",
        "content":   result["reply"],
        "intent":    "SAVE",
        "routed_by": result["routed_by"],
        "note":      result.get("note"),
        "sources":   None,
    })
    st.session_state.turn_count += 1
    st.rerun()

if _force == "history":
    with st.spinner("Fetching history…"):
        result = intent.route("what have I been thinking about most?", session, user_id)
    st.session_state.history.append({
        "role": "user", "content": "📜 Show my history", "intent": "HISTORY", "routed_by": "sidebar",
    })
    st.session_state.history.append({
        "role": "assistant", "content": result["reply"], "intent": "HISTORY",
        "routed_by": result["routed_by"], "sources": None, "note": None,
    })
    st.session_state.turn_count += 1
    st.rerun()

# ── Proactive suggestion banner ────────────────────────────────────────────────
if st.session_state.proactive_queue:
    suggestion = st.session_state.proactive_queue[0]
    st.markdown(f"""
    <div class="proactive-banner">
      <div style="font-size:0.8rem;color:#888;margin-bottom:4px;">💡 RELATED NOTE SURFACED</div>
      <div style="font-weight:600;color:#a89df7;font-size:1rem;">{suggestion['title']}</div>
      <div style="color:#999;font-size:0.85rem;margin-top:4px;">{suggestion['summary']}</div>
      <div style="color:#666;font-size:0.75rem;margin-top:6px;">{suggestion['age']} · similarity {suggestion['score']:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    col_v, col_d, _ = st.columns([1, 1, 6])
    if col_v.button("View note", key="proactive_view"):
        note = hist_engine  # just surface the title for now
        st.session_state.proactive_queue.pop(0)
        # Route a FIND query for this note
        result = intent.route(
            f"what did I write about {suggestion['title']}",
            session, user_id,
        )
        st.session_state.history.append({
            "role": "assistant", "content": result["reply"], "intent": "FIND",
            "routed_by": "proactive", "sources": result.get("sources"), "note": None,
        })
        st.session_state.turn_count += 1
        st.rerun()
    if col_d.button("Dismiss", key="proactive_dismiss"):
        st.session_state.proactive_queue.pop(0)
        st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
intent_colours = {
    "THINK":   "#a8d8ea",
    "SAVE":    "#a8e6cf",
    "FIND":    "#ffd3b6",
    "HISTORY": "#d4a5f5",
}

for msg in st.session_state.history:
    role    = msg["role"]
    content = msg["content"]
    msg_intent = msg.get("intent", "THINK")

    with st.chat_message(role, avatar="🧠" if role == "assistant" else "💭"):
        if role == "assistant":
            # Intent badge
            colour = intent_colours.get(msg_intent, "#fff")
            st.markdown(
                f'<span class="intent-badge intent-{msg_intent.lower()}">{msg_intent}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(content)

        # Source cards for FIND responses
        if role == "assistant" and msg.get("sources"):
            st.markdown("---")
            st.markdown("**📚 Sources:**")
            for src in msg["sources"]:
                tag_pills = "".join(
                    f'<span class="tag-pill">{t}</span>'
                    for t in src.get("tags", [])
                )
                with st.expander(f"📄 {src['title']}  ·  {src.get('created_at','')[:10]}  (score: {src.get('score',0):.2f})"):
                    st.markdown(src.get("body", "_Note body not available._"))
                    if tag_pills:
                        st.markdown(tag_pills, unsafe_allow_html=True)

        # Note save confirmation
        if role == "assistant" and msg.get("note"):
            note_info = msg["note"]
            if "file_path" in note_info:
                st.caption(f"📁 Saved to: `{note_info['file_path']}`")

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Think out loud, or type /save · /find <query> · /history",
    key="chat_input",
):
    # Pre-process slash commands
    display_prompt = prompt
    if prompt.strip().lower() == "/save":
        prompt = "let's save this"
        display_prompt = "💾 /save"
    elif prompt.strip().lower().startswith("/find "):
        prompt = "what did I write about " + prompt.strip()[6:].strip()
        display_prompt = f"🔍 /find {prompt.strip()[6:].strip() if len(prompt) > 6 else ''}"
    elif prompt.strip().lower() == "/history":
        prompt = "what have I been thinking about most?"
        display_prompt = "📜 /history"

    # Add user message to display history
    st.session_state.history.append({
        "role":      "user",
        "content":   display_prompt,
        "intent":    None,
        "routed_by": None,
    })

    # Show spinner while processing
    with st.spinner("…"):
        result = intent.route(prompt, session, user_id)

    st.session_state.turn_count += 1
    st.session_state.last_intent = result["intent"]

    # Add assistant response
    st.session_state.history.append({
        "role":      "assistant",
        "content":   result["reply"],
        "intent":    result["intent"],
        "routed_by": result["routed_by"],
        "sources":   result.get("sources"),
        "note":      result.get("note"),
    })

    # ── Proactive surfacing (every N THINK turns) ──────────────────────────────
    if (
        result["intent"] == "THINK"
        and st.session_state.turn_count > 0
        and st.session_state.turn_count % cfg.PROACTIVE_EVERY_N == 0
    ):
        suggestion = proactive.check_proactive(
            session.get_messages_for_llm(),
            exclude_note_ids=st.session_state.shown_note_ids,
        )
        if suggestion:
            st.session_state.proactive_queue.append(suggestion)
            st.session_state.shown_note_ids.add(suggestion["note_id"])

    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#444;font-size:0.75rem;padding:40px 0 10px;'>"
    "Second Brain · local · private · model-agnostic · "
    f"session {session.session_id[:8]}…</div>",
    unsafe_allow_html=True,
)
