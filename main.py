"""
main.py — Second Brain Terminal UI (Rich + prompt_toolkit).

Usage:
    python main.py [--user USERNAME]

Slash commands:
    /save       Save the current session as a note
    /find QUERY Find notes matching QUERY
    /history    Show usage history
    /clear      Clear session history
    /quit       Exit

Every message passes through the intent pipeline; slash commands are an
explicit shortcut that bypasses ambiguity.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# Load .env before importing brain modules (API keys etc.)
load_dotenv()

from brain import config, intent, profiles, proactive
from brain.session import SessionManager

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Rich console ─────────────────────────────────────────────────────────────
console = Console()

# prompt_toolkit style
PT_STYLE = Style.from_dict({
    "prompt": "bold #7c6af7",
})

BANNER = """[bold #7c6af7]
  ╔══════════════════════════════════╗
  ║     S E C O N D   B R A I N      ║
  ╚══════════════════════════════════╝[/bold #7c6af7]
[dim]Type to think out loud.  /save · /find <query> · /history · /quit[/dim]
"""


def print_banner(user_id: str, model: str) -> None:
    console.print(BANNER)
    console.print(f"  User: [cyan]{user_id[:8]}…[/cyan]   Model: [green]{model}[/green]\n")


def print_assistant(text: str, intent_label: str = "THINK") -> None:
    colour = {
        "THINK":   "#a8d8ea",
        "SAVE":    "#a8e6cf",
        "FIND":    "#ffd3b6",
        "HISTORY": "#d4a5f5",
    }.get(intent_label, "white")

    console.print(Panel(
        Markdown(text),
        border_style=colour,
        title=f"[dim]{intent_label}[/dim]",
        title_align="right",
        padding=(0, 1),
    ))


def print_proactive(suggestion: dict) -> None:
    console.print(Panel(
        f"💡 [bold]Related note:[/bold] {suggestion['title']}\n"
        f"[dim]{suggestion['summary']}[/dim]\n"
        f"[dim italic]{suggestion['age']}  ·  score {suggestion['score']:.2f}[/dim italic]",
        border_style="yellow",
        title="[yellow]Past note surfaced[/yellow]",
        title_align="left",
    ))


def print_user(text: str) -> None:
    console.print(f"\n[bold #7c6af7]You »[/bold #7c6af7] {text}\n")


# ─── Main loop ────────────────────────────────────────────────────────────────

async def main_loop(username: str) -> None:
    user_id  = profiles.load_or_create_user(username)
    session  = SessionManager(user_id=user_id)
    pq       = asyncio.Queue()   # proactive suggestion queue
    shown_notes: set[str] = set()

    profiles.increment_session_count(user_id)

    pt_session = PromptSession(
        history=InMemoryHistory(),
        style=PT_STYLE,
    )

    print_banner(user_id, config.get("model") or config.MODEL)

    while True:
        # ── Drain proactive queue ──────────────────────────────────────────
        while not pq.empty():
            suggestion = pq.get_nowait()
            print_proactive(suggestion)
            shown_notes.add(suggestion["note_id"])

        # ── Get user input ─────────────────────────────────────────────────
        try:
            raw = await pt_session.prompt_async("  You » ", style=PT_STYLE)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye. Your session has been saved.[/dim]")
            session.save_to_db()
            break

        raw = raw.strip()
        if not raw:
            continue

        # ── Slash command pre-processing ───────────────────────────────────
        if raw.lower() in ("/quit", "/exit", "/q"):
            console.print("\n[dim]Goodbye. Your session has been saved.[/dim]")
            session.save_to_db()
            break

        if raw.lower() == "/clear":
            session.history.clear()
            session.token_count = 0
            console.print("[dim]Session history cleared.[/dim]")
            continue

        if raw.lower() == "/history":
            raw = "what have I been thinking about most?"

        if raw.lower().startswith("/find "):
            raw = "what did I write about " + raw[6:].strip()

        if raw.lower() == "/save":
            raw = "let's save this"

        # ── Print user message ─────────────────────────────────────────────
        print_user(raw)

        # ── Route + respond ────────────────────────────────────────────────
        with console.status("[dim]Thinking…[/dim]", spinner="dots"):
            result = intent.route(raw, session, user_id)

        print_assistant(result["reply"], result["intent"])

        # ── Proactive surfacing (every N turns, THINK only) ────────────────
        if (
            result["intent"] == "THINK"
            and session.turn_count > 0
            and session.turn_count % config.PROACTIVE_EVERY_N == 0
        ):
            asyncio.create_task(
                proactive.async_check_proactive(
                    session.get_messages_for_llm(),
                    pq,
                    exclude_note_ids=shown_notes,
                )
            )

    # ── Update domain weights at session end ───────────────────────────────
    accessed_ids = list(shown_notes)
    if accessed_ids:
        from brain import profiles as prof
        prof.update_domain_weights(user_id, accessed_ids)


def main():
    parser = argparse.ArgumentParser(description="Second Brain — Terminal UI")
    parser.add_argument("--user", default="default", help="Username / profile name")
    args = parser.parse_args()

    try:
        asyncio.run(main_loop(args.user))
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")


if __name__ == "__main__":
    main()
