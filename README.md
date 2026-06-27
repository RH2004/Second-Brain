# Second Brain 🧠

> A local, Python-based thinking tool that remembers what you worked through and hands it back when it matters. Model-agnostic, multi-database, token-efficient — available as both a terminal TUI and a Streamlit web app.

---

## Features

| Feature | Detail |
|---------|--------|
| 🧠 Think out loud | Free-form chat with context-aware replies |
| 💾 Auto-structured notes | LLM writes a Markdown note with frontmatter at session end |
| 🔍 Hybrid retrieval | BM25 + dense (ChromaDB) + RRF + cross-encoder reranking |
| 📊 History analytics | MongoDB aggregations — frequency, recency, tag patterns |
| ⚡ Token-efficient | Regex fast-path saves ~45% of classification tokens |
| 💡 Proactive surfacing | Related past notes surface mid-session (no LLM call) |
| 🔌 Model-agnostic | Gemini, GPT-4o, Claude, Ollama — swap in config |

---

## Quick start

### 1. Clone and set up

```bash
git clone <repo-url>
cd second_brain
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Set your API key

```bash
# Create a .env file (or set environment variables)
echo GEMINI_API_KEY=your_key_here > .env
```

Supported models (set `LLM_MODEL` in `.env` to switch):

```env
LLM_MODEL=gemini/gemini-1.5-flash   # default — free tier
# LLM_MODEL=gpt-4o
# LLM_MODEL=claude-sonnet-4-6
# LLM_MODEL=ollama/llama3           # fully local, no API key needed
```

### 3. (Optional) MongoDB

The system works without a local MongoDB installation — it automatically falls back to an in-memory mock database (`mongomock`). For full persistence across restarts, install and run MongoDB Community Edition.

```bash
# Install MongoDB Community: https://www.mongodb.com/try/download/community
# Then start it:
mongod --dbpath ./data/db
```

### 4. Run the Streamlit web app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

### 5. Run the Terminal TUI

```bash
python main.py
# With a custom username:
python main.py --user yourname
```

### 6. Verify the system

```bash
python verify_system.py
```

---

## Usage

### Thinking (THINK)

Just type. The assistant engages with your ideas.

```
You » I've been thinking about the trade-off between lexical and semantic search...
```

### Saving (SAVE)

When you're done thinking, save the session as a structured note:

```
You » let's save this
You » wrap up
You » /save
```

The system generates a structured Markdown note with:
- YAML frontmatter (id, title, tags, summary)
- Core idea, key insights, open questions, threads to follow

Notes are saved to `storage/notes/YYYY-MM-DD-slug.md`.

### Finding (FIND)

In any session, ask about past thinking:

```
You » what did I figure out about retrieval?
You » what do I know about transformers?
You » /find reranking
```

The system runs:
1. Dense search (ChromaDB child chunks → parent resolve)
2. Sparse search (BM25)
3. Reciprocal Rank Fusion merge
4. Cross-encoder reranking (local, ~30ms)
5. LLM answer synthesis with citations

If nothing is found, it says so honestly — no hallucination.

### History (HISTORY)

```
You » what have I been thinking about most?
You » what did I work on last month?
You » do I have notes tagged 'retrieval'?
You » /history
```

---

## Architecture

```
User message
    ↓
Regex fast-path (router.py)
    ↓ SAVE / FIND / HISTORY: no LLM call
    ↓ ambiguous: LLM fallback (orchestrator.py)
    ↓
Intent dispatcher (intent.py)
    ├── THINK  → orchestrator.think_reply()
    ├── SAVE   → notes.generate_and_save()
    ├── FIND   → retrieval.find()
    └── HISTORY → history.surface_patterns()
```

### Storage

| Store | Technology | Purpose |
|-------|-----------|---------|
| Note files | Markdown on disk | Human-readable, git-friendly |
| Metadata & sessions | MongoDB / mongomock | Schemas, aggregations |
| Vector index | ChromaDB (HNSW) | Cosine ANN search |
| BM25 index | Pickle on disk | Sparse keyword search |

---

## File structure

```
second_brain/
├── main.py                 # TUI entry point
├── app.py                  # Streamlit entry point
├── verify_system.py        # Smoke tests
├── requirements.txt
├── .env                    # API keys (not committed)
│
├── brain/
│   ├── config.py           # All settings
│   ├── session.py          # Sliding token window
│   ├── router.py           # Regex fast-path
│   ├── intent.py           # Dispatch layer
│   ├── orchestrator.py     # All LLM calls
│   ├── notes.py            # Note generation pipeline
│   ├── index.py            # MongoDB + ChromaDB + BM25
│   ├── retrieval.py        # Hybrid retrieval pipeline
│   ├── reranker.py         # CrossEncoder wrapper
│   ├── history.py          # History analytics
│   ├── profiles.py         # User profiles
│   └── proactive.py        # Background similarity check
│
└── storage/
    ├── notes/              # Markdown note files
    ├── chroma/             # ChromaDB persistent index
    └── bm25_index.pkl      # BM25 pickle
```

---

## Configuration

All settings are in `brain/config.py`. Key knobs:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL` | `gemini/gemini-1.5-flash` | LLM provider |
| `TOKEN_BUDGET` | `80,000` | Sliding context window cap |
| `RERANK_THRESHOLD` | `0.5` | Minimum cross-encoder score |
| `PROACTIVE_THRESHOLD` | `0.78` | Similarity threshold for surfacing |
| `PROACTIVE_EVERY_N` | `4` | Check every N turns |

All can be overridden via environment variables or the Streamlit settings panel.

---

## Design document

See [DESIGN.md](./DESIGN.md) for the full technical design: architecture, storage schemas, retrieval design, token efficiency strategy, and trade-offs.
