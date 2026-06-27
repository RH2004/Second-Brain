# NOTES.md — What's Left and What'd Improve with More Time

## What works

- Full hybrid retrieval pipeline (BM25 + ChromaDB + RRF + cross-encoder)
- Regex fast-path saving ~45% of classification token spend
- Structured note generation via `instructor` + Pydantic
- MongoDB analytics (frequency, recency, tag, date-window) via aggregation pipelines
- Proactive surfacing (synchronous in Streamlit, async in TUI)
- User profile personalisation scoring
- Sliding token window with accurate `tiktoken` counting
- Both frontends: Rich TUI and Streamlit web app
- Mongomock fallback for zero-deployment setup

---

## Known limitations and failure modes

See `DESIGN.md §16` for full trade-off analysis. Key items:

1. **Regex false positives** — "saving this pattern for later" triggers SAVE. Conservative patterns reduce but don't eliminate this. Mitigation: SAVE always shows a preview first.

2. **Short-session notes** — a 2–3 turn session produces thin summaries. The note generator prompt enforces minimum body length, but retrieval quality degrades.

3. **Cross-encoder reads only title + summary** — a note with a weak summary but rich body can be missed. Fix: rerank on a snippet of the body instead.

4. **Streamlit re-run model** — async proactive surfacing runs synchronously every N turns in Streamlit (acceptable at ~30ms), but means it only checks on user turns, not mid-response.

5. **Mongomock persistence** — in-memory; data is lost on restart without a real MongoDB instance.

---

## What'd improve with more time

### High priority

| Improvement | Effort | Impact |
|-------------|--------|--------|
| Note editing after save | Medium | `notes.update()` + re-embed + reindex BM25 |
| Streaming LLM responses | Low | `litellm` stream=True + Streamlit `write_stream` |
| Better reranking input | Low | Pass snippet of body, not just title+summary |
| Streamlit auth | Low | `streamlit-authenticator` for shared deployments |

### Medium priority

| Improvement | Effort | Impact |
|-------------|--------|--------|
| Note versioning | Medium | Version array in MongoDB; diff on re-save |
| Note linking (co-retrieval graph) | Medium | Track co-retrieval in access_log; build link graph |
| Multi-user namespace isolation | Low | Separate ChromaDB collections per user |
| Export + sync | Trivial | Notes are already plain Markdown; git sync works |

### Stretch goals

| Feature | Notes |
|---------|-------|
| Graph knowledge connections | Neo4j layer; "how does my retrieval note connect to my UX note?" |
| Streaming TUI | Rich `Live` display + `litellm` stream=True |
| Mobile-friendly Streamlit | CSS tweaks; works in browser already |
| Multilingual | Swap `all-MiniLM-L6-v2` → `paraphrase-multilingual-MiniLM-L12-v2` in config.py |
| Voice input | `whisper.cpp` transcription → feed into same pipeline |

---

## Architecture decisions I'd revisit

1. **BM25 pickle vs. Tantivy/Typesense** — pickle works well for hundreds of notes but gets unwieldy at tens of thousands. A proper inverted index would be cleaner.

2. **ChromaDB vs. Qdrant** — ChromaDB is the right local-first choice, but Qdrant's filtering and metadata handling is more expressive. Worth considering at scale.

3. **`instructor` over raw function calling** — instructor adds reliable structured output at the cost of an extra library dependency. At scale, native structured output (Gemini response schema, OpenAI JSON mode) may be more predictable.

4. **Template strings for history** — covers 95% of cases cleanly. The LLM narration fallback could be replaced with a richer template DSL for better coverage without LLM cost.
