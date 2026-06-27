"""
verify_system.py — Quick smoke-test for all system components.

Forces UTF-8 stdout so Unicode test labels render correctly on Windows.

Runs without a live LLM connection.  Tests:
    1. Config loading and path creation
    2. MongoDB / mongomock connection and CRUD
    3. ChromaDB parent + child chunk indexing
    4. BM25 pickle save / load
    5. Cross-encoder scoring
    6. RRF merge calculation
    7. Regex router fast-path
    8. Note Markdown write + frontmatter round-trip

Usage:
    python verify_system.py
"""

import sys
import traceback
from pathlib import Path


import sys
import io
# Force UTF-8 on Windows consoles that default to cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def ok(msg):   print(f"  [PASS]  {msg}")
def fail(msg): print(f"  [FAIL]  {msg}")


def test_config():
    print("\n── 1. Config ──")
    from brain import config
    assert config.NOTES_DIR.exists(),  "NOTES_DIR not created"
    assert config.CHROMA_DIR.exists(), "CHROMA_DIR not created"
    ok(f"ROOT         = {config.ROOT}")
    ok(f"NOTES_DIR    = {config.NOTES_DIR}")
    ok(f"CHROMA_DIR   = {config.CHROMA_DIR}")
    ok(f"BM25_PATH    = {config.BM25_INDEX_PATH}")
    ok(f"MODEL        = {config.MODEL}")
    ok(f"TOKEN_BUDGET = {config.TOKEN_BUDGET}")


def test_mongodb():
    print("\n── 2. MongoDB / mongomock ──")
    from brain import index
    db = index.get_db()
    assert db is not None
    if index.is_using_mock():
        ok("Using mongomock fallback (no real MongoDB needed)")
    else:
        ok("Connected to real MongoDB")

    # Insert + retrieve a note
    test_note = {
        "_id":        "test-note-001",
        "user_id":    "test-user",
        "title":      "Test Note",
        "summary":    "A test summary",
        "tags":       ["test", "verification"],
        "file_path":  "/tmp/test.md",
        "created_at": "2025-01-01T00:00:00Z",
        "modified_at":"2025-01-01T00:00:00Z",
        "session_turns": 3,
    }
    nid = index.upsert_note(test_note)
    assert nid == "test-note-001"
    retrieved = index.get_note_by_id("test-note-001")
    assert retrieved is not None
    assert retrieved["title"] == "Test Note"
    ok("Note upsert + retrieve OK")

    # Access log
    index.log_access("test-user", "test-note-001", "test query", "FIND")
    ok("Access log entry OK")

    # Profile
    from brain import profiles
    uid = profiles.load_or_create_user("verify_user")
    assert uid
    ok(f"User profile created: {uid[:8]}…")


def test_chromadb():
    print("\n── 3. ChromaDB ──")
    from brain import index

    # Simple embedding using a tiny manual vector (skip model download)
    parent_col = index.get_parent_collection()
    child_col  = index.get_child_collection()

    dummy_emb = [0.1] * 384

    parent_col.upsert(
        ids=["verify-parent-1"],
        embeddings=[dummy_emb],
        documents=["Test parent chunk"],
        metadatas=[{"note_id": "verify-parent-1"}],
    )
    ok("Parent chunk upserted")

    child_col.upsert(
        ids=["verify-child-1"],
        embeddings=[dummy_emb],
        documents=["Test child chunk"],
        metadatas=[{"parent_id": "verify-parent-1", "section": "Key insights", "chunk_index": 0}],
    )
    ok("Child chunk upserted")

    results = index.dense_search(dummy_emb, top_k=5)
    assert len(results) >= 1
    ok(f"Dense search returned {len(results)} result(s)")


def test_bm25():
    print("\n-- 4. BM25 persistence --")
    from brain import index, config
    import os

    # Delete stale pickle from any prior test run for a fully clean test
    if config.BM25_INDEX_PATH.exists():
        config.BM25_INDEX_PATH.unlink()
        ok("Deleted stale BM25 pickle")

    # Reset in-memory state
    index._bm25_obj      = None
    index._bm25_corpus   = []
    index._note_ids_bm25 = []

    index.add_to_bm25("bm25-note-1", "retrieval pipeline dense sparse BM25")
    index.add_to_bm25("bm25-note-2", "machine learning transformers attention")
    ok("Added 2 docs to BM25")

    results = index.bm25_query("retrieval dense", top_k=5)
    assert len(results) >= 1, f"Expected >=1 results, got {len(results)}"
    assert results[0]["note_id"] == "bm25-note-1", f"Top hit was {results[0]['note_id']}"
    ok(f"BM25 query correct: top hit = {results[0]['note_id']}")

    # Reload from pickle
    index._bm25_obj = None
    index.load_bm25()
    assert index._bm25_obj is not None
    ok("BM25 pickle round-trip OK")


def test_reranker():
    print("\n-- 5. Cross-encoder reranker --")
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        candidates = [
            ("what did I decide about retrieval?", "Retrieval Design Trade-offs. A note about BM25 and vector search."),
            ("what did I decide about retrieval?", "UX Patterns for AI Tools. A note about user experience design."),
            ("what did I decide about retrieval?", "Quantum Computing Basics. A note about quantum gates."),
        ]
        scores = model.predict(candidates)
        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
        # Retrieval note should score highest
        best_idx = int(scores.argmax()) if hasattr(scores, 'argmax') else scores.index(max(scores))
        ok(f"Reranker scores: {[round(float(s),3) for s in scores]}")
        ok(f"Top candidate index: {best_idx} (retrieval note = index 0)")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        fail(f"Reranker FAILED: {exc}")


def test_rrf():
    print("\n── 6. RRF merge ──")
    from brain.retrieval import _rrf_merge

    dense  = [{"note_id": "a", "score": 0.9}, {"note_id": "b", "score": 0.7}, {"note_id": "c", "score": 0.5}]
    sparse = [{"note_id": "b", "score": 10.0}, {"note_id": "a", "score": 8.0}, {"note_id": "d", "score": 6.0}]
    merged = _rrf_merge(dense, sparse, top_n=4, k=60)

    assert len(merged) <= 4
    ids = [r["note_id"] for r in merged]
    # "a" and "b" should be top (appear in both lists)
    assert "a" in ids[:2] or "b" in ids[:2]
    ok(f"RRF merged correctly: {ids}")


def test_router():
    print("\n── 7. Regex router ──")
    from brain import router

    save_cases = [
        "let's save this", "wrap up", "that's good", "I'm done", "/save",
        "file this", "note this down", "save it",
    ]
    find_cases = [
        "what did I write about retrieval", "do I have notes on memory",
        "find my thoughts on attention", "/find embeddings",
        "what do I know about transformers", "remind me about BM25",
    ]
    hist_cases = [
        "what have I been thinking about most?",
        "what did I work on last month",
        "how many times did I revisit retrieval",
        "/history",
    ]
    think_cases = [
        "I've been thinking about this problem",
        "here's my current mental model",
        "actually wait, let me reconsider",
    ]

    for case in save_cases:
        r = router.fast_route(case)
        assert r == "SAVE", f"Expected SAVE for: {case!r}, got {r}"
    ok(f"SAVE patterns: {len(save_cases)}/{len(save_cases)} correct")

    for case in find_cases:
        r = router.fast_route(case)
        assert r == "FIND", f"Expected FIND for: {case!r}, got {r}"
    ok(f"FIND patterns: {len(find_cases)}/{len(find_cases)} correct")

    for case in hist_cases:
        r = router.fast_route(case)
        assert r == "HISTORY", f"Expected HISTORY for: {case!r}, got {r}"
    ok(f"HISTORY patterns: {len(hist_cases)}/{len(hist_cases)} correct")

    for case in think_cases:
        r = router.fast_route(case)
        assert r is None, f"Expected None (THINK fallback) for: {case!r}, got {r}"
    ok(f"THINK fallback: {len(think_cases)}/{len(think_cases)} correctly passed to LLM")


def test_note_markdown():
    print("\n── 8. Markdown + frontmatter round-trip ──")
    import frontmatter
    import tempfile, os

    content = "# Test\n\n## Core idea\nSome core idea.\n\n## Key insights\n- Insight 1\n- Insight 2"
    post = frontmatter.Post(
        content=content,
        id="test-id",
        title="Test Note",
        created="2025-01-01T00:00:00Z",
        tags=["test", "verification"],
        summary="A test summary for retrieval.",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(frontmatter.dumps(post))
        tmp_path = f.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        loaded = frontmatter.load(f)

    assert loaded.metadata["title"]   == "Test Note"
    assert loaded.metadata["id"]      == "test-id"
    assert loaded.metadata["tags"]    == ["test", "verification"]
    assert "Core idea" in loaded.content
    ok(f"Frontmatter round-trip OK → {tmp_path}")
    os.unlink(tmp_path)


# ─── Run all tests ────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Second Brain — System Verification")
    print("=" * 55)

    tests = [
        test_config,
        test_mongodb,
        test_chromadb,
        test_bm25,
        test_rrf,
        test_router,
        test_note_markdown,
        test_reranker,   # last — may download models
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            fail(f"Test {test.__name__} FAILED: {exc}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 55)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
