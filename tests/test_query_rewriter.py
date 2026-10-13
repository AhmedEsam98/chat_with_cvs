import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from retrieval.query_rewriter import rewrite_query


def test_coreference_resolution():
    print("\n--- Test 1: Conversational Coreference Resolution ---")
    history = [
        {"role": "user", "content": "Who is Abdelrahman Karawia?"},
        {
            "role": "assistant",
            "content": "Abdelrahman Karawia is a Flutter Developer specializing in cross-platform mobile apps with Bloc and Clean Architecture.",
        },
    ]
    follow_up = "What are his main projects?"
    print(f"Prior Turn: User asked about Abdelrahman Karawia")
    print(f"Follow-up:  \"{follow_up}\"")

    t0 = time.perf_counter()
    rewritten = rewrite_query(follow_up, history=history)
    t1 = time.perf_counter()

    print(f"Rewritten:  \"{rewritten}\"")
    print(f"Latency:    {t1 - t0:.4f}s")
    assert "abdelrahman" in rewritten.lower() or "karawia" in rewritten.lower(), (
        f"Expected candidate name in rewritten query, got: {rewritten}"
    )
    print("✅ Coreference successfully resolved!")


def test_synonym_expansion():
    print("\n--- Test 2: Domain & Technical Keyword Expansion ---")
    query = "mobile developer"
    print(f"Input:     \"{query}\"")

    t0 = time.perf_counter()
    rewritten = rewrite_query(query, history=[])
    t1 = time.perf_counter()

    print(f"Rewritten: \"{rewritten}\"")
    print(f"Latency:   {t1 - t0:.4f}s")
    assert len(rewritten) > len(query), "Expected expanded query for generic role"
    print("✅ Query expansion successfully applied!")


def test_cache_speedup():
    print("\n--- Test 3: Query Rewriter Cache Latency ---")
    history = [
        {"role": "user", "content": "Who knows Python?"},
        {"role": "assistant", "content": "Mohamed and Sarah have Python experience."},
    ]
    query = "Where did he study?"

    # First call (cold / LLM)
    t0 = time.perf_counter()
    res1 = rewrite_query(query, history=history)
    cold_time = time.perf_counter() - t0

    # Second call (warm / cached)
    t0 = time.perf_counter()
    res2 = rewrite_query(query, history=history)
    warm_time = time.perf_counter() - t0

    print(f"Cold Rewrite Latency: {cold_time:.4f}s -> \"{res1}\"")
    print(f"Warm Rewrite Latency: {warm_time:.6f}s -> \"{res2}\"")

    assert res1 == res2, "Cached rewritten query should match cold result"
    assert warm_time < 0.05, f"Cached query should be instantaneous (<50ms), took {warm_time}s"
    print(f"✅ Cache speedup verified! ({cold_time / max(warm_time, 0.000001):.1f}x faster)")


def test_disabled_bypass():
    print("\n--- Test 4: Disable Config Bypass ---")
    original_setting = config.QUERY_REWRITER_ENABLED
    try:
        config.QUERY_REWRITER_ENABLED = False
        raw = "What are his projects?"
        res = rewrite_query(raw, history=[{"role": "user", "content": "Tell me about John."}])
        assert res == raw, "Should return raw query when disabled"
        print(f"✅ Bypassed cleanly when disabled: \"{res}\"")
    finally:
        config.QUERY_REWRITER_ENABLED = original_setting


if __name__ == "__main__":
    print("=" * 60)
    print("Running Query Rewriter Test Suite")
    print("=" * 60)
    try:
        test_coreference_resolution()
        test_synonym_expansion()
        test_cache_speedup()
        test_disabled_bypass()
        print("\n" + "=" * 60)
        print("🎉 ALL QUERY REWRITER TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
