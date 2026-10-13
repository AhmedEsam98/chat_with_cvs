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
from retrieval.router import (
    LABEL_CV_RELATED,
    LABEL_NOT_CV_RELATED,
    classify_question,
    get_out_of_scope_message,
    is_cv_related,
)


def test_cv_related_queries():
    print("\n--- Test 1: CV-Related Queries (Expected: CV_RELATED) ---")
    queries = [
        "Who is Akram?",
        "Who is the Flutter developer?",
        "Which candidates have experience with Docker or Kubernetes?",
        "What degrees and certifications do the candidates have?",
        "Compare the AI engineers on our team",
        "Does Abdelrahman Karawia have mobile app projects?",
    ]

    for q in queries:
        t0 = time.perf_counter()
        label = classify_question(q)
        lat = time.perf_counter() - t0
        print(f"  • \"{q}\" -> {label} ({lat:.4f}s)")
        assert label == LABEL_CV_RELATED, f"Expected CV_RELATED for '{q}', got {label}"
    print("✅ All CV-related questions correctly classified!")


def test_followup_cv_queries():
    print("\n--- Test 2: Conversational Follow-up Queries with History ---")
    history = [
        {"role": "user", "content": "Who is Akram?"},
        {"role": "assistant", "content": "Akram Mohamed Hassan is an AI Engineer."},
    ]
    follow_up = "What are his main projects?"
    label = classify_question(follow_up, history=history)
    print(f"  • History: Prior turn about Akram")
    print(f"  • Follow-up: \"{follow_up}\" -> {label}")
    assert label == LABEL_CV_RELATED, f"Expected CV_RELATED for follow-up '{follow_up}'"
    print("✅ Conversational follow-up correctly classified!")


def test_not_cv_related_queries():
    print("\n--- Test 3: Off-Topic / Unrelated Queries (Expected: NOT_CV_RELATED) ---")
    off_topic = [
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "Who won the 2022 World Cup in Qatar?",
        "Write a poem about the ocean",
        "hello",
    ]

    for q in off_topic:
        t0 = time.perf_counter()
        label = classify_question(q)
        lat = time.perf_counter() - t0
        print(f"  • \"{q}\" -> {label} ({lat:.4f}s)")
        assert label == LABEL_NOT_CV_RELATED, f"Expected NOT_CV_RELATED for '{q}', got {label}"
        # Verify message generation
        msg = get_out_of_scope_message(q)
        assert len(msg) > 20
    print("✅ All off-topic queries correctly caught and refused!")


def test_cache_and_bypass():
    print("\n--- Test 4: Router Cache Speedup & Bypass Toggle ---")
    q = "What is quantum computing and superposition?"

    # Cold run
    t0 = time.perf_counter()
    label1 = classify_question(q)
    cold_lat = time.perf_counter() - t0

    # Warm run (cached)
    t0 = time.perf_counter()
    label2 = classify_question(q)
    warm_lat = time.perf_counter() - t0

    print(f"  • Cold Latency: {cold_lat:.4f}s -> {label1}")
    print(f"  • Warm Latency: {warm_lat:.6f}s -> {label2}")
    assert label1 == label2 == LABEL_NOT_CV_RELATED
    assert warm_lat < 0.05, f"Expected cached classification <50ms, took {warm_lat}s"
    print("  ✅ Router caching verified!")

    # Test disabled bypass
    orig = config.QUESTION_ROUTER_ENABLED
    try:
        config.QUESTION_ROUTER_ENABLED = False
        res = classify_question(q)
        assert res == LABEL_CV_RELATED, "Expected CV_RELATED fallback when router disabled"
        print("  ✅ Bypassed to CV_RELATED when disabled.")
    finally:
        config.QUESTION_ROUTER_ENABLED = orig


if __name__ == "__main__":
    print("=" * 70)
    print("🧭 Running Question Router Test Suite")
    print("=" * 70)
    test_cv_related_queries()
    test_followup_cv_queries()
    test_not_cv_related_queries()
    test_cache_and_bypass()
    print("\n" + "=" * 70)
    print("🎉 ALL QUESTION ROUTER TESTS PASSED!")
    print("=" * 70)
