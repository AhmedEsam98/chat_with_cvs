"""Unit and integration test for Semantic Cache."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for Windows consoles (e.g. cp1256 / cp1252)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from cache_manager import (
    cache_clear,
    semantic_cache_get,
    semantic_cache_set,
    is_redis_available,
    _compute_cosine_similarities,
)


def run_tests():
    print("=" * 70)
    print("🧪 RUNNING SEMANTIC CACHE VERIFICATION SUITE")
    print(f"Redis Connected: {is_redis_available()}")
    print(f"Semantic Cache Enabled: {config.SEMANTIC_CACHE_ENABLED}")
    print(f"Semantic Cache Cutoff Threshold: {config.SEMANTIC_CACHE_THRESHOLD}")
    print("=" * 70)

    # Clean up test prefix
    cache_clear("test:")

    # 1. Test Cosine Similarity Math
    print("\n[Test 1] Cosine Similarity Vector Computation...")
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]  # identical -> 1.0
    v3 = [0.0, 1.0, 0.0]  # orthogonal -> 0.0
    v4 = [0.7071, 0.7071, 0.0]  # ~45 deg -> ~0.7071

    sims = _compute_cosine_similarities(v1, [v2, v3, v4])
    assert abs(sims[0] - 1.0) < 1e-4, f"Expected 1.0, got {sims[0]}"
    assert abs(sims[1] - 0.0) < 1e-4, f"Expected 0.0, got {sims[1]}"
    assert abs(sims[2] - 0.7071) < 1e-3, f"Expected ~0.7071, got {sims[2]}"
    print("  ✅ Cosine similarity math is correct.")

    # 2. Test Semantic Set and Get (High Similarity Hit)
    print("\n[Test 2] Semantic Cache Hit with High Similarity...")
    # Base query vector (normalized)
    base_vec = [0.1] * 10
    norm = sum(x * x for x in base_vec) ** 0.5
    base_vec = [x / norm for x in base_vec]

    # Stored query
    semantic_cache_set(
        namespace="test",
        query_text="Who has Flutter experience?",
        query_vector=base_vec,
        value=[{"candidate": "Abdelrahman", "role": "Flutter Dev"}],
        filter_params={"k": 5},
    )

    # Slightly perturbed vector (simulating a semantically equivalent query with similarity > 0.98)
    query_vec = list(base_vec)
    query_vec[0] += 0.01
    q_norm = sum(x * x for x in query_vec) ** 0.5
    query_vec = [x / q_norm for x in query_vec]

    result, sim, matched_q = semantic_cache_get(
        namespace="test",
        query_vector=query_vec,
        threshold=0.92,
        filter_params={"k": 5},
    )

    assert result is not None, "Expected semantic cache HIT!"
    assert sim >= 0.92, f"Expected similarity >= 0.92, got {sim}"
    assert matched_q == "Who has Flutter experience?"
    assert result[0]["candidate"] == "Abdelrahman"
    print(f"  ✅ Semantic Cache HIT verified! Similarity: {sim:.4f}, Matched: '{matched_q}'")

    # 3. Test Semantic Cache Miss (Low Similarity)
    print("\n[Test 3] Semantic Cache Miss with Low Similarity...")
    # Completely different orthogonal query vector
    diff_vec = [0.0] * 10
    diff_vec[0] = 1.0
    # Make base_vec orthogonal to diff_vec
    ortho_vec = [0.0] * 10
    ortho_vec[1] = 1.0

    semantic_cache_set(
        namespace="test",
        query_text="Python Django backend developer",
        query_vector=ortho_vec,
        value=[{"candidate": "Mohamed", "role": "Backend"}],
        filter_params={"k": 5},
    )

    result_miss, sim_miss, _ = semantic_cache_get(
        namespace="test",
        query_vector=diff_vec,
        threshold=0.92,
        filter_params={"k": 5},
    )
    assert result_miss is None, f"Expected cache MISS, but got a hit with sim {sim_miss}"
    print(f"  ✅ Semantic Cache correctly MISSED when similarity is low ({sim_miss:.4f} < 0.92).")

    # 4. Test Filter Params Constraint (e.g. k=3 vs k=5)
    print("\n[Test 4] Filter Params Isolation (e.g. k=3 vs k=5)...")
    result_k_miss, _, _ = semantic_cache_get(
        namespace="test",
        query_vector=query_vec,
        threshold=0.92,
        filter_params={"k": 3},  # Query requested k=3, cache only has k=5
    )
    assert result_k_miss is None, "Expected cache miss due to mismatched filter_param 'k'"
    print("  ✅ Filter parameters properly isolate cache entries.")

    # 5. Clean Up
    cache_clear("test:")
    post_clear, _, _ = semantic_cache_get(
        namespace="test",
        query_vector=query_vec,
        threshold=0.5,
        filter_params={"k": 5},
    )
    assert post_clear is None, "Cache clear failed to remove semantic entries!"
    print("\n[Test 5] Cache Invalidation...")
    print("  ✅ Cache clear successfully purged all test entries.")

    print("\n" + "=" * 70)
    print("🎉 ALL SEMANTIC CACHE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
