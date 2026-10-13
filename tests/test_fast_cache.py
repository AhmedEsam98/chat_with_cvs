import os
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.searcher import retrieve
from generation.generator import answer_stream

q = "who has ai developer position"

print("=== Test 1: First retrieval call (warm or new) ===")
t0 = time.perf_counter()
hits1 = retrieve(q, 10)
t1 = time.perf_counter()
print(f"Retrieval 1: {t1 - t0:.4f}s (hits: {len(hits1)})")

print("\n=== Test 2: Exact same question retrieval ===")
t0 = time.perf_counter()
hits2 = retrieve(q, 10)
t1 = time.perf_counter()
print(f"Retrieval 2 (Exact): {t1 - t0:.6f}s (hits: {len(hits2)})")

print("\n=== Test 3: Same question with trailing question mark '?' ===")
t0 = time.perf_counter()
hits3 = retrieve(q + "?", 10)
t1 = time.perf_counter()
print(f"Retrieval 3 (Punctuation): {t1 - t0:.6f}s (hits: {len(hits3)})")

print("\n=== Test 4: Same question with smaller k=5 ===")
t0 = time.perf_counter()
hits4 = retrieve(q, 5)
t1 = time.perf_counter()
print(f"Retrieval 4 (min_k=5): {t1 - t0:.6f}s (hits: {len(hits4)})")

print("\n=== Test 5: Generation latency for first call ===")
t0 = time.perf_counter()
tokens = list(answer_stream(q, [], hits2))
t1 = time.perf_counter()
print(f"Generation Latency (First Call): {t1 - t0:.6f}s (tokens generated: {len(tokens)})")
print("Answer preview:", "".join(tokens)[:120])

print("\n=== Test 6: Generation latency on repeated call (Cache Hit) ===")
t0 = time.perf_counter()
tokens_cached = list(answer_stream(q, [], hits2))
t1 = time.perf_counter()
print(f"Generation Latency (Cache Hit): {t1 - t0:.6f}s")
assert t1 - t0 < 0.05, f"Expected cached answer <50ms, took {t1 - t0}s"
assert "".join(tokens) == "".join(tokens_cached)
print("✅ Answer cache verified! (<50ms)")

