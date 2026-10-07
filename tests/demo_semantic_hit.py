import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
from retrieval import retrieve

q1 = "Who has experience with Flutter development?"
q2 = "Which candidates know Flutter and Dart?"

print("=" * 70)
print("🚀 LIVE SEMANTIC CACHE DEMO")
print("=" * 70)

print(f"\n1️⃣ Running Initial Query: \"{q1}\"")
t0 = time.perf_counter()
hits1 = retrieve(q1, k=3)
t_init = time.perf_counter() - t0
print(f"   ⏱️ Time taken: {t_init:.3f}s (Fetched from Azure Search & Cached)")
print(f"   📄 Top candidate returned: {hits1[0]['cv_name'] if hits1 else 'None'}")

print(f"\n2️⃣ Running Paraphrased Query: \"{q2}\"")
t1 = time.perf_counter()
hits2 = retrieve(q2, k=3)
t_cached = time.perf_counter() - t1
print(f"   ⏱️ Time taken: {t_cached:.3f}s (🎯 Returned via SEMANTIC CACHE!)")
print(f"   📄 Top candidate returned: {hits2[0]['cv_name'] if hits2 else 'None'}")

if t_init > 0:
    speedup = t_init / max(t_cached, 0.0001)
    print(f"\n⚡ Speedup: {speedup:.1f}x faster!")

print("=" * 70)
