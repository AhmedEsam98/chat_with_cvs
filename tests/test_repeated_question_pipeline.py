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

from retrieval.router import classify_question
from retrieval.query_rewriter import rewrite_query
from retrieval.searcher import retrieve
from generation.generator import answer_stream


def run_pipeline_turn(question: str, history: list[dict], top_k: int = 10):
    t_start = time.perf_counter()

    # Step 0: Router
    t0 = time.perf_counter()
    route = classify_question(question, history=history)
    t_router = time.perf_counter() - t0

    # Step 1: Rewriter
    t0 = time.perf_counter()
    search_query = rewrite_query(question, history=history)
    t_rewrite = time.perf_counter() - t0

    # Step 2: Retrieval
    t0 = time.perf_counter()
    hits = retrieve(search_query, top_k)
    t_retrieval = time.perf_counter() - t0

    # Step 3: Generation
    t0 = time.perf_counter()
    tokens = list(answer_stream(question, history, hits, search_query=search_query))
    t_generation = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start
    answer = "".join(tokens)

    return {
        "answer": answer,
        "search_query": search_query,
        "hits_count": len(hits),
        "t_total": t_total,
        "t_router": t_router,
        "t_rewrite": t_rewrite,
        "t_retrieval": t_retrieval,
        "t_generation": t_generation,
    }


def test_repeated_question():
    q = "who is akram"
    print("=" * 70)
    print(f"Testing End-to-End Latency for Repeated Question: \"{q}\"")
    print("=" * 70)

    # Turn 1: Cold / First time asking
    print("\n[Turn 1] First time asking question:")
    history_turn1 = []
    res1 = run_pipeline_turn(q, history_turn1)
    print(f"  • Total:      {res1['t_total']:.4f}s")
    print(f"  • Router:     {res1['t_router']:.4f}s")
    print(f"  • Rewrite:    {res1['t_rewrite']:.4f}s (\" {res1['search_query']} \")")
    print(f"  • Retrieval:  {res1['t_retrieval']:.4f}s ({res1['hits_count']} hits)")
    print(f"  • Generation: {res1['t_generation']:.4f}s")
    print(f"  • Answer:     \"{res1['answer'][:80]}...\"")

    # Turn 2: Exact same question in conversational chat history
    print("\n[Turn 2] Exact same question asked as next turn:")
    history_turn2 = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": res1["answer"]},
    ]
    res2 = run_pipeline_turn(q, history_turn2)
    print(f"  • Total:      {res2['t_total']:.4f}s")
    print(f"  • Router:     {res2['t_router']:.4f}s")
    print(f"  • Rewrite:    {res2['t_rewrite']:.4f}s (\" {res2['search_query']} \")")
    print(f"  • Retrieval:  {res2['t_retrieval']:.4f}s ({res2['hits_count']} hits)")
    print(f"  • Generation: {res2['t_generation']:.4f}s")
    print(f"  • Answer:     \"{res2['answer'][:80]}...\"")

    # Assertions
    assert res2["t_total"] < 0.10, f"Expected Turn 2 to take <100ms, took {res2['t_total']:.4f}s"
    assert res2["answer"] == res1["answer"]
    speedup = res1["t_total"] / max(res2["t_total"], 0.000001)
    print(f"\n🎉 SPEEDUP ON REPEATED QUESTION: {speedup:.1f}x faster! ({res1['t_total']:.2f}s -> {res2['t_total']:.4f}s)")


if __name__ == "__main__":
    test_repeated_question()
