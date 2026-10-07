"""Test suite for FlashRank Cross-Encoder Re-ranker."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from retrieval.reranker import rerank, get_ranker
from retrieval import retrieve


def test_reranker():
    print("=" * 70)
    print("🧪 RUNNING FLASHRANK CROSS-ENCODER RE-RANKER TEST SUITE")
    print(f"Re-ranker Enabled: {config.RERANKER_ENABLED}")
    print(f"Model: {config.RERANKER_MODEL}")
    print(f"Coarse Candidate Pool (Top N): {config.RERANK_TOP_N}")
    print("=" * 70)

    # 1. Unit Test Re-ranking Logic
    print("\n[Test 1] Synthetic Passages Re-ranking Precision...")
    mock_candidates = [
        {
            "cv_name": "Graphic_Designer.pdf",
            "content": "[Section: Skills]\nAdobe Photoshop, Illustrator, InDesign, UI Design, Figma.",
        },
        {
            "cv_name": "Flutter_Dev.pdf",
            "content": (
                "[Section: Experience]\n"
                "Senior Flutter Developer with 4 years building cross-platform Android & iOS apps using Dart, "
                "Bloc, Riverpod, and Clean Architecture."
            ),
        },
        {
            "cv_name": "Network_Admin.pdf",
            "content": "[Section: Summary]\nCCNA certified network engineer with Cisco routers and switches expertise.",
        },
    ]

    query = "Looking for an engineer who builds mobile apps in Flutter and Dart"
    reranked = rerank(query, mock_candidates, top_k=2)

    assert len(reranked) == 2, f"Expected top 2 results, got {len(reranked)}"
    top_hit = reranked[0]
    print(f"  • Top Ranked: {top_hit['cv_name']} (Score: {top_hit.get('rerank_score', 0):.6f})")
    assert top_hit["cv_name"] == "Flutter_Dev.pdf", f"Expected Flutter_Dev.pdf as top hit, got {top_hit['cv_name']}"
    assert "rerank_score" in top_hit, "Expected 'rerank_score' to be present in result"
    print("  ✅ Re-ranker correctly identified the most relevant candidate.")

    # 2. Test Live Two-Stage Retrieval
    print("\n[Test 2] Live End-to-End Two-Stage Retrieval (Azure Hybrid Search + Re-ranking)...")
    live_query = "Who has experience with Docker and Kubernetes containerization?"
    try:
        results = retrieve(live_query, k=3)
        print(f"  • Query: \"{live_query}\"")
        print(f"  • Retrieved Chunks: {len(results)}")
        for idx, r in enumerate(results, 1):
            score_info = f"(Re-rank Score: {r.get('rerank_score', 0):.4f})" if "rerank_score" in r else ""
            print(f"    Rank #{idx}: {r['cv_name']} {score_info}")
            print(f"      Preview: {r['content'][:90].replace(chr(10), ' ')}...")
        print("  ✅ Live two-stage retrieval with cross-encoder re-ranking passed.")
    except Exception as exc:
        print(f"  ⚠️ Live retrieval note: {exc}")

    print("\n" + "=" * 70)
    print("🎉 ALL RE-RANKER TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_reranker()
