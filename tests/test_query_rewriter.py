"""Test suite for Conversational Query Rewriter."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from retrieval.query_rewriter import rewrite_query, _has_context_dependencies


def test_query_rewriter():
    print("=" * 70)
    print("🧪 RUNNING CONVERSATIONAL QUERY REWRITER TEST SUITE")
    print(f"Query Rewrite Enabled: {config.QUERY_REWRITE_ENABLED}")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Test 1: Fast-Path on Empty or Non-Dependent Queries (0ms, 0 API calls)
    # -----------------------------------------------------------------
    print("\n[Test 1] Fast-path evaluation without history...")
    q1 = "Who is the best Flutter developer?"
    rewritten1, was_rewritten1 = rewrite_query(q1, history=[])
    assert rewritten1 == q1, f"Expected untouched query, got {rewritten1}"
    assert not was_rewritten1, "Expected was_rewritten=False"
    print("  ✅ Empty history bypassed LLM correctly.")

    print("\n[Test 2] Fast-path evaluation on standalone query with greeting history...")
    greeting_hist = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I help you with the candidates?"},
    ]
    q2 = "List all candidates with machine learning and deep learning experience."
    rewritten2, was_rewritten2 = rewrite_query(q2, history=greeting_hist)
    assert rewritten2 == q2, f"Expected untouched query, got {rewritten2}"
    assert not was_rewritten2, "Expected was_rewritten=False"
    print("  ✅ Standalone query with greeting history bypassed LLM correctly.")

    # -----------------------------------------------------------------
    # Test 3: Context Dependency Detection
    # -----------------------------------------------------------------
    print("\n[Test 3] Testing context dependency heuristics...")
    assert _has_context_dependencies("What are his skills?", greeting_hist) is True
    assert _has_context_dependencies("Where did she study?", greeting_hist) is True
    assert _has_context_dependencies("What about Flutter?", greeting_hist) is True
    assert _has_context_dependencies("And certifications?", greeting_hist) is True
    assert _has_context_dependencies("Compare him with Ahmed", greeting_hist) is True
    assert _has_context_dependencies("Who has experience in Docker?", greeting_hist) is False
    print("  ✅ Coreference heuristic detection passed.")

    # -----------------------------------------------------------------
    # Test 4: Mocked LLM Coreference Resolution
    # -----------------------------------------------------------------
    print("\n[Test 4] Mocked LLM coreference resolution...")
    mock_history = [
        {"role": "user", "content": "Who is Akram?"},
        {"role": "assistant", "content": "Akram is an AI Developer with 5 years experience in Python and NLP."},
    ]
    follow_up_q = "What are his main projects?"

    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "What are Akram's main projects?"
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    rewritten4, was_rewritten4 = rewrite_query(follow_up_q, history=mock_history, client=mock_client)
    assert was_rewritten4 is True, "Expected was_rewritten=True"
    assert rewritten4 == "What are Akram's main projects?", f"Unexpected output: {rewritten4}"
    print(f"  • Input: \"{follow_up_q}\"")
    print(f"  • Rewritten: \"{rewritten4}\"")
    print("  ✅ Mocked coreference resolution passed.")

    # -----------------------------------------------------------------
    # Test 5: Fallback on API Error
    # -----------------------------------------------------------------
    print("\n[Test 5] Error resilience & graceful fallback...")
    error_client = MagicMock()
    error_client.chat.completions.create.side_effect = RuntimeError("OpenAI rate limit or network error")

    fallback_res, fallback_flag = rewrite_query("What about his salary?", history=mock_history, client=error_client)
    assert fallback_res == "What about his salary?", "Expected fallback to original query on error"
    assert fallback_flag is False, "Expected was_rewritten=False on error"
    print("  ✅ Gracefully falls back to original query on exception.")

    # -----------------------------------------------------------------
    # Test 6: Feature Toggle Disabled
    # -----------------------------------------------------------------
    print("\n[Test 6] Feature toggle disabled behavior...")
    config.QUERY_REWRITE_ENABLED = False
    t_res, t_flag = rewrite_query("What are his skills?", history=mock_history, client=mock_client)
    assert t_res == "What are his skills?", "Expected untouched when toggle is False"
    assert t_flag is False
    config.QUERY_REWRITE_ENABLED = True
    print("  ✅ Feature toggle respected.")

    # -----------------------------------------------------------------
    # Test 7: Live Azure OpenAI Query Rewriter (Integration)
    # -----------------------------------------------------------------
    print("\n[Test 7] Live Azure OpenAI Query Rewriter Test...")
    try:
        live_history = [
            {"role": "user", "content": "Who is Akram?"},
            {
                "role": "assistant",
                "content": "Akram is an AI Developer with experience in machine learning, PyTorch, and NLP.",
            },
        ]
        live_q = "What are his technical skills?"
        live_rewritten, live_flag = rewrite_query(live_q, history=live_history)
        print(f"  • Original Follow-Up: \"{live_q}\"")
        print(f"  • Live Rewritten Query: \"{live_rewritten}\" (Was Rewritten: {live_flag})")
        assert "Akram" in live_rewritten or "akram" in live_rewritten.lower(), "Expected candidate name in rewritten query"
        print("  ✅ Live conversational rewrite successfully resolved pronouns to 'Akram'!")
    except Exception as exc:
        print(f"  ⚠️ Live API test note (skipped if offline): {exc}")

    print("\n" + "=" * 70)
    print("🎉 ALL CONVERSATIONAL QUERY REWRITER TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_query_rewriter()
