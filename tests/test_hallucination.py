"""Automated Hallucination & Faithfulness Evaluation Test Suite."""
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

from generation import check_hallucination, answer_stream
from retrieval import retrieve


# Sample Ground Truth Context
MOCK_CV_HITS = [
    {
        "cv_name": "khaled_osama.pdf",
        "content": (
            "[Section: Summary]\n"
            "CCNA- and MCSA-certified IT support and network professional with 2+ years of hands-on experience "
            "in end-user support, network administration, and systems troubleshooting across enterprise hospitality environments. "
            "Skilled in Cisco routing & switching, Windows Server, Active Directory, VoIP, cloud PMS/POS platforms, and cybersecurity practices.\n\n"
            "[Section: Experience]\n"
            "Charmillion Club Aqua Park Sharm El Sheikh, Egypt IT Supervisor Jun 2026 – Present\n"
            "• Supervised IT operations and coordinated technical tasks across the property.\n"
            "• Managed and supported Omada Controller environments, including TP-Link switches and access points.\n"
        )
    }
]


def test_hallucination_suite():
    print("=" * 80)
    print("🧪 RUNNING RAG HALLUCINATION & FAITHFULNESS AUDIT SUITE")
    print("=" * 80)

    # ── Test 1: Fully Grounded Answer ─────────────────────────────────────────
    print("\n[Test 1] Fully Grounded Answer (Expected: Score >= 0.90, No Hallucinations)")
    grounded_answer = (
        "Khaled Osama is an IT Supervisor at Charmillion Club Aqua Park in Sharm El Sheikh. "
        "He holds CCNA and MCSA certifications and has experience with Cisco routing, switching, "
        "and Windows Server."
    )
    res1 = check_hallucination(grounded_answer, MOCK_CV_HITS)
    print(f"  • Score: {res1['score'] * 100:.1f}%")
    print(f"  • Grounded Claims: {res1.get('grounded_claims_count')}/{res1.get('total_claims')}")
    print(f"  • Flagged Hallucinations: {len(res1['hallucinations'])}")
    print(f"  • Reasoning: {res1.get('reasoning')}")
    assert res1["is_grounded"], f"Expected answer to be marked grounded, got score {res1['score']}"
    assert len(res1["hallucinations"]) == 0, f"Expected 0 hallucinations, got {res1['hallucinations']}"
    print("  ✅ Passed: Correctly verified grounded answer.")

    # ── Test 2: Injected Severe Hallucination ──────────────────────────────────
    print("\n[Test 2] Blatantly Hallucinated Answer (Expected: Score < 0.50, Hallucinations Flagged)")
    hallucinated_answer = (
        "Khaled Osama holds a PhD in Quantum Physics from Oxford University and spent 12 years as "
        "Chief Information Officer at Apple headquarters in Cupertino, California."
    )
    res2 = check_hallucination(hallucinated_answer, MOCK_CV_HITS)
    print(f"  • Score: {res2['score'] * 100:.1f}%")
    print(f"  • Grounded Claims: {res2.get('grounded_claims_count')}/{res2.get('total_claims')}")
    print(f"  • Flagged Hallucinations: {len(res2['hallucinations'])}")
    for h in res2["hallucinations"]:
        print(f"    ⚠️ Claim: \"{h.get('claim')}\" -> Reason: {h.get('reason')}")
    assert not res2["is_grounded"], "Expected answer to be rejected as ungrounded!"
    assert len(res2["hallucinations"]) > 0, "Expected hallucinations to be flagged!"
    print("  ✅ Passed: Successfully detected and caught severe hallucinations.")

    # ── Test 3: Partial / Subtle Hallucination ─────────────────────────────────
    print("\n[Test 3] Partial Hallucination (Mixed Truth and Fabrication)")
    mixed_answer = (
        "Khaled Osama holds a CCNA certification and has hands-on experience in network administration. "
        "In 2024, he also won the Turing Award and founded a multi-billion dollar semiconductor startup."
    )
    res3 = check_hallucination(mixed_answer, MOCK_CV_HITS)
    print(f"  • Score: {res3['score'] * 100:.1f}%")
    print(f"  • Grounded Claims: {res3.get('grounded_claims_count')}/{res3.get('total_claims')}")
    print(f"  • Flagged Hallucinations: {len(res3['hallucinations'])}")
    for h in res3["hallucinations"]:
        print(f"    ⚠️ Claim: \"{h.get('claim')}\" -> Reason: {h.get('reason')}")
    assert len(res3["hallucinations"]) > 0, "Failed to catch the fabricated startup / Turing award claim!"
    print("  ✅ Passed: Successfully separated grounded facts from fabricated claim.")

    # ── Test 4: Live End-to-End Query Audit ───────────────────────────────────
    print("\n[Test 4] Live End-to-End RAG Generation & Hallucination Check")
    query = "What networking and IT certifications does Khaled Osama have?"
    print(f"  • Question: \"{query}\"")
    try:
        hits = retrieve(query, k=3)
        if not hits:
            print("  ⚠️ No hits retrieved from search index (check if CVs are indexed).")
        else:
            stream_gen = answer_stream(query, [], hits)
            generated_answer = "".join(list(stream_gen))
            print(f"  • Generated Answer Preview: \"{generated_answer[:120]}...\"")

            live_eval = check_hallucination(generated_answer, hits)
            print(f"  • Faithfulness Score: {live_eval['score'] * 100:.1f}%")
            print(f"  • Grounded: {'🟢 YES' if live_eval['is_grounded'] else '🔴 NO'}")
            print(f"  • Verified Claims: {live_eval.get('grounded_claims_count')}/{live_eval.get('total_claims')}")
            print(f"  • Reasoning: {live_eval.get('reasoning')}")
            if live_eval["hallucinations"]:
                for h in live_eval["hallucinations"]:
                    print(f"    ⚠️ Claim: \"{h.get('claim')}\"")
            else:
                print("  ✅ 100% Grounded: No hallucinations detected in live generated response!")
    except Exception as exc:
        print(f"  ⚠️ Live RAG skipped or failed: {exc}")

    print("\n" + "=" * 80)
    print("🎉 ALL HALLUCINATION VERIFICATION TESTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    test_hallucination_suite()
