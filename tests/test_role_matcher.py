import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from generation.role_matcher import validate_position_query

MOCK_HITS = [
    {
        "cv_name": "Akram_Mohamed_Hassan_FlowCV_Resume_2026-03-25.pdf",
        "candidate_name": "Akram Mohamed Hassan",
        "content": (
            "[Section: Contact & Overview]\n"
            "Akram Mohamed Hassan\n"
            "AI Engineer\n"
            "Agentic AI Developer Trainee\n"
        ),
        "job_titles": ["AI Engineer", "Agentic AI Developer Trainee"],
    },
    {
        "cv_name": "Abdelrahman_Karawia_Flutter_CV.pdf",
        "candidate_name": "Abdelrahman Karawia",
        "content": (
            "[Section: Summary]\n"
            "Abdelrahman Karawia\n"
            "Mid-Senior Flutter Developer with 3+ years experience.\n"
        ),
        "job_titles": ["Flutter Developer"],
    },
    {
        "cv_name": "Hossam_Elshami_CV Graphic Designer.pdf",
        "candidate_name": "Hossam Elshami",
        "content": (
            "[Section: Profile]\n"
            "Hossam Elshami\n"
            "Graphic Designer and UI/UX Designer\n"
        ),
        "job_titles": ["Graphic Designer"],
    },
]


def test_candidate_name_queries_not_position():
    print("\n--- Test 1: Candidate Name Queries (Should NOT trigger position gate) ---")
    candidate_queries = [
        "who is akram",
        "who is Akram",
        "who is Akram Mohamed Hassan?",
        "who is abdelrahman",
        "who is Abdelrahman Karawia",
        "who is sherif",
        "tell me about akram",
        "what is akram's experience?",
    ]

    for q in candidate_queries:
        res = validate_position_query(q, MOCK_HITS)
        print(f"Query: \"{q}\" -> is_position_query={res.is_position_query}")
        assert not res.is_position_query, (
            f"Expected is_position_query=False for query '{q}', got target_role='{res.target_role}'"
        )
    print("✅ All candidate name queries successfully bypassed position gate!")


def test_valid_positions_with_matches():
    print("\n--- Test 2: Valid Position Queries With Matching Candidates ---")
    pos_queries = [
        ("who is the flutter developer", "Abdelrahman Karawia"),
        ("who is the ai engineer", "Akram Mohamed Hassan"),
        ("is there a graphic designer?", "Hossam Elshami"),
        ("who works as a flutter developer", "Abdelrahman Karawia"),
    ]

    for q, expected_cand in pos_queries:
        res = validate_position_query(q, MOCK_HITS)
        print(f"Query: \"{q}\" -> is_pos={res.is_position_query}, matched={res.has_matching_candidates}, cands={[c['cv_name'] for c in res.matching_candidates or []]}")
        assert res.is_position_query, f"Expected is_position_query=True for '{q}'"
        assert res.has_matching_candidates, f"Expected has_matching_candidates=True for '{q}'"
        matched_cvs = " ".join(c["cv_name"] for c in (res.matching_candidates or []))
        assert expected_cand.split()[0].lower() in matched_cvs.lower(), (
            f"Expected candidate '{expected_cand}' in matches for '{q}', got {matched_cvs}"
        )
    print("✅ All matching position queries verified successfully!")


def test_valid_positions_with_no_matches():
    print("\n--- Test 3: Genuine Position Queries With No Matches (Rejection) ---")
    absent_roles = [
        "is there a product manager?",
        "who holds the position of devops engineer",
        "who is the scrum master",
    ]

    for q in absent_roles:
        res = validate_position_query(q, MOCK_HITS)
        print(f"Query: \"{q}\" -> is_pos={res.is_position_query}, matched={res.has_matching_candidates}, rejection={res.rejection_message[:50] if res.rejection_message else 'None'}...")
        assert res.is_position_query, f"Expected is_position_query=True for '{q}'"
        assert not res.has_matching_candidates, f"Expected has_matching_candidates=False for '{q}'"
        assert res.rejection_message is not None, f"Expected rejection_message for '{q}'"
        assert "None of the candidates" in res.rejection_message
    print("✅ Genuine absent position queries correctly generated rejection message!")


def test_general_and_skill_queries():
    print("\n--- Test 4: General and Skill Queries (Should NOT trigger position gate) ---")
    general_queries = [
        "who is the best candidate?",
        "who has python experience",
        "who knows docker",
        "which candidate has the highest gpa",
    ]

    for q in general_queries:
        res = validate_position_query(q, MOCK_HITS)
        print(f"Query: \"{q}\" -> is_position_query={res.is_position_query}")
        assert not res.is_position_query, f"Expected is_position_query=False for '{q}'"
    print("✅ All general/skill queries bypassed position gate cleanly!")


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Running Role Matcher / Position Validation Test Suite")
    print("=" * 70)
    test_candidate_name_queries_not_position()
    test_valid_positions_with_matches()
    test_valid_positions_with_no_matches()
    test_general_and_skill_queries()
    print("\n" + "=" * 70)
    print("🎉 ALL ROLE MATCHER TESTS PASSED!")
    print("=" * 70)
