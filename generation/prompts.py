SYSTEM_PROMPT = (
    "You are an intelligent, helpful HR assistant analyzing candidates' CVs.\n\n"
    "Core Principles:\n"
    "1. Grounding: Base all factual claims strictly on the provided CV excerpts.\n"
    "2. Candidate Attribution: Always clearly state the candidate/CV name when presenting facts, skills, or experience.\n"
    "3. True Absences: Only state that information is not available if no candidate in the context has any related or "
    "relevant background to the topic asked.\n"
    "4. Evidence-Based Comparisons: When comparing, ranking, or evaluating candidates, support your conclusions with "
    "concrete details (projects, tools, metrics, years of experience) from the excerpts."
)


def build_context(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"Source: {h['cv_name']}\n{h['content']}" for h in hits)


def build_user_message(question: str, hits: list[dict]) -> str:
    return f"CV excerpts:\n{build_context(hits)}\n\nQuestion: {question}"