import logging
import re
from typing import Any, Optional

import config
from cache_manager import (
    cache_get,
    cache_set,
    hash_key,
    normalize_query_key,
)
from clients import get_openai_client

from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABEL_CV_RELATED = "CV_RELATED"
LABEL_NOT_CV_RELATED = "NOT_CV_RELATED"

ROUTER_SYSTEM_PROMPT = """You are a question router for an HR candidate CV/Resume retrieval system.
If the question is related to a CV, resume, candidate, job experience, education, skills, qualifications, projects, or any information contained in a CV, classify it as CV-related.
If the question is not related to a CV or candidate information, classify it as not CV-related.

Guidelines:
1. Questions asking about candidates, specific people, qualifications, job roles, skills, degrees, or projects are CV_RELATED.
2. Questions asking about general knowledge, trivia, cooking, weather, math, jokes, or non-CV topics are NOT_CV_RELATED.

Return only one of the following labels:

CV_RELATED
NOT_CV_RELATED
"""

GREETINGS = {
    "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
    "greetings", "howdy", "sup", "yo", "welcome",
}

CV_FAST_KEYWORDS = {
    "cv", "cvs", "resume", "resumes", "candidate", "candidates", "applicant", "applicants",
    "experience", "experienced", "education", "degree", "university", "college", "school",
    "gpa", "bachelor", "master", "phd", "graduated", "graduation", "study", "studied",
    "skills", "skill", "technologies", "tech stack", "tools", "projects", "project",
    "certification", "certifications", "certified", "courses", "intern", "internship",
    "trainee", "hire", "hiring", "job", "position", "role", "work", "worked", "employment",
    "developer", "engineer", "designer", "architect", "manager", "lead", "supervisor",
    "flutter", "python", "dart", "react", "docker", "rag", "langchain", "sql", "ai", "ml",
}

_CACHED_CANDIDATE_NAMES: Optional[set[str]] = None


def _extract_name_tokens(filename: str) -> set[str]:
    base = filename.rsplit(".", 1)[0]
    base = re.sub(r"[\(\)]", " ", base)
    base = re.sub(r"([a-z])([A-Z])", r"\1 \2", base)
    base = re.sub(r"[-_\d]+", " ", base)
    base = re.sub(
        r"\b(?:CV|Resume|FlowCV|Flow|Graphic|Designer|Flutter|Developer|Backend|Frontend|AI|Engineer)\b",
        " ", base, flags=re.IGNORECASE)
    base = re.sub(r"\s+", " ", base).strip().lower()
    tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", base))
    if base:
        tokens.add(base)
    return tokens


def get_known_candidate_names() -> set[str]:
    """Collect candidate names from data/ directory."""
    global _CACHED_CANDIDATE_NAMES
    if _CACHED_CANDIDATE_NAMES is not None:
        return _CACHED_CANDIDATE_NAMES

    names: set[str] = set()
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        for f in data_dir.glob("*"):
            if f.is_file() and f.suffix.lower() in (".pdf", ".docx", ".txt"):
                names.update(_extract_name_tokens(f.name))

    _CACHED_CANDIDATE_NAMES = names
    return names


def _build_history_summary(history: list[dict], max_turns: int = 3) -> str:
    """Format recent turns to provide context for follow-up questions."""
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = m.get("content", "").strip()
        if role == "Assistant" and len(content) > 200:
            content = content[:200] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def classify_question(
    question: str,
    history: Optional[list[dict]] = None,
    client: Optional[Any] = None,
) -> str:
    """Classify whether a user's question is CV-related or not.

    Returns:
        "CV_RELATED" or "NOT_CV_RELATED"
    """
    if not question or not question.strip():
        return LABEL_NOT_CV_RELATED

    raw_query = question.strip()
    clean_q = re.sub(r"[?!.,;]+$", "", raw_query.lower()).strip()

    # If router is disabled via config, fail open to CV_RELATED
    if not getattr(config, "QUESTION_ROUTER_ENABLED", True):
        return LABEL_CV_RELATED

    # 1. Fast-path: Greetings without other words are NOT_CV_RELATED
    if clean_q in GREETINGS:
        return LABEL_NOT_CV_RELATED

    # 2. Fast-path: Candidate names or clear CV/technical keywords
    words_3 = set(re.findall(r"\b[a-z]{3,}\b", clean_q))
    known_cands = get_known_candidate_names()
    if words_3 & known_cands:
        return LABEL_CV_RELATED

    words_all = set(re.findall(r"\b[a-z]{2,}\b", clean_q))
    if words_all & CV_FAST_KEYWORDS:
        return LABEL_CV_RELATED

    # 3. Cache lookup
    history_summary = _build_history_summary(history or [])
    norm_q = normalize_query_key(raw_query)
    cache_key = f"cv:router:{hash_key(norm_q, history_summary)}"

    cached_label = cache_get(cache_key)
    if cached_label in (LABEL_CV_RELATED, LABEL_NOT_CV_RELATED):
        return cached_label

    # 4. LLM Classification via Azure OpenAI
    prompt_lines = []
    if history_summary:
        prompt_lines.append(f"Conversation Context:\n{history_summary}\n")
    prompt_lines.append(f"Question to Classify:\n{raw_query}")
    user_prompt = "\n".join(prompt_lines)

    try:
        openai_client = client or get_openai_client()
        response = openai_client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw_output = response.choices[0].message.content.strip().upper()
        if LABEL_NOT_CV_RELATED in raw_output:
            label = LABEL_NOT_CV_RELATED
        elif LABEL_CV_RELATED in raw_output:
            label = LABEL_CV_RELATED
        else:
            # Default to CV_RELATED on ambiguous output
            label = LABEL_CV_RELATED

        # Cache result
        cache_set(cache_key, label, ttl=getattr(config, "REDIS_CACHE_TTL", 3600))
        return label

    except Exception as exc:
        logger.warning(f"Question router LLM call failed ({exc}). Failing open to {LABEL_CV_RELATED}.")
        return LABEL_CV_RELATED


def is_cv_related(question: str, history: Optional[list[dict]] = None) -> bool:
    """Convenience boolean helper."""
    return classify_question(question, history=history) == LABEL_CV_RELATED


def get_out_of_scope_message(question: str) -> str:
    """Friendly response message when a question is classified as NOT_CV_RELATED."""
    clean = re.sub(r"[?!.,;]+$", "", question.strip().lower()).strip()
    if clean in GREETINGS:
        return (
            "👋 **Hello! Welcome to Chat with CVs.**\n\n"
            "I am your dedicated HR assistant for analyzing the uploaded candidate CVs and resumes.\n\n"
            "You can ask me questions like:\n"
            "- *\"Who is Akram?\"*\n"
            "- *\"Who is the Flutter developer?\"*\n"
            "- *\"Which candidates have experience with Docker or Machine Learning?\"*\n"
            "- *\"Compare the AI engineers on our team.\"*\n"
            "- *\"What degrees and certifications do the candidates have?\"*"
        )
    return (
        "⚠️ **Out of Scope Question**\n\n"
        "I am an HR assistant designed exclusively to analyze and answer questions based on the "
        "**uploaded candidate CVs and resumes**.\n\n"
        "Your question does not appear to be related to candidate qualifications, work experience, "
        "technical skills, education, or projects.\n\n"
        "Please ask a question related to the candidates or their CVs to continue."
    )
