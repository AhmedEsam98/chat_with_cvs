import logging
import re
from dataclasses import dataclass
from typing import Optional

import config
from cache_manager import get_redis_client, hash_key
from clients import get_openai_client

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a question router. If the question is related to a CV, resume, candidate, job experience, education, skills, qualifications, projects, or any information contained in a CV, classify it as CV-related.
If the question is not related to a CV or candidate information, classify it as not CV-related.

Return only one of the following labels:

CV_RELATED
NOT_CV_RELATED"""

GREETING_PATTERNS = [
    r"^(?:hi|hello|hey|greetings|hola|salam|marhaba)(?:\s+there|\s+all)?\b[!.?]*$",
    r"^good\s+(?:morning|afternoon|evening|day)\b[!.?]*$",
    r"^(?:how\s+are\s+you|how's\s+it\s+going|how\s+do\s+you\s+do)\b[!.?]*$",
    r"^(?:who\s+are\s+you|what\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|how\s+can\s+you\s+help(?:\s+me)?|help)\b[!.?]*$",
    r"^(?:thank\s+you|thanks|thx|thank\s+you\s+so\s+much|appreciate\s+it)\b[!.?]*$",
]

GREETING_REGEX = re.compile("|".join(GREETING_PATTERNS), re.IGNORECASE)

# Fast-path keywords that immediately confirm a query is CV-related (0ms, 0 API calls)
CV_INDICATOR_KEYWORDS = {
    # Candidates & resumes
    "candidate", "candidates", "applicant", "applicants", "cv", "cvs", "resume", "resumes",
    "profile", "profiles", "person", "someone", "anyone", "who", "whom", "whose", "which",
    # Professional attributes
    "skill", "skills", "experience", "experienced", "education", "degree", "graduated", "graduate",
    "university", "college", "school", "bachelor", "master", "phd", "courses", "course",
    "qualification", "qualifications", "certificate", "certificates", "certification", "certifications",
    "project", "projects", "portfolio", "technologies", "tools", "stack", "knowledge",
    # Work & Roles
    "job", "jobs", "role", "roles", "position", "positions", "title", "titles",
    "work", "worked", "working", "career", "employment", "company", "companies", "years",
    "junior", "senior", "lead", "intern", "trainee", "mid-level", "fresh",
    # Common tech/engineering roles
    "developer", "engineer", "designer", "architect", "analyst", "specialist",
    "python", "flutter", "dart", "react", "node", "ai", "ml", "backend", "frontend",
    "mobile", "devops", "cloud", "docker", "sql", "aws", "azure", "agentic", "data",
    # Comparisons & evaluations
    "compare", "comparison", "difference", "better", "best", "rank", "ranking",
    "hire", "hiring", "fit", "suitable", "recommend", "shortlist", "interview",
}

DEFAULT_REFUSAL_MESSAGE = (
    "I am an intelligent HR assistant specialized in analyzing the uploaded candidate CVs.\n\n"
    "Please ask questions related to the candidates, such as their work experience, technical skills, "
    "qualifications, projects, or role comparisons."
)

DEFAULT_GREETING_RESPONSE = (
    "Hello! 👋 I am your AI HR Assistant for reviewing and analyzing candidate CVs.\n\n"
    "You can ask me questions such as:\n"
    "• *Who has experience with Flutter or Python?*\n"
    "• *Which candidates hold an AI Developer position?*\n"
    "• *Compare the qualifications of the candidates.*"
)

DEFAULT_THANKS_RESPONSE = (
    "You're very welcome! Let me know if you need any more insights about the candidates or their qualifications. 😊"
)


@dataclass
class RouteResult:
    is_cv_related: bool
    is_greeting: bool = False
    direct_response: Optional[str] = None
    label: str = "CV_RELATED"
    matched_by: str = "fast_path"


def route_question(query: str, client=None) -> RouteResult:
    """Classify user query into CV_RELATED vs NOT_CV_RELATED using hybrid routing.

    1. Greetings Fast-Path (0ms): Instant friendly response for greetings and pleasantries.
    2. CV Keyword Fast-Path (0ms): Instant pass-through for obvious candidate queries.
    3. Cached LLM classifier: Low-latency classification for ambiguous or off-topic queries.
    """
    clean = query.strip()
    lower_clean = clean.lower()
    words = set(re.findall(r"\b\w+\b", lower_clean))

    # 1. Greetings Fast-Path (0ms)
    if GREETING_REGEX.match(lower_clean):
        if any(w in words for w in ["thank", "thanks", "thx", "appreciate"]):
            return RouteResult(
                is_cv_related=False,
                is_greeting=True,
                direct_response=DEFAULT_THANKS_RESPONSE,
                label="NOT_CV_RELATED",
                matched_by="greeting",
            )
        return RouteResult(
            is_cv_related=False,
            is_greeting=True,
            direct_response=DEFAULT_GREETING_RESPONSE,
            label="NOT_CV_RELATED",
            matched_by="greeting",
        )

    # 2. Obvious CV Keyword Fast-Path (0ms, 0 API calls)
    if words.intersection(CV_INDICATOR_KEYWORDS):
        return RouteResult(
            is_cv_related=True,
            is_greeting=False,
            label="CV_RELATED",
            matched_by="fast_path",
        )

    # 3. Check Redis/Memory Router Cache for this exact query
    cache_key = f"cv:route:{hash_key(lower_clean)}"
    redis_client = get_redis_client()
    if redis_client is not None:
        try:
            cached_label = redis_client.get(cache_key)
            if cached_label:
                is_cv = cached_label == "CV_RELATED"
                return RouteResult(
                    is_cv_related=is_cv,
                    direct_response=None if is_cv else DEFAULT_REFUSAL_MESSAGE,
                    label=cached_label,
                    matched_by="cache",
                )
        except Exception:
            pass

    # 4. LLM Classifier for Ambiguous / Truly Off-Topic Questions
    client = client or get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": clean},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        content = (resp.choices[0].message.content or "").strip().upper()
        if "NOT_CV_RELATED" in content:
            final_label = "NOT_CV_RELATED"
            is_cv = False
        else:
            final_label = "CV_RELATED"
            is_cv = True

        # Cache decision for 24 hours
        if redis_client is not None:
            try:
                redis_client.set(cache_key, final_label, ex=86400)
            except Exception:
                pass

        return RouteResult(
            is_cv_related=is_cv,
            direct_response=None if is_cv else DEFAULT_REFUSAL_MESSAGE,
            label=final_label,
            matched_by="llm_classifier",
        )

    except Exception as exc:
        logger.warning(f"Router classifier fallback on error: {exc}")
        # Default to CV_RELATED on API error so we never block legitimate queries
        return RouteResult(
            is_cv_related=True,
            label="CV_RELATED",
            matched_by="error_fallback",
        )
