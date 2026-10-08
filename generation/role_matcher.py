import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionValidationResult:
    is_position_query: bool
    target_role: Optional[str] = None
    has_matching_candidates: bool = False
    matching_candidates: Optional[list[dict]] = None
    rejection_message: Optional[str] = None


# Taxonomy mapping broad role categories to recognized specializations, sub-fields, and titles
ROLE_TAXONOMY: dict[str, list[str]] = {
    "ai developer": [
        "agentic ai developer",
        "ai developer",
        "ai engineer",
        "agentic ai engineer",
        "junior ai engineer",
        "machine learning engineer",
    ],
    "ai engineer": [
        "agentic ai engineer",
        "ai engineer",
        "junior ai engineer",
        "agentic ai developer",
        "ai developer",
        "machine learning engineer",
    ],
    "machine learning engineer": [
        "machine learning engineer",
        "ai & machine learning engineer",
        "ml engineer",
        "ai engineer",
    ],
    "mobile developer": [
        "flutter developer",
        "ios developer",
        "android developer",
        "mobile developer",
        "mobile app developer",
    ],
    "flutter developer": [
        "flutter developer",
        "junior flutter developer",
        "mid-senior flutter developer",
    ],
    "backend developer": [
        "backend developer",
        "node.js developer",
        "python developer",
        "django developer",
    ],
    "frontend developer": [
        "frontend developer",
        "react developer",
        "web developer",
    ],
    "software engineer": [
        "software engineer",
        "developer",
        "flutter developer",
        "backend developer",
        "frontend developer",
        "ai developer",
    ],
    "graphic designer": [
        "graphic designer",
        "ui/ux designer",
    ],
}

# Recognized role nouns that confirm a target noun phrase is an occupational title
ROLE_NOUNS: set[str] = {
    "developer", "dev", "engineer", "designer", "architect", "manager", "lead", "leader",
    "specialist", "analyst", "consultant", "administrator", "admin", "scientist",
    "officer", "intern", "trainee", "programmer", "coder", "tester", "qa",
    "director", "coordinator", "technician", "expert", "executive", "assistant"
}

# Explicit regexes where the user explicitly asks for a position / role
EXPLICIT_POSITION_PATTERNS = [
    re.compile(r"^who\s+(?:holds|has|works\s+in)\s+(?:the\s+)?(?:position|role|job)\s+of\s+(.+)$", re.IGNORECASE),
    re.compile(r"^who\s+works\s+as\s+(?:an?\s+)?(.+?)(?:\s+position|\s+role|\s+job)?$", re.IGNORECASE),
    re.compile(r"^is\s+there\s+(?:an?\s+)?(.+?)(?:\s+position|\s+role|\s+job)$", re.IGNORECASE),
    re.compile(r"^(?:which\s+candidate|candidates)\s+(?:holds?|has|have)\s+(?:the\s+)?(?:position|role|job)\s+of\s+(.+)$", re.IGNORECASE),
]

# Implicit regexes that require a role article ("a", "an", "the") or candidate reference
IMPLICIT_POSITION_PATTERNS = [
    re.compile(r"^who\s+(?:is|are)\s+(?:an?|the)\s+(.+?)(?:\s+position|\s+role|\s+job)?$", re.IGNORECASE),
    re.compile(r"^is\s+there\s+(?:an?)\s+(.+?)$", re.IGNORECASE),
    re.compile(r"^(?:which\s+candidate|candidates)\s+(?:is|are)\s+(?:an?|the)\s+(.+?)$", re.IGNORECASE),
]

# Patterns that indicate a skill/technology or general question instead of a position
SKILL_INDICATORS = {
    "know", "knows", "experience", "experienced", "skills", "skill", "worked with",
    "using", "used", "study", "studied", "graduated", "education", "project", "projects",
    "background", "compare", "summary", "summarize", "tell me about"
}


def _clean_role(raw: str) -> str:
    """Normalize extracted role string."""
    r = raw.strip().lower()
    r = re.sub(r"^(?:an?|the)\s+", "", r)
    r = re.sub(r"\s+(?:position|role|job)$", "", r)
    return r.strip()


def extract_candidate_titles(hits: list[dict]) -> dict[str, set[str]]:
    """Extract candidate names and their mentioned job titles from retrieved chunks."""
    candidate_titles: dict[str, set[str]] = {}

    title_patterns = [
        re.compile(r"(?:Trainee\s*[–\-]\s*)?Agentic AI Developer(?:\s+(?:Intern|Trainee))?", re.IGNORECASE),
        re.compile(r"AI\s*&\s*Machine Learning Engineer", re.IGNORECASE),
        re.compile(r"(?:Junior\s+)?AI Engineer", re.IGNORECASE),
        re.compile(r"(?:Mid-Senior\s+|Junior\s+|Freelance\s+)?Flutter Developer", re.IGNORECASE),
        re.compile(r"Graphic Designer", re.IGNORECASE),
        re.compile(r"(?:Backend|Node\.js|Python)\s+Developer", re.IGNORECASE),
        re.compile(r"Frontend Developer", re.IGNORECASE),
        re.compile(r"Software Engineer", re.IGNORECASE),
    ]

    for h in hits:
        cv_name = h.get("cv_name", "Unknown")
        content = h.get("content", "")
        if cv_name not in candidate_titles:
            candidate_titles[cv_name] = set()

        for pat in title_patterns:
            for match in pat.finditer(content):
                matched_title = match.group(0).strip()
                if matched_title:
                    candidate_titles[cv_name].add(matched_title)

    return candidate_titles


def validate_position_query(question: str, hits: list[dict]) -> PositionValidationResult:
    """Validate whether the user's query is asking for a position and whether any candidate holds it."""
    clean_q = question.strip().lower()

    # If question contains skill/action verbs like "knows Python" or "experience with RAG", not a strict position query
    for ind in SKILL_INDICATORS:
        if f" {ind} " in f" {clean_q} " or clean_q.startswith(f"{ind} "):
            return PositionValidationResult(is_position_query=False)

    # Collect candidate names from hits to avoid mistaking candidates for job titles
    candidate_name_tokens: set[str] = set()
    for h in hits:
        raw_name = (h.get("candidate_name") or h.get("cv_name") or "").strip().lower()
        if raw_name:
            cleaned_name = re.sub(r"\.(?:pdf|docx|txt)$", "", raw_name, flags=re.IGNORECASE)
            cleaned_name = re.sub(r"[\(\)]", "", cleaned_name).strip()
            candidate_name_tokens.add(cleaned_name)
            for part in re.split(r"[\s_\-]+", cleaned_name):
                part = part.strip()
                if len(part) >= 2 and part not in ROLE_NOUNS and part not in {"cv", "resume"}:
                    candidate_name_tokens.add(part)

    target_role = None
    is_explicit_position = False

    # 1. Check explicit position patterns first
    for pattern in EXPLICIT_POSITION_PATTERNS:
        m = pattern.match(clean_q)
        if m:
            cand = _clean_role(m.group(1))
            if cand and len(cand.split()) <= 6:
                target_role = cand
                is_explicit_position = True
                break

    # 2. Check implicit position patterns
    if not target_role:
        for pattern in IMPLICIT_POSITION_PATTERNS:
            m = pattern.match(clean_q)
            if m:
                cand = _clean_role(m.group(1))
                if cand and len(cand.split()) <= 6:
                    target_role = cand
                    break

    if not target_role:
        return PositionValidationResult(is_position_query=False)

    # If target matches a candidate's name, it's a person inquiry, NOT a position query
    if target_role in candidate_name_tokens:
        return PositionValidationResult(is_position_query=False)

    # If the extracted target is clearly a skill/tool rather than a role (e.g. "python", "docker", "flutter" alone)
    tech_tools = {"python", "docker", "flutter", "dart", "react", "sql", "git", "rag", "langchain"}
    if target_role in tech_tools:
        return PositionValidationResult(is_position_query=False)

    # For implicit queries (e.g. "who is the X"), confirm target actually looks like an occupational role
    if not is_explicit_position:
        words = set(target_role.split())
        has_role_noun = bool(words.intersection(ROLE_NOUNS))
        in_taxonomy = target_role in ROLE_TAXONOMY
        if not (has_role_noun or in_taxonomy):
            # Not an occupational position query (likely asking about a person, comparative superlative, etc.)
            return PositionValidationResult(is_position_query=False)

    # Extract all candidate titles from retrieved chunks
    candidates_map = extract_candidate_titles(hits)

    # Find matching candidates
    matching_candidates: list[dict] = []
    related_titles_found: set[str] = set()

    for cv_name, titles in candidates_map.items():
        for title in titles:
            t_lower = title.lower()
            related_titles_found.add(title)

            # 1. Exact or normalized equality
            if target_role == t_lower:
                matching_candidates.append({"cv_name": cv_name, "title": title})
                continue

            # 2. Taxonomy mapping: Does target_role map to this title?
            if target_role in ROLE_TAXONOMY:
                acceptable_subtitles = ROLE_TAXONOMY[target_role]
                if any(sub in t_lower for sub in acceptable_subtitles):
                    matching_candidates.append({"cv_name": cv_name, "title": title})
                    continue

            # 3. Substring/hierarchy matching:
            # If target role words are all contained in candidate's title without conflicting modifiers
            target_words = set(target_role.split())
            title_words = set(t_lower.replace("-", " ").replace("–", " ").split())
            if target_words.issubset(title_words):
                matching_candidates.append({"cv_name": cv_name, "title": title})

    if matching_candidates:
        return PositionValidationResult(
            is_position_query=True,
            target_role=target_role,
            has_matching_candidates=True,
            matching_candidates=matching_candidates,
        )

    # If no candidate holds this specific position, craft an accurate deterministic rejection
    target_display = target_role.title()
    related_bullets = ""
    if related_titles_found:
        bullet_list = [f"- {t}" for t in sorted(related_titles_found)[:5]]
        related_bullets = (
            f"\n\nThe provided CVs include candidates with related positions such as:\n"
            + "\n".join(bullet_list)
            + f"\n\nHowever, none of them hold the specific position of **{target_display}**."
        )

    rejection_msg = (
        f"None of the candidates in the uploaded CVs hold the position of **{target_display}**."
        f"{related_bullets}"
    )

    return PositionValidationResult(
        is_position_query=True,
        target_role=target_role,
        has_matching_candidates=False,
        rejection_message=rejection_msg,
    )
