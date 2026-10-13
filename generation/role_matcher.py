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
    "scrum master": [
        "scrum master",
        "agile coach",
        "project manager",
    ],
    "devops engineer": [
        "devops engineer",
        "site reliability engineer",
        "sre",
        "cloud engineer",
    ],
    "product manager": [
        "product manager",
        "product owner",
    ],
    "data scientist": [
        "data scientist",
        "machine learning engineer",
        "data analyst",
    ],
    "data analyst": [
        "data analyst",
        "business intelligence analyst",
        "bi analyst",
    ],
    "it supervisor": [
        "it supervisor",
        "it support",
        "network administrator",
        "systems administrator",
    ],
    "network administrator": [
        "network administrator",
        "it supervisor",
        "it support",
        "network engineer",
    ],
}

# Explicit position query patterns: user explicitly used position/role/job/works as keywords
EXPLICIT_POSITION_PATTERNS = [
    # "who holds/has the position of ...", "who is in the position/role of ..."
    re.compile(r"^who\s+(?:is|are|has|holds)\s+(?:in\s+)?(?:the\s+)?(?:position|role|job)\s+of\s+(.+?)[?.!]*$", re.IGNORECASE),
    # "who holds/has the ... position/role/job"
    re.compile(r"^who\s+(?:holds|has)\s+(?:the\s+)?(.+?)\s+(?:position|role|job)[?.!]*$", re.IGNORECASE),
    # "who works as an? ..."
    re.compile(r"^who\s+works\s+as\s+(?:an?\s+)?(.+?)(?:\s+(?:position|role|job))?[?.!]*$", re.IGNORECASE),
    # "is there a/an ... position/role/job"
    re.compile(r"^is\s+there\s+(?:an?\s+)?(.+?)\s+(?:position|role|job)[?.!]*$", re.IGNORECASE),
    # "which candidate holds the position of ...", "which candidate has the ... position/role/job"
    re.compile(r"^(?:which\s+candidate|candidates)\s+(?:is|are|has|have|holds?)\s+(?:the\s+)?(?:position|role|job)\s+of\s+(.+?)[?.!]*$", re.IGNORECASE),
    re.compile(r"^(?:which\s+candidate|candidates)\s+(?:is|are|has|have|holds?)\s+(?:the\s+)?(.+?)\s+(?:position|role|job)[?.!]*$", re.IGNORECASE),
    # "do we have a/an ... position/role/job"
    re.compile(r"^(?:do\s+we\s+have|are\s+there)\s+(?:an?\s+)?(.+?)\s+(?:position|role|job)[?.!]*$", re.IGNORECASE),
]

# Implicit position query patterns: user asks "who is the ...", "is there a ...", "which candidate is the ...", etc.
# These MUST be validated against recognized roles (not people, names, or general questions).
IMPLICIT_POSITION_PATTERNS = [
    re.compile(r"^who\s+(?:is|are)\s+(?:the|an?)\s+(.+?)[?.!]*$", re.IGNORECASE),
    re.compile(r"^who\s+(?:is|are)\s+(.+?)[?.!]*$", re.IGNORECASE),
    re.compile(r"^is\s+there\s+(?:an?)\s+(.+?)[?.!]*$", re.IGNORECASE),
    re.compile(r"^(?:which\s+candidate|candidates)\s+(?:is|are)\s+(?:the|an?)\s+(.+?)[?.!]*$", re.IGNORECASE),
    re.compile(r"^(?:do\s+we\s+have|are\s+there)\s+(?:an?)\s+(.+?)[?.!]*$", re.IGNORECASE),
]

# Patterns that indicate a skill/technology or general question instead of a position
SKILL_INDICATORS = {
    "know", "knows", "experience", "experienced", "skills", "skill", "worked with",
    "using", "used", "study", "studied", "graduated", "education", "project", "projects",
    "background", "compare", "summary", "summarize", "tell me about", "details about",
    "information about", "highest gpa", "gpa", "degree", "university", "college",
    "school", "certification", "certifications", "courses", "license",
}

ROLE_NOUNS = {
    "developer", "dev", "engineer", "designer", "manager", "architect",
    "lead", "specialist", "analyst", "administrator", "admin", "consultant",
    "officer", "coordinator", "supervisor", "intern", "trainee", "scientist",
    "programmer", "tester", "qa", "director", "recruiter", "technician",
    "assistant", "auditor", "instructor", "lecturer", "professor", "expert",
    "master",
}

TECH_TOOLS = {
    "python", "docker", "flutter", "dart", "react", "sql", "git", "rag",
    "langchain", "aws", "azure", "linux", "c++", "java", "node.js", "nodejs",
    "fastapi", "flask", "django", "mongodb", "postgresql", "figma",
}

NON_ROLE_WORDS = {
    "candidate", "candidates", "person", "people", "someone", "anyone",
    "best", "top", "qualified", "eligible", "he", "she", "they", "him", "her",
    "cv", "cvs", "resume", "resumes", "profile", "profiles",
}


def _clean_role(raw: str) -> str:
    """Normalize extracted role string."""
    r = raw.strip().lower()
    r = re.sub(r"[?!.,;]+$", "", r)
    r = re.sub(r"^(?:an?|the)\s+", "", r)
    r = re.sub(r"\s+(?:position|role|job)$", "", r)
    return r.strip()


def _extract_candidate_name_from_filename(filename: str) -> str:
    base = filename.rsplit(".", 1)[0]
    base = re.sub(r"[\(\)]", " ", base)
    base = re.sub(r"([a-z])([A-Z])", r"\1 \2", base)
    base = re.sub(r"[-_\d]+", " ", base)
    base = re.sub(
        r"\b(?:CV|Resume|FlowCV|Flow|Graphic|Designer|Flutter|Developer|Backend|Frontend|AI|Engineer)\b",
        " ", base, flags=re.IGNORECASE)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def extract_candidate_names(hits: list[dict]) -> set[str]:
    """Collect candidate names and name tokens from retrieved chunks and filenames."""
    names: set[str] = set()
    for h in hits:
        cand_name = h.get("candidate_name")
        if cand_name:
            cn_lower = cand_name.strip().lower()
            names.add(cn_lower)
            for token in re.findall(r"\b[a-zA-Z]{3,}\b", cn_lower):
                names.add(token)

        cv_name = h.get("cv_name")
        if cv_name:
            fn_clean = _extract_candidate_name_from_filename(cv_name).lower()
            if fn_clean:
                names.add(fn_clean)
                for token in re.findall(r"\b[a-zA-Z]{3,}\b", fn_clean):
                    names.add(token)

    return names


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
        re.compile(r"IT\s+Supervisor", re.IGNORECASE),
        re.compile(r"Network\s+Admin(?:istrator)?", re.IGNORECASE),
        re.compile(r"(?:UI\s*/\s*UX|Product)\s+Designer", re.IGNORECASE),
        re.compile(r"(?:Data|Machine Learning)\s+Scientist", re.IGNORECASE),
    ]

    for h in hits:
        cv_name = h.get("cv_name", "Unknown")
        content = h.get("content", "")
        if cv_name not in candidate_titles:
            candidate_titles[cv_name] = set()

        for t in h.get("job_titles") or []:
            if t:
                candidate_titles[cv_name].add(t.strip())

        for pat in title_patterns:
            for match in pat.finditer(content):
                matched_title = match.group(0).strip()
                if matched_title:
                    candidate_titles[cv_name].add(matched_title)

    return candidate_titles


def is_recognized_role(role_candidate: str, hits: list[dict]) -> bool:
    """Determine whether a candidate string represents an actual recognizable job title/role."""
    cleaned = _clean_role(role_candidate)
    if not cleaned:
        return False

    if cleaned in TECH_TOOLS:
        return False

    # Check taxonomy keys and values
    if cleaned in ROLE_TAXONOMY:
        return True
    for tax_key, tax_vals in ROLE_TAXONOMY.items():
        if cleaned == tax_key or cleaned in tax_vals:
            return True

    # Check if any role noun is present in the target words
    words = set(re.findall(r"\b[a-z]+\b", cleaned))
    if words & ROLE_NOUNS:
        return True

    # Check against extracted titles from CV hits
    cand_titles = extract_candidate_titles(hits)
    for titles in cand_titles.values():
        for t in titles:
            t_lower = t.lower()
            if cleaned == t_lower or cleaned in t_lower:
                return True

    return False


def validate_position_query(question: str, hits: list[dict]) -> PositionValidationResult:
    """Validate whether the user's query is asking for a position and whether any candidate holds it."""
    clean_q = question.strip().lower()
    clean_q = re.sub(r"[?!.,;]+$", "", clean_q).strip()

    # 1. Skill/Action indicators: questions asking about skills, education, projects, etc.
    for ind in SKILL_INDICATORS:
        if f" {ind} " in f" {clean_q} " or clean_q.startswith(f"{ind} "):
            return PositionValidationResult(is_position_query=False)

    # 2. Extract candidate names to prevent treating candidate name questions (e.g. "who is akram") as position queries
    candidate_names = extract_candidate_names(hits)

    target_role = None
    is_explicit = False

    # Check explicit patterns first
    for pattern in EXPLICIT_POSITION_PATTERNS:
        m = pattern.match(clean_q)
        if m:
            extracted = _clean_role(m.group(1))
            if extracted and len(extracted.split()) <= 6:
                target_role = extracted
                is_explicit = True
                break

    # If not explicit, check implicit patterns
    if not target_role:
        for pattern in IMPLICIT_POSITION_PATTERNS:
            m = pattern.match(clean_q)
            if m:
                extracted = _clean_role(m.group(1))
                if extracted and len(extracted.split()) <= 6:
                    target_role = extracted
                    break

    if not target_role:
        return PositionValidationResult(is_position_query=False)

    # 3. Reject if target matches a known candidate name (e.g. "akram", "abdelrahman", "sherif")
    target_words = set(re.findall(r"\b[a-z]{3,}\b", target_role))
    if target_role in candidate_names or (target_words and target_words.issubset(candidate_names)):
        return PositionValidationResult(is_position_query=False)

    # 4. Filter out pure tech tools and non-role generic query terms
    if target_role in TECH_TOOLS:
        return PositionValidationResult(is_position_query=False)

    if target_words and target_words.issubset(NON_ROLE_WORDS):
        return PositionValidationResult(is_position_query=False)

    # 5. For implicit queries (or even explicit without role nouns), verify target looks like a valid role
    if not is_explicit and not is_recognized_role(target_role, hits):
        return PositionValidationResult(is_position_query=False)

    # 6. Extract candidate titles from retrieved chunks and map against target_role
    candidates_map = extract_candidate_titles(hits)

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

            # 2. Taxonomy mapping
            if target_role in ROLE_TAXONOMY:
                acceptable_subtitles = ROLE_TAXONOMY[target_role]
                if any(sub in t_lower for sub in acceptable_subtitles):
                    matching_candidates.append({"cv_name": cv_name, "title": title})
                    continue

            # 3. Substring/hierarchy matching
            target_words_set = set(target_role.split())
            title_words = set(t_lower.replace("-", " ").replace("–", " ").split())
            if target_words_set.issubset(title_words):
                matching_candidates.append({"cv_name": cv_name, "title": title})

    if matching_candidates:
        return PositionValidationResult(
            is_position_query=True,
            target_role=target_role,
            has_matching_candidates=True,
            matching_candidates=matching_candidates,
        )

    # Deterministic rejection for genuine position queries with 0 matches
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

