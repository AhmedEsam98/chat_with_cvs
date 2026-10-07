import re
from typing import Optional

import config

# Standard resume/CV section heading patterns
SECTION_PATTERNS = [
    r"(?:Professional\s+|Executive\s+|Career\s+)?Summary",
    r"(?:Career\s+)?Objective",
    r"Profile(?:\s+Summary)?",
    r"About\s+Me",
    r"(?:Work\s+|Professional\s+|Employment\s+|Career\s+)?Experience",
    r"Work\s+History",
    r"(?:Academic\s+|Educational\s+)?(?:Background|Qualifications|Education)",
    r"(?:Technical\s+|Core\s+|Key\s+)?Skills(?:\s*&\s*(?:Tools|Technologies|Proficiencies))?",
    r"(?:Areas\s+of\s+)?Expertise",
    r"(?:Key\s+|Selected\s+|Personal\s+|Recent\s+)?Projects",
    r"Certifications?(?:\s*(?:&|and)\s*Training)?",
    r"Licenses?(?:\s*(?:&|and)\s*Certifications?)?",
    r"Courses?(?:\s*(?:&|and)\s*Workshops?)?",
    r"(?:Honors?\s*(?:&|and)\s*)?Awards?",
    r"Achievements?",
    r"(?:Spoken\s+)?Languages?",
    r"(?:Extracurricular\s+)?Activities",
    r"Volunteer(?:ing)?(?:\s+Experience)?",
    r"Publications?",
    r"References?",
]

HEADING_REGEX = re.compile(
    r"^\s*(?:[-*#•\d.]+\s*)?(" + "|".join(SECTION_PATTERNS) + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_text(text: str) -> str:
    """Normalize whitespace and line endings without destroying layout."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3 or more newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Recursively split text on natural boundaries (paragraphs -> lines -> bullets -> sentences -> words)."""
    if len(text) <= max_chars:
        return [text]

    separators = ["\n\n", "\n", " • ", "\n•", " - ", "\n-", ". ", " "]
    chosen_sep = None
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    if not chosen_sep:
        # Fallback to word slicing without cutting words in half
        words = text.split(" ")
        chunks = []
        curr, curr_len = [], 0
        for w in words:
            if curr_len + len(w) + 1 > max_chars and curr:
                chunks.append(" ".join(curr))
                # Keep some overlap words
                overlap_words = []
                o_len = 0
                for ow in reversed(curr):
                    if o_len + len(ow) + 1 <= overlap_chars:
                        overlap_words.insert(0, ow)
                        o_len += len(ow) + 1
                    else:
                        break
                curr = overlap_words + [w]
                curr_len = sum(len(x) + 1 for x in curr)
            else:
                curr.append(w)
                curr_len += len(w) + 1
        if curr:
            chunks.append(" ".join(curr))
        return chunks

    parts = text.split(chosen_sep)
    chunks = []
    curr_chunk: list[str] = []
    curr_len = 0

    for part in parts:
        part_len = len(part) + len(chosen_sep)
        if curr_len + part_len > max_chars and curr_chunk:
            chunk_str = chosen_sep.join(curr_chunk)
            chunks.append(chunk_str)

            # Build overlap from the end of current chunk
            overlap_parts = []
            o_len = 0
            for op in reversed(curr_chunk):
                if o_len + len(op) + len(chosen_sep) <= overlap_chars:
                    overlap_parts.insert(0, op)
                    o_len += len(op) + len(chosen_sep)
                else:
                    break
            curr_chunk = overlap_parts + [part]
            curr_len = sum(len(x) for x in curr_chunk) + len(chosen_sep) * max(0, len(curr_chunk) - 1)
        else:
            curr_chunk.append(part)
            curr_len += part_len

    if curr_chunk:
        chunks.append(chosen_sep.join(curr_chunk))

    return [c.strip() for c in chunks if c.strip()]


def extract_sections(text: str) -> list[tuple[str, str]]:
    """Parse CV text into logical sections based on recognized headings.

    Returns:
        List of (heading_name, section_body) tuples.
    """
    cleaned = _clean_text(text)
    matches = list(HEADING_REGEX.finditer(cleaned))

    if not matches:
        return [("General", cleaned)]

    sections = []
    prev_end = 0
    prev_heading = "Contact & Overview"

    for m in matches:
        start = m.start()
        if start > prev_end:
            body = cleaned[prev_end:start].strip()
            if body:
                sections.append((prev_heading, body))
        prev_heading = m.group(1).strip().title()
        prev_end = m.end()

    if prev_end < len(cleaned):
        body = cleaned[prev_end:].strip()
        if body:
            sections.append((prev_heading, body))

    return sections


def chunk_text(text: str) -> list[str]:
    """Section-Aware Heading Chunking for CVs.

    1. Identifies logical CV sections (Summary, Experience, Education, Skills, etc.).
    2. Keeps each section intact as a coherent semantic unit.
    3. If a section exceeds CHUNK_SIZE, sub-chunks along natural boundaries
       (paragraphs, bullet points, lines) without cutting words or sentences.
    4. Annotates each chunk with its Section title for precise retrieval grounding.
    """
    max_size = getattr(config, "CHUNK_SIZE", 1000)
    overlap = getattr(config, "CHUNK_OVERLAP", 150)

    sections = extract_sections(text)
    chunks = []

    for heading, body in sections:
        if len(body) <= max_size:
            # Section fits completely within one chunk
            chunks.append(f"[Section: {heading}]\n{body}")
        else:
            # Section exceeds max size -> split recursively on paragraphs/bullets
            sub_parts = _split_long_text(body, max_size, overlap)
            total_parts = len(sub_parts)
            for idx, part in enumerate(sub_parts, start=1):
                part_header = f"[Section: {heading} (Part {idx}/{total_parts})]" if total_parts > 1 else f"[Section: {heading}]"
                chunks.append(f"{part_header}\n{part}")

    return chunks