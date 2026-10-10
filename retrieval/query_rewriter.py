"""Conversational Query Rewriter for Multi-Turn CV Search.

Resolves coreferences (pronouns like "he", "she", "his", "their"), follow-up questions,
and elliptical queries into standalone, search-optimized queries before retrieval.
"""
import logging
import re
from typing import Optional

import config
from cache_manager import (
    cache_get_json,
    cache_set_json,
    hash_key,
    is_redis_available,
    normalize_query_key,
)
from clients import get_openai_client

logger = logging.getLogger(__name__)

# System instructions for conversational reformulation
REWRITER_SYSTEM_PROMPT = """You are a conversational query reformulation engine for an HR Resume/CV search system.
Your job is to reformulate the user's latest question into a single, standalone search query that can be understood in complete isolation without the conversation history.

Rules:
1. Coreference Resolution: Replace pronouns (e.g., "he", "him", "his", "she", "her", "they", "their", "this candidate", "both of them") with the actual candidate name(s) or job titles established in the chat history.
2. Follow-Up Expansion: If the user asks an elliptical or shorthand question (e.g. "What about Flutter?", "Where did he study?", "And certifications?"), expand it into a complete search query (e.g., "Does Akram have Flutter experience?").
3. Preserve Intent: Do NOT attempt to answer the question. Do NOT hallucinate candidate names or skills that were never mentioned in the history or question.
4. Search Optimization: Strip away conversational filler (e.g. "Can you tell me", "I want to know", "Also,"). Keep technical terms, skill keywords, and candidate names intact.
5. If Already Standalone: If the latest question is already completely self-contained and does not rely on prior context, return the original question verbatim.
6. Format: Output ONLY the rewritten search query. Do NOT add quotes, introductory phrases, or explanations."""

# Pronouns and conversational follow-up triggers
COREFERENCE_PATTERNS = [
    r"\b(?:he|him|his|she|her|hers|they|them|their|theirs)\b",
    r"\b(?:this|that|these|those)\s+(?:candidate|person|applicant|one|profile|cv|resume)\b",
    r"\b(?:what\s+about|how\s+about|and\s+what\s+about|tell\s+me\s+more)\b",
    r"\b(?:both|either|neither|the\s+same|another|other|compare\s+(?:him|her|them|with))\b",
]
COREFERENCE_REGEX = re.compile("|".join(COREFERENCE_PATTERNS), re.IGNORECASE)

MAX_HISTORY_MESSAGES = 6


def _has_context_dependencies(query: str, history: list[dict]) -> bool:
    """Heuristic check: does the query likely depend on prior conversation context?"""
    if not history or len(history) == 0:
        return False

    clean_q = query.strip()
    words = clean_q.split()

    # Very short elliptical follow-ups (e.g., "Flutter?", "His skills?", "What about salary?")
    if len(words) <= 3 and any(ch in clean_q for ch in ["?", "about", "and", "his", "her"]):
        return True

    # Contains explicit coreference pronouns or follow-up markers
    if COREFERENCE_REGEX.search(clean_q):
        return True

    # Starts with conversational connectors
    if re.match(r"^(?:and|also|what\s+about|how\s+about|does\s+he|does\s+she|is\s+he|is\s+she)\b", clean_q, re.IGNORECASE):
        return True

    return False


def _format_history(history: list[dict]) -> str:
    """Format recent user and assistant dialogue turns into a concise transcript."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    formatted_lines = []
    for msg in recent:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        # Truncate very long assistant replies to keep prompt focused
        if role == "assistant" and len(content) > 300:
            content = content[:300].rsplit(" ", 1)[0] + "..."
        if role in ("user", "assistant"):
            label = "User" if role == "user" else "Assistant"
            formatted_lines.append(f"{label}: {content}")
    return "\n".join(formatted_lines)


def rewrite_query(
    query: str,
    history: Optional[list[dict]] = None,
    client=None,
) -> tuple[str, bool]:
    """Reformulate a conversational question into a standalone search query.

    Args:
        query: The user's latest raw question.
        history: The conversation message history list.
        client: Optional Azure OpenAI client override (useful for testing/mocking).

    Returns:
        tuple[str, bool]: (effective_search_query, was_rewritten)
    """
    clean_q = query.strip()
    if not clean_q:
        return "", False

    # 1. Feature Toggle Check
    if not getattr(config, "QUERY_REWRITE_ENABLED", True):
        return clean_q, False

    # 2. Fast-Path: If no history or query is clearly standalone, skip LLM (0ms, 0 cost)
    if not history or not _has_context_dependencies(clean_q, history):
        return clean_q, False

    formatted_history = _format_history(history)
    if not formatted_history:
        return clean_q, False

    # 3. Redis Cache Check (0ms for repeated follow-up questions)
    history_hash = hash_key(formatted_history)
    cache_key = f"cv:rewrite:{normalize_query_key(clean_q)}:{history_hash}"
    if is_redis_available():
        cached = cache_get_json(cache_key)
        if cached and isinstance(cached, dict) and "rewritten" in cached:
            rewritten = cached["rewritten"]
            was_rewritten = cached.get("was_rewritten", True)
            logger.info("Semantic Query Rewriter Cache Hit: '%s' -> '%s'", clean_q, rewritten)
            return rewritten, was_rewritten

    # 4. Azure OpenAI Query Reformulation
    try:
        user_message = (
            f"=== CONVERSATION HISTORY ===\n"
            f"{formatted_history}\n\n"
            f"=== LATEST USER QUESTION ===\n"
            f"{clean_q}\n\n"
            f"Standalone Search Query:"
        )

        ai_client = client or get_openai_client()
        response = ai_client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=100,
        )

        raw_output = response.choices[0].message.content or ""
        rewritten = raw_output.strip()

        # Clean any accidental conversational prefixes or quotation marks
        rewritten = re.sub(r'^(?:Standalone\s+Search\s+Query|Rewritten\s+Query|Query):\s*', '', rewritten, flags=re.IGNORECASE)
        rewritten = rewritten.strip('"`\' \n')

        # Fallback if model returned empty
        if not rewritten:
            return clean_q, False

        was_rewritten = rewritten.lower() != clean_q.lower()

        # Cache result in Redis (1 hour TTL)
        if is_redis_available():
            cache_set_json(
                cache_key,
                {"rewritten": rewritten, "was_rewritten": was_rewritten},
                ttl=getattr(config, "REDIS_CACHE_TTL", 3600),
            )

        if was_rewritten:
            logger.info("Conversational Query Rewritten: '%s' -> '%s'", clean_q, rewritten)

        return rewritten, was_rewritten

    except Exception as exc:
        logger.warning("Query rewrite failed, falling back to original query: %s", exc)
        return clean_q, False
