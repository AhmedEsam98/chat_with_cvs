import logging
import re
from typing import Any, Optional

import config
from cache_manager import (
    get_cached_rewrite,
    hash_key,
    normalize_query_key,
    set_cached_rewrite,
)
from clients import get_openai_client

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """You are an expert search query reformulation engine for an HR and CV/Resume Retrieval-Augmented Generation (RAG) system.

Your task is to transform the user's latest question into a single, standalone, optimized search query for hybrid search (BM25 keyword + semantic vector search) over candidate CVs.

Guidelines:
1. Coreference & Context Resolution:
   - If the conversation history references specific candidates, roles, projects, or education, replace all pronouns (he, she, they, him, her, it, his, their) and indirect references ("the candidate", "the previous person", "the Flutter dev") with their explicit names or positions.
2. Self-Contained:
   - The rewritten query must make complete sense on its own without requiring any prior context.
3. Domain Keywords & Expansion:
   - When searching for technical roles, skills, or experience, include closely related technical keywords or synonyms that commonly appear on resumes (e.g., "mobile developer" -> "mobile developer Flutter Dart Android iOS", "backend" -> "backend developer Python Node.js APIs").
4. Maintain Strict Intent:
   - Do NOT answer the question.
   - Do NOT assume candidate qualifications or fabricate details not present in the chat or question.
   - Keep it concise (1-2 sentences maximum).
5. Output format:
   - Return ONLY the rewritten search query text. No explanations, no markdown quotes, no conversational filler.
"""


def _build_history_summary(history: list[dict], max_turns: int = 4) -> str:
    """Format recent conversation turns into a compact summary for the rewriter."""
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for m in recent:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = m.get("content", "").strip()
        # Truncate lengthy assistant replies to keep prompt small and fast
        if role == "Assistant" and len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _is_context_dependent(query: str) -> bool:
    """Check if query contains pronouns or conversational follow-up references."""
    q_lower = query.lower()
    words = set(re.findall(r"\b[a-z']+\b", q_lower))
    pronouns = {"he", "she", "they", "him", "her", "his", "their", "them", "himself", "herself"}
    if words & pronouns:
        return True
    followup_phrases = (
        "the candidate", "this candidate", "that candidate", "the previous",
        "previous candidate", "tell me more", "what about", "what else",
        "more details", "and him", "and her", "both", "either",
    )
    return any(p in q_lower for p in followup_phrases)


def rewrite_query(
    query: str,
    history: Optional[list[dict]] = None,
    client: Optional[Any] = None,
    force_expansion: bool = False,
) -> str:
    """Rewrite and optimize a user query for CV hybrid retrieval.

    1. Fast paths: returns original query if rewriting is disabled or query is empty.
    2. Conversational resolution: resolves pronouns/references using conversation history.
    3. Query expansion: enriches domain/technical terms for hybrid search.
    4. Caching: caches rewritten queries in Redis/memory to ensure sub-millisecond repeated queries.
    """
    if not query or not query.strip():
        return ""

    raw_query = query.strip()

    # Check if rewriter is enabled globally
    if not getattr(config, "QUERY_REWRITER_ENABLED", True):
        return raw_query

    has_history = bool(history and len(history) > 0)
    expansion_enabled = getattr(config, "QUERY_EXPANSION_ENABLED", True) or force_expansion

    # If no history and expansion is disabled, fast-path return without LLM call
    if not has_history and not expansion_enabled:
        return raw_query

    history_summary = _build_history_summary(history or [])
    norm_q = normalize_query_key(raw_query)
    is_dependent = _is_context_dependent(raw_query)

    # ── Check Standalone Cache First for Non-Context Queries ──
    standalone_key = f"cv:rewrite:standalone:{norm_q}"
    if not is_dependent:
        cached_standalone = get_cached_rewrite(standalone_key)
        if cached_standalone:
            return cached_standalone

    # ── Check Conversational Cache ──
    cache_key = f"cv:rewrite:{hash_key(norm_q, history_summary)}"
    cached_val = get_cached_rewrite(cache_key)
    if cached_val:
        return cached_val

    # ── LLM Invocation ──
    prompt_lines = []
    if history_summary and is_dependent:
        prompt_lines.append(f"Conversation History:\n{history_summary}\n")
    prompt_lines.append(f"Current Question to Reformulate: {raw_query}")
    prompt_lines.append("\nRewritten Standalone Search Query:")

    user_prompt = "\n".join(prompt_lines)

    try:
        openai_client = client or get_openai_client()
        response = openai_client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        rewritten = response.choices[0].message.content.strip()
        # Clean quotes if wrapped
        if (rewritten.startswith('"') and rewritten.endswith('"')) or (
            rewritten.startswith("'") and rewritten.endswith("'")
        ):
            rewritten = rewritten[1:-1].strip()

        # If LLM returned empty, fallback to raw_query
        if not rewritten:
            rewritten = raw_query

        # ── Store in Cache ──
        ttl = getattr(config, "REDIS_CACHE_TTL", 3600)
        set_cached_rewrite(cache_key, rewritten, ttl=ttl)
        if not is_dependent:
            set_cached_rewrite(standalone_key, rewritten, ttl=ttl)
        return rewritten

    except Exception as exc:
        logger.warning(f"Query rewriting failed ({exc}). Falling back to original query.")
        return raw_query
