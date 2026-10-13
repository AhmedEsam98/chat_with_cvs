from typing import Optional
import logging

import config
from cache_manager import (
    cache_clear,
    cache_get,
    cache_set,
    get_query_vector,
    hash_key,
    normalize_query_key,
    semantic_cache_get,
    semantic_cache_set,
    store_query_vector,
)
from clients import get_openai_client
from generation.prompts import SYSTEM_PROMPT, build_user_message
from generation.role_matcher import validate_position_query
from ingestion.embeddings import embed

logger = logging.getLogger(__name__)

HISTORY_TURNS = 6


def _hit_hash(hits: list[dict]) -> str:
    """Deterministic hash of retrieved chunks content and sources."""
    raw = "|".join(sorted(h["cv_name"] + ":" + h["content"] for h in hits))
    return hash_key(raw)


def clear_answer_cache():
    """Clear all answer caches."""
    cache_clear("cv:ans:")
    cache_clear("cv:sem:answer:")


def answer_stream(question: str, history: list[dict], hits: list[dict], search_query: Optional[str] = None):
    """Stream an answer from Azure OpenAI, with Fast Exact Caching, Semantic Caching, and Position Validation.

    1. Deterministic Position Gate: If the question asks for a specific job position and no
       candidate holds it, immediately returns an honest refusal without calling the LLM.
    2. Exact Answer Cache Check: Sub-millisecond lookup in Redis/Memory for the identical question and retrieved chunks.
    3. Semantic Cache Check: Instant return if a semantically equivalent question with identical chunks was answered before.
    4. Cache Miss: Stream tokens from Azure OpenAI, yield each token, and store the full answer in both Exact and Semantic Cache.
    """
    clean_q = normalize_query_key(question)

    # 1. Deterministic Position Gate
    pos_check = validate_position_query(question, hits)
    if not pos_check.is_position_query and search_query and search_query != question:
        pos_check_rw = validate_position_query(search_query, hits)
        if pos_check_rw.is_position_query and pos_check_rw.has_matching_candidates:
            pos_check = pos_check_rw

    if pos_check.is_position_query and not pos_check.has_matching_candidates:
        if pos_check.rejection_message:
            yield pos_check.rejection_message
            return

    h_hash = _hit_hash(hits)

    # 2. Fast Exact Answer Cache Check (<1ms)
    exact_cache_key = f"cv:ans:exact:{hash_key(clean_q, h_hash)}"
    cached_exact = cache_get(exact_cache_key)
    if cached_exact is not None:
        logger.info(f"🎯 Exact Answer Cache HIT for '{question}'")
        yield cached_exact
        return

    # 3. Semantic Cache Check
    qvec = None
    if getattr(config, "SEMANTIC_CACHE_ENABLED", True):
        threshold = getattr(config, "SEMANTIC_CACHE_THRESHOLD", 0.92)
        qvec = get_query_vector(clean_q)
        if qvec is None and search_query:
            qvec = get_query_vector(normalize_query_key(search_query))
        if qvec is None:
            try:
                qvec = embed([clean_q])[0]
                store_query_vector(clean_q, qvec, ttl=getattr(config, "REDIS_CACHE_TTL", 3600))
            except Exception as e:
                logger.debug(f"Could not embed clean_q for answer cache: {e}")

        if qvec is not None:
            cached_answer, sim, matched_q = semantic_cache_get(
                namespace="answer",
                query_vector=qvec,
                threshold=threshold,
                filter_params={"hit_hash": h_hash},
            )
            if cached_answer is not None:
                logger.info(f"🎯 Semantic Answer Cache HIT! Sim: {sim:.4f} (Matched: '{matched_q}')")
                yield cached_answer
                return

    # 4. Cache Miss: Stream from Azure OpenAI
    user_prompt = build_user_message(question, hits)
    if pos_check.is_position_query and pos_check.has_matching_candidates and pos_check.matching_candidates:
        cand_notes = ", ".join(
            f"{c['cv_name']} ({c['title']})" for c in pos_check.matching_candidates)
        user_prompt += f"\n\n[Context: The role '{pos_check.target_role}' encompasses the following verified positions held by candidates: {cand_notes}]"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-HISTORY_TURNS:]
    messages.append({"role": "user", "content": user_prompt})

    stream = get_openai_client().chat.completions.create(
        model=config.CHAT_MODEL, messages=messages,
        temperature=0.0, stream=True)

    full_response = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response.append(token)
            yield token

    # Store completed response in Exact & Semantic Cache
    full_text = "".join(full_response)
    if full_text:
        ttl = getattr(config, "REDIS_CACHE_TTL", 3600)
        cache_set(exact_cache_key, full_text, ttl=ttl)

        if getattr(config, "SEMANTIC_CACHE_ENABLED", True):
            if qvec is None:
                qvec = get_query_vector(clean_q)
            if qvec is None and search_query:
                qvec = get_query_vector(normalize_query_key(search_query))
            if qvec is not None:
                semantic_cache_set(
                    namespace="answer",
                    query_text=question,
                    query_vector=qvec,
                    value=full_text,
                    filter_params={"hit_hash": h_hash},
                )
