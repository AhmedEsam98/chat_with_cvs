import config
from cache_manager import (
    cache_clear,
    get_query_vector,
    hash_key,
    normalize_query_key,
    semantic_cache_get,
    semantic_cache_set,
)
from clients import get_openai_client
from generation.prompts import SYSTEM_PROMPT, build_user_message
from generation.role_matcher import validate_position_query

HISTORY_TURNS = 6


def _hit_hash(hits: list[dict]) -> str:
    """Deterministic hash of retrieved chunks content and sources."""
    raw = "|".join(sorted(h["cv_name"] + ":" + h["content"] for h in hits))
    return hash_key(raw)


def clear_answer_cache():
    """Clear all answer caches."""
    cache_clear("cv:sem:answer:")


def answer_stream(question: str, history: list[dict], hits: list[dict]):
    """Stream an answer from Azure OpenAI, with Semantic Caching and Position Validation.

    1. Deterministic Position Gate: If the question asks for a specific job position and no
       candidate holds it, immediately returns an honest refusal without calling the LLM.
    2. Checks Semantic Cache: If a question with similar meaning and identical retrieved CV chunks
       was answered before, returns the cached answer instantly without calling Azure OpenAI.
    3. Cache miss: Stream tokens from Azure OpenAI, yield each token, and store the full answer in Semantic Cache.
    """
    clean_q = normalize_query_key(question)

    # 1. Deterministic Position Gate
    pos_check = validate_position_query(question, hits)
    if pos_check.is_position_query and not pos_check.has_matching_candidates:
        if pos_check.rejection_message:
            yield pos_check.rejection_message
            return

    h_hash = _hit_hash(hits)

    # 2. Semantic Cache Check
    if getattr(config, "SEMANTIC_CACHE_ENABLED", True):
        threshold = getattr(config, "SEMANTIC_CACHE_THRESHOLD", 0.92)
        qvec = get_query_vector(clean_q)
        if qvec is not None:
            cached_answer, sim, matched_q = semantic_cache_get(
                namespace="answer",
                query_vector=qvec,
                threshold=threshold,
                filter_params={"hit_hash": h_hash},
            )
            if cached_answer is not None:
                yield cached_answer
                return

    # 3. Cache Miss: Stream from Azure OpenAI
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

    # Store completed response in Semantic Cache
    full_text = "".join(full_response)
    if full_text and getattr(config, "SEMANTIC_CACHE_ENABLED", True):
        qvec = get_query_vector(clean_q)
        if qvec is not None:
            semantic_cache_set(
                namespace="answer",
                query_text=question,
                query_vector=qvec,
                value=full_text,
                filter_params={"hit_hash": h_hash},
            )
