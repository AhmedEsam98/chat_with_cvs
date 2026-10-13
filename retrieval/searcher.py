import json
import re
from azure.search.documents.models import VectorizedQuery

import config
from cache_manager import (
    cache_get,
    cache_set,
    get_query_vector,
    hash_key,
    normalize_query_key,
    semantic_cache_get,
    semantic_cache_set,
    store_query_vector,
)
from typing import Optional

from clients import get_search_client
from ingestion.embeddings import embed
from retrieval.query_rewriter import rewrite_query
from retrieval.reranker import rerank


def retrieve(query: str, k: int, history: Optional[list[dict]] = None) -> list[dict]:
    """Two-Stage Retrieval: Azure Hybrid Search + FlashRank Cross-Encoder Re-ranking.

    Optimized Caching Pipeline:
    0. Query Rewriting: If history is provided, reformulates conversational query into standalone search.
    1. Fast Exact Hits Cache: Sub-millisecond lookup (<1ms) for repeated queries.
    2. Query Vector Check: If query vector is cached in Redis, reuse it immediately (0.002s vs 0.35s Azure OpenAI API).
    3. Semantic Cache Check: If semantically equivalent search is in Redis, return top-k chunks instantly.
    4. Stage 1 (Coarse): If cache miss, retrieve candidate pool via Azure AI Search (BM25 + HNSW Cosine vector).
    5. Stage 2 (Fine): Cross-encoder re-ranks candidate pool to return top k most relevant chunks.
    6. Cache Results: Stores final re-ranked hits and query vector in exact and semantic cache.
    """
    if history:
        query = rewrite_query(query, history=history)

    clean_q = normalize_query_key(query)

    # 1. Fast Exact Hits Cache Check (<1ms)
    exact_hits_key = f"cv:hits:exact:{hash_key(clean_q, str(k))}"
    cached_exact = cache_get(exact_hits_key)
    if cached_exact is not None:
        if isinstance(cached_exact, str):
            try:
                return json.loads(cached_exact)[:k]
            except Exception:
                pass
        elif isinstance(cached_exact, list):
            return cached_exact[:k]

    # 2. Check if query vector is already in Redis cache from a previous run
    qvec = get_query_vector(clean_q)

    # 3. If vector is cached, check Semantic Retrieval Cache immediately (bypasses Azure OpenAI embed API)
    if qvec is not None and getattr(config, "SEMANTIC_CACHE_ENABLED", True):
        threshold = getattr(config, "SEMANTIC_CACHE_THRESHOLD", 0.92)
        cached_hits, sim, matched_q = semantic_cache_get(
            namespace="retrieve",
            query_vector=qvec,
            threshold=threshold,
            filter_params={"min_k": k},
        )
        if cached_hits is not None:
            return cached_hits[:k]

    # 3. Cache Miss / un-embedded query: Generate embedding vector via Azure OpenAI
    if qvec is None:
        qvec = embed([query])[0]
        store_query_vector(clean_q, qvec, ttl=getattr(
            config, "REDIS_CACHE_TTL", 3600))

        # Check semantic cache now that vector is generated (for semantically similar questions)
        if getattr(config, "SEMANTIC_CACHE_ENABLED", True):
            threshold = getattr(config, "SEMANTIC_CACHE_THRESHOLD", 0.92)
            cached_hits, sim, matched_q = semantic_cache_get(
                namespace="retrieve",
                query_vector=qvec,
                threshold=threshold,
                filter_params={"min_k": k},
            )
            if cached_hits is not None:
                return cached_hits[:k]

    # 2. Stage 1 (Coarse): Retrieve candidate pool from Azure AI Search
    top_candidates = max(k * 2, getattr(config, "RERANK_TOP_N", 20))
    select_fields = [
        "id", "cv_name", "candidate_name", "section_name",
        "chunk_index", "content", "skills", "job_titles", "file_type"
    ]
    results = get_search_client().search(
        search_text=query,
        vector_queries=[VectorizedQuery(
            vector=qvec, k_nearest_neighbors=top_candidates, fields="content_vector")],
        select=select_fields,
        top=top_candidates,
    )
    candidate_hits = [{
        "id": r.get("id"),
        "cv_name": r["cv_name"],
        "candidate_name": r.get("candidate_name") or r["cv_name"],
        "section_name": r.get("section_name", "General"),
        "chunk_index": r.get("chunk_index", 0),
        "content": r["content"],
        "skills": r.get("skills") or [],
        "job_titles": r.get("job_titles") or [],
        "file_type": r.get("file_type", "pdf"),
    } for r in results]

    # 3. Stage 2 (Fine): Cross-Encoder Re-ranking
    if getattr(config, "RERANKER_ENABLED", True) and len(candidate_hits) > k:
        hits = rerank(query, candidate_hits, top_k=k)
    else:
        hits = candidate_hits[:k]

    # 4. Store in Exact & Semantic Cache
    ttl = getattr(config, "REDIS_CACHE_TTL", 3600)
    cache_set(exact_hits_key, json.dumps(hits), ttl=ttl)

    if getattr(config, "SEMANTIC_CACHE_ENABLED", True):
        semantic_cache_set(
            namespace="retrieve",
            query_text=query,
            query_vector=qvec,
            value=hits,
            filter_params={"k": k},
        )

    return hits
