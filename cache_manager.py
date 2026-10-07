import json
import logging
import hashlib
from typing import Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

import redis

import config

logger = logging.getLogger(__name__)

_redis_client: Optional[redis.Redis] = None
_redis_initialized: bool = False
_memory_cache: dict[str, Any] = {}
_semantic_memory_cache: dict[str, dict[str, dict]] = {}


def get_redis_client() -> Optional[redis.Redis]:
    global _redis_client, _redis_initialized
    if _redis_initialized:
        return _redis_client

    _redis_initialized = True
    if not getattr(config, "REDIS_ENABLED", True):
        logger.info("Redis cache is explicitly disabled via configuration.")
        return None

    redis_url = getattr(config, "REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis.from_url(
            redis_url,
            decode_responses=True,       # Return strings instead of raw bytes (no need to .decode() manually)
            socket_connect_timeout=1.5,  # Max 1.5s to establish TCP connection, else give up (prevents app hanging)
            socket_timeout=1.5,          # Max 1.5s to wait for a response per command (GET/SET), else timeout
        )
        client.ping()
        _redis_client = client
        logger.info(f"✅ Successfully connected to Redis at {redis_url}")
    except Exception as exc:
        logger.warning(
            f"⚠️ Could not connect to Redis at {redis_url} ({exc}). "
            "Falling back to in-memory caching."
        )
        _redis_client = None

    return _redis_client


def is_redis_available() -> bool:
    client = get_redis_client()
    return client is not None


# ── Query Vector Transport (Internal) ──────────────────────────────────────────

def normalize_query_key(query_text: str) -> str:
    """Normalize query text for cache key consistency: lowercase, strip trailing punctuation and collapse whitespace."""
    import re
    cleaned = re.sub(r"[?!.,;:]+$", "", query_text.strip().lower()).strip()
    return " ".join(cleaned.split()) if cleaned else query_text.strip().lower()


def store_query_vector(query_text: str, vector: list[float], ttl: Optional[int] = None) -> None:
    """Hold a query vector so retrieval and generation steps can reuse it without re-embedding."""
    ttl = ttl or getattr(config, "REDIS_CACHE_TTL", 3600)
    key = f"cv:qvec:{hash_key(normalize_query_key(query_text))}"
    client = get_redis_client()
    if client is not None:
        try:
            client.set(key, json.dumps(vector), ex=ttl)
            return
        except Exception as exc:
            logger.error(f"Redis store_query_vector error: {exc}")
    _memory_cache[key] = vector


def get_query_vector(query_text: str) -> Optional[list[float]]:
    """Retrieve the cached query vector."""
    key = f"cv:qvec:{hash_key(normalize_query_key(query_text))}"
    client = get_redis_client()
    if client is not None:
        try:
            val = client.get(key)
            if val is not None:
                return json.loads(val)
        except Exception as exc:
            logger.error(f"Redis get_query_vector error: {exc}")
    return _memory_cache.get(key)


# ── Semantic Vector Cache ─────────────────────────────────────────────────────

def _compute_cosine_similarities(query_vec: list[float], matrix_vecs: list[list[float]]) -> list[float]:
    """Compute cosine similarities between query_vec and a list of vectors."""
    if not matrix_vecs:
        return []

    if HAS_NUMPY:
        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        mat = np.array(matrix_vecs, dtype=np.float32)
        mat_norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat_norms[mat_norms == 0] = 1.0
        mat = mat / mat_norms

        sims = np.dot(mat, q)
        return sims.tolist()

    # Pure Python fallback
    q_norm = sum(x * x for x in query_vec) ** 0.5
    results = []
    for v in matrix_vecs:
        v_norm = sum(x * x for x in v) ** 0.5
        if q_norm == 0 or v_norm == 0:
            results.append(0.0)
        else:
            dot = sum(a * b for a, b in zip(query_vec, v))
            results.append(dot / (q_norm * v_norm))
    return results


def _match_filter_params(item_params: dict, filter_params: dict) -> bool:
    """Match item params against filter criteria, supporting exact match and min_k range."""
    for k, v in filter_params.items():
        if k == "min_k":
            if item_params.get("k", 0) < v:
                return False
        elif item_params.get(k) != v:
            return False
    return True


def semantic_cache_get(
    namespace: str,
    query_vector: list[float],
    threshold: Optional[float] = None,
    filter_params: Optional[dict] = None,
) -> tuple[Optional[Any], float, Optional[str]]:
    """Look up semantically similar cached results based on vector cosine similarity.

    Args:
        namespace: Logical category, e.g. "retrieve" or "answer".
        query_vector: Embedding vector of the query (e.g. 1536 floats).
        threshold: Minimum cosine similarity required (e.g. 0.92).
        filter_params: Optional dict of exact attributes to match (e.g. {"k": 5}).

    Returns:
        (cached_value, similarity_score, matched_query_text) if hit, else (None, best_sim, None)
    """
    if threshold is None:
        threshold = getattr(config, "SEMANTIC_CACHE_THRESHOLD", 0.92)

    filter_params = filter_params or {}
    client = get_redis_client()

    matching_items: list[dict] = []

    if client is not None:
        try:
            keys = list(client.scan_iter(f"cv:sem:{namespace}:*", count=200))
            if keys:
                raw_items = client.mget(keys)
                for raw in raw_items:
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw)
                        # Check filter params (e.g. same k or same chunks)
                        item_params = item.get("params", {})
                        if _match_filter_params(item_params, filter_params):
                            matching_items.append(item)
                    except Exception:
                        continue
        except Exception as exc:
            logger.error(f"Redis semantic scan error [{namespace}]: {exc}")

    # Fallback / merge with in-memory semantic cache if Redis has no matches
    if not matching_items and namespace in _semantic_memory_cache:
        for item in _semantic_memory_cache[namespace].values():
            item_params = item.get("params", {})
            if _match_filter_params(item_params, filter_params):
                matching_items.append(item)

    if not matching_items:
        return None, 0.0, None

    # Compute similarity across all candidate vectors
    matrix_vecs = [item["vector"] for item in matching_items]
    sims = _compute_cosine_similarities(query_vector, matrix_vecs)

    best_idx = int(np.argmax(sims)) if HAS_NUMPY else max(range(len(sims)), key=lambda i: sims[i])
    best_sim = float(sims[best_idx])
    best_item = matching_items[best_idx]

    if best_sim >= threshold:
        logger.info(
            f"🎯 Semantic Cache HIT [{namespace}]! Similarity: {best_sim:.4f} >= {threshold:.2f} "
            f"(Matched: \"{best_item.get('query')}\")"
        )
        return best_item["value"], best_sim, best_item.get("query")

    logger.debug(f"Semantic Cache MISS [{namespace}]: Best similarity {best_sim:.4f} < {threshold:.2f}")
    return None, best_sim, None


def semantic_cache_set(
    namespace: str,
    query_text: str,
    query_vector: list[float],
    value: Any,
    ttl: Optional[int] = None,
    filter_params: Optional[dict] = None,
) -> None:
    """Store a query vector and its result in the semantic cache.

    Args:
        namespace: Logical category, e.g. "retrieve" or "answer".
        query_text: The user's original query text.
        query_vector: Embedding vector of the query.
        value: The result to cache (chunks list or answer text).
        ttl: Time to live in seconds.
        filter_params: Additional matching constraints (e.g. {"k": 5}).
    """
    ttl = ttl or getattr(config, "REDIS_CACHE_TTL", 3600)
    filter_params = filter_params or {}

    param_str = "_".join(f"{k}:{v}" for k, v in sorted(filter_params.items()))
    entry_id = hash_key(query_text.strip().lower(), param_str)
    key = f"cv:sem:{namespace}:{entry_id}"

    payload = {
        "query": query_text,
        "vector": query_vector,
        "params": filter_params,
        "value": value,
    }

    client = get_redis_client()
    if client is not None:
        try:
            client.set(key, json.dumps(payload), ex=ttl)
            return
        except Exception as exc:
            logger.error(f"Redis semantic set error [{namespace}]: {exc}")

    # Fallback to in-memory semantic cache
    _semantic_memory_cache.setdefault(namespace, {})[entry_id] = payload


# ── Invalidation & Utilities ──────────────────────────────────────────────────

def cache_clear(prefix: str = "cv:") -> None:
    """Clear all caches (both exact and semantic) matching the prefix."""
    global _memory_cache, _semantic_memory_cache
    client = get_redis_client()
    if client is not None:
        try:
            keys = set(client.scan_iter(f"{prefix}*"))
            clean_p = prefix.rstrip(":")
            if clean_p == "cv":
                keys.update(client.scan_iter("cv:sem:*"))
            elif clean_p.startswith("cv:"):
                sub_ns = clean_p[3:]
                keys.update(client.scan_iter(f"cv:sem:{sub_ns}*"))
            else:
                keys.update(client.scan_iter(f"cv:sem:{clean_p}*"))

            if keys:
                client.delete(*keys)
                logger.info(f"Cleared {len(keys)} Redis keys matching '{prefix}*'")
        except Exception as exc:
            logger.error(f"Redis clear error: {exc}")

    # Clear matching memory cache keys
    _memory_cache = {k: v for k, v in _memory_cache.items() if not k.startswith(prefix)}

    clean_p = prefix.rstrip(":")
    if clean_p == "cv":
        _semantic_memory_cache.clear()
    else:
        ns_target = clean_p[3:] if clean_p.startswith("cv:") else clean_p
        for ns in list(_semantic_memory_cache.keys()):
            if ns == ns_target or ns.startswith(ns_target):
                del _semantic_memory_cache[ns]


def hash_key(*parts: str) -> str:
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


