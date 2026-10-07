import logging
from typing import Optional

from flashrank import Ranker, RerankRequest

import config

logger = logging.getLogger(__name__)

_ranker_instance: Optional[Ranker] = None


def get_ranker() -> Ranker:
    global _ranker_instance
    if _ranker_instance is None:
        model_name = getattr(config, "RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2")
        logger.info(f"Initializing FlashRank cross-encoder with model '{model_name}'...")
        _ranker_instance = Ranker(model_name=model_name)
    return _ranker_instance


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Re-rank candidate chunks using a local cross-encoder scoring model.

    Args:
        query: User question.
        hits: Candidate chunks from Azure AI Search (each having 'cv_name' and 'content').
        top_k: Number of highest-relevance chunks to return.

    Returns:
        List of top-k re-ranked chunks with an added 'rerank_score' field.
    """
    if not hits:
        return []

    if not getattr(config, "RERANKER_ENABLED", True) or len(hits) <= 1:
        return hits[:top_k]

    try:
        ranker = get_ranker()
        passages = [
            {
                "id": idx,
                "text": h["content"],
                "meta": h,
            }
            for idx, h in enumerate(hits)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerank_req)

        output = []
        for r in reranked_results[:top_k]:
            item = dict(r["meta"])
            item["rerank_score"] = float(r.get("score", 0.0))
            output.append(item)

        return output
    except Exception as exc:
        logger.error(f"FlashRank re-ranking failed ({exc}). Falling back to original order.")
        return hits[:top_k]
