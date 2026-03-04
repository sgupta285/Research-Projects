"""Cross-encoder reranker using ms-marco-MiniLM-L-6-v2."""
from __future__ import annotations
import time
from typing import Any, Optional

_MODEL: Optional[Any] = None
_MODEL_LOAD_MS: float = 0.0
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_model():
    global _MODEL, _MODEL_LOAD_MS
    if _MODEL is None:
        from sentence_transformers import CrossEncoder
        t0 = time.perf_counter()
        _MODEL = CrossEncoder(_MODEL_NAME, max_length=512)
        _MODEL_LOAD_MS = (time.perf_counter() - t0) * 1000.0
    return _MODEL


def rerank_cross(query: str, candidates, top_k: int):
    """Score candidates with a cross-encoder and return top_k by score.

    Args:
        query: The query string.
        candidates: List of (score, Chunk) pairs from the retriever.
        top_k: Number of top candidates to return after reranking.

    Returns:
        List of (cross_encoder_score, Chunk) pairs sorted descending.
    """
    model = _load_model()
    if not candidates:
        return []
    pairs = [(query, c.text[:400]) for _, c in candidates]
    scores = model.predict(pairs, batch_size=len(pairs))
    rescored = [(float(s), c) for s, (_, c) in zip(scores, candidates)]
    rescored.sort(key=lambda x: x[0], reverse=True)
    return rescored[:top_k]
