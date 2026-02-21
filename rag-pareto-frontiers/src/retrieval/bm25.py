from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from rank_bm25 import BM25Okapi

@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list

def build_bm25(chunks) -> BM25Index:
    tok = [c.text.lower().split() for c in chunks]
    return BM25Index(bm25=BM25Okapi(tok), chunks=chunks)

def query_bm25(index: BM25Index, query: str, top_k: int) -> List[Tuple[float, object]]:
    scores = index.bm25.get_scores(query.lower().split())
    pairs = list(zip(scores, index.chunks))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:top_k]
