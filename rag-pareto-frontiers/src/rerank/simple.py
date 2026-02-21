from __future__ import annotations
from rapidfuzz import fuzz

def rerank_simple(query: str, candidates, top_k: int):
    rescored=[]
    for _,c in candidates:
        s = fuzz.token_set_ratio(query, c.text)/100.0
        rescored.append((float(s), c))
    rescored.sort(key=lambda x:x[0], reverse=True)
    return rescored[:top_k]
