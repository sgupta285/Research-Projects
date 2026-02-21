from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

@dataclass
class DenseIndex:
    model_name: str
    model: SentenceTransformer
    index: faiss.IndexFlatIP
    chunks: list

def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def build_dense(chunks, model: SentenceTransformer, model_name: str) -> DenseIndex:
    vecs = model.encode([c.text for c in chunks], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    vecs = _norm(vecs)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return DenseIndex(model_name=model_name, model=model, index=idx, chunks=chunks)

def query_dense(d: DenseIndex, query: str, top_k: int) -> List[Tuple[float, object]]:
    qv = d.model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    qv = _norm(qv)
    scores, ids = d.index.search(qv, top_k)
    out=[]
    for s,i in zip(scores[0].tolist(), ids[0].tolist()):
        if i >= 0: out.append((float(s), d.chunks[i]))
    return out
