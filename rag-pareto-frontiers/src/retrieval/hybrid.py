from __future__ import annotations
from .bm25 import query_bm25
from .dense import query_dense

def query_hybrid(bm25_index, dense_index, query: str, top_k: int, alpha: float = 0.5):
    bm = query_bm25(bm25_index, query, top_k=top_k*2)
    de = query_dense(dense_index, query, top_k=top_k*2)

    def norm(pairs):
        if not pairs: return {}
        scores=[p[0] for p in pairs]
        lo,hi=min(scores),max(scores)
        out={}
        for s,c in pairs:
            ns = 0.0 if hi==lo else (s-lo)/(hi-lo)
            out[c.chunk_id]=(ns,c)
        return out

    nb, nd = norm(bm), norm(de)
    ids=set(nb)|set(nd)
    merged=[]
    for cid in ids:
        sb = nb.get(cid,(0.0,None))[0]
        sd = nd.get(cid,(0.0,None))[0]
        c = (nb.get(cid,(None,None))[1] or nd.get(cid,(None,None))[1])
        merged.append((alpha*sd+(1-alpha)*sb, c))
    merged.sort(key=lambda x:x[0], reverse=True)
    return merged[:top_k]
