from __future__ import annotations
import re

def _norm(s: str) -> str:
    s=s.lower()
    s=re.sub(r"[^a-z0-9\s]"," ",s)
    s=re.sub(r"\s+"," ",s).strip()
    return s

def token_f1(pred: str, gold: str) -> float:
    p=_norm(pred).split()
    g=_norm(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    pc, gc = {}, {}
    for t in p: pc[t]=pc.get(t,0)+1
    for t in g: gc[t]=gc.get(t,0)+1
    common=sum(min(pc[t], gc.get(t,0)) for t in pc)
    if common==0: return 0.0
    prec=common/len(p); rec=common/len(g)
    return 2*prec*rec/(prec+rec)

def retrieval_title_recall_precision(retrieved, gold_titles):
    if not gold_titles: return (0.0,0.0)
    gt=set(_norm(t) for t in gold_titles)
    got=set(_norm(getattr(c,"title","")) for _,c in retrieved)
    tp=len(gt&got)
    return (tp/max(1,len(gt)), tp/max(1,len(got)))

def _overlap(a0,a1,b0,b1): return max(0, min(a1,b1)-max(a0,b0))

def retrieval_span_recall_precision(retrieved, gold_spans):
    if not gold_spans: return (0.0,0.0)
    hits=0
    for gs in gold_spans:
        ok=False
        for _,c in retrieved:
            if getattr(c,"source_path",None)!=gs["source_path"]: continue
            cs,ce=getattr(c,"char_start",None), getattr(c,"char_end",None)
            if cs is None or ce is None: continue
            if _overlap(cs,ce,int(gs["start"]),int(gs["end"]))>0:
                ok=True; break
        if ok: hits+=1
    recall=hits/max(1,len(gold_spans))
    good=0
    for _,c in retrieved:
        cs,ce=getattr(c,"char_start",None), getattr(c,"char_end",None)
        if cs is None or ce is None: continue
        cpath=getattr(c,"source_path",None)
        for gs in gold_spans:
            if gs["source_path"]!=cpath: continue
            if _overlap(cs,ce,int(gs["start"]),int(gs["end"]))>0:
                good+=1; break
    prec=good/max(1,len(retrieved))
    return (recall, prec)
