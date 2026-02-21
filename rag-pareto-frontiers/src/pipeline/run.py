from __future__ import annotations
import time, hashlib
from typing import Dict, Any, Tuple
from src.data.io import load_hotpotqa, load_legalbenchrag
from src.data.chunking import build_chunks
from src.retrieval.bm25 import build_bm25, query_bm25
from src.retrieval.dense import load_model, build_dense, query_dense
from src.retrieval.hybrid import query_hybrid
from src.rerank.simple import rerank_simple
from src.metrics.metrics import token_f1, retrieval_title_recall_precision, retrieval_span_recall_precision
from src.ops.cache import SimpleCache
from src.utils.config import sha1_text

_MODEL_CACHE: Dict[str, Tuple[object, float]] = {}
_INDEX_CACHE: Dict[str, Tuple[object, float]] = {}

def load_dataset(cfg_dataset: dict):
    n=cfg_dataset.get("name")
    if n=="hotpotqa":
        return load_hotpotqa(cfg_dataset["root_dir"], cfg_dataset["split"], int(cfg_dataset.get("max_examples",500)))
    if n=="legalbenchrag":
        return load_legalbenchrag(cfg_dataset["root_dir"], cfg_dataset["benchmark_file"], int(cfg_dataset.get("max_examples",800)))
    raise ValueError(n)

def _corpus_fingerprint(docs) -> str:
    sample = "\n".join((d.title + "|" + str(len(d.text))) for d in docs[:200])
    return sha1_text(sample)

def build_artifacts(cfg: dict) -> Tuple[Dict[str, Any], Dict[str, float]]:
    t0=time.perf_counter()
    corpus,_=load_dataset(cfg["dataset"])
    corpus_fp=_corpus_fingerprint(corpus)

    ch=cfg["pipeline"]["chunking"]
    chunks=build_chunks(corpus, mode=ch.get("mode","words"), chunk_size=int(ch["chunk_size"]), overlap=int(ch["overlap"]))

    r=cfg["pipeline"]["retrieval"]
    bm=dense=None
    model_load_ms=0.0
    index_build_ms=0.0

    if r["type"] in ("bm25","hybrid"):
        tb=time.perf_counter()
        bm=build_bm25(chunks)
        index_build_ms += (time.perf_counter()-tb)*1000.0

    if r["type"] in ("dense","hybrid"):
        model_name = cfg.get("dense_model_name","sentence-transformers/all-MiniLM-L6-v2")
        model_key = sha1_text(model_name)
        model_cache_hit = False
        model_load_ms_observed = 0.0
        if model_key in _MODEL_CACHE:
            model, ml_canonical = _MODEL_CACHE[model_key]
            model_cache_hit = True
        else:
            tm=time.perf_counter()
            model=load_model(model_name)
            ml_canonical=(time.perf_counter()-tm)*1000.0
            _MODEL_CACHE[model_key]=(model, ml_canonical)
            model_load_ms_observed = float(ml_canonical)
        model_load_ms += float(ml_canonical)


        index_key = sha1_text("|".join([corpus_fp, str(ch.get("mode")), str(ch["chunk_size"]), str(ch["overlap"]), model_name]))
        index_cache_hit = False
        index_build_ms_observed = 0.0
        if index_key in _INDEX_CACHE:
            dense, ib_canonical = _INDEX_CACHE[index_key]
            index_cache_hit = True
        else:
            ti=time.perf_counter()
            dense=build_dense(chunks, model=model, model_name=model_name)
            ib_canonical=(time.perf_counter()-ti)*1000.0
            _INDEX_CACHE[index_key]=(dense, ib_canonical)
            index_build_ms_observed = float(ib_canonical)
        index_build_ms += float(ib_canonical)


    timings={
        "model_load_ms": float(model_load_ms),
        "index_build_ms": float(index_build_ms),
        "model_load_ms_observed": float(locals().get("model_load_ms_observed", 0.0)),
        "index_build_ms_observed": float(locals().get("index_build_ms_observed", 0.0)),
        "artifact_total_ms": float((time.perf_counter()-t0)*1000.0),
        "model_cache_hit": bool(locals().get("model_cache_hit", False)),
        "index_cache_hit": bool(locals().get("index_cache_hit", False)),
    }
    return {"chunks":chunks,"bm25":bm,"dense":dense}, timings

def _retrieve(cfg, art, q):
    r=cfg["pipeline"]["retrieval"]; k=int(r["top_k"])
    if r["type"]=="bm25": return query_bm25(art["bm25"], q, k)
    if r["type"]=="dense": return query_dense(art["dense"], q, k)
    if r["type"]=="hybrid": return query_hybrid(art["bm25"], art["dense"], q, k, alpha=0.5)
    raise ValueError(r["type"])

def _answer(cands): return "" if not cands else cands[0][1].text[:220]

def _h(*parts):
    h=hashlib.sha256()
    for p in parts: h.update(str(p).encode("utf-8"))
    return h.hexdigest()

def run_eval(cfg: dict, art):
    _,qa=load_dataset(cfg["dataset"])
    retrieval_cache = SimpleCache() if cfg["pipeline"].get("caching",{}).get("retrieval_cache",False) else None

    rows=[]
    retrieval_cold_ms=None
    retrieval_warm=[]
    e2e_cold_ms=None
    e2e_warm=[]

    for i,ex in enumerate(qa):
        t0=time.perf_counter()

        tr0=time.perf_counter()
        if retrieval_cache is not None:
            key=_h(ex.question, cfg["pipeline"]["retrieval"], cfg["pipeline"]["chunking"], cfg["pipeline"]["rerank"])
            hit,val=retrieval_cache.get(key)
            cands=val if hit else _retrieve(cfg, art, ex.question)
            if not hit: retrieval_cache.set(key, cands)
        else:
            cands=_retrieve(cfg, art, ex.question)
        tr1=time.perf_counter()

        retrieval_ms=(tr1-tr0)*1000.0

        if bool(cfg["pipeline"]["rerank"].get("enabled",False)):
            cands=rerank_simple(ex.question, cands, top_k=int(cfg["pipeline"]["retrieval"]["top_k"]))

        pred=_answer(cands)
        qf1=token_f1(pred, ex.gold_answer) if ex.gold_answer else 0.0

        rrec=rprec=0.0
        if ex.gold_sources:
            if "source_path" in (ex.gold_sources[0] or {}):
                rrec,rprec=retrieval_span_recall_precision(cands, ex.gold_sources)
            else:
                titles=[g.get("title","") for g in ex.gold_sources if g.get("title")]
                rrec,rprec=retrieval_title_recall_precision(cands, titles)

        t1=time.perf_counter()
        e2e_ms=(t1-t0)*1000.0

        if i==0:
            retrieval_cold_ms=retrieval_ms
            e2e_cold_ms=e2e_ms
        else:
            retrieval_warm.append(retrieval_ms)
            e2e_warm.append(e2e_ms)

        cost=0.0
        if bool(cfg["pipeline"]["rerank"].get("enabled",False)):
            toks=max(80, len(ex.question.split())+240)
            cost += (toks/1000.0)*float(cfg["pricing"]["rerank"]["per_1k_tokens_usd"])

        rows.append({
            "qid":ex.qid,
            "quality_f1":float(qf1),
            "retrieval_recall_k":float(rrec),
            "retrieval_precision_k":float(rprec),
            "retrieval_ms":float(retrieval_ms),
            "latency_ms":float(e2e_ms),
            "cost_usd":float(cost),
            "retrieval_type":cfg["pipeline"]["retrieval"]["type"],
            "top_k":int(cfg["pipeline"]["retrieval"]["top_k"]),
            "chunk_mode":cfg["pipeline"]["chunking"].get("mode","words"),
            "chunk_size":int(cfg["pipeline"]["chunking"]["chunk_size"]),
            "overlap":int(cfg["pipeline"]["chunking"]["overlap"]),
            "rerank_enabled":bool(cfg["pipeline"]["rerank"].get("enabled",False)),
            "retrieval_cache_hit_rate": retrieval_cache.stats.hit_rate if retrieval_cache is not None else 0.0,
        })

    meta={
        "retrieval_cold_ms": float(retrieval_cold_ms or 0.0),
        "retrieval_warm_mean_ms": float(sum(retrieval_warm)/max(1,len(retrieval_warm))),
        "end_to_end_cold_ms": float(e2e_cold_ms or 0.0),
        "end_to_end_warm_mean_ms": float(sum(e2e_warm)/max(1,len(e2e_warm))),
    }
    return rows, meta
