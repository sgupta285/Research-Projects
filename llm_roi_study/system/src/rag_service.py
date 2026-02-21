import time, yaml
import numpy as np
from pathlib import Path
from openai import OpenAI

_base = Path(__file__).parent.parent
_cfg = yaml.safe_load(open(_base / "config" / "rag_config.yaml"))
_llm = yaml.safe_load(open(_base / "config" / "llm_config.yaml"))
_price = yaml.safe_load(open(_base / "config" / "pricing.yaml"))
client = OpenAI()

def _embed(text):
    t0 = time.perf_counter()
    r = client.embeddings.create(model=_cfg["embedding_model"], input=text)
    return r.data[0].embedding, round((time.perf_counter() - t0) * 1000, 2)

def _retrieve(q_emb, index, chunks, top_k):
    import faiss
    q = np.array([q_emb], dtype=np.float32)
    dists, idxs = index.search(q, top_k)
    return [{"rank": r+1, "chunk": chunks[i], "score": float(d)}
            for r, (d, i) in enumerate(zip(dists[0], idxs[0])) if i >= 0]

def rag_chat(query, messages, index, chunks):
    top_k = _cfg["retrieval_top_k"]; model = _llm["model"]
    q_emb, embed_ms = _embed(query)
    embed_cost = (len(query.split()) / 1000) * _price["models"][_cfg["embedding_model"]]["input_per_1k_tokens"]
    t_r0 = time.perf_counter()
    retrieved = _retrieve(q_emb, index, chunks, top_k)
    retrieval_ms = round((time.perf_counter() - t_r0) * 1000 + embed_ms, 2)
    ctx_str = "\n\n".join(f"[Source {r['rank']}]:\n{r['chunk']['text']}" for r in retrieved)
    aug_user = f"RETRIEVED CONTEXT:\n{ctx_str}\n\nTASK:\n{query}"
    full_msgs = [{"role": "system", "content": _cfg["rag_system_prompt"]},
                 *messages[:-1], {"role": "user", "content": aug_user}]
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=model, messages=full_msgs,
        temperature=_llm["temperature"], max_tokens=_llm["max_tokens"])
    gen_ms = round((time.perf_counter() - t0) * 1000, 2)
    u = resp.usage; p = _price["models"][model]
    llm_cost = (u.prompt_tokens / 1000) * p["input_per_1k_tokens"] + (u.completion_tokens / 1000) * p["output_per_1k_tokens"]
    return {"content": resp.choices[0].message.content,
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "cost_usd_llm": llm_cost, "cost_usd_embedding": embed_cost,
            "cost_usd_retrieval": _price["retrieval"]["faiss_query_cost_usd"],
            "retrieval_latency_ms": retrieval_ms, "generation_full_latency_ms": gen_ms,
            "_ragas_input": {"question": query, "answer": resp.choices[0].message.content,
                             "contexts": [r["chunk"]["text"] for r in retrieved]}}
