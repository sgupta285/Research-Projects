from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    source_path: Optional[str] = None

@dataclass
class QA:
    qid: str
    question: str
    gold_answer: str = ""
    gold_sources: Optional[List[Dict[str, Any]]] = None

def load_hotpotqa(root_dir: str, split: str, max_examples: int = 500):
    import json
    fname = {
        "train":"hotpot_train_v1.1.json",
        "dev_distractor":"hotpot_dev_distractor_v1.json",
        "dev_fullwiki":"hotpot_dev_fullwiki_v1.json",
    }[split]
    fp = os.path.join(root_dir, fname)
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs, qa = [], []
    for i, ex in enumerate(data[:max_examples]):
        qid = ex.get("_id", str(i))
        question = ex["question"]
        answer = ex.get("answer","")
        gold_titles = sorted(set(t for t,_ in ex.get("supporting_facts", [])))
        gold_sources = [{"title": t} for t in gold_titles] if gold_titles else None

        for j, (title, sents) in enumerate(ex.get("context", [])):
            docs.append(Doc(doc_id=f"{qid}::p{j}", title=title, text=title+"\n"+" ".join(sents)))
        qa.append(QA(qid=qid, question=question, gold_answer=answer, gold_sources=gold_sources))
    return docs, qa

def load_legalbenchrag(root_dir: str, benchmark_file: str, max_examples: int = 800):
    import json
    bench_path = os.path.join(root_dir, benchmark_file)
    with open(bench_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    cases = bench.get("test_cases", bench if isinstance(bench, list) else [])
    cases = cases[:max_examples]

    needed = set()
    for c in cases:
        for s in c.get("ground_truth", c.get("ground_truth_snippets", [])) or []:
            p = s.get("path") or s.get("file_path") or s.get("corpus_path")
            if p: needed.add(p)

    docs = []
    for p in sorted(needed):
        abs_p = os.path.join(root_dir, "corpus", p)
        if not os.path.exists(abs_p):
            abs_p = os.path.join(root_dir, p)
        with open(abs_p, "r", encoding="utf-8") as f:
            txt = f.read()
        docs.append(Doc(doc_id=p, title=os.path.basename(p), text=txt, source_path=p))

    qa = []
    for i, c in enumerate(cases):
        qid = c.get("id") or c.get("_id") or str(i)
        question = c.get("query") or c.get("question")
        gts = c.get("ground_truth", c.get("ground_truth_snippets", [])) or []
        gold_sources = []
        for s in gts:
            p = s.get("path") or s.get("file_path") or s.get("corpus_path")
            start = s.get("start") or s.get("start_idx") or s.get("char_start")
            end = s.get("end") or s.get("end_idx") or s.get("char_end")
            if p is not None and start is not None and end is not None:
                gold_sources.append({"source_path": p, "start": int(start), "end": int(end)})
        qa.append(QA(qid=qid, question=question, gold_sources=gold_sources or None))
    return docs, qa
