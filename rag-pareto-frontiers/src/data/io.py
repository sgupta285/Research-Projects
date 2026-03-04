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
    """Load LegalBench-RAG benchmark (ZeroEntropy format).

    Handles the actual corpus format from https://github.com/zeroentropy-ai/legalbenchrag:
      {"tests": [{"query": ..., "snippets": [{"file_path": ..., "span": [s, e], ...}]}]}
    Also handles legacy formats with test_cases/ground_truth keys.
    """
    import json
    bench_path = os.path.join(root_dir, benchmark_file)
    with open(bench_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    # Normalise to a list of cases
    if isinstance(bench, list):
        cases = bench
    elif "tests" in bench:          # actual ZeroEntropy format
        cases = bench["tests"]
    elif "test_cases" in bench:
        cases = bench["test_cases"]
    else:
        cases = list(bench.values())[0] if bench else []
    cases = cases[:max_examples]

    # Collect all corpus file paths needed
    needed: set = set()
    for c in cases:
        snippets = c.get("snippets") or c.get("ground_truth") or c.get("ground_truth_snippets") or []
        for s in snippets:
            p = s.get("file_path") or s.get("path") or s.get("corpus_path")
            if p:
                needed.add(p)

    # Load corpus documents (text files, one per legal document)
    docs: List[Doc] = []
    for p in sorted(needed):
        for candidate in [
            os.path.join(root_dir, "corpus", p),
            os.path.join(root_dir, p),
        ]:
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read()
                docs.append(Doc(doc_id=p, title=os.path.basename(p), text=txt, source_path=p))
                break

    # Build QA pairs
    qa: List[QA] = []
    for i, c in enumerate(cases):
        qid = str(c.get("id") or c.get("_id") or i)
        question = c.get("query") or c.get("question") or ""
        snippets = c.get("snippets") or c.get("ground_truth") or c.get("ground_truth_snippets") or []
        gold_sources: List[Dict[str, Any]] = []
        for s in snippets:
            p = s.get("file_path") or s.get("path") or s.get("corpus_path")
            span = s.get("span")      # [start, end] in ZeroEntropy format
            start = span[0] if span else (s.get("start") or s.get("start_idx") or s.get("char_start"))
            end   = span[1] if span else (s.get("end")   or s.get("end_idx")   or s.get("char_end"))
            if p is not None and start is not None and end is not None:
                gold_sources.append({"source_path": p, "start": int(start), "end": int(end)})
        qa.append(QA(qid=qid, question=question, gold_sources=gold_sources or None))
    return docs, qa
