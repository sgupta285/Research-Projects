from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_path: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

def build_chunks(docs, mode: str, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks = []
    if mode == "chars":
        step = max(1, chunk_size - overlap)
        for d in docs:
            txt = d.text
            n = len(txt)
            j = 0
            for start in range(0, n, step):
                end = min(n, start + chunk_size)
                chunks.append(Chunk(
                    chunk_id=f"{d.doc_id}::c{j}", doc_id=d.doc_id, title=d.title,
                    text=txt[start:end], source_path=getattr(d, "source_path", None),
                    char_start=start, char_end=end
                ))
                j += 1
                if end >= n: break
        return chunks

    step = max(1, chunk_size - overlap)
    for d in docs:
        words = d.text.split()
        for j in range(0, len(words), step):
            part = words[j:j+chunk_size]
            if not part: break
            chunks.append(Chunk(
                chunk_id=f"{d.doc_id}::w{j}", doc_id=d.doc_id, title=d.title,
                text=" ".join(part), source_path=getattr(d, "source_path", None)
            ))
            if j + chunk_size >= len(words): break
    return chunks
