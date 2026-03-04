from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Tuple, Any


_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the question based solely on the "
    "passages below. Be concise (1-2 sentences).\n\n"
    "Question: {question}\n\n"
    "Passages:\n{passages}\n\n"
    "Answer:"
)

_MAX_PASSAGE_CHARS = 600   # per passage, to stay within context


def _build_prompt(question: str, passages: List[str]) -> str:
    ptext = "\n\n".join(
        f"[{i+1}] {p[:_MAX_PASSAGE_CHARS]}" for i, p in enumerate(passages)
    )
    return _PROMPT_TEMPLATE.format(question=question, passages=ptext)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Backend: mock (deterministic, zero cost)
# ---------------------------------------------------------------------------

class _MockGenerator:
    def generate(self, question: str, passages: List[str]) -> GenerationResult:
        t0 = time.perf_counter()
        answer = passages[0][:80].strip() if passages else ""
        return GenerationResult(
            text=answer,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
        )


# ---------------------------------------------------------------------------
# Backend: Anthropic (claude-haiku-4-5-20251001 by default)
# ---------------------------------------------------------------------------

class _AnthropicGenerator:
    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 120,
        input_price_per_1k: float = 0.001,
        output_price_per_1k: float = 0.005,
    ):
        import anthropic as _anthropic
        self._client = _anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._input_price = input_price_per_1k / 1000.0
        self._output_price = output_price_per_1k / 1000.0

    def generate(self, question: str, passages: List[str]) -> GenerationResult:
        prompt = _build_prompt(question, passages)
        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = resp.content[0].text.strip() if resp.content else ""
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        cost = in_tok * self._input_price + out_tok * self._output_price
        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            cost_usd=cost,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )


# ---------------------------------------------------------------------------
# Backend: Ollama (local, zero cost)
# ---------------------------------------------------------------------------

class _OllamaGenerator:
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def generate(self, question: str, passages: List[str]) -> GenerationResult:
        import urllib.request, json
        prompt = _build_prompt(question, passages)
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 80, "temperature": 0.0},
        }).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read())
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return GenerationResult(
            text=body.get("response", "").strip(),
            latency_ms=latency_ms,
            cost_usd=0.0,
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_generator(cfg_gen: dict) -> Any:
    backend = cfg_gen.get("backend", "mock")
    if backend == "mock":
        return _MockGenerator()
    if backend == "anthropic":
        return _AnthropicGenerator(
            model=cfg_gen.get("model", "claude-haiku-4-5-20251001"),
            max_tokens=int(cfg_gen.get("max_tokens", 120)),
            input_price_per_1k=float(cfg_gen.get("input_price_per_1k", 0.001)),
            output_price_per_1k=float(cfg_gen.get("output_price_per_1k", 0.005)),
        )
    if backend == "ollama":
        return _OllamaGenerator(
            model=cfg_gen.get("model", "llama3.2:3b"),
            base_url=cfg_gen.get("base_url", "http://localhost:11434"),
        )
    raise ValueError(f"Unknown generation backend: {backend!r}")
