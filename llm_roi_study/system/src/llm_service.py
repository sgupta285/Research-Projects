import time, yaml
from pathlib import Path
from openai import OpenAI

_base = Path(__file__).parent.parent
config = yaml.safe_load(open(_base / "config" / "llm_config.yaml"))
pricing = yaml.safe_load(open(_base / "config" / "pricing.yaml"))
client = OpenAI()

def _cost(model, pt, ct):
    p = pricing["models"][model]
    return (pt / 1000) * p["input_per_1k_tokens"] + (ct / 1000) * p["output_per_1k_tokens"]

def chat(messages: list, model: str = None, temperature: float = None) -> dict:
    model = model or config["model"]
    temperature = config["temperature"] if temperature is None else temperature
    full_msgs = [{"role": "system", "content": config["system_prompt"]}] + messages
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=model, messages=full_msgs,
        temperature=temperature, max_tokens=config["max_tokens"])
    latency = round((time.perf_counter() - t0) * 1000, 2)
    u = resp.usage
    return {"content": resp.choices[0].message.content,
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "cost_usd": _cost(model, u.prompt_tokens, u.completion_tokens),
            "generation_full_latency_ms": latency, "model": model}
