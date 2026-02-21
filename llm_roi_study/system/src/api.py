from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from .logger import SessionLogger, hash_participant, hash_text
from .llm_service import chat
from .rag_service import rag_chat

app = FastAPI(title="LLM-ROI Study API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
logger = SessionLogger()
sessions: dict = {}

class StartReq(BaseModel):
    prolific_pid: str; task_id: str; condition: str

class InteractReq(BaseModel):
    session_token: str; message: str

class SubmitReq(BaseModel):
    session_token: str; final_response: str; nasa_tlx: dict; pause_log: list

@app.post("/start_task")
def start_task(req: StartReq):
    pid = hash_participant(req.prolific_pid)
    tok = hash_text(f"{pid}{req.task_id}{datetime.now().isoformat()}")
    sessions[tok] = {"participant_id": pid, "task_id": req.task_id, "condition": req.condition,
        "session_start_ts": datetime.now(timezone.utc).isoformat(), "task_start_ts": None,
        "messages": [], "tokens": {"prompt": 0, "completion": 0},
        "cost": {"llm": 0.0, "embedding": 0.0, "retrieval": 0.0},
        "latency": {"retrieval_ms": 0.0, "generation_ms": 0.0}, "num_interactions": 0}
    return {"session_token": tok}

@app.post("/task_started")
def task_started(session_token: str):
    if session_token not in sessions: raise HTTPException(404)
    sessions[session_token]["task_start_ts"] = datetime.now(timezone.utc).isoformat()
    return {"status": "ok"}

@app.post("/interact")
def interact(req: InteractReq):
    s = sessions.get(req.session_token)
    if not s: raise HTTPException(404)
    s["messages"].append({"role": "user", "content": req.message})
    s["num_interactions"] += 1
    if s["condition"] == "control":
        return {"content": "AI assistant not available in this condition."}
    result = chat(s["messages"]) if s["condition"] == "T1" else rag_chat(req.message, s["messages"], None, [])
    if s["condition"] == "T1":
        s["cost"]["llm"] += result["cost_usd"]
    else:
        s["cost"]["llm"] += result["cost_usd_llm"]
        s["cost"]["embedding"] += result["cost_usd_embedding"]
        s["cost"]["retrieval"] += result["cost_usd_retrieval"]
        s["latency"]["retrieval_ms"] += result.get("retrieval_latency_ms", 0)
    s["tokens"]["prompt"] += result.get("prompt_tokens", 0)
    s["tokens"]["completion"] += result.get("completion_tokens", 0)
    s["latency"]["generation_ms"] += result.get("generation_full_latency_ms", 0)
    s["messages"].append({"role": "assistant", "content": result["content"]})
    return {"content": result["content"]}

@app.post("/submit_final")
def submit_final(req: SubmitReq):
    s = sessions.get(req.session_token)
    if not s: raise HTTPException(404)
    end_ts = datetime.now(timezone.utc).isoformat()
    from datetime import datetime as dt
    wall_s = (dt.fromisoformat(end_ts) - dt.fromisoformat(s["task_start_ts"])).total_seconds()
    pause_s = sum(p["end"] - p["start"] for p in req.pause_log if (p["end"] - p["start"]) > 60)
    record = {"participant_id": s["participant_id"], "task_id": s["task_id"],
        "task_category": s["task_id"][0], "condition": s["condition"],
        "task_start_ts": s["task_start_ts"], "task_end_ts": end_ts,
        "time_to_complete_s": round(max(0, wall_s - pause_s), 3),
        "status": "completed", "num_interactions": s["num_interactions"],
        "cost_usd_total": round(sum(s["cost"].values()), 6),
        "retrieval_enabled": s["condition"] == "T2",
        "response_hash": hash_text(req.final_response),
        "rework_count": max(0, s["num_interactions"] - 1),
        **{f"nasa_tlx_{k}": v for k, v in req.nasa_tlx.items()},
        "nasa_tlx_composite": round(sum(req.nasa_tlx.values()) / 6, 2),
        "quality_score_final": None, "hallucination_rate_human": None}
    logger.write(record)
    del sessions[req.session_token]
    return {"status": "ok"}
