import hashlib, json, uuid
from datetime import datetime, timezone
from pathlib import Path

SALT = "REPLACE_WITH_SECRET"

def hash_participant(pid: str) -> str:
    return hashlib.sha256(f"{pid}{SALT}".encode()).hexdigest()[:16]

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]

class SessionLogger:
    def __init__(self, log_path: str = "data/raw/sessions.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        record["event_id"] = str(uuid.uuid4())
        record["collection_date"] = datetime.now(timezone.utc).date().isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
