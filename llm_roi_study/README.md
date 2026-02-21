# LLM-ROI Study
**Title:** Do LLM Assistants Actually Improve Productivity?
**Target Venue:** ACM CHI 2027

---

## Setup (macOS / Linux)

### Step 1 — Python environment
macOS ships `python3`, not `python`. Create a virtual environment first:
```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate.bat     # Windows
pip install -r requirements.txt
```

### Step 2 — Environment variables
```bash
cp .env.example .env
# Open .env and fill in OPENAI_API_KEY and PARTICIPANT_ID_SALT
```

### Step 3 — Run analysis scripts (no Docker needed)
```bash
python3 analysis/scripts/power_calculation.py --effect_size 0.35 --rho 0.4 --n_target 200
python3 scripts/generate_assignment.py --dry-run
python3 analysis/scripts/generate_synthetic_data.py --n 200 --output data/processed/sessions_synthetic.csv
python3 analysis/scripts/primary_analysis.py --data data/processed/sessions_synthetic.csv --output data/processed/
python3 analysis/scripts/roi_frontier.py --data data/processed/ate_results.csv --sessions data/processed/sessions_synthetic.csv --output figures/
```

### Step 4 — API server (optional, only needed for live data collection)
**With Docker:**
```bash
docker-compose up --build
```
**Without Docker:**
```bash
uvicorn system.src.api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Common Errors on macOS

| Error | Fix |
|---|---|
| `python: command not found` | Use `python3` instead of `python` |
| `cp: .env.example: No such file or directory` | You may have only extracted v2.zip. Extract the full zip — all files should be at root level. |
| `no configuration file provided: not found` | `docker-compose.yml` missing. Same fix — re-extract from this zip. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your `venv` first |
| `faiss-cpu` install fails on Apple Silicon | Run `pip install faiss-cpu --no-binary faiss-cpu` or use `conda install -c conda-forge faiss-cpu` |

---

## Project Structure
```
llm_roi_study/
├── paper/                   paper_draft.md (full CHI paper)
├── preregistration/         OSF pre-registration template (N=200)
├── protocol/                30-task bank, quality rubric, session protocol
├── system/
│   ├── src/                 api.py, llm_service.py, rag_service.py, logger.py
│   └── config/              llm_config.yaml, rag_config.yaml, pricing.yaml, seeds.yaml
├── analysis/scripts/        power_calculation.py, primary_analysis.py,
│                            roi_frontier.py, generate_synthetic_data.py
├── scripts/                 generate_assignment.py
├── checklists/              camera_ready_checklist.md (18 reviewer defenses)
├── data/processed/          sessions_synthetic.csv (1800 rows, N=200)
├── .env.example             ← copy to .env and fill in keys
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Design Summary
- **N = 200** (powered for d=0.35; CHI-competitive)
- **3 conditions:** Control | T1 (LLM-only) | T2 (RAG-augmented)
- **3 task categories:** Information Synthesis | Structured Writing | Coding
- **Primary theoretical contribution:** Welfare utility model trading quality gain vs. cognitive burden
- **Secondary:** RAG as distinct causal treatment (first experiment to do so)

License: MIT (code) | CC-BY 4.0 (paper)
