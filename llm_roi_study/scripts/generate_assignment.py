"""
Generate and commit the Latin-square task-condition assignment matrix.
Run ONCE before data collection. Commit output to OSF alongside seeds.yaml.
Usage: python3 scripts/generate_assignment.py [--dry-run] [--n 200]
"""
import json, argparse
import numpy as np
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
seeds = yaml.safe_load(open(ROOT / "system" / "config" / "seeds.yaml"))
RNG = np.random.default_rng(seeds["task_assignment_seed"])

CATEGORIES = {"A": ["A01","A02","A03","A04","A05","A06"],
              "B": ["B01","B02","B03","B04","B05","B06"],
              "C": ["C01","C02","C03","C04","C05","C06"]}
LATIN = [["control","T1","T2"],["T1","T2","control"],["T2","control","T1"]]


def generate(n):
    out = {}
    for pid in range(1, n + 1):
        ls = LATIN[(pid - 1) % 3]; tasks = []
        for cat, pool in CATEGORIES.items():
            shuf = RNG.permutation(pool).tolist()
            for i, cond in enumerate(ls):
                tasks.append({"task_id": shuf[i], "category": cat,
                               "condition": cond, "condition_order": i + 1})
        RNG.shuffle(tasks)
        for j, t in enumerate(tasks): t["session_order"] = j + 1
        out[str(pid)] = {"participant_index": pid, "latin_row": (pid-1) % 3, "tasks": tasks}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    assignments = generate(args.n)
    if args.dry_run:
        print("DRY RUN — not writing to disk")
        print(json.dumps(assignments["1"], indent=2))
        return

    out_path = ROOT / "system" / "config" / "task_assignment.json"
    out_path.write_text(json.dumps(assignments, indent=2))
    print(f"Assignment written to {out_path}")

    import pandas as pd
    df = pd.DataFrame([t for a in assignments.values() for t in a["tasks"]])
    print("\nCondition distribution per category:")
    print(df.groupby(["category", "condition"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
