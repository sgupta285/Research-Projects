from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import pandas as pd


def _round3(v: float) -> float:
    return round(float(v), 3)


def _keyed(df: pd.DataFrame, cols: List[str]) -> dict:
    out = {}
    for _, r in df.iterrows():
        key = tuple(str(r[c]) for c in cols)
        out[key] = r
    return out


def _check_sharpe_consistency(
    metrics: pd.DataFrame, fig_vals: pd.DataFrame, name: str
) -> List[str]:
    errors: List[str] = []
    m = _keyed(metrics[["strategy", "exec_model", "sharpe"]], ["strategy", "exec_model"])
    f = _keyed(fig_vals[["strategy", "exec_model", "sharpe"]], ["strategy", "exec_model"])

    for k, mr in m.items():
        if k not in f:
            errors.append(f"{name}: missing key {k}")
            continue
        mv = _round3(mr["sharpe"])
        fv = _round3(f[k]["sharpe"])
        if mv != fv:
            errors.append(f"{name}: {k} metrics={mv} fig={fv}")

    for k in f.keys():
        if k not in m:
            errors.append(f"{name}: unexpected key {k}")
    return errors


def _expected_inflation(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strat in ["tsmom_60", "csmom_60"]:
        sub = metrics[metrics["strategy"] == strat].copy()
        if sub.empty:
            continue
        naive = sub[sub["exec_model"] == "naive"]
        if naive.empty:
            continue
        s0 = float(naive.iloc[0]["sharpe"])
        for _, r in sub.iterrows():
            model = str(r["exec_model"])
            if model == "naive":
                continue
            sm = float(r["sharpe"])
            plotted = bool(s0 > 0 and sm > 0)
            ratio = float(s0 / sm) if plotted else float("nan")
            rows.append(
                {
                    "strategy": strat,
                    "exec_model": model,
                    "ratio": ratio,
                    "plotted": plotted,
                }
            )
    return pd.DataFrame(rows)


def _check_inflation_consistency(metrics: pd.DataFrame, fig4: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    exp = _expected_inflation(metrics)
    e = _keyed(exp, ["strategy", "exec_model"])
    f = _keyed(fig4[["strategy", "exec_model", "ratio", "plotted"]], ["strategy", "exec_model"])

    for k, er in e.items():
        if k not in f:
            errors.append(f"fig_inflation_ratio: missing key {k}")
            continue
        fr = f[k]
        ep = bool(er["plotted"])
        fp = bool(fr["plotted"])
        if ep != fp:
            errors.append(f"fig_inflation_ratio: {k} plotted expected={ep} got={fp}")
            continue
        if ep:
            ev = _round3(er["ratio"])
            fv = _round3(fr["ratio"])
            if ev != fv:
                errors.append(f"fig_inflation_ratio: {k} expected={ev} got={fv}")

    for k in f.keys():
        if k not in e:
            errors.append(f"fig_inflation_ratio: unexpected key {k}")
    return errors


def _check_inflation_table(metrics: pd.DataFrame, infl: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    idx = _keyed(infl[["strategy", "exec_model", "sharpe_inflation_ratio"]], ["strategy", "exec_model"])
    for strat in metrics["strategy"].unique():
        sub = metrics[metrics["strategy"] == strat].copy()
        naive = sub[sub["exec_model"] == "naive"]
        if naive.empty:
            continue
        s0 = float(naive.iloc[0]["sharpe"])
        for _, r in sub.iterrows():
            model = str(r["exec_model"])
            if model == "naive":
                continue
            key = (str(strat), model)
            if key not in idx:
                errors.append(f"inflation_ratios.csv: missing key {key}")
                continue
            ev = _round3(s0 / float(r["sharpe"]))
            fv = _round3(float(idx[key]["sharpe_inflation_ratio"]))
            if ev != fv:
                errors.append(f"inflation_ratios.csv: {key} expected={ev} got={fv}")
    return errors


def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    tdir = os.path.join(args.out_dir, "tables")
    metrics_path = os.path.join(tdir, "metrics.csv")
    infl_path = os.path.join(tdir, "inflation_ratios.csv")
    fig3_path = os.path.join(tdir, "fig_sharpe_by_model_values.csv")
    fig4_path = os.path.join(tdir, "fig_inflation_ratio_values.csv")
    fig5_path = os.path.join(tdir, "fig_heatmap_values.csv")

    for p in [metrics_path, infl_path, fig3_path, fig4_path, fig5_path]:
        _require(p)

    metrics = pd.read_csv(metrics_path)
    infl = pd.read_csv(infl_path)
    fig3 = pd.read_csv(fig3_path)
    fig4 = pd.read_csv(fig4_path)
    fig5 = pd.read_csv(fig5_path)

    errors: List[str] = []
    errors.extend(_check_sharpe_consistency(metrics, fig3, "fig_sharpe_by_model"))
    errors.extend(_check_sharpe_consistency(metrics, fig5, "fig_heatmap"))
    errors.extend(_check_inflation_consistency(metrics, fig4))
    errors.extend(_check_inflation_table(metrics, infl))

    if errors:
        print("[FAIL] figure/table consistency checks failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("[OK] Figure inputs match canonical tables to 3 decimals.")
    print(f"[OK] Checked files under: {tdir}")


if __name__ == "__main__":
    main()
