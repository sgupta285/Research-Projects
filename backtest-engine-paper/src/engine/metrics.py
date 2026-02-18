from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Metrics:
    cagr: float
    sharpe: float
    max_drawdown: float
    vol_ann: float
    turnover: float


def compute_metrics(equity: pd.Series, turnover: pd.Series) -> Metrics:
    eq = equity.astype(float)
    rets = eq.pct_change().dropna()
    if len(rets) < 2:
        return Metrics(0.0, 0.0, 0.0, 0.0, float(turnover.sum()))

    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)

    vol_ann = float(rets.std(ddof=1) * np.sqrt(252.0))
    sharpe = float((rets.mean() / (rets.std(ddof=1) + 1e-12)) * np.sqrt(252.0))

    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min())

    return Metrics(cagr=cagr, sharpe=sharpe, max_drawdown=mdd, vol_ann=vol_ann, turnover=float(turnover.sum()))
