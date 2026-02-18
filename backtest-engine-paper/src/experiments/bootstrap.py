from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def block_bootstrap_sharpe(returns: pd.Series, n_samples: int = 500, block_size: int = 10, seed: int = 0) -> Tuple[float, float]:
    r = returns.dropna().astype(float).values
    if len(r) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(r)
    k = max(1, int(np.ceil(n / block_size)))
    sharpes = []
    for _ in range(n_samples):
        starts = rng.integers(0, max(1, n - block_size + 1), size=k)
        sample = []
        for s in starts:
            sample.append(r[s:s + block_size])
        samp = np.concatenate(sample)[:n]
        mu = float(np.mean(samp))
        sd = float(np.std(samp, ddof=1))
        sh = (mu / (sd + 1e-12)) * np.sqrt(252.0)
        sharpes.append(sh)
    lo, hi = np.percentile(np.array(sharpes), [2.5, 97.5])
    return float(lo), float(hi)
