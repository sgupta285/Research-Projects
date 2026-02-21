"""
Power Calculation for LLM-ROI Study (CHI 2027, N=200).
Usage: python3 analysis/scripts/power_calculation.py
       python3 analysis/scripts/power_calculation.py --effect_size 0.35 --rho 0.4 --n_target 200
"""
import numpy as np
from scipy import stats
import argparse


def power_within(d, rho, n, alpha=0.05):
    sd_diff = np.sqrt(2 * (1 - rho))
    ncp = d / sd_diff * np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    return 1 - stats.t.cdf(t_crit, n - 1, loc=ncp) + stats.t.cdf(-t_crit, n - 1, loc=ncp)


def required_n(d, rho, power=0.80, alpha=0.05):
    for n in range(5, 1000):
        if power_within(d, rho, n, alpha) >= power:
            return n
    return 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--effect_size", type=float, default=0.35)
    p.add_argument("--rho", type=float, default=0.40)
    p.add_argument("--power", type=float, default=0.80)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n_target", type=int, default=200)
    args = p.parse_args()

    n_min = required_n(args.effect_size, args.rho, args.power, args.alpha)
    achieved = power_within(args.effect_size, args.rho, args.n_target, args.alpha)

    print("=" * 60)
    print("  LLM-ROI Study — Power Calculation (CHI 2027 target N=200)")
    print("=" * 60)
    print(f"  d = {args.effect_size}  |  rho = {args.rho}  |  alpha = {args.alpha}  |  target power = {args.power}")
    print(f"  Minimum required N:            {n_min}")
    print(f"  Recommended N (+20% attrition): {int(np.ceil(n_min * 1.2))}")
    print(f"  Power at N={args.n_target}:           {achieved:.4f}")
    print("=" * 60)

    print("\nPower at N=200 by d and rho:")
    rhos = [0.20, 0.30, 0.40, 0.50, 0.60]
    print(f"  {'d':>5}  " + "  ".join(f"rho={r:.2f}" for r in rhos))
    for d in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        vals = "  ".join(f"{power_within(d, r, 200):>8.3f}" for r in rhos)
        print(f"  d={d:.2f}  {vals}")


if __name__ == "__main__":
    main()
