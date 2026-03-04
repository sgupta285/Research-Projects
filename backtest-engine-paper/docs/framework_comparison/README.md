# Framework Comparison Evidence

This directory documents the evidence basis for Table 1 in the paper
("Backtest Engineering: A Modular Execution-Realism Framework for Credible Strategy Evaluation").

## Methodology

Table 1 features were assessed by:
1. Reviewing the public GitHub repository of each framework (source code + test suite)
2. Reviewing official documentation
3. Checking PyPI release pages for maintenance status

All assessments were made based on publicly available information as of Q1 2025.

## Framework Snapshots

| Framework     | GitHub URL                                      | Release used      | Notes |
|---------------|-------------------------------------------------|-------------------|-------|
| Zipline       | https://github.com/quantopian/zipline           | v1.3.0 (archived) | Quantopian shut down 2020; `zipline-reloaded` is a fork |
| Backtrader    | https://github.com/mementum/backtrader          | v1.9.78.123       | Active in 2024; has commission/slippage but no formal accounting tests |
| VectorBT      | https://github.com/polakowo/vectorbt            | v0.26.1           | Vectorised design precludes event-level causality by construction |
| LEAN          | https://github.com/QuantConnect/Lean            | v2.5 (C# engine)  | Extensive test suite but no backtesting-specific accounting invariant tests |

## Feature Verification Notes

### Formal causality unit tests (row 2)
- **Zipline**: No dedicated test verifying that signals computed at bar close cannot access bar open of the same bar.
- **Backtrader**: No such test found in `tests/` directory.
- **VectorBT**: Vectorised design makes per-bar causality enforcement structural; no event-level test.
- **LEAN**: Tests cover order routing but not the timestamp-ordering invariant as defined in §4.1.

### Accounting identity tests (row 3)
- All four frameworks: No test of the form `portfolio_value(t) == portfolio_value(t-1) × (1 + returns) - costs` at each bar, as implemented in this engine's `tests/test_invariants.py`.

### Fill-quantity bound tests (row 4)
- **LEAN**: Has partial fill logic but no explicit test that `q_filled ≤ ADV × ρ` at every bar.
- Others: No such tests found.

### Per-tier Sharpe attribution (row 6)
- None of the reviewed frameworks expose a mechanism to isolate the Sharpe contribution of each individual cost tier (M0→M1→...→M5).

### Multiple block-length CI (row 8)
- None of the reviewed frameworks include moving-block bootstrap CI computation over multiple block sizes.

## How to Reproduce This Verification

To verify any row in Table 1:
1. Clone the framework at the release listed above
2. Search for tests matching the feature description
3. Check documentation for first-class feature support

Command to search Zipline for accounting tests:
```bash
git clone https://github.com/quantopian/zipline --branch 1.3.0
grep -r "portfolio_value\|accounting" zipline/tests/ --include="*.py" -l
```
