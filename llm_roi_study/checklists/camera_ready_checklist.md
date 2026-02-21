# Camera-Ready Checklist — CHI 2027

## CHI-Specific Reviewer Concerns

| Attack | Defense |
|--------|---------|
| "Prolific ≠ real workers" | Scoped to online knowledge work; both AI-exp and naive workers; demographics table; same population as Noy & Zhang (2023) |
| "NASA-TLX is gameable self-report" | Administered per-task before feedback; validated instrument ICC 0.56-0.73 (Hart 2006); subscale-by-subscale consistency check |
| "Welfare utility lambda is arbitrary" | Pre-specified lambda=1.0; sensitivity for lambda in {0.5,1.5,2.0}; presented as decision tool not universal truth |
| "No design implications — this is measurement" | §7.4 maps each finding to a concrete interface design change (latency, citation affordances, skill-adaptive routing) |
| "Task bank is artificial" | Exit survey item: participant rates task similarity to real work; used as heterogeneity covariate |

## Causality / Identification

| Attack | Defense | Extra Ablation |
|--------|---------|----------------|
| "Learning confounds condition" | Latin square + order FEs | Drop first task per condition; re-estimate |
| "Carryover T2 → Control" | Order FE absorbs systematic carryover | Placebo: regress Control quality on whether T2 came first in that category |
| "Control compliance" | Honesty attestation + log inspection | Sensitivity: exclude anomalous completions |

## Statistics

| Attack | Defense |
|--------|---------|
| "Multiple comparisons" | BH FDR q=0.10 on 12-test primary family; all p & q reported; Bonferroni in appendix |
| "N for heterogeneity" | N=200 powered for d=0.25 interactions; effect sizes + CIs; no binary framing |
| "Welfare model untested" | Pre-registered; sensitivity to lambda; participant-rated welfare in exit survey as validation |
| "ROI ratio CIs wide" | Bootstrap BCa 95% CIs (1000 reps); numerator and denominator reported separately |

## Reproducibility

| Attack | Defense |
|--------|---------|
| "Cannot reproduce" | Full code, task bank, configs on OSF. Hashed logs. Full text under data-sharing agreement. |
| "Stochastic outputs" | temperature=0.2; model version pinned; response hashes logged |
