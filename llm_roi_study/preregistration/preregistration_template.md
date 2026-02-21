# Pre-Registration (OSF) — CHI 2027
Title: Do LLM Assistants Actually Improve Productivity?
Date: [YYYY-MM-DD — MUST be before data collection]

## Hypotheses
| ID  | Hypothesis | Direction | Outcome | Family |
|-----|-----------|-----------|---------|--------|
| H1 | T1 reduces time vs Control | Decrease | Time (min) | Primary |
| H2 | T2 reduces time vs Control | Decrease | Time (min) | Primary |
| H3 | T1 improves quality vs Control | Increase | Quality (0-10) | Primary |
| H4 | T2 improves quality vs Control | Increase | Quality (0-10) | Primary |
| H5 | T2 reduces hallucination vs T1 | Decrease | Hallucination rate | Primary |
| H6 | T2 increases cost vs T1 | Increase | Cost (USD) | Primary |
| H7 | T2 raises NASA-TLX Frustration vs T1 (latency burden) | Increase | TLX-Frustration | Primary |
| H8 | T1 NASA-TLX composite does not significantly differ from Control (null) | Null | TLX-composite | Primary |
| H9 | Low-skill participants benefit more from T1/T2 on quality | Interaction | Quality | Secondary |
| H10 | Quality gains larger for easy/medium vs hard tasks | Interaction | Quality | Secondary |
| H11 | T2 incremental quality > T1 for Cat-A; null for Cat-C | Moderation | Quality | Exploratory |
| H12 | Welfare utility W lower for T2 than quality gain alone predicts | Welfare | W composite | Exploratory |

## Sample Size
- N = 200 (max 250)
- Powers d=0.35, within-subj rho=0.40 at alpha=0.05: 99.4% power
- Powers d=0.25 for heterogeneity analyses: ~80% at N=200
- Powers d=0.30 for NASA-TLX Frustration (H7): ~94% power

## Welfare Utility (pre-specified functional form)
W = delta_Q / (1 + lambda * max(0, delta_TLX_composite) / 100)
lambda = 1.0 (pre-specified); sensitivity: lambda in {0.5, 1.5, 2.0}

## Analysis Plan
1. OLS with task FEs, participant clustered SEs, condition dummies
2. Heterogeneity: condition x skill-tercile and condition x task-difficulty
3. Multiple testing: BH FDR q=0.10 across 12-test primary family (H1-H8)
4. Robustness: (a) Winsorize time 97.5th pct (b) GPT-4o-judge quality
   (c) carryover exclusion (d) Wilcoxon (e) +-20% price sensitivity (f) leave-one-task-out

## Stopping Rules
- Enroll to N=250; no early stopping for efficacy
- Halt if >25% participants have logs inconsistent with genuine completion
