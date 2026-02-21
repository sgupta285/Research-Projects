# Quality Rubric — Blind Rater Scoring Guide
Raters see ONLY task prompt + submitted response. No condition, PID, or session metadata shown.

## Category A — Information Synthesis (10 pts)
| Dimension       | Max | 3 | 2 | 1 | 0 |
|-----------------|-----|---|---|---|---|
| Factual accuracy | 3 | All claims correct | ≤1 minor error | ≥1 material error | Fundamentally wrong |
| Completeness    | 3 | All key elements | Misses 1 minor | Misses ≥1 major | Substantially incomplete |
| Source attribution | 2 | Specific credible sources | Vague | None | — |
| Coherence       | 2 | Logically structured | Partially | Incoherent | — |

## Category B — Structured Writing (10 pts)
| Dimension       | Max | Anchors |
|-----------------|-----|---------|
| Content accuracy | 3 | 3=accurate on-task; 2=minor issues; 1=significant off-target; 0=fails objective |
| Format adherence | 2 | 2=follows genre exactly; 1=minor deviation; 0=wrong format |
| Clarity         | 2 | 2=clear no padding; 1=some verbosity; 0=unclear |
| Professional tone | 3 | 3=fully professional; 2=minor issues; 1=inappropriate; 0=unprofessional |

## Category C — Coding (10 pts)
| Dimension       | Max | Anchors |
|-----------------|-----|---------|
| Correctness     | 4 | 4=all tests pass; 3=main+1 edge fail; 2=partial; 1=right approach; 0=wrong |
| Code quality    | 2 | 2=readable idiomatic; 1=functional poor style; 0=unreadable |
| Edge cases      | 2 | 2=all pre-specified; 1=some; 0=none |
| Explanation     | 2 | 2=accurate docstring; 1=partial; 0=absent |

## Agreement Rules
- |r1 - r2| ≤ 1: average scores
- |r1 - r2| > 1: 3rd rater adjudicates
- Krippendorff α ≥ 0.70 per category required before live rating begins
