# Do LLM Assistants Actually Improve Productivity?
## A Causal Study of Latency, Cost, Output Quality, and Cognitive Wellbeing in Knowledge Work

**[Authors — Blinded for Review]** | **Target Venue: ACM CHI 2027**

---

## Abstract

Productivity metrics dominate evaluations of LLM deployment, yet time savings and quality improvements are welfare-improving *only if they do not increase cognitive burden*. We present a pre-registered randomized controlled experiment (N=200, Prolific Academic) measuring the causal impact of LLM and retrieval-augmented LLM (RAG) access on knowledge-work productivity. Participants complete tasks in information synthesis, structured writing, and light debugging under three conditions: no tool (Control), LLM-only (T1), and RAG-augmented LLM (T2). We measure time-to-complete, output quality (blind rater rubrics), per-task dollar cost, and cognitive workload via NASA-TLX. We introduce a **welfare utility model** that jointly trades quality gain against workload cost, surfacing cases where productivity gains are partially offset by cognitive burden increases. We also construct the first **ROI frontier** under randomized assignment, separately identifying the incremental causal value of retrieval augmentation — a gap in all prior field experiments. Design implications for HCI practitioners are derived from each empirical finding.

**Keywords:** LLM productivity, retrieval-augmented generation, RCT, cognitive workload, NASA-TLX, welfare utility, ROI frontier, appropriate reliance

---

## 1. Introduction

Generative AI tools promise to make knowledge workers faster and better. The evidence base is dominated by productivity metrics — time-to-complete, tasks per hour, rubric quality scores — that treat time savings as unambiguously beneficial. This assumption deserves scrutiny. A worker who completes a task 5 minutes faster but leaves the session frustrated, mentally exhausted, and uncertain about output accuracy has not necessarily experienced a welfare improvement. The productivity benefit accrues to the employer; the cognitive cost is borne by the worker.

This paper takes that asymmetry seriously. We present a pre-registered RCT measuring the **joint** distribution of time, quality, cost, and cognitive workload across three conditions: Control (no AI), T1 (LLM-only), and T2 (RAG-augmented LLM). We introduce a **welfare utility model** that formally trades quality gain against workload cost, enabling evaluation of whether AI assistance is beneficial to workers — not just to organizations.

We also address a methodological gap: all prior field experiments treat "AI access" as binary. None separately identifies RAG as a distinct causal treatment. This matters for deployment decisions — a chat bot and a document-grounded RAG pipeline both count as "AI access" in prior work, despite differing substantially in latency, cost, hallucination rate, and user experience.

**Primary contributions:**

1. **Welfare utility model:** \( \mathcal{W}^{(c)} = \Delta Q^{(c)} / (1 + \lambda \cdot \Delta \text{TLX}^{(c)}/100) \) — formally trading quality improvement against normalized workload cost
2. **RAG as a distinct causal treatment:** First RCT to separately estimate the incremental causal value of retrieval augmentation
3. **ROI frontier under randomization:** Joint (quality × time × cost × workload) decision surface for IT deployment
4. **HCI design implications:** Four concrete interface design recommendations derived from empirical findings

---

## 2. Related Work

### 2.1 Causal Evidence on LLM Productivity

Brynjolfsson, Li and Raymond (2023) study 5,179 customer-support agents, finding +15% throughput with largest gains for inexperienced workers. Dell'Acqua et al. (2023) randomize GPT-4 for 758 BCG consultants: on in-frontier tasks +12.2% tasks, 25.1% faster, 40%+ higher quality; on out-of-frontier tasks AI *hurts* — the "jagged frontier." Noy and Zhang (2023) find ~40% time reduction and ~18% quality improvement in professional writing (N=444). Peng et al. (2023) find 55% faster completion with GitHub Copilot (N=95).

None of these studies measures cognitive workload, separately identifies RAG, or constructs an ROI metric that weights workload cost. We address all three gaps.

### 2.2 Appropriate Reliance in HCI

Buccinca et al. (2021) find cognitive forcing functions reduce over-reliance on AI at a cost of increased task time. Schemmer et al. (2023) show reliance is modulated by task difficulty and worker confidence. Vasconcelos et al. (2023) demonstrate that fluent LLM explanations can increase over-reliance when wrong. This literature motivates our T2 design: if citations give false confidence, hallucination rates may decrease while appropriate skepticism also decreases — a welfare-negative outcome even when quality scores improve.

### 2.3 RAG Evaluation

Lewis et al. (2020) introduce RAG; Es et al. (2023) introduce RAGAS metrics (Faithfulness, Answer Relevance, Context Recall, Context Precision). We log RAGAS asynchronously for all T2 interactions as secondary automated measures. Shuster et al. (2021) show retrieval substantially reduces hallucination in knowledge-intensive tasks — our H5 tests this under randomization.

### 2.4 Cognitive Workload in HCI

Hart and Staveland (1988) introduce NASA-TLX (6 subscales, 0–100), validated across 20 years of HCI use (Hart 2006; ICC 0.56–0.73). Amershi et al. (2019) identify user frustration as a key failure mode when AI systems violate expectations about reliability, latency, and controllability. We pre-specify the **Frustration** subscale as the primary welfare indicator for AI tool evaluation.

### 2.5 Gap Table

| Study | N | Real workers | RAG separate | Workload | Cost |
|---|---|---|---|---|---|
| Brynjolfsson et al. (2023) | 5,179 | ✓ | ✗ | ✗ | ✗ |
| Dell'Acqua et al. (2023) | 758 | ✓ | ✗ | ✗ | ✗ |
| Noy & Zhang (2023) | 444 | ✓ | ✗ | ✗ | ✗ |
| Peng et al. (2023) | 95 | ✓ | ✗ | ✗ | ✗ |
| **This paper** | **200** | **Prolific** | **✓** | **✓** | **✓** |

---

## 3. Theory

### 3.1 The Productivity-Wellbeing Tradeoff

Standard productivity evaluation is incomplete for two reasons. First, quality improvements benefit output recipients; cognitive burden increases are paid by workers. A metric that ignores burden misattributes benefits and conceals costs. Second, RAG tools reduce hallucination on average but introduce a new failure mode: confident citation of retrieved but irrelevant passages. Workers who trust citations without scrutiny may produce worse outcomes than appropriately skeptical workers.

We formalize this as a welfare utility model. Let \( \Delta Q^{(c)} \), \( \Delta T^{(c)} \), and \( \Delta \text{TLX}^{(c)} \) denote ATEs for quality, time, and NASA-TLX composite under condition \( c \). Define:

\[ \mathcal{W}^{(c)} = \frac{\Delta Q^{(c)}}{1 + \lambda \cdot \max(0,\ \Delta \text{TLX}^{(c)}) / 100} \]

where \( \lambda > 0 \) is a welfare weight. We pre-specify \( \lambda = 1.0 \) and report sensitivity for \( \lambda \in \{0.5, 1.5, 2.0\} \). When \( \Delta \text{TLX}^{(c)} = 0 \), \( \mathcal{W}^{(c)} = \Delta Q^{(c)} \), reducing to the standard quality gain.

**Pre-specified predictions:**
- **P1:** T2 shows lower \( \mathcal{W} \) than T1 despite higher \( \Delta Q \), due to retrieval latency raising Frustration (H7)
- **P2:** For Category C, T2 may show near-zero \( \mathcal{W} \) if the corpus is unhelpful and raises workload without quality gain (H11 exploratory)

---

## 4. Method

### 4.1 Participants
N=200 via Prolific Academic (max 250). Eligibility: English proficiency, regular computer use, passing screening task. Prior AI use recorded (not exclusion). Pay: ~$18–22/hr.

### 4.2 Design
Within-subject 3×3 Latin Square crossover. Each participant: 9 tasks (3 categories × 3 conditions), condition order counterbalanced within category. Task-condition assignment from pre-committed seed.

### 4.3 Conditions
| Condition | Tool | Key notes |
|---|---|---|
| Control | Web browser only | Honesty attestation + log inspection |
| T1 (LLM-only) | GPT-4o-2024-11-20 | temperature=0.2; no retrieval; all costs logged |
| T2 (RAG) | GPT-4o + FAISS | Top-k=5 chunks, 512 tokens; citations required; RAGAS logged async |

### 4.4 Measures
- **Time-to-complete (min):** Active wall-clock; pauses >60s subtracted
- **Quality score (0-10):** 2 blind raters; adjudication if |r1-r2|>1; Krippendorff α ≥ 0.70
- **Cost (USD):** Token + retrieval overhead; Control = $0.00
- **NASA-TLX:** 6 subscales (0-100); administered per-task before any feedback; Frustration = primary welfare indicator
- **Hallucination rate:** Flagged claims/total (Cat A/B); failed tests/total (Cat C)
- **Welfare utility \( \mathcal{W} \):** Computed post-hoc from ATE estimates

---

## 5. Analysis

**Primary specification:**
\[ Y_{it} = \alpha + \beta_1 \mathbf{1}[T1]_{it} + \beta_2 \mathbf{1}[T2]_{it} + \gamma X_i + \delta_t + \lambda_{\text{order}} + \varepsilon_{it} \]

Task FEs \( \delta_t \), participant-clustered SEs, condition-order FEs \( \lambda_{\text{order}} \), covariates \( X_i \) = skill + prior AI use.

**Multiple testing:** BH FDR q=0.10 across 12-test primary family (H1–H8 across contrasts).

**ROI metrics:** Bootstrap BCa 95% CIs (1,000 reps) for all ratios.

---

## 6. Results *(Placeholders — Data Collection Pending)*

### Table 2: Primary ATE Estimates
| Outcome | T1 vs C (ATE) | 95% CI | q-BH | T2 vs C (ATE) | 95% CI | q-BH | T2 vs T1 (ATE) | 95% CI |
|---|---|---|---|---|---|---|---|---|
| Time (min) | — | — | — | — | — | — | — | — |
| Quality (0-10) | — | — | — | — | — | — | — | — |
| Cost (USD) | — | — | — | — | — | — | — | — |
| Hallucination | — | — | — | — | — | — | — | — |
| TLX Frustration | — | — | — | — | — | — | — | — |
| TLX Composite | — | — | — | — | — | — | — | — |

### Table 3: Welfare Utility Estimates
| Condition | ΔQ | ΔTLX | W(0.5) | W(1.0) | W(1.5) | W(2.0) |
|---|---|---|---|---|---|---|
| T1 vs Control | — | — | — | — | — | — |
| T2 vs Control | — | — | — | — | — | — |
| T2 vs T1 | — | — | — | — | — | — |

**[FIGURE 1]** ROI Frontier: ΔCost vs ΔQuality with welfare utility contours
**[FIGURE 2]** NASA-TLX 6-subscale profile by condition
**[FIGURE 3]** Welfare utility W by condition and λ
**[FIGURE 4]** Heterogeneity: quality ATE by skill × task difficulty

---

## 7. Discussion

### 7.1 Welfare Utility (Primary Contribution)
If P1 is confirmed — T2 raises quality but also raises Frustration, yielding lower welfare utility — the implication is direct: RAG deployment is not "add retrieval and win." The cognitive cost of evaluating citations can partially offset the quality benefit, especially for certain task types and skill levels.

### 7.2 Appropriate Reliance and Citation Effects
If H5 is confirmed (T2 reduces hallucination) but H7 is also confirmed (T2 raises Frustration), the tension suggests two competing effects: retrieval genuinely reduces errors, but the cognitive overhead of evaluating sources frustrates workers. Future work should directly manipulate citation display to isolate these effects.

### 7.3 Design Implications for HCI
Four actionable implications follow:

1. **Latency is a welfare feature.** If retrieval latency correlates with Frustration increases, caching and speculative retrieval are welfare interventions. Measure them as such.
2. **Citation display needs cognitive scaffolding.** Raw retrieved text raises evaluation burden. Interfaces should surface confidence indicators and explicit uncertainty signals to calibrate reliance without adding frustration.
3. **Skill-adaptive deployment.** If H9 is confirmed (low-skill workers benefit more), organizations should prioritize LLM access for lower-skill roles rather than uniform rollout.
4. **Task routing for the jagged frontier.** If quality gains are near-zero or negative for hard tasks (H10), interfaces should route out-of-frontier tasks away from AI assistance entirely.

---

## 8. Limitations
1. **Prolific vs. organizational workers** — different motivation, skill, domain familiarity
2. **Control compliance** — honesty attestation is imperfect; anomalous log sensitivity analysis conducted
3. **Single model and corpus** — GPT-4o-2024-11-20 specific; results expected to vary
4. **Welfare weight lambda** — pre-specified but normative; sensitivity reported; appropriate value depends on organizational context

---

## 9. Conclusion
We present a pre-registered RCT jointly measuring time, quality, cost, and cognitive wellbeing under randomized assignment, producing a welfare-respecting ROI frontier. Our primary contribution — the welfare utility model — reframes the standard AI productivity question from "does it improve output?" to "does the improvement justify the cognitive cost it imposes on workers?" All materials are released openly for replication and extension.

---

## References

Amershi, S., et al. (2019). Software engineering for machine learning. *ICSE-SEIP*.
Buccinca, Z., Malaya, M. B., & Gajos, K. Z. (2021). To Trust or to Think. *CSCW*.
Brynjolfsson, E., Li, D., & Raymond, L. (2023). Generative AI at Work. *NBER WP 31161*.
Dell'Acqua, F., et al. (2023). Navigating the Jagged Technological Frontier. *HBS WP 24-013*.
Es, S., et al. (2023). RAGAS. *arXiv:2309.15217*.
Hart, S. G., & Staveland, L. E. (1988). Development of NASA-TLX. *Human Mental Workload*, 139–183.
Hart, S. G. (2006). NASA-TLX: 20 Years Later. *HFES Annual Meeting*.
Lewis, P., et al. (2020). Retrieval-Augmented Generation. *NeurIPS*.
Noy, S., & Zhang, W. (2023). Experimental Evidence on GenAI Productivity. *Science*, 381, 187–192.
Peng, S., et al. (2023). The Impact of AI on Developer Productivity. *arXiv:2302.06590*.
Schemmer, M., et al. (2023). Appropriate Reliance on AI. *IUI*.
Shuster, K., et al. (2021). Retrieval Augmentation Reduces Hallucination. *EMNLP Findings*.
Vasconcelos, H., et al. (2023). Explanations Can Reduce Overreliance on AI. *CSCW*.
Benjamini, Y., & Hochberg, Y. (1995). Controlling the FDR. *JRSS-B*, 57(1), 289–300.
