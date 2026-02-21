# LLM-ROI Study — paper v3 (fully revised after editorial review)

## Files
- paper.tex       — Full manuscript (936 lines, standard article class)
- references.bib  — 18 fully verified bib entries
- figures/        — Drop fig1_roi_frontier.png, fig2_nasa_tlx.png, fig3_welfare_utility.png here

## What changed from v2 (all 30+ editorial fixes applied)
CRITICAL fixes:
  - OSF pre-registration placeholder added (Abstract + Methods + Appendix A)
  - Dell'Acqua citation corrected: "12.2% more tasks, 25.1% faster, 40% higher quality"
  - Brynjolfsson N clarified (5,179 NBER WP; footnote discloses arXiv discrepancy)
  - Noy & Zhang: N=453 recruited, 444 analyzed (footnote in bib + Table 1)
  - Power formula shown explicitly (Eq.2, within-subject crossover)
  - "Welfare utility" renamed -> "Quality-Workload Trade-off Index (QWTI)"
  - lambda=1.0 justified as symmetric reference; conclusions held for lambda in [0.5,2.0]
  - P2 removed as "prediction" (relabeled as mathematical property of formula)

MAJOR fixes:
  - Amershi et al. corrected to CHI 2019 "Guidelines for Human-AI Interaction" (doi fixed)
  - NASA-TLX ICC claim removed; replaced with Hart (2006) benchmark (10-15pt threshold)
  - Frustration increase for T1 (+2.15, d=0.14) now explicitly noted and contextualized
  - Cohen's d added to every row of ATE table
  - Control compliance sensitivity: Section 5.3 + Appendix E
  - Rater blinding: explicit statement that citation markers stripped before rating
  - IRB mention added (Section 3, placeholder)
  - Timeout rates reported (Section 3.4: Control 2.8%, T1 1.7%, T2 1.2%)
  - RAGAS metrics reported in Appendix D (Faithfulness=0.81, corr with hallucination rho=-0.62)
  - Table 1: Descriptive statistics by condition (NEW)
  - Table 2: Gap analysis fixed (Noy & Zhang coded as Prolific, not "real workers")
  - Table 4 (QWTI): Added Frustration-only specification as conservative robustness check
  - "Three field experiments" -> "four field experiments" fixed
  - "Quarter-century later" -> "Nearly three decades later"
  - "Are released openly" -> "Will be released publicly upon acceptance"

STRUCTURAL changes:
  - Method now before Theory (Section 3, 4 swapped)
  - Theory section renamed Section 4: "Theory: The Quality-Workload Trade-off Index"
  - Results split: Section 5 (Primary Results) + Section 6 (Welfare Analysis)
  - Robustness checks added: Section 5.3 + Appendix E
  - Limitations expanded to subsections (Section 7.4)
  - Appendix A: Full hypothesis list H1-H12
  - Appendix B: Descriptive stats by task category
  - Appendix C: Inter-rater reliability (quality alpha, hallucination kappa)
  - Appendix D: RAGAS validation metrics
  - Appendix E: Robustness checks (timeout, compliance, exclude-first-task, aggregation)
  - Appendix F: Task examples

CITATION additions:
  - Bansal et al. (2019) for appropriate reliance / complementary performance
  - Paas et al. (2003) and Sweller (1988) for cognitive load theory
  - Acemoglu & Restrepo (2018) for automation/labor welfare economics

## Compile
  pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
  open paper.pdf