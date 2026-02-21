# Paper: final LaTeX source
## Files
- paper.tex        — full manuscript (706 lines, ACM sigconf format)
- references.bib   — 14 fully formatted .bib entries
- figures/         — put fig1_roi_frontier.png, fig2_nasa_tlx.png, fig3_welfare_utility.png here

## Compile (local, 3 passes)
  cd paper_final
  pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex

## Overleaf
  1. New Project → Blank → upload all three files + figures/
  2. Menu → Compiler → pdfLaTeX
  3. Main document: paper.tex
  4. Compile