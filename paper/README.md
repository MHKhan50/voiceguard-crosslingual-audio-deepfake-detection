# Paper source

`main.tex`, `references.bib` and `figures/` build the preprint (pdfLaTeX + BibTeX, or upload the folder to Overleaf).
The figures are drawn only from the saved result files:

```bash
python paper/make_figures.py
cd paper && latexmk -pdf main.tex
```

Every number in the tables comes from `results/results.json` (ASVspoof 2019 LA) and
`results/in_the_wild/itw_results.json` (In-the-Wild).
