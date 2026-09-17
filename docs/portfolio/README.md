# Portfolio PDF — source

The PDF is generated, not hand-edited. To rebuild after changing a number:

```
python charts.py     # matplotlib -> fig_frontier.svg, fig_latency.svg
python build.py      # inlines the SVGs + architecture diagram into portfolio.final.html
python render.py     # Chromium headless -> Jayden_Lee_Cascade_Classifier.pdf
```

Requires `matplotlib` and `playwright` (Chromium). Every figure in the document traces to a
run in this repository: the frontier to `src/sweep.py`, the latency table to
`src/serve/bench.py --provider anthropic`, the parity bound to `src/serve/verify.py`.

Chart colours are slots 1-3 of a CVD-validated categorical palette (blue/orange/aqua);
they are not decorative choices and should not be swapped casually.
