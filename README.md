# Mathematics for the Observation Programme

This is a primer for engineering students. It covers the mathematics used in the observation research programme, starting from the standard engineering sequence: calculus, differential equations, a first course in probability and statistics, and introductory linear algebra. It continues where Primer L and Primer S of *Data Mining as Observation* stop.

| Ch | Topic | Feeds |
|---|---|---|
| 0 | Preface and map from projects to chapters | all |
| 1 | Linear algebra beyond the first course | everything |
| 2 | Random vectors and Gaussians | Observation Theory, estimation and control |
| 3 | Information theory and rate–distortion | Observation Theory, the rate and leakage paper |
| 4 | Optimization | Observation Theory, Geometric Evaluation Theory |
| 5 | Geometry | the Geometric series, Observation Theory |
| 6 | Observation Theory, the core mathematics | Observation Theory |
| 7 | Quantization and vector search | turboquant-pro |
| 8 | Evaluation and decisions | Geometric Evaluation Theory, geometric economics and ethics |
| 9 | Statistics for registration-first research | DPE, every campaign |
| 10 | Logic and machine-checked proof | DPE, the Lean folders |

## Build

From the repository root, with TeX Live or MiKTeX:

```
latexmk -pdf -interaction=nonstopmode -outdir=build main.tex          # whole book
latexmk -pdf -interaction=nonstopmode -outdir=build/ch06 drivers/ch06.tex   # one chapter
```

`.latexmkrc` points bibtex at `refs/`.

## Checks

`checks/chNN_examples.py` recomputes every number in chapter NN's examples, answers and figures, and prints PASS or FAIL for each one. The checks run on the Atlas workstation, not on a laptop:

```
MSYS_NO_PATHCONV=1 python tools/run_checks.py <atlas helper>
```

A passing check confirms arithmetic and algebra. It does not confirm any claim about the projects; those claims rest on the project files each chapter points to.

## Writing rules

See `STYLE.md`.

## License

Two licenses, split by what the file is.

| What | License | File |
|---|---|---|
| Prose and figures: chapters, preface, appendices, answers, figures, README and style guide | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `LICENSE-TEXT` |
| Source code: check scripts, build tools, LaTeX preamble and drivers | [MIT](https://opensource.org/licenses/MIT) | `LICENSE` |

CC BY asks you to attribute the work and to indicate whether you changed it. Attribution that names the author, this primer and this repository, and says whether the text was changed, is enough.

## Status

Draft 0.1, in progress. Chapters are being drafted and checked. A chapter is ready to read when its `checks/chNN_examples.py` passes and it builds without errors.
