# Style and notation brief for chapter authors

This book is **Mathematics for the Observation Programme. A primer for engineering students.** It gets a student with a standard engineering background up to the mathematics used in Andrew Bond's research programme. That background is calculus, differential equations, a first course in probability and statistics, and some linear algebra. The programme has four areas:
- Observation Theory
- the Geometric series of books, including Geometric Evaluation Theory
- turboquant-pro and vector search
- the registration-first research method, Discovery Philosophy Engineering (DPE)

## Audience and level

- The reader knows calculus (gradients, Lagrange multipliers, Taylor expansion) and ODEs. They know basic probability: random variables, expectation, variance, the normal distribution, a confidence interval and a p-value. They know matrix multiplication, determinants and eigenvalues of small matrices.
- First-course material is already written. It is in `C:\source\observation-data-mining\chapters\primer_l_linear_algebra.md` (Primer L) and `primer_s_probability_and_statistics.md` (Primer S), with chapter 0 of that book at `ch00_the_mathematics_this_book_uses.md`. **Read the parts relevant to your chapter before writing.** Do not re-teach that material. Cite it as "Primer L, section L.7" (or Primer S) of *Data Mining as Observation* when the reader needs to look back, and start one level higher.
- Every new concept is built from something the reader knows, with a worked example small enough to check by hand.

## Chapter structure

- 6 to 9 sections, and 5,000 to 7,000 words of prose plus the mathematics.
- **Section headings are sentences that state the point.** Examples are "The trace of a product is a weighted sum of variances" and "Water-filling is what KKT gives for a log objective". Do not use bare category names like "Traces".
- Each section introduces its object in words, gives the definition or result as a **numbered displayed equation** (or a `definition` / `theorem` environment), and then works an `example` with concrete numbers. It ends with `\buildson{...}` and `\usedin{...}`. `\usedin` points to later chapters of this book and to the project files where the tool appears.
- Use `\inproject{...}` boxes (one or two per section, short) to show the reader exactly where the object appears in the programme, with a repository path. An example is `geometric-observation/paper/observation-theory.tex`, Theorem "Omission floor". Only point at files you have opened and confirmed.
- Proofs are included when they are short and teach something, for example the KKT derivation of water-filling or the Schur-complement conditioning formula. Otherwise state the result and cite a textbook.
- End with `\section*{Exercises}` and three groups:
  - **Warm-up**: 3 to 4 exercises
  - **By hand**: 4 to 5
  - **Programming**: 2 to 3, numpy/sympy, small, runnable in seconds
- Write answers to at least half the exercises in `answers/chNN.tex`. Each answer is a `\paragraph{Exercise N.k}` followed by the answer.

## Prose rules (the author's standing standard)

These apply to prose, captions and headings. Mathematics, `\label`s, code and bibliography titles are exempt.
- **No em-dashes, no colons, no semicolons in prose.** Rewrite as separate sentences. "Namely" and "that is" are fine.
- No filler words. "Gap", "regime", "paradigm", "crucially", "notably", "delve" and "landscape" are banned.
- No self-justifying sentences about the book's own value or honesty. Every sentence states something.
- Precise beats smooth. If a word implies more than the mathematics shows, change the word.
- **Hypothesis discipline.** Students remember the sentence and forget the conditions. Every theorem, rule of thumb and memorable explanation states its scope in the same sentence or the next one: when it holds, and what breaks it. Where a natural reading overreaches, give the counterexample. A positive floor does not imply a divergent rate, for instance, and a permutation null preserves dependence only under the right exchangeability.
- **Never hard-code a chapter or exercise number.** Printed numbers change when chapters are added. Use `\cref`/`\ref` everywhere, including answer headings: `\paragraph{Exercise~\ref{exr:chNN:name}}`.
- Put each caveat next to its result. Say what a quantity is NOT when confusion is likely. For example, "this is a joint probability, not a conditional rate".
- The prose rules govern prose, not mathematics. **Every definition and every result gets a numbered displayed equation.** A textbook without formulas is a failure (the author has flagged this before).

## Honesty rules (mandatory)

- **Never present a guess as fact.** If you cannot confirm something about a project, leave it out, or say "the project files do not state this".
- **Credit prior art.** Much of Observation Theory renames standard objects. The author's own assessment is `C:\source\geometric-observation\articles\2026-10-01-ot-novelty-assessment.md`, and the chapter on OT must follow it. Consumer-relative distortion is a weighted MSE in the sense of Sakrison, and Constantine's active subspaces are another precedent. Task-based quantization and the "conditional content" leakage coordinate are the secure-source-coding leakage of Villard–Piantanida 2013 and Ekrem–Ulukus 2013. Never call a renamed object new.
- For Geometric Evaluation Theory, position it against Dawid and Lauritzen, "The geometry of decision theory" (2005), which is the unbudgeted, identity-evaluator special case. Do not call it GDT.
- In turboquant-pro, **reconstruction cosine is never an acceptance metric**. Acceptance is rank fidelity (recall@k, Kendall tau) or the downstream consumer's own metric. Teach why.
- Use the programme's claim status words exactly as the project files use them: `[proved]`, `[demonstrated]`, `[replicated]`, `[predicted]`, `[exploratory]`, `[refuted]` (see `geometric-observation/PROTOCOL.md`).
- **Citations.** Cite only works you have confirmed exist with the stated authors, title, venue and year. Confirm by fetching `https://api.crossref.org/works/<DOI>`, or a publisher, arXiv or PhilPapers page. Prefer standard textbooks:
  - Cover & Thomas, *Elements of Information Theory*, 2nd ed. (Wiley 2006)
  - Boyd & Vandenberghe, *Convex Optimization* (CUP 2004)
  - Horn & Johnson, *Matrix Analysis*, 2nd ed. (CUP 2012)
  - Gray & Neuhoff, "Quantization" (IEEE T-IT 1998)
  - Kay, *Fundamentals of Statistical Signal Processing*
  - Lee, *Introduction to Riemannian Manifolds*
  - Amari, *Information Geometry and Its Applications*
  - Efron & Tibshirani, *An Introduction to the Bootstrap*
  
  Put full BibTeX in `refs/chNN.bib` with keys of the form `authoryear` (e.g. `coverthomas2006`). An unconfirmed citation is worse than none.
- **Every number in a worked example must be checked by a script.** See the section on verification below.

## LaTeX conventions

- Write only your own files: `chapters/chNN_*.tex`, `answers/chNN.tex`, `refs/chNN.bib`, `checks/chNN_examples.py` and `figures/chNN_*`. Do NOT edit `preamble.tex`, `main.tex`, `STYLE.md` or other chapters. If you need a macro that is not in the preamble, define it at the top of your chapter with `\providecommand` and mention it in your report.
- The first line of the chapter is `\chapter{<sentence-style title>}\label{chNN}`. Label everything with your chapter prefix, e.g. `\label{eq:ch03:gauss-rd}`, `\label{sec:ch03:water}`, `\label{ex:ch03:two-dims}`. Use `\cref`.
- **Figures: 6 to 9 per chapter. A picture is worth a thousand words.** Most sections should have one. Requirements:
  - Inline TikZ or pgfplots only, with no external images, and self-contained in the chapter file. Wrap each in `\begin{figure}[tbp]\centering ... \caption{...}\label{fig:chNN:name}\end{figure}` and refer to it in the text with `\cref`.
  - Use the shared palette and styles from `preamble.tex`:
    - colours: `pblue` for the source or signal, `porange` for the consumer or read operator, `pgreen` for the optimum or what is kept, `pred` for the error, floor or rejected, `pgray` for guides, `plight` for fills
    - TikZ styles: `vec`, `guide`, `block`, `flow`, `world`, `lbl`
    - pgfplots style: `primer` (use `\begin{axis}[primer, ...]`)
  - Match the clean look of the author's accepted INFOCOM paper (`C:\source\nats-bursting\paper\infocom.tex`, lines 131–147, and `infocom_eval.tex`). That means thick strokes, few colours, labels in small font placed on the object rather than in a distant legend, and no chartjunk.
  - **Every plotted number is computed, never invented.** Coordinates for curves and bars are generated by your `checks/chNN_examples.py`, which prints them and checks them, and are pasted into the figure. A schematic diagram, such as a block diagram or a Kripke frame, carries no data and needs no check.
  - Captions are full sentences in the prose style, with no colons or semicolons. They tell the reader what to see, for example "The consumer-aware code reconstructs worse yet serves the consumer better". Do not write "Plot of X versus Y".
  - Prefer one "thesis in an image" figure per chapter, the single picture a student should remember.
- Use the shared macros in `preamble.tex`:
  - `\tr`, `\E`, `\Var`, `\Cov`, `\T` (transpose, written `x\T y`), `\psd` (Loewner order)
  - `\PC`, `\Sd`, `\dO`, `\RC`
  - `\KL{p}{q}`, `\MI{X}{Y}`, `\h`, `\Normal`, `\R`
  - `\inner{}{}`, `\norm{}`, `\diag`, `\rank`, `\range`, `\argmin`
- **Build your chapter** from the repository root:
  `latexmk -pdf -interaction=nonstopmode -outdir=build/chNN drivers/chNN.tex`
  It must compile with no errors and no undefined references or citations. Run it before you finish.

## Shared notation (use exactly this; see also the read-operator sections of ch00)

| Symbol | Meaning |
|---|---|
| $x\in\R^d$ | a source vector (a column); rows of a data table written as vectors |
| $\Sigma_x$ | source covariance; $\Sigma_x\pd 0$ unless stated |
| $C$ | a consumer, a computation that reads a representation and outputs something |
| $J=\partial C/\partial x$ | Jacobian of the consumer; $g=\nabla C$ for a scalar consumer |
| $G$ | output metric of the consumer, PSD |
| $O=(C,G,B)$ | an observer: consumer, output metric, budget |
| $P_C=J\T G J$ | read operator (pointwise); $\bar P_{C,\mu}=\E_\mu[P_C(x)]$ workload average; PSD |
| $\delta=\hat x-x$ | reconstruction error; $M_\delta=\E[\delta\delta\T]$; $\Sigma_\delta$ when centred |
| $d_O=\tr(P_C\Sigma_\delta)$ | observer distortion; identity reader $P_C=I$ gives $\tr\Sigma_\delta$ |
| $R(D)$, $R_C(D)$ | rate–distortion function; consumer-relative version |
| $\theta$ | water level in reverse water-filling |
| $\log$ | natural log (nats) unless written $\log_2$ (bits); always state the unit |
| $\KL{p}{q}$, $\MI{X}{Y}$, $H$, $h$ | KL divergence, mutual information, entropy, differential entropy |
| $A\psd B$ | Loewner order: $A-B$ is PSD |

## Verification (mandatory)

- **Put every numerical claim in a worked example, exercise answer or figure into `checks/chNN_examples.py`.** Use sympy where exact, numpy where not. Each claim gets `check("description", condition)`, which prints PASS or FAIL. The script exits nonzero on any FAIL and prints `k/n PASS` at the end.
- **Do not run compute on the laptop.** Run the check on the Atlas workstation with the helper (MSYS path conversion must be off):
  ```
  H="<path to the Atlas SSH helper; not part of this repository>"
  MSYS_NO_PATHCONV=1 python "$H" run "mkdir -p ~/omp-checks"
  MSYS_NO_PATHCONV=1 python "$H" put C:/source/observation-math-primer/checks/chNN_examples.py /home/claude/omp-checks/chNN_examples.py
  MSYS_NO_PATHCONV=1 python "$H" run "cd ~/omp-checks && OPENBLAS_NUM_THREADS=4 /home/claude/env/bin/python3 chNN_examples.py"
  ```
  (sympy 1.14 and numpy are installed there.) Fix the text or the check until everything passes. Paste the final `k/n PASS` line as a comment at the end of your chapter file.
- Building LaTeX locally with latexmk is allowed.
- Never stop or kill processes on Atlas other than your own, and never reboot it.

## Your report

When you finish, report:
- word count and section titles
- the check result
- every citation and how you confirmed it
- every project file you pointed to
- anything you could not confirm and therefore left out
