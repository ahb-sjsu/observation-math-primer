"""Write drivers/chNN.tex, a standalone build of one chapter with its answers and references.

Chapter numbers follow the order of the \\include lines in main.tex, so a chapter
whose file key is ch11 but which the book places third is numbered 3.

Run from the repository root: python tools/make_drivers.py
Build one chapter: latexmk -pdf -interaction=nonstopmode -outdir=build/chNN drivers/chNN.tex
"""
import re

BS = "\\"
main = open("main.tex", encoding="utf-8").read()
stems = re.findall(r"\\include\{chapters/(ch\d\d_[A-Za-z_]+)\}", main)
stems = [s for s in stems if not s.startswith("ch00")]
for position, stem in enumerate(stems, start=1):
    n = stem[2:4]
    lines = [
        BS + "documentclass[11pt,openany]{book}",
        BS + "input{preamble}",
        BS + "begin{document}",
        BS + "setcounter{chapter}{%d}" % (position - 1),
        BS + "input{chapters/%s}" % stem,
        BS + "section*{Answers to selected exercises}",
        BS + "input{answers/ch%s}" % n,
        BS + "bibliographystyle{plainnat}",
        BS + "bibliography{refs/ch%s}" % n,
        BS + "end{document}",
    ]
    with open("drivers/ch%s.tex" % n, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
print("drivers written for", len(stems), "chapters")
