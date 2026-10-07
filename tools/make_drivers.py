"""Write drivers/chNN.tex, a standalone build of one chapter with its answers and references.

Run from the repository root: python tools/make_drivers.py
Build one chapter: latexmk -pdf -interaction=nonstopmode -outdir=build/chNN drivers/chNN.tex
"""
import glob
import os

BS = "\\"
for path in sorted(glob.glob("chapters/ch[0-9][0-9]_*.tex")):
    stem = os.path.basename(path)[:-4]
    n = stem[2:4]
    if n == "00":
        continue
    lines = [
        BS + "documentclass[11pt,openany]{book}",
        BS + "input{preamble}",
        BS + "begin{document}",
        BS + "setcounter{chapter}{%d}" % (int(n) - 1),
        BS + "input{chapters/%s}" % stem,
        BS + "section*{Answers to selected exercises}",
        BS + "input{answers/ch%s}" % n,
        BS + "bibliographystyle{plainnat}",
        BS + "bibliography{refs/ch%s}" % n,
        BS + "end{document}",
    ]
    with open("drivers/ch%s.tex" % n, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
print("drivers written")
