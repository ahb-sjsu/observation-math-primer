"""Replace hard-coded "Exercise N.k" answer headings with \\ref to the exercise label.

Printed chapter numbers follow main.tex and change when chapters are added, so an
answer heading must reference the exercise's label, never a literal number. This
maps "Exercise N.k" in answers/chNN.tex to the k-th \\label{exr:...} in the
chapter file (exercise order), and prints each mapping for review.

Run from the repository root: python tools/fix_answer_refs.py
"""
import glob
import io
import re

BS = "\\"
for ans in sorted(glob.glob("answers/ch[0-9][0-9].tex")):
    key = ans[-8:-4]  # chNN
    text = io.open(ans, encoding="utf-8").read()
    if not re.search(r"\\paragraph\{Exercise \d+\.\d+\}", text):
        continue
    chapter = glob.glob("chapters/%s_*.tex" % key)[0]
    labels = re.findall(r"\\begin\{exercise\}(?:\[[^\]]*\])?\s*\\label\{(exr:[^}]+)\}",
                        io.open(chapter, encoding="utf-8").read())

    def repl(m):
        k = int(m.group(2))
        if k > len(labels):
            raise SystemExit("%s: Exercise %s.%d has no label" % (ans, m.group(1), k))
        print("%s  Exercise %s.%d -> %s" % (ans, m.group(1), k, labels[k - 1]))
        return BS + "paragraph{Exercise~" + BS + "ref{" + labels[k - 1] + "}}"

    text = re.sub(r"\\paragraph\{Exercise (\d+)\.(\d+)\}", repl, text)
    io.open(ans, "w", encoding="utf-8", newline="\n").write(text)
