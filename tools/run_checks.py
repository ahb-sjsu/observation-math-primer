"""Run every chapter's numerical checks on Atlas and summarize.

The laptop does not run compute. This uploads checks/chNN_examples.py to
~/omp-checks on Atlas, runs each with the shared Python environment, and prints
each script's final "k/n PASS" line.

Usage (from the repository root, Git Bash):
    MSYS_NO_PATHCONV=1 python tools/run_checks.py <path-to-atlas-helper.py>
"""
import glob
import os
import subprocess
import sys

helper = sys.argv[1]
scripts = sorted(glob.glob("checks/ch[0-9][0-9]_examples.py"))
# Helpers some scripts import (e.g. ch07_extra.py), and the chapter sources that
# some scripts read to compare pasted figure coordinates against computed ones.
extras = sorted(set(glob.glob("checks/*.py")) - set(scripts)) + sorted(glob.glob("chapters/ch[0-9][0-9]_*.tex"))
subprocess.run([sys.executable, helper, "run", "mkdir -p ~/omp-checks"], check=True)
args = []
for s in scripts + extras:
    args += [os.path.abspath(s).replace("\\", "/"), "/home/claude/omp-checks/" + os.path.basename(s)]
subprocess.run([sys.executable, helper, "put"] + args, check=True, stdout=subprocess.DEVNULL)
cmd = "cd ~/omp-checks && for f in ch[0-9][0-9]_examples.py; do printf '%s ' $f; " \
      "OPENBLAS_NUM_THREADS=4 /home/claude/env/bin/python3 $f > ${f%.py}.log 2>&1; " \
      "echo \"exit=$? $(tail -1 ${f%.py}.log)\"; grep FAIL ${f%.py}.log | head -3; done"
subprocess.run([sys.executable, helper, "run", cmd], check=True)
