"""Checks for every number in chapter 10 (logic and machine-checked proof).

Run on Atlas:  python3 ch10_examples.py
Prints PASS/FAIL per claim, exits nonzero on any FAIL, and prints k/n PASS.
The Lean snippets are checked separately (checks/lean/*.lean, Lean v4.32.2).
"""
import itertools
import math
import sys

import sympy as sp

results = []


def check(name, ok, detail=""):
    ok = bool(ok)
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + ((" | " + str(detail)) if detail else ""))


def imp(a, b):
    return (not a) or b


# ---------------------------------------------------------------------------
# Section 10.1  Truth tables, validity
def valid(premises, conclusion, nvars):
    bad = []
    for row in itertools.product([True, False], repeat=nvars):
        if all(pr(*row) for pr in premises) and not conclusion(*row):
            bad.append(row)
    return bad


check("modus ponens valid", valid([lambda p, q: p, lambda p, q: imp(p, q)], lambda p, q: q, 2) == [])
check("modus tollens valid", valid([lambda p, q: not q, lambda p, q: imp(p, q)], lambda p, q: not p, 2) == [])
bad = valid([lambda p, q: q, lambda p, q: imp(p, q)], lambda p, q: p, 2)
check("affirming the consequent invalid, only row p=F,q=T", bad == [(False, True)], bad)
bad = valid([lambda p, q: not p, lambda p, q: imp(p, q)], lambda p, q: not q, 2)
check("denying the antecedent invalid, only row p=F,q=T", bad == [(False, True)], bad)
check("hypothetical syllogism valid (8 rows)",
      valid([lambda p, q, r: imp(p, q), lambda p, q, r: imp(q, r)], lambda p, q, r: imp(p, r), 3) == [])
check("3 variables give 8 rows, 10 give 1024", 2 ** 3 == 8 and 2 ** 10 == 1024)
# p -> q equivalent to (not q) -> (not p), not to q -> p
rows = list(itertools.product([True, False], repeat=2))
check("contrapositive equivalent on all 4 rows", all(imp(p, q) == imp(not q, not p) for p, q in rows))
check("converse differs on 2 of 4 rows", sum(imp(p, q) != imp(q, p) for p, q in rows) == 2)
# implication true on 3 of 4 rows
check("p -> q true on 3 of 4 rows", sum(imp(p, q) for p, q in rows) == 3)
# De Morgan
check("De Morgan both forms", all((not (p and q)) == ((not p) or (not q)) and
                                  (not (p or q)) == ((not p) and (not q)) for p, q in rows))
# excluded middle tautology
check("p or not p is a tautology", all(p or (not p) for p in (True, False)))

# ---------------------------------------------------------------------------
# Section 10.2  Quantifiers on a finite domain
D = [1, 2, 3]
check("forall x exists y, x != y : true", all(any(x != y for y in D) for x in D))
check("exists y forall x, x != y : false", not any(all(x != y for x in D) for y in D))
check("forall x exists y, x <= y : true and exists y forall x x <= y : true (y=3)",
      all(any(x <= y for y in D) for x in D) and any(all(x <= y for x in D) for y in D))
check("not forall x, x even  ==  exists x, x odd (on D)",
      (not all(x % 2 == 0 for x in D)) == any(x % 2 == 1 for x in D))
check("counterexample n=1 refutes n*n > n", not (1 * 1 > 1))
check("n*n > n holds for n = 2..100", all(n * n > n for n in range(2, 101)))

# ---------------------------------------------------------------------------
# Section 10.3  Proof patterns
n = sp.symbols("n", integer=True, nonnegative=True)
k = sp.symbols("k", integer=True)
check("sum_{k=1}^{n} k = n(n+1)/2 (sympy)", sp.simplify(sp.summation(k, (k, 1, n)) - n * (n + 1) / 2) == 0)
check("sum to 10 is 55", sum(range(1, 11)) == 55)
check("induction step algebra n(n+1) + 2(n+1) = (n+1)(n+2)",
      sp.expand(n * (n + 1) + 2 * (n + 1) - (n + 1) * (n + 2)) == 0)
check("sqrt 2 irrational (sympy)", sp.sqrt(2).is_rational is False)
check("odd squared is odd: (2m+1)^2 = 2(2m^2+2m)+1",
      sp.expand((2 * k + 1) ** 2 - (2 * (2 * k ** 2 + 2 * k) + 1)) == 0)
check("n odd -> n+1 even for n < 1000", all((m + 1) % 2 == 0 for m in range(1, 1000, 2)))


# ---------------------------------------------------------------------------
# Section 10.4  Kripke model of the worked example
def box(R, worlds, p, w):
    return all(p[v] for v in worlds if (w, v) in R)


def dia(R, worlds, p, w):
    return any(p[v] for v in worlds if (w, v) in R)


W = [1, 2, 3]
R = {(1, 2), (1, 3), (2, 2)}
P = {1: False, 2: True, 3: False}
tbl = {w: (box(R, W, P, w), dia(R, W, P, w)) for w in W}
print("box/dia table", tbl)
check("w1: box p false, dia p true", tbl[1] == (False, True))
check("w2: box p true, dia p true", tbl[2] == (True, True))
check("w3 (dead end): box p true vacuously, dia p false", tbl[3] == (True, False))
check("T fails at w3: box p true, p false", tbl[3][0] and not P[3])
check("duality dia p = not box not p at every world",
      all(dia(R, W, P, w) == (not box(R, W, {v: not P[v] for v in W}, w)) for w in W))


# ---------------------------------------------------------------------------
# Section 10.5  Frame conditions and their axioms
def frames(nw):
    pairs = [(a, b) for a in range(nw) for b in range(nw)]
    for bits in itertools.product([0, 1], repeat=len(pairs)):
        yield frozenset(p for p, x in zip(pairs, bits) if x)


def props(R, nw):
    refl = all((a, a) in R for a in range(nw))
    sym = all((b, a) in R for (a, b) in R)
    trans = all((a, d) in R for (a, b) in R for (c, d) in R if b == c)
    eucl = all((b, c) in R for (a, b) in R for (a2, c) in R if a == a2)
    return refl, sym, trans, eucl


cnt = dict(refl=0, sym=0, trans=0, equiv=0, total=0)
for Rf in frames(2):
    rf, sy, tr, eu = props(Rf, 2)
    cnt["total"] += 1
    cnt["refl"] += rf
    cnt["sym"] += sy
    cnt["trans"] += tr
    cnt["equiv"] += rf and sy and tr
print("2-world frame counts", cnt)
check("2 worlds: 16 relations", cnt["total"] == 16)
check("2 worlds: 4 reflexive", cnt["refl"] == 4)
check("2 worlds: 8 symmetric", cnt["sym"] == 8)
check("2 worlds: 13 transitive", cnt["trans"] == 13)
check("2 worlds: 2 equivalence relations", cnt["equiv"] == 2)


def axiom_valid_on_frame(Rf, nw, ax):
    worlds = list(range(nw))
    for vals in itertools.product([False, True], repeat=nw):
        p = dict(zip(worlds, vals))
        bp = {w: box(Rf, worlds, p, w) for w in worlds}
        dp = {w: dia(Rf, worlds, p, w) for w in worlds}
        for w in worlds:
            if ax == "T" and not imp(bp[w], p[w]):
                return False
            if ax == "B" and not imp(p[w], box(Rf, worlds, dp, w)):
                return False
            if ax == "4" and not imp(bp[w], box(Rf, worlds, bp, w)):
                return False
            if ax == "5" and not imp(dp[w], box(Rf, worlds, dp, w)):
                return False
    return True


ok = {"T": True, "B": True, "4": True, "5": True}
nframes = 0
for nw in (1, 2, 3):
    for Rf in frames(nw):
        nframes += 1
        rf, sy, tr, eu = props(Rf, nw)
        ok["T"] &= axiom_valid_on_frame(Rf, nw, "T") == rf
        ok["B"] &= axiom_valid_on_frame(Rf, nw, "B") == sy
        ok["4"] &= axiom_valid_on_frame(Rf, nw, "4") == tr
        ok["5"] &= axiom_valid_on_frame(Rf, nw, "5") == eu
check("frames with 1-3 worlds: 2+16+512 = 530", nframes == 530)
for ax, cond in (("T", "reflexive"), ("B", "symmetric"), ("4", "transitive"), ("5", "Euclidean")):
    check(f"axiom {ax} valid on a frame iff frame is {cond} (all 530 frames)", ok[ax])
# preorders on 3 points
pre3 = sum(1 for Rf in frames(3) if props(Rf, 3)[0] and props(Rf, 3)[2])
check("29 reflexive transitive relations on 3 worlds", pre3 == 29, pre3)
sym3 = sum(1 for Rf in frames(3) if props(Rf, 3)[1])
check("64 symmetric relations on 3 worlds", sym3 == 64, sym3)
eq3 = sum(1 for Rf in frames(3) if all(props(Rf, 3)[:3]))
check("5 equivalence relations on 3 worlds (Bell number)", eq3 == 5, eq3)
# reflexive + Euclidean = equivalence on all 530 frames
re_eq = all((props(Rf, nw)[0] and props(Rf, nw)[3]) == all(props(Rf, nw)[:3])
            for nw in (1, 2, 3) for Rf in frames(nw))
check("reflexive and Euclidean iff equivalence (all 530 frames)", re_eq)


# ---------------------------------------------------------------------------
# Section 10.6  The modal ontological argument
def prem2(Rf, worlds, G, w):
    bG = {u: box(Rf, worlds, G, u) for u in worlds}
    return all(imp(G[v], bG[v]) for v in worlds if (w, v) in Rf)


c = dict(sym_cases=0, sym_fail=0, s4_cm=0, refl_cases=0, rev_fail=0, joint=0)
for nw in (1, 2, 3):
    worlds = list(range(nw))
    for Rf in frames(nw):
        rf, sy, tr, eu = props(Rf, nw)
        for vals in itertools.product([False, True], repeat=nw):
            G = dict(zip(worlds, vals))
            nG = {u: not G[u] for u in worlds}
            for w in worlds:
                p1, p2 = dia(Rf, worlds, G, w), prem2(Rf, worlds, G, w)
                if sy:
                    c["sym_cases"] += 1
                    c["sym_fail"] += p1 and p2 and not G[w]
                if rf:
                    c["refl_cases"] += 1
                    c["rev_fail"] += dia(Rf, worlds, nG, w) and p2 and G[w]
                if rf and tr and not sy and p1 and p2 and not G[w]:
                    c["s4_cm"] += 1
                if rf and sy and p1 and dia(Rf, worlds, nG, w) and p2:
                    c["joint"] += 1
print("ontological counts", c)
check("argument holds in all 1604 symmetric frame/valuation/world cases", c["sym_cases"] == 1604 and c["sym_fail"] == 0)
check("reverse argument holds in all 1570 reflexive cases", c["refl_cases"] == 1570 and c["rev_fail"] == 0)
check("68 S4 countermodel cases on 1-3 worlds", c["s4_cm"] == 68, c["s4_cm"])
check("rival premises never jointly hold on reflexive symmetric frames", c["joint"] == 0)
# the two-world countermodel used in the text: worlds 0=false, 1=true
R4 = {(0, 0), (1, 1), (0, 1)}
Gm = {0: False, 1: True}
pr = props(R4, 2)
check("R4 reflexive, transitive, not symmetric", pr[0] and pr[2] and not pr[1])
check("R4: dia G and premise 2 hold at world 0, G fails there",
      dia(R4, [0, 1], Gm, 0) and prem2(R4, [0, 1], Gm, 0) and not Gm[0])
pB = {0: True, 1: False}
dpB = {u: dia(R4, [0, 1], pB, u) for u in (0, 1)}
check("R4: axiom B fails at world 0 with p true only at world 0",
      pB[0] and not box(R4, [0, 1], dpB, 0))
# adding the missing edge (1,0) makes the frame symmetric and premise 2 fails at 0
R5 = R4 | {(1, 0)}
check("symmetric closure: premise 2 fails at world 0 (G at 1 but not box G at 1)",
      not prem2(R5, [0, 1], Gm, 0))

# ---------------------------------------------------------------------------
# Exercise answers
check("Ex 10.1: (p->q) and (q->p) true on 2 of 4 rows (p<->q)",
      sum(imp(p, q) and imp(q, p) for p, q in rows) == 2)
check("Ex 10.2: on D, forall x exists y x<y false, its negation exists x forall y x>=y true (x=3)",
      (not all(any(x < y for y in D) for x in D)) and any(all(x >= y for y in D) for x in D))
R1 = set()
check("Ex 10.3: one world, no successor, p false: box p true, p false",
      box(R1, [0], {0: False}, 0) and not False)
check("Ex 10.5: 1+3+...+(2n-1) = n^2 (sympy)",
      sp.simplify(sp.summation(2 * k - 1, (k, 1, n)) - n ** 2) == 0)
check("Ex 10.5: n = 1..50 numerically", all(sum(range(1, 2 * m, 2)) == m * m for m in range(1, 51)))
Rc = {(1, 2), (2, 3)}
Wc = [1, 2, 3]
pc = {1: False, 2: True, 3: False}
bpc = {w: box(Rc, Wc, pc, w) for w in Wc}
check("Ex 10.6: box p true at w1, box box p false at w1", bpc[1] and not box(Rc, Wc, bpc, 1))
check("Ex 10.6: frame not transitive", not props({(a - 1, b - 1) for a, b in Rc}, 3)[2])
refl3 = sum(1 for Rf in frames(3) if props(Rf, 3)[0])
check("Ex 10.9: 64 reflexive relations on 3 worlds", refl3 == 64)
check("Ex 10.9: 2^9 = 512 relations, 2^6 symmetric", 2 ** 9 == 512 and 2 ** 6 == 64)

print(f"\n{sum(results)}/{len(results)} PASS")
sys.exit(0 if all(results) else 1)
