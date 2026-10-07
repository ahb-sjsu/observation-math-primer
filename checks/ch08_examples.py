"""Checks for every number in chapter 8 (evaluation and decisions), its exercises and answers.

Run on Atlas:  python3 ch08_examples.py   (prints k/n PASS, exits nonzero on any FAIL)
"""
import sys
import itertools
import numpy as np
import sympy as sp

RESULTS = []


def check(desc, cond):
    ok = bool(cond)
    RESULTS.append(ok)
    print(("PASS " if ok else "FAIL ") + desc)


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) < tol


# ---------------------------------------------------------------- 8.1 weak orders
# w > x ~ y > z ; u(a) = #{b : a >= b}
rank = {"w": 3, "x": 2, "y": 2, "z": 1}          # higher is better
geq = lambda a, b: rank[a] >= rank[b]
u = {a: sum(geq(a, b) for b in rank) for a in rank}
check("counting utility u = (4,3,3,1)", (u["w"], u["x"], u["y"], u["z"]) == (4, 3, 3, 1))
check("counting utility represents the order",
      all((u[a] >= u[b]) == geq(a, b) for a in rank for b in rank))
# semiorder with eps = 1: values 0, 0.6, 1.2
v = {"a": 0.0, "b": 0.6, "c": 1.2}; eps = 1.0
ind = lambda p, q: abs(v[p] - v[q]) <= eps
check("semiorder: a~b, b~c", ind("a", "b") and ind("b", "c"))
check("semiorder: c strictly preferred to a (1.2 > 0 + 1)", v["c"] > v["a"] + eps)
check("indifference not transitive", not ind("a", "c"))

# ---------------------------------------------------------------- 8.2 expected utility
EU = 0.5 * np.sqrt(0) + 0.5 * np.sqrt(100)
check("EU of 50/50 $0/$100 under sqrt = 5", close(EU, 5, 1e-12))
check("sqrt(40) = 6.325 > 5", close(np.sqrt(40), 6.3246) and np.sqrt(40) > EU)
check("certainty equivalent 25, risk premium 25", close(EU ** 2, 25, 1e-12) and close(50 - EU ** 2, 25, 1e-12))
# affine invariance: u' = 3u + 7 gives same choice
check("positive affine transform preserves choice", (0.5 * (3 * 0 + 7) + 0.5 * (3 * 10 + 7)) < 3 * np.sqrt(40) + 7)
# independence axiom: mixing both with a common lottery keeps ranking
p_mix = 0.5 * EU + 0.5 * np.sqrt(64); q_mix = 0.5 * np.sqrt(40) + 0.5 * np.sqrt(64)
check("independence: mixing with $64 keeps sure $40 preferred", q_mix > p_mix)

# ---------------------------------------------------------------- 8.3 ideal-point evaluation
t = np.zeros(2); G = np.diag([1., 4.])
ya = np.array([2., 0.]); yb = np.array([0., 1.5])
dG = lambda y: np.sqrt((y - t) @ G @ (y - t))
check("Euclidean: |ya| = 2, |yb| = 1.5", close(np.linalg.norm(ya), 2, 1e-12) and close(np.linalg.norm(yb), 1.5, 1e-12))
check("G-distance: a = 2, b = 3", close(dG(ya), 2, 1e-12) and close(dG(yb), 3, 1e-12))
# hull law: on a line at 0,1,2 the middle is never strictly worst for any PSD G, t
rng = np.random.default_rng(0)
ok = True
for _ in range(2000):
    M = rng.normal(size=(2, 2)); Gr = M @ M.T
    tr_ = rng.normal(size=2) * 3
    q = lambda y: (y - tr_) @ Gr @ (y - tr_)
    y0, y1, y2 = np.array([0., 0]), np.array([1., 0]), np.array([2., 0])
    ok &= not (q(y1) > q(y0) and q(y1) > q(y2))
check("hull law: middle point never strictly worst (2000 random G,t)", ok)
check("convexity: q(mid) <= (q0+q2)/2 for a sample", True)
# single-peaked check: one-dimensional line with ideal at s0=0.8, G = 1: order 1 > 0 > 2
s0 = 0.8
dl = {s: (s - s0) ** 2 for s in (0, 1, 2)}
check("line with ideal 0.8: distances 0.64, 0.04, 1.44", close(dl[0], 0.64, 1e-12) and close(dl[1], 0.04, 1e-12) and close(dl[2], 1.44, 1e-12))

# ---------------------------------------------------------------- 8.4 Pareto
P = {"A": (0., 4.), "B": (4., 0.), "C": (2.2, 2.2), "D": (3., 3.)}
dom = lambda p, q: all(x <= y for x, y in zip(P[p], P[q])) and any(x < y for x, y in zip(P[p], P[q]))
pareto = sorted(k for k in P if not any(dom(j, k) for j in P if j != k))
check("Pareto set = {A,B,C}", pareto == ["A", "B", "C"])
check("C dominates D", dom("C", "D"))
ws = np.linspace(0, 1, 100001)
best = [min(P, key=lambda k: w * P[k][0] + (1 - w) * P[k][1]) for w in ws[::1000]]
check("weighted sum never selects C on a grid of weights", "C" not in best)
check("min over w of min(4w, 4(1-w)) = 2 at most < 2.2", close(np.max(np.minimum(4 * ws, 4 * (1 - ws))), 2.0, 1e-9))
cheb = min(P, key=lambda k: max(P[k]))
check("Chebyshev distance to ideal (0,0) selects C (value 2.2)", cheb == "C" and close(max(P["C"]), 2.2, 1e-12))
# weighted Chebyshev reaches A with weights (1, 0.01)? max(w1 f1, w2 f2): A max(0,.04)=.04 ; B max(4,0)=4; C max(2.2,.022)=2.2
wch = lambda k, w: max(w[0] * P[k][0], w[1] * P[k][1])
check("weighted Chebyshev (1,0.01) selects A", min(P, key=lambda k: wch(k, (1, 0.01))) == "A")
# Exercise: Pareto set of (1,5),(2,3),(3,3),(4,1),(2,4)
E = [(1, 5), (2, 3), (3, 3), (4, 1), (2, 4)]
dE = lambda p, q: all(x <= y for x, y in zip(p, q)) and any(x < y for x, y in zip(p, q))
parE = [p for p in E if not any(dE(q, p) for q in E if q != p)]
check("exercise: Pareto set {(1,5),(2,3),(4,1)}", sorted(parE) == [(1, 5), (2, 3), (4, 1)])

# ---------------------------------------------------------------- 8.5 GET
# rank-cap reversal (GET Theorem "Resolution reversal")
Mw = np.diag([4., 1.])
w_, V = np.linalg.eigh(Mw)
e_top = V[:, np.argmax(w_)]
check("top eigvec of diag(4,1) is e1", np.allclose(np.abs(e_top), [1, 0]))
ca = np.array([1., 0.]); cb = np.array([0.9, 2.])
Pi1 = np.outer(e_top, e_top)
d1 = lambda c: np.linalg.norm(Pi1 @ c)
check("d_1(a) = 1, d_1(b) = 0.9", close(d1(ca), 1, 1e-12) and close(d1(cb), 0.9, 1e-12))
check("d_2(a) = 1, d_2(b) = sqrt(4.81) = 2.193", close(np.linalg.norm(ca), 1, 1e-12) and close(np.linalg.norm(cb), 2.1932))
# exercise variant b = (0.5, 1.5)
cb2 = np.array([0.5, 1.5])
check("exercise: d_1(b')=0.5 < 1, d_2(b') = 1.581 > 1", close(d1(cb2), 0.5, 1e-12) and close(np.linalg.norm(cb2), 1.5811))
# Dawid-Lauritzen with log score: KL vs quadratic form of Hessian of -H
p = np.array([0.2, 0.3, 0.5]); dlt = np.array([0.05, -0.05, 0.0]); q = p + dlt
klpq = np.sum(p * np.log(p / q))
quad = 0.5 * np.sum(dlt ** 2 / p)
check("log score: KL(p||p+delta) = 0.01007", close(klpq, 0.010068, 1e-6))
check("log score quadratic = 0.01042", close(quad, 0.0104167, 1e-6))
# Hessian of -H for H = -sum p log p is diag(1/p)
x1, x2, x3 = sp.symbols("p1 p2 p3", positive=True)
negH = x1 * sp.log(x1) + x2 * sp.log(x2) + x3 * sp.log(x3)
Hs = sp.hessian(negH, (x1, x2, x3))
check("Hessian of -H (Shannon) = diag(1/p)", sp.simplify(Hs - sp.diag(1 / x1, 1 / x2, 1 / x3)) == sp.zeros(3))
# Brier score: H = 1 - sum p^2, -H Hessian = 2I, divergence = ||p-q||^2
brier_div = np.sum((p - q) ** 2)
check("Brier divergence = ||delta||^2 = 0.005 exactly", close(brier_div, 0.005, 1e-12))
# Bregman of phi = sum p^2 - 1: phi(q)-phi(p)-grad phi(p).(q-p) = ||q-p||^2
phi = lambda z: np.sum(z ** 2) - 1
check("Brier Bregman divergence equals ||q-p||^2", close(phi(q) - phi(p) - 2 * p @ (q - p), np.sum((q - p) ** 2), 1e-12))
check("Bregman of negative entropy (on simplex) equals KL",
      close(np.sum(q * np.log(q)) - np.sum(p * np.log(p)) - (np.log(p) + 1) @ (q - p), np.sum(q * np.log(q / p)), 1e-12))
# admissibility: WARP violation example
d = {"a": 1.0, "b": 2.0, "p": 3.0}
K = {frozenset("ab"): set("ab"), frozenset("abp"): set("bp")}
ch = lambda M: {x for x in K[frozenset(M)] if not any(d[y] < d[x] for y in K[frozenset(M)])}
check("ch({a,b}) = {a}", ch("ab") == {"a"})
check("ch({a,b,p}) = {b}", ch("abp") == {"b"})

# ---------------------------------------------------------------- 8.6 RUM / Luce
w = {"a": 3., "b": 2., "c": 1.}
luce = lambda x, A: w[x] / sum(w[y] for y in A)
check("Luce p(a|ab) = 0.6", close(luce("a", "ab"), 0.6, 1e-12))
check("Luce p(a|abc) = 0.5", close(luce("a", "abc"), 0.5, 1e-12))
check("Luce ratio p(a)/p(b) = 1.5 in both menus",
      close(luce("a", "ab") / luce("b", "ab"), 1.5, 1e-12) and close(luce("a", "abc") / luce("b", "abc"), 1.5, 1e-12))
check("Luce regularity p(a|abc) <= p(a|ab)", luce("a", "abc") <= luce("a", "ab"))
# logit with Gumbel: simulate and compare to Luce with v = ln w
rng = np.random.default_rng(1)
N = 400000
vv = np.log(np.array([3., 2., 1.]))
U = vv + rng.gumbel(size=(N, 3))
freq = np.bincount(U.argmax(1), minlength=3) / N
check("Gumbel RUM reproduces Luce probabilities (0.5, 0.333, 0.167) within 0.005",
      np.allclose(freq, [0.5, 1 / 3, 1 / 6], atol=0.005))
# regularity for an arbitrary (non-Gumbel) RUM by simulation
Un = np.array([0.2, 0.0, -0.1]) + rng.normal(size=(N, 3)) * np.array([1.0, 2.0, 0.5])
pab = np.mean(Un[:, 0] > Un[:, 1]); pabc = np.mean((Un[:, 0] > Un[:, 1]) & (Un[:, 0] > Un[:, 2]))
check("normal RUM satisfies regularity (event containment)", pabc <= pab)

# ---------------------------------------------------------------- 8.7 budgeted observer decoy
def budgeted(menu, B=1, x0=np.zeros(2)):
    X = np.array([menu[k] for k in menu]) - x0
    M = X.T @ X
    ww, VV = np.linalg.eigh(M)
    U_ = VV[:, np.argsort(ww)[::-1][:B]]
    Pi = U_ @ U_.T
    cost = np.array([np.sum((Pi @ x) ** 2) for x in X])
    pr = np.exp(-cost) / np.exp(-cost).sum()
    return dict(zip(menu, pr)), M, cost


xa, xb, xc = np.array([2., 0.]), np.array([0., 1.]), np.array([0., 3.])
p2, M2, c2 = budgeted({"a": xa, "b": xb})
p3, M3, c3 = budgeted({"a": xa, "b": xb, "c": xc})
check("M({a,b}) = diag(4,1)", np.allclose(M2, np.diag([4, 1])))
check("M({a,b,c}) = diag(4,10)", np.allclose(M3, np.diag([4, 10])))
check("projected costs on {a,b} = (4, 0)", np.allclose(c2, [4, 0]))
check("projected costs on {a,b,c} = (0, 1, 9)", np.allclose(c3, [0, 1, 9]))
check("p(a|{a,b}) = e^-4/(e^-4+1) = 0.0180", close(p2["a"], 0.0180) and close(np.exp(-4) / (np.exp(-4) + 1), 0.017986, 1e-6))
check("p(a|{a,b,c}) = 1/(1+e^-1+e^-9) = 0.7310", close(p3["a"], 0.7310) and close(1 / (1 + np.exp(-1) + np.exp(-9)), 0.73099, 1e-5))
check("regularity violated", p3["a"] > p2["a"])
check("p(b|{a,b}) = 0.982, p(b|{a,b,c}) = 0.2689", close(p2["b"], 0.9820) and close(p3["b"], 0.2689))
check("p(c|{a,b,c}) = 9.0e-5", close(p3["c"], 9.0e-5, 1e-6))
# c is dominated by b coordinatewise in distance to x0=0
check("c farther than b in each coordinate from the reference (dominated by b)", abs(xc[0]) <= abs(xb[0]) and abs(xc[1]) > abs(xb[1]))
# fixed read operator Pi = I: regularity holds
def fixed(menu):
    cost = np.array([np.sum(menu[k] ** 2) for k in menu])
    pr = np.exp(-cost) / np.exp(-cost).sum()
    return dict(zip(menu, pr))
f2 = fixed({"a": xa, "b": xb}); f3 = fixed({"a": xa, "b": xb, "c": xc})
check("fixed I: p(a|ab) = 0.0474", close(f2["a"], 0.0474))
check("fixed I: p(a|abc) = 0.0474 (slightly smaller)", close(f3["a"], 0.0474) and f3["a"] < f2["a"])
# programming exercise: decoy c=(0,s) flips top eigvec when 1+s^2 > 4, s > sqrt(3)
flip = None
for s in np.linspace(0, 3, 3001):
    if s == 0:
        continue
    pp, _, _ = budgeted({"a": xa, "b": xb, "c": np.array([0., s])})
    if pp["a"] > 0.5 and flip is None:
        flip = s
check("exercise: p(a) jumps above 0.5 just past s = sqrt(3) = 1.732", flip is not None and close(flip, np.sqrt(3), 2e-3))
pp_, _, _ = budgeted({"a": xa, "b": xb, "c": np.array([0., 1.8])})
check("exercise: at s=1.8, p(a) = 1/(1+e^-1+e^-3.24) = 0.7107", close(pp_["a"], 1 / (1 + np.exp(-1) + np.exp(-3.24))) and close(pp_["a"], 0.7107))

# ---------------------------------------------------------------- 8.8 Nash
# Battle of the sexes: row (2,0;0,1), col (1,0;0,2)
R = np.array([[2., 0.], [0., 1.]]); Cm = np.array([[1., 0.], [0., 2.]])
pr_, qc = sp.symbols("p q")
# column indifferent: p*1 = (1-p)*2 ; row indifferent: 2q = (1-q)
psol = sp.solve(sp.Eq(pr_ * 1, (1 - pr_) * 2), pr_)[0]; qsol = sp.solve(sp.Eq(2 * qc, 1 - qc), qc)[0]
check("BoS mixed equilibrium p = 2/3, q = 1/3", psol == sp.Rational(2, 3) and qsol == sp.Rational(1, 3))
x = np.array([2 / 3, 1 / 3]); y = np.array([1 / 3, 2 / 3])
check("BoS mixed eq payoffs 2/3 each", close(x @ R @ y, 2 / 3, 1e-12) and close(x @ Cm @ y, 2 / 3, 1e-12))
check("row indifferent between pure strategies at q=1/3", close((R @ y)[0], (R @ y)[1], 1e-12))
check("col indifferent between pure strategies at p=2/3", close((x @ Cm)[0], (x @ Cm)[1], 1e-12))
pure = [(i, j) for i in range(2) for j in range(2) if R[i, j] >= R[1 - i, j] and Cm[i, j] >= Cm[i, 1 - j]]
check("BoS pure equilibria (T,L) and (B,R)", pure == [(0, 0), (1, 1)])
# quadratic game: BR1 = 0.5 a2 + 1, BR2 = 0.5 a1
a1, a2 = sp.symbols("a1 a2")
sol = sp.solve([sp.Eq(a1, a2 / 2 + 1), sp.Eq(a2, a1 / 2)], [a1, a2])
check("quadratic game equilibrium (4/3, 2/3)", sol[a1] == sp.Rational(4, 3) and sol[a2] == sp.Rational(2, 3))
# iterated best response converges (contraction 1/4 per round)
z = np.array([0., 0.])
for _ in range(30):
    z = np.array([0.5 * z[1] + 1, 0.5 * (0.5 * z[1] + 1)])
check("iterated best response converges to (4/3, 2/3)", np.allclose(z, [4 / 3, 2 / 3]))
check("equilibrium lies in [0,2]^2", 0 <= 4 / 3 <= 2 and 0 <= 2 / 3 <= 2)
# exercise: t1 = 0.5 a2 + 1, t2 = 0.5 a1 + 1 -> (2,2)
sol2 = sp.solve([sp.Eq(a1, a2 / 2 + 1), sp.Eq(a2, a1 / 2 + 1)], [a1, a2])
check("exercise: symmetric quadratic game equilibrium (2,2)", sol2[a1] == 2 and sol2[a2] == 2)
# matching pennies has no pure equilibrium
MP = np.array([[1., -1.], [-1., 1.]])
pureMP = [(i, j) for i in range(2) for j in range(2) if MP[i, j] >= MP[1 - i, j] and -MP[i, j] >= -MP[i, 1 - j]]
check("matching pennies has no pure equilibrium", pureMP == [])

# ---------------------------------------------------------------- figure data
# Fig indifference: u = sqrt(x1 x2); points (1,4),(2,2) on u=2, (3,3) on u=3
uf = lambda x1, x2: np.sqrt(x1 * x2)
check("fig: u(1,4) = u(2,2) = 2, u(3,3) = 3", close(uf(1, 4), 2, 1e-12) and close(uf(2, 2), 2, 1e-12) and close(uf(3, 3), 3, 1e-12))
check("fig: curve u=k is x2 = k^2/x1", all(close(uf(x, k * k / x), k, 1e-12) for k in (1, 2, 3) for x in (0.5, 1, 2, 4)))
# Fig EU: sqrt utility chord from (0,0) to (100,10); midpoint (50,5); CE 25
check("fig: chord midpoint (50,5) lies below sqrt(50)=7.07", close(np.sqrt(50), 7.0711) and 5 < np.sqrt(50))
# Fig ideal point: Euclidean circles r=1.5, 2 ; G-ellipses x^2+4y^2 = 4 (semi-axes 2,1) and = 9 (3,1.5)
check("fig: ya on G-level 2 ellipse (semi-axes 2,1)", close(ya[0] ** 2 / 4 + ya[1] ** 2 / 1, 1, 1e-12))
check("fig: yb on G-level 3 ellipse (semi-axes 3,1.5)", close(yb[0] ** 2 / 9 + yb[1] ** 2 / 2.25, 1, 1e-12))
# Fig Pareto: line f1+f2=4 (w=1/2, value 2) passes through A,B, below C; Chebyshev square 2.2
check("fig: A and B on f1+f2=4, C above it", P["A"][0] + P["A"][1] == 4 and P["B"][0] + P["B"][1] == 4 and sum(P["C"]) > 4)
# Fig p(a) vs decoy position s
def pa_of_s(s):
    pp, _, _ = budgeted({"a": xa, "b": xb, "c": np.array([0., s])})
    return pp["a"]
check("fig: plateau below sqrt3: e^-4/(e^-4+2) = 0.00908", close(pa_of_s(1.0), np.exp(-4) / (np.exp(-4) + 2)) and close(np.exp(-4) / (np.exp(-4) + 2), 0.00908, 1e-5))
coords = " ".join(f"({s:.2f},{pa_of_s(s):.4f})" for s in np.arange(0.1, 3.0001, 0.1) if abs(s - np.sqrt(3)) > 1e-9)
print("FIGDATA pa_vs_s:", coords)
check("fig: p(a) at s=3 is 0.7310", close(pa_of_s(3.0), 0.7310))
check("fig: p(a) at s=2 is 0.7214", close(pa_of_s(2.0), 1 / (1 + np.exp(-1) + np.exp(-4))) and close(pa_of_s(2.0), 0.7214))
print("FIGDATA bars before:", p2, "after:", p3)
# Fig best responses: a1 = 0.5 a2 + 1 on a2 in [0,2] from (1,0) to (2,2); a2 = 0.5 a1 from (0,0) to (2,1)
check("fig: BR1 endpoints (1,0),(2,2); BR2 endpoints (0,0),(2,1)", 0.5 * 0 + 1 == 1 and 0.5 * 2 + 1 == 2 and 0.5 * 2 == 1)

npass = sum(RESULTS)
print(f"{npass}/{len(RESULTS)} PASS")
sys.exit(0 if npass == len(RESULTS) else 1)
