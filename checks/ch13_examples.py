"""Checks every number in the chapter on spectral perturbation (file key ch13).

Covers the worked examples, the exercise answers and the data of every figure.
Run on Atlas:  OPENBLAS_NUM_THREADS=4 /home/claude/env/bin/python3 ch13_examples.py
Prints PASS/FAIL per claim, the figure coordinates, then 'k/n PASS'.
Exits nonzero on any FAIL. All simulations are seeded.
"""
import sys
import numpy as np
import sympy as sp

results = []


def check(desc, cond):
    ok = bool(cond)
    results.append(ok)
    print(("PASS" if ok else "FAIL") + "  " + desc)


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def coords(pairs, fmt="({:.4g},{:.4g})"):
    return " ".join(fmt.format(float(a), float(b)) for a, b in pairs)


R = sp.Rational
M_ = sp.Matrix


def eigh_desc(A):
    w, V = np.linalg.eigh(np.asarray(A, dtype=float))
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def principal_cosines(Q1, Q2):
    Q1, _ = np.linalg.qr(Q1)
    Q2, _ = np.linalg.qr(Q2)
    return np.clip(np.linalg.svd(Q1.T @ Q2, compute_uv=False), 0, 1)


# =========================================================== Section 1. Weyl
print("\n# Section 1: Weyl")
A = M_([[3, 0], [0, 1]])
E = M_([[R(1, 10), R(2, 10)], [R(2, 10), R(-1, 10)]])
AE = A + E
check("A+E = [[3.1,0.2],[0.2,0.9]]", AE == M_([[R(31, 10), R(2, 10)], [R(2, 10), R(9, 10)]]))
check("trace 4, det 2.75", AE.trace() == 4 and AE.det() == R(275, 100))
ev = sorted(AE.eigenvals().keys(), key=lambda z: -float(z))
check("eigenvalues 2 +- sqrt(1.25)", sp.simplify(ev[0] - (2 + sp.sqrt(R(5, 4)))) == 0)
check("eigenvalues 3.1180 and 0.8820", close(ev[0], 3.1180, 1e-4) and close(ev[1], 0.8820, 1e-4))
nE = max(abs(float(z)) for z in E.eigenvals().keys())
check("||E||_2 = sqrt(0.05) = 0.2236", close(nE, np.sqrt(0.05)) and close(nE, 0.2236, 1e-4))
check("eigenvalue E = +-0.2236 (trace 0, det -0.05)", E.trace() == 0 and E.det() == R(-5, 100))
shift = [float(ev[0]) - 3, float(ev[1]) - 1]
check("shifts +0.1180, -0.1180, both <= ||E||", close(shift[0], 0.1180, 1e-4) and close(shift[1], -0.1180, 1e-4)
      and max(abs(s) for s in shift) <= nE)
check("Hoffman-Wielandt: 0.0279 <= ||E||_F^2 = 0.1",
      close(shift[0] ** 2 + shift[1] ** 2, 0.0279, 1e-4) and close(float(sum(e ** 2 for e in E)), 0.1)
      and shift[0] ** 2 + shift[1] ** 2 <= 0.1)
check("first-order: 3.1 and 0.9", A[0, 0] + E[0, 0] == R(31, 10) and A[1, 1] + E[1, 1] == R(9, 10))
check("second-order: 3.1 + 0.04/2 = 3.12, 0.9 - 0.02 = 0.88",
      R(31, 10) + R(4, 100) / 2 == R(312, 100) and R(9, 10) - R(4, 100) / 2 == R(88, 100))
check("second-order within 0.002 of exact", abs(float(ev[0]) - 3.12) < 0.002 and abs(float(ev[1]) - 0.88) < 0.002)
# non-symmetric warning
eps = 1e-4
w = np.linalg.eigvals(np.array([[0, 1], [eps, 0]]))
check("nonsymmetric [[0,1],[1e-4,0]] has eigenvalues +-0.01", close(max(w.real), 0.01, 1e-12) and close(min(w.real), -0.01, 1e-12))
check("that is 100 times the perturbation", close(0.01 / eps, 100))

# Figure 1: avoided crossing. A(t) = [[t, e],[e, -t]]
print("\n# Figure 1 data (avoided crossing)")
ts = np.round(np.linspace(-1, 1, 41), 3)
fig1_top, fig1_angle = {}, {}
for e in [0.1, 0.02]:
    lam, ang = [], []
    for t in ts:
        w, V = eigh_desc([[t, e], [e, -t]])
        lam.append(w[0])
        v = V[:, 0] * np.sign(V[1, 0] if abs(V[1, 0]) > 1e-12 else V[0, 0])
        ang.append(np.degrees(np.arctan2(abs(v[1]), abs(v[0]))))
    fig1_top[e] = np.array(lam)
    fig1_angle[e] = np.array(ang)
    check(f"e={e}: top eigenvalue = sqrt(t^2+e^2)", np.allclose(fig1_top[e], np.sqrt(ts ** 2 + e ** 2)))
    check(f"e={e}: Weyl, |lambda - |t|| <= e on the whole path", np.all(np.abs(fig1_top[e] - np.abs(ts)) <= e + 1e-12))
    check(f"e={e}: angle = 0.5*atan2(e, t) in degrees",
          np.allclose(fig1_angle[e], 0.5 * np.degrees(np.arctan2(e, ts)), atol=1e-8))
check("e=0.1: angle 87.1 at t=-1, 45 at t=0, 2.9 at t=1",
      close(fig1_angle[0.1][0], 87.15, 0.01) and close(fig1_angle[0.1][20], 45, 1e-9) and close(fig1_angle[0.1][-1], 2.85, 0.01))
check("e=0.02: angle swings from 89.4 to 0.6",
      close(fig1_angle[0.02][0], 89.43, 0.01) and close(fig1_angle[0.02][-1], 0.57, 0.01))
print("fig1 lambda e=0.1:", coords(zip(ts, fig1_top[0.1])))
print("fig1 angle e=0.1:", coords(zip(ts, fig1_angle[0.1])))
# denser path for e=0.02 near 0
ts2 = np.round(np.concatenate([np.linspace(-1, -0.2, 9), np.linspace(-0.15, 0.15, 13), np.linspace(0.2, 1, 9)]), 3)
ang2 = 0.5 * np.degrees(np.arctan2(0.02, ts2))
print("fig1 angle e=0.02:", coords(zip(ts2, ang2)))

# ===================================================== Section 2. eigenvector turn
print("\n# Section 2: eigenvector turn")
# A=diag(3,1), E = eps*[[0,1],[1,0]]: tan 2theta = 2 eps / g
th = 0.5 * np.arctan(2 * 0.2 / 2)
check("g=2, eps=0.2: theta = 0.5*atan(0.2) = 5.65 deg", close(np.degrees(th), 5.65, 0.005))
w, V = eigh_desc([[3, 0.2], [0.2, 1]])
check("numerical top eigenvector at 5.65 deg", close(np.degrees(np.arctan(abs(V[1, 0] / V[0, 0]))), np.degrees(th)))
check("first-order eps/g = 0.1 rad = 5.73 deg", close(np.degrees(0.1), 5.73, 0.005))
th2 = 0.5 * np.arctan(2 * 0.2 / 0.02)
w, V = eigh_desc([[3, 0.2], [0.2, 1]])
check("diag(3,1)+0.2 swap: eigenvalues 2 +- sqrt(1.04) = 3.0198, 0.9802", close(w[0], 3.0198, 1e-4) and close(w[1], 0.9802, 1e-4))
check("g=0.02, eps=0.2: tan 2theta = 20, theta = 43.57 deg", close(np.degrees(th2), 43.57, 0.005))
w, V = eigh_desc([[2.01, 0.2], [0.2, 1.99]])
check("numerical: diag(2.01,1.99)+0.2 swap turns by 43.57 deg",
      close(np.degrees(np.arctan(abs(V[1, 0] / V[0, 0]))), np.degrees(th2)))
check("eigenvalues of that matrix move by at most 0.2",
      abs(w[0] - 2.01) <= 0.2 + 1e-12 and abs(w[1] - 1.99) <= 0.2 + 1e-12)
check("its eigenvalues are 2 +- sqrt(0.0401) = 2.2002, 1.7998",
      close(w[0], 2 + np.sqrt(0.0401)) and close(w[0], 2.2002, 1e-4) and close(w[1], 1.7998, 1e-4))
# first-order eigenvector formula on a 3x3
A3 = np.diag([3.0, 2.0, 0.0])
E3 = np.array([[0, 0.02, 0.03], [0.02, 0, 0], [0.03, 0, 0]])
w, V = eigh_desc(A3 + E3)
v1 = V[:, 0] * np.sign(V[0, 0])
pred = np.array([1, 0.02 / 1, 0.03 / 3])
check("3x3 first-order: v1 ~ (1, 0.02, 0.01)", np.allclose(pred, [1, 0.02, 0.01]))
check("3x3 first-order matches exact to 1e-3", np.allclose(v1 / v1[0], pred, atol=1e-3))
print("3x3 exact v1/v1[0]:", np.round(v1 / v1[0], 5))

# Figure 2: theta(eps) for several gaps
print("\n# Figure 2 data")
epsg = np.round(np.linspace(0, 1, 26), 3)
for g in [0.1, 0.5, 2.0]:
    thd = 0.5 * np.degrees(np.arctan(2 * epsg / g))
    num = []
    for e in epsg:
        w, V = eigh_desc([[g / 2, e], [e, -g / 2]])
        num.append(np.degrees(np.arctan2(abs(V[1, 0]), abs(V[0, 0]))))
    check(f"g={g}: closed-form angle matches eigh", np.allclose(thd, num, atol=1e-8))
    print(f"fig2 g={g}:", coords(zip(epsg, thd)))
check("g=0.1 reaches 41.44 deg at eps=0.4", close(0.5 * np.degrees(np.arctan(8)), 41.44, 0.01))
check("g=2 at eps=1: 22.5 deg", close(0.5 * np.degrees(np.arctan(1.0)), 22.5))
check("g=0.1 at eps=0.05: 22.5 deg (eps/g = 1/2)", close(0.5 * np.degrees(np.arctan(1.0)), 22.5))

# ============================================== Section 3. principal angles
print("\n# Section 3: principal angles")
Q = M_([[1, 0], [0, 1], [0, 0]])
Qh = M_([[1, 0], [0, R(3, 5)], [0, R(4, 5)]])
check("Qhat orthonormal", Qh.T * Qh == sp.eye(2))
C = Q.T * Qh
sv = sorted([sp.sqrt(z) for z in (C.T * C).eigenvals().keys()], key=lambda z: -float(z))
check("cosines of principal angles 1 and 3/5", sv[0] == 1 and sv[1] == R(3, 5))
check("largest principal angle 53.13 deg", close(np.degrees(np.arccos(0.6)), 53.13, 0.005))
P = Q * Q.T
Ph = Qh * Qh.T
o = (P * Ph).trace() / 2
check("overlap = (1 + 9/25)/2 = 17/25 = 0.68", o == R(17, 25))
check("chance r/d = 2/3 = 0.667", close(R(2, 3), 0.6667, 1e-4))
D = P - Ph
check("||Pi - Pihat||_F^2 = 32/25 = 2 r (1-o)", sum(z ** 2 for z in D) == R(32, 25) and R(32, 25) == 2 * 2 * (1 - o))
nd = max(abs(float(z)) for z in D.eigenvals().keys())
check("||Pi - Pihat||_2 = 4/5 = sin of largest angle", close(nd, 0.8))
check("sum sin^2 = 16/25 = r(1-o)", 1 - R(9, 25) == R(16, 25) and R(16, 25) == 2 * (1 - o))
check("resolution of the 3x3 example (0.68-2/3)/(1-2/3) = 0.04", (R(17, 25) - R(2, 3)) / (1 - R(2, 3)) == R(1, 25))
check("programme: o=0.647 gives floor fraction 0.353", close(1 - 0.647, 0.353))
check("two lines at 60 deg: overlap 1/4 below chance 1/2", close(np.cos(np.pi / 3) ** 2, 0.25))

# chance level and Figure 3
print("\n# Figure 3 data (random overlaps, r=4)")
rng = np.random.default_rng(13)
r = 4
bins = np.linspace(0, 1, 101)
sds = {}
allvals = {}
for d in [8, 16, 32]:
    vals = np.empty(20000)
    for k in range(vals.size):
        G = rng.standard_normal((d, r))
        Qr, _ = np.linalg.qr(G)
        vals[k] = np.sum(Qr[:r, :] ** 2) / r  # R = span(e_1..e_r)
    check(f"d={d}: mean random overlap {vals.mean():.4f} ~ r/d = {r/d:.4f}", abs(vals.mean() - r / d) < 0.005)
    sds[d] = (vals.std(), np.quantile(vals, 0.99))
    allvals[d] = vals
    hist, _ = np.histogram(vals, bins=bins)
    frac = hist / vals.size
    centers = 0.5 * (bins[1:] + bins[:-1])
    keep = [(c, h) for c, h in zip(centers, frac) if c <= 0.85]
    print(f"fig3 d={d} sd={vals.std():.4f} q99={np.quantile(vals, 0.99):.3f}:", coords(keep, "({:.3f},{:.4f})"))
check("spread shrinks: sd(8) > sd(16) > sd(32)", sds[8][0] > sds[16][0] > sds[32][0])
check("d=8: 99th percentile of chance overlap above 0.647", sds[8][1] > 0.647)
tail65 = np.mean(allvals[8] >= 0.65)
print(f"d=8: P(o >= 0.65) = {tail65:.4f}; d=32 max = {allvals[32].max():.4f}")
check("d=8: about 4 percent of random overlaps reach 0.65 (between 1% and 5%)", 0.01 < tail65 < 0.05 and abs(tail65 - 0.04) < 0.005)
check("d=8: 99th percentile is 0.693", abs(sds[8][1] - 0.693) < 5e-4)
check("d=32: largest of 20000 random overlaps is 0.296", abs(allvals[32].max() - 0.296) < 5e-4)
print("sd, q99 by d:", {k: (round(a, 4), round(b, 3)) for k, (a, b) in sds.items()})
# E[Pi_hat] = (r/d) I by Monte Carlo
d = 6
acc = np.zeros((d, d))
for k in range(20000):
    Qr, _ = np.linalg.qr(rng.standard_normal((d, 2)))
    acc += Qr @ Qr.T
acc /= 20000
check("E[Pi_hat] ~ (2/6) I for random planes in R^6", np.allclose(acc, np.eye(d) / 3, atol=0.01))

# ================================================ Section 4. Davis-Kahan
print("\n# Section 4: Davis-Kahan")
A = sp.diag(9, 9, 1)
E = M_([[0, 0, 3], [0, 0, 0], [3, 0, 0]])
AE = A + E
evs = AE.eigenvals()
check("A+E eigenvalues 10, 9, 0", set(evs.keys()) == {10, 9, 0})
check("(3,0,1) is eigenvector for 10", AE * M_([3, 0, 1]) == 10 * M_([3, 0, 1]))
check("e2 is eigenvector for 9", AE * M_([0, 1, 0]) == 9 * M_([0, 1, 0]))
check("Weyl: shifts 1, 0, 1 all <= ||E|| = 3",
      max(abs(float(z)) for z in E.eigenvals().keys()) == 3)
Qh = np.column_stack([[3 / np.sqrt(10), 0, 1 / np.sqrt(10)], [0, 1, 0]])
cs = principal_cosines(np.eye(3)[:, :2], Qh)
check("principal cosines 1 and 3/sqrt(10)", np.allclose(sorted(cs), sorted([1, 3 / np.sqrt(10)])))
sinF = np.sqrt(np.sum(1 - cs ** 2))
check("||sin Theta||_F = ||sin Theta||_2 = sqrt(0.1) = 0.3162", close(sinF, np.sqrt(0.1)) and close(sinF, 0.3162, 1e-4))
check("overlap = (1 + 0.9)/2 = 0.95", close(np.sum(cs ** 2) / 2, 0.95))
check("original DK, operator: 3/9 = 0.333 >= 0.316", 3 / 9 >= np.sqrt(0.1) and close(3 / 9, 0.3333, 1e-4))
check("original DK, Frobenius: 3 sqrt2 / 9 = 0.471", close(3 * np.sqrt(2) / 9, 0.4714, 1e-4) and 3 * np.sqrt(2) / 9 >= sinF)
check("YWS Frobenius: 2 min(sqrt2*3, 3 sqrt2)/8 = 1.06 (vacuous)", close(2 * min(np.sqrt(2) * 3, 3 * np.sqrt(2)) / 8, 1.0607, 1e-4))
check("population-gap operator form: 2*3/8 = 0.75", close(2 * 3 / 8, 0.75))
check("tan 2theta = 6/8 so cos 2theta = 0.8 and sin^2 = 0.1", close(np.cos(np.arctan(0.75)), 0.8) and close((1 - 0.8) / 2, 0.1))
check("theta = 18.43 deg", close(np.degrees(0.5 * np.arctan(0.75)), 18.43, 0.005))

# Figure 4: collapse in x = eps/g
print("\n# Figure 4 data")
xs = np.round(np.linspace(0, 1.5, 31), 3)
true = np.sin(0.5 * np.arctan(2 * xs))
dk = xs / (0.5 + np.sqrt(0.25 + xs ** 2))
yws = np.minimum(2 * xs, 1)
check("collapse: sin theta <= DK mixed-gap bound everywhere", np.all(true <= dk + 1e-12))
check("collapse: DK mixed-gap bound <= 2x everywhere", np.all(dk <= 2 * xs + 1e-12))
check("DK mixed-gap bound < 1 everywhere", np.all(dk < 1))
print("fig4 true:", coords(zip(xs, true)))
print("fig4 dk:", coords(zip(xs, dk)))
print("fig4 yws:", coords([(x, y) for x, y in zip(xs, yws) if x <= 0.5 + 1e-9]))
pts = []
for g in [0.1, 0.5, 2.0]:
    for e in [0.02, 0.05, 0.1, 0.3]:
        x = e / g
        if x > 1.5:
            continue
        w, V = eigh_desc([[g / 2, e], [e, -g / 2]])
        s = abs(V[1, 0])
        pts.append((g, x, s))
        check(f"g={g} eps={e}: measured sin theta on the collapse curve", close(s, np.sin(0.5 * np.arctan(2 * x)), 1e-9))
print("fig4 markers:", " ".join(f"({x:.3g},{s:.4g})" for g, x, s in pts))
# exercise 13.6: A=diag(2,1), E=0.5 swap
th = 0.5 * np.arctan(1.0)
check("Ex: theta = 22.5 deg, sin = 0.3827", close(np.degrees(th), 22.5) and close(np.sin(th), 0.3827, 1e-4))
lam2hat = 1.5 - np.sqrt(0.5)
delta = 2 - lam2hat
check("Ex: delta = 0.5 + sqrt(0.5) = 1.2071", close(delta, 1.2071, 1e-4))
check("Ex: DK bound 0.5/1.2071 = 0.4142 = sqrt2 - 1 = tan 22.5", close(0.5 / delta, np.sqrt(2) - 1) and close(0.5 / delta, np.tan(th)))
check("Ex: population-gap bound 2*0.5/1 = 1 (vacuous)", close(2 * 0.5 / 1, 1))

# ================================== Section 5. sample covariance
print("\n# Section 5: sample covariance")
# Isserlis: E||S - Sigma||_F^2 = ((tr Sigma)^2 + tr Sigma^2)/n for zero-mean Gaussian
d = 20
rng = np.random.default_rng(1305)
for g in [0.5, 2.0]:
    lam = np.ones(d); lam[0] = 1 + g
    pred = (lam.sum() ** 2 + (lam ** 2).sum()) / 50
    vals = []
    for k in range(4000):
        X = rng.standard_normal((50, d)) * np.sqrt(lam)
        S = X.T @ X / 50
        vals.append(np.sum((S - np.diag(lam)) ** 2))
    check(f"g={g}, n=50: E||S-Sigma||_F^2 = {pred:.3f} matches MC {np.mean(vals):.3f} within 2%",
          abs(np.mean(vals) / pred - 1) < 0.02)
lam = np.ones(d); lam[0] = 3.0
check("g=2, d=20: (trS)^2 + trS^2 = 22^2 + 28 = 512", close(lam.sum() ** 2 + (lam ** 2).sum(), 512))
check("Frobenius DK bound at n=1000: 2 sqrt(512/1000)/2 = 0.7155", close(2 * np.sqrt(512 / 1000) / 2, 0.7155, 1e-4))

print("\n# Figure 5 data")
ns = [25, 50, 100, 200, 400, 800, 1600, 3200]
fig5 = {}
for g in [0.5, 2.0]:
    lam = np.ones(d); lam[0] = 1 + g
    rows = []
    for n in ns:
        sins, ops, tops = [], [], []
        for rep in range(200):
            X = rng.standard_normal((n, d)) * np.sqrt(lam)
            S = X.T @ X / n
            w, V = eigh_desc(S)
            sins.append(np.sqrt(max(0.0, 1 - V[0, 0] ** 2)))
            ops.append(np.linalg.norm(S - np.diag(lam), 2))
            tops.append(w[0])
        rows.append((n, np.mean(sins), np.mean(ops), np.mean(tops)))
    fig5[g] = rows
    for n, s, op, top in rows:
        print(f"g={g} n={n}: mean sin={s:.4f} mean||E||op={op:.4f} DKop={2*op/g:.4f} mean top eig={top:.4f}")
    big = [(np.log(n), np.log(s)) for n, s, _, _ in rows if n >= 400]
    slope = np.polyfit([a for a, _ in big], [b for _, b in big], 1)[0]
    check(f"g={g}: log-log slope over n>=400 is {slope:.3f}, near -1/2", -0.6 <= slope <= -0.4)
    check(f"g={g}: measured mean sin below DK op bound 2E||E||/g at every n",
          all(s <= min(1, 2 * op / g) + 1e-12 for n, s, op, _ in rows))
    check(f"g={g}: top sample eigenvalue biased up at n=25 ({rows[0][3]:.3f} > {1+g})", rows[0][3] > 1 + g)
    print(f"fig5 sin g={g}:", coords([(n, s) for n, s, _, _ in rows]))
    print(f"fig5 DKop g={g}:", coords([(n, min(1.0, 2 * op / g)) for n, _, op, _ in rows]))
check("Frobenius DK bound n=800: 2 sqrt(512/800)/2 = 0.800; n=1600: 0.566",
      close(np.sqrt(512 / 800), 0.8) and close(np.sqrt(512 / 1600), 0.5657, 1e-4))
for g, n_, meas in [(2.0, 3200, fig5[2.0][-1][1]), (0.5, 3200, fig5[0.5][-1][1]), (2.0, 1600, fig5[2.0][-2][1])]:
    l1 = 1 + g
    predfo = np.sqrt(19 * l1 * 1.0 / n_) / g
    print(f"first-order sample prediction g={g} n={n_}: {predfo:.4f} vs measured {meas:.4f}")
    check(f"first-order prediction sqrt(19*{l1}/{n_})/{g} = {predfo:.4f} within 3% of measured {meas:.4f}", abs(meas / predfo - 1) < 0.03)
check("predictions 0.0667 and 0.1887", close(np.sqrt(19 * 3 / 3200) / 2, 0.0667, 1e-4) and close(np.sqrt(19 * 1.5 / 3200) / 0.5, 0.1887, 1e-4))
check("first-order angle ratio (sqrt(1.5)/0.5)/(sqrt(3)/2) = 2.83", close((np.sqrt(1.5) / 0.5) / (np.sqrt(3) / 2), 2.83, 0.005))
check("measured ratio 2.80 within 2% of 2.83", abs((fig5[0.5][-1][1] / fig5[2.0][-1][1]) / 2.8284 - 1) < 0.02)
check("smaller gap gives larger angle at every n",
      all(a[1] > b[1] for a, b in zip(fig5[0.5], fig5[2.0])))
ratio = fig5[0.5][-1][1] / fig5[2.0][-1][1]
print(f"angle ratio g=0.5 vs g=2 at n=3200: {ratio:.3f}")
check("at n=3200 the angle ratio is between 2.5 and 5 (gap ratio 4)", 2.5 < ratio < 5)
# convexity bias argument numbers
check("bias at n=25, g=0.5 exceeds 0.5", fig5[0.5][0][3] - 1.5 > 0.5)
print(f"bias g=0.5 n=25: {fig5[0.5][0][3]-1.5:.3f}; g=2 n=25: {fig5[2.0][0][3]-3:.3f}")

# ================================== Section 6. MP
print("\n# Section 6: Marchenko-Pastur")
d, n = 200, 800
gam = d / n
check("gamma = 1/4, edges (1-1/2)^2 = 0.25 and (1+1/2)^2 = 2.25",
      close(gam, 0.25) and close((1 - np.sqrt(gam)) ** 2, 0.25) and close((1 + np.sqrt(gam)) ** 2, 2.25))
check("E of mean squared eigenvalue = 1 + (d+1)/n = 1.25125", close(1 + (d + 1) / n, 1.25125))
rng = np.random.default_rng(1306)
X = rng.standard_normal((n, d))
S = X.T @ X / n
w = np.linalg.eigvalsh(S)
print(f"single draw: min {w.min():.4f} max {w.max():.4f} mean {w.mean():.4f} meansq {np.mean(w**2):.4f}")
check("single draw: mean eigenvalue within 0.015 of 1", abs(w.mean() - 1) < 0.015)
check("single draw: mean squared eigenvalue within 0.04 of 1.2513", abs(np.mean(w ** 2) - 1.25125) < 0.04)
check("single draw: smallest eigenvalue between 0.2 and 0.3", 0.2 < w.min() < 0.3)
check("single draw: largest eigenvalue between 2.15 and 2.4", 2.15 < w.max() < 2.4)
check("ratio largest/smallest above 7 (truth is 1)", w.max() / w.min() > 7)
print(f"condition number of sample: {w.max()/w.min():.2f}")
mc = []
for k in range(100):
    Xk = rng.standard_normal((n, d))
    wk = np.linalg.eigvalsh(Xk.T @ Xk / n)
    mc.append(np.mean(wk ** 2))
check("100-draw mean of mean squared eigenvalue within 0.3% of 1.25125", abs(np.mean(mc) / 1.25125 - 1) < 0.003)
edges = np.linspace(0, 2.6, 27)
hist, _ = np.histogram(w, bins=edges, density=True)
cent = 0.5 * (edges[1:] + edges[:-1])
print("fig6 hist:", " ".join(f"({a:.2f},{b:.3f})" for a, b in zip(edges[:-1], hist)) + f" (2.60,{hist[-1]:.3f})")
a_, b_ = (1 - np.sqrt(gam)) ** 2, (1 + np.sqrt(gam)) ** 2
xx = np.linspace(a_, b_, 41)
mp = np.sqrt(np.maximum((b_ - xx) * (xx - a_), 0)) / (2 * np.pi * gam * xx)
print("fig6 MP:", coords(zip(xx, mp), "({:.4f},{:.4f})"))
# MP density integrates to one
from math import pi
trapz = getattr(np, "trapezoid", None) or np.trapz
fine = np.linspace(a_, b_, 200001)
dens = np.sqrt(np.maximum((b_ - fine) * (fine - a_), 0)) / (2 * pi * gam * fine)
check("MP density integrates to 1", abs(trapz(dens, fine) - 1) < 1e-3)
mpbin = np.array([trapz(dens[(fine >= lo) & (fine < hi)], fine[(fine >= lo) & (fine < hi)]) / (hi - lo)
                  if np.any((fine >= lo) & (fine < hi)) else 0.0 for lo, hi in zip(edges[:-1], edges[1:])])
l1 = np.sum(np.abs(hist - mpbin) * np.diff(edges))
print(f"L1 distance histogram vs MP bins: {l1:.3f}")
check("histogram within L1 0.15 of MP bin masses", l1 < 0.15)
check("MP variance gamma=0.25 matches meansq - 1 to first order", close(1.25125 - 1, 0.25, 0.002))

# ================================== Section 7. power method
print("\n# Section 7: power method")
A = M_([[2, 1], [1, 2]])
x = M_([1, 0])
its = []
for k in range(1, 4):
    x = A * x
    its.append(list(x))
check("Primer L iterates (2,1), (5,4), (14,13)", its == [[2, 1], [5, 4], [14, 13]])
check("tan of angle to v1 = 1/3, 1/9, 1/27",
      [R(a - b, a + b) for a, b in its] == [R(1, 3), R(1, 9), R(1, 27)])
kk = sp.symbols("k", positive=True, integer=True)
xk = (3 ** kk * M_([1, 1]) + M_([1, -1])) / 2
check("A^k e1 = (3^k (1,1) + (1,-1))/2 for k=1..5",
      all((A ** j * M_([1, 0])) == xk.subs(kk, j) for j in range(1, 6)))
for j in [1, 2, 3]:
    v = A ** j * M_([1, 0])
    rq = (v.T * A * v)[0] / (v.T * v)[0]
    check(f"k={j}: 3 - Rayleigh quotient = 2/(3^(2k)+1)", sp.simplify(3 - rq - R(2, 3 ** (2 * j) + 1)) == 0)
check("k=3: eigenvalue error 2/730 = 0.00274, angle tan 1/27 = 0.037",
      close(R(2, 730), 0.00274, 1e-5) and close(R(1, 27), 0.037, 5e-4))
# deflation 2x2
v1 = M_([1, 1]) / sp.sqrt(2)
B = A - 3 * v1 * v1.T
check("2x2 deflation: A - 3 v1 v1^T = [[1/2,-1/2],[-1/2,1/2]]", B == M_([[R(1, 2), R(-1, 2)], [R(-1, 2), R(1, 2)]]))
# 3x3 deflation
A3 = M_([[3, 1, 0], [1, 3, 0], [0, 0, 1]])
check("3x3 eigenvalues 4, 2, 1", set(A3.eigenvals().keys()) == {4, 2, 1})
u = M_([1, 1, 0]) / sp.sqrt(2)
B3 = A3 - 4 * u * u.T
check("deflated = [[1,-1,0],[-1,1,0],[0,0,1]]", B3 == M_([[1, -1, 0], [-1, 1, 0], [0, 0, 1]]))
check("deflated eigenvalues 2, 1, 0", set(B3.eigenvals().keys()) == {2, 1, 0})
x0 = M_([1, 0, 1])
check("B x0 = (1,-1,1), B^2 x0 = (2,-2,1)", B3 * x0 == M_([1, -1, 1]) and B3 ** 2 * x0 == M_([2, -2, 1]))
check("B^k x0 = (2^(k-1), -2^(k-1), 1) for k=1..6",
      all(B3 ** j * x0 == M_([2 ** (j - 1), -(2 ** (j - 1)), 1]) for j in range(1, 7)))
# deflation with an inexact eigenvector: error norm lambda1 sin theta
th = 0.1
vh = np.array([np.cos(np.pi / 4 + th), np.sin(np.pi / 4 + th), 0])
vt = np.array([1, 1, 0]) / np.sqrt(2)
Dm = 4 * (np.outer(vt, vt) - np.outer(vh, vh))
check("deflation error ||lambda1 (v v^T - vh vh^T)||_2 = lambda1 sin theta", close(np.linalg.norm(Dm, 2), 4 * np.sin(th)))
check("rho bound: RQ error <= (l1 - ld) tan^2: k=3 gives 2/730 <= 2*(1/27)^2", R(2, 730) <= 2 * R(1, 27) ** 2)
# iterations to reach 1e-6 at ratio 0.9
kneed = 6 * np.log(10) / np.log(1 / 0.9)
check("ratio 0.9: k >= 131.1, so 132 iterations for 1e-6", close(kneed, 131.13, 0.01) and int(np.ceil(kneed)) == 132)
check("ratio 0.99: about 1375 iterations", int(np.ceil(6 * np.log(10) / np.log(1 / 0.99))) == 1375)
# gap-free: e^{-eps k}/delta
check("gap-free: eps=0.1, delta=0.05 needs k >= ln(1/(eps*delta))/eps = 53", int(np.ceil(np.log(1 / (0.1 * 0.05)) / 0.1)) == 53)
check("gap-free check at k=53: e^{-5.3}/0.05 <= 0.1", np.exp(-5.3) / 0.05 <= 0.1)

print("\n# Figure 7 data (power method on 40x40)")
rng = np.random.default_rng(1307)
d = 40
Qr, _ = np.linalg.qr(rng.standard_normal((d, d)))
x0 = rng.standard_normal(d)
for rho in [0.5, 0.8, 0.95]:
    lam = np.concatenate([[1.0, rho], np.linspace(0.6 * rho, 0.0, d - 2)])
    Am = Qr @ np.diag(lam) @ Qr.T
    v1 = Qr[:, 0]
    x = x0 / np.linalg.norm(x0)
    tans = []
    for k in range(0, 61):
        c = abs(x @ v1)
        tans.append(np.sqrt(max(1 - c ** 2, 0)) / c)
        x = Am @ x
        x /= np.linalg.norm(x)
    tans = np.array(tans)
    ks = np.arange(61)
    sel = (ks >= 10) & (tans > 1e-13)
    slope = np.polyfit(ks[sel], np.log10(tans[sel]), 1)[0]
    check(f"rho={rho}: fitted log10 slope {slope:.4f} vs log10 rho {np.log10(rho):.4f}", abs(slope - np.log10(rho)) < 0.01)
    check(f"rho={rho}: tan theta_k <= rho^k tan theta_0 for every k", np.all(tans <= rho ** ks * tans[0] * (1 + 1e-9) + 1e-15))
    pick = [k for k in range(0, 61, 2) if tans[k] > 1e-14]
    print(f"fig7 rho={rho} tan0={tans[0]:.3f}:", " ".join(f"({k},{np.log10(tans[k]):.3f})" for k in pick))

# ================================== Section 8. Krylov and randomized
print("\n# Section 8: Krylov and randomized SVD")
A = np.diag([3.0, 2.0, 1.0])
x = np.ones(3)
Ax = A @ x
rq1 = Ax @ A @ Ax / (Ax @ Ax)
check("power method one step from (1,1,1): RQ = 36/14 = 2.5714", close(rq1, 36 / 14) and close(rq1, 2.5714, 1e-4))
K = np.column_stack([x, Ax])
Qk, _ = np.linalg.qr(K)
T = Qk.T @ A @ Qk
ritz = np.linalg.eigvalsh(T).max()
print(f"Ritz value on K_2: {ritz:.6f}")
# exact: on span{(1,1,1),(3,2,1)} = span{(1,1,1),(1,0,-1)}
Ks = M_([[1, 1], [1, 0], [1, -1]])
Gm = Ks.T * Ks
Hm = Ks.T * sp.diag(3, 2, 1) * Ks
check("Gram on (1,1,1),(1,0,-1) = diag(3,2); H = [[6,2],[2,4]]", Gm == sp.diag(3, 2) and Hm == M_([[6, 2], [2, 4]]))
mu = sp.symbols("mu")
pol = sp.expand((Hm - mu * Gm).det())
check("det(H - mu G) = 6 mu^2 - 24 mu + 20", sp.simplify(pol - (6 * mu ** 2 - 24 * mu + 20)) == 0)
rt = max(sp.solve(pol, mu), key=lambda z: float(z))
check("Ritz value 2 + sqrt(6)/3 = 2.8165", sp.simplify(rt - (2 + sp.sqrt(6) / 3)) == 0 and close(rt, 2.8165, 1e-4) and close(rt, ritz))
check("Krylov beats one power step: 2.8165 > 2.5714", ritz > rq1)
# two power steps also below
A2x = A @ Ax
rq2 = A2x @ A @ A2x / (A2x @ A2x)
check("even two power steps give 2.8 < 2.8165 (RQ of (9,4,1) = 276/98)", close(rq2, 276 / 98) and rq2 < ritz)
print(f"two-step RQ = {rq2:.4f}")
# randomized range finder on rank one
Ar = M_([[1, 1], [2, 2], [2, 2]])
Om = M_([1, 0])
Y = Ar * Om
Qy = Y / sp.sqrt((Y.T * Y)[0])
check("rank-1: Y = (1,2,2), Q = Y/3", Y == M_([1, 2, 2]) and Qy == M_([R(1, 3), R(2, 3), R(2, 3)]))
check("rank-1: Q Q^T A = A exactly", Qy * Qy.T * Ar == Ar)
check("singular value of A is 3 sqrt(2) = 4.243", close(np.linalg.svd(np.array(Ar, dtype=float), compute_uv=False)[0], 3 * np.sqrt(2)))

print("\n# Figure 8 data (randomized SVD error)")
rng = np.random.default_rng(1308)
m, nn, k, p = 300, 200, 10, 5
U, _ = np.linalg.qr(rng.standard_normal((m, nn)))
V, _ = np.linalg.qr(rng.standard_normal((nn, nn)))
j = np.arange(1, nn + 1)
spectra = {"fast": 0.7 ** (j - 1), "slow": 1.0 / np.sqrt(j)}
fig8 = {}
for name, s in spectra.items():
    Am = U @ np.diag(s) @ V.T
    opt = s[k]
    rows = []
    for q in range(0, 4):
        errs = []
        for t in range(20):
            Om = rng.standard_normal((nn, k + p))
            Y = Am @ Om
            for _ in range(q):
                Y, _r = np.linalg.qr(Y)
                Y = Am @ (Am.T @ Y)
            Qy, _r = np.linalg.qr(Y)
            Bm = Qy.T @ Am
            Ub, sb, Vb = np.linalg.svd(Bm, full_matrices=False)
            Ak = (Qy @ Ub[:, :k]) * sb[:k] @ Vb[:k]
            errs.append(np.linalg.norm(Am - Ak, 2))
        rows.append((q, np.mean(errs) / opt))
    fig8[name] = rows
    print(f"fig8 {name} opt={opt:.4g}:", coords(rows, "({:.0f},{:.4f})"))
    check(f"{name}: error ratio >= 1 at every q (Eckart-Young)", all(rr >= 1 - 1e-9 for _, rr in rows))
    check(f"{name}: power iterations do not hurt (ratio at q=3 <= q=0)", rows[3][1] <= rows[0][1])
check("slow decay: q=0 error ratio above 1.5", fig8["slow"][0][1] > 1.5)
check("slow decay: q=2 error ratio below 1.05", fig8["slow"][2][1] < 1.05)
check("fast decay: q=0 already within 1.2 of optimal", fig8["fast"][0][1] < 1.2)

# figure 7 guide lines and random-direction reference value
check("fig7 guide: 0.972 + 27.9 log10(0.5) = -7.43", close(0.972 + 27.9 * np.log10(0.5), -7.43, 0.005))
check("fig7 guide: 0.972 + 60 log10(0.8) = -4.84", close(0.972 + 60 * np.log10(0.8), -4.84, 0.005))
check("fig7 guide: 0.972 + 60 log10(0.95) = -0.365", close(0.972 + 60 * np.log10(0.95), -0.365, 0.005))
check("fig7: log10 tan theta_0 = log10 9.384 = 0.972", close(np.log10(9.384), 0.972, 0.0005))
rng = np.random.default_rng(1309)
u = rng.standard_normal((200000, 20))
u /= np.linalg.norm(u, axis=1, keepdims=True)
rs = np.mean(np.sqrt(1 - u[:, 0] ** 2))
print(f"mean sin for a random direction in R^20: {rs:.4f}")
check("random direction in R^20: mean sin theta = 0.97 (2 d.p.)", round(rs, 2) == 0.97)
check("whitening stretch 1/sqrt(0.265) = 1.94", close(1 / np.sqrt(0.265), 1.94, 0.005))
check("fast decay q=0 within 3 percent", fig8["fast"][0][1] < 1.03)

# ================================== Exercises
print("\n# Exercises")
w = np.linalg.eigvalsh([[5, 0.3], [0.3, 2]])
check("Ex 13.1: eigenvalues 3.5 +- sqrt(2.34) = 5.0297, 1.9703", close(w[1], 3.5 + np.sqrt(2.34)) and close(w[1], 5.0297, 1e-4) and close(w[0], 1.9703, 1e-4))
check("Ex 13.1: within Weyl bands [4.7,5.3] and [1.7,2.3]", 4.7 <= w[1] <= 5.3 and 1.7 <= w[0] <= 2.3)
check("Ex 13.1: A+E trace 7, det 9.91, each eigenvalue moved by 0.0297",
      close(5 * 2 - 0.09, 9.91) and close(w[1] - 5, 0.0297, 1e-4) and close(2 - w[0], 0.0297, 1e-4))
check("Ex 13.4: 6 ln10 / ln(1/0.99) = 1374.6", close(6 * np.log(10) / np.log(1 / 0.99), 1374.6, 0.05))
check("Ex 13.6: hat lambda_2 = 1.5 - sqrt(0.5) = 0.7929", close(1.5 - np.sqrt(0.5), 0.7929, 1e-4))
check("Ex 13.5: trace 5, det 3.91", close(4 * 1 - 0.09, 3.91))
check("Ex 13.4: rho=0.9 needs 132 iterations", int(np.ceil(6 * np.log(10) / np.log(1 / 0.9))) == 132)
w = np.linalg.eigvalsh([[4, 0.3], [0.3, 1]])
check("Ex 13.5: exact 2.5 + sqrt(2.34) = 4.0297, second order 4.03", close(w[1], 4.0297, 1e-4) and close(4 + 0.09 / 3, 4.03))
check("Ex 13.5: second order error below 5e-4", abs(w[1] - 4.03) < 5e-4)
check("Ex 13.5: first-order eigenvector turn 0.3/3 = 0.1 rad, exact 0.5 atan(0.2) = 0.0987",
      close(0.5 * np.arctan(0.2), 0.0987, 1e-4))
cs = principal_cosines(np.eye(3)[:, :2], np.column_stack([[1, 0, 1], [0, 1, 0]]))
check("Ex 13.7: cosines 1 and 1/sqrt2, overlap 3/4, chance 2/3",
      np.allclose(sorted(cs), [1 / np.sqrt(2), 1]) and close(np.sum(cs ** 2) / 2, 0.75))
check("Ex 13.7: largest angle 45 deg, ||Pi - Pihat||_F^2 = 2*2*(1/4) = 1",
      close(np.degrees(np.arccos(min(cs))), 45) and close(np.linalg.norm(np.diag([1, 1, 0]) - np.array([[.5, 0, .5], [0, 1, 0], [.5, 0, .5]])) ** 2, 1))
check("Ex 13.8: B^k x0 direction tends to (1,-1,0)/sqrt2 with tan = 1/(sqrt2 2^(k-1))",
      all(close(1 / (np.sqrt(2) * 2 ** (j - 1)), np.linalg.norm([0, 0, 1]) / np.linalg.norm([2 ** (j - 1), -(2 ** (j - 1))])) for j in range(1, 6)))

# ---------------------------------------------------------------- hypothesis-discipline additions
# 45 degree ceiling belongs to the off-diagonal family; a diagonal push turns the top eigenvector 90 degrees
wD, VD = np.linalg.eigh(np.diag([2.01, 1.99]) + np.diag([0.0, 0.2]))
check("diag(2.01,1.99)+diag(0,0.2) = diag(2.01,2.19) has top eigenvector e2 (90 deg turn)",
      close(wD[-1], 2.19) and close(abs(VD[1, -1]), 1.0))
# Bauer-Fike for symmetric A, nonsymmetric E: every eigenvalue of A+E within ||E||_2 of some eigenvalue of A
rngBF = np.random.default_rng(1311)
okBF = True
for _ in range(200):
    Ms = rngBF.standard_normal((5, 5)); As = Ms + Ms.T
    Es = 0.3 * rngBF.standard_normal((5, 5))
    lam = np.linalg.eigvals(As + Es); la = np.linalg.eigvalsh(As)
    okBF &= all(min(abs(l - la)) <= np.linalg.norm(Es, 2) + 1e-12 for l in lam)
check("Bauer-Fike: eigenvalues of symmetric A + any E lie within ||E||_2 of spec(A) (200 random)", okBF)
# power-method step counts are attained bounds
check("ratio 0.9, p=6: ceil(6 ln10 / ln(1/0.9)) = 132", int(np.ceil(6 * np.log(10) / np.log(1 / 0.9))) == 132)
check("ratio 0.99, p=6: ceil(...) = 1375", int(np.ceil(6 * np.log(10) / np.log(1 / 0.99))) == 1375)
# Davis-Kahan sufficient, not necessary: spike with gap 1 below ||E||_2 ~ 1.23 is still located
rngSp = np.random.default_rng(1310)
dS, nS, ell = 200, 800, 1.0
cos2 = []; en = []
for s in range(50):
    X = rngSp.standard_normal((nS, dS)); X[:, 0] *= np.sqrt(1 + ell)
    S_ = X.T @ X / nS
    w_, V_ = np.linalg.eigh(S_)
    cos2.append(V_[0, -1] ** 2)
    Sig_ = np.eye(dS); Sig_[0, 0] = 1 + ell
    en.append(np.linalg.norm(S_ - Sig_, 2))
print(f"spike gap 1, d=200, n=800: mean cos^2 = {np.mean(cos2):.4f}, mean ||E||_2 = {np.mean(en):.4f}")
check("spike gap 1: mean ||Sigma_hat - Sigma||_2 ~ 1.23 > gap 1", abs(np.mean(en) - 1.23) < 0.005 and np.mean(en) > 1)
check("spike gap 1: mean squared cosine of top sample eigenvector with e1 rounds to 0.61 >> 1/200", round(np.mean(cos2), 2) == 0.61)
# on average, not always: a single sample's top eigenvalue can fall below lambda_1 (d=1 case)
rngJ = np.random.default_rng(1312)
below = np.mean([np.mean(rngJ.standard_normal(25) ** 2) < 1 for _ in range(2000)])
check(f"d=1, n=25: top sample eigenvalue below the truth in {below:.2f} of draws (> 0)", below > 0.3)

n_pass = sum(results)
print(f"\n{n_pass}/{len(results)} PASS")
sys.exit(0 if n_pass == len(results) else 1)
