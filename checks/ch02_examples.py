"""Checks every number in chapter 2 (random vectors and the Gaussian), its exercise
answers and figures.

Run on Atlas:  /home/claude/env/bin/python3 ch02_examples.py
Prints PASS/FAIL per claim, FIGDATA lines for figure coordinates, then 'k/n PASS';
exits nonzero on any FAIL.
"""
import sys
import numpy as np
import sympy as sp

results = []


def check(desc, cond):
    ok = bool(cond)
    results.append(ok)
    print(("PASS" if ok else "FAIL") + "  " + desc)


def figdata(name, text):
    print(f"FIGDATA {name}|{text}")


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


R = sp.Rational
M_ = sp.Matrix
rng = np.random.default_rng(2026)


def ell(S):
    w, V = np.linalg.eigh(np.array(S, dtype=float))
    v = V[:, 1]
    if v[0] < 0:
        v = -v
    return np.sqrt(w[1]), np.sqrt(w[0]), np.degrees(np.arctan2(v[1], v[0]))


# ---------------------------------------------------------------- Example 2.1 (linear map)
mu = M_([1, 2]); S = M_([[2, 1], [1, 3]]); A = M_([[1, 1], [1, -1]])
check("A mu = (3,-1)", A * mu == M_([3, -1]))
check("A Sigma = [[3,4],[1,-2]]", A * S == M_([[3, 4], [1, -2]]))
check("A Sigma A^T = [[7,-1],[-1,3]]", A * S * A.T == M_([[7, -1], [-1, 3]]))
check("Var(x1+x2) = 7, Var(x1-x2) = 3, Cov = -1", 2 + 3 + 2 == 7 and 2 + 3 - 2 == 3 and 2 - 3 == -1)
# Monte Carlo of the linear-map rule (Exercise 2.10)
L = np.linalg.cholesky(np.array(S, dtype=float))
X = np.array([1., 2.]) + rng.standard_normal((100000, 2)) @ L.T
Y = X @ np.array(A, dtype=float).T
check("MC: sample Cov(Ax) near [[7,-1],[-1,3]]", np.max(np.abs(np.cov(Y.T) - np.array([[7., -1.], [-1., 3.]]))) < 0.1)
# trace chain tr(G J S J^T) = tr(J^T G J S)
J = rng.standard_normal((3, 2)); Gm = rng.standard_normal((3, 3)); Gm = Gm @ Gm.T; Sd = np.array([[2., 1.], [1., 3.]])
check("tr(G J S J^T) = tr(J^T G J S)", close(np.trace(Gm @ J @ Sd @ J.T), np.trace(J.T @ Gm @ J @ Sd), 1e-9))
# figure: ellipses before and after the map
a1, b1, t1 = ell(np.array(S, dtype=float)); a2, b2, t2 = ell(np.array(A * S * A.T, dtype=float))
check("fig map: Sigma axes 1.902, 1.176 at 58.283 deg", close(a1, 1.902, 5e-4) and close(b1, 1.176, 5e-4) and close(t1, 58.283, 1e-3))
check("fig map: A Sigma A^T axes 2.690, 1.663 at -13.283 deg", close(a2, 2.690, 5e-4) and close(b2, 1.663, 5e-4) and close(t2, -13.283, 1e-3))
figdata("ch02mapS", f"{a1:.3f}/{b1:.3f}/{t1:.3f}")
figdata("ch02mapASA", f"{a2:.3f}/{b2:.3f}/{t2:.3f}")

# ---------------------------------------------------------------- Example 2.2 (density, ellipses)
Sg = M_([[5, 2], [2, 2]])
check("det 6, inverse (1/6)[[2,-2],[-2,5]]", Sg.det() == 6 and Sg.inv() == M_([[2, -2], [-2, 5]]) / 6)
check("density at mean 1/(2 pi sqrt6) ~ 0.06497", close(1 / (2 * np.pi * np.sqrt(6)), 0.06497, 5e-6))
check("Mahalanobis at (1,1) = 1/2", (M_([1, 1]).T * Sg.inv() * M_([1, 1]))[0] == R(1, 2))
check("density at (1,1) ~ 0.0506", close(np.exp(-0.25) / (2 * np.pi * np.sqrt(6)), 0.0506, 5e-5))
check("P(r^2 <= 1/2) = 1 - e^-1/4 ~ 0.2212", close(1 - np.exp(-0.25), 0.2212, 5e-5))
c95 = -2 * np.log(0.05)
check("c95 = -2 log 0.05 ~ 5.991", close(c95, 5.991, 5e-4))
check("95% semi-axes 5.996 and 2.448", close(np.sqrt(6 * c95), 5.996, 5e-4) and close(np.sqrt(c95), 2.448, 5e-4))
from scipy.stats import chi2  # noqa: E402
check("chi2(2) cdf matches 1 - e^{-c/2}", close(chi2.cdf(0.5, 2), 1 - np.exp(-0.25), 1e-12) and close(chi2.cdf(c95, 2), 0.95, 1e-12))
v2 = M_([1, -2]) / sp.sqrt(5) * sp.sqrt(2)
check("point sqrt2 v2 has Euclidean length sqrt2 and Mahalanobis 2", sp.simplify((v2.T * v2)[0]) == 2 and sp.simplify((v2.T * Sg.inv() * v2)[0]) == 2)
aG, bG, tG = ell(np.array(Sg, dtype=float))
check("fig gauss: axes sqrt6 = 2.449, 1 at 26.565", close(aG, 2.449, 5e-4) and close(bG, 1) and close(tG, 26.565, 1e-3))
# marginal densities N(0,5) and N(0,2) for the figure, scaled by 12 for display
ts = np.arange(-7, 7.01, 0.5)
m1 = 12 * np.exp(-ts ** 2 / 10) / np.sqrt(2 * np.pi * 5)
ts2 = np.arange(-4.5, 4.51, 0.5)
m2 = 12 * np.exp(-ts2 ** 2 / 4) / np.sqrt(2 * np.pi * 2)
check("fig gauss: marginal peak heights 12/sqrt(10 pi)=2.141, 12/sqrt(4 pi)=3.385", close(m1.max(), 2.141, 5e-4) and close(m2.max(), 3.385, 5e-4))
figdata("ch02marg1", " ".join(f"({t:.1f},{-8.5 + v:.3f})" for t, v in zip(ts, m1)))
figdata("ch02marg2", " ".join(f"({-11.0 + v:.3f},{t:.1f})" for t, v in zip(ts2, m2)))

# ---------------------------------------------------------------- Example 2.3 (conditioning)
check("scalar: K = 1, cond var 4 - 2*2/2 = 2", R(2, 2) == 1 and 4 - R(4, 2) == 2)
Sj = M_([[4, 2, 2], [2, 2, 1], [2, 1, 3]])
Syy = Sj[1:, 1:]; Sxy = Sj[:1, 1:]
check("Syy^-1 = (1/5)[[3,-1],[-1,2]]", Syy.inv() == M_([[3, -1], [-1, 2]]) / 5)
K = Sxy * Syy.inv()
check("K = (0.8, 0.4)", K == M_([[R(4, 5), R(2, 5)]]))
cv = Sj[0, 0] - (K * Sxy.T)[0]
check("cond var = 1.6", cv == R(8, 5))
check("(Sigma^-1)_11 = 5/8 = 0.625, reciprocal 1.6", Sj.inv()[0, 0] == R(5, 8) and 1 / Sj.inv()[0, 0] == R(8, 5))
check("det Sigma = 8, cofactor 5", Sj.det() == 8 and Syy.det() == 5)
check("cond mean at y = (1,2) is 1.6", (K * M_([1, 2]))[0] == R(8, 5))
Kx = Sj[1:, :1] / Sj[0, 0]
check("given x: gain (1/2,1/2), cond cov diag(1,2)", Kx == M_([R(1, 2), R(1, 2)]) and Syy - Sj[1:, :1] * Sj[:1, 1:] / 4 == sp.diag(1, 2))
check("precision block = inverse of conditional covariance (y block)", Sj.inv()[1:, 1:] == (Syy - Sj[1:, :1] * Sj[:1, 1:] / 4).inv())
# MC check of conditioning (scalar)
XY = rng.multivariate_normal([0, 0], [[4, 2], [2, 2]], size=400000)
sel = np.abs(XY[:, 1] - 1.5) < 0.05
check("MC: x | y ~ 1.5 has mean ~1.5 and var ~2", abs(XY[sel, 0].mean() - 1.5) < 0.08 and abs(XY[sel, 0].var() - 2) < 0.15)
# figure: joint of (y, x) with cov [[2,2],[2,4]], slice at y=1.5, conditional N(1.5, 2)
aJ, bJ, tJ = ell(np.array([[2., 2.], [2., 4.]]))
check("fig cond: joint ellipse axes 2.288, 0.874 at 58.283", close(aJ, 2.288, 5e-4) and close(bJ, 0.874, 5e-4) and close(tJ, 58.283, 1e-3))
figdata("ch02joint", f"{aJ:.3f}/{bJ:.3f}/{tJ:.3f}")
figdata("ch02joint95", f"{aJ*np.sqrt(c95):.3f}/{bJ*np.sqrt(c95):.3f}/{tJ:.3f}")
xs = np.arange(-2.5, 5.51, 0.25)
cd = np.exp(-(xs - 1.5) ** 2 / 4) / np.sqrt(4 * np.pi)
check("fig cond: conditional peak 1/sqrt(4 pi) = 0.282", close(cd.max(), 0.282, 5e-4))
figdata("ch02slice", " ".join(f"({1.5 + 6 * v:.3f},{x:.2f})" for x, v in zip(xs, cd)))

# ---------------------------------------------------------------- Example 2.4 (MMSE/LMMSE)
check("K = 4/5, error var 4 - 16/5 = 0.8", R(4, 5) == R(4, 5) and 4 - R(16, 5) == R(4, 5))
check("(1/4 + 1)^-1 = 0.8", 1 / (R(1, 4) + 1) == R(4, 5))
check("y = 2.5 gives estimate 2", R(4, 5) * R(5, 2) == 2)
yv = np.array([-1, 0, 1]); xv = yv ** 2
Ex = xv.mean(); covxy = (xv * yv).mean() - xv.mean() * yv.mean()
check("x = y^2: E[x] = 2/3, Cov(x,y) = 0", close(Ex, 2 / 3) and close(covxy, 0))
check("LMMSE MSE = Var x = 2/3 - 4/9 = 2/9", close((xv ** 2).mean() - Ex ** 2, 2 / 9) and R(2, 3) - R(4, 9) == R(2, 9))
check("MMSE error 0 (E[x|y] = y^2)", np.all(xv - yv ** 2 == 0))
# orthogonality numerically
Xs = rng.multivariate_normal([0, 0, 0], np.array(Sj, dtype=float), size=200000)
Kn = np.array(K, dtype=float)
e = Xs[:, 0] - Xs[:, 1:] @ Kn.ravel()
check("MC: LMMSE error uncorrelated with data", np.max(np.abs(Xs[:, 1:].T @ e / len(e))) < 0.02)
check("MC: LMMSE error var ~ 1.6", abs(e.var() - 1.6) < 0.03)

# ---------------------------------------------------------------- Example 2.5 (scalar Kalman)
def kf_scalar(m, P, ys, F=1, Q=1, H=1, Rn=2):
    out = []
    for y in ys:
        mp, Pp = F * m, F * P * F + Q
        Sk = H * Pp * H + Rn; Kk = Pp * H / Sk
        m = mp + Kk * (y - H * mp); P = Pp - Kk * Sk * Kk
        out.append((mp, Pp, Sk, Kk, m, P))
    return out


o = kf_scalar(R(0), R(4), [R(2), R(1)])
check("step1: P- = 5, S = 7, K = 5/7, m = 10/7, P = 10/7", o[0][1] == 5 and o[0][2] == 7 and o[0][3] == R(5, 7) and o[0][4] == R(10, 7) and o[0][5] == R(10, 7))
check("10/7 ~ 1.4286", close(10 / 7, 1.4286, 5e-5))
check("step2: P- = 17/7 ~ 2.4286, S = 31/7, K = 17/31 ~ 0.5484", o[1][1] == R(17, 7) and o[1][2] == R(31, 7) and o[1][3] == R(17, 31) and close(17 / 7, 2.4286, 5e-5) and close(17 / 31, 0.5484, 5e-5))
check("step2: innovation -3/7, m = 37/31 ~ 1.1935, P = 34/31 ~ 1.0968", R(1) - R(10, 7) == R(-3, 7) and o[1][4] == R(37, 31) and o[1][5] == R(34, 31) and close(37 / 31, 1.1935, 5e-5) and close(34 / 31, 1.0968, 5e-5))
Ps = sp.symbols('P', positive=True)
check("steady state: P^2 + P - 2 = 0 -> P = 1", sp.solve(sp.Eq(Ps, (Ps + 1) * 2 / (Ps + 3)), Ps) == [1])
# sawtooth figure: k = 0..8
saw = [(0, 4.0)]; m, P = 0.0, 4.0
for k in range(1, 9):
    Pp = P + 1; P = Pp * 2 / (Pp + 2)
    saw += [(k, Pp), (k, P)]
check("fig sawtooth: last prior ~2 and posterior ~1 (within 1e-3)", abs(saw[-2][1] - 2) < 1e-3 and abs(saw[-1][1] - 1) < 1e-3)
figdata("ch02saw", " ".join(f"({k},{v:.4f})" for k, v in saw))
# Monte Carlo of the filter (Exercise 2.11 logic)
nrep, T = 2000, 30
errs = np.zeros((nrep, T)); rr = np.random.default_rng(11)
for r in range(nrep):
    x = rr.normal(0, 2.0); m, P = 0.0, 4.0
    for k in range(T):
        x = x + rr.normal(0, 1.0); y = x + rr.normal(0, np.sqrt(2))
        Pp = P + 1; Kk = Pp / (Pp + 2); m = m + Kk * (y - m); P = Pp - Kk * Pp
        errs[r, k] = x - m
check("MC: filter error variance at step 30 ~ 1", abs(errs[:, -1].var() - 1) < 0.1)

# ---------------------------------------------------------------- Example 2.6 (vector Kalman)
F = M_([[1, 1], [0, 1]]); Q = sp.diag(0, 1); H = M_([[1, 0]]); Rr = M_([[1]])
m0 = M_([0, 1]); S0 = sp.eye(2)
mp = F * m0; Spr = F * S0 * F.T + Q
check("m1- = (1,1), Sigma1- = [[2,1],[1,2]]", mp == M_([1, 1]) and Spr == M_([[2, 1], [1, 2]]) and F * F.T == M_([[2, 1], [1, 1]]))
Sk = H * Spr * H.T + Rr; Kk = Spr * H.T * Sk.inv()
check("S = 3, K = (2,1)/3", Sk == M_([[3]]) and Kk == M_([2, 1]) / 3)
m1 = mp + Kk * (M_([[R(3, 2)]]) - H * mp)
check("m1 = (4/3, 7/6)", m1 == M_([R(4, 3), R(7, 6)]))
S1 = Spr - Kk * Sk * Kk.T
check("Sigma1 = [[2/3,1/3],[1/3,5/3]]", S1 == M_([[R(2, 3), R(1, 3)], [R(1, 3), R(5, 3)]]))
check("Joseph form and (I-KH)Sigma- agree", (sp.eye(2) - Kk * H) * Spr == S1 and (sp.eye(2) - Kk * H) * Spr * (sp.eye(2) - Kk * H).T + Kk * Rr * Kk.T == S1)
check("velocity reader gain 2 - 5/3 = 1/3, position reader 2 - 2/3 = 4/3", Spr[1, 1] - S1[1, 1] == R(1, 3) and Spr[0, 0] - S1[0, 0] == R(4, 3))
h = M_([1, 0])
for g, val in [(M_([0, 1]), R(1, 3)), (M_([1, 0]), R(4, 3))]:
    num = (g.T * Spr * h)[0]
    check(f"rank-one VoI formula for g={list(g)}: ({num})^2/3 = {val}", num ** 2 / ((h.T * Spr * h)[0] + 1) == val and (g * g.T * (Spr - S1)).trace() == val)
# figure: ellipses before/after
aP, bP, tP = ell(np.array(Spr, dtype=float)); aQ, bQ, tQ = ell(np.array(S1, dtype=float))
check("fig update: prior axes sqrt3, 1 at 45", close(aP, np.sqrt(3)) and close(bP, 1) and close(tP, 45, 1e-9))
check("fig update: posterior axes 1.330, 0.752 at 73.155", close(aQ, 1.330, 5e-4) and close(bQ, 0.752, 5e-4) and close(tQ, 73.155, 1e-3))
check("fig update: posterior ellipse inside prior (Loewner)", min(np.linalg.eigvalsh(np.array(Spr - S1, dtype=float))) >= -1e-12)
figdata("ch02prior", f"{aP:.3f}/{bP:.3f}/{tP:.3f}")
figdata("ch02post", f"{aQ:.3f}/{bQ:.3f}/{tQ:.3f}")

# ---------------------------------------------------------------- Example 2.7 (Fisher)
Hf = M_([[1, 0], [1, 1], [0, 1]])
I_ = Hf.T * Hf
check("I = [[2,1],[1,2]], I^-1 = (1/3)[[2,-1],[-1,2]]", I_ == M_([[2, 1], [1, 2]]) and I_.inv() == M_([[2, -1], [-1, 2]]) / 3)
check("per-parameter bound 2/3", I_.inv()[0, 0] == R(2, 3) and I_.inv()[1, 1] == R(2, 3))
check("sum reader 2/3, difference reader 2", (M_([1, 1]).T * I_.inv() * M_([1, 1]))[0] == R(2, 3) and (M_([1, -1]).T * I_.inv() * M_([1, -1]))[0] == 2)
info = Spr.inv() + H.T * Rr.inv() * H
check("info form: (Sigma-)^-1 = (1/3)[[2,-1],[-1,2]]", Spr.inv() == M_([[2, -1], [-1, 2]]) / 3)
check("info form sum = [[5/3,-1/3],[-1/3,2/3]] det 1, inverse Sigma1", info == M_([[R(5, 3), R(-1, 3)], [R(-1, 3), R(2, 3)]]) and info.det() == 1 and info.inv() == S1)
# CRB attained by least squares (MC)
th = np.array([1., -1.]); Hn = np.array(Hf, dtype=float)
ests = []
for _ in range(20000):
    yv_ = Hn @ th + rng.standard_normal(3)
    ests.append(np.linalg.solve(Hn.T @ Hn, Hn.T @ yv_))
check("MC: LS covariance ~ I^-1", np.max(np.abs(np.cov(np.array(ests).T) - np.array(I_.inv(), dtype=float))) < 0.03)
# figure: log-likelihood curves for n=2 and n=16, sigma^2=1
d = np.arange(-1.5, 1.51, 0.1)
ll2 = -2 * d ** 2 / 2; ll16 = -16 * d ** 2 / 2
check("fig Fisher: curvature equals information (2 and 16)", close(-np.polyfit(d, ll2, 2)[0] * 2, 2, 1e-9) and close(-np.polyfit(d, ll16, 2)[0] * 2, 16, 1e-9))
figdata("ch02ll2", " ".join(f"({a:.1f},{b:.3f})" for a, b in zip(d, ll2)))
figdata("ch02ll16", " ".join(f"({a:.1f},{b:.3f})" for a, b in zip(d, ll16) if b >= -6.5))
check("fig Fisher: 1/sqrt(I) widths 0.707 and 0.25", close(1 / np.sqrt(2), 0.707, 5e-4) and close(1 / np.sqrt(16), 0.25))

# ---------------------------------------------------------------- Example 2.8 (mutual information)
check("rho^2 = 4/8 = 1/2", R(4, 8) == R(1, 2))
check("I = 1/2 log 2 ~ 0.3466 nats = 0.5 bits", close(0.5 * np.log(2), 0.3466, 5e-5) and close(0.5 * np.log2(2), 0.5))
check("det form 8/4 = 2", R(4 * 2, 4 * 2 - 4) == 2)
check("Kalman: det Sigma1- = 3, det Sigma1 = 1", Spr.det() == 3 and S1.det() == 1)
check("1/2 log 3 ~ 0.5493 nats", close(0.5 * np.log(3), 0.5493, 5e-5))
check("det S / det R = 3", Sk.det() / Rr.det() == 3)
# general identity det(S-)/det(S+) = det S / det R (random)
okd = True
for _ in range(100):
    n_, p_ = 3, 2
    Pm = rng.standard_normal((n_, n_)); Pm = Pm @ Pm.T + np.eye(n_)
    Hm = rng.standard_normal((p_, n_)); Rm = rng.standard_normal((p_, p_)); Rm = Rm @ Rm.T + np.eye(p_)
    Sm = Hm @ Pm @ Hm.T + Rm; Kn_ = Pm @ Hm.T @ np.linalg.inv(Sm); Pp_ = Pm - Kn_ @ Sm @ Kn_.T
    okd &= close(np.linalg.det(Pm) / np.linalg.det(Pp_), np.linalg.det(Sm) / np.linalg.det(Rm), 1e-6 * np.linalg.det(Sm) / np.linalg.det(Rm))
check("random: det Sigma-/det Sigma+ = det S/det R", okd)
# figure: I(rho) curve
rhos = np.arange(0, 0.951, 0.05)
Irho = -0.5 * np.log(1 - rhos ** 2)
check("fig MI: I at rho = 0.95 is 1.164 nats", close(Irho[-1], 1.164, 5e-4))
figdata("ch02mi", " ".join(f"({r:.2f},{v:.4f})" for r, v in zip(rhos, Irho)))
check("fig MI: rho = 1/sqrt2 = 0.707 gives 0.347", close(1 / np.sqrt(2), 0.707, 5e-4) and close(-0.5 * np.log(0.5), 0.347, 5e-4))

# ---------------------------------------------------------------- Exercise answers
Se = M_([[1, R(1, 2)], [R(1, 2), 1]])
check("Ex2.1: Var(x1-x2) = 1, Var(2x1+x2) = 7", (M_([1, -1]).T * Se * M_([1, -1]))[0] == 1 and (M_([2, 1]).T * Se * M_([2, 1]))[0] == 7)
xs_ = rng.standard_normal(200000); s_ = rng.choice([-1, 1], size=200000); ys_ = s_ * xs_
check("Ex2.2: y = s x has var 1, corr 0, |y| = |x|, P(x+y=0) ~ 1/2", abs(ys_.var() - 1) < 0.02 and abs(np.mean(xs_ * ys_)) < 0.02 and np.all(np.abs(ys_) == np.abs(xs_)) and abs(np.mean(xs_ + ys_ == 0) - 0.5) < 0.01)
check("Ex2.3: r^2 = 2, probability 1 - e^-1 ~ 0.6321", R(4, 4) + 1 == 2 and close(1 - np.exp(-1), 0.6321, 5e-5))
check("Ex2.4: MC error uncorrelated with estimate", abs(np.mean(e * (Xs[:, 1:] @ Kn.ravel()))) < 0.02)
check("Ex2.5: K = 1/2, mean 1/2, var 5/2", R(1, 2) * 1 == R(1, 2) and 3 - R(1, 2) == R(5, 2))
o2 = kf_scalar(R(0), R(1), [R(1), R(0)], Q=0, Rn=1)
check("Ex2.6: step1 P-=1 S=2 K=1/2 m=1/2 P=1/2", o2[0][1] == 1 and o2[0][2] == 2 and o2[0][3] == R(1, 2) and o2[0][4] == R(1, 2) and o2[0][5] == R(1, 2))
check("Ex2.6: step2 P-=1/2 S=3/2 K=1/3 m=1/3 P=1/3", o2[1][1] == R(1, 2) and o2[1][2] == R(3, 2) and o2[1][3] == R(1, 3) and o2[1][4] == R(1, 3) and o2[1][5] == R(1, 3))
check("Ex2.6: average of 0,1,0 is 1/3", R(0 + 1 + 0, 3) == R(1, 3))
Pm_ = sp.symbols('Pm', positive=True); q, r = sp.symbols('q r', positive=True)
check("Ex2.7: steady-state equation", sp.expand((Pm_ - q) * (Pm_ + r) - Pm_ * r - (Pm_ ** 2 - q * Pm_ - q * r)) == 0)
root = sp.solve(Pm_ ** 2 - 2 * Pm_ - 2, Pm_)
check("Ex2.7: q=2, r=1 gives 1+sqrt3 ~ 2.732 and sqrt3-1 ~ 0.732", root == [1 + sp.sqrt(3)] and close(1 + np.sqrt(3), 2.732, 5e-4) and close(np.sqrt(3) - 1, 0.732, 5e-4))
check("Ex2.7: q=1, r=2 gives 2", sp.solve(Pm_ ** 2 - Pm_ - 2, Pm_) == [2])
vv = sp.symbols('v', positive=True); yy, tt = sp.symbols('y theta', real=True)
ll = -sp.log(vv) / 2 - (yy - tt) ** 2 / (2 * vv)
# -E[d2 ll/dv2] is linear in (y-theta)^2, so substituting E[(y-theta)^2] = v is exact
Ifish = sp.simplify(-sp.diff(ll, vv, 2).subs(yy, tt + sp.sqrt(vv)))
check("Ex2.8: theta bound sigma^2/n = 0.5", R(2, 4) == R(1, 2))
check("Ex2.8: Fisher for variance per reading 1/(2v^2), n=4, v=2 -> 1/2, bound 2", sp.simplify(Ifish - 1 / (2 * vv ** 2)) == 0 and 4 / (2 * R(2) ** 2) == R(1, 2))
check("Ex2.8: estimator variance 2 v^2 / n = 2", 2 * 4 / 4 == 2)
vs = np.mean((rng.normal(0, np.sqrt(2), size=(200000, 4))) ** 2, axis=1)
check("Ex2.8: MC variance of estimator ~ 2", abs(vs.var() - 2) < 0.05)
check("Ex2.9: 1/2 log(5/(10/7)) = 1/2 log 3.5 ~ 0.6264 = 1/2 log(7/2)", close(0.5 * np.log(5 / (10 / 7)), 0.6264, 5e-5) and close(0.5 * np.log(3.5), 0.5 * np.log(7 / 2)))

n_pass = sum(results)
print(f"{n_pass}/{len(results)} PASS")
sys.exit(0 if n_pass == len(results) else 1)
