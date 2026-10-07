"""Checks every number in chapter 12 (tail bounds and concentration), its exercise
answers and figures.

Run on Atlas:  /home/claude/env/bin/python3 ch12_examples.py
Prints PASS/FAIL per claim, FIGDATA lines for figure coordinates, then 'k/n PASS';
exits nonzero on any FAIL.
"""
import math
import sys

import numpy as np
import sympy as sp
from scipy import stats, integrate

results = []


def check(desc, cond):
    ok = bool(cond)
    results.append(ok)
    print(("PASS" if ok else "FAIL") + "  " + desc)


def figdata(name, pts, fmt="{:.6g}"):
    print("FIGDATA " + name + "|" + " ".join("(" + fmt.format(x) + "," + fmt.format(y) + ")" for x, y in pts))


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def kl_bern(q, p):
    """KL divergence D(Bern(q)||Bern(p)) in nats."""
    out = 0.0
    if q > 0:
        out += q * math.log(q / p)
    if q < 1:
        out += (1 - q) * math.log((1 - q) / (1 - p))
    return out


Q = stats.norm.sf
phi = stats.norm.pdf

# ============================================================ Section 1 Markov / Chebyshev
n, p = 100, 0.5
mu, var = n * p, n * p * (1 - p)
B = stats.binom(n, p)
check("Binomial(100,1/2) mean 50, variance 25", mu == 50 and var == 25)
check("Markov Pr[S>=60] <= 50/60 = 0.833", close(50 / 60, 0.8333, 5e-5))
check("Chebyshev Pr[|S-50|>=10] <= 25/100 = 0.25", 25 / 100 == 0.25)
exact60 = B.sf(59)
print("exact Pr[S>=60] =", exact60)
check("exact Pr[S>=60] = 0.0284", close(exact60, 0.0284, 5e-5))
exact_two = B.sf(59) + B.cdf(40)
check("exact Pr[|S-50|>=10] = 0.0569", close(exact_two, 0.0569, 5e-5))
check("Chebyshev / exact two-sided = 4.4", close(0.25 / exact_two, 4.4, 0.05))
check("Markov / exact ratio about 29", close((50 / 60) / exact60, 29.3, 0.05))
# Markov is tight: X = a w.p. m/a, 0 otherwise
check("two-point law attains Markov: E = a*(m/a) = m", 60 * (50 / 60) == 50)
# Chebyshev is tight: X = +-k with prob 1/(2k^2), 0 otherwise has variance 1
k = 3
check("three-point law has variance 1 and Pr[|X|>=3] = 1/9", close(2 * k**2 / (2 * k**2), 1, 1e-12) and close(2 / (2 * k**2), 1 / 9, 1e-12))

# Figure 1: Binomial(100,1/2) upper tail and the bounds
f1 = {"true": [], "markov": [], "cheb": [], "chern": []}
for kk in range(52, 82, 2):
    q = kk / n
    f1["true"].append((kk, B.sf(kk - 1)))
    f1["markov"].append((kk, min(1.0, 50 / kk)))
    f1["cheb"].append((kk, min(1.0, 25 / (kk - 50) ** 2)))
    f1["chern"].append((kk, math.exp(-n * kl_bern(q, p))))
check("fig1 Chernoff within a factor 10 of the exact tail on 52..80",
      all(c[1] / t_[1] < 10 for c, t_ in zip(f1["chern"], f1["true"])))
for key in f1:
    figdata("fig1-" + key, f1[key])
check("fig1 ordering true <= Chernoff <= ... at k=70",
      f1["true"][9][1] < f1["chern"][9][1] < f1["cheb"][9][1] < f1["markov"][9][1])
check("fig1 at k=70: true 3.93e-5, Chernoff 2.67e-4, Chebyshev 0.0625, Markov 0.714",
      close(f1["true"][9][1], 3.93e-5, 5e-7) and close(f1["chern"][9][1], 2.67e-4, 5e-6)
      and close(f1["cheb"][9][1], 0.0625, 1e-9) and close(f1["markov"][9][1], 0.7143, 5e-5))
print("k=70 values", f1["true"][9], f1["chern"][9])

# ============================================================ Section 2 Chernoff method
# Gaussian Chernoff: inf_l exp(-l t + l^2 s^2/2) = exp(-t^2/(2 s^2)) at l = t/s^2
l, t, s = sp.symbols("l t s", positive=True)
expo = -l * t + l**2 * s**2 / 2
lstar = sp.solve(sp.diff(expo, l), l)[0]
check("Gaussian Chernoff optimum lambda = t/s^2", sp.simplify(lstar - t / s**2) == 0)
check("Gaussian Chernoff exponent -t^2/(2 s^2)", sp.simplify(expo.subs(l, lstar) + t**2 / (2 * s**2)) == 0)
# Bernoulli sum, KL form, at n=100 p=1/2 a=60
lam_star = math.log(0.6 * 0.5 / (0.5 * 0.4))
check("Bernoulli optimum lambda* = ln 1.5 = 0.4055", close(lam_star, 0.405465, 5e-6))
fl = lambda la: math.exp(-60 * la) * ((1 + math.exp(la)) / 2) ** 100
check("f(lambda*) equals exp(-n D(0.6||0.5)) = 0.1335", close(fl(lam_star), math.exp(-100 * kl_bern(0.6, 0.5)), 1e-12)
      and close(fl(lam_star), 0.1335, 5e-5))
check("D(0.6||0.5) = 0.02014 nats", close(kl_bern(0.6, 0.5), 0.020136, 5e-7))
check("f(0)=1 and f(1) = 7.47 (bound useless at lambda = 1)", fl(0) == 1 and close(fl(1), 7.47, 0.01))
print("f(1) =", fl(1))
lgrid = np.linspace(0, 1, 21)
figdata("fig2-f", [(x, fl(x)) for x in lgrid])
check("Chernoff 0.1335 / exact 0.0284 ratio about 4.7", close(0.13353 / exact60, 4.70, 0.01))
# multiplicative Chernoff, delta = 0.2, mu = 50
mult = (math.exp(0.2) / 1.2**1.2) ** 50
check("multiplicative Chernoff (e^d/(1+d)^(1+d))^mu at d=0.2, mu=50 is 0.391", close(mult, 0.391, 5e-4))
check("simplified exp(-d^2 mu/3) at d=0.2 mu=50 is 0.513", close(math.exp(-0.04 * 50 / 3), 0.513, 5e-4))
# e^d/(1+d)^(1+d) <= exp(-d^2/3) on (0,1]; and lower tail form
dg = np.linspace(1e-4, 1, 10001)
check("e^d/(1+d)^(1+d) <= exp(-d^2/3) for d in (0,1]",
      np.all(dg - (1 + dg) * np.log1p(dg) <= -dg**2 / 3 + 1e-15))
check("e^-d/(1-d)^(1-d) <= exp(-d^2/2) for d in (0,1)",
      np.all(-dg[:-1] - (1 - dg[:-1]) * np.log1p(-dg[:-1]) <= -dg[:-1] ** 2 / 2 + 1e-15))
# rare-event case n=1000, p=0.01, Pr[S>=20]
B2 = stats.binom(1000, 0.01)
ex20 = B2.sf(19)
print("exact Pr[S>=20], n=1000 p=0.01:", ex20)
check("rare case exact Pr[S>=20] = 0.00329", close(ex20, 0.00329, 5e-6))
check("rare case Hoeffding exp(-2*1000*0.01^2) = 0.819", close(math.exp(-0.2), 0.8187, 5e-5))
check("rare case multiplicative Chernoff (e/4)^10 = 0.0210", close((math.e / 4) ** 10, 0.0210, 5e-5))
chkl = math.exp(-1000 * kl_bern(0.02, 0.01))
print("KL Chernoff rare:", chkl, kl_bern(0.02, 0.01))
check("rare case KL Chernoff exp(-1000 D(0.02||0.01)) = 0.0200", close(chkl, 0.0200, 5e-5))

# ============================================================ Section 3 Hoeffding and Bernstein
def bern(nn, pp, eps, c=1.0):
    return math.exp(-nn * eps**2 / (2 * pp * (1 - pp) + 2 * c * eps / 3))


hb = bern(1000, 0.01, 0.01)
print("Bernstein rare:", hb)
check("Bernstein n=1000 p=0.01 eps=0.01 (c=1) = 0.0229", close(hb, 0.0229, 5e-5))
check("Hoeffding n=100 eps=0.1 gives e^-2 = 0.1353", close(math.exp(-2 * 100 * 0.01), 0.1353, 5e-5))
# Figure 3: n=1000, eps=0.01, tail vs p
f3 = {"exact": [], "hoef": [], "bern": []}
for pp in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
    kk = math.ceil(round(1000 * (pp + 0.01), 9))
    f3["exact"].append((pp, stats.binom(1000, pp).sf(kk - 1)))
    f3["hoef"].append((pp, math.exp(-2 * 1000 * 0.01**2)))
    f3["bern"].append((pp, bern(1000, pp, 0.01)))
check("Bernstein at p=0.005 is 336 times smaller than Hoeffding", close(0.818731 / f3["bern"][1][1], 336, 1))
check("variance-aware bounds 0.0200..0.0229 sit 6.1 to 7.0 times above exact 0.00329",
      close(chkl / ex20, 6.1, 0.05) and close(hb / ex20, 7.0, 0.05) and close(hb / chkl, 1.145, 0.005))
check("Hoeffding 0.819 is about 250 times exact 0.00329", close(math.exp(-0.2) / ex20, 249, 1))
for key in f3:
    figdata("fig3-" + key, f3[key])
check("fig3 Bernstein below Hoeffding for every grid p < 0.5", all(b[1] < h[1] for b, h in zip(f3["bern"], f3["hoef"]) if b[0] < 0.5))
check("fig3 Bernstein above Hoeffding at p = 0.5", f3["bern"][-1][1] > f3["hoef"][-1][1])
check("fig3 exact below both bounds everywhere",
      all(e[1] <= min(h[1], b[1]) for e, h, b in zip(f3["exact"], f3["hoef"], f3["bern"])))
print("fig3 p=0.5 bern", f3["bern"][-1][1])
# Bernstein vs Hoeffding at p=1/2: exponents 2 n eps^2 vs n eps^2 / (1/2 + 2 eps/3)
check("at p=1/2 Bernstein exponent 0.1/(0.5+0.00667) = 0.1974 < Hoeffding 0.2",
      close(1000 * 1e-4 / (0.5 + 2 * 0.01 / 3), 0.1974, 5e-5))

# ============================================================ Section 4 sub-Gaussian, Mills
check("Rademacher cosh(1)=1.5431 <= e^(1/2)=1.6487", close(math.cosh(1), 1.5431, 5e-5) and close(math.exp(0.5), 1.6487, 5e-5)
      and math.cosh(1) <= math.exp(0.5))
xs = np.linspace(-5, 5, 2001)
check("cosh(x) <= exp(x^2/2) on a grid", np.all(np.cosh(xs) <= np.exp(xs**2 / 2) + 1e-12))
# Hoeffding lemma numeric check: Bernoulli(p) centred, MGF <= exp(l^2/8)
ok = True
for pp in np.linspace(0.01, 0.99, 50):
    for la in np.linspace(-6, 6, 121):
        mgf = pp * math.exp(la * (1 - pp)) + (1 - pp) * math.exp(-la * pp)
        ok &= mgf <= math.exp(la**2 / 8) * (1 + 1e-12)
check("Hoeffding lemma: centred Bernoulli MGF <= exp(l^2 (b-a)^2/8) on a grid", ok)
q3 = Q(3)
check("Q(3) = 0.001350", close(q3, 0.0013499, 5e-8))
check("Mills upper phi(3)/3 = 0.001477", close(phi(3) / 3, 0.0014773, 5e-8))
check("Mills lower phi(3)(1/3-1/27) = 0.001313", close(phi(3) * (1 / 3 - 1 / 27), 0.0013131, 5e-8))
check("Chernoff e^-4.5 = 0.01111", close(math.exp(-4.5), 0.011109, 5e-7))
check("Chernoff/true at t=3 about 8.2", close(math.exp(-4.5) / q3, 8.23, 0.01))
check("Mills upper/true at t=3 = 1.094", close(phi(3) / 3 / q3, 1.094, 5e-4))
check("Q(5) = 2.87e-7, Mills upper 2.97e-7", close(Q(5), 2.867e-7, 5e-10) and close(phi(5) / 5, 2.973e-7, 5e-10))
f4 = {"true": [], "up": [], "lo": [], "ch": []}
for tt in np.arange(1.5, 6.01, 0.25):
    f4["true"].append((tt, Q(tt)))
    f4["up"].append((tt, phi(tt) / tt))
    f4["lo"].append((tt, phi(tt) * (1 / tt - 1 / tt**3)))
    f4["ch"].append((tt, math.exp(-tt**2 / 2)))
for key in f4:
    figdata("fig4-" + key, f4[key])
for key in ["up", "lo", "ch"]:
    figdata("fig4-ratio-" + key, [(u[0], u[1] / v[1]) for u, v in zip(f4[key], f4["true"])], "{:.4g}")
check("fig4 ratio of Chernoff to Q reaches 15.4 at t=6",
      f4["ch"][-1][1] / f4["true"][-1][1] > 15 and close(f4["ch"][-1][1] / f4["true"][-1][1], 15.44, 0.01))
check("fig4 ratios of Mills bounds within 3 percent of 1 for t >= 6",
      close(f4["up"][-1][1] / f4["true"][-1][1], 1.026, 0.001) and close(f4["lo"][-1][1] / f4["true"][-1][1], 0.998, 0.001))
print("t=6 ratios", f4["ch"][-1][1] / f4["true"][-1][1], f4["up"][-1][1] / f4["true"][-1][1], f4["lo"][-1][1] / f4["true"][-1][1])
check("fig4 lo <= true <= up <= ch on grid (t >= 1.5)",
      all(a[1] <= b[1] <= c[1] for a, b, c in zip(f4["lo"], f4["true"], f4["up"])) and
      all(c[1] <= d[1] for c, d in zip(f4["up"], f4["ch"])))
# z for one-sided 0.05/n via union bound
check("one-sided 0.025 critical value 1.96", close(stats.norm.isf(0.025), 1.95996, 5e-5))

# ============================================================ Section 5 maximum of n Gaussians
def emax(nn):
    f = lambda x: x * nn * phi(x) * stats.norm.cdf(x) ** (nn - 1)
    v, _ = integrate.quad(f, -10, 12, limit=400, points=[0, math.sqrt(2 * math.log(max(nn, 2)))])
    return v


check("E max of 2 standard normals = 1/sqrt(pi) = 0.5642", close(emax(2), 1 / math.sqrt(math.pi), 1e-7))
e40 = emax(40)
print("E max 40 =", e40, " sqrt(2 ln 40) =", math.sqrt(2 * math.log(40)))
check("E max of 40 = 2.161", close(e40, 2.161, 5e-4))
check("sqrt(2 ln 40) = 2.716", close(math.sqrt(2 * math.log(40)), 2.716, 5e-4))
p40 = 1 - 0.975**40
check("Pr[some of 40 null z >= 1.96] = 1 - 0.975^40 = 0.637", close(p40, 0.637, 5e-4))
check("union bound 40*0.025 = 1 (vacuous)", close(40 * 0.025, 1.0, 1e-12))
# median of max of 40
med40 = stats.norm.ppf(0.5 ** (1 / 40))
check("median of max of 40 = 2.116", close(med40, 2.116, 5e-4))
print("median max 40", med40)
ns = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
f5e = [(nn, emax(nn)) for nn in ns]
f5b = [(nn, math.sqrt(2 * math.log(nn))) for nn in ns]
check("E max 1 = 0", close(f5e[0][1], 0, 1e-7))
check("E max <= sqrt(2 ln n) for n >= 2 in the grid", all(e[1] <= b[1] for e, b in zip(f5e[1:], f5b[1:])))
check("ratio E max / sqrt(2 ln n) rises toward 1 (0.826 at 100, 0.897 at 10^4)",
      close(f5e[6][1] / f5b[6][1], 0.826, 5e-4) and close(f5e[-1][1] / f5b[-1][1], 0.897, 5e-4))
print("ratios 100, 1e4:", f5e[6][1] / f5b[6][1], f5e[-1][1] / f5b[-1][1])
print("E max 100, 10000:", f5e[6][1], f5e[-1][1])
nx = 2
while emax(nx) < 1.96:
    nx += 1
print("first n with E max >= 1.96:", nx)
check("E max first exceeds 1.96 at n = 25", nx == 25)
check("E max increments 100->1000 = 0.73, 1000->10^4 = 0.61",
      close(emax(1000) - emax(100), 0.73, 0.005) and close(emax(10000) - emax(1000), 0.61, 0.005))
figdata("fig5-exact", f5e)
figdata("fig5-bound", f5b)
rng = np.random.default_rng(12)
sim = []
for nn in [10, 100, 1000, 10000]:
    m = np.array([rng.standard_normal(nn).max() for _ in range(1000)])
    sim.append((nn, m.mean()))
figdata("fig5-sim", sim)
check("simulated means within 0.03 of exact", all(close(sm[1], emax(sm[0]), 0.03) for sm in sim))
# proof steps: E max <= (ln n)/l + l/2, minimized at l = sqrt(2 ln n)
nn_ = sp.symbols("n", positive=True)
g = sp.log(nn_) / l + l / 2
ls = sp.solve(sp.diff(g, l), l)
check("minimizer of ln n / l + l/2 is sqrt(2 ln n) with value sqrt(2 ln n)",
      any(sp.simplify(sp.sqrt(2 * sp.log(nn_)) - r) == 0 for r in ls)
      and sp.simplify(g.subs(l, sp.sqrt(2 * sp.log(nn_))) - sp.sqrt(2 * sp.log(nn_))) == 0)

# ============================================================ Section 6 union bound
def n_hoef(m, eps, delta):
    return math.ceil(math.log(2 * m / delta) / (2 * eps**2))


def n_cheb(m, eps, delta):
    return math.ceil(m / (4 * delta * eps**2) - 1e-9)


check("Hoeffding n for one estimate eps=0.02 delta=0.05 is 4612", n_hoef(1, 0.02, 0.05) == 4612)
check("Hoeffding n for 1000 estimates is 13246", n_hoef(1000, 0.02, 0.05) == 13246)
check("factor 13246/4612 = 2.87", close(13246 / 4612, 2.87, 0.005))
check("Chebyshev n for one estimate is 12500", n_cheb(1, 0.02, 0.05) == 12500)
check("Chebyshev n for 1000 estimates is 12,500,000", n_cheb(1000, 0.02, 0.05) == 12500000)
print("n_hoef 1e4", n_hoef(10**4, 0.02, 0.05))
check("Hoeffding n for 10^4 estimates is 16125", n_hoef(10**4, 0.02, 0.05) == 16125)
check("10^4 estimates cost 16125/4612 = 3.50 times one", close(16125 / 4612, 3.50, 0.005))
ms = [1, 10, 100, 1000, 10000]
figdata("fig6-hoef", [(m, n_hoef(m, 0.02, 0.05)) for m in ms])
figdata("fig6-cheb", [(m, n_cheb(m, 0.02, 0.05)) for m in ms])
# Gaussian threshold with union bound, delta = 0.05: z = Q^{-1}(delta/m)
zs = [(m, stats.norm.isf(0.05 / m)) for m in [1, 10, 100, 1000, 10000, 10**6]]
print("union z thresholds", zs)
check("one-sided union-bound thresholds 1.645, 2.576, 3.291, 3.891, 4.417 for m=1..10^4",
      all(close(a[1], b, 5e-4) for a, b in zip(zs, [1.645, 2.576, 3.291, 3.891, 4.417])))
check("m = 10^6 threshold 5.327 and sqrt(2 ln(10^6/0.05)) = 5.798",
      close(zs[-1][1], 5.327, 5e-4) and close(math.sqrt(2 * math.log(1e6 / 0.05)), 5.798, 5e-4))
# independent events: exact vs union
check("20 events of 0.001: union 0.02, independent 0.01981",
      close(20 * 0.001, 0.02, 1e-12) and close(1 - 0.999**20, 0.01981, 5e-6))

# ============================================================ Section 7 Clopper-Pearson
def cp(x, nn, alpha=0.05):
    lo = 0.0 if x == 0 else stats.beta.ppf(alpha / 2, x, nn - x + 1)
    hi = 1.0 if x == nn else stats.beta.ppf(1 - alpha / 2, x + 1, nn - x)
    return lo, hi


def wald(x, nn, z=1.959963984540054):
    ph = x / nn
    h = z * math.sqrt(ph * (1 - ph) / nn)
    return ph - h, ph + h


# 0 of 300
up1 = 1 - 0.05 ** (1 / 300)
check("0/300 one-sided 95% upper = 1-0.05^(1/300) = 0.00994", close(up1, 0.009936, 5e-7))
check("rule of three 3/300 = 0.0100", close(3 / 300, 0.01, 1e-12))
check("-ln 0.05 = 2.996", close(-math.log(0.05), 2.9957, 5e-5))
up2 = cp(0, 300)[1]
check("0/300 two-sided CP upper = 1-0.025^(1/300) = 0.01222", close(up2, 1 - 0.025 ** (1 / 300), 1e-10) and close(up2, 0.01222, 5e-6))
check("Wald at 0/300 is [0,0]", wald(0, 300) == (0.0, 0.0))
# 3 of 50
lo, hi = cp(3, 50)
wl, wh = wald(3, 50)
print("CP 3/50", lo, hi, "Wald", wl, wh)
check("CP 3/50 = [0.0125, 0.1655]", close(lo, 0.0125, 5e-5) and close(hi, 0.1655, 5e-5))
check("Wald 3/50 = [-0.0058, 0.1258]", close(wl, -0.0058, 5e-5) and close(wh, 0.1258, 5e-5))
# CP definition: Pr[Bin(50,hi) <= 3] = 0.025 and Pr[Bin(50,lo) >= 3] = 0.025
check("CP endpoints invert the binomial tails",
      close(stats.binom(50, hi).cdf(3), 0.025, 1e-9) and close(stats.binom(50, lo).sf(2), 0.025, 1e-9))
# 20/20 (exercise)
check("20/20 CP lower = 0.025^(1/20) = 0.8316", close(cp(20, 20)[0], 0.8316, 5e-5) and close(0.025 ** (1 / 20), 0.8316, 5e-5))


def coverage(nn, pp, method):
    tot = 0.0
    pm = stats.binom(nn, pp).pmf(np.arange(nn + 1))
    for x in range(nn + 1):
        a, b = method(x, nn)
        if a <= pp <= b:
            tot += pm[x]
    return tot


f7w, f7c = [], []
for pp in np.arange(0.01, 0.501, 0.01):
    f7w.append((pp, coverage(50, pp, wald)))
    f7c.append((pp, coverage(50, pp, cp)))
figdata("fig7-wald", f7w, "{:.4f}")
figdata("fig7-cp", f7c, "{:.4f}")
check("Wald misses when x=0, probability 0.99^50 = 0.605", close(0.99**50, 0.605, 5e-4) and f7w[0][1] <= 1 - 0.99**50 and close(1 - 0.99**50, f7w[0][1], 1e-3))
check("CP coverage >= 0.95 on the grid", min(c[1] for c in f7c) >= 0.95 - 1e-12)
wmin = min(f7w, key=lambda z: z[1])
print("Wald min coverage", wmin, "Wald at p=0.01", f7w[0], "at 0.05", f7w[4], "CP mean", np.mean([c[1] for c in f7c]))
check("Wald coverage at p=0.01 is 0.395", close(f7w[0][1], 0.395, 5e-4))
check("Wald coverage at p=0.05 is 0.920", close(f7w[4][1], 0.920, 5e-4))
check("Wald coverage below 0.95 at most grid points", sum(w[1] < 0.95 for w in f7w) >= 35)
print("wald below .95 count", sum(w[1] < 0.95 for w in f7w))
check("CP mean coverage over grid about 0.97", close(np.mean([c[1] for c in f7c]), 0.97, 0.01))

# ============================================================ Section 8 order statistics
n99 = math.ceil(math.log(0.05) / math.log(0.99))
check("smallest n with 1-0.99^n >= 0.95 is 299", n99 == 299 and 1 - 0.99**299 >= 0.95 and 1 - 0.99**298 < 0.95)
check("1 - 0.99^299 = 0.9505", close(1 - 0.99**299, 0.9505, 5e-5))
check("n for p=0.999 is 2995", math.ceil(math.log(0.05) / math.log(0.999)) == 2995)
check("exchangeability: Pr[new > max of 299] = 1/300 = 0.00333", close(1 / 300, 0.003333, 5e-7))


def conf_k(nn, k, pp=0.99):
    return stats.binom(nn, 1 - pp).sf(k - 1)


def nmin(k, pp=0.99, d=0.05):
    nn = k
    while conf_k(nn, k, pp) < 1 - d:
        nn += 1
    return nn


n1, n2, n5 = nmin(1), nmin(2), nmin(5)
print("n for k=1,2,5:", n1, n2, n5)
check("k=1 needs 299, k=2 needs 473, k=5 needs 913", (n1, n2, n5) == (299, 473, 913))
c1000 = conf_k(1000, 5)
print("conf 5th largest of 1000:", c1000)
check("confidence that the 5th largest of 1000 is >= q_0.99 is 0.971", close(c1000, 0.971, 5e-4))
f8 = {}
for k in [1, 2, 5]:
    f8[k] = [(nn, conf_k(nn, k)) for nn in range(50, 1501, 50)]
    figdata("fig8-k%d" % k, f8[k], "{:.4g}")
# DKW with Massart constant
ndkw = math.ceil(math.log(2 / 0.05) / (2 * 0.01**2))
check("DKW n for eps=0.01 delta=0.05 is 18445", ndkw == 18445)
check("DKW n for eps=0.05 delta=0.05 is 738", math.ceil(math.log(40) / (2 * 0.0025)) == 738)
# simulation: distribution-free, exponential and Cauchy
rng = np.random.default_rng(7)
for name, sampler, q99 in [("exponential", lambda size: rng.exponential(size=size), -math.log(0.01)),
                           ("Cauchy", lambda size: rng.standard_cauchy(size=size), math.tan(math.pi * 0.49))]:
    mx = sampler((20000, 299)).max(axis=1)
    rate = np.mean(mx >= q99)
    print(name, "coverage of max>=q99", rate)
    check(f"{name}: simulated Pr[max of 299 >= q_0.99] within 0.01 of 0.9505", close(rate, 0.9505, 0.01))

# ============================================================ certificate illustration
check("200 anchors give 19900 pairs", 200 * 199 // 2 == 19900)
check("tau floor 1-2*0.0664 = 0.8672 (spec prints 0.8671 from unrounded mu_hat)", close(1 - 2 * 0.0664, 0.8672, 1e-9))
check("Spearman floor 1-3*0.0664 = 0.8008 (spec prints 0.8006)", close(1 - 3 * 0.0664, 0.8008, 1e-9))
check("spec floors imply mu_hat in [0.06645, 0.06647]",
      close((1 - 0.8671) / 2, 0.06645, 1e-9) and close((1 - 0.8006) / 3, 0.06647, 5e-6))
hw = math.sqrt(math.log(2 / 0.05) / (2 * 19900))
print("Hoeffding half-width 19900:", hw)
check("Hoeffding half-width with 19900 independent pairs at delta=0.05 is 0.0096", close(hw, 0.0096, 5e-5))
check("so tau floor would move by 2*0.0096 = 0.019", close(2 * hw, 0.019, 5e-4))
hw200 = math.sqrt(math.log(40) / (2 * 200))
check("with 200 independent units the half-width is 0.096", close(hw200, 0.096, 5e-4))

# ============================================================ Section 9 sample sizes
z = stats.norm.isf(0.025)
nnorm = math.ceil(z**2 * 0.25 / 0.02**2)
check("normal-approximation n for +-0.02 at 95% (p=1/2) is 2401", nnorm == 2401)


def cp_halfwidth_ok(nn, eps=0.02):
    x = nn // 2
    a, b = cp(x, nn)
    ph = x / nn
    return max(ph - a, b - ph) <= eps


ncp = 2000
while not cp_halfwidth_ok(ncp):
    ncp += 1
print("CP n:", ncp)
check("smallest n whose CP interval at phat about 1/2 is within +-0.02 is 2449", ncp == 2449)
# rare event: p = 0.001, relative 20 percent, eps = 0.0002
nn_norm_rare = math.ceil(z**2 * 0.001 * 0.999 / 0.0002**2)
nn_hoef_rare = math.ceil(math.log(40) / (2 * 0.0002**2))
nn_chern_rare = math.ceil(3 * math.log(40) / (0.001 * 0.2**2))
print("rare n:", nn_norm_rare, nn_hoef_rare, nn_chern_rare)
check("rare event n normal = 95,941", nn_norm_rare == 95941)
check("rare event n Hoeffding = 46,110,994", nn_hoef_rare == 46110994)
check("rare event n multiplicative Chernoff = 276,666", nn_chern_rare == 276666)
nn_bern_rare = math.ceil(math.log(40) * (2 * 0.001 * 0.999 + 2 * 0.0002 / 3) / 0.0002**2)
print("rare n Bernstein:", nn_bern_rare)
check("rare event n Bernstein = 196,556", nn_bern_rare == 196556)
check("Hoeffding rare n is 480 times the normal n", close(nn_hoef_rare / nn_norm_rare, 480.6, 0.1))
check("CP 2449 is 2 percent above 2401", close(2449 / 2401, 1.02, 0.001))
check("Chernoff rare mu = 3 ln 40 / 0.04 = 276.7", close(3 * math.log(40) / 0.04, 276.7, 0.05))
# exercise 12.6
check("Exercise: Hoeffding n for eps=0.05 delta=0.01 is 1060", n_hoef(1, 0.05, 0.01) == 1060)

# ============================================================ exercises
check("Ex 12.1: Markov 2/10 = 0.2, exponential mean 2 tail e^-5 = 0.00674", close(math.exp(-5), 0.00674, 5e-6))
check("Ex 12.2: Chebyshev 4/36 = 0.111, Gaussian 2Q(3) = 0.0027", close(4 / 36, 0.1111, 5e-5) and close(2 * Q(3), 0.0027, 5e-5))
check("Ex 12.3: 0/600 upper 1-0.05^(1/600) = 0.00498, 3/600 = 0.005",
      close(1 - 0.05 ** (1 / 600), 0.004980, 5e-6))
check("Ex 12.5: e^-2 = 0.1353 vs Q(2) = 0.02275", close(math.exp(-2), 0.1353, 5e-5) and close(Q(2), 0.02275, 5e-6))
check("Ex 12.8: sqrt(2 ln 100) = 3.035, 100 Q(3) = 0.135, exact 1-(1-Q(3))^100 = 0.1264",
      close(math.sqrt(2 * math.log(100)), 3.035, 5e-4) and close(100 * Q(3), 0.135, 5e-4)
      and close(1 - (1 - Q(3)) ** 100, 0.1264, 5e-5))
check("Ex 12.8: E max of 100 = 2.508", close(emax(100), 2.508, 5e-4))
hb9 = bern(10000, 0.001, 0.001)
print("Ex 12.9 Bernstein:", hb9)
check("Ex 12.9: Hoeffding e^-0.02 = 0.980, Bernstein 0.0235",
      close(math.exp(-0.02), 0.980, 5e-4) and close(hb9, 0.0235, 5e-5))
ex9 = stats.binom(10000, 0.001).sf(19)
print("Ex 12.9 exact:", ex9)
check("Ex 12.9: exact Pr[S>=20] = 0.00344", close(ex9, 0.00344, 5e-6))

check("answer ratios: 0.2/e^-5 = 29.7, 0.111/0.0027 = 41, 0.005/0.00498 = 1.004, e^-2/Q(2) = 5.95, 0.25/0.000999 = 250",
      close(0.2 / math.exp(-5), 29.7, 0.05) and close((4 / 36) / (2 * Q(3)), 41.2, 0.1)
      and close(0.005 / (1 - 0.05 ** (1 / 600)), 1.004, 5e-4) and close(math.exp(-2) / Q(2), 5.95, 0.005)
      and close(0.25 / 0.000999, 250, 0.3))
check("Ex 12.6: ln(200)/0.005 = 1059.7", close(math.log(200) / 0.005, 1059.7, 0.05))

# ---------------------------------------------------------------- hypothesis-discipline additions
# rule of three is conservative at every n: 1 - 0.05^(1/n) < 2.996/n < 3/n
check("rule of three conservative for n = 1..5000", all(1 - 0.05 ** (1 / n) < -math.log(0.05) / n < 3 / n for n in range(1, 5001)))
check("-ln 0.05 = 2.996", close(-math.log(0.05), 2.996, 5e-4))
# maximum tail bound needs u >= 0: n exp(-(a+u)^2/2) <= exp(-u^2/2) for u >= 0, fails for u < 0
a40 = math.sqrt(2 * math.log(40))
check("max tail: holds at u = 0, 0.5, 2 for n = 40", all(40 * math.exp(-(a40 + u) ** 2 / 2) <= math.exp(-u ** 2 / 2) + 1e-15 for u in [0, 0.5, 2]))
check("max tail: inequality fails at u = -1 for n = 40", 40 * math.exp(-(a40 - 1) ** 2 / 2) > math.exp(-1 / 2))
# Slepian direction: equicorrelated Gaussians (rho = 0.5) have a smaller expected max than independent ones
rgS = np.random.default_rng(1201)
Zi = rgS.standard_normal((20000, 40))
Zc = np.sqrt(0.5) * rgS.standard_normal((20000, 1)) + np.sqrt(0.5) * rgS.standard_normal((20000, 40))
check(f"n=40: E max independent {Zi.max(1).mean():.3f} > E max equicorrelated rho=0.5 {Zc.max(1).mean():.3f}", Zi.max(1).mean() > Zc.max(1).mean() + 0.3)
# heavy tail: the three-point law attains Chebyshev exactly at its threshold (k=10)
k10 = 10
check("three-point law with k=10: variance 1, Pr[|X|>=10] = 1/100 = Chebyshev bound", close(2 * k10 ** 2 / (2 * k10 ** 2), 1, 1e-12) and close(2 / (2 * k10 ** 2), 1 / k10 ** 2, 1e-15))

npass = sum(results)
print(f"{npass}/{len(results)} PASS")
sys.exit(0 if npass == len(results) else 1)
