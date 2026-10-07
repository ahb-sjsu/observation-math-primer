"""Checks for every number in chapter 9 (statistics for registration-first research).

Run on Atlas:  python3 ch09_examples.py
Prints PASS/FAIL per claim, exits nonzero on any FAIL, and prints k/n PASS.
Uses the standard library and numpy only.
"""
import hashlib
import itertools
import math
import sys
from statistics import NormalDist

import numpy as np

Z = NormalDist()
Phi = Z.cdf
results = []


def check(name, ok, detail=""):
    ok = bool(ok)
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + ((" | " + str(detail)) if detail else ""))


def close(a, b, tol):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Section 9.1  Paired scores, effect sizes and standard errors
A = np.array([0.82, 0.75, 0.91, 0.68, 0.88, 0.79, 0.85, 0.72])
B = np.array([0.78, 0.74, 0.86, 0.65, 0.83, 0.80, 0.81, 0.70])
d = A - B
n = len(d)
check("mean of A is 0.800", close(A.mean(), 0.800, 1e-12))
check("mean of B is 0.77125", close(B.mean(), 0.77125, 1e-12))
check("differences in hundredths are 4,1,5,3,5,-1,4,2",
      np.allclose(np.round(d * 100), [4, 1, 5, 3, 5, -1, 4, 2]))
dbar = d.mean()
check("mean difference 0.02875", close(dbar, 0.02875, 1e-12))
ss = ((d - dbar) ** 2).sum()
check("sum of squared deviations 0.0030875", close(ss, 0.0030875, 1e-12))
sd = d.std(ddof=1)
check("sd of differences 0.0210", close(sd, 0.0210, 5e-5), sd)
se_pair = sd / math.sqrt(n)
check("paired SE 0.00743", close(se_pair, 0.00743, 5e-6), se_pair)
t_pair = dbar / se_pair
check("paired t = 3.87", close(t_pair, 3.87, 5e-3), t_pair)
dz = dbar / sd
check("standardized paired effect d_z = 1.37", close(dz, 1.37, 5e-3), dz)
sA, sB = A.std(ddof=1), B.std(ddof=1)
se_unpair = math.sqrt(sA ** 2 / n + sB ** 2 / n)
check("sd of A 0.0800", close(sA, 0.0800, 5e-5), sA)
check("sd of B 0.0702", close(sB, 0.0702, 5e-5), sB)
check("unpaired SE 0.0376", close(se_unpair, 0.0376, 5e-5), se_unpair)
check("unpaired SE about five times paired SE", close(se_unpair / se_pair, 5.1, 0.05),
      se_unpair / se_pair)
sp = math.sqrt((sA ** 2 + sB ** 2) / 2)
check("pooled sd 0.0753", close(sp, 0.0753, 5e-5), sp)
check("Cohen d (unpaired) 0.38", close(dbar / sp, 0.38, 5e-3), dbar / sp)
r = np.corrcoef(A, B)[0, 1]
check("correlation of A and B 0.97", close(r, 0.97, 5e-3), r)
# variance of a difference identity
lhs = d.var(ddof=1)
rhs = sA ** 2 + sB ** 2 - 2 * np.cov(A, B, ddof=1)[0, 1]
check("Var(A-B) = VarA + VarB - 2Cov", close(lhs, rhs, 1e-15))

# ---------------------------------------------------------------------------
# Section 9.2  Sign-flip permutation test and bootstrap
h = np.round(d * 100).astype(int)
obs = abs(h.sum())
count = 0
for signs in itertools.product([1, -1], repeat=n):
    s = sum(si * abs(hi) for si, hi in zip(signs, h))
    if abs(s) >= obs:
        count += 1
check("observed sum 23 hundredths, |d| total 25", obs == 23 and np.abs(h).sum() == 25)
check("sign-flip: 6 of 256 assignments at least as extreme", count == 6, count)
check("exact two-sided p = 6/256 = 0.0234", close(count / 256, 0.0234, 5e-5), count / 256)
check("smallest attainable two-sided p with n=8 is 2/256 = 0.0078", close(2 / 256, 0.0078, 5e-5))

rng = np.random.default_rng(20261006)
Bboot = 10000
idx = rng.integers(0, n, size=(Bboot, n))
boot_means = d[idx].mean(axis=1)
lo, hi = np.percentile(boot_means, [2.5, 97.5])
print("bootstrap percentile interval", lo, hi)
check("paired bootstrap 95% interval lower end 0.015", close(lo, 0.015, 5e-4), lo)
check("paired bootstrap 95% interval upper end 0.041", close(hi, 0.041, 5e-4), hi)
boot_se = boot_means.std(ddof=1)
check("bootstrap SE 0.0069 (plug-in, below 0.00743)", close(boot_se, 0.0069, 1e-4), boot_se)
check("plug-in SE sqrt((n-1)/n) * 0.00743 = 0.00695",
      close(math.sqrt((n - 1) / n) * se_pair, 0.00695, 5e-5))

# ---------------------------------------------------------------------------
# Section 9.3  Multiple comparisons
p = np.array([0.001, 0.008, 0.012, 0.03, 0.04, 0.20])
m, alpha = len(p), 0.05
bonf = int((p <= alpha / m).sum())
check("Bonferroni threshold 0.00833", close(alpha / m, 0.00833, 5e-6))
check("Bonferroni rejects 2", bonf == 2)
ps = np.sort(p)
holm = 0
for i, pi in enumerate(ps):
    if pi <= alpha / (m - i):
        holm += 1
    else:
        break
check("Holm thresholds 0.00833 0.01 0.0125 0.0167", np.allclose(
    [alpha / (m - i) for i in range(4)], [0.008333, 0.01, 0.0125, 0.016667], atol=1e-6))
check("Holm rejects 3", holm == 3)
bh_k = max([k for k in range(1, m + 1) if ps[k - 1] <= k * alpha / m] + [0])
check("BH thresholds k*alpha/m", np.allclose([k * alpha / m for k in range(1, 7)],
                                             [0.008333, 0.016667, 0.025, 0.033333, 0.041667, 0.05], atol=1e-6))
check("BH rejects 5", bh_k == 5)
check("40 null tests: P(at least one) = 0.87", close(1 - 0.95 ** 40, 0.87, 5e-3))
# Holm adjusted p for the third hypothesis
holm_adj = [min(1, max((m - j) * ps[j] for j in range(i + 1))) for i in range(m)]
check("Holm adjusted p-values 0.006 0.040 0.048 0.090 0.090 0.200",
      np.allclose(holm_adj, [0.006, 0.040, 0.048, 0.090, 0.090, 0.200], atol=1e-9), holm_adj)

# ---------------------------------------------------------------------------
# Section 9.4  Power, minimum detectable effect, bars in noise units
za = Z.inv_cdf(0.975)
zb = Z.inv_cdf(0.80)
check("z_0.975 = 1.960, z_0.80 = 0.842", close(za, 1.960, 5e-4) and close(zb, 0.842, 5e-4))
mde8 = (za + zb) * 0.021 / math.sqrt(8)
check("MDE at n=8, sd 0.021: 0.0208", close(mde8, 0.0208, 5e-5), mde8)
nreq = ((za + zb) * 0.021 / 0.01) ** 2
check("n to detect 0.01 at 80%: 34.6 -> 35", close(nreq, 34.6, 0.05) and math.ceil(nreq) == 35, nreq)
pw = Phi(0.01 * math.sqrt(8) / 0.021 - za)
check("power at n=8 for true 0.01 is 0.27", close(pw, 0.27, 5e-3), pw)


def power_bar(delta, k, n):
    return Phi((delta - k) * math.sqrt(n))


vals = {(dl, nn): power_bar(dl, 0.5, nn) for dl in (0.45, 0.55) for nn in (40, 640)}
print("bar powers", vals)
check("bar 0.5, true 0.45, n=40: power 0.38", close(vals[(0.45, 40)], 0.38, 5e-3))
check("bar 0.5, true 0.45, n=640: power 0.10", close(vals[(0.45, 640)], 0.10, 5e-3))
check("bar 0.5, true 0.55, n=40: power 0.62", close(vals[(0.55, 40)], 0.62, 5e-3))
check("bar 0.5, true 0.55, n=640: power 0.90", close(vals[(0.55, 640)], 0.90, 5e-3))
check("bar exactly at true effect: power 0.5 at any n", close(power_bar(0.5, 0.5, 999), 0.5, 1e-12))
# SE-unit bar: estimate beyond 2 SE, true 0.45: power rises with n
pse = [Phi(0.45 * math.sqrt(nn) - 2) for nn in (40, 640)]
check("2-SE bar, true 0.45: power 0.80 at n=40, 1.00 at n=640",
      close(pse[0], 0.80, 5e-3) and pse[1] > 0.9999, pse)
# absolute tolerance vs SE
pa = 2 * Phi(0.06 / 0.101) - 1
check("tolerance 0.06 with SE 0.101 passes a correct pipeline 45% of the time", close(pa, 0.45, 5e-3), pa)
p4 = 2 * Phi(4) - 1
check("4-SE tolerance passes 0.99994", close(p4, 0.99994, 5e-6), p4)
check("4 SE of 0.101 is 0.404", close(4 * 0.101, 0.404, 1e-12))
# house floor 1.3x, protocol registry 052 miss arithmetic
check("052: 0.1316 vs bar 0.1331 is 1.1% short", close((0.1331 - 0.1316) / 0.1331, 0.011, 5e-4))

# ---------------------------------------------------------------------------
# Section 9.5  Equivalence testing (TOST)
est, se, Delta = 0.003, 0.004, 0.01
z1 = (est + Delta) / se
z2 = (est - Delta) / se
p_tost = max(1 - Phi(z1), Phi(z2))
check("TOST z1 = 3.25, z2 = -1.75", close(z1, 3.25, 1e-12) and close(z2, -1.75, 1e-12))
check("TOST p = 0.040", close(p_tost, 0.040, 5e-4), p_tost)
z90 = Z.inv_cdf(0.95)
ci = (est - z90 * se, est + z90 * se)
check("90% CI (-0.0036, 0.0096) inside (-0.01, 0.01)",
      close(ci[0], -0.0036, 5e-5) and close(ci[1], 0.0096, 5e-5) and -Delta < ci[0] and ci[1] < Delta, ci)
est2, se2 = 0.01, 0.008
p2 = 2 * (1 - Phi(est2 / se2))
check("non-significant case p = 0.21", close(p2, 0.21, 5e-3), p2)
ci2 = (est2 - z90 * se2, est2 + z90 * se2)
check("its 90% CI (-0.0032, 0.0232) leaves the band", close(ci2[0], -0.0032, 5e-5)
      and close(ci2[1], 0.0232, 5e-5) and ci2[1] > Delta, ci2)
p_tost2 = max(1 - Phi((est2 + Delta) / se2), Phi((est2 - Delta) / se2))
check("its TOST p = 0.50", close(p_tost2, 0.50, 5e-3), p_tost2)

# ---------------------------------------------------------------------------
# Section 9.6  Clustering
mbar, rho = 50, 0.1
de = 1 + (mbar - 1) * 1.0 * rho
check("Moulton factor, covariate constant in cluster: 5.9", close(de, 5.9, 1e-12))
check("SE inflation sqrt(5.9) = 2.43", close(math.sqrt(de), 2.43, 5e-3))
check("effective sample 1000/5.9 = 169", int(1000 / de) == 169)
check("covariate uncorrelated within cluster: factor 1", 1 + (mbar - 1) * 0.0 * rho == 1)
check("rho_x = 0.3: factor 2.47", close(1 + (mbar - 1) * 0.3 * rho, 2.47, 1e-12))
# Simulation: cluster-level covariate, naive vs cluster bootstrap SE
rng = np.random.default_rng(7)
G, M = 20, 50
reps = 2000
slopes = []
naive_se = []
for _ in range(reps):
    xg = rng.normal(size=G)
    ug = rng.normal(scale=math.sqrt(rho), size=G)
    x = np.repeat(xg, M)
    y = 0.0 * x + np.repeat(ug, M) + rng.normal(scale=math.sqrt(1 - rho), size=G * M)
    xc = x - x.mean()
    b = (xc * (y - y.mean())).sum() / (xc ** 2).sum()
    res = y - y.mean() - b * xc
    naive_se.append(math.sqrt((res ** 2).sum() / (G * M - 2) / (xc ** 2).sum()))
    slopes.append(b)
true_sd = np.std(slopes, ddof=1)
ratio = true_sd / np.mean(naive_se)
print("simulated SE ratio", ratio)
check("simulation: true sd of slope / naive SE near sqrt(5.9)=2.43 (within 0.15)",
      close(ratio, math.sqrt(de), 0.15), ratio)
naive_rej = np.mean(np.abs(np.array(slopes) / np.array(naive_se)) > 1.96)
print("naive rejection rate", naive_rej)
check("simulation: naive 5% test rejects a true null 0.43 of the time",
      close(naive_rej, 0.43, 0.005), naive_rej)
check("theory: 2 Phi(-1.96/2.43) = 0.42", close(2 * Phi(-1.96 / math.sqrt(de)), 0.42, 5e-3))

# ---------------------------------------------------------------------------
# Section 9.7  Forking paths, attenuation
check("best of 5 null analyses: 0.226", close(1 - 0.95 ** 5, 0.226, 5e-4))
check("best of 20 null analyses: 0.642", close(1 - 0.95 ** 20, 0.642, 5e-4))
lam = 1.0 / (1.0 + 0.5)
check("reliability 2/3, slope 2 -> 1.333", close(lam, 2 / 3, 1e-12) and close(2 * lam, 1.3333, 5e-5))
rng = np.random.default_rng(11)
xs = rng.normal(size=200000)
ys = 2 * xs + rng.normal(size=200000)
w = xs + rng.normal(scale=math.sqrt(0.5), size=200000)
bw = np.cov(w, ys)[0, 1] / w.var(ddof=1)
check("simulation: OLS slope on noisy x near 1.333", close(bw, 1.3333, 0.02), bw)
# attenuated scale makes a bar easier
check("effect 0.4 fails bar 0.5x1.0 but passes bar 0.5x0.6", 0.4 < 0.5 * 1.0 and 0.4 >= 0.5 * 0.6)

# ---------------------------------------------------------------------------
# Section 9.8  Sealing
reg = b"demo-001: primary = mean paired difference; bar = 2 SE; n = 35; alpha = 0.05\n"
h1 = hashlib.sha256(reg).hexdigest()
reg2 = reg.replace(b"n = 35", b"n = 36")
h2 = hashlib.sha256(reg2).hexdigest()
print("sha256 reg :", h1)
print("sha256 reg2:", h2)
check("sha256 of the registration begins 34b0bff6623a63cb", h1.startswith("34b0bff6623a63cb"))
check("changed registration hash begins 40fbf8c7c7886ca5", h2.startswith("40fbf8c7c7886ca5"))
diffbits = bin(int(h1, 16) ^ int(h2, 16)).count("1")
print("differing bits", diffbits)
check("one-character change flips 135 of the 256 bits", diffbits == 135, diffbits)
check("file drawer: 20 null studies at 0.05 give 1 expected positive", close(20 * 0.05, 1.0, 1e-12))
# registry accounting: IDs 001..091 with 029, 030 never-run
check("pigeonhole: 2^256 digests", 2 ** 256 > 10 ** 77)

# ---------------------------------------------------------------------------
# Extra values quoted in the text and figure
check("z sum 1.960 + 0.842 = 2.802", close(za + zb, 2.802, 5e-4))
check("plug-in factor sqrt(7/8) = 0.935", close(math.sqrt(7 / 8), 0.935, 5e-4))
check("figure dots 0.376 0.624 0.103 0.897", close(vals[(0.45, 40)], 0.376, 5e-4) and close(vals[(0.55, 40)], 0.624, 5e-4)
      and close(vals[(0.45, 640)], 0.103, 5e-4) and close(vals[(0.55, 640)], 0.897, 5e-4))
xs_ = np.linspace(-4, 4, 801)
logi = 1 / (1 + np.exp(-1.702 * xs_))
check("logistic approximation to Phi within 0.01", max(abs(logi[i] - Phi(x)) for i, x in enumerate(xs_)) < 0.01)
check("TOST one-sided first tail 0.0006", close(1 - Phi(3.25), 0.0006, 5e-5))

# ---------------------------------------------------------------------------
# Exercise answers
check("Ex 9.2: 1-0.95^10 = 0.401, level 0.005", close(1 - 0.95 ** 10, 0.401, 5e-4) and close(0.05 / 10, 0.005, 1e-15))
check("Ex 9.4: smallest sign-flip p with n=4 is 2/16 = 0.125", close(2 / 16, 0.125, 1e-15))
pe = np.sort(np.array([0.004, 0.01, 0.02, 0.03, 0.30]))
me = 5
bonf_e = int((pe <= 0.05 / me).sum())
holm_e = 0
for i, pi in enumerate(pe):
    if pi <= 0.05 / (me - i):
        holm_e += 1
    else:
        break
bh_e = max([k for k in range(1, me + 1) if pe[k - 1] <= k * 0.05 / me] + [0])
check("Ex 9.5: Bonferroni 2, Holm 2, BH 4", (bonf_e, holm_e, bh_e) == (2, 2, 4), (bonf_e, holm_e, bh_e))
z1e, z2e = (-0.004 + 0.01) / 0.003, (-0.004 - 0.01) / 0.003
pte = max(1 - Phi(z1e), Phi(z2e))
cie = (-0.004 - z90 * 0.003, -0.004 + z90 * 0.003)
check("Ex 9.6: z1 = 2, z2 = -4.67, p = 0.023, CI (-0.0089, 0.0009)",
      close(z1e, 2, 1e-12) and close(z2e, -4.67, 5e-3) and close(pte, 0.023, 5e-4)
      and close(cie[0], -0.0089, 5e-5) and close(cie[1], 0.0009, 5e-5), (z1e, z2e, pte, cie))
dee = 1 + 19 * 0.05
check("Ex 9.7: factor 1.95, SE x1.40, effective 308", close(dee, 1.95, 1e-12) and close(math.sqrt(dee), 1.40, 5e-3)
      and round(600 / dee) == 308)
check("Ex 9.8: 0.8 x 0.75 = 0.6 < 0.7 < 0.8", close(0.8 * 0.75, 0.6, 1e-12) and 0.6 < 0.7 < 0.8)
ne = ((za + zb) * 1 / 0.25) ** 2
check("Ex 9.9: n = 125.6 -> 126", close(ne, 125.6, 0.05) and math.ceil(ne) == 126, ne)
pbe = Phi((0.25 - 0.3) * math.sqrt(126))
check("Ex 9.9: bar 0.3 pass probability 0.29", close(pbe, 0.29, 5e-3), pbe)


def holm_mask(p, alpha):
    order = np.argsort(p)
    m = len(p)
    rej = np.zeros(m, bool)
    for i, j in enumerate(order):
        if p[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej


def bh_mask(p, alpha):
    order = np.argsort(p)
    m = len(p)
    ok = [k for k in range(1, m + 1) if p[order[k - 1]] <= k * alpha / m]
    rej = np.zeros(m, bool)
    if ok:
        rej[order[:max(ok)]] = True
    return rej


pu = np.array([0.03, 0.001, 0.20, 0.012, 0.04, 0.008])
check("Ex 9.11: holm/bh masks on shuffled example give 3 and 5",
      holm_mask(pu, 0.05).sum() == 3 and bh_mask(pu, 0.05).sum() == 5 and not bh_mask(pu, 0.05)[2])

# ---------------------------------------------------------------------------
# Figure data. Each list is printed (to paste into the chapter) and, when the
# chapter file sits next to this script, compared with the coordinates pasted
# there after a "% data:NAME" marker.
import os
import re

fig = {}
# F1 sampling distribution: CI endpoints and the zero line
ci_lo, ci_hi = dbar - 1.96 * se_pair, dbar + 1.96 * se_pair
fig["ci"] = [(round(ci_lo, 4), 0.0), (round(ci_hi, 4), 0.0)]
check("normal 95% interval for mean difference (0.0142, 0.0433)",
      close(ci_lo, 0.0142, 5e-5) and close(ci_hi, 0.0433, 5e-5), (ci_lo, ci_hi))
# F2 sign-flip null distribution of the sum (hundredths)
dist = {}
for signs in itertools.product([1, -1], repeat=n):
    s_ = sum(si * abs(hi_) for si, hi_ in zip(signs, h))
    dist[s_] = dist.get(s_, 0) + 1
fig["perm"] = [(k_, dist[k_]) for k_ in sorted(dist)]
check("sign-flip null has 256 assignments, symmetric, 6 at |sum| >= 23",
      sum(dist.values()) == 256 and all(dist[k_] == dist[-k_] for k_ in dist)
      and sum(c_ for k_, c_ in dist.items() if abs(k_) >= 23) == 6)
check("null is flat at 19 for sums -5..5", all(dist[k_] == 19 for k_ in (-5, -3, -1, 1, 3, 5)))
# F3 BH
fig["bhp"] = [(i + 1, float(ps[i])) for i in range(m)]
# F4 power curves, delta in [0, 0.03]
deltas = [round(0.0025 * i, 4) for i in range(13)]
for nn in (8, 35):
    fig[f"pow{nn}"] = [(dl, round(Phi(dl * math.sqrt(nn) / 0.021 - za), 3)) for dl in deltas]
check("power at n=35 for 0.01 is 0.80", close(Phi(0.01 * math.sqrt(35) / 0.021 - za), 0.80, 5e-3))
fig["powpts"] = [(0.01, round(pw, 3)), (round(mde8, 4), 0.8)]
# F5 effect-size bar step, exact Phi
dgrid = [round(0.3 + 0.01 * i, 2) for i in range(41)]
for nn in (40, 640):
    fig[f"bar{nn}"] = [(dl, round(Phi((dl - 0.5) * math.sqrt(nn)), 3)) for dl in dgrid]
fig["bardots"] = [(0.45, 0.376), (0.55, 0.624), (0.45, 0.103), (0.55, 0.897)]
# F6 TOST intervals: (estimate, lo, hi) for three studies, plotted as y = 3,2,1
est3, se3 = 0.03, 0.004
ci3 = (est3 - z90 * se3, est3 + z90 * se3)
check("present case 90% CI (0.0234, 0.0366) wholly outside the band",
      close(ci3[0], 0.0234, 5e-5) and close(ci3[1], 0.0366, 5e-5) and ci3[0] > Delta, ci3)
fig["tost"] = [(round(est, 4), 3), (round(est2, 4), 2), (round(est3, 4), 1)]
fig["tostlo"] = [(round(ci[0], 4), 3), (round(ci2[0], 4), 2), (round(ci3[0], 4), 1)]
fig["tosthi"] = [(round(ci[1], 4), 3), (round(ci2[1], 4), 2), (round(ci3[1], 4), 1)]
# F7 Moulton factor at the example point
fig["moulton"] = [(50, round(1 + 49 * 0.1, 2)), (50, round(1 + 49 * 0.03, 2))]
check("Moulton factor at m=50 with rho_x rho_u = 0.03 is 2.47", close(1 + 49 * 0.03, 2.47, 1e-12))
# F8 attenuation scatter, 40 points
rng = np.random.default_rng(2026)
xa = rng.normal(size=40)
ya = 2 * xa + rng.normal(size=40)
wa = xa + rng.normal(scale=math.sqrt(0.5), size=40)
fig["attx"] = [(round(a_, 2), round(b_, 2)) for a_, b_ in zip(xa, ya)]
fig["attw"] = [(round(a_, 2), round(b_, 2)) for a_, b_ in zip(wa, ya)]
bx = np.polyfit(xa, ya, 1)[0]
bw40 = np.polyfit(wa, ya, 1)[0]
print("attenuation sample slopes", bx, bw40)
check("40-point sample: slope on x above slope on w", bx > bw40, (bx, bw40))
check("40-point sample slopes 2.37 and 1.36", close(bx, 2.37, 5e-3) and close(bw40, 1.36, 5e-3), (bx, bw40))


def fmt(pts):
    return " ".join(f"({a},{b})" for a, b in pts)


for k_, v_ in fig.items():
    print(f"DATA {k_}: {fmt(v_)}")

texf = "ch09_registration_statistics.tex"
if os.path.exists(texf):
    tex = open(texf, encoding="utf8").read()
    found = re.findall(r"coordinates\s*\{([^}]*)\}\s*;\s*%\s*data:(\w+)", tex)
    check("figure data blocks found in chapter (>= 12)", len(found) >= 12, len(found))
    for body, name in found:
        pts = [tuple(float(t) for t in p.split(",")) for p in re.findall(r"\(([^)]*)\)", body)]
        ref = fig.get(name)
        ok = ref is not None and len(ref) == len(pts) and all(
            abs(a - c) <= 5e-4 * max(1, abs(c)) and abs(b - d) <= 5e-4 * max(1, abs(d))
            for (a, b), (c, d) in zip(pts, ref))
        check(f"figure coordinates '{name}' match the computed values", ok)
else:
    print("NOTE chapter file not present, figure coordinates not compared")

print(f"\n{sum(results)}/{len(results)} PASS")
sys.exit(0 if all(results) else 1)
