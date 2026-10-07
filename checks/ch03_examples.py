"""Numerical checks for chapter 3 (information theory and rate-distortion).

Every number quoted in a worked example, exercise answer or figure of
chapters/ch03_information_theory.tex and answers/ch03.tex is checked here.
Run on Atlas:  python3 ch03_examples.py
"""
import sys

import numpy as np
import sympy as sp

RESULTS = []


def check(desc, cond):
    ok = bool(cond)
    RESULTS.append(ok)
    print(("PASS " if ok else "FAIL ") + desc)


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


LN2 = np.log(2.0)


def H2(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def hb(p):
    return H2([p, 1 - p])


def revwf(gam, D):
    """Reverse water-filling: returns theta, per-component distortion, rate in bits."""
    gam = np.asarray(gam, dtype=float)
    lo, hi = 0.0, gam.max()
    for _ in range(200):
        th = 0.5 * (lo + hi)
        if np.minimum(gam, th).sum() > D:
            hi = th
        else:
            lo = th
    th = 0.5 * (lo + hi)
    Di = np.minimum(gam, th)
    R = float(sum(0.5 * np.log2(g / th) for g in gam if g > th))
    return th, Di, R


# ---------------------------------------------------------------- section 1
P = np.array([[3, 1], [1, 3]]) / 8.0
HX = H2(P.sum(1))
HXY = H2(P.ravel())
HYgX = HXY - HX
check("ex entropy: H(X)=1 bit", close(HX, 1.0))
check("ex entropy: H(X,Y)=1.8113 bits", close(HXY, 1.8113))
check("ex entropy: H(Y|X)=h(1/4)=0.8113 bits", close(HYgX, 0.8113) and close(hb(0.25), 0.8113))
# exact: H(X,Y) = 1 + h(1/4)
check("ex entropy: H(X,Y)=1+h(1/4) exactly (sympy)",
      sp.simplify(-2 * sp.Rational(3, 8) * sp.log(sp.Rational(3, 8), 2)
                  - 2 * sp.Rational(1, 8) * sp.log(sp.Rational(1, 8), 2)
                  - 1 - (-sp.Rational(1, 4) * sp.log(sp.Rational(1, 4), 2)
                         - sp.Rational(3, 4) * sp.log(sp.Rational(3, 4), 2))) == 0)

# ---------------------------------------------------------------- section 2
p, q = np.array([0.9, 0.1]), np.array([0.5, 0.5])
Dpq = float((p * np.log2(p / q)).sum())
Dqp = float((q * np.log2(q / p)).sum())
check("ex KL: D(p||q)=0.531 bits", close(Dpq, 0.531))
check("ex KL: D(q||p)=0.737 bits", close(Dqp, 0.737))
IXY = 1 - hb(0.25)
check("ex MI: I(X;Y)=0.1887 bits", close(IXY, 0.1887))
# MI as KL from product
prod = np.outer(P.sum(1), P.sum(0))
check("ex MI: I = D(p_XY||p_X p_Y)", close((P * np.log2(P / prod)).sum(), IXY, 1e-12))
# cascade of two BSC(1/4) is BSC(3/8)
eps = 2 * 0.25 * 0.75
check("ex DPI: cascade crossover 3/8", close(eps, 0.375, 1e-12))
IXZ = 1 - hb(eps)
check("ex DPI: I(X;Z)=0.0456 bits", close(IXZ, 0.0456))
check("ex DPI: I(X;Z) <= I(X;Y)", IXZ <= IXY)
check("ex DPI: h(3/8)=0.9544", close(hb(0.375), 0.9544))

# ---------------------------------------------------------------- section 3
hneg = 0.5 * np.log2(2 * np.pi * np.e * 0.01)
check("ex diff entropy: sigma^2=0.01 gives -1.275 bits", close(hneg, -1.275))
S = np.array([[2.0, 1.0], [1.0, 2.0]])
h_nats = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * S))
check("ex gauss h: det Sigma = 3", close(np.linalg.det(S), 3.0, 1e-12))
check("ex gauss h: ln(2 pi e)=2.8379", close(np.log(2 * np.pi * np.e), 2.8379))
check("ex gauss h: h=3.3872 nats", close(h_nats, 3.3872))
check("ex gauss h: h=4.8867 bits", close(h_nats / LN2, 4.8867))
I_pair = 0.5 * np.log2(S[0, 0] * S[1, 1] / np.linalg.det(S))
check("ex gauss MI: I(x1;x2)=0.2075 bits (det ratio)", close(I_pair, 0.2075))
check("ex gauss MI: equals -1/2 log2(1-rho^2), rho=1/2", close(-0.5 * np.log2(1 - 0.25), I_pair, 1e-12))
check("ex gauss MI: rho=0.8 gives 0.5108 nats", close(-0.5 * np.log(1 - 0.64), 0.5108))
check("ex gauss MI: rho=0.8 gives 0.7370 bits", close(-0.5 * np.log2(1 - 0.64), 0.7370))
# conditional variance form
check("ex gauss MI: Var(x1|x2)=1.5", close(S[0, 0] - S[0, 1] ** 2 / S[1, 1], 1.5, 1e-12))

# ---------------------------------------------------------------- section 4
pp = np.array([0.5, 0.25, 0.25])
check("ex Huffman: H(1/2,1/4,1/4)=1.5 bits", close(H2(pp), 1.5, 1e-12))
check("ex Huffman: mean length 1.5", close((pp * np.array([1, 2, 2])).sum(), 1.5, 1e-12))
check("ex Huffman: Kraft sum = 1", close(2 ** -1 + 2 ** -2 + 2 ** -2, 1.0, 1e-12))
check("ex binary R(D): 1-h(0.1)=0.531 bits", close(1 - hb(0.1), 0.531))
check("ex binary R(D): h(0.1)=0.469", close(hb(0.1), 0.469))

# ---------------------------------------------------------------- section 5
check("ex scalar RD: sigma^2=1, D=1/4 gives 1 bit", close(0.5 * np.log2(1 / 0.25), 1.0, 1e-12))
check("ex scalar RD: D(R)=sigma^2 2^{-2R} at R=2 is 1/16", close(2.0 ** -4, 1 / 16, 1e-15))
gap = 0.5 * np.log2(np.pi * np.e / 6)
check("ex ECSQ gap: 1/2 log2(pi e/6)=0.2546 bits", close(gap, 0.2546, 5e-5))
check("ex ECSQ gap: 10 log10(pi e/6)=1.53 dB", close(10 * np.log10(np.pi * np.e / 6), 1.53, 5e-3))
gam = [4, 2, 1, 0.25]
th, Di, R = revwf(gam, 1.75)
check("ex/fig vector RD: theta=0.5 at D=1.75", close(th, 0.5, 1e-9))
check("ex/fig vector RD: D_i=(0.5,0.5,0.5,0.25)", np.allclose(Di, [0.5, 0.5, 0.5, 0.25], atol=1e-9))
check("ex/fig vector RD: R=3 bits", close(R, 3.0, 1e-9))
check("ex/fig vector RD: per-component bits 1.5,1,0.5,0",
      np.allclose([0.5 * np.log2(g / 0.5) for g in [4, 2, 1]], [1.5, 1.0, 0.5]))
check("ex vector RD: identity-reader D_max=7.25", close(sum(gam), 7.25, 1e-12))

# ---------------------------------------------------------------- section 6
Sx = np.array([[2.0, 1.0], [1.0, 2.0]])
Pc = np.diag([1.0, 0.0])


def whitened_spectrum(Sx, P):
    w, V = np.linalg.eigh(Sx)
    Sh = V @ np.diag(np.sqrt(w)) @ V.T
    return np.sort(np.linalg.eigvalsh(Sh @ P @ Sh))[::-1]


g1 = whitened_spectrum(Sx, Pc)
check("ex R_C: whitened spectrum (2,0)", np.allclose(g1, [2, 0], atol=1e-12))
check("ex R_C: tr(P Sigma_x)=2", close(np.trace(Pc @ Sx), 2.0, 1e-12))
check("ex R_C: R_C(0.5)=1 bit", close(0.5 * np.log2(2 / 0.5), 1.0, 1e-12))
thI, DiI, RI = revwf(np.linalg.eigvalsh(Sx), 0.5)
check("ex R_C: identity reader eigenvalues (3,1)", np.allclose(np.sort(np.linalg.eigvalsh(Sx)), [1, 3]))
check("ex R_C: identity reader theta=0.25", close(thI, 0.25, 1e-9))
check("ex R_C: identity reader R(0.5)=2.792 bits", close(RI, 2.792))
# the optimal Sigma for the consumer: error only along e1, Sigma* = Sigma_x - (Sigma_x e1)(..)/..*(1-D/2)
# check via max-det closed form X = Phi(Q~/theta) in whitened coords
w, V = np.linalg.eigh(Sx)
Sh = V @ np.diag(np.sqrt(w)) @ V.T
Qt = Sh @ Pc @ Sh
lam, E = np.linalg.eigh(Qt)
theta = 0.5
xdiag = np.array([1.0 if l <= theta else theta / l for l in lam])
X = E @ np.diag(xdiag) @ E.T
Sig = Sh @ X @ Sh
check("ex R_C: optimal Sigma has tr(P Sigma)=0.5", close(np.trace(Pc @ Sig), 0.5, 1e-9))
check("ex R_C: optimal Sigma has rate 1 bit",
      close(0.5 * np.log2(np.linalg.det(Sx) / np.linalg.det(Sig)), 1.0, 1e-9))
check("ex R_C: optimal Sigma = [[0.5,0.25],[0.25,1.625]]",
      np.allclose(Sig, [[0.5, 0.25], [0.25, 1.625]], atol=1e-9))
check("ex R_C: Sigma_x - Sigma* is PSD rank one",
      np.linalg.eigvalsh(Sx - Sig).min() > -1e-9 and np.linalg.matrix_rank(Sx - Sig, tol=1e-9) == 1)

# ---------------------------------------------------------------- section 7
s2, t2, D = 1.0, 1.0, 0.25
vXS = s2 * t2 / (s2 + t2)
check("ex side info: Var(X|S)=0.5", close(vXS, 0.5, 1e-12))
check("ex side info: R_{X|S}(0.25)=0.5 bit", close(0.5 * np.log2(vXS / D), 0.5, 1e-12))
check("ex side info: R(0.25)=1 bit without S", close(0.5 * np.log2(s2 / D), 1.0, 1e-12))
# Wyner-Ziv Gaussian test channel U = X + N achieving it: choose Var N so that Var(X|U,S)=D
# 1/D = 1/s2 + 1/t2 + 1/vN  -> vN
vN = 1.0 / (1 / D - 1 / s2 - 1 / t2)
check("ex WZ: auxiliary noise variance 0.5", close(vN, 0.5, 1e-12))
# I(X;U|S) = 1/2 log Var(X|S)/Var(X|U,S)
check("ex WZ: I(X;U|S)=0.5 bit", close(0.5 * np.log2(vXS / D), 0.5, 1e-12))
# I(X;U)-I(S;U)
SigXUS = np.array([[s2, s2, s2], [s2, s2 + vN, s2], [s2, s2, s2 + t2]])


def gmi(S, a, b):
    Sa = S[np.ix_(a, a)]
    Sb = S[np.ix_(b, b)]
    Sab = S[np.ix_(a + b, a + b)]
    return 0.5 * np.log2(np.linalg.det(Sa) * np.linalg.det(Sb) / np.linalg.det(Sab))


check("ex WZ: I(X;U)-I(S;U)=0.5 bit", close(gmi(SigXUS, [0], [1]) - gmi(SigXUS, [2], [1]), 0.5, 1e-9))

# ---------------------------------------------------------------- section 8
r = 0.5 * np.log2(s2 / D)
vXgMS = 1.0 / (1 / D + 1 / t2)
ell = 0.5 * np.log2(vXS / vXgMS)
check("ex leakage: Var(X|Xhat,S)=0.2", close(vXgMS, 0.2, 1e-12))
check("ex leakage: rate 1 bit", close(r, 1.0, 1e-12))
check("ex leakage: leakage 0.661 bits", close(ell, 0.661))
# I(Xhat;S) directly: Xhat ~ N(0, s2-D), S = Xhat + Zb + U
IhS = 0.5 * np.log2((s2 + t2) / (D + t2))
check("ex leakage: I(Xhat;S)=0.339 bits", close(IhS, 0.339))
check("ex leakage: ell = r - I(Xhat;S)", close(ell, r - IhS, 1e-12))
# Monte-Carlo-free joint covariance check: (X, Xhat, S)
vh = s2 - D
SigXhS = np.array([[s2, vh, s2], [vh, vh, vh], [s2, vh, s2 + t2]])
IXXh_given_S = gmi(SigXhS, [0], [1, 2]) - gmi(SigXhS, [0], [2])
check("ex leakage: I(X;Xhat|S) from covariance = 0.661", close(IXXh_given_S, ell, 1e-9))

# ---------------------------------------------------------------- section 9
check("ex refinement: D1=1/4 costs 1 bit", close(0.5 * np.log2(1 / 0.25), 1.0, 1e-12))
check("ex refinement: D2=1/16 costs 2 bits total", close(0.5 * np.log2(16), 2.0, 1e-12))
# two observers on Sigma_x = I_2
S1 = np.diag([0.25, 1.0])
S2 = np.diag([1.0, 0.25])
check("ex two observers: Sigma2* not <= Sigma1*", np.linalg.eigvalsh(S1 - S2).min() < 0)
check("ex two observers: R1(D1)=1 bit", close(0.5 * np.log2(1 / np.linalg.det(S1)), 1.0, 1e-12))
check("ex two observers: R2(D2)=1 bit", close(0.5 * np.log2(1 / np.linalg.det(S2)), 1.0, 1e-12))
Scirc = np.diag([0.25, 0.25])
check("ex two observers: total with stage-1-optimal base 2 bits",
      close(0.5 * np.log2(1 / np.linalg.det(Scirc)), 2.0, 1e-12))
check("ex two observers: rate loss 1 bit",
      close(0.5 * np.log2(np.linalg.det(S2) / np.linalg.det(Scirc)), 1.0, 1e-12))

# ---------------------------------------------------------------- exercises / answers
check("Ex3.1: H(1/2,1/4,1/8,1/8)=1.75 bits", close(H2([0.5, 0.25, 0.125, 0.125]), 1.75, 1e-12))
check("Ex3.3: h(Unif[0,1/2]) = -1 bit", close(np.log2(0.5), -1.0, 1e-12))
th0, Di0, R0 = revwf([8, 2, 0.1], 1.1)
check("Ex3.4: ch00 example theta=0.5", close(th0, 0.5, 1e-9))
check("Ex3.4: ch00 example R=3 bits", close(R0, 3.0, 1e-9))
check("Ex3.4: D=1.1", close(Di0.sum(), 1.1, 1e-9))
klA = 0.5 * (0.25 - 1 + np.log(4))
klB = 0.5 * (4 - 1 - np.log(4))
check("Ex3.5: D(N(0,1)||N(0,4))=0.3181 nats", close(klA, 0.3181))
check("Ex3.5: D(N(0,4)||N(0,1))=0.8069 nats", close(klB, 0.8069))
# numerical integral check of the Gaussian KL
xs = np.linspace(-20, 20, 400001)
f = np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi)
g = np.exp(-xs ** 2 / 8) / np.sqrt(8 * np.pi)
check("Ex3.5: KL formula matches quadrature", close(np.trapezoid(f * np.log(f / g), xs), klA, 1e-6))
SigXY = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)
check("Ex3.6: I(X;Y1,Y2)=1/2 log2 3=0.7925 bits", close(gmi(SigXY, [0], [1, 2]), 0.7925))
check("Ex3.6: single look 1/2 bit", close(gmi(SigXY, [0], [1]), 0.5, 1e-12))
check("Ex3.6: second look adds I(X;Y2|Y1)=0.2925",
      close(gmi(SigXY, [0], [1, 2]) - gmi(SigXY, [0], [1]), 0.2925))
P2 = np.diag([1.0, 0.25])
g2 = whitened_spectrum(Sx, P2)
check("Ex3.7: whitened spectrum (2.1514, 0.3486)", np.allclose(g2, [2.1514, 0.3486], atol=1e-4))
check("Ex3.7: same as eig of P^1/2 Sx P^1/2 and of P Sx",
      np.allclose(np.sort(np.linalg.eigvalsh(np.sqrt(P2) @ Sx @ np.sqrt(P2))), np.sort(g2))
      and np.allclose(np.sort(np.linalg.eigvals(P2 @ Sx).real), np.sort(g2)))
th7, Di7, R7 = revwf(g2, 0.5)
check("Ex3.7: theta=0.25", close(th7, 0.25, 1e-9))
check("Ex3.7: R_C(0.5)=1.7926 bits", close(R7, 1.7926))
# coordinatewise scheme in P's eigenbasis: products lambda_i * var_i = (2, 0.5)
prods = np.diag(P2) * np.diag(Sx)
check("Ex3.7: products (2,0.5)", np.allclose(prods, [2, 0.5]))
thp, Dip, Rp = revwf(prods, 0.5)
check("Ex3.7: coordinatewise scheme 2 bits", close(Rp, 2.0, 1e-9))
check("Ex3.7: excess 0.2075 bits = I(x1;x2)", close(Rp - R7, I_pair, 1e-6))
for tt, lab in [(0.25, "tau2=0.25"), (100.0, "tau2=100"), (1e-4, "tau2=1e-4")]:
    vxs = s2 * tt / (s2 + tt)
    vxms = 1 / (1 / D + 1 / tt)
    ell_t = 0.5 * np.log2(vxs / vxms)
    if tt == 0.25:
        check("Ex3.8: tau2=0.25 leakage 0.339 bits", close(ell_t, 0.339))
    if tt == 100.0:
        check("Ex3.8: tau2=100 leakage 0.9946 bits", close(ell_t, 0.9946, 1e-3))
    if tt == 1e-4:
        check("Ex3.8: tau2=1e-4 leakage near 0", ell_t < 1e-3)
# symbolic: ell = 1/2 log2 ((s2 + t2)/... ) closed form
s2s, t2s, Ds = sp.symbols("s2 t2 D", positive=True)
ell_s = sp.Rational(1, 2) * sp.log((s2s * t2s / (s2s + t2s)) * (1 / Ds + 1 / t2s), 2)
r_s = sp.Rational(1, 2) * sp.log(s2s / Ds, 2)
I_s = sp.Rational(1, 2) * sp.log((s2s + t2s) / (Ds + t2s), 2)
check("Ex3.8: ell = r - I(Xhat;S) symbolically",
      sp.simplify(sp.expand_log(ell_s - (r_s - I_s), force=True)) == 0)
# Ex3.9: P1 <= P2 but not refinable
S1b = np.diag([0.25, 1.0])
S2b = np.diag([0.5, 0.5])
check("Ex3.9: P1=diag(1,0) <= P2=I", np.linalg.eigvalsh(np.eye(2) - np.diag([1.0, 0.0])).min() >= 0)
check("Ex3.9: Sigma2*=0.5 I at D2=1", close(revwf([1, 1], 1.0)[0], 0.5, 1e-9))
check("Ex3.9: Sigma2* not <= Sigma1*", np.linalg.eigvalsh(S1b - S2b).min() < 0)
check("Ex3.9: R1=1 bit, R2=1 bit",
      close(0.5 * np.log2(1 / np.linalg.det(S1b)), 1.0, 1e-12)
      and close(0.5 * np.log2(1 / np.linalg.det(S2b)), 1.0, 1e-12))
# rate loss: max logdet Sigma <= diag(.25,1), tr Sigma <= 1 -> diag(.25,.75)
Sc9 = np.diag([0.25, 0.75])
check("Ex3.9: refinement-optimal Sigma diag(0.25,0.75), loss 0.2075 bits",
      close(0.5 * np.log2(np.linalg.det(S2b) / np.linalg.det(Sc9)), 0.2075))
# verify diag(.25,.75) is the max-det point by a grid over the feasible diagonal family and random PSD
best = -1e9
rng = np.random.default_rng(0)
for _ in range(20000):
    a = rng.uniform(0.01, 0.25)
    b = rng.uniform(0.01, 1.0)
    c = rng.uniform(-0.5, 0.5) * np.sqrt(a * b)
    Sg = np.array([[a, c], [c, b]])
    if np.linalg.eigvalsh(S1b - Sg).min() < 0 or np.trace(Sg) > 1.0:
        continue
    best = max(best, np.log(np.linalg.det(Sg)))
check("Ex3.9: random feasible points never beat diag(0.25,0.75)", best <= np.log(0.25 * 0.75) + 1e-12)

check("Ex3.9: total rate 1.2075 bits", close(0.5 * np.log2(1 / np.linalg.det(Sc9)), 1.2075))
check("Ex3.7: P Sx trace 2.5 det 0.75", close(np.trace(P2 @ Sx), 2.5, 1e-12) and close(np.linalg.det(P2 @ Sx), 0.75, 1e-12))
check("Ex3.7: theta P^-1 = diag(0.25,1) is the optimal error",
      np.allclose(0.25 * np.linalg.inv(P2), np.diag([0.25, 1.0])))
check("Ex3.6: Var(X|Y1,Y2)=1/3", close(1 / 3.0, 1 - SigXY[0, 1:] @ np.linalg.inv(SigXY[1:, 1:]) @ SigXY[1:, 0], 1e-12))
check("Ex3.8: tau2=0.25 Var(X|S)=0.2 and Var(X|Xhat,S)=0.125", close(0.25 / 1.25, 0.2, 1e-12) and close(1 / 8.0, 0.125, 1e-12))
check("ex R_C: consumer code saves 1.792 bits", close(RI - 1.0, 1.792))

w7, V7 = np.linalg.eigh(Sx)
Sh7 = V7 @ np.diag(np.sqrt(w7)) @ V7.T
Q7 = Sh7 @ P2 @ Sh7
l7, E7 = np.linalg.eigh(Q7)
X7 = E7 @ np.diag([1.0 if l <= 0.25 else 0.25 / l for l in l7]) @ E7.T
Sig7 = Sh7 @ X7 @ Sh7
check("Ex3.7: max-det optimal error (whitened water-filling) equals diag(0.25,1)", np.allclose(Sig7, np.diag([0.25, 1.0]), atol=1e-9))
check("Ex3.7: its rate is 1.7926 bits", close(0.5 * np.log2(np.linalg.det(Sx) / np.linalg.det(Sig7)), 1.7926))

# ---------------------------------------------------------------- additions (Gaussian KL, slope, trade-off)


def gkl(m0, S0, m1, S1):
    d = len(m0)
    S1i = np.linalg.inv(S1)
    dm = m1 - m0
    return 0.5 * (np.trace(S1i @ S0) - d + dm @ S1i @ dm + np.log(np.linalg.det(S1) / np.linalg.det(S0)))


z2 = np.zeros(2)
check("ex gauss KL: vs diag(2,2) gives 0.1438 nats", close(gkl(z2, S, z2, np.diag([2.0, 2.0])), 0.1438))
check("ex gauss KL: equals I(x1;x2)=0.2075 bits", close(gkl(z2, S, z2, np.diag([2.0, 2.0])) / LN2, 0.2075))
check("ex gauss KL: trace term = d", close(np.trace(np.linalg.inv(np.diag([2.0, 2.0])) @ S), 2.0, 1e-12))
check("ex gauss KL: vs N(0,I) gives 0.4507 nats", close(gkl(z2, S, z2, np.eye(2)), 0.4507))
# Monte Carlo sanity of the Gaussian KL formula
rng2 = np.random.default_rng(1)
L = np.linalg.cholesky(S)
xx = rng2.standard_normal((400000, 2)) @ L.T
logf = -0.5 * np.einsum('ij,jk,ik->i', xx, np.linalg.inv(S), xx) - 0.5 * np.log(np.linalg.det(2 * np.pi * S))
logg = -0.5 * (xx ** 2).sum(1) - np.log(2 * np.pi)
check("ex gauss KL: Monte Carlo agrees to 0.01", close((logf - logg).mean(), 0.4507, 1e-2))
# slope
th165, _, R165 = revwf([4, 2, 1, 0.25], 1.65)
check("ex slope: theta at D=1.65 is 0.4667", close(th165, 0.4667))
check("ex slope: R(1.65)=3.149 bits", close(R165, 3.149))
check("ex slope: 1/(2 theta ln2) at theta=0.5 = 1.4427 bits", close(1 / (2 * 0.5 * LN2), 1.4427))
check("ex slope: linear prediction 3.144", close(3 + 0.1 * 1.4427, 3.144))
# numerical slope
eps_ = 1e-6
check("ex slope: numerical dR/dD at 1.75 = -1/(2 theta ln 2) bits",
      close((revwf([4, 2, 1, 0.25], 1.75 + eps_)[2] - revwf([4, 2, 1, 0.25], 1.75 - eps_)[2]) / (2 * eps_), -1.4427, 1e-3))
# trade-off example
K = np.diag([1.0, 0.5])
Jm = np.diag([0.0, 1.0])


def r_ell(Sd):
    Se = np.linalg.inv(np.linalg.inv(Sd) + Jm)
    r_ = 0.5 * np.log2(1.0 / np.linalg.det(Sd))
    l_ = 0.5 * np.log2(np.linalg.det(K) / np.linalg.det(Se))
    return r_, l_


r0, l0 = r_ell(0.5 * np.eye(2))
check("ex trade-off: rate-optimal r=1 bit", close(r0, 1.0, 1e-12))
check("ex trade-off: rate-optimal leakage 0.7925 bits", close(l0, 0.7925))
check("ex trade-off: Cov(X|Xhat,S)=diag(0.5,1/3)",
      np.allclose(np.linalg.inv(np.linalg.inv(0.5 * np.eye(2)) + Jm), np.diag([0.5, 1 / 3])))
a_ = 2 - np.sqrt(2)
check("ex trade-off: a=2-sqrt2=0.5858 solves a^2-4a+2=0", close(a_, 0.5858) and close(a_ ** 2 - 4 * a_ + 2, 0, 1e-12))
r1_, l1_ = r_ell(np.diag([a_, 1 - a_]))
check("ex trade-off: leakage-optimal leakage 0.7716 bits", close(l1_, 0.7716))
check("ex trade-off: leakage-optimal rate 1.0216 bits", close(r1_, 1.0216))
check("ex trade-off: rate cost 0.0216, leakage saving 0.0209",
      close(r1_ - r0, 0.0216) and close(l0 - l1_, 0.0209))
check("ex trade-off: closed form l(a)", close(0.5 * np.log2((2 - a_) / 2 / (a_ * (1 - a_))), l1_, 1e-12))
# global check: random search over all 2x2 PD Sd <= I with tr Sd <= 1
best = 1e9
rng3 = np.random.default_rng(2)
for _ in range(100000):
    A = rng3.uniform(-1, 1, (2, 2))
    Sd = A @ A.T + 1e-3 * np.eye(2)
    Sd = Sd / np.trace(Sd) * rng3.uniform(0.5, 1.0)
    if np.linalg.eigvalsh(np.eye(2) - Sd).min() < 0:
        continue
    best = min(best, r_ell(Sd)[1])
check("ex trade-off: no feasible Sd beats 0.7716 (random search)", best >= l1_ - 1e-9)
# fine grid on the active face tr Sd = 1: Sd = [[a, c], [c, 1-a]]
ag = np.linspace(1e-4, 1 - 1e-4, 4001)[:, None]
cg = np.linspace(-0.5, 0.5, 2001)[None, :]
det_ = ag * (1 - ag) - cg ** 2
ok_ = det_ > 0
lg = np.where(ok_, 0.5 * np.log2(0.5 * (1 + (1 - ag)) / np.where(ok_, det_, 1.0)), np.inf)
i_, j_ = np.unravel_index(np.argmin(lg), lg.shape)
check("ex trade-off: grid minimum at a=0.5858, c=0", close(ag[i_, 0], 0.5858, 1e-3) and abs(cg[0, j_]) < 1e-9)
check("ex trade-off: grid minimum value 0.7716", close(lg[i_, j_], 0.7716))

n = len(RESULTS)
k = sum(RESULTS)
print(f"{k}/{n} PASS")
sys.exit(0 if k == n else 1)
