"""Checks for every number in chapter 5 (geometry), its exercises and answers.

Run on Atlas:  python3 ch05_examples.py   (prints k/n PASS, exits nonzero on any FAIL)
"""
import sys
import numpy as np
import sympy as sp
from scipy.linalg import expm, logm, sqrtm, eigh

RESULTS = []


def check(desc, cond):
    ok = bool(cond)
    RESULTS.append(ok)
    print(("PASS " if ok else "FAIL ") + desc)


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) < tol


# ---------------------------------------------------------------- 5.1 Mahalanobis
S = sp.Matrix([[2, 1], [1, 2]])
Si = S.inv()
check("Sigma^{-1} = (1/3)[[2,-1],[-1,2]]", Si == sp.Rational(1, 3) * sp.Matrix([[2, -1], [-1, 2]]))
a = sp.Matrix([1, 1]); b = sp.Matrix([1, -1])
dMa2 = (a.T * Si * a)[0]; dMb2 = (b.T * Si * b)[0]
check("d_M(a)^2 = 2/3", dMa2 == sp.Rational(2, 3))
check("d_M(b)^2 = 2", dMb2 == 2)
check("d_M(a) = 0.816", close(sp.sqrt(dMa2), 0.8165))
check("d_M(b) = 1.414", close(sp.sqrt(dMb2), 1.4142))
check("Euclidean |a| = |b| = sqrt2", a.norm() == b.norm() == sp.sqrt(2))
ev = S.eigenvals()
check("eigenvalues of Sigma are 3 and 1", ev == {3: 1, 1: 1})
check("(1,1) is eigvec with eigenvalue 3", S * a == 3 * a)
check("(1,-1) is eigvec with eigenvalue 1", S * b == b)
# whitening form
Sn = np.array([[2., 1.], [1., 2.]])
W = np.linalg.inv(sqrtm(Sn)).real
check("whitening ||Sigma^{-1/2} a|| = d_M(a)", close(np.linalg.norm(W @ np.array([1., 1.])), np.sqrt(2 / 3)))
# triangle inequality spot check for the Mahalanobis metric
rng = np.random.default_rng(0)
Sin = np.linalg.inv(Sn)
dm = lambda x, y: np.sqrt((x - y) @ Sin @ (x - y))
pts = rng.normal(size=(200, 3, 2))
check("Mahalanobis triangle inequality on 200 random triples",
      all(dm(p[0], p[2]) <= dm(p[0], p[1]) + dm(p[1], p[2]) + 1e-12 for p in pts))

# ---------------------------------------------------------------- 5.2 sphere
th = sp.pi / 3
lat_len = sp.sin(th) * sp.pi / 2
check("latitude path at colatitude 60deg, 90deg of longitude = 1.360", close(lat_len, 1.3603))
p = sp.Matrix([sp.sin(th), 0, sp.cos(th)]); q = sp.Matrix([0, sp.sin(th), sp.cos(th)])
check("p.q = 1/4", sp.simplify((p.T * q)[0]) == sp.Rational(1, 4))
gc = sp.acos(sp.Rational(1, 4))
check("great-circle distance arccos(1/4) = 1.318", close(gc, 1.3181))
check("great circle shorter than latitude path", float(gc) < float(lat_len))
check("chord |p-q| = sqrt(3/2) = 1.225", close((p - q).norm(), 1.2247))
check("chord shorter than great circle", float((p - q).norm()) < float(gc))
# latitude path length by integrating sqrt(gamma' G gamma') with G = diag(1, sin^2 theta)
t = sp.symbols("t")
integrand = sp.sqrt(0 + sp.sin(th) ** 2 * 1)
check("arc-length integral of latitude path = sin(th)*pi/2",
      sp.simplify(sp.integrate(integrand, (t, 0, sp.pi / 2)) - lat_len) == 0)
# polar metric: x = r cos phi, y = r sin phi pulls back I to diag(1, r^2)
r, ph = sp.symbols("r phi", positive=True)
Jp = sp.Matrix([[sp.cos(ph), -r * sp.sin(ph)], [sp.sin(ph), r * sp.cos(ph)]])
check("polar coordinates pull back I to diag(1, r^2)", sp.simplify(Jp.T * Jp) == sp.diag(1, r ** 2))
# sphere chart pulls back I to diag(1, sin^2 theta)
TH, PH = sp.symbols("theta phi")
X = sp.Matrix([sp.sin(TH) * sp.cos(PH), sp.sin(TH) * sp.sin(PH), sp.cos(TH)])
Js = X.jacobian([TH, PH])
check("sphere chart pulls back I to diag(1, sin^2 theta)", sp.simplify(Js.T * Js - sp.diag(1, sp.sin(TH) ** 2)) == sp.zeros(2))

# ---------------------------------------------------------------- 5.3 pullback
J = sp.Matrix([[1, 2]])
P = J.T * J
check("P_C = [[1,2],[2,4]]", P == sp.Matrix([[1, 2], [2, 4]]))
check("kernel of P_C is span(2,-1)", P * sp.Matrix([2, -1]) == sp.zeros(2, 1))
check("nonzero eigenvalue of P_C is 5", P.eigenvals() == {0: 1, 5: 1})
check("delta=(2,-1) leaves C unchanged: 1*2+2*(-1)=0", 1 * 2 + 2 * (-1) == 0)
check("delta=(1,2): delta^T P delta = 25 = (J delta)^2", (sp.Matrix([1, 2]).T * P * sp.Matrix([1, 2]))[0] == 25)
# normalization map C(x) = x/||x|| at x = (3,4)
x0 = sp.Matrix([3, 4]); rr = x0.norm(); u = x0 / rr
Jn = (sp.eye(2) - u * u.T) / rr
Pn = Jn.T * Jn
check("r = 5, u = (0.6, 0.8)", rr == 5 and u == sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5)]))
check("P = (I-uu^T)/25", sp.simplify(Pn - (sp.eye(2) - u * u.T) / 25) == sp.zeros(2))
check("P entries 0.0256, -0.0192, 0.0144",
      Pn == sp.Matrix([[sp.Rational(16, 625), sp.Rational(-12, 625)], [sp.Rational(-12, 625), sp.Rational(9, 625)]]))
check("16/625=0.0256, 12/625=0.0192, 9/625=0.0144",
      close(16 / 625, 0.0256, 1e-9) and close(12 / 625, 0.0192, 1e-9) and close(9 / 625, 0.0144, 1e-9))
check("P u = 0 (radius unread)", Pn * u == sp.zeros(2, 1))
v = sp.Matrix([-sp.Rational(4, 5), sp.Rational(3, 5)])
check("tangent v=(-0.8,0.6): v^T P v = 1/25 = 0.04", (v.T * Pn * v)[0] == sp.Rational(1, 25))
# numeric Jacobian of normalization equals the formula
xs = sp.symbols("x1 x2")
Cn = sp.Matrix(xs) / sp.sqrt(xs[0] ** 2 + xs[1] ** 2)
Jsym = Cn.jacobian(xs).subs({xs[0]: 3, xs[1]: 4})
check("Jacobian of x/||x|| at (3,4) equals (I-uu^T)/r", sp.simplify(Jsym - Jn) == sp.zeros(2))
# softmax Fisher metric
pv = sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 4)])
G = sp.diag(*pv) - pv * pv.T
check("softmax G entries", G == sp.Matrix([[sp.Rational(1, 4), -sp.Rational(1, 8), -sp.Rational(1, 8)],
                                          [-sp.Rational(1, 8), sp.Rational(3, 16), -sp.Rational(1, 16)],
                                          [-sp.Rational(1, 8), -sp.Rational(1, 16), sp.Rational(3, 16)]]))
check("softmax G * 1 = 0", G * sp.ones(3, 1) == sp.zeros(3, 1))
check("softmax G has rank 2", G.rank() == 2)
check("softmax G is PSD (eigenvalues >= 0)", all(sp.N(e) >= -1e-12 for e in G.eigenvals()))
# KL(softmax(l) || softmax(l + eps*h)) ~ 1/2 eps^2 h^T G h
l = np.log(np.array([0.5, 0.25, 0.25]))
h = np.array([1.0, -0.5, 0.3]); eps = 1e-3
sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
p1, p2 = sm(l), sm(l + eps * h)
kl = np.sum(p1 * np.log(p1 / p2))
Gn = np.array(G, dtype=float)
check("softmax KL matches 1/2 eps^2 h^T G h", abs(kl / (0.5 * eps ** 2 * h @ Gn @ h) - 1) < 1e-2)
check("shifting all logits by a constant leaves softmax unchanged", np.allclose(sm(l), sm(l + 7.0)))

# ---------------------------------------------------------------- 5.4 Fisher
kl_b = 0.5 * np.log(0.5 / 0.6) + 0.5 * np.log(0.5 / 0.4)
check("Bernoulli KL(0.5||0.6) = 0.02041", close(kl_b, 0.020411, 1e-6))
F = 1 / (0.5 * 0.5)
check("Bernoulli Fisher at 0.5 is 4", F == 4)
check("quadratic approx 1/2 * 4 * 0.01 = 0.02", close(0.5 * F * 0.01, 0.02, 1e-12))
check("relative error about 2 percent", close((kl_b - 0.02) / kl_b, 0.0201, 1e-3))
# Fisher info of Bernoulli symbolically
th_ = sp.symbols("theta", positive=True)
ll1 = sp.log(th_); ll0 = sp.log(1 - th_)
Fsym = sp.simplify(th_ * sp.diff(ll1, th_) ** 2 + (1 - th_) * sp.diff(ll0, th_) ** 2)
check("Bernoulli Fisher info = 1/(theta(1-theta))", sp.simplify(Fsym - 1 / (th_ * (1 - th_))) == 0)
# Gaussian location: KL exactly 1/2 delta^T Sigma^{-1} delta
Sig = np.array([[2., 1.], [1., 2.]]); dl = np.array([0.3, -0.2])
klg = 0.5 * dl @ np.linalg.inv(Sig) @ dl
# closed form KL between N(mu1,S) and N(mu2,S)
check("Gaussian location KL = 1/2 delta^T Sigma^{-1} delta (= 0.0633)", close(klg, 0.06333, 1e-4))
# Fisher-Rao distance Bernoulli 0.5 -> 0.9
dfr = 2 * (np.arcsin(np.sqrt(0.9)) - np.arcsin(np.sqrt(0.5)))
check("Fisher-Rao distance 0.5->0.9 = 0.9273", close(dfr, 0.9273))
check("Bhattacharyya check cos(d/2) = sqrt(.45)+sqrt(.05)", close(np.cos(dfr / 2), np.sqrt(0.45) + np.sqrt(0.05), 1e-9))
# sqrt map: theta = sin^2(phi) gives Fisher metric 4 dphi^2
phi = sp.symbols("phi", positive=True)
thph = sp.sin(phi) ** 2
check("pullback of Fisher metric under theta=sin^2 phi is 4",
      sp.simplify(sp.diff(thph, phi) ** 2 / (thph * (1 - thph)) - 4) == 0)
# Fisher bridge: KL of channel through C
# channel N(C(x), s^2 I) with C(x) = (x1 + 2 x2): KL = (C(x+d)-C(x))^2/(2 s^2) = 1/2 d^T J^T F J d, F=1/s^2
s2 = 0.25; d = np.array([0.1, 0.05]); Jv = np.array([1., 2.])
kl_ch = (Jv @ d) ** 2 / (2 * s2)
check("Fisher bridge exact for Gaussian channel: 1/2 d^T J^T F J d", close(kl_ch, 0.5 * d @ np.outer(Jv, Jv) / s2 @ d, 1e-12))
check("Fisher bridge value 0.08", close(kl_ch, 0.08, 1e-12))

# ---------------------------------------------------------------- 5.5 SPD
def d_ai(A, B):
    Ai = np.linalg.inv(sqrtm(A)).real
    w = np.linalg.eigvalsh(Ai @ B @ Ai)
    return np.sqrt(np.sum(np.log(w) ** 2))


I2 = np.eye(2)
S2 = np.diag([4., 0.25]); S3 = np.diag([4., 1.])
check("d_AI(I, diag(4,1/4)) = sqrt2 ln4 = 1.9605", close(d_ai(I2, S2), np.sqrt(2) * np.log(4)) and close(d_ai(I2, S2), 1.9605))
check("d_AI(I, diag(4,1)) = ln4 = 1.3863", close(d_ai(I2, S3), 1.3863))
check("Frobenius ||I - diag(4,1/4)|| = 3.0923", close(np.linalg.norm(I2 - S2), 3.0923))
check("Frobenius ||I - diag(4,1)|| = 3", close(np.linalg.norm(I2 - S3), 3.0))
A = np.array([[1., 1.], [0., 1.]])
check("affine invariance d(A I A^T, A S2 A^T) = d(I,S2)", close(d_ai(A @ I2 @ A.T, A @ S2 @ A.T), d_ai(I2, S2), 1e-9))
check("A I A^T = [[2,1],[1,1]]", np.allclose(A @ A.T, [[2, 1], [1, 1]]))
check("A S2 A^T = [[4.25,0.25],[0.25,0.25]]", np.allclose(A @ S2 @ A.T, [[4.25, 0.25], [0.25, 0.25]]))
check("Frobenius NOT invariant: ||A(I-S2)A^T|| != ||I-S2||",
      abs(np.linalg.norm(A @ (I2 - S2) @ A.T) - np.linalg.norm(I2 - S2)) > 0.1)
check("1-D: d(1,2) = d(100,200) = ln 2 = 0.6931",
      close(d_ai(np.eye(1), 2 * np.eye(1)), 0.6931) and close(d_ai(100 * np.eye(1), 200 * np.eye(1)), 0.6931))
check("d_AI formula via eigenvalues of S1^{-1}S2 equals ||log(S1^{-1/2} S2 S1^{-1/2})||_F",
      close(np.linalg.norm(logm(np.linalg.inv(sqrtm(S3)) @ S2 @ np.linalg.inv(sqrtm(S3))).real), d_ai(S3, S2), 1e-9))
# log-Euclidean equals AI for commuting matrices; differs for non-commuting
d_le = lambda X, Y: np.linalg.norm(logm(X).real - logm(Y).real)
check("log-Euclidean = AI for commuting diag matrices", close(d_le(S3, S2), d_ai(S3, S2), 1e-9))
Rm = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
S4 = Rm @ S2 @ Rm.T
check("log-Euclidean != AI for non-commuting pair", abs(d_le(S3, S4) - d_ai(S3, S4)) > 1e-3)
# symmetric and triangle inequality spot check
rng = np.random.default_rng(1)
def rspd():
    M = rng.normal(size=(3, 3)); return M @ M.T + 0.3 * np.eye(3)
ok = True
for _ in range(100):
    X, Y, Z = rspd(), rspd(), rspd()
    ok &= d_ai(X, Z) <= d_ai(X, Y) + d_ai(Y, Z) + 1e-9
    ok &= abs(d_ai(X, Y) - d_ai(Y, X)) < 1e-8
check("AI distance symmetric and satisfies triangle inequality on 100 random triples", ok)
# geodesic midpoint S1^{1/2}(S1^{-1/2}S2S1^{-1/2})^{1/2}S1^{1/2}; for I and diag(4,1/4) it is diag(2,1/2)
mid = sqrtm(S2).real
check("AI midpoint of I and diag(4,1/4) is diag(2,1/2)", np.allclose(mid, np.diag([2, 0.5])))
check("midpoint equidistant: d(I,mid)=d(mid,S2)=half", close(d_ai(I2, mid), d_ai(I2, S2) / 2, 1e-9) and close(d_ai(mid, S2), d_ai(I2, S2) / 2, 1e-9))
check("arithmetic mean diag(2.125,0.625) differs from geodesic midpoint", not np.allclose((I2 + S2) / 2, mid))
check("det of midpoint = 1 = geometric mean of dets", close(np.linalg.det(mid), 1.0, 1e-9))
# Fisher metric on Gaussian covariance: KL(N(0,S)||N(0,S+eA)) ~ (1/4) e^2 tr((S^-1 A)^2)
def kl_gauss0(S1_, S2_):
    k = S1_.shape[0]
    S2i = np.linalg.inv(S2_)
    return 0.5 * (np.trace(S2i @ S1_) - k + np.log(np.linalg.det(S2_) / np.linalg.det(S1_)))
Sb = np.array([[2., 0.5], [0.5, 1.]]); Ab = np.array([[1., 0.3], [0.3, -0.5]]); e = 1e-3
lhs = kl_gauss0(Sb, Sb + e * Ab)
Sbi = np.linalg.inv(Sb)
rhs = 0.25 * e ** 2 * np.trace(Sbi @ Ab @ Sbi @ Ab)
check("Gaussian covariance KL ~ (1/4) e^2 tr((S^-1 A)^2)", abs(lhs / rhs - 1) < 1e-2)
# Exercise: d_AI(diag(1,1), diag(e, e^-1)) = sqrt2
check("exercise: d_AI(I, diag(e,1/e)) = sqrt 2", close(d_ai(I2, np.diag([np.e, 1 / np.e])), np.sqrt(2), 1e-9))
# Exercise: inverse map isometry d(S1^-1,S2^-1) = d(S1,S2)
X, Y = rspd(), rspd()
check("exercise: inversion is an isometry of d_AI", close(d_ai(np.linalg.inv(X), np.linalg.inv(Y)), d_ai(X, Y), 1e-8))
# Exercise: d_AI(diag(1,1), diag(9,1)) = ln 9 = 2.1972, Frobenius 8
check("exercise: d_AI(I,diag(9,1)) = ln9 = 2.197", close(d_ai(I2, np.diag([9., 1.])), 2.1972))
check("exercise: d_AI(diag(100,1),diag(108,1)) = ln1.08 = 0.0770", close(d_ai(np.diag([100., 1.]), np.diag([108., 1.])), 0.0770))

# ---------------------------------------------------------------- 5.6 hyperbolic
def dpd(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.arccosh(1 + 2 * np.sum((x - y) ** 2) / ((1 - x @ x) * (1 - y @ y)))


check("d(0,(0.5,0)) = ln3 = 1.0986", close(dpd([0, 0], [0.5, 0]), np.log(3)) and close(np.log(3), 1.0986))
check("d(0,(0.9,0)) = ln19 = 2.9444", close(dpd([0, 0], [0.9, 0]), np.log(19)) and close(np.log(19), 2.9444))
check("d(0,r) = 2 artanh r", close(dpd([0, 0], [0.7, 0]), 2 * np.arctanh(0.7), 1e-9))
dop = dpd([0.9, 0], [-0.9, 0])
check("d((0.9,0),(-0.9,0)) = 2 ln 19 = 5.8889", close(dop, 2 * np.log(19)) and close(dop, 5.8889))
dq = dpd([0.9, 0], [0, 0.9])
check("argument 1 + 3.24/0.0361 = 90.75", close(1 + 3.24 / 0.0361, 90.7507))
check("d((0.9,0),(0,0.9)) = 5.2012", close(dq, 5.2012))
check("ratio to path through origin 0.883", close(dq / (2 * np.log(19)), 0.8832))
check("Euclidean ratio chord/through-origin = 1.2728/1.8 = 0.707", close(np.sqrt(1.62) / 1.8, 0.7071))
# conformal metric check: small step at radius 0.9 has length 2/(1-0.81) * step
h_ = 1e-6
check("conformal factor at r=0.9 is 2/0.19 = 10.526", close(dpd([0.9, 0], [0.9 + h_, 0]) / h_, 2 / 0.19, 1e-3) and close(2 / 0.19, 10.526))
# circumference of hyperbolic circle radius rho: 2 pi sinh rho; rho=3
check("hyperbolic circumference at rho=3 is 2pi sinh3 = 62.94 vs Euclidean 18.85",
      close(2 * np.pi * np.sinh(3), 62.94, 0.01) and close(2 * np.pi * 3, 18.85, 0.01))
# figure geodesic: circle centre (k,k), k = 1.81/1.8, radius R
k = 1.81 / 1.8; R2 = 2 * k ** 2 - 1; Rr = np.sqrt(R2)
check("figure: k = 1.0056, R = 1.0111", close(k, 1.0056) and close(Rr, 1.0111))
check("figure: circle passes through (0.9,0)", close((0.9 - k) ** 2 + k ** 2, R2, 1e-9))
check("figure: circle orthogonal to unit circle (|c|^2 = 1 + R^2)", close(2 * k ** 2, 1 + R2, 1e-9))
a1 = np.degrees(np.arctan2(0 - k, 0.9 - k)) % 360; a2 = np.degrees(np.arctan2(0.9 - k, 0 - k)) % 360
check("figure: arc angles 186.0 and 264.0 degrees", close(a2, 186.0, 0.05) and close(a1, 264.0, 0.05))
# exercise: tree leaves at radius r = 0.99
check("exercise: d(0,0.99) = ln199 = 5.293", close(dpd([0, 0], [0.99, 0]), np.log(199)) and close(np.log(199), 5.2933))
# exercise: Mobius invariance: the map z -> (z - a)/(1 - conj(a) z) preserves distance
def mob(z, aa):
    return (z - aa) / (1 - np.conj(aa) * z)
z1, z2, aa = 0.3 + 0.4j, -0.5 + 0.1j, 0.2 - 0.6j
w1, w2 = mob(z1, aa), mob(z2, aa)
check("exercise: disk automorphism preserves d",
      close(dpd([z1.real, z1.imag], [z2.real, z2.imag]), dpd([w1.real, w1.imag], [w2.real, w2.imag]), 1e-9))

# ---------------------------------------------------------------- 5.7 Laplacian
f = np.array([0., 1., 3.])
Ap = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
Lp = np.diag(Ap.sum(1)) - Ap
check("path P3 f=(0,1,3): f^T L f = 5", close(f @ Lp @ f, 5.0, 1e-12))
check("Lp = [[1,-1,0],[-1,2,-1],[0,-1,1]]", np.allclose(Lp, [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]))
check("Lp eigenvalues 0,1,3", np.allclose(np.linalg.eigvalsh(Lp), [0, 1, 3]))
n = 6
A6 = np.zeros((n, n))
for i in range(n):
    A6[i, (i + 1) % n] = A6[(i + 1) % n, i] = 1
D6 = np.diag(A6.sum(1))
L6 = np.eye(n) - A6 / 2
w6, U6 = np.linalg.eigh(L6)
check("C6 normalized Laplacian eigenvalues 0,.5,.5,1.5,1.5,2", np.allclose(w6, [0, .5, .5, 1.5, 1.5, 2]))
ang = 2 * np.pi * np.arange(n) / n
c_ = np.cos(ang); s_ = np.sin(ang)
check("cos(2 pi i/6) is eigvec with eigenvalue 0.5", np.allclose(L6 @ c_, 0.5 * c_))
check("sin(2 pi i/6) is eigvec with eigenvalue 0.5", np.allclose(L6 @ s_, 0.5 * s_))
Y = np.stack([c_, s_], 1)
# angle between embedded nodes = 60 deg * hop distance
okang = True
for i in range(n):
    for j in range(n):
        hop = min(abs(i - j), n - abs(i - j))
        cosang = Y[i] @ Y[j] / (np.linalg.norm(Y[i]) * np.linalg.norm(Y[j]))
        okang &= close(np.degrees(np.arccos(np.clip(cosang, -1, 1))), 60 * hop, 1e-6)
check("C6 embedding angle = 60 deg x hop distance", okang)
check("C6 embedding radius constant", np.allclose(np.linalg.norm(Y, axis=1), 1))
# any orthonormal basis of the 0.5 eigenspace gives same angles (rotation)
E = U6[:, 1:3]
G6 = E @ E.T
okang2 = True
for i in range(n):
    for j in range(n):
        hop = min(abs(i - j), n - abs(i - j))
        cosang = G6[i, j] / np.sqrt(G6[i, i] * G6[j, j])
        okang2 &= close(np.degrees(np.arccos(np.clip(cosang, -1, 1))), 60 * hop, 1e-6)
check("numerical eigenbasis gives same angles", okang2)
# effective resistance on C6: R = k(6-k)/6
Lu = D6 - A6
Lplus = np.linalg.pinv(Lu)
Reff = lambda i, j: Lplus[i, i] + Lplus[j, j] - 2 * Lplus[i, j]
check("C6 resistance hop1 = 5/6", close(Reff(0, 1), 5 / 6, 1e-9))
check("C6 resistance hop3 = 3/2", close(Reff(0, 3), 1.5, 1e-9))
check("C6 resistance hop2 = 4/3", close(Reff(0, 2), 4 / 3, 1e-9))
# commute embedding identity ||Psi_i - Psi_j||^2 = R(i,j) (normalized-L spectral form)
dg = A6.sum(1)
Psi = np.stack([U6[:, k_] / (np.sqrt(w6[k_]) * np.sqrt(dg)) for k_ in range(1, n)], 1)
check("commute embedding ||Psi_i-Psi_j||^2 = R(i,j) on C6",
      all(close(np.sum((Psi[i] - Psi[j]) ** 2), Reff(i, j), 1e-9) for i in range(n) for j in range(n)))
# von Luxburg limit 1/d_i + 1/d_j on C6 vs exact
check("C6: 1/d_i+1/d_j = 1, exact hop1 R = 0.833 (small graph not in the limit)", close(1 / 2 + 1 / 2, 1) and close(Reff(0, 1), 0.8333))
# Laplacian eigenmap generalized problem L y = lam D y on C6 has lam = 1 - cos(2 pi k/6)
wg = np.sort(np.real(np.linalg.eigvals(np.linalg.inv(D6) @ Lu)))
check("generalized eigenvalues L y = lam D y on C6 = 0,.5,.5,1.5,1.5,2", np.allclose(wg, [0, .5, .5, 1.5, 1.5, 2]))
# star graph exercise: centre degree 4 leaves degree 1; R(leaf,leaf)=2, R(centre, leaf)=1
As = np.zeros((5, 5))
for i in range(1, 5):
    As[0, i] = As[i, 0] = 1
Ls = np.diag(As.sum(1)) - As
Lsp = np.linalg.pinv(Ls)
Rs = lambda i, j: Lsp[i, i] + Lsp[j, j] - 2 * Lsp[i, j]
check("star: R(leaf,leaf) = 2 = 1/1 + 1/1", close(Rs(1, 2), 2.0, 1e-9))
check("star: R(centre,leaf) = 1", close(Rs(0, 1), 1.0, 1e-9))
check("star: 1/d_c + 1/d_leaf = 1.25 differs from exact 1", close(1 / 4 + 1, 1.25))
# exercise: Rayleigh quotient smallest nonconstant on P3 is 1 with y=(1,0,-1)
yv = np.array([1., 0., -1.])
check("P3: (1,0,-1) eigvec of L with eigenvalue 1", np.allclose(Lp @ yv, yv))
check("P3: (1,-2,1) eigvec of L with eigenvalue 3", np.allclose(Lp @ np.array([1., -2., 1.]), 3 * np.array([1., -2., 1.])))

# ---------------------------------------------------------------- exercises misc
# Ex: Mahalanobis with Sigma=diag(9,1): points (3,0),(0,1) both distance 1
Sd9 = np.diag([1 / 9, 1.])
check("exercise: diag(9,1): d_M((3,0)) = d_M((0,1)) = 1",
      close(np.sqrt(np.array([3., 0]) @ Sd9 @ np.array([3., 0])), 1, 1e-12) and close(np.sqrt(np.array([0, 1.]) @ Sd9 @ np.array([0, 1.])), 1, 1e-12))
# Ex: pullback of C(x)=(x1, x1 x2) at (1,2): J = [[1,0],[2,1]]
Jx = np.array([[1., 0.], [2., 1.]])
check("exercise: J^T J at (1,2) = [[5,2],[2,1]]", np.allclose(Jx.T @ Jx, [[5, 2], [2, 1]]))
check("exercise: at (0,2) J=[[1,0],[2,0]] has kernel e2", np.allclose(np.array([[1., 0.], [2., 0.]]) @ np.array([0., 1.]), 0))
check("exercise: J^T J at (0,2) = [[5,0],[0,0]]", np.allclose(np.array([[1., 0.], [2., 0.]]).T @ np.array([[1., 0.], [2., 0.]]), [[5, 0], [0, 0]]))
# Ex: Gaussian channel N(theta, sigma^2), F = 1/sigma^2; sigma=0.5 -> 4
check("exercise: Fisher of N(theta, 0.25) is 4", close(1 / 0.25, 4, 1e-12))
# Ex: Poisson Fisher 1/lambda, KL(Pois(4)||Pois(4.4)) vs 1/2 * 0.16/4 = 0.02
klp = 4 * np.log(4 / 4.4) + 4.4 - 4
check("exercise: KL(Pois 4 || Pois 4.4) = 0.01876", close(klp, 0.01876, 1e-5))
check("exercise: quadratic 1/2 * 0.16 / 4 = 0.02", close(0.5 * 0.16 / 4, 0.02, 1e-12))

# ---------------------------------------------------------------- additions
u1 = np.array([0.6, 0.8]); u2 = np.array([0.8, -0.6])
P1 = (np.eye(2) - np.outer(u1, u1)) / 25; P2 = (np.eye(2) - np.outer(u2, u2)) / 25
check("workload average of angle reader at (3,4),(4,-3) = I/50", np.allclose((P1 + P2) / 2, np.eye(2) / 50))
check("|(4,-3)| = 5", close(np.linalg.norm([4, -3]), 5, 1e-12))
check("each pointwise P has rank 1", np.linalg.matrix_rank(P1) == 1 and np.linalg.matrix_rank(P2) == 1)
klr = 0.6 * np.log(1.2) + 0.4 * np.log(0.8)
check("reverse KL(0.6||0.5) = 0.02014", close(klr, 0.02014, 1e-5))
kv = lambda s1, s2: 0.5 * (s1 / s2 - 1 + np.log(s2 / s1))
check("KL(N(0,1)||N(0,1.1)) = 0.0022", close(kv(1, 1.1), 0.0022, 5e-5))
check("KL(N(0,100)||N(0,110)) = KL(N(0,1)||N(0,1.1))", close(kv(100, 110), kv(1, 1.1), 1e-12))
check("quadratic eps^2/4 = 0.0025", close(0.1 ** 2 / 4, 0.0025, 1e-12))
# metric transformation rule: whitening gives identity
Sh = np.real(sqrtm(Sn))
check("G_z = A^T G_x A with A=Sigma^{1/2}, G_x=Sigma^{-1} gives I", np.allclose(Sh.T @ np.linalg.inv(Sn) @ Sh, np.eye(2)))
# eigenmap trace identity: tr(Y^T L Y) = 1/2 sum w ||yi-yj||^2
Yr = np.random.default_rng(3).normal(size=(6, 2))
lhs6 = np.trace(Yr.T @ Lu @ Yr)
rhs6 = 0.5 * sum(A6[i, j] * np.sum((Yr[i] - Yr[j]) ** 2) for i in range(6) for j in range(6))
check("tr(Y^T L Y) = 1/2 sum w_ij ||y_i - y_j||^2", close(lhs6, rhs6, 1e-9))

# ---------------------------------------------------------------- figure data (printed as FIGDATA)
def fmt(pts):
    return " ".join(f"({x:.3f},{y:.3f})" for x, y in pts)
# Fig sphere: rotate about z by -45 deg so p, q are symmetric; view elevation 20 deg
el = np.radians(70)
def proj(v):
    c, s_ = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    x, y, z = c * v[0] - s_ * v[1], s_ * v[0] + c * v[1], v[2]
    return (y, -x * np.sin(el) + z * np.cos(el))
th0 = np.pi / 3
pn = np.array([np.sin(th0), 0, np.cos(th0)]); qn = np.array([0, np.sin(th0), np.cos(th0)])
lat_full = [proj([np.sin(th0) * np.cos(f), np.sin(th0) * np.sin(f), np.cos(th0)]) for f in np.linspace(0, 2 * np.pi, 73)]
lat_arc = [proj([np.sin(th0) * np.cos(f), np.sin(th0) * np.sin(f), np.cos(th0)]) for f in np.linspace(0, np.pi / 2, 31)]
om = np.arccos(pn @ qn)
gc_arc = [proj((np.sin((1 - t_) * om) * pn + np.sin(t_ * om) * qn) / np.sin(om)) for t_ in np.linspace(0, 1, 31)]
equator = [proj([np.cos(f), np.sin(f), 0]) for f in np.linspace(0, 2 * np.pi, 73)]
check("fig sphere: slerp points have unit norm", all(close(np.linalg.norm((np.sin((1 - t_) * om) * pn + np.sin(t_ * om) * qn) / np.sin(om)), 1, 1e-12) for t_ in np.linspace(0, 1, 11)))
check("fig sphere: great-circle arc is above latitude arc (higher z at midpoint)",
      ((pn + qn) / np.linalg.norm(pn + qn))[2] > np.cos(th0))
print("FIGDATA sphere_p:", fmt([proj(pn)]), "q:", fmt([proj(qn)]), "north:", fmt([proj([0, 0, 1])]))
print("FIGDATA sphere_lat_full:", fmt(lat_full))
print("FIGDATA sphere_lat_arc:", fmt(lat_arc))
print("FIGDATA sphere_gc_arc:", fmt(gc_arc))
print("FIGDATA sphere_equator:", fmt(equator))
# Fig Poincare: geodesic arcs through pairs, and metric unit-ball circles of radius eps (1-r^2)/2
def geo_arc(pz, qz, n=40):
    pz, qz = complex(*pz), complex(*qz)
    # hyperbolic geodesic: map pz to 0 by Mobius, straight line to image of qz, map back
    m = lambda z, a: (z - a) / (1 - np.conj(a) * z)
    mi = lambda w, a: (w + a) / (1 + np.conj(a) * w)
    w = m(qz, pz)
    pts = [mi(w * t_, pz) for t_ in np.linspace(0, 1, n)]
    return [(z.real, z.imag) for z in pts]
pairs = [((0.9, 0), (0, 0.9)), ((-0.6, 0.6), (-0.7, -0.5)), ((0.2, -0.85), (0.8, -0.4)), ((-0.9, 0), (0.9, 0))]
for k_, (pa, pb) in enumerate(pairs):
    arc = geo_arc(pa, pb)
    # length along arc equals d_H
    L = sum(dpd(arc[i], arc[i + 1]) for i in range(len(arc) - 1))
    check(f"fig poincare arc {k_}: polyline length matches d_H", close(L, dpd(pa, pb), 2e-2 * dpd(pa, pb)))
    print(f"FIGDATA poincare_arc{k_}:", fmt(arc))
arc0 = geo_arc((0.9, 0), (0, 0.9))
check("fig poincare: Mobius arc agrees with circle centre (k,k) radius R", all(close((x - k) ** 2 + (y - k) ** 2, R2, 1e-6) for x, y in arc0))
balls = []
for r_ in (0.0, 0.35, 0.6, 0.8, 0.92):
    nang = 1 if r_ == 0 else (6 if r_ < 0.5 else 12)
    for j in range(nang):
        a_ = 2 * np.pi * j / nang + (0.3 if r_ > 0.7 else 0)
        balls.append((r_ * np.cos(a_), r_ * np.sin(a_), 0.07 * (1 - r_ ** 2)))
check("fig poincare: unit-ball radius scales as (1-r^2)", close(balls[0][2] * (1 - 0.8 ** 2), [b for b in balls if close(np.hypot(b[0], b[1]), 0.8, 1e-9)][0][2], 1e-12))
print("FIGDATA poincare_balls:", " ".join(f"{x:.3f}/{y:.3f}/{rb:.4f}" for x, y, rb in balls))
# Fig Fisher: exact KL contours for N(mu, sigma) around grid points, eps = 0.03
def kl_ns(m0, s0, m1, s1):
    return np.log(s1 / s0) + (s0 ** 2 + (m0 - m1) ** 2) / (2 * s1 ** 2) - 0.5
def contour(m0, s0, eps=0.03, n=48):
    out = []
    for a_ in np.linspace(0, 2 * np.pi, n + 1):
        lo, hi = 0.0, 0.9 * s0
        for _ in range(60):
            mid_ = (lo + hi) / 2
            if kl_ns(m0, s0, m0 + mid_ * np.cos(a_), s0 + mid_ * np.sin(a_)) < eps:
                lo = mid_
            else:
                hi = mid_
        out.append((m0 + lo * np.cos(a_), s0 + lo * np.sin(a_)))
    return out
for m0 in (-1.0, 0.0, 1.0):
    for s0 in (0.5, 1.0, 1.6):
        print(f"FIGDATA fisher_{m0:+.0f}_{s0}:", fmt(contour(m0, s0)))
# quadratic ellipse at (0,1): 1/2 [dm^2/s^2 + 2 ds^2/s^2] = eps -> semi-axes sqrt(2 eps) s and sqrt(eps) s
check("fig fisher: Fisher metric of N(mu,sigma) is diag(1/s^2, 2/s^2)",
      close(kl_ns(0, 1, 1e-3, 1) / (0.5 * 1e-6), 1, 1e-3) and close(kl_ns(0, 1, 0, 1 + 1e-3) / (0.5 * 2e-6), 1, 1e-2))
check("fig fisher: quadratic semi-axes at (0,1), eps .03: 0.245 (mu), 0.173 (sigma)",
      close(np.sqrt(2 * 0.03), 0.2449) and close(np.sqrt(0.03), 0.1732))
cc = contour(0.0, 1.0)
check("fig fisher: exact contour at (0,1) extends ~0.245 along mu", close(max(x for x, y in cc), 0.245, 0.01))
check("fig fisher: contour at sigma=1.6 is 1.6x wider in mu than at sigma=1",
      close(max(x for x, y in contour(0.0, 1.6)) / max(x for x, y in cc), 1.6, 0.02))
# Fig SPD: covariance ellipse semi-axes along AI and Euclidean paths from I to diag(4,1/4)
for t_ in (0, 0.25, 0.5, 0.75, 1):
    ai = (2 ** t_, 2 ** -t_); eu = (np.sqrt(1 + 3 * t_), np.sqrt(1 - 0.75 * t_))
    print(f"FIGDATA spd t={t_}: AI {ai[0]:.3f} {ai[1]:.3f} det {np.prod(ai)**2:.3f} | EU {eu[0]:.3f} {eu[1]:.3f} det {(np.prod(eu))**2:.3f}")
check("fig spd: AI path det stays 1", all(close((2 ** t_ * 2 ** -t_) ** 2, 1, 1e-12) for t_ in (0, .25, .5, .75, 1)))
check("fig spd: Euclidean midpoint det = 2.5*0.625 = 1.5625? no: (1+1.5)(1-0.375)=1.5625",
      close((1 + 3 * 0.5) * (1 - 0.75 * 0.5), 1.5625, 1e-12))
check("fig spd: AI path is geodesic (d(I,S(t)) = t d(I,S2))",
      all(close(d_ai(I2, np.diag([4 ** t_, 4 ** -t_])), t_ * d_ai(I2, S2), 1e-9) for t_ in (0.25, 0.5, 0.75)))
# Fig ring graph C12: scrambled layout and eigenmap
n12 = 12
A12 = np.zeros((n12, n12))
for i in range(n12):
    A12[i, (i + 1) % n12] = A12[(i + 1) % n12, i] = 1
L12 = np.eye(n12) - A12 / 2
w12, U12 = np.linalg.eigh(L12)
check("fig ring: smallest nonzero eigenvalue of C12 is 1-cos(30deg) = 0.134, multiplicity 2",
      close(w12[1], 1 - np.cos(np.pi / 6), 1e-9) and close(w12[2], w12[1], 1e-9) and close(w12[1], 0.134))
emb = np.stack([np.cos(2 * np.pi * np.arange(n12) / n12), np.sin(2 * np.pi * np.arange(n12) / n12)], 1)
check("fig ring: cos/sin columns lie in that eigenspace", np.allclose(L12 @ emb, w12[1] * emb))
rs = np.random.default_rng(7)
scr = rs.uniform(-1, 1, size=(n12, 2))
print("FIGDATA ring_scrambled:", fmt(scr))
print("FIGDATA ring_embed:", fmt(emb))

# hypothesis audit: a zero Jacobian does not make a finite step invisible
Cq = lambda x: x[0] ** 2
Jq0 = np.array([2 * 0.0, 0.0])
check("kernel is first order: C=x1^2 has J=0 at origin, step (0.1,0) changes C by 0.01",
      np.allclose(Jq0, 0) and close(Cq((0.1, 0.0)) - Cq((0.0, 0.0)), 0.01, 1e-12))
# hypothesis audit: on C6 the degree formula 1/d_i+1/d_j = 1 for every pair, resistances vary
A6 = np.zeros((6, 6))
for i in range(6):
    A6[i, (i + 1) % 6] = A6[(i + 1) % 6, i] = 1
L6p = np.linalg.pinv(np.diag(A6.sum(1)) - A6)
R6 = [L6p[0, 0] + L6p[k, k] - 2 * L6p[0, k] for k in (1, 2, 3)]
check("C6: degree formula predicts 1, resistances are 5/6, 4/3, 3/2",
      close(1 / 2 + 1 / 2, 1, 1e-12) and np.allclose(R6, [5 / 6, 4 / 3, 3 / 2], atol=1e-9))

npass = sum(RESULTS)
print(f"{npass}/{len(RESULTS)} PASS")
sys.exit(0 if npass == len(RESULTS) else 1)
