"""Checks every number in chapter 1 (linear algebra), its exercise answers and figure.

Run on Atlas:  /home/claude/env/bin/python3 ch01_examples.py
Prints PASS/FAIL per claim, then 'k/n PASS'; exits nonzero on any FAIL.
"""
import sys
import numpy as np
import sympy as sp

results = []


def check(desc, cond):
    ok = bool(cond)
    results.append(ok)
    print(("PASS" if ok else "FAIL") + "  " + desc)


R = sp.Rational
M_ = sp.Matrix
sqrt = sp.sqrt


def is_psd(A):
    return all(ev >= -1e-12 for ev in np.linalg.eigvalsh(np.array(A, dtype=float)))


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


# ---------------------------------------------------------------- Example 1.1 (G-projection)
u = M_([1, 1]); x = M_([1, -1]); G = sp.diag(2, 1)
Pe = u * u.T / 2
check("Euclidean projector sends x=(1,-1) to 0", Pe * x == M_([0, 0]))
check("u^T G u = 3", (u.T * G * u)[0] == 3)
check("u^T G = (2,1)", u.T * G == M_([[2, 1]]))
PG = u * (u.T * G * u).inv() * u.T * G
check("Pi_G = (1/3)[[2,1],[2,1]]", PG == M_([[2, 1], [2, 1]]) / 3)
check("Pi_G x = (1/3)(1,1)", PG * x == M_([R(1, 3), R(1, 3)]))
check("Pi_G idempotent", PG * PG == PG)
check("Pi_G first entry of square: (4+2)/9 = 2/3", R(4 + 2, 9) == R(2, 3) and (PG * PG)[0, 0] == R(2, 3))
check("Pi_G not symmetric", PG != PG.T)
check("G Pi_G = Pi_G^T G", G * PG == PG.T * G)
res = x - PG * x
check("residual = (2/3,-4/3)", res == M_([R(2, 3), R(-4, 3)]))
check("residual G-orthogonal to u", (res.T * G * u)[0] == 0)
check("residual dot u = -2/3", (res.T * u)[0] == R(-2, 3))
check("squared G-distance to line = 8/3", (res.T * G * res)[0] == R(8, 3))
check("||x||_G^2 = 3", (x.T * G * x)[0] == 3)
# general basis formula equals QQ^T
q1 = M_([1, 1, 0]) / sqrt(2); q2 = M_([0, 0, 1])
Pi3 = q1 * q1.T + q2 * q2.T
check("Pi (R^3) matrix", Pi3 == M_([[R(1, 2), R(1, 2), 0], [R(1, 2), R(1, 2), 0], [0, 0, 1]]))
check("Pi (3,1,2) = (2,2,2)", Pi3 * M_([3, 1, 2]) == M_([2, 2, 2]))
check("residual (1,-1,0) in kernel", Pi3 * M_([1, -1, 0]) == M_([0, 0, 0]) and M_([3, 1, 2]) - Pi3 * M_([3, 1, 2]) == M_([1, -1, 0]))
check("tr Pi = rank Pi = 2", Pi3.trace() == 2 and Pi3.rank() == 2)
A3 = M_([[1, 0], [1, 0], [0, 1]])
check("A(A^TA)^{-1}A^T = QQ^T", A3 * (A3.T * A3).inv() * A3.T == Pi3)
# figure: ellipse semi-axes and tangency
check("figure: ellipse x-radius^2 = (8/3)/2 = 4/3", R(8, 3) / 2 == R(4, 3))
check("figure: ellipse y-radius^2 = 8/3", R(8, 3) / 1 == R(8, 3))
check("figure: Euclidean circle radius sqrt2 touches at origin", (x.T * x)[0] == 2)
# tangency of the G-ellipse: minimum over the line of G-distance is 8/3 at t=1/3
t = sp.symbols('t')
f = ((M_([t, t]) - x).T * G * (M_([t, t]) - x))[0]
check("figure: G-distance minimized at t=1/3 with value 8/3", sp.solve(sp.diff(f, t), t) == [R(1, 3)] and f.subs(t, R(1, 3)) == R(8, 3))

# ---------------------------------------------------------------- Example 1.2 (spectral)
A = M_([[5, 2], [2, 2]])
check("A trace 7 det 6", A.trace() == 7 and A.det() == 6)
check("A eigenvalues 6, 1", set(A.eigenvals().keys()) == {6, 1})
v1 = M_([2, 1]) / sqrt(5); v2 = M_([1, -2]) / sqrt(5)
check("A v1 = 6 v1, A v2 = v2", sp.simplify(A * v1 - 6 * v1) == M_([0, 0]) and sp.simplify(A * v2 - v2) == M_([0, 0]))
P1 = v1 * v1.T; P2 = v2 * v2.T
check("Pi_1 = (1/5)[[4,2],[2,1]]", sp.simplify(P1 - M_([[4, 2], [2, 1]]) / 5) == sp.zeros(2))
check("Pi_2 = (1/5)[[1,-2],[-2,4]]", sp.simplify(P2 - M_([[1, -2], [-2, 4]]) / 5) == sp.zeros(2))
check("6 Pi1 + Pi2 = A", sp.simplify(6 * P1 + P2 - A) == sp.zeros(2))
check("Pi1 Pi2 = 0, Pi1 + Pi2 = I", sp.simplify(P1 * P2) == sp.zeros(2) and sp.simplify(P1 + P2) == sp.eye(2))
Ah = sqrt(6) * P1 + P2
check("sqrt(A) squares to A", sp.simplify(Ah * Ah - A) == sp.zeros(2))
check("sqrt(A)_11 = (4 sqrt6 + 1)/5 ~ 2.1596", sp.simplify(Ah[0, 0] - (4 * sqrt(6) + 1) / 5) == 0 and close(Ah[0, 0], 2.1596, 5e-5))
check("sqrt(A)_22 = (sqrt6 + 4)/5 ~ 1.2899", sp.simplify(Ah[1, 1] - (sqrt(6) + 4) / 5) == 0 and close(Ah[1, 1], 1.2899, 5e-5))
check("sqrt(A)_12 = (2 sqrt6 - 2)/5 ~ 0.5798", sp.simplify(Ah[0, 1] - (2 * sqrt(6) - 2) / 5) == 0 and close(Ah[0, 1], 0.5798, 5e-5))
check("2.1596^2 + 0.5798^2 ~ 5.000", close(2.1596 ** 2 + 0.5798 ** 2, 5.000, 5e-4))
check("sqrt(A) is PSD", is_psd(Ah.evalf()))
D_ = sp.diag(2, 0) - sp.eye(2)
check("diag(2,0) - I = diag(1,-1) indefinite", D_ == sp.diag(1, -1) and not is_psd(D_) and not is_psd(-D_))
A2 = M_([[2, 1], [1, 1]]); B2 = M_([[1, 0], [0, 0]])
check("A - B = ones, eigen 2 and 0, so A >= B >= 0", A2 - B2 == sp.ones(2) and set((A2 - B2).eigenvals()) == {2, 0} and is_psd(B2))
check("A^2 - B^2 = [[4,3],[3,2]] det -1", A2 ** 2 - B2 ** 2 == M_([[4, 3], [3, 2]]) and (A2 ** 2 - B2 ** 2).det() == -1)
check("A^2 - B^2 not PSD", not is_psd(A2 ** 2 - B2 ** 2))
A3_ = M_([[3, 1], [1, 2]])
check("A - I trace 3 det 1 (PSD)", (A3_ - sp.eye(2)).trace() == 3 and (A3_ - sp.eye(2)).det() == 1 and is_psd(A3_ - sp.eye(2)))
check("A^{-1} = (1/5)[[2,-1],[-1,3]]", A3_.inv() == M_([[2, -1], [-1, 3]]) / 5)
check("I - A^{-1} = (1/5)[[3,1],[1,2]], det 0.2, PSD", sp.eye(2) - A3_.inv() == M_([[3, 1], [1, 2]]) / 5 and (sp.eye(2) - A3_.inv()).det() == R(1, 5) and is_psd(sp.eye(2) - A3_.inv()))
# random check of inverse antitonicity and trace monotonicity
rng = np.random.default_rng(1)
okinv = True; oktr = True
for _ in range(200):
    Bm = rng.standard_normal((3, 3)); Bm = Bm @ Bm.T + 0.1 * np.eye(3)
    Cm = rng.standard_normal((3, 3)); Am = Bm + Cm @ Cm.T
    okinv &= is_psd(np.linalg.inv(Bm) - np.linalg.inv(Am))
    Pm = rng.standard_normal((3, 2)); Pm = Pm @ Pm.T
    oktr &= np.trace(Pm @ Am) >= np.trace(Pm @ Bm) - 1e-10
check("random: A >= B > 0 implies B^-1 >= A^-1", okinv)
check("random: A >= B, P >= 0 implies tr(PA) >= tr(PB)", oktr)

# ---------------------------------------------------------------- Example 1.3 (trace)
P = M_([[2, 1], [1, 2]])
check("P eigen 3 along (1,1), 1 along (1,-1)", P * M_([1, 1]) == 3 * M_([1, 1]) and P * M_([1, -1]) == M_([1, -1]))
Sa = sp.diag(1, 3); Sb = M_([[2, 1], [1, 2]])
w1 = M_([1, 1]) / sqrt(2); w2 = M_([1, -1]) / sqrt(2)
check("Sigma_a variances along eigvecs 2 and 2", (w1.T * Sa * w1)[0] == 2 and (w2.T * Sa * w2)[0] == 2)
check("P Sigma_a = [[2,3],[1,6]] trace 8", P * Sa == M_([[2, 3], [1, 6]]) and (P * Sa).trace() == 8)
check("weighted-variance form 3*2+1*2 = 8", 3 * 2 + 1 * 2 == 8)
check("Sigma_b variances along eigvecs 3 and 1", (w1.T * Sb * w1)[0] == 3 and (w2.T * Sb * w2)[0] == 1)
check("P Sigma_b = [[5,4],[4,5]] trace 10", P * Sb == M_([[5, 4], [4, 5]]) and (P * Sb).trace() == 10)
check("both traces 4", Sa.trace() == 4 and Sb.trace() == 4)
mu = M_([1, 0])
check("bias term mu^T P mu = 2, total 10", (mu.T * P * mu)[0] == 2 and (P * Sa).trace() + 2 == 10)
# Exercise 1.10 variance of quadratic form
check("Ex1.10: 2 tr(PSPS) = 92", 2 * (P * Sa * P * Sa).trace() == 92)
check("Ex1.10: 4 mu^T P S P mu = 28", 4 * (mu.T * P * Sa * P * mu)[0] == 28)
check("Ex1.10: se = sqrt(120/1e5) ~ 0.035", close(np.sqrt(120 / 1e5), 0.035, 1e-3))
z = np.random.default_rng(0).standard_normal((100000, 2))
d = np.array([1., 0.]) + z @ np.sqrt(np.diag([1., 3.]))
Pn = np.array([[2., 1.], [1., 2.]])
mc = np.mean(np.einsum('ni,ij,nj->n', d, Pn, d))
check(f"Ex1.10: Monte Carlo mean {mc:.4f} within 0.1 of 10", abs(mc - 10) < 0.1)
# tracepos identity
Ps = np.array([[2., 1.], [1., 2.]]); Ss = np.array([[1., 0.], [0., 3.]])
from scipy.linalg import sqrtm  # noqa: E402
check("tr(P Sigma) = ||Sigma^1/2 P^1/2||_F^2", close(np.trace(Ps @ Ss), np.linalg.norm(np.real(sqrtm(Ss)) @ np.real(sqrtm(Ps)), 'fro') ** 2, 1e-9))

# ---------------------------------------------------------------- Example 1.4 (whitening)
Sx = sp.diag(4, 1); g = M_([1, 1]); Pg = g * g.T
check("P = gg^T eigen 2 and 0", set(Pg.eigenvals()) == {2, 0})
Pt = sp.diag(2, 1) * Pg * sp.diag(2, 1)
check("P~ = [[4,2],[2,1]]", Pt == M_([[4, 2], [2, 1]]))
check("P~ eigen 5 along (2,1), 0 along (1,-2)", Pt * M_([2, 1]) == 5 * M_([2, 1]) and Pt * M_([1, -2]) == M_([0, 0]))
check("tr(P Sigma_x) = 5", (Pg * Sx).trace() == 5)
check("P Sigma_x = [[4,1],[4,1]] trace 5 det 0", Pg * Sx == M_([[4, 1], [4, 1]]) and (Pg * Sx).det() == 0)
check("Sigma^1/2 (1,-2) = (2,-2) in ker P", sp.diag(2, 1) * M_([1, -2]) == M_([2, -2]) and Pg * M_([2, -2]) == M_([0, 0]))
Ph = sp.Matrix(Pg / sqrt(2))  # P^{1/2} of rank-one gg^T is gg^T/||g||
check("P^1/2 Sigma P^1/2 has same spectrum {5,0}", set(sp.simplify(Ph * Sx * Ph).eigenvals()) == {5, 0})
# general whitening Q Sigma^{-1/2}
Sg = np.array([[5., 2.], [2., 2.]]); evals, V = np.linalg.eigh(Sg)
W_pca = np.diag(evals ** -0.5) @ V.T
check("PCA whitening Lambda^-1/2 V^T whitens", np.allclose(W_pca @ Sg @ W_pca.T, np.eye(2)))
Qw = W_pca @ np.real(sqrtm(Sg))
check("W Sigma^1/2 is orthogonal", np.allclose(Qw @ Qw.T, np.eye(2)))

# ---------------------------------------------------------------- Example 1.5 (pencil)
Ap = M_([[2, 2], [2, 8]]); Bp = sp.diag(1, 4); mu_ = sp.symbols('mu')
check("det(A - mu B) = 4mu^2 - 16mu + 12", sp.expand((Ap - mu_ * Bp).det() - (4 * mu_ ** 2 - 16 * mu_ + 12)) == 0)
check("roots 3 and 1", set(sp.solve((Ap - mu_ * Bp).det(), mu_)) == {1, 3})
check("A - 3B = [[-1,2],[2,-4]], (2,1) in kernel", Ap - 3 * Bp == M_([[-1, 2], [2, -4]]) and (Ap - 3 * Bp) * M_([2, 1]) == M_([0, 0]))
check("A - B = [[1,2],[2,4]], (2,-1) in kernel", Ap - Bp == M_([[1, 2], [2, 4]]) and (Ap - Bp) * M_([2, -1]) == M_([0, 0]))
check("v^T B v = 8 for both", (M_([2, 1]).T * Bp * M_([2, 1]))[0] == 8 and (M_([2, -1]).T * Bp * M_([2, -1]))[0] == 8)
check("B-orthogonal, dot product 3", (M_([2, 1]).T * Bp * M_([2, -1]))[0] == 0 and M_([2, 1]).dot(M_([2, -1])) == 3)
V = sp.Matrix.hstack(M_([2, 1]), M_([2, -1])) / (2 * sqrt(2))
check("V^T B V = I, V^T A V = diag(3,1)", sp.simplify(V.T * Bp * V) == sp.eye(2) and sp.simplify(V.T * Ap * V) == sp.diag(3, 1))
check("A(2,1) = (6,12), 24/8 = 3", Ap * M_([2, 1]) == M_([6, 12]) and M_([2, 1]).dot(M_([6, 12])) == 24)
check("Rayleigh at e1 = 2", Ap[0, 0] / Bp[0, 0] == 2)
check("ordinary eigenvalues 5 +- sqrt13 ~ 8.606, 1.394", set(Ap.eigenvals()) == {5 + sqrt(13), 5 - sqrt(13)} and close(5 + np.sqrt(13), 8.606, 5e-4) and close(5 - np.sqrt(13), 1.394, 5e-4))
check("B^-1 A eigenvalues 1, 3", set((Bp.inv() * Ap).eigenvals()) == {1, 3})
Ak = M_([[3, 1, 0], [1, 3, 0], [0, 0, 1]])
check("Ky Fan matrix eigen 4,2,1", set(Ak.eigenvals()) == {4, 2, 1})
check("tr over span(e1,e2) = 6, over span(e1,e3) = 4", Ak[0, 0] + Ak[1, 1] == 6 and Ak[0, 0] + Ak[2, 2] == 4)
# Ky Fan / Courant-Fischer random spot check (Exercise 1.12 logic)
Ar = rng.standard_normal((6, 6)); Ar = (Ar + Ar.T) / 2; lam = np.sort(np.linalg.eigvalsh(Ar))[::-1]
best = -np.inf
for _ in range(2000):
    U, _r = np.linalg.qr(rng.standard_normal((6, 2)))
    best = max(best, np.trace(U.T @ Ar @ U))
_, Ve = np.linalg.eigh(Ar)
check("Ky Fan: random U never exceed top-2 sum", best <= lam[0] + lam[1] + 1e-10)
check("Ky Fan: top eigenvectors attain it", close(np.trace(Ve[:, -2:].T @ Ar @ Ve[:, -2:]), lam[0] + lam[1], 1e-9))
# simultaneous diagonalization of two PSD via (A, A+B)
Apsd = np.array([[1., 0.], [0., 0.]]); Bpsd = np.array([[1., 1.], [1., 1.]])
from scipy.linalg import eigh  # noqa: E402
mus, Vs = eigh(Apsd, Apsd + Bpsd)
check("PSD pair via pencil (A, A+B): both diagonal", np.allclose(Vs.T @ Apsd @ Vs, np.diag(np.diag(Vs.T @ Apsd @ Vs))) and np.allclose(Vs.T @ Bpsd @ Vs, np.diag(np.diag(Vs.T @ Bpsd @ Vs))))

# ---------------------------------------------------------------- Example 1.6 (Schur)
M = M_([[4, 2, 2], [2, 2, 1], [2, 1, 3]])
Bs = M_([[2, 2]]); Ds = M_([[2, 1], [1, 3]])
check("B^T A^-1 B = ones", Bs.T * Bs / 4 == sp.ones(2))
S = Ds - Bs.T * Bs / 4
check("S = diag(1,2)", S == sp.diag(1, 2))
check("det M = 8 = 4 * det S", M.det() == 8 and 4 * S.det() == 8)
check("cofactor expansion 20 - 8 - 4 = 8", 4 * (6 - 1) - 2 * (6 - 2) + 2 * (2 - 4) == 8)
check("M PD", is_psd(M) and min(np.linalg.eigvalsh(np.array(M, dtype=float))) > 0)
check("lower-right block of M^-1 = diag(1, 1/2)", M.inv()[1:, 1:] == sp.diag(1, R(1, 2)))
M2 = M_([[4, 2, 2], [2, 2, 1], [2, 1, R(1, 2)]])
check("with corner 1/2: S = diag(1,-1/2) and M not PSD", M2[1:, 1:] - Bs.T * Bs / 4 == sp.diag(1, R(-1, 2)) and not is_psd(M2))
# block inverse formula and Woodbury on random
Ar_ = rng.standard_normal((5, 5)); Mr = Ar_ @ Ar_.T + np.eye(5)
A_, B_, D_b = Mr[:2, :2], Mr[:2, 2:], Mr[2:, 2:]
Sr = D_b - B_.T @ np.linalg.inv(A_) @ B_
Ai = np.linalg.inv(A_); Si = np.linalg.inv(Sr)
blk = np.block([[Ai + Ai @ B_ @ Si @ B_.T @ Ai, -Ai @ B_ @ Si], [-Si @ B_.T @ Ai, Si]])
check("block inverse formula", np.allclose(blk, np.linalg.inv(Mr)))
check("Woodbury identity", np.allclose(np.linalg.inv(A_ - B_ @ np.linalg.inv(D_b) @ B_.T), Ai + Ai @ B_ @ Si @ B_.T @ Ai))
check("det M = det A det S", close(np.linalg.det(Mr), np.linalg.det(A_) * np.linalg.det(Sr), 1e-8 * abs(np.linalg.det(Mr))))

# ---------------------------------------------------------------- Example 1.7 (Hadamard)
def sylvester(d):
    H = np.array([[1.]])
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    return H


H4 = sylvester(4)
check("H4 rows", np.array_equal(H4, np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])))
check("H4 H4^T = 4 I", np.array_equal(H4 @ H4.T, 4 * np.eye(4)))
Ds_ = np.diag([1., -1., 1., 1.]); Rh = 0.5 * H4 @ Ds_
check("R orthogonal", np.allclose(Rh @ Rh.T, np.eye(4)))
e1 = np.array([1., 0, 0, 0])
check("R e1 = (1/2)(1,1,1,1), squares 1/4, length 1", np.allclose(Rh @ e1, 0.5 * np.ones(4)) and np.isclose(np.linalg.norm(Rh @ e1), 1))
xh = 0.5 * np.ones(4)
check("unsigned (1/2) H4 x = e1", np.allclose(0.5 * H4 @ xh, e1))
check("row dots with D_s x*2 = (2,2,-2,2)", np.allclose(H4 @ np.array([1., -1, 1, 1]), [2, 2, -2, 2]))
check("R x = (1/2,1/2,-1/2,1/2)", np.allclose(Rh @ xh, [0.5, 0.5, -0.5, 0.5]))
# second moment over random signs: exact enumeration, d=4, arbitrary x
import itertools  # noqa: E402
xr = np.array([3., -1., 0.5, 2.]); acc = np.zeros(4)
for s in itertools.product([1., -1.], repeat=4):
    acc += (0.5 * H4 @ (np.array(s) * xr)) ** 2
acc /= 16
check("randomized Hadamard: each coord second moment = ||x||^2/d", np.allclose(acc, xr @ xr / 4))
# Haar spread: Monte Carlo with sign correction
d = 3; xq = np.array([1., 2., 2.]); C = np.zeros((3, 3)); N = 40000
r2 = np.random.default_rng(5)
for _ in range(N):
    Q, Rr = np.linalg.qr(r2.standard_normal((d, d)))
    Q = Q @ np.diag(np.sign(np.diag(Rr)))
    y = Q @ xq; C += np.outer(y, y)
C /= N
check("Haar: E[Qx x^T Q^T] ~ ||x||^2/d I (MC, tol 0.15)", np.max(np.abs(C - (xq @ xq / d) * np.eye(3))) < 0.15)
# sign flip + permutation maps e_j to +-e_k (Exercise 1.8)
perm = np.array([2, 0, 3, 1]); sg = np.array([1., -1., -1., 1.])
ok = all(np.count_nonzero((np.eye(4)[j] * sg)[perm]) == 1 for j in range(4))
check("sign flip + permutation maps e_j to a signed coordinate vector", ok)
check("(1/sqrt d) H D e_j has all magnitudes 1/sqrt d", all(np.allclose(np.abs(Rh @ np.eye(4)[j]), 0.5) for j in range(4)))

# ---------------------------------------------------------------- Example 1.8 (log det)
X = M_([[2, 1], [1, 1]])
check("det X = 1, X^-1 = [[1,-1],[-1,2]]", X.det() == 1 and X.inv() == M_([[1, -1], [-1, 2]]))
H = sp.diag(R(1, 10), 0)
check("det(X+H) = 1.1", (X + H).det() == R(11, 10))
check("log 1.1 ~ 0.09531", close(np.log(1.1), 0.09531, 5e-6))
check("first order tr(X^-1 H) = 0.1", (X.inv() * H).trace() == R(1, 10))
check("second order -1/2 tr(X^-1 H X^-1 H) = -0.005", -R(1, 2) * (X.inv() * H * X.inv() * H).trace() == R(-5, 1000))
check("expansion 0.095 within 0.0004 of exact", abs(0.1 - 0.005 - np.log(1.1)) < 0.0004)
check("log det diag(1,4) = log 4 ~ 1.386", close(np.log(4), 1.386, 5e-4))
check("midpoint 2 log 2.5 ~ 1.833 > 1.386", close(2 * np.log(2.5), 1.833, 5e-4) and 2 * np.log(2.5) > np.log(4))
# gradient numerically
Xn = np.array([[3., 1.], [1., 2.]]); Hn = np.array([[0.2, -0.1], [-0.1, 0.3]]); eps = 1e-6
num = (np.log(np.linalg.det(Xn + eps * Hn)) - np.log(np.linalg.det(Xn - eps * Hn))) / (2 * eps)
check("d logdet = tr(X^-1 dX) numerically", close(num, np.trace(np.linalg.inv(Xn) @ Hn), 1e-7))
num2 = (np.linalg.inv(Xn + eps * Hn) - np.linalg.inv(Xn - eps * Hn)) / (2 * eps)
check("d(X^-1) = -X^-1 dX X^-1 numerically", np.allclose(num2, -np.linalg.inv(Xn) @ Hn @ np.linalg.inv(Xn), atol=1e-6))

# ---------------------------------------------------------------- Example 1.9 (omission floor)
Sf = M_([[2, 1], [1, 1]])
Sh = (Sf + sqrt(Sf.det()) * sp.eye(2)) / sqrt(Sf.trace() + 2 * sqrt(Sf.det()))
check("2x2 sqrt formula gives (1/sqrt5)[[3,1],[1,2]]", sp.simplify(Sh - M_([[3, 1], [1, 2]]) / sqrt(5)) == sp.zeros(2))
check("that matrix squares to Sigma_x", sp.simplify(Sh * Sh - Sf) == sp.zeros(2))
e1v = M_([1, 0]); e2v = M_([0, 1])
Phat = e1v * e1v.T; Ptrue = e2v * e2v.T
s1 = Sh * e1v
check("||s1||^2 = 2", sp.simplify((s1.T * s1)[0]) == 2)
Pi_f = sp.eye(2) - s1 * s1.T / 2
Kw = Sh * Phat * Sh
check("Pi projects onto ker(Sigma^1/2 Phat Sigma^1/2)", sp.simplify(Kw * Pi_f) == sp.zeros(2) and sp.simplify(Pi_f * Pi_f - Pi_f) == sp.zeros(2) and sp.simplify(Pi_f - Pi_f.T) == sp.zeros(2))
Ptil = Sh * Ptrue * Sh
Dfl = sp.simplify((Ptil * Pi_f).trace())
check("D_floor = 1/2", Dfl == R(1, 2))
check("D_floor = tr P~ - s1^T P~ s1 / 2 = 1 - 1/2", sp.simplify(Ptil.trace()) == 1 and sp.simplify((s1.T * Ptil * s1)[0]) == 1)
check("D_floor equals Schur complement / conditional variance", Sf[1, 1] - Sf[1, 0] ** 2 / Sf[0, 0] == R(1, 2))
check("naive kernel number ((Sigma^1/2)_22)^2 = 4/5", sp.simplify((Ptil * Ptrue).trace()) == R(4, 5) and sp.simplify(Sh[1, 1] ** 2) == R(4, 5))
# Monte Carlo: code x1 perfectly (conditional mean of x2 given x1), error variance in x2 = 1/2
rs = np.random.default_rng(7); Xs = rs.multivariate_normal([0, 0], [[2, 1], [1, 1]], size=200000)
err = Xs[:, 1] - 0.5 * Xs[:, 0]
check("MC: Var(x2 - E[x2|x1]) ~ 0.5", abs(err.var() - 0.5) < 0.01)
# G-Pythagoras in Example 1.1
check("||Pi_G x||_G^2 = 1/3 and 1/3 + 8/3 = 3", ((PG * x).T * G * (PG * x))[0] == R(1, 3) and R(1, 3) + R(8, 3) == 3)
# Loewner converse via P = xx^T (random spot check)
okc = True
for _ in range(200):
    Am = rng.standard_normal((3, 3)); Am = Am + Am.T; Bm = rng.standard_normal((3, 3)); Bm = Bm + Bm.T
    psd_rel = is_psd(Am - Bm)
    worst = min(np.linalg.eigvalsh(Am - Bm))
    vmin = np.linalg.eigh(Am - Bm)[1][:, 0]
    okc &= (psd_rel or np.trace(np.outer(vmin, vmin) @ (Am - Bm)) < 0) and abs(np.trace(np.outer(vmin, vmin) @ (Am - Bm)) - worst) < 1e-9
check("Loewner converse: if not A>=B, some P=xx^T has tr(PA) < tr(PB)", okc)

# ---------------------------------------------------------------- Exercise answers
Pi = M_([[1, 1], [0, 0]])
check("Ex1.2: Pi^2 = Pi, not symmetric", Pi * Pi == Pi and Pi != Pi.T)
check("Ex1.2: range e1, kernel (1,-1), angle 45 deg", Pi * M_([1, -1]) == M_([0, 0]) and close(np.degrees(np.arccos(1 / np.sqrt(2))), 45, 1e-9))
Ae = sp.diag(3, 1); Be = M_([[2, 1], [1, 2]])
check("Ex1.3: A - B = [[1,-1],[-1,-1]] det -2, indefinite", Ae - Be == M_([[1, -1], [-1, -1]]) and (Ae - Be).det() == -2 and not is_psd(Ae - Be) and not is_psd(Be - Ae))
Ac = M_([[1, 0], [0, 0]]); Bc = M_([[0, 1], [0, 0]]); Cc = M_([[0, 0], [1, 0]])
check("Ex1.4: tr(ABC) = 1, tr(ACB) = 0", (Ac * Bc * Cc).trace() == 1 and (Ac * Cc * Bc).trace() == 0 and Bc * Cc == sp.diag(1, 0) and Cc * Bc == sp.diag(0, 1))
gg = M_([1, -1]); Pt2 = sp.diag(2, 1) * gg * gg.T * sp.diag(2, 1)
check("Ex1.5: P~ = [[4,-2],[-2,1]], eigen 5 along (2,-1), tr(P Sigma) = 5", Pt2 == M_([[4, -2], [-2, 1]]) and Pt2 * M_([2, -1]) == 5 * M_([2, -1]) and (gg * gg.T * Sx).trace() == 5)
Aq = sp.diag(1, 0); Bq = M_([[2, 1], [1, 1]])
check("Ex1.6: det(A - mu B) = mu^2 - mu", sp.expand((Aq - mu_ * Bq).det() - (mu_ ** 2 - mu_)) == 0)
check("Ex1.6: eigvecs (0,1) for 0 and (1,-1) for 1", Aq * M_([0, 1]) == M_([0, 0]) and (Aq - Bq) * M_([1, -1]) == M_([0, 0]))
check("Ex1.6: B(1,-1) = (1,0), B-orthogonal", Bq * M_([1, -1]) == M_([1, 0]) and (M_([0, 1]).T * Bq * M_([1, -1]))[0] == 0)
check("Ex1.6: Rayleigh at (1,-1) = 1/1 = 1", (M_([1, -1]).T * Aq * M_([1, -1]))[0] == 1 and (M_([1, -1]).T * Bq * M_([1, -1]))[0] == 1)
Me = M_([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
Se = Me[1:, 1:] - Me[1:, :1] * Me[:1, 1:] / 2
check("Ex1.7: S = [[1.5,.5],[.5,1.5]] trace 3 det 2", Se == M_([[R(3, 2), R(1, 2)], [R(1, 2), R(3, 2)]]) and Se.trace() == 3 and Se.det() == 2)
check("Ex1.7: det M = 4, eigen 4,1,1", Me.det() == 4 and Me.eigenvals() == {4: 1, 1: 2})
Xe = sp.diag(2, 4); He = M_([[0, R(1, 10)], [R(1, 10), 0]])
check("Ex1.9: det(X+H) = 7.99", (Xe + He).det() == R(799, 100))
check("Ex1.9: log(7.99/8) ~ -0.00125078", close(np.log(7.99 / 8), -0.00125078, 5e-9))
check("Ex1.9: first order 0", (Xe.inv() * He).trace() == 0)
check("Ex1.9: second order -0.00125", -R(1, 2) * (Xe.inv() * He * Xe.inv() * He).trace() == R(-125, 100000))
check("Ex1.9: agreement to ~8e-7", abs(np.log(7.99 / 8) + 0.00125) < 1e-6 and abs(np.log(7.99 / 8) + 0.00125) > 7e-7)

# ---------------------------------------------------------------- figure data
# Lines starting with FIGDATA are pasted into the chapter's figures.
def figdata(name, text):
    print(f"FIGDATA {name}|{text}")


def ell(S):
    """semi-axes (sqrt eigenvalues) and angle (deg) of the major axis of covariance S."""
    w, V = np.linalg.eigh(np.array(S, dtype=float))
    v = V[:, 1]
    if v[0] < 0:
        v = -v
    return np.sqrt(w[1]), np.sqrt(w[0]), np.degrees(np.arctan2(v[1], v[0]))


# fig unit circle -> ellipse under A=[[5,2],[2,2]]: image ellipse has semi-axes 6 and 1
a_, b_, ang = ell(np.array([[5., 2.], [2., 2.]]) @ np.array([[5., 2.], [2., 2.]]))
check("fig circle->ellipse: semi-axes 6 and 1, angle 26.565", close(a_, 6, 1e-9) and close(b_, 1, 1e-9) and close(ang, 26.565, 1e-3))
figdata("ch01circle", f"{a_:.3f}/{b_:.3f}/{ang:.3f}")
# fig PD vs PSD level sets: x^T A x = 1 has semi-axes 1/sqrt(6), 1
check("fig level set semi-axes 1/sqrt6 = 0.408 and 1", close(1 / np.sqrt(6), 0.408, 5e-4))
figdata("ch01pdlevel", f"{1/np.sqrt(6):.3f}/1/{ang:.3f}")
# fig Loewner: I inside cov-ellipse of [[3,1],[1,2]]; diag(3,1) vs [[2,1],[1,2]] crossing
aL, bL, angL = ell(np.array([[3., 1.], [1., 2.]]))
check("fig Loewner: [[3,1],[1,2]] ellipse axes 1.902, 1.176 at 31.717 deg", close(aL, 1.902, 5e-4) and close(bL, 1.176, 5e-4) and close(angL, 31.717, 1e-3))
check("fig Loewner: unit circle inside (min axis >= 1)", bL >= 1)
figdata("ch01loewnerA", f"{aL:.3f}/{bL:.3f}/{angL:.3f}")
aC, bC, angC = ell(np.array([[2., 1.], [1., 2.]]))
check("fig Loewner: [[2,1],[1,2]] axes sqrt3, 1 at 45 deg; diag(3,1) axes sqrt3, 1", close(aC, np.sqrt(3)) and close(bC, 1) and close(angC, 45, 1e-9))
figdata("ch01loewnerC", f"{aC:.3f}/{bC:.3f}/{angC:.3f}")
# fig whitening cloud
rw = np.random.default_rng(3); Z = rw.standard_normal((60, 2))
Shalf = np.real(sqrtm(np.array([[5., 2.], [2., 2.]])))
Xc = Z @ Shalf.T
check("fig whitening: Sigma^-1/2 x recovers z", np.allclose(np.linalg.solve(Shalf, Xc.T).T, Z))
figdata("ch01cloudx", " ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in Xc))
figdata("ch01cloudz", " ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in Z))
# fig Hadamard spreading, d = 8
H8 = sylvester(8); s8 = np.random.default_rng(0).choice([-1., 1.], size=8)
R8 = H8 @ np.diag(s8) / np.sqrt(8)
x8 = np.array([2., 0.4, 0, 0, 0, 0, 0, 0])
y8 = R8 @ x8
check("fig Hadamard: R orthogonal and norm preserved", np.allclose(R8 @ R8.T, np.eye(8)) and close(np.linalg.norm(y8), np.linalg.norm(x8)))
check("fig Hadamard: largest magnitude drops from 2 to below 0.85", np.max(np.abs(y8)) < 0.85)
check("fig Hadamard: magnitudes are |2 +- 0.4|/sqrt8", np.allclose(np.sort(np.unique(np.round(np.abs(y8), 9))), np.sort(np.unique(np.round([2.4 / np.sqrt(8), 1.6 / np.sqrt(8)], 9)))))
figdata("ch01hadx", " ".join(f"({i+1},{v:.3f})" for i, v in enumerate(x8)))
figdata("ch01hady", " ".join(f"({i+1},{v:.3f})" for i, v in enumerate(y8)))
figdata("ch01hadmax", f"{2.4/np.sqrt(8):.3f}")
# fig Rayleigh quotient of pencil (A,B) vs angle
An = np.array([[2., 2.], [2., 8.]]); Bn = np.diag([1., 4.])
angs = np.arange(0, 181, 5)
rq = [(np.array([np.cos(np.radians(t)), np.sin(np.radians(t))]) @ An @ np.array([np.cos(np.radians(t)), np.sin(np.radians(t))])) /
      (np.array([np.cos(np.radians(t)), np.sin(np.radians(t))]) @ Bn @ np.array([np.cos(np.radians(t)), np.sin(np.radians(t))])) for t in angs]
fine = np.linspace(0, 180, 18001)
rqf = [(np.cos(np.radians(t)) ** 2 * 2 + 4 * np.cos(np.radians(t)) * np.sin(np.radians(t)) + 8 * np.sin(np.radians(t)) ** 2) /
       (np.cos(np.radians(t)) ** 2 + 4 * np.sin(np.radians(t)) ** 2) for t in fine]
check("fig Rayleigh: max 3 at 26.565 deg, min 1 at 153.435 deg", close(max(rqf), 3, 1e-6) and close(fine[int(np.argmax(rqf))], 26.565, 0.01) and close(min(rqf), 1, 1e-6) and close(fine[int(np.argmin(rqf))], 153.435, 0.01))
check("fig Rayleigh: value 2 at 0 deg", close(rq[0], 2))
figdata("ch01rayleigh", " ".join(f"({t},{v:.4f})" for t, v in zip(angs, rq)))

check("fig plane: |x|^2 = 14 = 12 + 2", 9 + 1 + 4 == 14 and 4 * 3 == 12 and 1 + 1 == 2)
# Loewner picture: support function sqrt(u^T S u) compares ellipses
D31 = np.diag([3., 1.]); C21 = np.array([[2., 1.], [1., 2.]])
e1_ = np.array([1., 0.]); u45 = np.array([1., 1.]) / np.sqrt(2)
check("fig Loewner: along e1 diag(3,1) reaches 1.732 > 1.414 of the other", close(np.sqrt(e1_ @ D31 @ e1_), 1.732, 5e-4) and close(np.sqrt(e1_ @ C21 @ e1_), 1.414, 5e-4))
check("fig Loewner: along 45 deg the other reaches 1.732 > 1.414; tip (1.225,1.225)", close(np.sqrt(u45 @ C21 @ u45), 1.732, 5e-4) and close(np.sqrt(u45 @ D31 @ u45), 1.414, 5e-4) and close(np.sqrt(3) / np.sqrt(2), 1.225, 5e-4))
okE = True
for _ in range(200):
    S1_ = rng.standard_normal((2, 2)); S1_ = S1_ @ S1_.T; S2_ = rng.standard_normal((2, 2)); S2_ = S2_ @ S2_.T
    th_ = np.linspace(0, np.pi, 721); U_ = np.stack([np.cos(th_), np.sin(th_)])
    inside = np.all(np.einsum('iu,ij,ju->u', U_, S2_, U_) <= np.einsum('iu,ij,ju->u', U_, S1_, U_) + 1e-12)
    okE &= (inside == is_psd(S1_ - S2_)) or abs(min(np.linalg.eigvalsh(S1_ - S2_))) < 1e-3
check("ellipse of S2 inside ellipse of S1 iff S1 >= S2 (random)", okE)
check("fig Rayleigh: extreme directions 126.87 deg apart, not orthogonal", close(153.435 - 26.565, 126.87, 1e-9))
check("fig Hadamard: 2.4/sqrt8 ~ 0.849", close(2.4 / np.sqrt(8), 0.849, 5e-4))

# ---------------------------------------------------------------- hypothesis-discipline additions
# G-projector is symmetric when the subspace is spanned by an eigenvector of G
Gd = sp.diag(2, 1); a_ = M_([1, 0])
PGe = a_ * (a_.T * Gd * a_).inv() * a_.T * Gd
check("Pi_G symmetric when span is an eigenvector of G", PGe == PGe.T)
check("Pi_G symmetric when G = cI", (lambda P_: P_ == P_.T)(u * (u.T * (3 * sp.eye(2)) * u).inv() * u.T * (3 * sp.eye(2))))
# a few consumers agreeing does not give the Loewner order
C09 = M_([[1, R(9, 10)], [R(9, 10), 1]])
check("I and [[1,.9],[.9,1]] tie for e1 e1^T and e2 e2^T", C09[0, 0] == 1 and C09[1, 1] == 1)
check("I - [[1,.9],[.9,1]] has eigenvalues +-0.9 (incomparable)", set((sp.eye(2) - C09).eigenvals().keys()) == {R(9, 10), -R(9, 10)})
# indefinite pair not simultaneously diagonalizable by a real congruence
mu_ = sp.symbols('mu')
Ai = sp.diag(1, -1); Bi = M_([[0, 1], [1, 0]])
dp = sp.expand((Ai - mu_ * Bi).det())
check("det(diag(1,-1) - mu [[0,1],[1,0]]) = -1 - mu^2", dp == -1 - mu_ ** 2)
check("... which has no real root", all(not r.is_real for r in sp.solve(dp, mu_)))
# Rayleigh quotient unbounded when the denominator form is only PSD
rq_ = [(e ** 2 + 1) / e ** 2 for e in [1e-1, 1e-2, 1e-3]]
check("x^T I x / x^T diag(1,0) x grows without bound as x -> e2 (101, 10001, 1000001)", close(rq_[0], 101, 1e-6) and close(rq_[1], 10001, 1e-4) and close(rq_[2], 1000001, 1e-1))
# strict concavity of log det: tr(E^2) > 0 for symmetric E != 0 (random)
okS = all(np.trace(E_ @ E_) > 0 for E_ in [(lambda Z: Z + Z.T)(rng.standard_normal((3, 3))) for _ in range(100)])
check("tr(E^2) > 0 for random symmetric E != 0", okS)
# smaller determinant does not imply Loewner-smaller
check("det diag(1,1) = 1 < det diag(4,1/2) = 2", sp.diag(1, 1).det() == 1 and sp.diag(4, R(1, 2)).det() == 2)
check("diag(1,1) and diag(4,1/2) incomparable", set(sp.diag(4 - 1, R(1, 2) - 1).eigenvals().keys()) == {3, -R(1, 2)})
# naive kernel is exact when Sigma_x = diag(2,1), Phat = e1 e1^T
Sd_ = np.diag([2., 1.]); Sh_ = np.sqrt(Sd_)
ok_naive = True
for _ in range(50):
    g_ = rng.standard_normal((2, 2)); P_ = g_ @ g_.T
    Pt_ = Sh_ @ P_ @ Sh_
    K_ = Sh_ @ np.diag([1., 0.]) @ Sh_
    w_, V_ = np.linalg.eigh(K_); k_ = V_[:, np.argmin(np.abs(w_))]
    Pi_w = np.outer(k_, k_)
    Pi_naive = np.diag([0., 1.])
    ok_naive &= close(np.trace(Pt_ @ Pi_w), np.trace(Pt_ @ Pi_naive), 1e-9)
check("Sigma=diag(2,1), Phat=e1e1^T: whitened kernel = span(e2), naive floor exact", ok_naive)
# Example floor: Schur complement 1/2 is also the best linear predictor error variance
S2 = np.array([[2., 1.], [1., 1.]])
check("best linear predictor of x2 from x1 has error variance 1/2", close(S2[1, 1] - S2[0, 1] ** 2 / S2[0, 0], 0.5))

# ---------------------------------------------------------------- turboquant-pro two-window Hadamard (2026-10-07 fix)
def fwht(v):
    v = v.copy(); h = 1; n = v.shape[-1]
    while h < n:
        for i in range(0, n, 2 * h):
            a = v[..., i:i + h].copy(); b = v[..., i + h:i + 2 * h].copy()
            v[..., i:i + h] = a + b; v[..., i + h:i + 2 * h] = a - b
        h *= 2
    return v


def two_window(x, d, rg):
    m = 1 << (d.bit_length() - 1)
    s1 = rg.choice([-1.0, 1.0], size=d); p1 = rg.permutation(d)
    s2 = rg.choice([-1.0, 1.0], size=d); q2 = rg.permutation(m)
    y = (x * s1)[..., p1]
    y = np.concatenate((fwht(y[..., :m]) / np.sqrt(m), y[..., m:]), axis=-1)
    if m != d:
        y = y * s2; lo = d - m
        w = fwht(y[..., lo:][..., q2]) / np.sqrt(m)
        y = np.concatenate((y[..., :lo], w), axis=-1)
    return y


rgT = np.random.default_rng(1007)
for d_ in [12, 6144]:
    m_ = 1 << (d_.bit_length() - 1)
    check(f"d={d_}: m={m_} > d/2 and windows [0,m), [d-m,d) cover [0,d)", m_ > d_ / 2 and d_ - m_ < m_)
Rm = two_window(np.eye(12), 12, np.random.default_rng(7))
check("two-window rotation is orthogonal (d=12)", np.allclose(Rm @ Rm.T, np.eye(12)))
yT = two_window(np.eye(6144)[0], 6144, rgT)
check("d=6144: one-hot input keeps length 1 and its largest coordinate falls below 0.05",
      close(np.linalg.norm(yT), 1.0) and np.max(np.abs(yT)) < 0.05)
check("d=6144: windows are [0,4096) and [2048,6144)", (1 << (6144).bit_length() - 1) == 4096 and 6144 - 4096 == 2048)
check("d=8192 is a power of two, so only one window is used", (1 << (8192).bit_length() - 1) == 8192)
sgn = np.random.default_rng(3).choice([-1.0, 1.0], size=8); perm = np.random.default_rng(4).permutation(8)
check("sign flip plus permutation leaves a one-hot vector one-hot (spreads no energy)",
      np.count_nonzero((np.eye(8)[0] * sgn)[perm]) == 1)
# numbers quoted from turboquant-pro CHANGELOG.md, 2026-10-07
check("quoted: old path 0.89 vs spreading 0.054, more than ten times worse", 0.89 / 0.054 > 10)
check("quoted: 0.0444 lies within one SE (0.0034) of 0.0453", abs(0.0444 - 0.0453) < 0.0034)

n_pass = sum(results)
print(f"{n_pass}/{len(results)} PASS")
sys.exit(0 if n_pass == len(results) else 1)
