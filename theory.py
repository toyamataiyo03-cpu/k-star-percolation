"""Semi-analytic excluded area of k-armed stars for large k.

Two arms (angles alpha, beta measured from the line of centres, towards each other,
same side) of length a with centre distance r intersect iff
    r sin(beta) <= a sin(alpha+beta)  and  r sin(alpha) <= a sin(alpha+beta).
For a star the pairs (alpha_i, beta_j) form a square lattice of spacing h=2pi/k with
a uniformly random offset. Treating positions along the diagonal as continuous
(valid for k >> 1), contact occurs iff the offset delta0 = alpha-beta (mod h) lies in the
projection of the contact set S(r) on the delta axis, of half-width dmax(r):
    P(r) = min(1, 2 dmax(r)/h).
Small-angle limit: dmax = c eps^{3/2}, c=(4/3)sqrt(2/3), r = 2a(1-eps), giving
    4pi a^2 - <A_ex> ~ C k^{-2/3},  C = (24 pi/5) (pi/c)^{2/3}.
"""
import numpy as np
from scipy.optimize import minimize_scalar

a = 1.0
c = 4 / 3 * np.sqrt(2 / 3)
C = 24 * np.pi / 5 * (np.pi / c) ** (2 / 3)


def dmax(r):
    # maximise alpha-beta s.t. r sin(alpha) <= a sin(alpha+beta), 0<beta<=alpha<pi/2
    # for fixed u=(alpha+beta)/2 the feasible delta is up to the root of
    # r sin(u+d/2) = a sin(2u); scan u.
    best = 0.0
    us = np.linspace(1e-6, np.pi / 2, 4000)
    for u in us:
        rhs = a * np.sin(2 * u) / r
        if rhs < np.sin(u):
            continue
        if rhs >= 1:
            d = 2 * (np.pi / 2 - u)  # alpha can reach pi/2 ... bounded by beta>=0: d<=2u
        else:
            d = 2 * (np.arcsin(rhs) - u)
        d = min(d, 2 * u)
        best = max(best, d)
    return best


def aex_semi(k, nr=600):
    h = 2 * np.pi / k
    rs = (np.arange(nr) + 0.5) / nr * 2 * a
    P = np.array([min(1.0, 2 * dmax(r) / h) for r in rs])
    return np.sum(2 * np.pi * rs * P) * (2 * a / nr), rs, P


if __name__ == "__main__":
    # small-angle check
    for eps in (1e-2, 1e-3):
        print("dmax/eps^1.5", dmax(2 * a * (1 - eps)) / eps**1.5, "c=", c)
    print("C =", C)
    rows = []
    for k in (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024):
        A, _, _ = aex_semi(k)
        rows.append((k, A, 4 * np.pi - A, C * k ** (-2 / 3)))
        print(k, "semi A_ex/a^2=%.4f deficit=%.4f  leading=%.4f" % rows[-1][1:])
    np.save("data/aex_semi.npy", np.array(rows))
