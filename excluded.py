"""Mean excluded area <A_ex>(k) of two randomly oriented k-armed stars (arm length a=0.5).
Also the contact probability P(r) as a function of center distance r (for theory)."""
import numpy as np
from numba import njit, prange
import sys
sys.path.insert(0, ".")
from sim import stars_touch, A


@njit(parallel=True, cache=True)
def pr_curve(k, rs, nsamp, seed):
    out = np.zeros(len(rs))
    for i in prange(len(rs)):
        np.random.seed(seed + i)
        r = rs[i]; c = 0
        for s in range(nsamp):
            thA = np.random.random() * 2 * np.pi
            thB = np.random.random() * 2 * np.pi
            if stars_touch(r, 0.0, thA, thB, k):
                c += 1
        out[i] = c / nsamp
    return out


if __name__ == "__main__":
    ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]
    nr = 400
    # midpoint rule in r on [0, 2a]
    rs = (np.arange(nr) + 0.5) / nr * 2 * A
    res = {}
    for k in ks:
        ns = 200000 if k <= 32 else 60000
        if k == 1:
            # k=1: single arm of length a (stick of length a); use exact formula
            res[k] = (2 / np.pi) * A * A
            continue
        p = pr_curve(k, rs, ns, 7 + k)
        aex = np.sum(2 * np.pi * rs * p) * (2 * A / nr)
        res[k] = aex
        np.save(f"data/Pr_k{k}.npy", np.vstack([rs, p]))
        print(k, "A_ex/a^2 =", aex / A**2, "deficit (4pi - A_ex/a^2) =", 4 * np.pi - aex / A**2, flush=True)
    np.save("data/aex.npy", np.array([[k, res[k] / A**2] for k in ks]))
