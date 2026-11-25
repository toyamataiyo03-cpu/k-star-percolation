"""High-precision thresholds from data2/ (histogram-based grand-canonical estimator + bootstrap)."""
import numpy as np, glob, re, json
from scipy.stats import poisson
from scipy.optimize import brentq

A = 0.5
RH = 0.521058290; RE = 0.690473725


def load(d="data2"):
    out = {}
    for f in glob.glob(f"{d}/k*_L*_c*.npy"):
        k, L, c = map(int, re.findall(r"k(\d+)_L(\d+)_c(\d+)", f)[0])
        out.setdefault((k, L), []).append(np.load(f))
    return {key: np.vstack(v) for key, v in out.items()}


def est_hist(vals, wts, L, Rstar):
    W = wts.sum()
    def R(lam):
        return (poisson.sf(vals - 1, lam) * wts).sum() / W
    lo, hi = vals.min() * 0.8, vals.max() * 1.2
    return brentq(lambda x: R(x) - Rstar, lo, hi, xtol=1e-7) / L**2 * A**2


def est(N, L, Rstar, nb=100, seed=0):
    vals, cnt = np.unique(N, return_counts=True)
    v = est_hist(vals, cnt, L, Rstar)
    rng = np.random.default_rng(seed)
    p = cnt / cnt.sum()
    bs = [est_hist(vals, rng.multinomial(cnt.sum(), p), L, Rstar) for _ in range(nb)]
    return v, float(np.std(bs))


if __name__ == "__main__":
    d = load()
    res = {}
    for (k, L) in sorted(d):
        arr = d[(k, L)]
        h, eh = est(arr[:, 0], L, RH)
        e, ee = est(arr[:, 1], L, RE)
        res[f"{k}_{L}"] = dict(k=k, L=L, n=len(arr), h=h, eh=eh, e=e, ee=ee)
        print(f"k={k:4d} L={L:4d} n={len(arr):7d}  h={h:.6f}({eh:.6f})  e={e:.6f}({ee:.6f})", flush=True)
    json.dump(res, open("data2/thresholds.json", "w"), indent=1)
