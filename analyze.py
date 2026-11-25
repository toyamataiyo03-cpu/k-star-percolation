"""Grand-canonical threshold estimates n_c(L) from first-wrapping data, then L->inf extrapolation."""
import numpy as np, glob, re, json
from scipy.stats import poisson
from scipy.optimize import brentq, curve_fit

A = 0.5
RH = 0.521058290  # horizontal wrapping prob. at p_c, square torus (Pinson 1994)
RE = 0.690473725  # wrapping in either direction


def load():
    d = {}
    for f in glob.glob("data/k*_L*_c*.npy"):
        k, L, c = map(int, re.findall(r"k(\d+)_L(\d+)_c(\d+)", f)[0])
        d.setdefault((k, L), []).append(np.load(f))
    return {key: np.vstack(v) for key, v in d.items()}


def est(N, L, Rstar):
    N = np.sort(N)
    def R(lam):
        return poisson.sf(N - 1, lam).mean()  # P(Pois(lam) >= N)
    lo, hi = N.min() * 0.5, N.max() * 1.5
    lam = brentq(lambda x: R(x) - Rstar, lo, hi, xtol=1e-6)
    return lam / L**2 * A**2


def est_err(N, L, Rstar, nb=40, rng=np.random.default_rng(0)):
    v = est(N, L, Rstar)
    bs = [est(rng.choice(N, len(N)), L, Rstar) for _ in range(nb)]
    return v, np.std(bs)


def extrap(Ls, ys, es, power=2.0):
    Ls = np.asarray(Ls, float)
    f = lambda L, a, b: a + b * L ** (-power)
    p, cov = curve_fit(f, Ls, ys, sigma=es, absolute_sigma=True)
    chi2 = np.sum(((f(Ls, *p) - ys) / es) ** 2)
    return p[0], np.sqrt(cov[0, 0]), chi2, len(Ls) - 2


if __name__ == "__main__":
    d = load()
    ks = sorted({k for k, _ in d})
    out = {}
    for k in ks:
        Ls = sorted(L for kk, L in d if kk == k)
        rows = []
        for L in Ls:
            arr = d[(k, L)]
            h, eh = est_err(arr[:, 0], L, RH)
            e, ee = est_err(arr[:, 1], L, RE)
            rows.append((L, len(arr), h, eh, e, ee))
            print(f"k={k:3d} L={L:4d} n={len(arr):6d}  h: {h:.6f}±{eh:.6f}  e: {e:.6f}±{ee:.6f}")
        out[k] = rows
        if len(Ls) >= 3:
            r = np.array(rows)
            # use L>=32 for extrapolation; combine h and e estimates separately
            m = r[:, 0] >= 32
            for name, ci, cei in (("h", 2, 3), ("e", 4, 5)):
                if m.sum() >= 3:
                    a, ea, chi2, dof = extrap(r[m, 0], r[m, ci], r[m, cei])
                    print(f"   extrap[{name}] L>=32: n_c a^2 = {a:.6f} ± {ea:.6f}  chi2/dof={chi2:.2f}/{dof}")
                a, ea, chi2, dof = extrap(r[:, 0], r[:, ci], r[:, cei])
                print(f"   extrap[{name}] all L : n_c a^2 = {a:.6f} ± {ea:.6f}  chi2/dof={chi2:.2f}/{dof}")
    json.dump({str(k): v for k, v in out.items()}, open("data/finiteL.json", "w"), indent=1)
