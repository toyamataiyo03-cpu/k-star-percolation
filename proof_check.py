"""Numerical checks of each step of the proof (a = 1)."""
import numpy as np, json
from numba import njit

c = 4 * np.sqrt(6) / 9
C = 18 / 5 * 2 ** (1 / 3) * np.pi ** (5 / 3)
Q = (np.pi / c) ** (2 / 3)          # eps* = Q k^{-2/3}
C2 = C * (-20 / 21) * Q
NC_DISK = 1.12808737 / np.pi


def g(u, eps):
    w = np.sin(2 * u) / (2 * (1 - eps))
    return np.where(w <= 1, 2 * (np.arcsin(np.minimum(w, 1)) - u), np.nan)


def dmax(eps, n=200001):
    um = np.arccos(1 - eps)
    u = np.linspace(0, um, n)
    return np.nanmax(g(u, eps))


print("--- Step 1: delta_max expansion ---")
for eps in (0.1, 0.03, 0.01, 0.003, 0.001):
    ex = dmax(eps); a1 = c * eps**1.5; a2 = a1 * (1 + 1.25 * eps)
    print(f"eps={eps:6.3f}  exact={ex:.6e}  lead rel.err={ex/a1-1:+.2e}  2-term rel.err={ex/a2-1:+.2e}")

print("--- Step 2: concavity of g on (0, arccos(1-eps)) ---")
worst = -np.inf
for eps in np.linspace(1e-4, 0.3, 300):
    um = np.arccos(1 - eps); u = np.linspace(um * 1e-3, um * (1 - 1e-6), 4000)
    gg = g(u, eps); d2 = np.gradient(np.gradient(gg, u), u)
    worst = max(worst, np.nanmax(d2[2:-2]))
print("max g'' over eps<=0.3:", worst, "(negative => concave)")


@njit(cache=True)
def touch(alpha, beta, r):
    # arms at angles alpha (from A towards B) and beta (from B towards A), same side
    if alpha <= 0 or beta <= 0 or alpha + beta >= np.pi:
        return False
    s = np.sin(alpha + beta)
    return r * np.sin(beta) <= s and r * np.sin(alpha) <= s


@njit(cache=True)
def all_shifts_hit(eps, k, N):
    """True iff, on an N x N grid of lattice shifts, every shift gives contact (upper side only suffices
    for the covering test; lower side is the mirror image)."""
    h = 2 * np.pi / k; r = 2 * (1 - eps)
    umax = np.arccos(1 - eps) * 2 + h
    m = int(umax / h) + 2
    for a in range(N):
        for b in range(N):
            s1 = (a + 0.5) * h / N; s2 = (b + 0.5) * h / N
            hit = False
            for i in range(m):
                if hit: break
                for j in range(m):
                    if touch(s1 + i * h, s2 + j * h, r) or touch(-(s1 - (i + 1) * h), -(s2 - (j + 1) * h), r):
                        hit = True; break
            if not hit:
                return False
    return True


def eps1(k, N=300):
    lo, hi = 0.0, 0.9
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if all_shifts_hit(mid, k, N):
            hi = mid
        else:
            lo = mid
    return hi


print("--- Step 3: excluded-area deficit, theory vs semi-analytic vs Monte Carlo ---")
semi = {int(r[0]): r[2] for r in np.load("data/aex_semi.npy")}
mc = {int(k): 4 * np.pi - v for k, v in np.load("data/aex.npy")}
for k in (16, 32, 64, 128, 256, 512, 1024):
    t1 = C * k ** (-2 / 3); t2 = t1 + C2 * k ** (-4 / 3)
    print(f"k={k:5d}  1-term={t1:.4f}  2-term={t2:.4f}  semi={semi.get(k, np.nan):.4f}  MC={mc.get(k, np.nan):.4f}")

print("--- Step 4: rigorous sandwich n_c^disk <= n_c(k) <= n_c^disk/(1-eps1)^2 ---")
R = json.load(open("data/results.json"))
out = []
for k in (8, 12, 16, 24, 32, 48, 64, 128, 256, 1024):
    e1 = eps1(k)
    up = NC_DISK / (1 - e1) ** 2
    meas = R.get(str(k), {}).get("nc", np.nan)
    print(f"k={k:5d}  eps1={e1:.4f}  eps*={Q*k**(-2/3):.4f}  bounds [{NC_DISK:.4f}, {up:.4f}]  measured={meas:.4f}")
    out.append((k, e1, up, meas))
np.save("data/sandwich.npy", np.array(out))
