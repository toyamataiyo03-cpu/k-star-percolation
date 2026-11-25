import numpy as np, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, ".")
from sim import stars_touch
from theory import C as C_LEAD
from analyze import load, est_err, extrap, RH, RE

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.7,
                     "figure.dpi": 200})
STICK = 5.6372858 / 4        # n_c a^2 for k=2 (stick length 2a)
DISK = 1.12808737 / np.pi    # n_c a^2 for k->inf (disk radius a)
BC_STICK = 5.6372858 * 2 / np.pi
BC_DISK = 4 * 1.12808737

# ---------- thresholds ----------
d = load()
ks = sorted({k for k, _ in d})
res = {}
fin = {}
for k in ks:
    Ls = sorted(L for kk, L in d if kk == k)
    rows = []
    for L in Ls:
        arr = d[(k, L)]
        h, eh = est_err(arr[:, 0], L, RH, nb=60)
        e, ee = est_err(arr[:, 1], L, RE, nb=60)
        rows.append([L, len(arr), h, eh, e, ee])
    r = np.array(rows); fin[k] = r
    m = r[:, 0] >= 32
    ah, eah, c2h, dofh = extrap(r[m, 0], r[m, 2], r[m, 3])
    ae, eae, c2e, dofe = extrap(r[m, 0], r[m, 4], r[m, 5])
    # combine h and e (correlated; take mean and the larger error plus half the spread)
    nc = 0.5 * (ah + ae)
    err = max(eah, eae) + 0.5 * abs(ah - ae)
    res[k] = dict(nc=nc, err=err, h=ah, eh=eah, e=ae, ee=eae, chi2h=c2h, chi2e=c2e, dof=dofh)
    print(f"k={k:3d}  h={ah:.5f}({eah:.5f}) e={ae:.5f}({eae:.5f}) chi2 {c2h:.1f},{c2e:.1f}/{dofh} -> {nc:.5f}±{err:.5f}")

aex = dict((int(k), v) for k, v in np.load("data/aex.npy"))
semi = np.load("data/aex_semi.npy")
for k in ks:
    res[k]["aex"] = aex[k]
    res[k]["Bc"] = res[k]["nc"] * aex[k]
    res[k]["eBc"] = res[k]["err"] * aex[k]
json.dump({str(k): v for k, v in res.items()}, open("data/results.json", "w"), indent=1)
K = np.array(ks)
NC = np.array([res[k]["nc"] for k in ks]); ENC = np.array([res[k]["err"] for k in ks])

# ---------- Fig 1: star shapes + snapshot ----------
fig = plt.figure(figsize=(6.8, 2.6))
ax0 = fig.add_axes([0.0, 0.04, 0.5, 0.92])
for i, k in enumerate([2, 3, 4, 6, 12, 48]):
    cx = (i % 3) * 2.6; cy = -(i // 3) * 3.0
    th = 0.3 + 2 * np.pi * np.arange(k) / k
    for t in th:
        ax0.plot([cx, cx + np.cos(t)], [cy, cy + np.sin(t)], color="k", lw=1.0 if k < 20 else 0.5)
    circ = plt.Circle((cx, cy), 1, fill=False, ls=":", lw=0.5, color="0.5")
    ax0.add_patch(circ)
    ax0.text(cx, cy - 1.35, f"$k={k}$", ha="center", va="top")
ax0.set_xlim(-1.3, 6.5); ax0.set_ylim(-5.0, 1.15); ax0.set_aspect("equal"); ax0.axis("off")
ax0.text(-1.3, 1.3, "(a)", va="top")

ax1 = fig.add_axes([0.56, 0.03, 0.40, 0.94])
rng = np.random.default_rng(3); k = 3; Lb = 12.0; A = 0.5
n = int(res[3]["nc"] / A**2 * Lb * Lb)
X = rng.random(n) * Lb; Y = rng.random(n) * Lb; T = rng.random(n) * 2 * np.pi / k
par = np.arange(n)
def f(i):
    while par[i] != i:
        par[i] = par[par[i]]; i = par[i]
    return i
for i in range(n):
    for j in range(i + 1, n):
        dx = X[j] - X[i]; dy = Y[j] - Y[i]
        if dx * dx + dy * dy < 1 and stars_touch(dx, dy, T[i], T[j], k):
            par[f(i)] = f(j)
roots = np.array([f(i) for i in range(n)])
u, cnt = np.unique(roots, return_counts=True)
big = u[np.argmax(cnt)]
segs, cols = [], []
for i in range(n):
    for j in range(k):
        t = T[i] + 2 * np.pi * j / k
        segs.append([(X[i], Y[i]), (X[i] + A * np.cos(t), Y[i] + A * np.sin(t))])
        cols.append("#c0392b" if roots[i] == big else "0.65")
ax1.add_collection(LineCollection(segs, colors=cols, linewidths=0.6))
ax1.set_xlim(0, Lb); ax1.set_ylim(0, Lb); ax1.set_aspect("equal")
ax1.set_xticks([]); ax1.set_yticks([])
ax1.text(-0.6, Lb, "(b)", va="top", ha="right")
fig.savefig("fig1_stars.png", dpi=300); plt.close(fig)

# ---------- Fig 2: n_c vs k ----------
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
ax = axs[0]
ax.errorbar(K, NC, yerr=ENC, fmt="o", ms=3.5, color="k", mfc="w", label="this work")
ax.axhline(STICK, ls="--", lw=0.7, color="C0"); ax.text(4, STICK + 0.03, "stick ($k=2$, exact ref.)", fontsize=7, color="C0")
ax.axhline(DISK, ls="--", lw=0.7, color="C3"); ax.text(4, DISK + 0.03, "disk ($k\\to\\infty$)", fontsize=7, color="C3")
# excluded-area prediction with disk B_c
kk = semi[:, 0]
ax.plot(kk, BC_DISK / semi[:, 1], ":", color="0.4", lw=0.9, label=r"$B_c^{\rm disk}/\langle A_{\rm ex}\rangle$")
ax.set_xscale("log"); ax.set_xlabel("$k$"); ax.set_ylabel(r"$n_c a^2$")
ax.set_xlim(1.7, 300); ax.set_ylim(0.3, 1.55); ax.legend(fontsize=7, frameon=False, loc="center right")
ax.text(0.97, 0.97, "(a)", transform=ax.transAxes, va="top", ha="right")
ax = axs[1]
for k_, mk in ((2, "o"), (4, "s"), (16, "^"), (64, "D")):
    r = fin[k_]
    ax.errorbar(r[:, 0] ** -2.0, r[:, 2] / res[k_]["h"], yerr=r[:, 3] / res[k_]["h"], fmt=mk, ms=3, mfc="w", label=f"$k={k_}$")
ax.set_xlabel("$L^{-2}$  ($L$ in units of $2a$)"); ax.set_ylabel(r"$n_c(L)/n_c(\infty)$")
ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower center", handletextpad=0.1, columnspacing=0.6); ax.set_ylim(0.9975, 1.0035); ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, va="top")
fig.tight_layout(); fig.savefig("fig2_threshold.png", dpi=300); plt.close(fig)

# ---------- Fig 3: excluded-area deficit ----------
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
ax = axs[0]
ka = np.array(sorted(aex)); va = np.array([aex[k] for k in ka])
m = ka >= 2
ax.loglog(ka[m], 4 * np.pi - va[m], "o", ms=3.5, mfc="w", color="k", label="Monte Carlo")
ax.loglog(semi[:, 0], semi[:, 2], "-", color="C0", lw=0.9, label="upper bound of Eq. (6)")
ax.loglog(semi[:, 0], semi[:, 3], "--", color="C3", lw=0.9, label=r"$C\,k^{-2/3}$")
kq = semi[:, 0]; C2 = 36 / 7 * 2 ** (-1 / 3) * np.pi ** (7 / 3)
ax.loglog(kq[kq >= 8], (C_LEAD * kq ** (-2 / 3) - C2 * kq ** (-4 / 3))[kq >= 8], "-.", color="C2", lw=0.9,
          label=r"$Ck^{-2/3}-C'k^{-4/3}$, Eq. (7)")
ax.set_xlabel("$k$"); ax.set_ylabel(r"$4\pi - \langle A_{\rm ex}\rangle/a^2$")
ax.legend(fontsize=7, frameon=False); ax.text(0.02, 0.05, "(a)", transform=ax.transAxes)
ax = axs[1]
for kk_ in (8, 32, 128):
    try:
        rs, p = np.load(f"data/Pr_k{kk_}.npy")
    except FileNotFoundError:
        continue
    eps = 1 - rs / (2 * 0.5)
    from theory import dmax
    h = 2 * np.pi / kk_
    epsg = np.linspace(1e-4, 0.6, 300)
    ps = [min(1, 2 * dmax(2 * (1 - e)) / h) for e in epsg]
    l, = ax.plot(eps, p, ".", ms=2, label=f"$k={kk_}$")
    ax.plot(epsg, ps, "-", lw=0.7, color=l.get_color())
ax.set_xlim(0, 0.5); ax.set_xlabel(r"$\varepsilon = 1 - r/2a$"); ax.set_ylabel(r"$P(r)$")
ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.text(0.02, 0.92, "(b)", transform=ax.transAxes)
fig.tight_layout(); fig.savefig("fig3_excluded.png", dpi=300); plt.close(fig)

# ---------- Fig 4: B_c ----------
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
BC = np.array([res[k]["Bc"] for k in ks]); EBC = np.array([res[k]["eBc"] for k in ks])
ax = axs[0]
ax.errorbar(K, BC, yerr=EBC, fmt="o", ms=3.5, color="k", mfc="w")
HP = json.load(open("data2/conjecture_test.json")); KH = np.array(sorted(int(x) for x in HP))
ax.errorbar(KH, [HP[str(k)]["Bc"] for k in KH], yerr=[HP[str(k)]["eBc"] for k in KH], fmt="s", ms=3.5, color="C1", label="high precision")
ax.axhline(BC_STICK, ls="--", lw=0.7, color="C0"); ax.axhline(BC_DISK, ls="--", lw=0.7, color="C3")
ax.text(100, BC_STICK + 0.03, "stick", color="C0", fontsize=7); ax.text(100, BC_DISK - 0.12, "disk", color="C3", fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("$k$"); ax.set_ylabel(r"$B_c = n_c\langle A_{\rm ex}\rangle$")
ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512]); ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256", "512"]); ax.minorticks_off()
ax.text(0.96, 0.45, "(a)", transform=ax.transAxes, ha="right")
ax = axs[1]
dn = NC - DISK
mk = K >= 6
ax.loglog(K, dn, "o", ms=3.5, mfc="w", color="k", label=r"$n_c a^2 - n_c^{\rm disk}a^2$")
ax.loglog(KH, [HP[str(k)]["dn"] for k in KH], "s", ms=3.5, color="C1", label="high precision")
kk = semi[:, 0]
ax.loglog(kk, BC_DISK / semi[:, 1] - DISK, ":", color="0.4", lw=0.9, label="excluded-area estimate")
ax.loglog(kk, BC_DISK * C_LEAD / (16 * np.pi**2) * kk ** (-2 / 3), "--", color="C3", lw=0.7, label=r"$0.873\,k^{-2/3}$, Eq. (9)")
sw = np.load("data/sandwich.npy")
ax.loglog(sw[:, 0], DISK / (1 - sw[:, 1]) ** 2 - DISK, "^-", ms=3, color="C0", lw=0.7, mfc="w",
          label="rigorous upper bound, Eq. (8)")
ax.set_xlabel("$k$"); ax.legend(fontsize=7, frameon=False)
ax.text(0.02, 0.05, "(b)", transform=ax.transAxes)
fig.tight_layout(); fig.savefig("fig4_Bc.png", dpi=300); plt.close(fig)
# ---------- Fig 5: test of the conjecture ----------
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
ax = axs[0]
ratio_old = [(k, (res[k]["nc"] - DISK) / (0.873 * k ** (-2 / 3)), res[k]["err"] / (0.873 * k ** (-2 / 3))) for k in ks if k >= 16 and k not in KH]
ax.errorbar([r[0] for r in ratio_old], [r[1] for r in ratio_old], yerr=[r[2] for r in ratio_old], fmt="o", ms=3.5, color="k", mfc="w")
ax.errorbar(KH, [HP[str(k)]["ratio_asym"] for k in KH], yerr=[HP[str(k)]["enc"] / (0.873 * k ** (-2 / 3)) for k in KH], fmt="s", ms=3.5, color="C1")
ax.axhline(1, ls="--", lw=0.7, color="C3")
ax.set_xscale("log"); ax.set_xlabel("$k$"); ax.set_ylabel(r"$(n_c-n_c^{\rm disk})a^2\,/\,0.873k^{-2/3}$")
ax.set_xticks([16, 32, 64, 128, 256, 512]); ax.set_xticklabels(["16", "32", "64", "128", "256", "512"]); ax.minorticks_off()
ax.text(0.03, 0.08, "(a)", transform=ax.transAxes)
ax = axs[1]
bd_old = [(k, BC_DISK - res[k]["Bc"], res[k]["eBc"]) for k in ks if k >= 16 and k not in KH]
ax.errorbar([r[0] for r in bd_old], [r[1] for r in bd_old], yerr=[r[2] for r in bd_old], fmt="o", ms=3.5, color="k", mfc="w")
ax.errorbar(KH, [BC_DISK - HP[str(k)]["Bc"] for k in KH], yerr=[HP[str(k)]["eBc"] for k in KH], fmt="s", ms=3.5, color="C1")
kk2 = np.array([16, 512.0])
ax.loglog(kk2, 0.06 * (kk2 / 16) ** (-2 / 3), "--", color="0.5", lw=0.7, label=r"slope $-2/3$")
ax.loglog(kk2, 0.06 * (kk2 / 16) ** (-1.4), ":", color="C3", lw=0.9, label=r"slope $-1.4$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$k$"); ax.set_ylabel(r"$B_c^{\rm disk}-B_c$")
ax.set_xticks([16, 32, 64, 128, 256, 512]); ax.set_xticklabels(["16", "32", "64", "128", "256", "512"]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.legend(fontsize=7, frameon=False); ax.text(0.03, 0.08, "(b)", transform=ax.transAxes)
fig.tight_layout(); fig.savefig("fig5_conjecture.png", dpi=300); plt.close(fig)
print("figures done")
