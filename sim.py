"""Continuum percolation of randomly placed, randomly oriented k-armed stars in 2D.

A star = k segments (arms) of length a=0.5 emanating from a common center at
angles theta0 + 2*pi*j/k, theta0 uniform. Two stars are connected iff any arm of
one intersects any arm of the other. Stars are added one by one into an LxL
periodic box (Mertens-Moore / Newman-Ziff style); union-find with displacement
vectors detects the first horizontal wrapping and the first wrapping in any
direction. Output: N at first wrap for each sample.
"""
import numpy as np
from numba import njit
import sys, time

A = 0.5  # arm length; cell size 2A = 1 (max center distance for contact)


@njit(cache=True)
def seg_intersect(px, py, qx, qy, rx, ry, sx, sy):
    # segments P->Q and R->S (proper or touching intersection)
    d1x = qx - px; d1y = qy - py
    d2x = sx - rx; d2y = sy - ry
    den = d1x * d2y - d1y * d2x
    if den == 0.0:
        return False
    ex = rx - px; ey = ry - py
    t = (ex * d2y - ey * d2x) / den
    u = (ex * d1y - ey * d1x) / den
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


@njit(cache=True)
def stars_touch(dx, dy, thA, thB, k):
    # star A at origin, star B at (dx,dy)
    if dx * dx + dy * dy >= 4 * A * A:
        return False
    for i in range(k):
        ai = thA + 2 * np.pi * i / k
        ax = A * np.cos(ai); ay = A * np.sin(ai)
        for j in range(k):
            bj = thB + 2 * np.pi * j / k
            bx = dx + A * np.cos(bj); by = dy + A * np.sin(bj)
            if seg_intersect(0.0, 0.0, ax, ay, dx, dy, bx, by):
                return True
    return False


@njit(cache=True)
def find(parent, ox, oy, i):
    # returns root; compresses path and accumulates offsets (pos_i - pos_root)
    r = i
    sx = 0.0; sy = 0.0
    while parent[r] != r:
        r = parent[r]
    # second pass: compress
    # compute offsets by walking up
    j = i
    # collect chain
    chain = []
    while parent[j] != j:
        chain.append(j)
        j = parent[j]
    # process from top to bottom
    for idx in range(len(chain) - 1, -1, -1):
        c = chain[idx]
        p = parent[c]
        if p != r:
            ox[c] += ox[p]; oy[c] += oy[p]
        parent[c] = r
    return r


@njit(cache=True)
def run_sample(L, k, nmax, seed):
    np.random.seed(seed)
    M = L  # cells per side (cell size 1)
    head = -np.ones(M * M, np.int64)
    nxt = -np.ones(nmax, np.int64)
    X = np.empty(nmax); Y = np.empty(nmax); TH = np.empty(nmax)
    parent = np.arange(nmax)
    ox = np.zeros(nmax); oy = np.zeros(nmax)
    size = np.ones(nmax, np.int64)
    Nh = -1; Na = -1
    period = 2 * np.pi / k
    for n in range(nmax):
        x = np.random.random() * L; y = np.random.random() * L
        th = np.random.random() * period
        X[n] = x; Y[n] = y; TH[n] = th
        cx = int(x); cy = int(y)
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                c = ((cx + ddx) % M) * M + ((cy + ddy) % M)
                m = head[c]
                while m != -1:
                    # vector from n to m (minimal image)
                    vx = X[m] - x; vy = Y[m] - y
                    vx -= L * np.round(vx / L); vy -= L * np.round(vy / L)
                    if stars_touch(vx, vy, th, TH[m], k):
                        rn = find(parent, ox, oy, n)
                        on_x = ox[n] if parent[n] != n else 0.0
                        on_y = oy[n] if parent[n] != n else 0.0
                        rm = find(parent, ox, oy, m)
                        om_x = ox[m] if parent[m] != m else 0.0
                        om_y = oy[m] if parent[m] != m else 0.0
                        # pos_m - pos_rm = om ; pos_n - pos_rn = on; pos_m - pos_n = v
                        if rn == rm:
                            wx = on_x + vx - om_x; wy = on_y + vy - om_y
                            if abs(wx) > 0.5 * L or abs(wy) > 0.5 * L:
                                if Na < 0:
                                    Na = n + 1
                                if abs(wx) > 0.5 * L and Nh < 0:
                                    Nh = n + 1
                        else:
                            # attach smaller under larger
                            # pos_rm - pos_rn = on + v - om
                            dxr = on_x + vx - om_x; dyr = on_y + vy - om_y
                            if size[rn] >= size[rm]:
                                parent[rm] = rn; ox[rm] = dxr; oy[rm] = dyr
                                size[rn] += size[rm]
                            else:
                                parent[rn] = rm; ox[rn] = -dxr; oy[rn] = -dyr
                                size[rm] += size[rn]
                    m = nxt[m]
        c0 = cx * M + cy
        nxt[n] = head[c0]; head[c0] = n
        if Nh > 0 and Na > 0:
            return Nh, Na
    return Nh, Na


if __name__ == "__main__":
    k = int(sys.argv[1]); L = int(sys.argv[2]); ns = int(sys.argv[3])
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 12345
    out = sys.argv[5] if len(sys.argv) > 5 else f"data/k{k}_L{L}.npy"
    # density guess: interpolate sticks..disks, generous upper bound
    nmax = int(8.0 * L * L / (4 * A * A) * 1.0) + 1000
    t = time.time()
    res = np.empty((ns, 2), np.int64)
    for s in range(ns):
        res[s] = run_sample(L, k, nmax, seed0 * 1000003 + s)
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out):
        res = np.vstack([np.load(out), res])
    np.save(out, res)
    print(k, L, ns, "mean Nh/L^2*a^2", res[:, 0].mean() / L**2 * A**2, "time", time.time() - t)
