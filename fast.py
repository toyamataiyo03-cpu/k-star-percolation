"""Fast exact contact test for k-armed stars (O(k * window) instead of O(k^2)), and simulation drivers.

For each arm of A that can reach B's disk, clip the arm to B's disk (radius A). The B arms that hit
the clipped piece are exactly those whose angle lies in the angular interval the piece subtends from
B's centre; B's arm angles form a lattice of spacing h, so this is an O(1) check.
"""
import numpy as np
from numba import njit
import sys, os, time

A = 0.5
TWO_PI = 2 * np.pi


@njit(cache=True)
def wrap(x):
    # to (-pi, pi]
    x = x % TWO_PI
    if x > np.pi:
        x -= TWO_PI
    return x


@njit(cache=True)
def touch_fast(dx, dy, thA, thB, k):
    r2 = dx * dx + dy * dy
    if r2 >= 4 * A * A:
        return False
    r = np.sqrt(r2)
    h = TWO_PI / k
    phi = np.arctan2(dy, dx)
    if r <= A:
        amax = np.pi
    else:
        amax = np.arcsin(A / r)
    # A-arm indices with |wrap(thA + i h - phi)| <= amax
    i0 = int(np.round((phi - thA) / h))
    nw = int(amax / h) + 2
    if 2 * nw + 1 >= k:
        lo_i = 0; hi_i = k - 1
    else:
        lo_i = i0 - nw; hi_i = i0 + nw
    # iterate outward from i0 for early exit
    span = hi_i - lo_i + 1
    for s in range(span):
        off = (s + 1) // 2 if s % 2 == 1 else -(s // 2)
        i = i0 + off if span < k else lo_i + s
        ang = thA + i * h
        al = wrap(ang - phi)
        if abs(al) > amax + 1e-12:
            continue
        ex = np.cos(ang); ey = np.sin(ang)
        ed = ex * dx + ey * dy
        D = ed * ed - r2 + A * A
        if D < 0:
            continue
        sq = np.sqrt(D)
        t1 = max(ed - sq, 0.0); t2 = min(ed + sq, A)
        if t2 <= t1:
            continue
        # endpoints relative to B centre
        px = t1 * ex - dx; py = t1 * ey - dy
        qx = t2 * ex - dx; qy = t2 * ey - dy
        a1 = np.arctan2(py, px); a2 = np.arctan2(qy, qx)
        w = wrap(a2 - a1)
        if w < 0:
            a1, w = a2, -w
        # any B arm angle thB + j h in [a1, a1 + w] (mod 2pi)?
        m = np.ceil((a1 - thB) / h)
        psi = thB + m * h
        if psi <= a1 + w + 1e-13:
            return True
    return False


@njit(cache=True)
def find(parent, ox, oy, i):
    r = i
    while parent[r] != r:
        r = parent[r]
    j = i
    chain = []
    while parent[j] != j:
        chain.append(j)
        j = parent[j]
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
    M = L
    head = -np.ones(M * M, np.int64)
    nxt = -np.ones(nmax, np.int64)
    X = np.empty(nmax); Y = np.empty(nmax); TH = np.empty(nmax)
    parent = np.arange(nmax)
    ox = np.zeros(nmax); oy = np.zeros(nmax)
    size = np.ones(nmax, np.int64)
    Nh = -1; Na = -1
    period = TWO_PI / k
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
                    vx = X[m] - x; vy = Y[m] - y
                    vx -= L * np.round(vx / L); vy -= L * np.round(vy / L)
                    if touch_fast(vx, vy, th, TH[m], k):
                        rn = find(parent, ox, oy, n)
                        on_x = ox[n] if parent[n] != n else 0.0
                        on_y = oy[n] if parent[n] != n else 0.0
                        rm = find(parent, ox, oy, m)
                        om_x = ox[m] if parent[m] != m else 0.0
                        om_y = oy[m] if parent[m] != m else 0.0
                        if rn == rm:
                            wx = on_x + vx - om_x; wy = on_y + vy - om_y
                            if abs(wx) > 0.5 * L or abs(wy) > 0.5 * L:
                                if Na < 0:
                                    Na = n + 1
                                if abs(wx) > 0.5 * L and Nh < 0:
                                    Nh = n + 1
                        else:
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


@njit(cache=True)
def pr_point(k, r, nsamp, seed):
    np.random.seed(seed)
    c = 0
    for s in range(nsamp):
        if touch_fast(r, 0.0, np.random.random() * TWO_PI, np.random.random() * TWO_PI, k):
            c += 1
    return c


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sim":
        k = int(sys.argv[2]); L = int(sys.argv[3]); ns = int(sys.argv[4]); seed0 = int(sys.argv[5]); out = sys.argv[6]
        nmax = int(8.0 * L * L / (4 * A * A)) + 1000
        t = time.time()
        res = np.empty((ns, 2), np.int64)
        for s in range(ns):
            res[s] = run_sample(L, k, nmax, seed0 * 1000003 + s)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.save(out, res)
        print(k, L, ns, "time", round(time.time() - t, 1), flush=True)
    elif mode == "aex":
        # P(r) on midpoint grid restricted to the near-tangent band, P=1 checked elsewhere
        k = int(sys.argv[2]); nr = int(sys.argv[3]); ns = int(sys.argv[4]); seed = int(sys.argv[5]); out = sys.argv[6]
        rs = (np.arange(nr) + 0.5) / nr * 2 * A
        p = np.array([pr_point(k, r, ns, seed * 100003 + i) / ns for i, r in enumerate(rs)])
        np.save(out, np.vstack([rs, p]))
        print(k, "done", flush=True)
