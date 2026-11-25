"""High-precision excluded-area deficit D(k) = 4pi a^2 - <A_ex> (in units a^2), via the near-tangent band only.
For eps beyond EPS_MAX the contact is certain (covering argument; checked numerically on the grid edge)."""
import numpy as np, json, sys
from fast import pr_point, A

ks = [int(x) for x in sys.argv[1:]] or [64, 128, 256, 512]
c = 4 * np.sqrt(6) / 9
out = {}
for k in ks:
    es = (np.pi / (c * k)) ** (2 / 3)
    emax = min(0.45, 2.2 * es)
    nr, ns = 400, 200000
    eps = (np.arange(nr) + 0.5) / nr * emax
    P = np.array([pr_point(k, 2 * A * (1 - e), ns, 1000 * k + i) / ns for i, e in enumerate(eps)])
    w = 8 * np.pi * (1 - eps) * (emax / nr)           # D/a^2 = 8 pi int (1-eps)(1-P) d eps
    D = float(np.sum(w * (1 - P)))
    err = float(np.sqrt(np.sum(w**2 * P * (1 - P) / ns)))
    tail = float(P[-20:].min())
    out[k] = dict(D=D, err=err, emax=emax, P_tail_min=tail)
    print(f"k={k}  D={D:.5f} +- {err:.5f}   (P at band edge min = {tail})", flush=True)
json.dump(out, open("data2/aex_fast.json", "w"), indent=1)
