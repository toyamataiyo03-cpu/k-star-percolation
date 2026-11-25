import numpy as np, sys
sys.path.insert(0,".")
from excluded import pr_curve
from sim import A
rows=[list(r) for r in np.load("data/aex.npy")]
nr=400; rs=(np.arange(nr)+0.5)/nr*2*A
for k in (7,9):
    p=pr_curve(k, rs, 200000, 7+k)
    aex=np.sum(2*np.pi*rs*p)*(2*A/nr); np.save(f"data/Pr_k{k}.npy", np.vstack([rs,p]))
    rows.append([k, aex/A**2]); print(k, aex/A**2)
rows.sort(); np.save("data/aex.npy", np.array(rows))
# statistical error estimate of A_ex for k=5 by independent seed
p=pr_curve(5, rs, 200000, 999); print("k=5 reseed", np.sum(2*np.pi*rs*p)*(2*A/nr)/A**2)
