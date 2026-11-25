import numpy as np, sys
sys.path.insert(0,".")
from excluded import pr_curve
from sim import A
rows=[[1,2/np.pi*0.25/A**2]]
for line in open("data/aex_log.txt"):
    p=line.split(); rows.append([int(p[0]), float(p[3])])
nr=400; rs=(np.arange(nr)+0.5)/nr*2*A
k=128; p=pr_curve(k, rs, 60000, 7+k)
aex=np.sum(2*np.pi*rs*p)*(2*A/nr)
np.save(f"data/Pr_k{k}.npy", np.vstack([rs,p]))
rows.append([k, aex/A**2]); print(k, aex/A**2, 4*np.pi-aex/A**2)
np.save("data/aex.npy", np.array(rows))
