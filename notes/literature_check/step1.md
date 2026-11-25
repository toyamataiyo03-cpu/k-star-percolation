# Step 1 — Decompose the Novelty
Timestamp: 2026-11-24T22:05:08Z

- **Problem framing**: 2D continuum percolation of overlapping, randomly oriented k-armed stars (k zero-width segments of length a from a common centre at equal angles); quantities: critical number density n_c(k), mean excluded area <A_ex>(k), critical mean contacts B_c(k); regime k=2..64 and k→∞ (crossover sticks→disks).
- **Core mechanism**: (a) Newman–Ziff/Mertens–Moore union-find Monte Carlo with wrapping estimators; (b) exact law-of-sines contact condition and representation of two-star contact probability as area of a contact set modulo the square lattice (2π/k)Z² of arm-angle pairs, with two-sided bounds; (c) asymptotic expansion giving 4πa²−<A_ex> = C k^{-2/3} − C′ k^{-4/3} + O(k^{-2}); (d) coupling sandwich n_c^disk ≤ n_c(k) ≤ n_c^disk/(1−ε1)².
- **Key insight**: near-tangent contact is decided by a single nearly-facing arm pair whose admissible angular window grows only as ε^{3/2}, so the crossover scale is ε* ∝ k^{-2/3} (anomalously slow vs polygon k^{-2}); beyond a distance 2a(1−ε1) contact is deterministic, enabling a rigorous threshold bound without knowing η_c.
- **Application domain**: statistical physics of continuum percolation; branched/multipod nanoparticle and nanowire networks (onset of conductivity).
