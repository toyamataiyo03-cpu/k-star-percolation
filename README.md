# Continuum percolation of k-armed stars in two dimensions

**Taiyo Toyama** (遠山 大陽) — Hokkaido Asahikawa Higashi High School, Asahikawa, Hokkaido, Japan

Two-dimensional continuum percolation of randomly placed, randomly oriented *k-armed stars*
(k zero-width segments of length *a* from a common centre at equal angles), which interpolate between
sticks (k = 2) and disks (k → ∞).

## Main results

| | Statement | Status |
|---|---|---|
| Thresholds | n_c(k) for 2 ≤ k ≤ 512 (relative precision 1e-4 for k ≥ 64) | numerical |
| Theorem 1 | 4π − ⟨A_ex⟩/a² = C k^(−2/3) − C′ k^(−4/3) + O(k^(−2)), C = (18/5)2^(1/3)π^(5/3), C′ = (36/7)2^(−1/3)π^(7/3); error ≤ 2396/k² for k ≥ 106 | proven (partly computer-assisted) |
| Theorem 2 | n_c^disk ≤ n_c(k) ≤ n_c^disk/(1−ε₁)², ε₁ ≤ 1.1(π/ck)^(2/3), c = 4√6/9 | proven |
| Conjecture | (n_c − n_c^disk)a² ≈ 0.873 k^(−2/3) | heuristic; agrees with data to 0.5% at k = 512 |
| B_c | critical mean number of contacts varies from 3.59 (sticks) to 4.51 (disks) | numerical |

## Contents

| Path | What it is |
|---|---|
| `paper/main.tex`, `paper/main.pdf` | English manuscript (REVTeX 4.2) and compiled PDF |
| `paper_ja/main.tex`, `paper_ja/main.pdf` | Japanese manuscript (XeLaTeX or LuaLaTeX) and compiled PDF |
| `notes/proof.html` | Detailed proof note (Japanese, open in a browser) |
| `notes/literature_check/` | Log of the prior-art search |
| `sim.py`, `fast.py` | Monte Carlo simulation (Newman–Ziff union–find with wrapping detection; `fast.py` has the O(k) contact test) |
| `analyze.py`, `analyze2.py` | Threshold estimation (wrapping probabilities, bootstrap errors) |
| `excluded.py`, `aex_fast.py`, `aex_extra.py`, `aex128.py` | Excluded areas and contact probabilities |
| `theory.py`, `proof_check.py` | Semi-analytic formula and numerical checks of each proof step |
| `rigorous.py` | Computer-assisted bounds in ball arithmetic (python-flint / Arb) |
| `figures.py`, `gen_tex_data.py` | Figures and LaTeX tables |
| `data/`, `data2/` | Raw data: first-wrapping counts per sample (`k*_L*_c*.npy`), contact probabilities, results (JSON) |

## Reproduction

```bash
pip install numpy scipy matplotlib numba sympy python-flint
python sim.py <k> <L> <samples> <seed> data/k<k>_L<L>_c<chunk>.npy   # or: zsh jobs.sh (see joblist*.txt)
python fast.py sim <k> <L> <samples> <seed> data2/k<k>_L<L>_c<chunk>.npy
python excluded.py && python aex_fast.py 64 128 256 512
python theory.py && python proof_check.py && python rigorous.py
python analyze2.py && python figures.py && python gen_tex_data.py
```

The full runs took about 2 hours on a 10-core laptop.

## Use of generative AI

The research design, theoretical analysis, simulation code, data analysis, figures and manuscript drafts
were produced with substantial use of a generative AI system (Claude, Anthropic). T. Toyama initiated and directed the project (choice of research area, priority on novelty, the decision to seek a mathematical proof and to test the conjectured prefactor numerically), checked the results, and takes full responsibility for the content.

## Licence

Code: MIT (`LICENSE`). Data, figures, manuscripts and notes: CC BY 4.0 (`LICENSE-data-and-text.md`).

---

# 二次元における k 本腕の星形物体の連続体パーコレーション（日本語要約）

**遠山 大陽**（北海道旭川東高等学校 3年）

腕の本数 k を増やして円板に近づけたとき，平均排除面積の円板との差が k^(−2/3) という遅い速さで縮むことを，
係数まで含めて証明した（定理1）。また，臨界密度が円板の値に O(k^(−2/3)) で近づくことを，
円板の臨界値を使わずに証明した（定理2）。係数 0.873 は予想であり，k = 512 までの数値と 0.5% 以内で一致する。
研究の設計・解析・コード・原稿は生成AI（Claude）を大幅に用いて作成した。
