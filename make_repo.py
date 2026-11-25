"""Assemble the public repository folder ../star-percolation-repo from this working folder."""
import json, shutil, os, glob

SRC = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(os.path.dirname(SRC), "star-percolation-repo")
A = json.load(open(os.path.join(SRC, "author.json")))

CODE = ["sim.py", "fast.py", "analyze.py", "analyze2.py", "excluded.py", "aex_extra.py", "aex128.py", "aex_fast.py",
        "theory.py", "proof_check.py", "rigorous.py", "figures.py", "gen_tex_data.py", "make_repo.py",
        "jobs.sh", "jobs_fast.sh", "joblist.txt", "joblist2.txt", "joblist3.txt", "author.json"]
DIRS = ["data", "data2", "paper", "paper_ja"]
FIGS = sorted(glob.glob(os.path.join(SRC, "fig*.png")))

if os.path.exists(DST):
    for name in os.listdir(DST):
        if name == ".git":
            continue
        p = os.path.join(DST, name)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
os.makedirs(DST, exist_ok=True)
for f in CODE:
    shutil.copy2(os.path.join(SRC, f), DST)
for d in DIRS:
    shutil.copytree(os.path.join(SRC, d), os.path.join(DST, d), ignore=shutil.ignore_patterns("*.aux", "*.log", "~$*"))
for f in FIGS:
    shutil.copy2(f, DST)
os.makedirs(os.path.join(DST, "notes", "literature_check"), exist_ok=True)
shutil.copy2(os.path.join(SRC, "proof.html"), os.path.join(DST, "notes", "proof.html"))
for f in sorted(glob.glob(os.path.join(os.path.dirname(SRC), "step*.md"))):
    shutil.copy2(f, os.path.join(DST, "notes", "literature_check"))

MIT = f"""MIT License

Copyright (c) 2026 {A['name_en']}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
open(os.path.join(DST, "LICENSE"), "w").write(MIT)
open(os.path.join(DST, "LICENSE-data-and-text.md"), "w").write(
    f"The data (`data/`, `data2/`), figures, manuscripts (`paper/`, `paper_ja/`) and notes (`notes/`) are\n"
    f"© 2026 {A['name_en']} and licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0):\n"
    f"https://creativecommons.org/licenses/by/4.0/\n")
open(os.path.join(DST, ".gitignore"), "w").write("__pycache__/\n*.aux\n*.log\n*.out\n*.synctex.gz\n.DS_Store\n~$*\n")

README = f"""# Continuum percolation of k-armed stars in two dimensions

**{A['name_en']}** ({A['name_ja']}) — {A['affil_en']}

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

{A['contrib_en']} Generative AI (Claude, Anthropic) was used extensively in this work, including for the analysis,
code and drafting of the manuscript. All code and data are in this repository.

## Licence

Code: MIT (`LICENSE`). Data, figures, manuscripts and notes: CC BY 4.0 (`LICENSE-data-and-text.md`).

---

# 二次元における k 本腕の星形物体の連続体パーコレーション（日本語要約）

**{A['name_ja']}**（{A['affil_ja']}）

腕の本数 k を増やして円板に近づけたとき，平均排除面積の円板との差が k^(−2/3) という遅い速さで縮むことを，
係数まで含めて証明した（定理1）。また，臨界密度が円板の値に O(k^(−2/3)) で近づくことを，
円板の臨界値を使わずに証明した（定理2）。係数 0.873 は予想であり，k = 512 までの数値と 0.5% 以内で一致する。
解析・コード作成・原稿の起草を含め，生成AI（Claude）を広く利用した。
"""
open(os.path.join(DST, "README.md"), "w").write(README)
import re
for sh in ("jobs.sh", "jobs_fast.sh"):
    p = os.path.join(DST, sh)
    t = open(p).read()
    t = re.sub(r'PY="[^"]*"', 'PY="${PYTHON:-python}"', t)
    open(p, "w").write(t)
# no machine-specific paths may leak into the public repo
for root, _, files in os.walk(DST):
    if ".git" in root:
        continue
    for f in files:
        if f.endswith((".py", ".sh", ".md", ".tex", ".txt", ".json", ".html")):
            t = open(os.path.join(root, f), encoding="utf8", errors="ignore").read()
            assert ("/Us" + "ers/") not in t and ("/priv" + "ate/tmp") not in t, f"local path in {f}"
print("repo assembled at", DST)
