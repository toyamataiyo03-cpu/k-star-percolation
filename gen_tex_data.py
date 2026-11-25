"""Generate LaTeX fragments (table, author block) from results and author.json, for paper/ and paper_ja/."""
import json, os
R = json.load(open("data/results.json")); H = json.load(open("data2/conjecture_test.json"))
import numpy as np
rows = []
for k in sorted(int(x) for x in R):
    if str(k) in H:
        continue
    v = R[str(k)]
    rows.append((k, v["nc"], v["err"], v["aex"], None, v["Bc"], v["eBc"]))
for k in sorted(int(x) for x in H):
    v = H[str(k)]
    rows.append((k, v["nc"], v["enc"], 4 * np.pi - v["D"], v["eD"], v["Bc"], v["eBc"]))

def pm(x, e, nd):
    q = round(e * 10**nd)
    return f"{x:.{nd}f}({max(q,1)})"

lines = []
for k, nc, enc, aex, eaex, bc, ebc in rows:
    nd = 6 if enc < 1e-4 else 4
    a = f"{aex:.5f}({max(1,round(eaex*1e5))})" if eaex else f"{aex:.4f}"
    lines.append(f"{k} & {pm(nc, enc, nd)} & {a} & {pm(bc, ebc, 4 if ebc < 1e-3 else 3)} \\\\")
tab = "\n".join(lines)
for d in ("paper", "paper_ja"):
    os.makedirs(d, exist_ok=True)
    open(f"{d}/table_data.tex", "w").write(tab + "\n")

A = json.load(open("author.json"))
def esc(s): return s.replace("&", "\\&")
en = f"\\author{{{esc(A['name_en'])}}}\n\\affiliation{{{esc(A['affil_en'])}}}\n"
if A.get("email"):
    en += f"\\email{{{A['email']}}}\n"
open("paper/authors.tex", "w").write(en)
ja = f"\\newcommand{{\\AuthorJa}}{{{esc(A['name_ja'])}}}\n\\newcommand{{\\AuthorEn}}{{{esc(A['name_en'])}}}\n\\newcommand{{\\AffilJa}}{{{esc(A['affil_ja'])}}}\n\\newcommand{{\\EmailLine}}{{{A['email'] or ''}}}\n"
open("paper_ja/authors.tex", "w").write(ja)
for d in ("paper", "paper_ja"):
    open(f"{d}/meta.tex", "w").write(
        f"\\newcommand{{\\RepoURL}}{{{A['repo_url']}}}\n"
        f"\\newcommand{{\\ContribEn}}{{{esc(A['contrib_en'])}}}\n\\newcommand{{\\ContribJa}}{{{esc(A['contrib_ja'])}}}\n")
print("ok", len(rows), "rows")
