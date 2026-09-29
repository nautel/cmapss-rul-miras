"""Statistical tests for RUL results — instead of just comparing two means.

Two sources of uncertainty, which must be kept separate:
  (a) network initialization / batch order  -> run several seeds
  (b) finite test set (100-259 engines) -> bootstrap over ENGINES

Main functions:
  `ci`      : bootstrap confidence interval for one method's RMSE / Score
  `compare` : confidence interval of the DIFFERENCE between two methods, paired bootstrap
              over the same engine set — a CI that excludes 0 means a significant difference
  `paired`  : Wilcoxon signed-rank on per-engine errors + paired t-test over seeds
  Multiple-comparison correction: Holm-Bonferroni.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats as sps

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def score(p, t):
    r = p - t
    return float(np.sum(np.where(r > 0, np.exp(r / 10.0) - 1.0, np.exp(-r / 13.0) - 1.0)))


def load(dirs, pattern="pred_*.npz", merge_cn=True):
    """-> {(method, subset): {seed: (pred, true)}}  ; filename: pred_<sub>_<method>_s<seed>*.npz

    `merge_cn=True`: drop the `_cn` suffix. Operating-condition normalization is applied
    only to FD002/FD004 (FD001/FD003 have 1 condition, so both approaches coincide), so
    `bl_dcnn` and `bl_dcnn_cn` are the SAME method run on two different groups of subsets,
    not two methods.
    """
    out = defaultdict(dict)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, pattern))):
            b = os.path.basename(f)[5:-4]
            m = re.match(r"(FD00\d)_(.+)_s(\d+)(.*)$", b)
            if not m:
                continue
            sub, meth, sd, suf = m.groups()
            if merge_cn:
                suf = suf.replace("_cn", "")
            z = np.load(f)
            out[(meth + suf, sub)][int(sd)] = (z["pred"], z["true"])
    return out


def _boot_idx(n, n_boot, rng):
    return rng.integers(0, n, size=(n_boot, n))


def ci(data, metric=rmse, n_boot=2000, level=0.95, rng=None):
    """Engine-level bootstrap CI, averaged over seeds."""
    rng = rng or np.random.default_rng(0)
    seeds = sorted(data)
    n = len(data[seeds[0]][1])
    idx = _boot_idx(n, n_boot, rng)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = idx[b]
        vals[b] = np.mean([metric(data[s][0][i], data[s][1][i]) for s in seeds])
    point = np.mean([metric(data[s][0], data[s][1]) for s in seeds])
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return point, float(lo), float(hi)


def compare(a, b, metric=rmse, n_boot=2000, level=0.95, rng=None):
    """Bootstrap CI of the difference (A - B), paired over the same resampled engine set."""
    rng = rng or np.random.default_rng(0)
    sa, sb = sorted(a), sorted(b)
    n = len(a[sa[0]][1])
    assert len(b[sb[0]][1]) == n, "both methods must share the same test set"
    idx = _boot_idx(n, n_boot, rng)
    d = np.empty(n_boot)
    for k in range(n_boot):
        i = idx[k]
        d[k] = (np.mean([metric(a[s][0][i], a[s][1][i]) for s in sa])
                - np.mean([metric(b[s][0][i], b[s][1][i]) for s in sb]))
    point = (np.mean([metric(a[s][0], a[s][1]) for s in sa])
             - np.mean([metric(b[s][0], b[s][1]) for s in sb]))
    lo, hi = np.quantile(d, [(1 - level) / 2, 1 - (1 - level) / 2])
    # two-sided p from the bootstrap distribution of the difference
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(diff=float(point), lo=float(lo), hi=float(hi),
                p_boot=float(min(1.0, p)), sig=bool(lo > 0 or hi < 0))


def paired(a, b):
    """Wilcoxon on per-engine errors (seeds pooled) + paired t-test over seeds.

    WARNING: the Wilcoxon column here is for REFERENCE only, not for conclusions. It
    treats 100 engines x 10 seeds = 1000 independent observations, while the same 100
    engines are repeated 10 times. Result: p of 1e-04 to 1e-59 for EVERY pair, even
    pairs the engine-level bootstrap shows are not different. The correct sampling unit
    is the ENGINE -> use the paired bootstrap column.
    """
    sa, sb = sorted(a), sorted(b)
    ea = np.concatenate([(a[s][0] - a[s][1]) ** 2 for s in sa])
    eb = np.concatenate([(b[s][0] - b[s][1]) ** 2 for s in sb])
    n = min(len(ea), len(eb))
    try:
        w = sps.wilcoxon(ea[:n], eb[:n])
        p_w = float(w.pvalue)
    except ValueError:
        p_w = float("nan")
    common = sorted(set(sa) & set(sb))
    if len(common) >= 3:
        ra = [rmse(*a[s]) for s in common]
        rb = [rmse(*b[s]) for s in common]
        p_t = float(sps.ttest_rel(ra, rb).pvalue)
    else:
        p_t = float("nan")
    return dict(p_wilcoxon=p_w, p_ttest_seed=p_t, n_seeds=len(common))


def holm(pvals):
    """Holm-Bonferroni correction; returns adjusted p in the original input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--ref", required=True, help="reference method to compare against")
    p.add_argument("--methods", default=None, help="filter, comma-separated")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--md", default=None)
    a = p.parse_args()

    data = load(a.dirs)
    methods = sorted({k[0] for k in data})
    if a.methods:
        keep = set(a.methods.split(","))
        methods = [m for m in methods if m in keep]
    assert a.ref in methods, f"reference {a.ref} not found; available: {methods}"

    out = ["# Statistical tests", "",
           f"Reference: `{a.ref}`. 95% bootstrap CI, {a.n_boot} resamples **over engines** "
           "(the same engine set resampled for both sides — paired), averaged over seeds. "
           "`p` Holm-Bonferroni-adjusted within each subset.", ""]

    for sub in SUBSETS:
        rows = [m for m in methods if data.get((m, sub))]
        if len(rows) < 2:
            continue
        out += [f"## {sub}", "",
                "| method | RMSE [95% CI] | diff vs. reference [CI] | p (boot) | "
                "p (Holm) | p Wilcoxon | seed |",
                "|---|---|---|---|---|---|---|"]
        recs, ps = [], []
        for m in rows:
            if not data.get((m, sub)) or not data.get((a.ref, sub)):
                continue
            pt, lo, hi = ci(data[(m, sub)], n_boot=a.n_boot)
            if m == a.ref:
                recs.append((m, pt, lo, hi, None, None))
                continue
            c = compare(data[(m, sub)], data[(a.ref, sub)], n_boot=a.n_boot)
            w = paired(data[(m, sub)], data[(a.ref, sub)])
            recs.append((m, pt, lo, hi, c, w))
            ps.append(c["p_boot"])
        adj = holm(ps) if ps else []
        k = 0
        for m, pt, lo, hi, c, w in sorted(recs, key=lambda r: r[1]):
            if c is None:
                out.append(f"| **`{m}`** (ref.) | {pt:.2f} [{lo:.2f}, {hi:.2f}] | — | — | — | — | "
                           f"{len(data[(m, sub)])} |")
                continue
            i = [r[0] for r in recs if r[4] is not None].index(m)
            mark = "**" if adj[i] < 0.05 else ""
            out.append(f"| `{m}` | {pt:.2f} [{lo:.2f}, {hi:.2f}] | "
                       f"{mark}{c['diff']:+.2f}{mark} [{c['lo']:+.2f}, {c['hi']:+.2f}] | "
                       f"{c['p_boot']:.4f} | {adj[i]:.4f} | {w['p_wilcoxon']:.2e} | "
                       f"{len(data[(m, sub)])} |")
            k += 1
        out.append("")
        out.append("**Negative** difference = better than reference. "
                   "Bold = significant after Holm correction (p < 0.05).")
        out.append("")
        out.append("The `p Wilcoxon` column is for reference only — it treats "
                   "n_engine x n_seed as independent observations, so it comes out tiny for "
                   "every pair; use the paired bootstrap column for conclusions.")
        out.append("")

    md = "\n".join(out)
    print(md)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
