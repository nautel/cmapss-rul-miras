"""Report for the Miras-family rerun after the bug fix (09-2026).

Part A: effect of the chunk approximation (1 = exact recurrence).
Part B: Miras (fixed) vs. the SOTA reference in the SAME pipeline, engine-level bootstrap
        CIs, paired differences vs. STA-HPINN (no physics) and DCNN, Holm correction.
"""
import argparse
import statistics as st

import numpy as np

import stats as S
from paper_values import SOTA_RMSE

SUBSETS = S.SUBSETS
LABEL = {
    "sta_nophys": "STA-HPINN, no physics (SOTA re-implemented)",
    "mf_dcnn_c5": "DCNN (Li 2018)",
    "mf_titans+bilstm_c5": "Titans + BiLSTM",
    "mf_titans_R4_c5": "Titans, rank-4 memory",
    "mf_titans_B3trip_c5": "Titans, bottleneck 3 + triplet",
    "mf_titans_S_c5": "Titans + sensor branch",
}


def lab(m):
    if m in LABEL:
        return LABEL[m]
    if m.startswith("mf_") and m.endswith("_c5"):
        return m[3:-3]
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="../results/rul_mfix")
    p.add_argument("--n-boot", type=int, default=1500)
    p.add_argument("--md", default=None)
    a = p.parse_args()
    data = S.load([a.dir])
    methods = sorted({k[0] for k in data})
    out = ["# Miras after the bug fix — vs. SOTA in the same pipeline", "",
           "Shared pipeline: 14 classic sensors, window 40/60/60/60, min-max "
           "(FD001/FD003), operating-condition normalization (FD002/FD004), label cap 125, "
           "val = 20% of engines. `[..]` = 95% engine-level bootstrap CI.", ""]

    # ---- A: chunk ----
    out += ["## A. Effect of the chunk approximation (FD001, FD004 · 5 seeds)", "",
            "| variant | chunk 1 (exact) | chunk 5 | chunk 20 |", "|---|---|---|---|"]
    for v in ["linear_attn", "deltanet", "gated_deltanet", "titans"]:
        cells = []
        for c in (1, 5, 20):
            m = f"mf_{v}_c{c}"
            vals = []
            for s in ("FD001", "FD004"):
                d = data.get((m, s))
                if d:
                    seeds = sorted(d)[:5]
                    vals.append(np.mean([S.rmse(*d[k]) for k in seeds]))
            cells.append(f"{np.mean(vals):.2f}" if len(vals) == 2 else "—")
        out.append(f"| `{v}` | " + " | ".join(cells) + " |")
    out += ["", "Mean of FD001 and FD004. If `deltanet` and `linear_attn` separate only at "
            "small chunk sizes, the chunk approximation is erasing the differences between "
            "attentional biases.", ""]

    # ---- B: main table ----
    main_m = [m for m in methods if (m.endswith("_c5") or m == "sta_nophys")
              and all(data.get((m, s)) for s in SUBSETS)]
    rows = []
    for m in main_m:
        cells = {s: S.ci(data[(m, s)], n_boot=a.n_boot) for s in SUBSETS}
        sc = np.mean([np.mean([S.score(*data[(m, s)][k]) for k in data[(m, s)]])
                      for s in SUBSETS])
        n = min(len(data[(m, s)]) for s in SUBSETS)
        rows.append((st.mean(cells[s][0] for s in SUBSETS), m, cells, sc, n))
    rows.sort(key=lambda r: r[0])
    out += ["## B. Leaderboard (RMSE)", "",
            "| # | model | " + " | ".join(SUBSETS) + " | mean | mean Score | seed |",
            "|---|---|" + "---|" * 7]
    for i, (avg, m, c, sc, n) in enumerate(rows, 1):
        out.append(f"| {i} | {lab(m)} | " +
                   " | ".join(f"{c[s][0]:.2f} [{c[s][1]:.1f}, {c[s][2]:.1f}]" for s in SUBSETS) +
                   f" | **{avg:.2f}** | {sc:.0f} | {n} |")
    v = SOTA_RMSE["STA-HPINN (2024)"]
    out.append(f"| — | *STA-HPINN — published* | " + " | ".join(f"*{x:.2f}*" for x in v) +
               f" | *{np.mean(v):.2f}* | — | — |")
    out.append("")

    # ---- C: paired tests ----
    for ref in ("sta_nophys", "mf_dcnn_c5"):
        if not all(data.get((ref, s)) for s in SUBSETS):
            continue
        out += [f"## C. Paired difference vs. {lab(ref)}", "",
                "RMSE(model) − RMSE(reference); **negative = better than reference**. "
                "Bold = significant after Holm (p < 0.05) within each subset.", "",
                "| model | " + " | ".join(SUBSETS) + " |", "|---|" + "---|" * 4]
        per = {m: {} for _, m, *_ in rows if m != ref}
        for s in SUBSETS:
            ms = [m for m in per]
            cs = [S.compare(data[(m, s)], data[(ref, s)], n_boot=a.n_boot) for m in ms]
            adj = S.holm([c["p_boot"] for c in cs])
            for m, c, pa in zip(ms, cs, adj):
                t = f"{c['diff']:+.2f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"
                per[m][s] = f"**{t}**" if pa < 0.05 else t
        for _, m, *_ in rows:
            if m in per:
                out.append(f"| {lab(m)} | " + " | ".join(per[m][s] for s in SUBSETS) + " |")
        out.append("")

    md = "\n".join(out)
    print(md)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
