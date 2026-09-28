"""Bao cao lan chay lai ho Miras sau khi sua loi (09-2026).

Phan A: anh huong cua xap xi chunk (1 = truy hoi chinh xac).
Phan B: Miras (da sua) so voi moc SOTA trong CUNG pipeline, CI bootstrap tren engine,
        hieu ghep cap so voi STA-HPINN (bo physics) va DCNN, hieu chinh Holm.
"""
import argparse
import statistics as st

import numpy as np

import stats as S
from paper_values import SOTA_RMSE

SUBSETS = S.SUBSETS
LABEL = {
    "sta_nophys": "STA-HPINN, bo physics (SOTA tai tao)",
    "mf_dcnn_c5": "DCNN (Li 2018)",
    "mf_titans+bilstm_c5": "Titans + BiLSTM",
    "mf_titans_R4_c5": "Titans, bo nho hang 4",
    "mf_titans_B3trip_c5": "Titans, nut that 3 + triplet",
    "mf_titans_S_c5": "Titans + nhanh cam bien",
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
    out = ["# Miras sau khi sua loi — so voi SOTA cung pipeline", "",
           "Pipeline chung: 14 sensor kinh dien, cua so 40/60/60/60, min-max (FD001/FD003), "
           "chuan hoa theo che do van hanh (FD002/FD004), nhan cat 125, val 20% engine. "
           "`[..]` = CI 95% bootstrap tren engine.", ""]

    # ---- A: chunk ----
    out += ["## A. Anh huong cua xap xi chunk (FD001, FD004 · 5 seed)", "",
            "| bien the | chunk 1 (chinh xac) | chunk 5 | chunk 20 |", "|---|---|---|---|"]
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
    out += ["", "Trung binh FD001 va FD004. Neu `deltanet` va `linear_attn` chi tach nhau khi "
            "chunk nho thi xap xi chunk dang xoa khac biet giua cac attentional bias.", ""]

    # ---- B: bang chinh ----
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
    out += ["## B. Bang xep hang (RMSE)", "",
            "| # | mo hinh | " + " | ".join(SUBSETS) + " | TB | Score TB | seed |",
            "|---|---|" + "---|" * 7]
    for i, (avg, m, c, sc, n) in enumerate(rows, 1):
        out.append(f"| {i} | {lab(m)} | " +
                   " | ".join(f"{c[s][0]:.2f} [{c[s][1]:.1f}, {c[s][2]:.1f}]" for s in SUBSETS) +
                   f" | **{avg:.2f}** | {sc:.0f} | {n} |")
    v = SOTA_RMSE["STA-HPINN (2024)"]
    out.append(f"| — | *STA-HPINN — so cong bo* | " + " | ".join(f"*{x:.2f}*" for x in v) +
               f" | *{np.mean(v):.2f}* | — | — |")
    out.append("")

    # ---- C: kiem dinh ghep cap ----
    for ref in ("sta_nophys", "mf_dcnn_c5"):
        if not all(data.get((ref, s)) for s in SUBSETS):
            continue
        out += [f"## C. Hieu ghep cap so voi {lab(ref)}", "",
                "RMSE(mo hinh) − RMSE(moc); **am = tot hon moc**. In dam = co y nghia sau "
                "Holm (p < 0,05) trong tung bo.", "",
                "| mo hinh | " + " | ".join(SUBSETS) + " |", "|---|" + "---|" * 4]
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
