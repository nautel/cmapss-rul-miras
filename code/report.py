"""Gom ket qua -> bang markdown doi chieu voi Table 3/4/5/6 cua paper."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from paper_values import TABLE3, ABL_RMSE, ABL_SCORE, TABLE6, ABL_LABEL, SUBSETS


def load(outdir):
    agg = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(outdir, "res_*.json"))):
        r = json.load(open(f))
        agg[(r["subset"], r["ablation"], bool(r.get("cond_norm")),
             bool(r.get("full_epochs")), int(r["cfg"]["seq_len"]))].append(r)
    return agg


def mstd(vals):
    a = np.asarray(vals, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def cell(vals, paper=None, fmt="{:.2f}"):
    m, s = mstd(vals)
    t = fmt.format(m) + (f" ±{fmt.format(s)}" if len(vals) > 1 else "")
    if paper is not None:
        t += f" ({(m - paper) / paper * 100:+.1f}%)"
    return t


def table(agg, key, paper_map, cond_norm=False, title="", fmt="{:.2f}", fe=False, L=45):
    lines = [f"### {title}", "",
             "| Cau hinh (Transformer / Sparse / BiLSTM) | " +
             " | ".join(SUBSETS) + " | Trung binh |",
             "|---|" + "---|" * (len(SUBSETS) + 1)]
    for ab, lab in ABL_LABEL.items():
        row, means = [], []
        ok = True
        for s in SUBSETS:
            rs = agg.get((s, ab, cond_norm, fe, L), [])
            if not rs:
                row.append("—"); ok = False; continue
            v = [r[key] for r in rs]
            row.append(cell(v, paper_map[ab][SUBSETS.index(s)], fmt))
            means.append(np.mean(v))
        avg = fmt.format(np.mean(means)) if ok and means else "—"
        pavg = np.mean(paper_map[ab])
        lines.append(f"| {lab} | " + " | ".join(row) + f" | {avg} (paper {pavg:.2f}) |")
    lines.append("")
    lines.append("Trong ngoac: lech tuong doi so voi tri paper. `±` = do lech chuan giua cac seed.")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="./out")
    p.add_argument("--md", default=None)
    a = p.parse_args()
    agg = load(a.outdir)
    if not agg:
        raise SystemExit(f"Khong co res_*.json trong {a.outdir}")

    out = ["# Tai lap SBi-Transformer — Ren et al., Results in Engineering 29 (2026) 109187", ""]
    n_seeds = max(len(v) for v in agg.values())
    out += [f"Chay tren cassio (Tesla V100). Moi o = trung binh {n_seeds} seed.", ""]

    # --- Table 3 ---
    out += ["## Table 3 — do chinh xac so voi cac phuong phap khac", "",
            "| Algorithm | " + " | ".join(f"RMSE {s}" for s in SUBSETS) + " | " +
            " | ".join(f"Score {s}" for s in SUBSETS) + " |",
            "|---|" + "---|" * 8]
    for name, v in TABLE3.items():
        out.append(f"| {name} (paper) | " +
                   " | ".join(f"{x:.2f}" for x in v["rmse"]) + " | " +
                   " | ".join(f"{x:.2f}" for x in v["score"]) + " |")
    for ab, lab in [("yes_yes_yes", "**SBi-Transformer (tai lap)**"),
                    ("yes_no_no", "Transformer only (tai lap)"),
                    ("no_no_yes", "BiLSTM only (tai lap)")]:
        r_, s_ = [], []
        for s in SUBSETS:
            rs = agg.get((s, ab, False, False, 45), [])
            r_.append(cell([x["rmse"] for x in rs]) if rs else "—")
            s_.append(cell([x["score"] for x in rs], fmt="{:.1f}") if rs else "—")
        out.append(f"| {lab} | " + " | ".join(r_) + " | " + " | ".join(s_) + " |")
    out.append("")

    # --- Table 4 / 5 ---
    out.append(table(agg, "rmse", ABL_RMSE, False,
                     "Table 4 — ablation, RMSE (dung som nhu Table 2: patience=10)"))
    out.append(table(agg, "score", ABL_SCORE, False,
                     "Table 5 — ablation, Score (dung som)", "{:.1f}"))
    if any((s, ab, False, True, 45) in agg for s in SUBSETS for ab in ABL_LABEL):
        out += ["> Early stopping ban rat som (FD001/FD003: 22-24 epoch) va cat mo hinh LON "
                "nang hon mo hinh nho, nen bang tren KHONG so sanh ablation cong bang. "
                "Hai bang duoi chay du 200/600 epoch cua Table 2.", ""]
        out.append(table(agg, "rmse", ABL_RMSE, False,
                         "Table 4 — ablation, RMSE (du epoch Table 2)", fe=True))
        out.append(table(agg, "score", ABL_SCORE, False,
                         "Table 5 — ablation, Score (du epoch Table 2)", "{:.1f}", fe=True))

    # --- Table 6 ---
    out += ["### Table 6 — thoi gian moi epoch (giay)", "",
            "| Phuong phap | " + " | ".join(SUBSETS) + " |", "|---|" + "---|" * 4]
    for k, v in TABLE6.items():
        out.append(f"| {k} (paper) | " + " | ".join(f"{x:.2f}" for x in v) + " |")
    row = []
    for s in SUBSETS:
        rs = agg.get((s, "yes_yes_yes", False, False, 45), [])
        row.append(cell([x["sec_per_epoch"] for x in rs]) if rs else "—")
    out.append("| **SBi-Transformer (tai lap, V100)** | " + " | ".join(row) + " |")
    out += ["", "Paper do tren CPU i7-13800H; cot tai lap do tren 1 GPU V100 (may dung chung, "
            "tai bien dong) nen chi so sanh duoc theo ty le giua cac subset.", ""]

    # --- do bat dinh + so epoch ---
    out += ["### Do bat dinh (Bootstrap + t-distribution, Eq. 13-14)", "",
            "| Subset | RMSE (mean MC-dropout) | Score | CI Eq.14: do phu / be rong | CI du bao: do phu / be rong | Epoch chay | Tham so |",
            "|---|---|---|---|---|---|---|"]
    for s in SUBSETS:
        rs = agg.get((s, "yes_yes_yes", False, False, 45), [])
        if not rs:
            continue
        g = lambda k, f="{:.2f}": cell([x[k] for x in rs if k in x], fmt=f) if any(k in x for x in rs) else "—"
        out.append(f"| {s} | {g('rmse_mc')} | {g('score_mc','{:.1f}')} | "
                   f"{g('ci_coverage','{:.3f}')} / {g('ci_width')} | "
                   f"{g('ci_pred_coverage','{:.3f}')} / {g('ci_pred_width')} | "
                   f"{g('epochs_run','{:.0f}')} | {rs[0]['n_params']} |")
    out += ["", "Eq. (14) chia sigma cho `sqrt(n_bootstrap)` sau khi sigma da la do lech chuan cua "
            "mot ensemble trung binh -> khoang hep den muc do phu ~0. Cot 'CI du bao' dung do tan "
            "MC-dropout (`f*t*sigma_mc`), la thu tuong ung voi dai mau trong Fig. 5 cua paper.", ""]

    # --- cac bien the: dung som / du epoch, Z-score toan cuc / theo che do van hanh ---
    VARIANTS = [((False, False, 45), "Nhu paper (seq_len=45, patience=10, Z-score toan cuc)"),
                ((False, True, 45),  "+ du epoch Table 2 (bo dung som)"),
                ((True, False, 45),  "+ chuan hoa theo che do van hanh"),
                ((True, True, 45),   "+ du epoch + chuan hoa theo che do"),
                ((False, True, 30),  "seq_len=30 (theo Table 1), du epoch"),
                ((True, True, 30),   "seq_len=30 + chuan hoa theo che do, du epoch")]
    have = [(k, lab) for k, lab in VARIANTS
            if any((s, "yes_yes_yes") + k in agg for s in SUBSETS)]
    if len(have) > 1:
        out += ["### Bien the — tach rieng anh huong cua dung som va cua chuan hoa", "",
                "| Bien the | " + " | ".join(f"RMSE {s}" for s in SUBSETS) +
                " | " + " | ".join(f"Score {s}" for s in SUBSETS) + " |",
                "|---|" + "---|" * 8]
        for (cn, fe, L), lab in have:
            r_, s_ = [], []
            for s in SUBSETS:
                rs = agg.get((s, "yes_yes_yes", cn, fe, L), [])
                r_.append(cell([x["rmse"] for x in rs]) if rs else "—")
                s_.append(cell([x["score"] for x in rs], fmt="{:.1f}") if rs else "—")
            out.append(f"| {lab} | " + " | ".join(r_) + " | " + " | ".join(s_) + " |")
        out.append("| **paper** | " + " | ".join(f"{x:.2f}" for x in ABL_RMSE["yes_yes_yes"]) +
                   " | " + " | ".join(f"{x:.2f}" for x in ABL_SCORE["yes_yes_yes"]) + " |")
        out += ["", "FD001/FD003 chi co 1 che do van hanh nen chuan hoa theo che do trung voi "
                "Z-score toan cuc — khong chay.", ""]
        ep = []
        for s in SUBSETS:
            rs = agg.get((s, "yes_yes_yes", False, False, 45), [])
            if rs:
                ep.append(f"{s}: {np.mean([x['epochs_run'] for x in rs]):.0f}")
        if ep:
            out += [f"So epoch thuc su chay khi bat dung som: {', '.join(ep)} "
                    f"(Table 2 ghi 200 cho FD001/FD003 va 600 cho FD002/FD004).", ""]

    md = "\n".join(out)
    print(md)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
