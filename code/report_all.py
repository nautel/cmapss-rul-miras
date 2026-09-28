"""Bang tong hop MOI phuong phap da chay tren C-MAPSS, kem CI bootstrap tren engine."""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np
from paper_values import SUBSETS, TABLE3, SOTA_RMSE, SOTA_SCORE
import stats as S

# ten hien thi + nguon
LABEL = {
    "yes_yes_yes": ("SBi-Transformer (paper 1, tai lap)", "paper 1"),
    "yes_yes_no": ("Transformer + sparse attn", "paper 1, ablation"),
    "yes_no_yes": ("Transformer + BiLSTM", "paper 1, ablation"),
    "yes_no_no": ("Transformer", "paper 1, ablation"),
    "no_no_yes": ("BiLSTM", "paper 1, ablation"),
    "bl_dcnn": ("DCNN", "Li 2018"),
    "bl_lstm": ("LSTM", "Zheng 2017"),
    "bl_bilstm": ("BiLSTM (baseline)", "van lieu"),
    "bl_gru": ("GRU", "van lieu"),
    "bl_tcn": ("TCN", "Bai 2018"),
    "bl_cnn_lstm": ("CNN-LSTM", "van lieu"),
    "bl_mlp": ("MLP (can duoi)", "—"),
    "titans": ("Titans", "paper 2"),
    "titans+bilstm": ("Titans + BiLSTM", "paper 2 x paper 1"),
    "gated_deltanet": ("Gated DeltaNet", "paper 2"),
    "gated_deltanet+bilstm": ("Gated DeltaNet + BiLSTM", "paper 2 x paper 1"),
    "mamba2": ("Mamba2", "paper 2"),
    "deltanet": ("DeltaNet", "paper 2"),
    "linear_attn": ("Linear Attention", "paper 2"),
    "retnet": ("RetNet", "paper 2"),
    "moneta": ("Moneta", "paper 2"),
    "yaad": ("Yaad", "paper 2"),
    "memora": ("Memora", "paper 2"),
    "elastic": ("Elastic-net retention", "paper 2"),
    "robust": ("Robust bias", "paper 2"),
    "mamba2+bilstm": ("Mamba2 + BiLSTM", "paper 2 x paper 1"),
    "titans_mlp": ("Titans, bo nho MLP sau", "paper 2"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", default=["../results/rul_sbi", "../results/rul_miras",
                                                 "../results/rul_bl", "../results/rul_main10"])
    p.add_argument("--ar", default="../results/rul_ar2")
    p.add_argument("--n-boot", type=int, default=1500)
    p.add_argument("--md", default=None)
    a = p.parse_args()

    dirs = [d for d in a.dirs if os.path.isdir(d)]
    data = S.load(dirs)
    # gop: moi phuong phap lay bien the co nhieu seed nhat tren tung subset
    best = defaultdict(dict)
    for (m, sub), v in data.items():
        base = m.split("_fe")[0].split("_L")[0]
        cur = best[base].get(sub)
        if cur is None or len(v) > len(cur):
            best[base][sub] = v

    # cau hinh cuoi cua autoresearch — CHI lay cau hinh co val tot nhat, khong tron 3 cai
    fin = []
    for f in sorted(glob.glob(os.path.join(a.ar, "final_results.jsonl*"))):
        fin += [json.loads(l) for l in open(f)]
    if fin:
        pick = min(fin, key=lambda r: r["val_stage"])["id"]
        for f in sorted(glob.glob(os.path.join(a.ar, f"pred_fin_{pick}_*.npz"))):
            b = os.path.basename(f)[:-4].replace(f"pred_fin_{pick}_", "")   # <sub>_s<seed>
            sub, sd = b.rsplit("_s", 1)
            z = np.load(f)
            best.setdefault("autoresearch", {}).setdefault(sub, {})[int(sd)] = (
                z["pred"], z["true"])

    rows = []
    for m, per in best.items():
        if len(per) < 4:
            continue
        cells = {}
        for sub in SUBSETS:
            pt, lo, hi = S.ci(per[sub], n_boot=a.n_boot)
            sc = np.mean([S.score(*per[sub][s]) for s in per[sub]])
            cells[sub] = (pt, lo, hi, sc, len(per[sub]))
        rows.append((st.mean(cells[s][0] for s in SUBSETS), m, cells))
    rows.sort()

    out = ["# Tong hop moi phuong phap tren C-MAPSS", "",
           "Cung pipeline: cung tien xu ly (z-score cho FD001/FD003, chuan hoa theo che do "
           "van hanh cho FD002/FD004), cung nhan cat 125, cung tach val theo engine. "
           "`[.., ..]` = CI 95% bootstrap tren engine.", "",
           "| # | phuong phap | nguon | " + " | ".join(SUBSETS) + " | TB | seed |",
           "|---|---|---|" + "---|" * 6]
    for i, (avg, m, c) in enumerate(rows, 1):
        lab, src = LABEL.get(m, (m, "—"))
        out.append(f"| {i} | {lab} | {src} | " +
                   " | ".join(f"{c[s][0]:.2f} [{c[s][1]:.1f}, {c[s][2]:.1f}]" for s in SUBSETS) +
                   f" | **{avg:.2f}** | {min(c[s][4] for s in SUBSETS)} |")
    out.append("")
    out += ["### Doi chieu voi so CONG BO (khong chay lai duoc)", "",
            "| nguon | " + " | ".join(SUBSETS) + " | TB |", "|---|" + "---|" * 5]
    ref = dict(SOTA_RMSE)
    ref["**SBi-Transformer (paper 1) — cong bo**"] = TABLE3["SBi-Transformer"]["rmse"]
    ref["ICL4RUL (paper 1 trich)"] = TABLE3["ICL4RUL [41]"]["rmse"]
    for k, v in sorted(ref.items(), key=lambda kv: st.mean(kv[1])):
        out.append(f"| {k} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {st.mean(v):.2f} |")
    out.append("")

    out += ["### Score (tong, thap hon = tot hon)", "",
            "| phuong phap | " + " | ".join(SUBSETS) + " | TB |", "|---|" + "---|" * 5]
    for avg, m, c in sorted(rows, key=lambda r: st.mean(r[2][s][3] for s in SUBSETS))[:12]:
        lab, _ = LABEL.get(m, (m, ""))
        out.append(f"| {lab} | " + " | ".join(f"{c[s][3]:.1f}" for s in SUBSETS) +
                   f" | **{st.mean(c[s][3] for s in SUBSETS):.1f}** |")
    out.append(f"| **paper 1 — cong bo** | " +
               " | ".join(f"{x:.2f}" for x in TABLE3['SBi-Transformer']['score']) +
               f" | {st.mean(TABLE3['SBi-Transformer']['score']):.1f} |")
    out.append(f"| STA-HPINN (2024) — cong bo | " +
               " | ".join(f"{x:.2f}" for x in SOTA_SCORE['STA-HPINN (2024)']) +
               f" | {st.mean(SOTA_SCORE['STA-HPINN (2024)']):.1f} |")
    out.append("")

    md = "\n".join(out)
    print(md)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
