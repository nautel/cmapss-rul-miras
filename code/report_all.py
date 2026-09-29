"""Summary table of ALL methods run on C-MAPSS, with engine-level bootstrap CIs."""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np
from paper_values import SUBSETS, TABLE3, SOTA_RMSE, SOTA_SCORE
import stats as S

# display name + source
LABEL = {
    "yes_yes_yes": ("SBi-Transformer (paper 1, reproduced)", "paper 1"),
    "yes_yes_no": ("Transformer + sparse attn", "paper 1, ablation"),
    "yes_no_yes": ("Transformer + BiLSTM", "paper 1, ablation"),
    "yes_no_no": ("Transformer", "paper 1, ablation"),
    "no_no_yes": ("BiLSTM", "paper 1, ablation"),
    "bl_dcnn": ("DCNN", "Li 2018"),
    "bl_lstm": ("LSTM", "Zheng 2017"),
    "bl_bilstm": ("BiLSTM (baseline)", "prior work"),
    "bl_gru": ("GRU", "prior work"),
    "bl_tcn": ("TCN", "Bai 2018"),
    "bl_cnn_lstm": ("CNN-LSTM", "prior work"),
    "bl_mlp": ("MLP (lower bound)", "—"),
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
    "titans_mlp": ("Titans, deep MLP memory", "paper 2"),
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
    # merge: for each method, take the variant with the most seeds on each subset
    best = defaultdict(dict)
    for (m, sub), v in data.items():
        base = m.split("_fe")[0].split("_L")[0]
        cur = best[base].get(sub)
        if cur is None or len(v) > len(cur):
            best[base][sub] = v

    # final autoresearch config — take ONLY the config with the best val, do not mix all 3
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

    out = ["# All methods on C-MAPSS — summary", "",
           "Same pipeline: same preprocessing (z-score for FD001/FD003, "
           "operating-condition normalization for FD002/FD004), same label cap 125, "
           "same engine-wise val split. `[.., ..]` = 95% engine-level bootstrap CI.", "",
           "| # | method | source | " + " | ".join(SUBSETS) + " | mean | seed |",
           "|---|---|---|" + "---|" * 6]
    for i, (avg, m, c) in enumerate(rows, 1):
        lab, src = LABEL.get(m, (m, "—"))
        out.append(f"| {i} | {lab} | {src} | " +
                   " | ".join(f"{c[s][0]:.2f} [{c[s][1]:.1f}, {c[s][2]:.1f}]" for s in SUBSETS) +
                   f" | **{avg:.2f}** | {min(c[s][4] for s in SUBSETS)} |")
    out.append("")
    out += ["### Comparison with PUBLISHED numbers (not re-runnable)", "",
            "| source | " + " | ".join(SUBSETS) + " | mean |", "|---|" + "---|" * 5]
    ref = dict(SOTA_RMSE)
    ref["**SBi-Transformer (paper 1) — published**"] = TABLE3["SBi-Transformer"]["rmse"]
    ref["ICL4RUL (cited in paper 1)"] = TABLE3["ICL4RUL [41]"]["rmse"]
    for k, v in sorted(ref.items(), key=lambda kv: st.mean(kv[1])):
        out.append(f"| {k} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {st.mean(v):.2f} |")
    out.append("")

    out += ["### Score (sum, lower = better)", "",
            "| method | " + " | ".join(SUBSETS) + " | mean |", "|---|" + "---|" * 5]
    for avg, m, c in sorted(rows, key=lambda r: st.mean(r[2][s][3] for s in SUBSETS))[:12]:
        lab, _ = LABEL.get(m, (m, ""))
        out.append(f"| {lab} | " + " | ".join(f"{c[s][3]:.1f}" for s in SUBSETS) +
                   f" | **{st.mean(c[s][3] for s in SUBSETS):.1f}** |")
    out.append(f"| **paper 1 — published** | " +
               " | ".join(f"{x:.2f}" for x in TABLE3['SBi-Transformer']['score']) +
               f" | {st.mean(TABLE3['SBi-Transformer']['score']):.1f} |")
    out.append(f"| STA-HPINN (2024) — published | " +
               " | ".join(f"{x:.2f}" for x in SOTA_SCORE['STA-HPINN (2024)']) +
               f" | {st.mean(SOTA_SCORE['STA-HPINN (2024)']):.1f} |")
    out.append("")

    md = "\n".join(out)
    print(md)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
