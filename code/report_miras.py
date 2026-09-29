"""Miras family (paper 2) vs. SBi-Transformer and the baselines (paper 1) on C-MAPSS.

Merges results from both run dirs: `--sbi` (phases 1-4) and `--miras` (phases 5-6).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from paper_values import SUBSETS, TABLE3

# design groups per Miras: (attentional bias, retention gate)
MIRAS_INFO = {
    "linear_attn":    ("dot product (Hebbian)", "no forgetting (alpha=1)"),
    "retnet":         ("dot product", "learned constant alpha"),
    "mamba2":         ("dot product", "data-dependent alpha_t"),
    "deltanet":       ("l2 (delta rule)", "no forgetting (alpha=1)"),
    "gated_deltanet": ("l2 (delta rule)", "data-dependent alpha_t"),
    "titans":         ("l2 + momentum", "data-dependent alpha_t"),
    "moneta":         ("l_p, p=3", "l_q normalization, q=4"),
    "yaad":           ("Huber (l2 / l1)", "data-dependent alpha_t"),
    "memora":         ("l2", "KL / softmax"),
    "elastic":        ("l2", "elastic net, soft-threshold"),
    "robust":         ("l2 + worst-case shift", "data-dependent alpha_t"),
    "titans_mlp":     ("l2 + momentum, deep MLP memory", "data-dependent alpha_t"),
}
SBI_LABEL = {
    "yes_yes_yes": "SBi-Transformer (paper 1)",
    "yes_yes_no":  "Transformer + sparse attn",
    "yes_no_no":   "Transformer",
    "no_no_yes":   "BiLSTM",
}


def load(dirs):
    agg = defaultdict(list)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "res_*.json"))):
            r = json.load(open(f))
            name = r.get("arch", "sbi")
            if name == "sbi":
                name = r["ablation"]
            agg[(r["subset"], name, bool(r.get("cond_norm")),
                 bool(r.get("full_epochs")), int(r["cfg"]["seq_len"]))].append(r)
    return agg


def best_variant(agg, subset, name):
    """Pick a model's best configuration (by mean RMSE) on one subset."""
    cands = [(k, v) for k, v in agg.items() if k[0] == subset and k[1] == name]
    if not cands:
        return None
    return min(cands, key=lambda kv: np.mean([r["rmse"] for r in kv[1]]))


def cell(vals, fmt="{:.2f}"):
    a = np.asarray(vals, float)
    t = fmt.format(a.mean())
    return t + (f" ±{fmt.format(a.std(ddof=1))}" if len(a) > 1 else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sbi", default="./out_sbi")
    p.add_argument("--miras", default="./out_miras")
    p.add_argument("--md", default=None)
    a = p.parse_args()
    agg = load([a.sbi, a.miras])
    names = sorted({k[1] for k in agg})
    miras_names = ([n for n in MIRAS_INFO if n in names] +
                   [n + "+bilstm" for n in MIRAS_INFO if n + "+bilstm" in names])
    sbi_names = [n for n in SBI_LABEL if n in names]

    rows = []
    for n in sbi_names + miras_names:
        rmse, score, cfgs, secs = [], [], [], []
        for s in SUBSETS:
            b = best_variant(agg, s, n)
            if b is None:
                rmse.append(None); score.append(None); continue
            k, v = b
            rmse.append([r["rmse"] for r in v])
            score.append([r["score"] for r in v])
            secs.append(np.mean([r["sec_per_epoch"] for r in v]))
            cfgs.append(("cond-norm" if k[2] else "z-score") + f"/L{k[4]}")
        if any(x is None for x in rmse):
            continue
        rows.append(dict(name=n, rmse=rmse, score=score, cfg=cfgs,
                         sec=np.mean(secs) if secs else float("nan"),
                         avg=np.mean([np.mean(x) for x in rmse]),
                         avg_score=np.mean([np.mean(x) for x in score]),
                         n_params=v[0]["n_params"]))
    rows.sort(key=lambda r: r["avg"])

    out = ["# Trying the Miras family on C-MAPSS — and which direction suits RUL", "",
           "Paper 2: Behrouz, Razaviyayn, Zhong, Mirrokni, *It's All Connected* "
           "(arXiv:2504.13173, Google Research 2025).", "",
           "Every architecture uses the SAME backbone (linear projection + learned "
           "position embedding -> N blocks -> FC on the last step) and the SAME Table 2 "
           "hyperparameters of paper 1; only the sequence-mixing block differs. "
           "Each cell = mean over 3 seeds, using the best preprocessing config "
           "(z-score for FD001/FD003, operating-condition normalization for FD002/FD004).",
           "",
           "## Leaderboard (RMSE, lower = better)", "",
           "| # | Model | attentional bias | retention gate | " +
           " | ".join(SUBSETS) + " | mean | params |",
           "|---|---|---|---|" + "---|" * 6]
    for i, r in enumerate(rows, 1):
        base = r["name"].replace("+bilstm", "")
        bias, ret = MIRAS_INFO.get(base, ("—", "—"))
        lab = SBI_LABEL.get(r["name"], r["name"])
        if r["name"].endswith("+bilstm"):
            lab += " + BiLSTM head"
        out.append(f"| {i} | {lab} | {bias} | {ret} | " +
                   " | ".join(cell(x) for x in r["rmse"]) +
                   f" | **{r['avg']:.2f}** | {r['n_params']} |")
    out.append(f"| — | SBi-Transformer *(paper 1 published)* | — | — | " +
               " | ".join(f"{x:.2f}" for x in TABLE3["SBi-Transformer"]["rmse"]) +
               f" | {np.mean(TABLE3['SBi-Transformer']['rmse']):.2f} | — |")
    out.append("")

    out += ["## Score (lower = better)", "",
            "| Model | " + " | ".join(SUBSETS) + " | mean |", "|---|" + "---|" * 5]
    for r in sorted(rows, key=lambda r: r["avg_score"]):
        lab = SBI_LABEL.get(r["name"], r["name"])
        out.append(f"| {lab} | " + " | ".join(cell(x, "{:.1f}") for x in r["score"]) +
                   f" | **{r['avg_score']:.1f}** |")
    out.append("")

    out += ["## Compute cost (seconds / epoch, mean over 4 subsets, shared V100)", "",
            "| Model | s/epoch | params |", "|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["sec"]):
        out.append(f"| {SBI_LABEL.get(r['name'], r['name'])} | {r['sec']:.2f} | {r['n_params']} |")
    out.append("")

    md = "\n".join(out)
    print(md)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
