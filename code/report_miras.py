"""So sanh ho Miras (paper 2) voi SBi-Transformer va cac baseline (paper 1) tren C-MAPSS.

Gom ket qua tu ca hai thu muc chay: `--sbi` (giai doan 1-4) va `--miras` (giai doan 5-6).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from paper_values import SUBSETS, TABLE3

# nhom thiet ke theo Miras: (attentional bias, retention gate)
MIRAS_INFO = {
    "linear_attn":    ("dot product (Hebbian)", "khong quen (alpha=1)"),
    "retnet":         ("dot product", "alpha hang so hoc duoc"),
    "mamba2":         ("dot product", "alpha_t theo du lieu"),
    "deltanet":       ("l2 (delta rule)", "khong quen (alpha=1)"),
    "gated_deltanet": ("l2 (delta rule)", "alpha_t theo du lieu"),
    "titans":         ("l2 + momentum", "alpha_t theo du lieu"),
    "moneta":         ("l_p, p=3", "chuan hoa l_q, q=4"),
    "yaad":           ("Huber (l2 / l1)", "alpha_t theo du lieu"),
    "memora":         ("l2", "KL / softmax"),
    "elastic":        ("l2", "elastic net, soft-threshold"),
    "robust":         ("l2 + worst-case shift", "alpha_t theo du lieu"),
    "titans_mlp":     ("l2 + momentum, bo nho MLP sau", "alpha_t theo du lieu"),
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
    """Lay cau hinh tot nhat (theo RMSE trung binh) cua mot mo hinh tren mot subset."""
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

    out = ["# Thu ho Miras tren C-MAPSS — va huong nao hop voi bai toan RUL", "",
           "Paper 2: Behrouz, Razaviyayn, Zhong, Mirrokni, *It's All Connected* "
           "(arXiv:2504.13173, Google Research 2025).", "",
           "Moi kien truc dung CUNG backbone (chieu tuyen tinh + position embedding hoc duoc "
           "-> N khoi -> FC tren buoc cuoi) va CUNG sieu tham so Table 2 cua paper 1, chi khac "
           "khoi tron chuoi. Moi o = trung binh 3 seed, lay cau hinh tien xu ly tot nhat "
           "(z-score cho FD001/FD003, chuan hoa theo che do van hanh cho FD002/FD004).", "",
           "## Bang xep hang (RMSE, thap hon = tot hon)", "",
           "| # | Mo hinh | attentional bias | retention gate | " +
           " | ".join(SUBSETS) + " | TB | tham so |",
           "|---|---|---|---|" + "---|" * 6]
    for i, r in enumerate(rows, 1):
        base = r["name"].replace("+bilstm", "")
        bias, ret = MIRAS_INFO.get(base, ("—", "—"))
        lab = SBI_LABEL.get(r["name"], r["name"])
        if r["name"].endswith("+bilstm"):
            lab += " + dau BiLSTM"
        out.append(f"| {i} | {lab} | {bias} | {ret} | " +
                   " | ".join(cell(x) for x in r["rmse"]) +
                   f" | **{r['avg']:.2f}** | {r['n_params']} |")
    out.append(f"| — | SBi-Transformer *(so paper 1 cong bo)* | — | — | " +
               " | ".join(f"{x:.2f}" for x in TABLE3["SBi-Transformer"]["rmse"]) +
               f" | {np.mean(TABLE3['SBi-Transformer']['rmse']):.2f} | — |")
    out.append("")

    out += ["## Score (thap hon = tot hon)", "",
            "| Mo hinh | " + " | ".join(SUBSETS) + " | TB |", "|---|" + "---|" * 5]
    for r in sorted(rows, key=lambda r: r["avg_score"]):
        lab = SBI_LABEL.get(r["name"], r["name"])
        out.append(f"| {lab} | " + " | ".join(cell(x, "{:.1f}") for x in r["score"]) +
                   f" | **{r['avg_score']:.1f}** |")
    out.append("")

    out += ["## Chi phi tinh toan (giay / epoch, trung binh 4 subset, V100 dung chung)", "",
            "| Mo hinh | s/epoch | tham so |", "|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["sec"]):
        out.append(f"| {SBI_LABEL.get(r['name'], r['name'])} | {r['sec']:.2f} | {r['n_params']} |")
    out.append("")

    md = "\n".join(out)
    print(md)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
