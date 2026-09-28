"""Bao cao ket qua autoresearch: xu huong thiet ke tren VAL + bang TEST cuoi cung."""
import argparse
import glob
import json
import os
import statistics as st
from collections import Counter, defaultdict

from paper_values import SUBSETS, TABLE3

PAPER_RMSE = TABLE3["SBi-Transformer"]["rmse"]
PAPER_SCORE = TABLE3["SBi-Transformer"]["score"]
KEYS = ["arch", "seq_len", "rul_cap", "feature_mode", "cond_norm", "num_hidden",
        "ffn_hidden", "encoder_layers", "bilstm_size", "n_heads", "lr", "dropout",
        "weight_decay", "batch_size"]


def load_stage(d, stage):
    rs = []
    for f in sorted(glob.glob(os.path.join(d, f"stage{stage}_results.jsonl*"))):
        rs += [json.loads(l) for l in open(f)]
    return rs


def load_final(d):
    rows = defaultdict(dict)
    cfg = {}
    for f in sorted(glob.glob(os.path.join(d, "final_results.jsonl*"))):
        for l in open(f):
            r = json.loads(l)
            rows[r["id"]][r["subset"]] = r
            cfg[r["id"]] = r["cfg"]
    return rows, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="../results/rul_ar")
    p.add_argument("--md", default=None)
    a = p.parse_args()

    out = ["# Autoresearch — tim phuong phap RUL manh tren C-MAPSS", "",
           "Vong lap successive halving 3 tang, **chon loc hoan toan tren tap validation**; "
           "tap test chi duoc mo mot lan o buoc cuoi cho top-3.", ""]

    # --- tien trinh cac tang ---
    out += ["## Cac tang tim kiem (xep hang theo val)", "",
            "| tang | so cau hinh | ngan sach | val tot nhat | val trung vi |",
            "|---|---|---|---|---|"]
    budgets = {1: "30 epoch", 2: "70 epoch", 3: "150 epoch, 2 seed"}
    survivors = None
    for s in (1, 2, 3):
        rs = load_stage(a.dir, s)
        if not rs:
            continue
        v = sorted(r["val"] for r in rs)
        out.append(f"| {s} | {len(rs)} | {budgets[s]} | {v[0]:.3f} | {st.median(v):.3f} |")
        survivors = rs
    out.append("")

    # --- xu huong thiet ke o tang cuoi ---
    if survivors:
        keep = sorted(survivors, key=lambda r: r["val"])[:max(3, len(survivors) // 2)]
        out += ["## Xu huong thiet ke — nhom song sot o tang cuoi so voi cau hinh cua paper", "",
                "| tham so | paper (Table 2) | pho bien nhat trong nhom song sot |",
                "|---|---|---|"]
        paper_cfg = {"arch": "sbi:yes_yes_yes", "seq_len": 45, "rul_cap": 125,
                     "feature_mode": "paper", "cond_norm": "khong neu",
                     "num_hidden": 16, "ffn_hidden": 32, "encoder_layers": "3 / 2",
                     "bilstm_size": 32, "n_heads": 2, "lr": "5e-4 / 1e-4",
                     "dropout": "0.2 / 0.3", "weight_decay": 1e-5, "batch_size": 256}
        for k in KEYS:
            c = Counter(str(r["cfg"][k]) for r in keep)
            top = ", ".join(f"`{v}` ({n}/{len(keep)})" for v, n in c.most_common(2))
            out.append(f"| `{k}` | {paper_cfg.get(k, '—')} | {top} |")
        out.append("")

    # --- #5: tach bo tham gia tim kiem khoi bo held-out that ---
    SEARCH = ["FD001", "FD004"]      # tim kiem chay tren 2 bo nay (qua val cua chinh chung)
    HELD = ["FD002", "FD003"]        # hai bo nay chua he tham gia chon loc

    # --- bang TEST cuoi ---
    rows, cfg = load_final(a.dir)
    if rows:
        order = sorted(rows, key=lambda k: st.mean(
            [st.mean(rows[k][s]["rmse"]) for s in SUBSETS if s in rows[k]]))
        out += ["## Ket qua TEST cuoi cung (3 seed moi o)", "",
                "| cau hinh | " + " | ".join(SUBSETS) + " | TB | tham so |",
                "|---|" + "---|" * 6]
        for i, k in enumerate(order, 1):
            v = rows[k]
            rm = [st.mean(v[s]["rmse"]) for s in SUBSETS if s in v]
            sd = [st.pstdev(v[s]["rmse"]) for s in SUBSETS if s in v]
            out.append(f"| #{i} `{k}` | " +
                       " | ".join(f"{m:.2f} ±{d:.2f}" for m, d in zip(rm, sd)) +
                       f" | **{st.mean(rm):.2f}** | {v[SUBSETS[0]]['n_params']} |")
        out.append("| **paper cong bo** | " + " | ".join(f"{x:.2f}" for x in PAPER_RMSE) +
                   f" | {st.mean(PAPER_RMSE):.2f} | — |")
        out.append("")
        out += ["**#5 — tach bo tham gia tim kiem khoi bo held-out.** Vong lap chi chay tren "
                "`FD001` + `FD004` (qua tap val cua chinh chung), nen `FD002` + `FD003` la "
                "held-out that. Neu con so tren hai nhom lech nhau nhieu thi do la dau hieu "
                "cau hinh bi chuyen biet hoa cho cap dung de tim.", "",
                "| cau hinh | tim kiem (FD001, FD004) | held-out (FD002, FD003) | chenh |",
                "|---|---|---|---|"]
        for i, k in enumerate(order, 1):
            v = rows[k]
            a_ = st.mean([st.mean(v[s]["rmse"]) for s in SEARCH if s in v])
            b_ = st.mean([st.mean(v[s]["rmse"]) for s in HELD if s in v])
            out.append(f"| #{i} `{k}` | {a_:.2f} | {b_:.2f} | {b_ - a_:+.2f} |")
        out.append("| ICL4RUL [41] (paper trich) | " +
                   " | ".join(f"{x:.2f}" for x in TABLE3["ICL4RUL [41]"]["rmse"]) +
                   f" | {st.mean(TABLE3['ICL4RUL [41]']['rmse']):.2f} | — |")
        out.append("")

        out += ["### Score", "", "| cau hinh | " + " | ".join(SUBSETS) + " | TB |",
                "|---|" + "---|" * 5]
        for i, k in enumerate(order, 1):
            v = rows[k]
            sc = [st.mean(v[s]["score"]) for s in SUBSETS if s in v]
            out.append(f"| #{i} `{k}` | " + " | ".join(f"{x:.1f}" for x in sc) +
                       f" | **{st.mean(sc):.1f}** |")
        out.append("| **paper cong bo** | " + " | ".join(f"{x:.2f}" for x in PAPER_SCORE) +
                   f" | {st.mean(PAPER_SCORE):.1f} |")
        out.append("")

        out += ["### Cau hinh chi tiet", ""]
        for i, k in enumerate(order, 1):
            c = cfg[k]
            v = rows[k]
            out.append(f"**#{i} `{k}`** — val (tang 3) = {v[SUBSETS[0]]['val_stage']:.3f}")
            out.append("")
            out.append("```")
            out.append("  ".join(f"{x}={c[x]}" for x in KEYS[:5]))
            out.append("  ".join(f"{x}={c[x]}" for x in KEYS[5:10]))
            out.append("  ".join(f"{x}={c[x]}" for x in KEYS[10:]))
            out.append("```")
            out.append("")
        vals = [(k, rows[k][SUBSETS[0]]["val_stage"]) for k in order]
        best_val = min(vals, key=lambda x: x[1])[0]
        if best_val != order[0]:
            out += ["> Luu y ve phuong phap: cau hinh co val TOT NHAT "
                    f"(`{best_val}`) khong phai cau hinh co test tot nhat (`{order[0]}`). "
                    "Val khong du bao hoan hao test — do chinh la ly do phai chot top-K "
                    "theo val roi moi mo test, thay vi xep hang lai theo test.", ""]

    md = "\n".join(out)
    print(md)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
