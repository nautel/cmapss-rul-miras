"""Autoresearch results report: design trends on VAL + final TEST table."""
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

    out = ["# Autoresearch — searching for a strong RUL method on C-MAPSS", "",
           "3-stage successive-halving loop, **selection done entirely on the validation "
           "set**; the test set is opened only once, at the final step, for the top-3.", ""]

    # --- stage progression ---
    out += ["## Search stages (ranked by val)", "",
            "| stage | # configs | budget | best val | median val |",
            "|---|---|---|---|---|"]
    budgets = {1: "30 epochs", 2: "70 epochs", 3: "150 epochs, 2 seeds"}
    survivors = None
    for s in (1, 2, 3):
        rs = load_stage(a.dir, s)
        if not rs:
            continue
        v = sorted(r["val"] for r in rs)
        out.append(f"| {s} | {len(rs)} | {budgets[s]} | {v[0]:.3f} | {st.median(v):.3f} |")
        survivors = rs
    out.append("")

    # --- design trends in the last stage ---
    if survivors:
        keep = sorted(survivors, key=lambda r: r["val"])[:max(3, len(survivors) // 2)]
        out += ["## Design trends — last-stage survivors vs. the paper's configuration", "",
                "| parameter | paper (Table 2) | most common among survivors |",
                "|---|---|---|"]
        paper_cfg = {"arch": "sbi:yes_yes_yes", "seq_len": 45, "rul_cap": 125,
                     "feature_mode": "paper", "cond_norm": "not stated",
                     "num_hidden": 16, "ffn_hidden": 32, "encoder_layers": "3 / 2",
                     "bilstm_size": 32, "n_heads": 2, "lr": "5e-4 / 1e-4",
                     "dropout": "0.2 / 0.3", "weight_decay": 1e-5, "batch_size": 256}
        for k in KEYS:
            c = Counter(str(r["cfg"][k]) for r in keep)
            top = ", ".join(f"`{v}` ({n}/{len(keep)})" for v, n in c.most_common(2))
            out.append(f"| `{k}` | {paper_cfg.get(k, '—')} | {top} |")
        out.append("")

    # --- #5: separate the search subsets from the truly held-out ones ---
    SEARCH = ["FD001", "FD004"]      # search ran on these 2 subsets (via their own val)
    HELD = ["FD002", "FD003"]        # these two never took part in selection

    # --- final TEST table ---
    rows, cfg = load_final(a.dir)
    if rows:
        order = sorted(rows, key=lambda k: st.mean(
            [st.mean(rows[k][s]["rmse"]) for s in SUBSETS if s in rows[k]]))
        out += ["## Final TEST results (3 seeds per cell)", "",
                "| config | " + " | ".join(SUBSETS) + " | mean | params |",
                "|---|" + "---|" * 6]
        for i, k in enumerate(order, 1):
            v = rows[k]
            rm = [st.mean(v[s]["rmse"]) for s in SUBSETS if s in v]
            sd = [st.pstdev(v[s]["rmse"]) for s in SUBSETS if s in v]
            out.append(f"| #{i} `{k}` | " +
                       " | ".join(f"{m:.2f} ±{d:.2f}" for m, d in zip(rm, sd)) +
                       f" | **{st.mean(rm):.2f}** | {v[SUBSETS[0]]['n_params']} |")
        out.append("| **published paper** | " + " | ".join(f"{x:.2f}" for x in PAPER_RMSE) +
                   f" | {st.mean(PAPER_RMSE):.2f} | — |")
        out.append("")
        out += ["**#5 — search subsets vs. held-out subsets.** The loop ran only on "
                "`FD001` + `FD004` (via their own val sets), so `FD002` + `FD003` are "
                "truly held out. A large gap between the two groups would signal that the "
                "configs are over-specialized to the pair used for the search.", "",
                "| config | search (FD001, FD004) | held-out (FD002, FD003) | gap |",
                "|---|---|---|---|"]
        for i, k in enumerate(order, 1):
            v = rows[k]
            a_ = st.mean([st.mean(v[s]["rmse"]) for s in SEARCH if s in v])
            b_ = st.mean([st.mean(v[s]["rmse"]) for s in HELD if s in v])
            out.append(f"| #{i} `{k}` | {a_:.2f} | {b_:.2f} | {b_ - a_:+.2f} |")
        out.append("| ICL4RUL [41] (cited in paper) | " +
                   " | ".join(f"{x:.2f}" for x in TABLE3["ICL4RUL [41]"]["rmse"]) +
                   f" | {st.mean(TABLE3['ICL4RUL [41]']['rmse']):.2f} | — |")
        out.append("")

        out += ["### Score", "", "| config | " + " | ".join(SUBSETS) + " | mean |",
                "|---|" + "---|" * 5]
        for i, k in enumerate(order, 1):
            v = rows[k]
            sc = [st.mean(v[s]["score"]) for s in SUBSETS if s in v]
            out.append(f"| #{i} `{k}` | " + " | ".join(f"{x:.1f}" for x in sc) +
                       f" | **{st.mean(sc):.1f}** |")
        out.append("| **published paper** | " + " | ".join(f"{x:.2f}" for x in PAPER_SCORE) +
                   f" | {st.mean(PAPER_SCORE):.1f} |")
        out.append("")

        out += ["### Configuration details", ""]
        for i, k in enumerate(order, 1):
            c = cfg[k]
            v = rows[k]
            out.append(f"**#{i} `{k}`** — val (stage 3) = {v[SUBSETS[0]]['val_stage']:.3f}")
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
            out += ["> Methodological note: the config with the BEST val "
                    f"(`{best_val}`) is not the config with the best test (`{order[0]}`). "
                    "Val does not predict test perfectly — that is exactly why top-K must "
                    "be fixed by val before opening test, instead of re-ranking by test.", ""]

    md = "\n".join(out)
    print(md)
    if a.md:
        open(a.md, "w").write(md)


if __name__ == "__main__":
    main()
