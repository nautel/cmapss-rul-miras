"""#8 — ablation co lap cho cau hinh cuoi cua autoresearch.

Doc cau hinh tot nhat (theo VAL) tu thu muc autoresearch, roi sinh cac bien the
"doi MOT yeu to ve gia tri cua paper" (one-factor-at-a-time). Nho vay biet duoc
tung khac biet so voi paper dong gop bao nhieu, thay vi chi biet ca goi tot hon.
"""
import argparse
import glob
import json
import os
import traceback

import train as T

# gia tri tuong ung cua paper 1 (Table 2) cho tung yeu to
PAPER = {
    "arch": "sbi:yes_yes_yes", "seq_len": 45, "rul_cap": 125,
    "feature_mode": "paper", "cond_norm": False, "num_hidden": 16,
    "ffn_hidden": 32, "encoder_layers": 3, "bilstm_size": 32, "n_heads": 2,
    "lr": 5e-4, "dropout": 0.2, "weight_decay": 1e-5, "batch_size": 256,
}
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def best_cfg(d):
    """Cau hinh co val tot nhat o tang cuoi co du lieu."""
    for stage in (3, 2, 1):
        rs = []
        for f in sorted(glob.glob(os.path.join(d, f"stage{stage}_results.jsonl*"))):
            rs += [json.loads(l) for l in open(f)]
        if rs:
            return min(rs, key=lambda r: r["val"])["cfg"], stage
    raise SystemExit(f"khong co ket qua trong {d}")


def variants(cfg):
    """[(ten, cfg)] — ban day du + moi bien the tra MOT yeu to ve gia tri paper."""
    out = [("full", dict(cfg))]
    for k, pv in PAPER.items():
        if k not in cfg or str(cfg[k]) == str(pv):
            continue                      # yeu to nay von da giong paper
        c = dict(cfg)
        c[k] = pv
        out.append((f"-{k}", c))
    return out


def run_cfg(c, name, seeds, root, outdir, budget, subsets, patience):
    arch, ablation = c["arch"], "yes_yes_yes"
    if str(arch).startswith("sbi:"):
        ablation, arch = arch.split(":")[1], "sbi"
    for s in subsets:
        for sd in seeds:
            tag = f"abl_{name}_{s}_s{sd}"
            if os.path.exists(os.path.join(outdir, f"res_{tag}.json")):
                continue
            ov = {k: c[k] for k in
                  ("seq_len", "rul_cap", "num_hidden", "ffn_hidden", "encoder_layers",
                   "bilstm_size", "n_heads", "lr", "dropout", "weight_decay",
                   "batch_size") if k in c}
            ov.update(epochs=budget, patience=patience,
                      skip_uncertainty=True, tag=tag)
            try:
                r = T.run(s, ablation, sd, root, outdir, None, bool(c["cond_norm"]),
                          c["feature_mode"], quiet=True, arch=arch, overrides=ov)
                print(f"  {name:22s} {s} s{sd}  RMSE={r['rmse']:.2f}", flush=True)
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ar-dir", default="/home/lab/letuan/runs/rul_ar2")
    p.add_argument("--outdir", default="/home/lab/letuan/runs/rul_abl")
    p.add_argument("--root", default="/home/lab/letuan/data/cmapss")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    cfg, stage = best_cfg(a.ar_dir)
    vs = variants(cfg)
    if a.list:
        print(f"cau hinh goc (tang {stage}):")
        for k in PAPER:
            print(f"   {k:16s} = {cfg.get(k)}   (paper: {PAPER[k]})")
        print(f"\n{len(vs)} bien the: " + ", ".join(n for n, _ in vs))
        raise SystemExit

    os.makedirs(a.outdir, exist_ok=True)
    json.dump({n: c for n, c in vs}, open(os.path.join(a.outdir, "variants.json"), "w"),
              indent=1, default=str)
    i, n = (int(x) for x in a.shard.split("/"))
    jobs = [(nm, c, s) for nm, c in vs for s in SUBSETS]
    jobs = [j for k, j in enumerate(jobs) if k % n == i]
    seeds = [int(x) for x in a.seeds.split(",")]
    for k, (nm, c, s) in enumerate(jobs):
        print(f"[{k+1}/{len(jobs)}] {nm} / {s}", flush=True)
        run_cfg(c, nm, seeds, a.root, a.outdir, a.budget, [s], a.patience)
