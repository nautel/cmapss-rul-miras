"""Run the full reproduction grid: 4 subsets x 5 ablation configs x n seeds."""
import argparse
import json
import os
import time
import traceback

import train as T
from model import ABLATIONS
from miras import VARIANTS as MIRAS_VARIANTS
from baselines import BASELINES

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(SUBSETS))
    p.add_argument("--ablations", default=",".join(ABLATIONS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--root", default="../data/CMAPSSData")
    p.add_argument("--outdir", default=os.environ.get("OUTDIR", "./out"))
    p.add_argument("--device", default=None)
    p.add_argument("--cond-norm", action="store_true")
    p.add_argument("--feature-mode", default="paper")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--full-epochs", action="store_true")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--archs", default=None, help="Miras archs, replaces --ablations")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--shard", default="0/1", help="i/n — split grid over n parallel processes")
    a = p.parse_args()

    ds = a.datasets.split(",")
    abl = a.archs.split(",") if a.archs else a.ablations.split(",")
    is_miras = bool(a.archs)
    seeds = [int(s) for s in a.seeds.split(",")]
    si, sn = (int(x) for x in a.shard.split("/"))
    # sort by estimated cost, descending -> every shard gets heavy jobs first
    COST = {"no_no_yes": 0.2, "yes_no_yes": 0.6, "yes_no_no": 0.65,
            "yes_yes_no": 0.8, "yes_yes_yes": 1.0, "titans_mlp": 6.0}
    SIZE = {"FD001": 1.0, "FD003": 1.25, "FD002": 7.7, "FD004": 9.2}
    jobs = sorted([(s, ab, sd) for s in ds for ab in abl for sd in seeds],
                  key=lambda j: -SIZE.get(j[0], 1) * COST.get(j[1], 1))
    jobs = [j for k, j in enumerate(jobs) if k % sn == si]
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    done, failed = [], []
    total = len(jobs)
    i = 0
    for (s, ab, sd) in jobs:
        i += 1
        tag = (f"{s}_{ab}_s{sd}" + ("_cn" if a.cond_norm else "")
               + ("_fe" if a.full_epochs else "")
               + (f"_L{a.seq_len}" if a.seq_len and a.seq_len != 45 else ""))
        path = os.path.join(a.outdir, f"res_{tag}.json")
        if os.path.exists(path):
            print(f"[{i}/{total}] SKIP {tag} (exists)", flush=True)
            done.append(json.load(open(path)))
            continue
        print(f"[{i}/{total}] === {tag}  (+{time.time()-t0:.0f}s) ===", flush=True)
        try:
            done.append(T.run(s, "yes_yes_yes" if is_miras else ab, sd, a.root,
                              a.outdir, a.device, a.cond_norm, a.feature_mode,
                              a.epochs, quiet=False, full_epochs=a.full_epochs,
                              seq_len=a.seq_len, arch=ab if is_miras else "sbi",
                              patience=a.patience))
        except Exception:
            traceback.print_exc()
            failed.append(tag)
    if sn == 1:
        with open(os.path.join(a.outdir, "all_results.json"), "w") as f:
            json.dump(done, f, indent=1)
    print(f"\nDONE shard {si}/{sn}: {len(done)}/{total} in {time.time()-t0:.0f}s. Errors: {failed}",
          flush=True)


if __name__ == "__main__":
    main()
