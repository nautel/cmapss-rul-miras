"""Vong lap nghien cuu tu dong (autoresearch) tim phuong phap RUL manh tren C-MAPSS.

NGUYEN TAC BAT BUOC: moi quyet dinh chon loc chi dua tren tap VALIDATION.
Tap test chi duoc mo DUNG MOT LAN, o buoc cuoi, cho vai cau hinh da chot. Chay hang
tram cau hinh roi chon theo test la tu lua minh — sai so chon loc se lon hon chinh
khoang cach ma ta dang do.

Ba giai doan (successive halving):
  1. `propose --stage 1` : lay mau ngau nhien N cau hinh trong khong gian thiet ke
  2. `select  --stage k` : giu top theo val, sinh giai doan k+1 bang dot bien cuc bo
  3. `final`             : lay top-K, chay nhieu seed, danh gia TEST + thu ensemble

Khong gian thiet ke duoc gieo bang nhung gi da do duoc o giai doan 1-6:
tien xu ly quan trong hon kien truc, nen no chiem phan lon cac bac tu do.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import time
import traceback

import numpy as np

import train as T
from miras import VARIANTS as MIRAS_VARIANTS

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# --- khong gian thiet ke -----------------------------------------------------
SPACE = {
    # tien xu ly — bac tu do quan trong nhat theo do luong o giai doan 1-4
    "cond_norm":     [True, False],
    "seq_len":       [30, 45, 60],
    "rul_cap":       [110, 125, 140],
    "feature_mode":  ["paper", "classic14"],
    # backbone
    "arch":          (["sbi:yes_yes_yes", "sbi:yes_yes_no", "sbi:yes_no_yes", "sbi:no_no_yes"]
                      + ["titans", "gated_deltanet", "yaad", "moneta", "linear_attn",
                         "deltanet", "mamba2", "memora"]
                      + ["titans+bilstm", "gated_deltanet+bilstm", "yaad+bilstm"]),
    # dung luong
    "num_hidden":    [16, 32, 64],
    "ffn_hidden":    [32, 64, 128],
    "encoder_layers": [1, 2, 3, 4],
    "bilstm_size":   [32, 64],
    "n_heads":       [2, 4],
    # huan luyen
    "lr":            [1e-4, 3e-4, 5e-4, 1e-3],
    "dropout":       [0.1, 0.2, 0.3],
    "weight_decay":  [1e-5, 1e-4],
    "batch_size":    [128, 256],
}
# rang buoc: seq_len phai chia het cho chunk 15 cua lop Miras
VALID_SEQ = {"miras": [30, 45, 60], "sbi": [30, 45, 60]}


def sample(rng):
    c = {k: rng.choice(v) for k, v in SPACE.items()}
    c["cond_norm"] = bool(c["cond_norm"])
    if c["num_hidden"] % c["n_heads"]:
        c["n_heads"] = 2
    return c


def mutate(base, rng, n_change=2):
    c = copy.deepcopy(base)
    for k in rng.sample(list(SPACE), n_change):
        c[k] = rng.choice(SPACE[k])
    c["cond_norm"] = bool(c["cond_norm"])
    if c["num_hidden"] % c["n_heads"]:
        c["n_heads"] = 2
    return c


def cfg_id(c):
    """Bam on dinh giua cac tien trinh — hash() cua Python co PYTHONHASHSEED ngau nhien."""
    keys = sorted(SPACE)
    raw = "|".join(f"{k}={c[k]}" for k in keys)
    return "h" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def to_overrides(c, budget):
    o = {k: c[k] for k in ("seq_len", "rul_cap", "num_hidden", "ffn_hidden",
                           "encoder_layers", "bilstm_size", "n_heads", "lr",
                           "dropout", "weight_decay", "batch_size")}
    o["epochs"] = budget
    o["patience"] = max(8, budget // 5)
    o["skip_uncertainty"] = True     # MC-dropout 50 luot qua dat cho vong tim kiem
    return o


# --- chay mot cau hinh -------------------------------------------------------
def run_one(c, budget, seeds, root, outdir, subsets):
    """Tra ve val RMSE trung binh tren cac subset — KHONG cham vao test."""
    arch = c["arch"]
    ablation = "yes_yes_yes"
    if arch.startswith("sbi:"):
        ablation, arch = arch.split(":")[1], "sbi"
    vals, tests, secs = [], [], []
    for s in subsets:
        for sd in seeds:
            ov = to_overrides(c, budget)
            ov["tag"] = f"ar_{cfg_id(c)}_{s}_s{sd}_b{budget}"
            r = T.run(s, ablation, sd, root, outdir, None,
                      c["cond_norm"], c["feature_mode"], quiet=True,
                      arch=arch, overrides=ov)
            vals.append(r["val_rmse"])
            tests.append(r["rmse"])          # ghi lai nhung KHONG dung de chon
            secs.append(r["sec_per_epoch"])
    return dict(id=cfg_id(c), cfg=c, budget=budget, seeds=list(seeds),
                subsets=list(subsets), val=float(np.mean(vals)),
                val_all=vals, test_hidden=tests, sec=float(np.mean(secs)))


def stage_path(outdir, stage):
    return os.path.join(outdir, f"stage{stage}_configs.json")


def results_path(outdir, stage):
    return os.path.join(outdir, f"stage{stage}_results.jsonl")


def cmd_propose(a):
    rng = random.Random(a.seed)
    os.makedirs(a.outdir, exist_ok=True)
    cfgs = [sample(rng) for _ in range(a.n)]
    # gieo them cac cau hinh da biet la manh (baseline de so sanh trong cung khung)
    seeds_known = [
        dict(cond_norm=True, seq_len=45, rul_cap=125, feature_mode="paper",
             arch="sbi:yes_yes_yes", num_hidden=16, ffn_hidden=32, encoder_layers=3,
             bilstm_size=32, n_heads=2, lr=5e-4, dropout=0.2, weight_decay=1e-5,
             batch_size=256),
        dict(cond_norm=True, seq_len=45, rul_cap=125, feature_mode="paper",
             arch="titans", num_hidden=16, ffn_hidden=32, encoder_layers=3,
             bilstm_size=32, n_heads=2, lr=5e-4, dropout=0.2, weight_decay=1e-5,
             batch_size=256),
    ]
    cfgs = seeds_known + cfgs
    json.dump(cfgs, open(stage_path(a.outdir, a.stage), "w"), indent=1)
    print(f"stage {a.stage}: {len(cfgs)} cau hinh -> {stage_path(a.outdir, a.stage)}")


def cmd_run(a):
    cfgs = json.load(open(stage_path(a.outdir, a.stage)))
    i, n = (int(x) for x in a.shard.split("/"))
    mine = [c for k, c in enumerate(cfgs) if k % n == i]
    seeds = [int(x) for x in a.seeds.split(",")]
    subsets = a.subsets.split(",")
    out = results_path(a.outdir, a.stage) + f".{i}"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    with open(out, "a") as f:
        for k, c in enumerate(mine):
            if cfg_id(c) in done:
                print(f"[{k+1}/{len(mine)}] SKIP {cfg_id(c)}", flush=True)
                continue
            t0 = time.time()
            try:
                r = run_one(c, a.budget, seeds, a.root, a.outdir, subsets)
            except Exception:
                traceback.print_exc()
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{k+1}/{len(mine)}] {r['id']} val={r['val']:.3f} "
                  f"({time.time()-t0:.0f}s) {c['arch']} L{c['seq_len']} "
                  f"h{c['num_hidden']} cn={c['cond_norm']}", flush=True)


def load_results(outdir, stage):
    rs = []
    for f in sorted(os.listdir(outdir)):
        if f.startswith(f"stage{stage}_results.jsonl"):
            rs += [json.loads(l) for l in open(os.path.join(outdir, f))]
    return rs


def cmd_select(a):
    rs = load_results(a.outdir, a.stage)
    rs.sort(key=lambda r: r["val"])
    keep = rs[:max(1, int(len(rs) * a.frac))]
    print(f"stage {a.stage}: {len(rs)} ket qua -> giu {len(keep)}")
    for r in keep[:12]:
        c = r["cfg"]
        print(f"  val={r['val']:.3f}  {c['arch']:22s} L{c['seq_len']} cap{c['rul_cap']} "
              f"h{c['num_hidden']} l{c['encoder_layers']} lr{c['lr']} "
              f"do{c['dropout']} cn={c['cond_norm']} {c['feature_mode']}")
    rng = random.Random(a.seed + a.stage)
    nxt = [r["cfg"] for r in keep]
    while len(nxt) < a.n:                      # dot bien quanh cac cau hinh song sot
        nxt.append(mutate(rng.choice(keep)["cfg"], rng))
    json.dump(nxt, open(stage_path(a.outdir, a.stage + 1), "w"), indent=1)
    print(f"-> stage {a.stage+1}: {len(nxt)} cau hinh")


def cmd_final(a):
    """Buoc DUY NHAT duoc cham vao test: top-K x 4 subset x nhieu seed.

    Chia shard theo cap (cau hinh, subset) de chay song song — chay tuan tu mat ~15 gio.
    """
    rs = load_results(a.outdir, a.stage)
    rs.sort(key=lambda r: r["val"])
    top = rs[:a.k]
    seeds = [int(x) for x in a.seeds.split(",")]
    jobs = [(r, s) for r in top for s in SUBSETS]
    i, n = (int(x) for x in a.shard.split("/"))
    jobs = [j for k, j in enumerate(jobs) if k % n == i]
    out = os.path.join(a.outdir, f"final_results.jsonl.{i}")
    done = {tuple(json.loads(l)["key"]) for l in open(out)} if os.path.exists(out) else set()
    with open(out, "a") as f:
        for j, (r, s) in enumerate(jobs):
            if (r["id"], s) in done:
                print(f"[{j+1}/{len(jobs)}] SKIP {r['id']}/{s}", flush=True)
                continue
            c = r["cfg"]
            arch, ablation = c["arch"], "yes_yes_yes"
            if arch.startswith("sbi:"):
                ablation, arch = arch.split(":")[1], "sbi"
            rr = []
            for sd in seeds:
                ov = to_overrides(c, a.budget)
                ov["patience"] = max(20, a.budget // 4)
                ov["tag"] = f"fin_{r['id']}_{s}_s{sd}"
                rr.append(T.run(s, ablation, sd, a.root, a.outdir, None,
                                c["cond_norm"], c["feature_mode"], quiet=True,
                                arch=arch, overrides=ov))
            rec = dict(key=[r["id"], s], id=r["id"], subset=s, cfg=c,
                       val_stage=r["val"],
                       rmse=[x["rmse"] for x in rr], score=[x["score"] for x in rr],
                       val=[x["val_rmse"] for x in rr],
                       sec=float(np.mean([x["sec_per_epoch"] for x in rr])),
                       n_params=rr[0]["n_params"],
                       tags=[f"fin_{r['id']}_{s}_s{sd}" for sd in seeds])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{j+1}/{len(jobs)}] {r['id']}/{s} TEST rmse={np.mean(rec['rmse']):.3f} "
                  f"score={np.mean(rec['score']):.1f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("propose", "run", "select", "final"):
        q = sub.add_parser(name)
        q.add_argument("--outdir", default="./ar")
        q.add_argument("--stage", type=int, default=1)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--n", type=int, default=48)
        q.add_argument("--budget", type=int, default=60)
        q.add_argument("--frac", type=float, default=0.25)
        q.add_argument("--k", type=int, default=3)
        q.add_argument("--shard", default="0/1")
        q.add_argument("--seeds", default="0")
        q.add_argument("--subsets", default="FD001,FD004")
        q.add_argument("--root", default="/home/lab/letuan/data/cmapss")
    a = p.parse_args()
    {"propose": cmd_propose, "run": cmd_run,
     "select": cmd_select, "final": cmd_final}[a.cmd](a)
