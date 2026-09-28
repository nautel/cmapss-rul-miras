"""Chay lai ho Miras sau khi sua loi (09-2026), cung pipeline voi moc so sanh.

Loi da sua trong miras.py:
  1. memora sai dau (di LEN gradient)
  2. titans mat momentum qua ranh gioi chunk
  3. nhanh cam bien dung mask nhan qua tren truc khong co thu tu
  4. bo `rank_bias` (V la chieu hoc duoc, khong phai RUL -> y tuong khong dung)
Va do anh huong cua xap xi chunk bang `--chunks` (1 = truy hoi chinh xac).

Pipeline chung cho MOI mo hinh (ke ca moc DCNN): 14 sensor kinh dien, cua so
40 (FD001) / 60 (con lai), min-max cho FD001/FD003, chuan hoa theo che do van hanh
cho FD002/FD004, nhan cat 125, val 20% engine, dung som tren val.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import data as D
from baselines import build_baseline
from miras import build_miras
from rve import batch_hard_triplet
from train import rmse, score, config_for

SEQ = {"FD001": 40, "FD002": 60, "FD003": 60, "FD004": 60}
MIRAS = ["linear_attn", "mamba2", "deltanet", "gated_deltanet", "titans",
         "moneta", "yaad", "memora"]
# ten -> (arch, ghi de cau hinh, trong so triplet)
IDEAS = {v: (v, {}, 0.0) for v in MIRAS}
IDEAS.update({
    "titans+bilstm": ("titans+bilstm", {}, 0.0),
    "titans_R4":     ("titans", {"mem_rank": 4}, 0.0),
    "titans_B3trip": ("titans", {"bottleneck": 3}, 150.0),
    "titans_S":      ("titans", {"sensor_branch": 1}, 0.0),
    "dcnn":          ("bl:dcnn", {}, 0.0),               # moc, cung pipeline
})


def predict(model, X, cap, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs]).float().cpu().numpy())
    return np.clip(np.concatenate(out) * cap, 0.0, cap)


def run(subset, idea, seed, chunk, root, outdir, cond_norm, epochs=200, patience=25,
        quiet=True):
    arch, ov, w_trip = IDEAS[idea]
    cfg = config_for(subset, epochs=epochs, patience=patience, seq_len=SEQ[subset],
                     num_hidden=32, ffn_hidden=64, encoder_layers=2, dropout=0.2,
                     lr=5e-4, batch_size=256)
    cfg.update(ov)
    cfg["chunk"] = chunk
    assert SEQ[subset] % chunk == 0, f"seq_len {SEQ[subset]} khong chia het chunk {chunk}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = D.build(root, subset, seq_len=cfg["seq_len"], feature_mode="classic14",
                cond_norm=cond_norm, seed=seed, cap=125.0, eval_cap=D.RUL_CAP,
                val_frac=0.2, norm="minmax")
    cfg["input_size"] = len(d["features"])
    cap = 125.0
    T = lambda a: torch.as_tensor(a, device=dev)
    Xtr, ytr = T(d["Xtr"]), T(d["ytr"] / cap)
    Xva, yva, Xte, yte = T(d["Xval"]), d["yval"], T(d["Xte"]), d["yte"]

    model = (build_baseline(cfg, arch[3:]) if arch.startswith("bl:")
             else build_miras(cfg, arch)).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    N, bs = len(Xtr), cfg["batch_size"]
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, times = np.inf, None, 0, []
    for ep in range(cfg["epochs"]):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(N, generator=g).to(dev)
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            if w_trip:
                y, z = model(Xtr[b], return_z=True)
                loss = F.mse_loss(y, ytr[b]) + w_trip * batch_hard_triplet(z, ytr[b])
            else:
                loss = F.mse_loss(model(Xtr[b]), ytr[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        vr = rmse(predict(model, Xva, cap), yva)
        if vr < best - 1e-6:
            best, bad = vr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        if not quiet and ep % 25 == 0:
            print(f"[{subset}/{idea}/c{chunk}/s{seed}] ep{ep:3d} val={vr:.3f} "
                  f"best={best:.3f}", flush=True)

    model.load_state_dict(best_state)
    pred = predict(model, Xte, cap)
    tag = f"{subset}_mf_{idea}_c{chunk}_s{seed}" + ("_cn" if cond_norm else "")
    res = dict(subset=subset, arch=f"mf_{idea}", idea=idea, chunk=chunk, seed=seed,
               rmse=rmse(pred, yte), score=score(pred, yte), val_rmse=best,
               epochs_run=len(times), sec_per_epoch=float(np.median(times)),
               n_params=nparam, cond_norm=cond_norm,
               cfg={k: v for k, v in cfg.items()})
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, f"pred_{tag}.npz"),
                        pred=pred, true=yte, unit=d["ute"])
    json.dump(res, open(os.path.join(outdir, f"res_{tag}.json"), "w"), indent=1)
    print(f"RESULT {tag}: RMSE={res['rmse']:.2f} Score={res['score']:.1f} "
          f"({res['epochs_run']} ep, {res['sec_per_epoch']:.2f}s/ep, {nparam} params)",
          flush=True)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    p.add_argument("--ideas", default=",".join(IDEAS))
    p.add_argument("--chunks", default="5")
    p.add_argument("--seeds", default="0")
    p.add_argument("--root", default="/home/lab/letuan/data/cmapss")
    p.add_argument("--outdir", default="/home/lab/letuan/runs/rul_mfix")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    jobs = [(s, k, int(c), int(sd)) for s in a.subsets.split(",")
            for k in a.ideas.split(",") for c in a.chunks.split(",")
            for sd in a.seeds.split(",")]
    i, n = (int(x) for x in a.shard.split("/"))
    for j, (s, k, c, sd) in enumerate(jobs):
        if j % n != i:
            continue
        cn = s in ("FD002", "FD004")          # chuan hoa theo che do van hanh
        if k == "dcnn" and c != int(a.chunks.split(",")[0]):
            continue                          # DCNN khong co chunk: chay 1 lan
        tag = f"{s}_mf_{k}_c{c}_s{sd}" + ("_cn" if cn else "")
        if os.path.exists(os.path.join(a.outdir, f"res_{tag}.json")):
            print(f"SKIP {tag}", flush=True)
            continue
        run(s, k, sd, c, a.root, a.outdir, cn, a.epochs, quiet=not a.verbose)
