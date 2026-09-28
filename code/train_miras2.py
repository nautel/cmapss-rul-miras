"""Thi nghiem: dua bai hoc tu TSHAE + STA-HPINN vao khung Miras.

Miras dinh nghia mo hinh bang 4 lua chon. Bon y tuong duoi day la 4 lua chon do
day theo huong ma hai phuong phap manh nhat (TSHAE, STA-HPINN) da di, va nguoc voi
huong paper Miras chon:

| ma  | y tuong                       | lua chon cua Miras | nguon |
|-----|-------------------------------|--------------------|-------|
| `R` | bo nho HANG THAP (rank 2/4/8) | 1 — cau truc bo nho| TSHAE latent=2, STA-HPINN hidden=3 |
| `B` | nut that o dau ra + triplet   | (dau doc bo nho)   | TSHAE triplet w=150 |
| `S` | them nhanh tren chieu CAM BIEN| (chieu tac dong)   | STA-HPINN 2 nhanh song song |
| `K` | attentional bias theo THU TU  | 2 — attentional bias| RUL don dieu |

Moi bien the deu so voi `base` = Miras nguyen ban (titans) trong CUNG pipeline.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import data as D
from miras import build_miras
from rve import batch_hard_triplet
from train import rmse, score, config_for

SEQ = {"FD001": 40, "FD002": 60, "FD003": 60, "FD004": 60}

# ten bien the -> ghi de cau hinh
IDEAS = {
    "base":        dict(),
    "R2":          dict(mem_rank=2),
    "R4":          dict(mem_rank=4),
    "R8":          dict(mem_rank=8),
    "B3":          dict(bottleneck=3),
    "B3trip":      dict(bottleneck=3, w_trip=150.0),
    "B8trip":      dict(bottleneck=8, w_trip=150.0),
    "S":           dict(sensor_branch=1),
    "R4_B3trip":   dict(mem_rank=4, bottleneck=3, w_trip=150.0),
    "R4_B3trip_S": dict(mem_rank=4, bottleneck=3, w_trip=150.0, sensor_branch=1),
}
# LUU Y (09-2026): da thay bang train_miras_fix.py sau khi sua loi trong miras.py.
# `K`/`all` (rank_bias) da bo vi y tuong khong dung (V khong phai RUL).


def predict(model, X, cap, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs]).float().cpu().numpy())
    return np.clip(np.concatenate(out) * cap, 0.0, cap)


def run(subset, idea="base", seed=0, root="../data/CMAPSSData", outdir="./out",
        cond_norm=False, epochs=200, patience=25, quiet=True):
    ov = dict(IDEAS[idea])
    arch = ov.pop("arch", "titans")
    w_trip = ov.pop("w_trip", 0.0)
    cfg = config_for(subset, epochs=epochs, patience=patience,
                     seq_len=SEQ[subset], num_hidden=32, ffn_hidden=64,
                     encoder_layers=2, dropout=0.2, lr=5e-4, batch_size=256)
    cfg.update(ov)
    cfg["chunk"] = 20 if SEQ[subset] == 60 else 20     # 60/20=3, 40/20=2
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

    model = build_miras(cfg, arch).to(dev)
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
        tot = 0.0
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            if w_trip:
                y, z = model(Xtr[b], return_z=True)
                l = F.mse_loss(y, ytr[b])
                loss = l + w_trip * batch_hard_triplet(z, ytr[b])
            else:
                y = model(Xtr[b])
                loss = l = F.mse_loss(y, ytr[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(l) * len(b)
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
            print(f"[{subset}/{idea}/s{seed}] ep{ep:3d} mse={tot/N:.5f} val={vr:.3f} "
                  f"best={best:.3f}", flush=True)

    model.load_state_dict(best_state)
    pred = predict(model, Xte, cap)
    tag = f"{subset}_m2_{idea}_s{seed}" + ("_cn" if cond_norm else "")
    res = dict(subset=subset, ablation="miras2", arch=f"m2_{idea}", idea=idea,
               seed=seed, rmse=rmse(pred, yte), score=score(pred, yte), val_rmse=best,
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
    p.add_argument("--seeds", default="0")
    p.add_argument("--root", default="/home/lab/letuan/data/cmapss")
    p.add_argument("--outdir", default="/home/lab/letuan/runs/rul_m2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--cond-norm", action="store_true")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    jobs = [(s, k, int(sd)) for s in a.subsets.split(",")
            for k in a.ideas.split(",") for sd in a.seeds.split(",")]
    i, n = (int(x) for x in a.shard.split("/"))
    for j, (s, k, sd) in enumerate(jobs):
        if j % n != i:
            continue
        tag = f"{s}_m2_{k}_s{sd}" + ("_cn" if a.cond_norm else "")
        if os.path.exists(os.path.join(a.outdir, f"res_{tag}.json")):
            print(f"SKIP {tag}", flush=True); continue
        run(s, k, sd, a.root, a.outdir, a.cond_norm, a.epochs, quiet=not a.verbose)
