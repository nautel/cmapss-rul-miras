"""Train RVE + ablation of the "narrow bottleneck" idea itself.

Variants (`--variant`):
  full     : latent 2 + recon + triplet          — the original TSHAE
  notrip   : drop triplet                        — contribution of metric learning
  norecon  : drop reconstruction                 — contribution of reconstruction branch
  plain    : regression through bottleneck only  — pure bottleneck, no regularization
`--latent` sweeps the bottleneck width: 2 / 3 / 8 / 32.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import data as D
from rve import build_rve, batch_hard_triplet
from train import rmse, score

CFG = dict(feature_mode="classic14", norm="minmax", cap=125.0, val_frac=0.2,
           batch_size=256, lr=1e-3, epochs=200, patience=25, dropout=0.2,
           rve_hidden=64, rve_layers=1, rve_reg=100,
           w_recon=1.0, w_reg=1.0, w_trip=150.0, w_kl=0.0)
SEQ = {"FD001": 40, "FD002": 60, "FD003": 60, "FD004": 60}
VARIANTS = {"full": (1, 1), "notrip": (1, 0), "norecon": (0, 1), "plain": (0, 0)}


def predict(model, X, cap, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs])[0].float().cpu().numpy())
    return np.clip(np.concatenate(out) * cap, 0.0, cap)


def run(subset, seed=0, variant="full", latent=2, root="../data/CMAPSSData",
        outdir="./out", cond_norm=False, epochs=None, quiet=True):
    cfg = dict(CFG)
    cfg["seq_len"], cfg["rve_latent"] = SEQ[subset], latent
    recon, trip = VARIANTS[variant]
    cfg["rve_recon"] = recon
    if epochs:
        cfg["epochs"] = epochs
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = D.build(root, subset, seq_len=cfg["seq_len"], feature_mode=cfg["feature_mode"],
                cond_norm=cond_norm, seed=seed, cap=cfg["cap"], eval_cap=D.RUL_CAP,
                val_frac=cfg["val_frac"], norm=cfg["norm"])
    cfg["input_size"] = len(d["features"])
    cap = cfg["cap"]
    T = lambda a: torch.as_tensor(a, device=dev)
    Xtr, ytr = T(d["Xtr"]), T(d["ytr"] / cap)
    Xva, yva, Xte, yte = T(d["Xval"]), d["yval"], T(d["Xte"]), d["yte"]

    model = build_rve(cfg).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
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
            y, xh, z, mean, logvar = model(Xtr[b])
            lreg = F.mse_loss(y, ytr[b])
            loss = cfg["w_reg"] * lreg
            if recon:
                loss = loss + cfg["w_recon"] * F.mse_loss(xh, Xtr[b], reduction="none") \
                    .view(len(b), -1).sum(1).mean() / (cfg["seq_len"] * cfg["input_size"])
            if trip:
                loss = loss + cfg["w_trip"] * batch_hard_triplet(z, ytr[b])
            if cfg["w_kl"]:
                loss = loss + cfg["w_kl"] * (
                    -0.5 * (1 + logvar - mean ** 2 - logvar.exp()).sum(1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(lreg) * len(b)
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
        if not quiet and ep % 20 == 0:
            print(f"[{subset}/{variant}/L{latent}/s{seed}] ep{ep:3d} reg={tot/N:.5f} "
                  f"val={vr:.3f} best={best:.3f}", flush=True)

    model.load_state_dict(best_state)
    pred = predict(model, Xte, cap)
    arch = f"rve_{variant}_L{latent}"
    tag = f"{subset}_{arch}_s{seed}" + ("_cn" if cond_norm else "")
    res = dict(subset=subset, ablation="rve", arch=arch, seed=seed, variant=variant,
               latent=latent, rmse=rmse(pred, yte), score=score(pred, yte),
               val_rmse=best, epochs_run=len(times),
               sec_per_epoch=float(np.median(times)), n_params=nparam,
               cond_norm=cond_norm, cfg={k: v for k, v in cfg.items()})
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
    p.add_argument("--seeds", default="0")
    p.add_argument("--variants", default="full")
    p.add_argument("--latents", default="2")
    p.add_argument("--root", default="/home/lab/letuan/data/cmapss")
    p.add_argument("--outdir", default="/home/lab/letuan/runs/rul_rve")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--cond-norm", action="store_true")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    jobs = [(s, v, int(L), int(sd))
            for s in a.subsets.split(",") for v in a.variants.split(",")
            for L in a.latents.split(",") for sd in a.seeds.split(",")]
    i, n = (int(x) for x in a.shard.split("/"))
    for k, (s, v, L, sd) in enumerate(jobs):
        if k % n != i:
            continue
        tag = f"{s}_rve_{v}_L{L}_s{sd}" + ("_cn" if a.cond_norm else "")
        if os.path.exists(os.path.join(a.outdir, f"res_{tag}.json")):
            print(f"SKIP {tag}", flush=True); continue
        run(s, sd, v, L, a.root, a.outdir, a.cond_norm, a.epochs, quiet=not a.verbose)
