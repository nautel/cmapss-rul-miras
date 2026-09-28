"""Huan luyen STA-HPINN theo dung muc 3.1-3.2 cua arXiv:2405.12377.

Khac voi train.py: loss co them thanh phan vat ly, va model can them input `t`
(so chu ky), nen co vong huan luyen rieng. Ket qua ghi ra cung dinh dang
`res_*.json` / `pred_*.npz` de dung chung cong cu thong ke.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import data as D
from sta_hpinn import build_sta, ReLoBRaLo
from train import rmse, score

# muc 3.1-3.2: cua so 40 cho FD001, 60 cho cac bo con lai; min-max; 14 sensor kinh dien;
# 20% train lam val; batch 512; lr 1e-3 cho 50 epoch dau roi 1e-4; trung binh 10 lan chay
CFG = dict(feature_mode="classic14", norm="minmax", cap=125.0, val_frac=0.2,
           batch_size=512, lr1=1e-3, lr2=1e-4, lr_switch=50, epochs=250, patience=30,
           sta_d=32, sta_hidden=3, sta_neurons=10, sta_layers=3, n_heads=1, dropout=0.1)
SEQ = {"FD001": 40, "FD002": 60, "FD003": 60, "FD004": 60}


def evaluate(model, X, t, cap, bs=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs], t[i:i + bs]).float().cpu().numpy())
    return np.clip(np.concatenate(out) * cap, 0.0, cap)


def run(subset, seed=0, root="../data/CMAPSSData", outdir="./out", device=None,
        physics=True, cond_norm=False, epochs=None, quiet=True, tag=None):
    cfg = dict(CFG)
    cfg["seq_len"] = SEQ[subset]
    if epochs:
        cfg["epochs"] = epochs
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    d = D.build(root, subset, seq_len=cfg["seq_len"], feature_mode=cfg["feature_mode"],
                cond_norm=cond_norm, seed=seed, cap=cfg["cap"], eval_cap=D.RUL_CAP,
                val_frac=cfg["val_frac"], norm=cfg["norm"])
    cfg["input_size"] = len(d["features"])
    cap = cfg["cap"]
    T = lambda a: torch.as_tensor(a, device=dev)
    Xtr, ytr, ttr = T(d["Xtr"]), T(d["ytr"] / cap), T(d["ttr"])
    Xva, yva, tva = T(d["Xval"]), d["yval"], T(d["tval"])
    Xte, yte, tte = T(d["Xte"]), d["yte"], T(d["tte"])

    model = build_sta(cfg).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr1"])
    bal = ReLoBRaLo()

    N, bs = len(Xtr), cfg["batch_size"]
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, times = np.inf, None, 0, []
    for ep in range(cfg["epochs"]):
        if ep == cfg["lr_switch"]:
            for pg in opt.param_groups:
                pg["lr"] = cfg["lr2"]
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(N, generator=g).to(dev)
        tot = 0.0
        for i in range(0, N - 1, bs):                 # bo batch le kich thuoc 1 (BatchNorm)
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            if physics:
                u, f = model(Xtr[b], ttr[b], physics=True)
                ld = ((u - ytr[b]) ** 2).mean()
                lf = (f ** 2).mean()
                lam = bal([ld, lf])
                loss = lam[0] * ld + lam[1] * lf
            else:
                u = model(Xtr[b], ttr[b])
                loss = ld = ((u - ytr[b]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(ld) * len(b)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        vr = rmse(evaluate(model, Xva, tva, cap), yva)
        if vr < best - 1e-6:
            best, bad = vr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        if not quiet and ep % 20 == 0:
            print(f"[{subset}/s{seed}] ep{ep:3d} data={tot/N:.5f} val={vr:.3f} "
                  f"best={best:.3f}", flush=True)

    model.load_state_dict(best_state)
    pred = evaluate(model, Xte, tte, cap)
    tag = tag or (f"{subset}_sta{'' if physics else '_nophys'}_s{seed}"
                  + ("_cn" if cond_norm else ""))
    res = dict(subset=subset, ablation="sta", arch="sta" if physics else "sta_nophys",
               seed=seed, rmse=rmse(pred, yte), score=score(pred, yte), val_rmse=best,
               epochs_run=len(times), sec_per_epoch=float(np.median(times)),
               n_params=nparam, cond_norm=cond_norm, physics=physics,
               cfg={k: v for k, v in cfg.items()})
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, f"pred_{tag}.npz"),
                        pred=pred, true=yte, unit=d["ute"])
    json.dump(res, open(os.path.join(outdir, f"res_{tag}.json"), "w"), indent=1)
    print(f"RESULT {tag}: RMSE={res['rmse']:.2f} Score={res['score']:.2f} "
          f"({res['epochs_run']} ep, {res['sec_per_epoch']:.2f}s/ep, {nparam} params)",
          flush=True)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    p.add_argument("--seeds", default="0")
    p.add_argument("--root", default="/home/lab/letuan/data/cmapss")
    p.add_argument("--outdir", default="/home/lab/letuan/runs/rul_sta")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--no-physics", action="store_true")
    p.add_argument("--cond-norm", action="store_true")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    jobs = [(s, int(sd)) for s in a.subsets.split(",") for sd in a.seeds.split(",")]
    i, n = (int(x) for x in a.shard.split("/"))
    for k, (s, sd) in enumerate(jobs):
        if k % n != i:
            continue
        tag = (f"{s}_sta{'' if not a.no_physics else '_nophys'}_s{sd}"
               + ("_cn" if a.cond_norm else ""))
        if os.path.exists(os.path.join(a.outdir, f"res_{tag}.json")):
            print(f"SKIP {tag}", flush=True)
            continue
        run(s, sd, a.root, a.outdir, None, not a.no_physics, a.cond_norm,
            a.epochs, quiet=not a.verbose)
