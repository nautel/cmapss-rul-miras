"""Huan luyen + danh gia SBi-Transformer tren mot subset C-MAPSS.

Sieu tham so lay nguyen tu Table 2 cua paper (cot FD001/3 va FD002/4).
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

import data as D
from model import build_model, ABLATIONS
from miras import build_miras, VARIANTS as MIRAS_VARIANTS
from baselines import build_baseline, BASELINES

# Table 2 — cot "FD0013" (FD001+FD003) va "FD0024" (FD002+FD004)
BASE = dict(val_frac=0.2,          # 10% engine (= 10 engine tren FD001) qua nho de xep hang
            batch_size=256, num_hidden=16, ffn_hidden=32, bilstm_size=32, num_layers=2,
            weight_decay=1e-5, n_heads=2, seq_len=45, input_size=17, num_samples=50,
            block_size=16, confidence_level=0.95, n_bootstrap=1000,
            narrowing_factor=0.5, patience=10, optimizer="Adam")
PER_SET = {
    "FD001": dict(lr=5e-4, encoder_layers=3, dropout=0.2, epochs=200),
    "FD003": dict(lr=5e-4, encoder_layers=3, dropout=0.2, epochs=200),
    "FD002": dict(lr=1e-4, encoder_layers=2, dropout=0.3, epochs=600),
    "FD004": dict(lr=1e-4, encoder_layers=2, dropout=0.3, epochs=600),
}


def config_for(subset, **over):
    c = dict(BASE)
    c.update(PER_SET[subset])
    c.update({k: v for k, v in over.items() if v is not None})
    return c


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))          # Eq. (15)


def score(pred, true):
    r = pred - true                                              # Eq. (16)
    return float(np.sum(np.where(r > 0, np.exp(r / 10.0) - 1.0,
                                 np.exp(-r / 13.0) - 1.0)))


@torch.no_grad()
def predict(model, X, cap, bs=4096, train_mode=False, clip=None):
    model.train(train_mode)                                      # train_mode=True => MC-dropout
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs]).float().cpu().numpy())
    model.eval()
    return np.clip(np.concatenate(out) * cap, 0.0, clip if clip else cap)


def bootstrap_ci(samples, level=0.95, narrowing=0.5):
    """Eq. (13), (14): mean/std cua ensemble Bootstrap + khoang tin cay t-Student.

    `samples` : (n_boot, N) cac du doan Bootstrap cho tung diem test.
    """
    from scipy import stats
    nb = samples.shape[0]
    mu = samples.mean(0)
    sd = samples.std(0, ddof=1)
    t = stats.t.ppf(1 - (1 - level) / 2, df=nb - 1)
    half = narrowing * t * sd / np.sqrt(nb)
    return mu, mu - half, mu + half


def uncertainty(model, X, cfg, cap, rng, n_boot=None):
    """MC-dropout (num_samples) -> resample Bootstrap (n_bootstrap) -> hai kieu CI.

    `ci_paper` : ap dung dung Eq. (14) — nua be rong = f * t * sigma/sqrt(n_bootstrap).
    `ci_pred`  : khoang DU BAO tu do tan MC-dropout — f * t * sigma_mc.

    Eq. (14) chia sigma cho sqrt(n_bootstrap) mot lan NUA sau khi sigma da la do lech
    chuan cua mot ensemble trung binh, nen khoang thu duoc hep di ~sqrt(50*1000) lan va
    do phu thuc te ve ~0. Bao cao ca hai de thay ro dieu do.
    """
    nb = n_boot or cfg["n_bootstrap"]
    S = cfg["num_samples"]
    mc = np.stack([predict(model, X, cap, train_mode=True) for _ in range(S)])   # (S, N)
    boot = mc[rng.randint(0, S, size=(nb, S))].mean(1)                          # (n_boot, N)
    ci_paper = bootstrap_ci(boot, cfg["confidence_level"], cfg["narrowing_factor"])
    from scipy import stats
    mu = boot.mean(0)
    half = cfg["narrowing_factor"] * stats.t.ppf(
        1 - (1 - cfg["confidence_level"]) / 2, df=S - 1) * mc.std(0, ddof=1)
    ci_pred = (mu, mu - half, mu + half)
    return mc, ci_paper, ci_pred


def run(subset, ablation="yes_yes_yes", seed=0, root="../data/CMAPSSData",
        outdir=".", device=None, cond_norm=False, feature_mode="paper",
        epochs=None, quiet=False, save_ckpt=False, full_epochs=False, seq_len=None,
        arch="sbi", patience=None, overrides=None):
    """`full_epochs=True`: chay het so epoch cua Table 2, khong dung som.

    Table 2 ghi DONG THOI `epochs = 200/600` va `patience = 10`; voi val RMSE nhieu thi
    patience 10 ban o epoch 36-106, khong bao gio toi 600. Hai con so khong the cung dung,
    nen chay ca hai kieu de biet phan lech den tu dau. Van lay checkpoint tot nhat theo val.
    """
    cfg = config_for(subset, epochs=epochs, seq_len=seq_len, patience=patience)
    if overrides:                      # autoresearch: ghi de bat ky khoa nao cua cfg
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    d = D.build(root, subset, seq_len=cfg["seq_len"], feature_mode=feature_mode,
                cond_norm=cond_norm, seed=seed, cap=cfg.get("rul_cap", D.RUL_CAP),
                eval_cap=D.RUL_CAP,          # thuoc do CO DINH, khong bao gio tune
                val_frac=cfg.get("val_frac", 0.1))
    cfg["input_size"] = len(d["features"])
    cap = d["cap"]                      # thang lai nhan train
    ecap = d["eval_cap"]                # kep du doan theo thuoc do

    t = lambda a: torch.as_tensor(a, device=dev)
    Xtr, ytr = t(d["Xtr"]), t(d["ytr"] / cap)
    Xva, yva = t(d["Xval"]), d["yval"]
    Xte, yte = t(d["Xte"]), d["yte"]

    if arch.startswith("bl:"):
        model = build_baseline(cfg, arch[3:])
    elif arch != "sbi":
        model = build_miras(cfg, arch)
    else:
        model = build_model(cfg, ablation)
    model = model.to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    lossf = nn.MSELoss()

    best, best_state, bad, epoch_times = np.inf, None, 0, []
    stopped_early = False
    hist = []
    N, bs = len(Xtr), cfg["batch_size"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(cfg["epochs"]):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(N, generator=g).to(dev)
        tot = 0.0
        for i in range(0, N, bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            l = lossf(model(Xtr[b]), ytr[b])
            l.backward()
            opt.step()
            tot += float(l) * len(b)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)

        vp = predict(model, Xva, cap, clip=ecap)
        vr = rmse(vp, yva)
        hist.append(dict(epoch=ep, train_loss=tot / N, val_rmse=vr))
        if vr < best - 1e-6:
            best, bad = vr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"] and not full_epochs:        # early stopping
                stopped_early = True
                break
        if not quiet and ep % 20 == 0:
            print(f"[{subset}/{ablation}/s{seed}] ep{ep:4d} loss={tot/N:.5f} "
                  f"val_rmse={vr:.3f} best={best:.3f}", flush=True)

    model.load_state_dict(best_state)
    pred = predict(model, Xte, cap, clip=ecap)
    res = dict(subset=subset, ablation=ablation, arch=arch, seed=seed,
               rmse=rmse(pred, yte), score=score(pred, yte),
               val_rmse=best, epochs_run=len(epoch_times),
               sec_per_epoch=float(np.median(epoch_times)),
               total_train_sec=float(np.sum(epoch_times)),
               n_params=nparam, n_train=int(N), n_test=int(len(yte)),
               features=d["features"], cond_norm=cond_norm,
               full_epochs=full_epochs, stopped_early=stopped_early,
               device=str(dev), cfg={k: v for k, v in cfg.items()})

    os.makedirs(outdir, exist_ok=True)
    name = ablation if arch == "sbi" else arch.replace("bl:", "bl_")
    if cfg.get("tag"):                     # autoresearch dat ten rieng cho tung cau hinh
        tag = cfg["tag"]
    else:
        tag = (f"{subset}_{name}_s{seed}"
               + ("_cn" if cond_norm else "")
               + ("_fe" if full_epochs else "")
               + (f"_L{cfg['seq_len']}" if cfg["seq_len"] != BASE["seq_len"] else ""))
    np.savez_compressed(os.path.join(outdir, f"pred_{tag}.npz"),
                        pred=pred, true=yte, unit=d["ute"])
    with open(os.path.join(outdir, f"hist_{tag}.json"), "w") as f:
        json.dump(hist, f)
    if save_ckpt:
        torch.save(best_state, os.path.join(outdir, f"ckpt_{tag}.pt"))

    # do bat dinh + quy dao — chi lam cho mo hinh day du de tiet kiem thoi gian
    if arch == "sbi" and ablation == "yes_yes_yes" and not cfg.get("skip_uncertainty"):
        rng = np.random.RandomState(seed)
        mc, (mu, lo, hi), (_, plo, phi) = uncertainty(model, Xte, cfg, cap, rng)
        res["rmse_mc"] = rmse(mu, yte)
        res["score_mc"] = score(mu, yte)
        res["ci_coverage"] = float(np.mean((yte >= lo) & (yte <= hi)))
        res["ci_width"] = float(np.mean(hi - lo))
        res["ci_pred_coverage"] = float(np.mean((yte >= plo) & (yte <= phi)))
        res["ci_pred_width"] = float(np.mean(phi - plo))
        np.savez_compressed(os.path.join(outdir, f"unc_{tag}.npz"),
                            mu=mu, lo=lo, hi=hi, plo=plo, phi=phi,
                            true=yte, unit=d["ute"])
        trajs = {}
        units = np.unique(d["ute"])
        lens = np.array([(d["raw_test"][1] == u).sum() for u in units])
        for u in units[np.argsort(-lens)[:4]]:      # 4 engine test dai nhat -> Fig. 5
            xs, ys = D.test_trajectory(d, u)
            xb = t(xs)
            _, (m2, l2, h2), (_, pl2, ph2) = uncertainty(model, xb, cfg, cap, rng, n_boot=200)
            trajs[str(int(u))] = dict(true=ys.tolist(), pred=predict(model, xb, cap).tolist(),
                                      mu=m2.tolist(), lo=l2.tolist(), hi=h2.tolist(),
                                      plo=pl2.tolist(), phi=ph2.tolist())
        with open(os.path.join(outdir, f"traj_{tag}.json"), "w") as f:
            json.dump(trajs, f)

    with open(os.path.join(outdir, f"res_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"RESULT {tag}: RMSE={res['rmse']:.2f} Score={res['score']:.2f} "
          f"({res['epochs_run']} ep, {res['sec_per_epoch']:.2f}s/ep, {nparam} params)",
          flush=True)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="FD001")
    p.add_argument("--ablation", default="yes_yes_yes", choices=list(ABLATIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="../data/CMAPSSData")
    p.add_argument("--outdir", default=os.environ.get("OUTDIR", "./out"))
    p.add_argument("--device", default=None)
    p.add_argument("--cond-norm", action="store_true")
    p.add_argument("--feature-mode", default="paper")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--save-ckpt", action="store_true")
    p.add_argument("--full-epochs", action="store_true")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--arch", default="sbi",
                   choices=["sbi"] + MIRAS_VARIANTS + [f"bl:{b}" for b in BASELINES])
    p.add_argument("--patience", type=int, default=None)
    a = p.parse_args()
    run(a.subset, a.ablation, a.seed, a.root, a.outdir, a.device,
        a.cond_norm, a.feature_mode, a.epochs, save_ckpt=a.save_ckpt,
        full_epochs=a.full_epochs, seq_len=a.seq_len, arch=a.arch,
        patience=a.patience)
