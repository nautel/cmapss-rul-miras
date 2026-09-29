"""Redraw Fig. 5 (RUL trajectory + CI), Fig. 6 (raincloud), Fig. 7 (heatmap), Fig. 8 (time)."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_values import ABL_RMSE, ABL_SCORE, ABL_LABEL, SUBSETS, TABLE6


def collect(outdir):
    agg = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(outdir, "res_*.json"))):
        r = json.load(open(f))
        if r.get("cond_norm"):
            continue
        agg[(r["subset"], r["ablation"])].append(r)
    return agg


def fig5(outdir, seed=0):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, s in zip(axes.ravel(), SUBSETS):
        p = os.path.join(outdir, f"traj_{s}_yes_yes_yes_s{seed}.json")
        if not os.path.exists(p):
            ax.set_visible(False)
            continue
        tj = json.load(open(p))
        u = sorted(tj, key=lambda k: -len(tj[k]["true"]))[0]
        t = tj[u]
        x = np.arange(len(t["true"]))
        ax.plot(x, t["true"], "k-", lw=1.8, label="True RUL")
        ax.plot(x, t["mu"], "-", color="tab:red", lw=1.5, label="Prediction (MC-dropout mean)")
        if "plo" in t:
            ax.fill_between(x, t["plo"], t["phi"], color="tab:red", alpha=0.20,
                            label="Prediction CI (t x sigma_MC x 0.5)")
        ax.fill_between(x, t["lo"], t["hi"], color="tab:blue", alpha=0.9,
                        label="CI per Eq. (14)")
        ax.set_title(f"{s} — test engine #{u}")
        ax.set_xlabel("Flight cycle"); ax.set_ylabel("RUL")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Fig. 5 (reproduced) — predicted RUL trajectories on test engines")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_rul_curves.png"), dpi=150)
    plt.close(fig)


def _raincloud(ax, groups, labels, colors):
    for i, (g, c) in enumerate(zip(groups, colors)):
        pos = i + 1
        v = ax.violinplot([g], positions=[pos + 0.15], widths=0.5,
                          showextrema=False, showmedians=False)
        for b in v["bodies"]:
            m = b.get_paths()[0].vertices[:, 0].mean()
            b.get_paths()[0].vertices[:, 0] = np.clip(
                b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_facecolor(c); b.set_alpha(0.45)
        ax.boxplot([g], positions=[pos], widths=0.12, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor="white", color=c),
                   medianprops=dict(color=c))
        j = np.random.RandomState(0).normal(0, 0.035, len(g))
        ax.scatter(pos - 0.22 + j, g, s=5, alpha=0.35, color=c)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)


def fig6(outdir, seed=0):
    show = ["no_no_yes", "yes_no_no", "yes_yes_no", "yes_yes_yes"]
    lab = ["BiLSTM", "Transformer", "+Sparse", "SBi-Transformer"]
    colors = ["tab:gray", "tab:blue", "tab:green", "tab:red"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, s in zip(axes.ravel(), SUBSETS):
        gs, ls, cs = [], [], []
        for ab, l, c in zip(show, lab, colors):
            p = os.path.join(outdir, f"pred_{s}_{ab}_s{seed}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            gs.append(z["pred"] - z["true"]); ls.append(l); cs.append(c)
        if not gs:
            ax.set_visible(False); continue
        _raincloud(ax, gs, ls, cs)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{s}"); ax.set_ylabel("Prediction error (pred − true)")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Fig. 6 (reproduced) — distribution of prediction errors on the test set")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig6_raincloud.png"), dpi=150)
    plt.close(fig)


def fig7(outdir):
    agg = collect(outdir)
    abls = list(ABL_LABEL)
    for key, paper, name, fmt in [("rmse", ABL_RMSE, "RMSE", "{:.2f}"),
                                  ("score", ABL_SCORE, "Score", "{:.0f}")]:
        M = np.full((len(abls), len(SUBSETS)), np.nan)
        P = np.array([paper[a] for a in abls], float)
        for i, a in enumerate(abls):
            for j, s in enumerate(SUBSETS):
                rs = agg.get((s, a), [])
                if rs:
                    M[i, j] = np.mean([r[key] for r in rs])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, A, t in [(axes[0], P, "Paper"), (axes[1], M, "Our reproduction")]:
            im = ax.imshow(A, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(SUBSETS))); ax.set_xticklabels(SUBSETS)
            ax.set_yticks(range(len(abls)))
            ax.set_yticklabels([ABL_LABEL[a] for a in abls], fontsize=8)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if not np.isnan(A[i, j]):
                        ax.text(j, i, fmt.format(A[i, j]), ha="center", va="center",
                                color="w", fontsize=8)
            ax.set_title(f"{t} — {name}")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"Fig. 7 (reproduced) — ablation heatmap, {name}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fig7_heatmap_{key}.png"), dpi=150)
        plt.close(fig)


def fig8(outdir):
    agg = collect(outdir)
    ours = [np.mean([r["sec_per_epoch"] for r in agg.get((s, "yes_yes_yes"), [])] or [np.nan])
            for s in SUBSETS]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(SUBSETS)); w = 0.2
    for i, (k, v) in enumerate(TABLE6.items()):
        ax.bar(x + (i - 1.5) * w, v, w, label=f"{k} (paper)")
    ax.bar(x + 1.5 * w, ours, w, label="SBi-Transformer (reproduced, V100)", color="tab:red")
    for i, v in enumerate(ours):
        if not np.isnan(v):
            ax.text(x[i] + 1.5 * w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(SUBSETS)
    ax.set_ylabel("Seconds / epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Fig. 8 (reproduced) — training time per epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig8_time.png"), dpi=150)
    plt.close(fig)


def fig_loss(outdir, seed=0):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, s in zip(axes.ravel(), SUBSETS):
        p = os.path.join(outdir, f"hist_{s}_yes_yes_yes_s{seed}.json")
        if not os.path.exists(p):
            ax.set_visible(False); continue
        h = json.load(open(p))
        ep = [x["epoch"] for x in h]
        ax.plot(ep, [x["train_loss"] for x in h], label="train MSE (normalized)")
        ax2 = ax.twinx()
        ax2.plot(ep, [x["val_rmse"] for x in h], color="tab:orange", label="val RMSE")
        ax.set_title(s); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8); ax2.legend(loc="center right", fontsize=8)
    fig.suptitle("Learning curves (equivalent to the paper's Fig. 8, left)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_loss_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="./out")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    fig5(a.outdir, a.seed); fig6(a.outdir, a.seed); fig7(a.outdir)
    fig8(a.outdir); fig_loss(a.outdir, a.seed)
    print("Figures saved:", sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.outdir, "fig*.png"))))
