"""Figures for the README: Miras vs. references on C-MAPSS (shared pipeline).

Reads prediction files from results/rul_mfix and writes PNGs to figures/.
  fig1_forest.png   RMSE with 95% CI (bootstrap over test engines), one panel per subset
  fig2_chunk.png    effect of the chunk approximation (chunk 1 = exact recurrence)
  fig3_ideas.png    RMSE change of each borrowed idea relative to plain Titans

Palette: fixed categorical order teal / purple / amber / crimson, validated for CVD
separation and contrast on light and dark surfaces. Diverging map: teal (better) —
neutral gray — crimson (worse).
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
TEAL, PURPLE, AMBER, CRIMSON = "#0E8C7A", "#6E51B8", "#AD7A0F", "#A62F52"
INK, INK2, MUTED, GRID = "#121A19", "#3A4A47", "#6B7A77", "#E3E8E6"
PUBLISHED_STA = {"FD001": 11.27, "FD002": 13.21, "FD003": 8.30, "FD004": 13.31}

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#BFCCC9", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})


def load(d):
    """{(method, subset): {seed: (pred, true)}} from pred_<sub>_<method>_s<seed>[_cn].npz"""
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(d, "pred_*.npz")):
        m = re.match(r"pred_(FD00\d)_(.+)_s(\d+)(_cn)?\.npz$", os.path.basename(f))
        if not m:
            continue
        z = np.load(f)
        out[(m.group(2), m.group(1))][int(m.group(3))] = (z["pred"], z["true"])
    return out


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def ci(runs, n_boot=2000, seed=0):
    """Mean RMSE over seeds, with a 95% CI from resampling test engines."""
    rng = np.random.default_rng(seed)
    seeds = sorted(runs)
    n = len(runs[seeds[0]][1])
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([np.mean([rmse(runs[s][0][i], runs[s][1][i]) for s in seeds])
                      for i in idx])
    point = np.mean([rmse(*runs[s]) for s in seeds])
    return point, *np.quantile(boots, [0.025, 0.975])


def style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------
def fig_forest(data, out):
    methods = [  # (key in file names, label, colour)
        ("mf_dcnn_c5", "DCNN (2018)", PURPLE),
        ("sta_nophys", "STA-HPINN, no physics", AMBER),
        ("mf_memora_c5", "Memora", TEAL),
        ("mf_titans_c5", "Titans", TEAL),
        ("mf_titans_R4_c5", "Titans, rank-4 memory", TEAL),
        ("mf_deltanet_c5", "DeltaNet", TEAL),
        ("mf_gated_deltanet_c5", "Gated DeltaNet", TEAL),
        ("mf_linear_attn_c5", "Linear Attention", TEAL),
        ("mf_mamba2_c5", "Mamba2", TEAL),
        ("mf_moneta_c5", "Moneta", TEAL),
        ("mf_yaad_c5", "Yaad", TEAL),
    ]
    methods = [m for m in methods if all(data.get((m[0], s)) for s in SUBSETS)]
    stats = {(k, s): ci(data[(k, s)]) for k, _, _ in methods for s in SUBSETS}
    order = sorted(methods, key=lambda m: np.mean([stats[(m[0], s)][0] for s in SUBSETS]))
    order = order[::-1]                                     # best at the top

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.6), sharey=True)
    y = np.arange(len(order))
    for ax, s in zip(axes, SUBSETS):
        for yi, (k, lab, col) in zip(y, order):
            p, lo, hi = stats[(k, s)]
            ax.plot([lo, hi], [yi, yi], color=col, lw=2, alpha=0.45, solid_capstyle="round")
            ax.plot(p, yi, "o", ms=6.5, color=col, mec="white", mew=1.5, zorder=3)
        ax.axvline(PUBLISHED_STA[s], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_title(s)
        ax.set_xlabel("RMSE")
        style(ax)
    axes[0].set_yticks(y, [m[1] for m in order])
    handles = [plt.Line2D([], [], marker="o", ls="", ms=6.5, color=c, label=l) for l, c in
               [("Miras family", TEAL), ("literature baseline", PURPLE),
                ("re-implemented SOTA", AMBER)]]
    handles.append(plt.Line2D([], [], color=MUTED, ls=(0, (4, 3)),
                              label="STA-HPINN published (not reproduced)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle("RMSE with 95% CI (bootstrap over test engines), 10 seeds each",
                 y=1.09, fontsize=11.5, fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_chunk(data, out):
    variants = [("linear_attn", "Linear Attention", TEAL), ("deltanet", "DeltaNet", PURPLE),
                ("gated_deltanet", "Gated DeltaNet", AMBER), ("titans", "Titans", CRIMSON)]
    chunks = [1, 5, 20]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    x = np.arange(len(chunks))
    for vi, (v, lab, col) in enumerate(variants):
        means, sds = [], []
        for c in chunks:
            per_seed = []
            for sd in range(5):
                r = [rmse(*data[(f"mf_{v}_c{c}", s)][sd]) for s in ("FD001", "FD004")
                     if sd in data.get((f"mf_{v}_c{c}", s), {})]
                if len(r) == 2:
                    per_seed.append(np.mean(r))
            means.append(np.mean(per_seed))
            sds.append(np.std(per_seed, ddof=1))
        off = (vi - 1.5) * 0.06
        ax.errorbar(x + off, means, yerr=sds, color=col, lw=2, marker="o", ms=6.5,
                    mec="white", mew=1.5, capsize=0, elinewidth=1.2, label=lab)
        ax.annotate(lab, (x[-1] + off, means[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, ["1 (exact)", "5", "20"])
    ax.set_xlabel("chunk size")
    ax.set_ylabel("RMSE, mean of FD001 and FD004")
    ax.set_xlim(-0.3, len(chunks) - 1 + 0.75)
    style(ax, "y")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    ax.set_title("Chunk size: differences stay within seed noise (bars: ±1 s.d., 5 seeds)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_ideas(data, out):
    base = "mf_titans_c5"
    ideas = [("mf_titans_R4_c5", "rank-4 memory"),
             ("mf_titans_B3trip_c5", "3-d bottleneck + triplet"),
             ("mf_titans_S_c5", "sensor-axis branch"),
             ("mf_titans+bilstm_c5", "BiLSTM head")]
    ideas = [i for i in ideas if all(data.get((i[0], s)) for s in SUBSETS)]
    cols = SUBSETS + ["mean"]
    M = np.zeros((len(ideas), len(cols)))
    for r, (k, _) in enumerate(ideas):
        for c, s in enumerate(SUBSETS):
            a = np.mean([rmse(*v) for v in data[(k, s)].values()])
            b = np.mean([rmse(*v) for v in data[(base, s)].values()])
            M[r, c] = a - b
        M[r, -1] = M[r, :4].mean()
    M = M + 0.0                                   # no negative zero in labels
    lim = 3.0                                     # cells beyond ±3 saturate
    cmap = LinearSegmentedColormap.from_list("div", [TEAL, "#F2F4F3", CRIMSON])
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(ideas)), [i[1] for i in ideas])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(ideas)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            v = M[r, c]
            txt = "0.00" if abs(v) < 0.005 else f"{v:+.2f}"
            ax.text(c, r, txt, ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > lim * 0.6 else INK,
                    fontweight="semibold" if c == len(cols) - 1 else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Δ RMSE (clipped ±3)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Ideas borrowed from TSHAE / STA-HPINN, applied to Titans "
                 "(negative = better)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="../results/rul_mfix")
    p.add_argument("--out", default="../figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load(a.dir)
    fig_forest(data, os.path.join(a.out, "fig1_forest.png"))
    fig_chunk(data, os.path.join(a.out, "fig2_chunk.png"))
    fig_ideas(data, os.path.join(a.out, "fig3_ideas.png"))
    print("wrote", sorted(os.listdir(a.out)))
