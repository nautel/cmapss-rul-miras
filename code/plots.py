"""Qualitative RUL figures, laid out like MathWorks' "Similarity-Based Remaining Useful
Life Estimation" example (openExample("predmaint/SimilarityBasedRULExample")) on the same
subset (FD002), with our four models in place of the similarity model; plus two figures
covering all four subsets.

  figures/data_raw.png            10 training engines, raw signals
  figures/data_regimes.png        6 operating regimes found by K-means
  figures/data_normalized.png     most trendable sensors after per-regime normalization
  figures/data_target.png         training target (piecewise-linear RUL, capped at 125)
  figures/fd002_rul_over_life.png one validation engine: predicted vs true RUL, 50/70/90% cuts
  figures/fd002_rul_estimation.png RUL estimate at 50/70/90% (density over seeds, true RUL,
                                   estimate, 90% interval), one row per method
  figures/fd002_error_hist.png    error histogram + KDE per breakpoint and method
  figures/fd002_error_box.png     error box plot per breakpoint
  figures/fd002_error_bar.png     mean error ± 1 s.d. per breakpoint
  figures/all_rul_trajectories.png predicted vs true RUL over one engine life, every subset
  figures/all_test_sorted.png     every test engine sorted by true RUL, every subset
  results/BREAKPOINT_EVAL.md      error mean / median / s.d. at 50/70/90%, every subset

Inputs: results/rul_traj/traj_*.npz (code/cluster/run_traj.sh: 5 seeds, same train/val
split for every seed) and results/rul_mfix/pred_*.npz (last-window test predictions).

Differences from the MathWorks example:
  * "Validation data" = held-out run-to-failure engines (20% of the training engines).
    They are never trained on; they are used for early stopping.
  * True RUL at a breakpoint is `life - ceil(p * life)`, as in the example. Models are
    trained on RUL capped at 125, so early-life errors on long engines are negative by
    construction; `--truth capped` scores against the capped label instead.
  * The estimate is the mean over 5 seeds; density and 90% interval are over seeds (our
    models are point predictors; the example's density comes from 50 nearest neighbours).
Prediction error = estimated − true RUL (positive = late, the dangerous side).
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
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

import data as D

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
BREAKS = [0.5, 0.7, 0.9]
# fixed categorical order, validated for CVD separation and contrast (light and dark)
METHODS = [("mf_memora_c5", "Memora (Miras)", "#0E8C7A"),
           ("mf_dcnn_c5", "DCNN", "#6E51B8"),
           ("sta_nophys", "STA-HPINN, no physics", "#AD7A0F"),
           ("mf_titans_c5", "Titans (Miras)", "#A62F52")]
ENSEMBLE = ["#0E8C7A", "#6E51B8", "#AD7A0F", "#A62F52", "#3E7CB1",
            "#7A8B3A", "#B0567E", "#5B6770", "#C27C2C", "#2F6F6A"]
INK, INK2, MUTED, GRID = "#121A19", "#3A4A47", "#6B7A77", "#E3E8E6"
TRUE_C = INK

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#BFCCC9", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})


# ------------------------------------------------------------------ helpers
def style(ax, grid="both"):
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def header(fig, title, extra=(), top=0.93, methods=True):
    """Title on top, legend under it, panels below `top`."""
    fig.suptitle(title, y=0.995, fontsize=11, fontweight="semibold", color=INK)
    h = ([plt.Line2D([], [], color=c, lw=2.2, label=l) for _, l, c in METHODS]
         if methods else []) + list(extra)
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False,
               bbox_to_anchor=(0.5, top + (0.995 - top) * 0.55), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, top))


def _load(d, prefix):
    """{(method, subset): {seed: npz-dict}} from <prefix>_<sub>_<method>_s<seed>[_cn].npz"""
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(d, f"{prefix}_*.npz")):
        m = re.match(rf"{prefix}_(FD00\d)_(.+)_s(\d+)(_cn)?\.npz$", os.path.basename(f))
        if m:
            out[(m.group(2), m.group(1))][int(m.group(3))] = dict(np.load(f))
    return out


def per_engine(runs, part="val"):
    """{unit: (cycle, true, pred[seeds, windows])}, windows ordered by cycle."""
    seeds = sorted(runs)
    ref = runs[seeds[0]]
    out = {}
    for u in np.unique(ref[f"{part}_unit"]):
        m = ref[f"{part}_unit"] == u
        o = np.argsort(ref[f"{part}_cycle"][m])
        P = np.stack([runs[s][f"{part}_pred"][runs[s][f"{part}_unit"] == u][o] for s in seeds])
        out[int(u)] = (ref[f"{part}_cycle"][m][o], ref[f"{part}_true"][m][o], P)
    return out


def at_breakpoint(cyc, true, P, frac, truth):
    """Seed estimates and true RUL once `frac` of the engine life has been observed."""
    life = int(cyc.max())
    c = int(np.ceil(frac * life))
    i = np.searchsorted(cyc, c)
    if i >= len(cyc) or cyc[i] != c:
        return None                    # breakpoint earlier than the first full window
    return P[:, i], (life - c if truth == "actual" else float(true[i]))


def breakpoint_errors(runs, truth):
    """{frac: one error per validation engine (seed-mean estimate − true)}"""
    out = {b: [] for b in BREAKS}
    for cyc, true, P in per_engine(runs).values():
        for b in BREAKS:
            r = at_breakpoint(cyc, true, P, b, truth)
            if r is not None:
                out[b].append(r[0].mean() - r[1])
    return {b: np.array(v) for b, v in out.items()}


# ------------------------------------------------------------ data figures
def data_figures(root, subset, out):
    tr = np.loadtxt(os.path.join(root, f"train_{subset}.txt"))
    pick = np.sort(np.random.default_rng(0).choice(np.unique(tr[:, 0]), 10, replace=False))
    col = D.COLS.index

    def ensemble(arr, names, labels, title, path):
        fig, axes = plt.subplots(len(names), 1, figsize=(8, 7), sharex=True)
        for ax, v, lab in zip(axes, names, labels):
            for i, u in enumerate(pick):
                m = arr[:, 0] == u
                ax.plot(arr[m, 1], arr[m, col(v)], lw=0.8, color=ENSEMBLE[i])
            ax.set_ylabel(lab)
            style(ax)
        axes[-1].set_xlabel("time (cycle)")
        fig.suptitle(title, fontsize=11, fontweight="semibold", color=INK)
        fig.tight_layout()
        save(fig, path)

    ensemble(tr, ["op1", "op2", "s1", "s2"],
             ["op_setting_1", "op_setting_2", "sensor_1", "sensor_2"],
             f"{subset} training ensemble, 10 engines — no visible degradation trend",
             os.path.join(out, "data_raw.png"))

    km = KMeans(6, n_init=5, random_state=0).fit(tr[:, 2:5])
    lab = km.labels_
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for k in range(6):
        m = lab == k
        ax.scatter(tr[m, 2], tr[m, 3], s=160, color=ENSEMBLE[k], alpha=0.55, lw=0,
                   label=f"Cluster {k + 1} ({m.sum():,} rows)")
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=45, marker="x",
               color=INK, linewidths=1.6, label="Centroids", zorder=3)
    ax.set_xlabel("op_setting_1")
    ax.set_ylabel("op_setting_2")
    ax.margins(0.08)
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left", markerscale=0.5)
    fig.suptitle("K-means finds the 6 working regimes (points in each regime nearly coincide)",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, os.path.join(out, "data_regimes.png"))

    Z = tr.copy()                                       # per-regime z-score
    for s in range(5, 26):
        for k in range(6):
            m = lab == k
            sd = tr[m, s].std()
            Z[m, s] = 0.0 if sd < 1e-6 else (tr[m, s] - tr[m, s].mean()) / sd
    ensemble(Z, ["s4", "s7", "s11", "s12"], ["sensor_4", "sensor_7", "sensor_11", "sensor_12"],
             "After per-regime normalization, degradation trends appear",
             os.path.join(out, "data_normalized.png"))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, u in enumerate(pick):
        n = int((tr[:, 0] == u).sum())
        ax.plot(np.arange(1, n + 1), D.piecewise_rul(n), lw=1.1, color=ENSEMBLE[i])
    ax.set_xlabel("time (cycle)")
    ax.set_ylabel("RUL target")
    style(ax)
    fig.suptitle("Training target: piecewise-linear RUL, capped at 125",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, os.path.join(out, "data_target.png"))


# ------------------------------------------------ evaluation figures (one subset)
def rul_over_life(tr, s, unit, path):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for k, lab, c in METHODS:
        cyc, true, P = per_engine(tr[(k, s)])[unit]
        ax.fill_between(cyc, P.min(0), P.max(0), color=c, alpha=0.15, lw=0)
        ax.plot(cyc, P.mean(0), color=c, lw=1.6, label=lab)
    life = cyc.max()
    ax.plot(cyc, life - cyc, color=MUTED, lw=1, ls=(0, (4, 3)), label="cycles to failure")
    ax.plot(cyc, true, color=TRUE_C, lw=1.8, label="true RUL (capped at 125)")
    top = ax.get_ylim()[1]
    for b in BREAKS:
        x = np.ceil(b * life)
        ax.axvline(x, color=INK2, lw=0.8, ls=":")
        ax.text(x, top * 0.97, f"{int(b*100)}%", ha="center", va="top", fontsize=8.5,
                color=INK2, backgroundcolor="white")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    style(ax)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower left")
    fig.suptitle(f"{s} validation engine {unit}: predicted RUL over its life "
                 "(line = mean of 5 seeds, band = seed range)",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, path)


def rul_estimation(tr, s, unit, truth, path):
    fig, axes = plt.subplots(len(METHODS), 3, figsize=(12, 9.5), sharex="col")
    for r, (k, lab, c) in enumerate(METHODS):
        cyc, true, P = per_engine(tr[(k, s)])[unit]
        for j, b in enumerate(BREAKS):
            ax = axes[r, j]
            est, t = at_breakpoint(cyc, true, P, b, truth)
            lo, hi = np.percentile(est, [5, 95])
            span = max(hi - lo, 4.0)
            xs = np.linspace(min(lo, t) - 1.5 * span, max(hi, t) + 1.5 * span, 300)
            dens = (gaussian_kde(est, bw_method=0.6)(xs) if np.ptp(est) > 1e-6 else
                    np.exp(-0.5 * (xs - est.mean()) ** 2))
            ax.fill_between(xs, dens, color=c, alpha=0.25, lw=0)
            ax.axvspan(lo, hi, color=c, alpha=0.10, lw=0)
            ax.axvline(t, color=TRUE_C, lw=1.8)
            ax.axvline(np.median(est), color=c, lw=1.8, ls=(0, (5, 2)))
            ax.text(0.98, 0.92, f"true {t:.0f}\nest. {np.median(est):.0f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
            ax.set_yticks([])
            style(ax, "x")
            if r == 0:
                ax.set_title(f"{int(b*100)}% of life observed")
            if j == 0:
                ax.set_ylabel(f"{lab}\nProbability Density", fontsize=9)
            if r == len(METHODS) - 1:
                ax.set_xlabel("Cycle")
    h = [plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.35, label="Probability Density Function"),
         plt.Line2D([], [], color=TRUE_C, lw=1.8, label="True RUL"),
         plt.Line2D([], [], color=MUTED, lw=1.8, ls=(0, (5, 2)), label="Estimated RUL"),
         plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.15, label="90% Confidence Interval")]
    header(fig, f"RUL Estimation — {s} validation engine {unit} (density and interval over "
                "5 seeds)", h, top=0.93, methods=False)
    save(fig, path)


def error_hist(errs, s, path):
    allv = np.concatenate([e for m in errs.values() for e in m.values()])
    bins = np.arange(np.floor(allv.min() / 5) * 5, np.ceil(allv.max() / 5) * 5 + 5, 5)
    xs = np.linspace(bins[0], bins[-1], 300)
    fig, axes = plt.subplots(3, len(METHODS), figsize=(13, 7.5), sharex=True, sharey=True)
    for j, (k, lab, c) in enumerate(METHODS):
        for i, b in enumerate(BREAKS):
            ax = axes[i, j]
            e = errs[k][b]
            ax.hist(e, bins=bins, density=True, color=c, alpha=0.45, edgecolor="white")
            ax.plot(xs, gaussian_kde(e)(xs), color=c, lw=1.6)
            ax.axvline(0, color=INK2, lw=0.8)
            style(ax, "y")
            if i == 0:
                ax.set_title(lab)
            if j == 0:
                ax.set_ylabel(f"first {int(b*100)}%\ndensity")
            if i == 2:
                ax.set_xlabel("Prediction Error")
    fig.suptitle(f"RUL Prediction Error using first 50 / 70 / 90% of each {s} validation "
                 "engine (histogram, bin width 5, with kernel density)",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, path)


def error_box(errs, s, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.18
    for mi, (k, lab, c) in enumerate(METHODS):
        pos = np.arange(3) + (mi - 1.5) * (w + 0.02)
        bp = ax.boxplot([errs[k][b] for b in BREAKS], positions=pos, widths=w,
                        patch_artist=True, medianprops=dict(color="white", lw=1.5),
                        flierprops=dict(marker="+", mec=c, ms=5),
                        whiskerprops=dict(color=c), capprops=dict(color=c))
        for bx in bp["boxes"]:
            bx.set(facecolor=c, edgecolor=c, alpha=0.85)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(3), ["50%", "70%", "90%"])
    ax.set_ylabel("Prediction Error")
    style(ax, "y")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in METHODS],
              [l for _, l, _ in METHODS], fontsize=8.5, frameon=False, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Prediction error using different percentages of each {s} validation engine",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, path)


def error_bar(errs, s, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for mi, (k, lab, c) in enumerate(METHODS):
        x = np.array([50, 70, 90]) + (mi - 1.5) * 1.5
        ax.errorbar(x, [errs[k][b].mean() for b in BREAKS],
                    yerr=[errs[k][b].std(ddof=1) for b in BREAKS], fmt="-o", color=c,
                    lw=1.6, ms=5, mec="white", capsize=3, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(40, 100)
    ax.set_xlabel("Percentage of validation data used for RUL prediction")
    ax.set_ylabel("Prediction Error")
    style(ax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", title_fontsize=8.5,
              title="Mean Prediction Error with 1 Standard Deviation Error bar")
    fig.suptitle(f"{s}: error concentrates around 0 as more of the life is observed",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, path)


# -------------------------------------------------- all-subset figures + table
def all_trajectories(tr, path):
    fig, axes = plt.subplots(4, 4, figsize=(13, 10.5), sharex="row", sharey=True)
    for r, s in enumerate(SUBSETS):
        eng = per_engine(tr[(METHODS[0][0], s)])
        lives = {u: v[0].max() for u, v in eng.items()}
        unit = sorted(lives, key=lives.get)[len(lives) // 2]        # median-life engine
        for c, (k, lab, col) in enumerate(METHODS):
            ax = axes[r, c]
            cyc, true, P = per_engine(tr[(k, s)])[unit]
            ax.plot(cyc, cyc.max() - cyc, color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.plot(cyc, true, color=TRUE_C, lw=1.6)
            ax.fill_between(cyc, P.min(0), P.max(0), color=col, alpha=0.22, lw=0)
            ax.plot(cyc, P.mean(0), color=col, lw=1.8)
            style(ax)
            if r == 0:
                ax.set_title(lab)
            if c == 0:
                ax.set_ylabel(f"{s} · engine {unit}\nRUL (cycles)")
            if r == 3:
                ax.set_xlabel("cycle")
        axes[r, 0].set_ylim(-5, 160)
    extra = [plt.Line2D([], [], color=TRUE_C, lw=1.6, label="true RUL (capped at 125)"),
             plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3)), label="cycles to failure")]
    header(fig, "Predicted RUL over the life of a held-out run-to-failure engine "
                "(median-life engine per subset; line = mean of 5 seeds, band = seed range)",
           extra, top=0.95)
    save(fig, path)


def all_test_sorted(runs, path):
    fig, axes = plt.subplots(4, 4, figsize=(13, 10.5), sharey=True)
    for r, s in enumerate(SUBSETS):
        true = runs[(METHODS[0][0], s)][0]["true"]
        o = np.argsort(true, kind="stable")
        x = np.arange(len(o))
        for c, (k, lab, col) in enumerate(METHODS):
            ax = axes[r, c]
            P = np.stack([q["pred"] for q in runs[(k, s)].values()]).mean(0)
            ax.plot(x, true[o], color=TRUE_C, lw=1.6, zorder=2)
            ax.scatter(x, P[o], s=9, color=col, alpha=0.75, lw=0, zorder=3)
            ax.text(0.03, 0.95, f"ensemble RMSE {np.sqrt(np.mean((P - true) ** 2)):.2f}",
                    transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
            style(ax)
            if r == 0:
                ax.set_title(lab)
            if c == 0:
                ax.set_ylabel(f"{s} ({len(o)} engines)\nRUL at last cycle")
            if r == 3:
                ax.set_xlabel("test engine, sorted by true RUL")
    header(fig, "Every test engine, sorted by true RUL (dot = prediction, mean of 10 seeds)",
           [plt.Line2D([], [], color=TRUE_C, lw=1.6, label="true RUL (capped at 125)")],
           top=0.95, methods=False)
    save(fig, path)


def breakpoint_table(tr, truth, path):
    lines = ["# Breakpoint evaluation (MathWorks similarity-RUL protocol)", "",
             "Held-out run-to-failure validation engines, cut at 50 / 70 / 90% of their life. "
             "Error = estimated − true RUL (cycles); estimate = mean over 5 seeds. True RUL: "
             + ("remaining cycles, as in the example (models predict RUL capped at 125, so "
                "early-life errors on long engines are negative by construction)."
                if truth == "actual" else "the capped label (125)."), ""]
    for s in SUBSETS:
        if not all((k, s) in tr for k, _, _ in METHODS):
            continue
        lines += [f"## {s}", "",
                  "| method | mean 50% | mean 70% | mean 90% | median 50% | median 70% | "
                  "median 90% | s.d. 50% | s.d. 70% | s.d. 90% | engines |",
                  "|---|" + "---|" * 10]
        for k, lab, _ in METHODS:
            e = breakpoint_errors(tr[(k, s)], truth)
            cells = ([f"{e[b].mean():.2f}" for b in BREAKS] +
                     [f"{np.median(e[b]):.2f}" for b in BREAKS] +
                     [f"{e[b].std(ddof=1):.2f}" for b in BREAKS])
            lines.append(f"| {lab} | " + " | ".join(cells) + f" | {len(e[0.5])} |")
        lines.append("")
    open(path, "w").write("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="../data/CMAPSSData")
    p.add_argument("--traj", default="../results/rul_traj")
    p.add_argument("--mfix", default="../results/rul_mfix")
    p.add_argument("--out", default="../figures")
    p.add_argument("--table", default="../results/BREAKPOINT_EVAL.md")
    p.add_argument("--subset", default="FD002", help="subset for the MathWorks-style set")
    p.add_argument("--truth", default="actual", choices=["actual", "capped"])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    o = lambda f: os.path.join(a.out, f)
    s = a.subset.lower()

    data_figures(a.root, a.subset, a.out)
    tr = _load(a.traj, "traj")
    unit = sorted(per_engine(tr[(METHODS[0][0], a.subset)]))[2]   # example: validationData{3}
    rul_over_life(tr, a.subset, unit, o(f"{s}_rul_over_life.png"))
    rul_estimation(tr, a.subset, unit, a.truth, o(f"{s}_rul_estimation.png"))
    errs = {k: breakpoint_errors(tr[(k, a.subset)], a.truth) for k, _, _ in METHODS}
    error_hist(errs, a.subset, o(f"{s}_error_hist.png"))
    error_box(errs, a.subset, o(f"{s}_error_box.png"))
    error_bar(errs, a.subset, o(f"{s}_error_bar.png"))
    all_trajectories(tr, o("all_rul_trajectories.png"))
    all_test_sorted(_load(a.mfix, "pred"), o("all_test_sorted.png"))
    breakpoint_table(tr, a.truth, a.table)
    print("wrote", sorted(os.listdir(a.out)))
