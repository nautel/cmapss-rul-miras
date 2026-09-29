# Summary — reproducing two papers, re-implementing SOTA, and applying Miras to RUL

All runs on a shared Tesla V100 cluster. Detailed reports:
[`results/REPRODUCTION.md`](results/REPRODUCTION.md) ·
[`results/MIRAS.md`](results/MIRAS.md) · [`results/AUTORESEARCH2.md`](results/AUTORESEARCH2.md) ·
[`results/ALL_METHODS.md`](results/ALL_METHODS.md) · [`results/STATS_BASELINES.md`](results/STATS_BASELINES.md) ·
[`results/MIRAS_FIX.md`](results/MIRAS_FIX.md)

---

## 0. Update 09-2026 — Miras after bug fixes

Details: [`results/MIRAS_FIX.md`](results/MIRAS_FIX.md) (640 runs, shared pipeline, 10 seeds).
Three bugs fixed in `code/miras.py` (Memora sign error, Titans losing momentum across
chunks, causal mask on the sensor branch) — the Miras numbers in section 1 below predate
the fix.

| | mean RMSE |
|---|---|
| DCNN (reference) | **12.50** |
| STA-HPINN without physics loss | 12.95 |
| Best Miras — Memora | 13.20 (not significantly different from DCNN on FD001–FD003) |
| Titans / Titans rank-4 memory | 13.77 |
| 5 other Miras variants | 13.98 – 14.75 |

The chunk approximation is not the cause (chunk 1 / 5 / 20 differ by ≤ 0.6). Bottleneck +
triplet loss hurts; the sensor-axis branch is good on FD001/FD003 but collapses on
FD002/FD004.

---

## 1. Leaderboard (RMSE, 95% CI by bootstrap over engines)

| method | source | FD001 | FD002 | FD003 | FD004 | mean | seeds |
|---|---|---|---|---|---|---|---|
| STA-HPINN, **without** physics loss | SOTA 2024, modified | 12.22 | 13.72 | 10.29 | 14.05 | **12.57** | 5 |
| DCNN | Li 2018 | 13.12 | 12.71 | 12.01 | 13.60 | **12.86** | 10 |
| Autoresearch (found config) | this study | 14.14 | 12.63 | 12.63 | 13.70 | **13.28** | 5 |
| GRU | literature | 14.08 | 13.52 | 12.80 | 14.15 | 13.64 | 10 |
| Transformer + sparse attn | paper 1, ablation | 13.59 | 12.96 | 14.11 | 14.53 | 13.80 | 10 |
| STA-HPINN (full) | SOTA 2024 | 13.40 | 14.56 | 11.69 | 15.91 | 13.89 | 10 |
| SBi-Transformer | paper 1 | 14.64 | 13.06 | 14.51 | 14.62 | 14.21 | 10 |
| Flattened-window MLP (lower bound) | — | 14.43 | 14.77 | 12.86 | 15.27 | 14.33 | 10 |
| 10 Miras variants (pre-fix) | paper 2 | — | — | — | — | 14.87–15.50 | 3 |
| *SBi-Transformer — **published*** | paper 1 | *11.37* | *12.05* | *11.13* | *11.18* | *11.43* | — |
| *STA-HPINN — **published*** | SOTA 2024 | *11.27* | *13.21* | *8.30* | *13.31* | *11.52* | — |

All 29 methods fall within a 3.5-RMSE band, while the CI of **a single cell** is already
**±2.4** wide.

## 2. Four findings

**a. Paper 1 is not reproducible** — the published numbers lie OUTSIDE the 95% CI on all four
subsets. Three hypothesised causes were tested experimentally: early stopping is *not* it
(600 epochs is no better than 80); window 30 (per Table 1) is *worse* than 45 (per Table 2);
missing operating-condition normalization is real but only rescues FD002 (−24%) and
FD004 (−22%).

**b. Paper 1's ablation reverses** — the paper reports a monotone decrease 18.48 → 11.43;
re-running all five configurations to full epochs puts them within 14.94–15.65, with the
**full model worst**. The independent automated search also selected `yes_yes_no`, i.e.
**without the BiLSTM**.

**c. The SOTA's physics loss hurts** — removing STA-HPINN's physics branch (the contribution
the paper is named after) gives **1.32 RMSE better** results and trains twice as fast. The
full model is even significantly worse than DCNN (2018) on FD002 (+1.85, p = 0.002).

**d. Preprocessing matters 3.4× more than architecture** — isolated ablation of the best
configuration, reverting one factor at a time to the paper's value:

| factor reverted to paper value | RMSE degradation |
|---|---|
| operating-condition normalization | **+1.37** |
| architecture (drop BiLSTM) | +0.40 |
| window 60 → 45 | +0.39 |
| weight decay | +0.20 |
| FFN width / hidden width | +0.09 |
| dropout / number of encoder layers | −0.02 / −0.03 |

## 3. Three methodological pitfalls (all made by this workflow itself)

1. **The metric leaked into the search space.** `rul_cap` also changes the TEST labels: at a
   cap of 110, 28–85 engines per subset get lowered labels — exactly the hardest,
   high-RUL engines. The search exploited it at once: it reported 10.63, which measures
   13.70 under the standard metric. After separating the training cap from the evaluation
   cap, the search **re-selected** 125 on its own.
2. **Wilcoxon over per-window errors** gives p = 1e-04 to 1e-59 for EVERY pair, including
   pairs the bootstrap cannot distinguish — because it treats 100 engines × 10 seeds as
   independent. The correct sampling unit is the ENGINE.
3. **Concluding from a partially finished grid.** Twice a conclusion was drawn at 48/96 runs;
   both were wrong once the grid completed.

## 4. What is still missing for publication

The reproduction of paper 1 is close to sufficient for a *reproducibility study* — **four
internal inconsistencies** can be read off the paper's own tables without trusting any
experiment: Table 1 contradicts Table 2 (three different window lengths), `epochs=600`
contradicts `patience=10` (measured: stopping at epoch 22–24), Eq. (14) yields CIs with ~0
coverage, and Table 6 contradicts the data sizes (FD002 with 48,819 samples runs *faster*
than FD001 with 17,731).

Still missing: contacting the authors before publishing; extending to N-CMAPSS or real
data; more seeds for cells that currently have 3–5. As for a "new method": after fixing the
metric, the found configuration only matches the reproduction and loses to DCNN (2018).
