# Statistical tests

Reference: `bl_dcnn`. 95% bootstrap CI, 1500 resamples **over engines** (the same engine set resampled for both sides — paired), averaged over seeds. `p` Holm-Bonferroni-adjusted within each subset.

## FD001

| method | RMSE [95% CI] | diff vs. reference [CI] | p (boot) | p (Holm) | p Wilcoxon | seed |
|---|---|---|---|---|---|---|
| **`bl_dcnn`** (ref.) | 13.12 [11.01, 15.16] | — | — | — | — | 10 |
| `bl_gru` | 14.08 [11.90, 16.15] | +0.96 [-0.09, +1.99] | 0.0813 | 0.1320 | 1.29e-04 | 10 |
| `bl_mlp` | 14.43 [12.24, 16.52] | **+1.31** [+0.36, +2.24] | 0.0027 | 0.0107 | 5.76e-12 | 10 |
| `bl_bilstm` | 14.53 [12.30, 16.57] | +1.41 [+0.07, +2.77] | 0.0440 | 0.1320 | 2.16e-05 | 10 |
| `bl_tcn` | 14.67 [12.52, 16.60] | +1.54 [-0.02, +3.12] | 0.0533 | 0.1320 | 1.93e-08 | 10 |
| `bl_lstm` | 15.10 [12.83, 17.28] | **+1.98** [+0.82, +3.12] | 0.0000 | 0.0000 | 1.78e-09 | 10 |
| `bl_cnn_lstm` | 15.53 [13.38, 17.66] | **+2.41** [+0.81, +3.96] | 0.0013 | 0.0067 | 2.33e-09 | 10 |

**Negative** difference = better than reference. Bold = significant after Holm correction (p < 0.05).

The `p Wilcoxon` column is for reference only — it treats n_engine x n_seed as independent observations, so it comes out tiny for every pair; use the paired bootstrap column for conclusions.

## FD002

| method | RMSE [95% CI] | diff vs. reference [CI] | p (boot) | p (Holm) | p Wilcoxon | seed |
|---|---|---|---|---|---|---|
| **`bl_dcnn`** (ref.) | 12.71 [11.38, 14.01] | — | — | — | — | 10 |
| `bl_tcn` | 13.44 [12.27, 14.61] | +0.72 [-0.02, +1.47] | 0.0680 | 0.1333 | 6.34e-16 | 10 |
| `bl_gru` | 13.52 [12.38, 14.63] | +0.80 [+0.08, +1.58] | 0.0333 | 0.1333 | 4.30e-20 | 10 |
| `bl_lstm` | 13.58 [12.35, 14.80] | +0.86 [+0.04, +1.70] | 0.0373 | 0.1333 | 9.60e-17 | 10 |
| `bl_bilstm` | 13.62 [12.47, 14.73] | +0.90 [+0.06, +1.73] | 0.0360 | 0.1333 | 2.95e-22 | 10 |
| `bl_cnn_lstm` | 14.47 [13.23, 15.72] | **+1.75** [+0.93, +2.63] | 0.0000 | 0.0000 | 6.05e-21 | 10 |
| `bl_mlp` | 14.77 [13.53, 16.00] | **+2.05** [+1.30, +2.77] | 0.0000 | 0.0000 | 8.15e-59 | 10 |

**Negative** difference = better than reference. Bold = significant after Holm correction (p < 0.05).

The `p Wilcoxon` column is for reference only — it treats n_engine x n_seed as independent observations, so it comes out tiny for every pair; use the paired bootstrap column for conclusions.

## FD003

| method | RMSE [95% CI] | diff vs. reference [CI] | p (boot) | p (Holm) | p Wilcoxon | seed |
|---|---|---|---|---|---|---|
| **`bl_dcnn`** (ref.) | 12.01 [9.89, 14.00] | — | — | — | — | 10 |
| `bl_gru` | 12.80 [10.70, 14.91] | +0.78 [-0.27, +1.90] | 0.1453 | 0.1493 | 4.60e-04 | 10 |
| `bl_mlp` | 12.86 [11.15, 14.48] | +0.84 [-0.09, +1.78] | 0.0747 | 0.1493 | 2.12e-10 | 10 |
| `bl_tcn` | 14.86 [12.71, 16.84] | **+2.85** [+0.95, +4.87] | 0.0013 | 0.0067 | 9.91e-13 | 10 |
| `bl_bilstm` | 14.87 [12.51, 17.05] | **+2.85** [+0.61, +5.08] | 0.0120 | 0.0480 | 3.19e-11 | 10 |
| `bl_lstm` | 14.90 [12.28, 17.55] | +2.88 [+0.53, +5.64] | 0.0187 | 0.0560 | 6.23e-07 | 10 |
| `bl_cnn_lstm` | 15.99 [13.67, 18.22] | **+3.97** [+2.29, +5.76] | 0.0000 | 0.0000 | 1.04e-22 | 10 |

**Negative** difference = better than reference. Bold = significant after Holm correction (p < 0.05).

The `p Wilcoxon` column is for reference only — it treats n_engine x n_seed as independent observations, so it comes out tiny for every pair; use the paired bootstrap column for conclusions.

## FD004

| method | RMSE [95% CI] | diff vs. reference [CI] | p (boot) | p (Holm) | p Wilcoxon | seed |
|---|---|---|---|---|---|---|
| **`bl_dcnn`** (ref.) | 13.60 [11.72, 15.44] | — | — | — | — | 10 |
| `bl_gru` | 14.15 [12.55, 15.76] | +0.56 [-0.20, +1.36] | 0.1547 | 0.1547 | 5.42e-28 | 10 |
| `bl_bilstm` | 14.62 [13.11, 16.13] | **+1.02** [+0.23, +1.84] | 0.0107 | 0.0213 | 3.44e-38 | 10 |
| `bl_lstm` | 14.95 [13.37, 16.45] | **+1.35** [+0.46, +2.26] | 0.0027 | 0.0080 | 3.23e-46 | 10 |
| `bl_mlp` | 15.27 [13.67, 16.86] | **+1.67** [+0.92, +2.48] | 0.0000 | 0.0000 | 5.16e-54 | 10 |
| `bl_tcn` | 15.54 [13.86, 17.20] | **+1.95** [+0.94, +3.01] | 0.0000 | 0.0000 | 2.65e-33 | 10 |
| `bl_cnn_lstm` | 15.61 [14.00, 17.25] | **+2.02** [+0.91, +3.12] | 0.0000 | 0.0000 | 2.25e-31 | 10 |

**Negative** difference = better than reference. Bold = significant after Holm correction (p < 0.05).

The `p Wilcoxon` column is for reference only — it treats n_engine x n_seed as independent observations, so it comes out tiny for every pair; use the paired bootstrap column for conclusions.
