# Trying the Miras family on C-MAPSS — and which direction suits RUL

Paper 2: Behrouz, Razaviyayn, Zhong, Mirrokni, *It's All Connected* (arXiv:2504.13173, Google Research 2025).

Every architecture uses the SAME backbone (linear projection + learned position embedding -> N blocks -> FC on the last step) and the SAME Table 2 hyperparameters of paper 1; only the sequence-mixing block differs. Each cell = mean over 3 seeds, using the best preprocessing config (z-score for FD001/FD003, operating-condition normalization for FD002/FD004).

## Leaderboard (RMSE, lower = better)

| # | Model | attentional bias | retention gate | FD001 | FD002 | FD003 | FD004 | mean | params |
|---|---|---|---|---|---|---|---|---|---|
| 1 | SBi-Transformer (paper 1) | — | — | 14.26 ±0.29 | 13.04 ±0.18 | 13.80 ±0.44 | 14.74 ±0.55 | **13.96** | 45649 |
| 2 | titans+bilstm + BiLSTM head | l2 + momentum | data-dependent alpha_t | 14.92 ±0.14 | 13.07 ±0.30 | 14.98 ±0.61 | 14.44 ±0.52 | **14.35** | 44503 |
| 3 | gated_deltanet+bilstm + BiLSTM head | l2 (delta rule) | data-dependent alpha_t | 15.13 ±0.31 | 13.84 ±0.28 | 14.46 ±1.11 | 15.07 ±0.32 | **14.63** | 44469 |
| 4 | titans | l2 + momentum | data-dependent alpha_t | 14.70 ±0.37 | 14.51 ±1.39 | 14.58 ±0.74 | 15.70 ±2.25 | **14.87** | 6567 |
| 5 | BiLSTM | — | — | 14.12 ±0.71 | 14.53 ±0.27 | 15.35 ±1.11 | 15.73 ±0.23 | **14.94** | 38961 |
| 6 | yaad | Huber (l2 / l1) | data-dependent alpha_t | 13.67 ±0.68 | 13.48 ±0.49 | 16.04 ±0.96 | 16.58 ±0.51 | **14.94** | 6567 |
| 7 | Transformer + sparse attn | — | — | 13.75 ±0.71 | 16.06 ±0.62 | 14.63 ±0.60 | 15.38 ±0.27 | **14.96** | 7713 |
| 8 | Transformer | — | — | 13.94 ±0.47 | 16.65 ±0.83 | 14.39 ±0.63 | 15.84 ±0.60 | **15.21** | 5473 |
| 9 | retnet | dot product | learned constant alpha | 15.37 ±0.47 | 14.06 ±0.16 | 16.69 ±0.03 | 14.80 ±0.46 | **15.23** | 6501 |
| 10 | memora | l2 | KL / softmax | 14.92 ±0.46 | 13.74 ±0.58 | 16.43 ±1.08 | 16.19 ±0.75 | **15.32** | 6535 |
| 11 | moneta | l_p, p=3 | l_q normalization, q=4 | 15.53 ±1.14 | 14.06 ±0.50 | 16.34 ±1.08 | 15.45 ±0.74 | **15.35** | 6533 |
| 12 | deltanet | l2 (delta rule) | no forgetting (alpha=1) | 15.85 ±1.86 | 14.30 ±0.82 | 14.78 ±1.09 | 16.49 ±1.67 | **15.35** | 6499 |
| 13 | linear_attn | dot product (Hebbian) | no forgetting (alpha=1) | 15.82 ±1.99 | 13.83 ±0.07 | 16.11 ±0.26 | 15.78 ±1.19 | **15.38** | 6499 |
| 14 | gated_deltanet | l2 (delta rule) | data-dependent alpha_t | 14.69 ±0.87 | 16.83 ±1.96 | 13.96 ±0.77 | 16.28 ±1.73 | **15.44** | 6533 |
| 15 | mamba2 | dot product | data-dependent alpha_t | 16.20 ±1.34 | 14.20 ±0.66 | 16.01 ±0.70 | 15.59 ±0.36 | **15.50** | 6533 |
| 16 | elastic | l2 | elastic net, soft-threshold | 14.27 ±0.15 | 17.84 ±1.37 | 14.09 ±1.28 | 17.27 ±0.94 | **15.87** | 6535 |
| 17 | robust | l2 + worst-case shift | data-dependent alpha_t | 14.58 ±0.89 | 18.61 ±1.79 | 14.15 ±0.33 | 16.33 ±1.56 | **15.92** | 6535 |
| 18 | mamba2+bilstm + BiLSTM head | dot product | data-dependent alpha_t | 15.94 ±0.91 | 14.38 ±0.48 | 18.25 ±0.41 | 15.52 ±0.54 | **16.03** | 44469 |
| — | SBi-Transformer *(paper 1 published)* | — | — | 11.37 | 12.05 | 11.13 | 11.18 | 11.43 | — |

## Score (lower = better)

| Model | FD001 | FD002 | FD003 | FD004 | mean |
|---|---|---|---|---|---|
| SBi-Transformer (paper 1) | 417.1 ±61.8 | 876.8 ±93.8 | 384.5 ±45.3 | 1285.1 ±47.2 | **740.9** |
| titans+bilstm | 441.2 ±32.0 | 941.2 ±165.4 | 562.6 ±165.3 | 1279.9 ±94.2 | **806.2** |
| BiLSTM | 319.1 ±11.4 | 1202.0 ±216.6 | 585.4 ±270.4 | 1234.4 ±53.8 | **835.2** |
| retnet | 446.4 ±32.8 | 910.8 ±29.1 | 871.6 ±179.5 | 1298.0 ±32.6 | **881.7** |
| gated_deltanet+bilstm | 477.1 ±47.6 | 1091.4 ±81.8 | 451.0 ±86.1 | 1508.8 ±334.7 | **882.1** |
| titans | 449.0 ±66.5 | 1100.9 ±223.1 | 424.4 ±95.2 | 1594.1 ±376.4 | **892.1** |
| linear_attn | 546.9 ±230.5 | 878.1 ±39.7 | 866.7 ±197.4 | 1341.8 ±65.6 | **908.4** |
| yaad | 320.9 ±69.8 | 798.2 ±93.5 | 810.9 ±216.9 | 1761.9 ±77.3 | **923.0** |
| memora | 383.8 ±67.7 | 932.2 ±107.2 | 909.2 ±338.5 | 1487.9 ±96.8 | **928.3** |
| deltanet | 478.7 ±145.4 | 1033.7 ±249.8 | 722.8 ±403.1 | 1569.1 ±510.0 | **951.1** |
| Transformer | 430.4 ±67.7 | 1608.4 ±179.8 | 457.7 ±79.7 | 1312.0 ±237.5 | **952.1** |
| mamba2 | 575.9 ±316.6 | 1022.2 ±231.5 | 756.5 ±176.8 | 1541.8 ±200.0 | **974.1** |
| Transformer + sparse attn | 396.8 ±69.4 | 2111.3 ±689.3 | 494.4 ±62.1 | 1326.6 ±205.1 | **1082.3** |
| gated_deltanet | 412.5 ±95.0 | 1732.5 ±572.9 | 490.1 ±58.6 | 1698.0 ±432.2 | **1083.3** |
| moneta | 505.3 ±51.1 | 1153.6 ±219.1 | 1368.4 ±119.9 | 1406.1 ±179.4 | **1108.3** |
| elastic | 358.0 ±29.9 | 2021.6 ±198.7 | 464.9 ±205.9 | 1659.5 ±104.7 | **1126.0** |
| mamba2+bilstm | 588.5 ±90.1 | 1050.4 ±208.4 | 1864.7 ±352.8 | 1657.9 ±111.9 | **1290.4** |
| robust | 386.3 ±95.9 | 4403.7 ±2835.3 | 407.8 ±31.5 | 1593.8 ±661.9 | **1697.9** |

## Compute cost (seconds / epoch, mean over 4 subsets, shared V100)

| Model | s/epoch | params |
|---|---|---|
| BiLSTM | 1.23 | 38961 |
| Transformer | 2.87 | 5473 |
| Transformer + sparse attn | 4.44 | 7713 |
| SBi-Transformer (paper 1) | 5.40 | 45649 |
| retnet | 6.97 | 6501 |
| linear_attn | 7.13 | 6499 |
| deltanet | 7.39 | 6499 |
| mamba2+bilstm | 7.44 | 44469 |
| elastic | 7.93 | 6535 |
| gated_deltanet+bilstm | 7.97 | 44469 |
| robust | 8.02 | 6535 |
| mamba2 | 8.63 | 6533 |
| gated_deltanet | 9.03 | 6533 |
| titans+bilstm | 9.49 | 44503 |
| yaad | 10.19 | 6567 |
| memora | 10.58 | 6535 |
| moneta | 10.92 | 6533 |
| titans | 11.24 | 6567 |
