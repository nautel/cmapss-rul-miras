# Tai lap SBi-Transformer — Ren et al., Results in Engineering 29 (2026) 109187

Chay tren cassio (Tesla V100). Moi o = trung binh 3 seed.

## Table 3 — do chinh xac so voi cac phuong phap khac

| Algorithm | RMSE FD001 | RMSE FD002 | RMSE FD003 | RMSE FD004 | Score FD001 | Score FD002 | Score FD003 | Score FD004 |
|---|---|---|---|---|---|---|---|---|
| BiLSTM (paper) | 17.31 | 25.82 | 17.76 | 28.17 | 395.33 | 6117.25 | 936.04 | 7618.01 |
| Transformer [37] (paper) | 13.33 | 13.37 | 13.28 | 13.02 | 293.70 | 1628.34 | 302.10 | 3108.32 |
| Informer [38] (paper) | 13.13 | 13.20 | 12.58 | 14.16 | 263.00 | 715.00 | 228.00 | 1023.00 |
| AutoFormer [39] (paper) | 23.04 | 16.51 | 25.40 | 20.31 | 1063.00 | 1248.00 | 2034.00 | 2291.00 |
| CATA-TCN [40] (paper) | 12.80 | 17.61 | 13.16 | 21.04 | 234.31 | 1361.12 | 290.63 | 2303.42 |
| ICL4RUL [41] (paper) | 10.26 | 14.21 | 10.11 | 16.38 | 185.29 | 915.97 | 146.56 | 1244.58 |
| SBi-Transformer (paper) | 11.37 | 12.05 | 11.13 | 11.18 | 267.54 | 841.02 | 273.44 | 926.23 |
| **SBi-Transformer (tai lap)** | 14.26 ±0.29 | 17.12 ±0.59 | 13.80 ±0.44 | 18.92 ±1.77 | 417.1 ±61.8 | 2588.3 ±282.2 | 384.5 ±45.3 | 3216.4 ±1399.4 |
| Transformer only (tai lap) | 13.94 ±0.47 | 16.65 ±0.83 | 14.39 ±0.63 | 18.37 ±1.90 | 430.4 ±67.7 | 1608.4 ±179.8 | 457.7 ±79.7 | 2516.8 ±1040.5 |
| BiLSTM only (tai lap) | 14.12 ±0.71 | 15.37 ±0.21 | 15.92 ±2.96 | 17.63 ±0.18 | 319.1 ±11.4 | 1247.3 ±192.6 | 846.1 ±737.6 | 2136.6 ±301.0 |

### Table 4 — ablation, RMSE (dung som nhu Table 2: patience=10)

| Cau hinh (Transformer / Sparse / BiLSTM) | FD001 | FD002 | FD003 | FD004 | Trung binh |
|---|---|---|---|---|---|
| No / No / Yes | 14.12 ±0.71 (+2.6%) | 15.37 ±0.21 (-34.3%) | 15.92 ±2.96 (+19.5%) | 17.63 ±0.18 (-24.8%) | 15.76 (paper 18.48) |
| Yes / No / No | 13.94 ±0.47 (+4.6%) | 16.65 ±0.83 (+24.6%) | 14.39 ±0.63 (+8.4%) | 18.37 ±1.90 (+41.1%) | 15.84 (paper 13.25) |
| Yes / No / Yes | 14.32 ±0.91 (+10.3%) | 16.42 ±0.57 (+24.9%) | 13.81 ±0.73 (+5.3%) | 17.83 ±0.64 (+40.3%) | 15.59 (paper 12.99) |
| Yes / Yes / No | 13.75 ±0.71 (+7.8%) | 17.25 ±0.75 (+33.2%) | 14.75 ±0.75 (+17.5%) | 17.68 ±2.08 (+42.9%) | 15.86 (paper 12.65) |
| Yes / Yes / Yes | 14.26 ±0.29 (+25.4%) | 17.12 ±0.59 (+42.0%) | 13.80 ±0.44 (+24.0%) | 18.92 ±1.77 (+69.3%) | 16.02 (paper 11.43) |

Trong ngoac: lech tuong doi so voi tri paper. `±` = do lech chuan giua cac seed.

### Table 5 — ablation, Score (dung som)

| Cau hinh (Transformer / Sparse / BiLSTM) | FD001 | FD002 | FD003 | FD004 | Trung binh |
|---|---|---|---|---|---|
| No / No / Yes | 319.1 ±11.4 (-7.6%) | 1247.3 ±192.6 (-37.6%) | 846.1 ±737.6 (+164.7%) | 2136.6 ±301.0 (-39.1%) | 1137.3 (paper 1543.40) |
| Yes / No / No | 430.4 ±67.7 (+27.8%) | 1608.4 ±179.8 (-8.3%) | 457.7 ±79.7 (+48.0%) | 2516.8 ±1040.5 (+15.4%) | 1253.3 (paper 1145.13) |
| Yes / No / Yes | 391.2 ±53.0 (+31.3%) | 2164.0 ±443.9 (+33.0%) | 412.3 ±104.0 (+1.3%) | 2192.2 ±500.0 (+25.4%) | 1289.9 (paper 1020.04) |
| Yes / Yes / No | 396.8 ±69.4 (+46.1%) | 3057.7 ±396.5 (+183.8%) | 507.8 ±102.2 (+66.4%) | 2208.1 ±933.4 (+82.3%) | 1542.6 (paper 716.38) |
| Yes / Yes / Yes | 417.1 ±61.8 (+55.9%) | 2588.3 ±282.2 (+207.8%) | 384.5 ±45.3 (+40.6%) | 3216.4 ±1399.4 (+247.3%) | 1651.6 (paper 577.06) |

Trong ngoac: lech tuong doi so voi tri paper. `±` = do lech chuan giua cac seed.

> Early stopping ban rat som (FD001/FD003: 22-24 epoch) va cat mo hinh LON nang hon mo hinh nho, nen bang tren KHONG so sanh ablation cong bang. Hai bang duoi chay du 200/600 epoch cua Table 2.

### Table 4 — ablation, RMSE (du epoch Table 2)

| Cau hinh (Transformer / Sparse / BiLSTM) | FD001 | FD002 | FD003 | FD004 | Trung binh |
|---|---|---|---|---|---|
| No / No / Yes | 14.12 ±0.71 (+2.6%) | 14.53 ±0.27 (-37.9%) | 15.35 ±1.11 (+15.3%) | 15.73 ±0.23 (-32.9%) | 14.94 (paper 18.48) |
| Yes / No / No | 14.17 ±0.64 (+6.3%) | 17.22 ±1.62 (+28.8%) | 14.87 ±0.23 (+12.0%) | 15.84 ±0.60 (+21.7%) | 15.52 (paper 13.25) |
| Yes / No / Yes | 14.46 ±1.15 (+11.4%) | 15.45 ±0.47 (+17.5%) | 13.70 ±0.54 (+4.5%) | 17.02 ±0.22 (+33.9%) | 15.16 (paper 12.99) |
| Yes / Yes / No | 14.13 ±1.38 (+10.9%) | 16.06 ±0.62 (+24.0%) | 14.63 ±0.60 (+16.6%) | 15.38 ±0.27 (+24.3%) | 15.05 (paper 12.65) |
| Yes / Yes / Yes | 14.44 ±0.23 (+27.0%) | 16.71 ±0.38 (+38.7%) | 13.80 ±0.44 (+24.0%) | 17.64 ±0.32 (+57.8%) | 15.65 (paper 11.43) |

Trong ngoac: lech tuong doi so voi tri paper. `±` = do lech chuan giua cac seed.

### Table 5 — ablation, Score (du epoch Table 2)

| Cau hinh (Transformer / Sparse / BiLSTM) | FD001 | FD002 | FD003 | FD004 | Trung binh |
|---|---|---|---|---|---|
| No / No / Yes | 319.1 ±11.4 (-7.6%) | 1202.0 ±216.6 (-39.8%) | 585.4 ±270.4 (+83.1%) | 1234.4 ±53.8 (-64.8%) | 835.2 (paper 1543.40) |
| Yes / No / No | 441.5 ±87.9 (+31.1%) | 7978.6 ±9052.4 (+354.9%) | 555.6 ±45.4 (+79.7%) | 1312.0 ±237.5 (-39.8%) | 2571.9 (paper 1145.13) |
| Yes / No / Yes | 419.9 ±101.7 (+40.9%) | 1660.2 ±86.5 (+2.0%) | 404.9 ±92.2 (-0.5%) | 1539.9 ±124.8 (-11.9%) | 1006.2 (paper 1020.04) |
| Yes / Yes / No | 431.0 ±128.5 (+58.6%) | 2111.3 ±689.3 (+96.0%) | 494.4 ±62.1 (+62.0%) | 1326.6 ±205.1 (+9.5%) | 1090.8 (paper 716.38) |
| Yes / Yes / Yes | 429.6 ±69.5 (+60.6%) | 2228.3 ±168.3 (+164.9%) | 384.5 ±45.3 (+40.6%) | 1873.9 ±215.9 (+102.3%) | 1229.0 (paper 577.06) |

Trong ngoac: lech tuong doi so voi tri paper. `±` = do lech chuan giua cac seed.

### Table 6 — thoi gian moi epoch (giay)

| Phuong phap | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| CATA-TCN [42] (paper) | 6.91 | 6.95 | 7.84 | 7.79 |
| LSTM [43] (paper) | 6.20 | 16.40 | 7.50 | 19.50 |
| SBi-Transformer (paper) | 4.80 | 4.50 | 4.20 | 5.20 |
| **SBi-Transformer (tai lap, V100)** | 3.41 ±0.31 | 5.82 ±0.23 | 4.25 ±0.53 | 6.71 ±0.48 |

Paper do tren CPU i7-13800H; cot tai lap do tren 1 GPU V100 (may dung chung, tai bien dong) nen chi so sanh duoc theo ty le giua cac subset.

### Do bat dinh (Bootstrap + t-distribution, Eq. 13-14)

| Subset | RMSE (mean MC-dropout) | Score | CI Eq.14: do phu / be rong | CI du bao: do phu / be rong | Epoch chay | Tham so |
|---|---|---|---|---|---|---|
| FD001 | 13.64 ±0.35 | 292.5 ±21.9 | 0.000 ±0.000 / 0.05 ±0.00 | 0.410 ±0.030 / 11.62 ±0.30 | 22 ±2 | 48993 |
| FD002 | 16.43 ±0.33 | 1996.0 ±90.9 | 0.003 ±0.002 / 0.06 ±0.00 | 0.328 ±0.010 / 13.11 ±0.79 | 84 ±8 | 45649 |
| FD003 | 13.15 ±0.19 | 309.2 ±24.1 | 0.000 ±0.000 / 0.05 ±0.00 | 0.403 ±0.021 / 10.98 ±0.88 | 24 ±7 | 48993 |
| FD004 | 18.37 ±2.03 | 1998.7 ±855.5 | 0.004 ±0.000 / 0.06 ±0.00 | 0.238 ±0.026 / 14.45 ±0.34 | 72 ±35 | 45649 |

Eq. (14) chia sigma cho `sqrt(n_bootstrap)` sau khi sigma da la do lech chuan cua mot ensemble trung binh -> khoang hep den muc do phu ~0. Cot 'CI du bao' dung do tan MC-dropout (`f*t*sigma_mc`), la thu tuong ung voi dai mau trong Fig. 5 cua paper.

### Bien the — tach rieng anh huong cua dung som va cua chuan hoa

| Bien the | RMSE FD001 | RMSE FD002 | RMSE FD003 | RMSE FD004 | Score FD001 | Score FD002 | Score FD003 | Score FD004 |
|---|---|---|---|---|---|---|---|---|
| Nhu paper (seq_len=45, patience=10, Z-score toan cuc) | 14.26 ±0.29 | 17.12 ±0.59 | 13.80 ±0.44 | 18.92 ±1.77 | 417.1 ±61.8 | 2588.3 ±282.2 | 384.5 ±45.3 | 3216.4 ±1399.4 |
| + du epoch Table 2 (bo dung som) | 14.44 ±0.23 | 16.71 ±0.38 | 13.80 ±0.44 | 17.64 ±0.32 | 429.6 ±69.5 | 2228.3 ±168.3 | 384.5 ±45.3 | 1873.9 ±215.9 |
| + chuan hoa theo che do van hanh | — | 13.04 ±0.18 | — | 14.84 ±0.61 | — | 876.8 ±93.8 | — | 1301.8 ±72.0 |
| + du epoch + chuan hoa theo che do | — | 13.40 ±0.73 | — | 14.74 ±0.55 | — | 937.2 ±201.0 | — | 1285.1 ±47.2 |
| seq_len=30 (theo Table 1), du epoch | 14.67 ±0.20 | 15.92 ±0.62 | 15.64 ±0.72 | 19.79 ±0.52 | 419.7 ±54.3 | 1390.8 ±157.0 | 670.3 ±236.5 | 2657.8 ±79.2 |
| seq_len=30 + chuan hoa theo che do, du epoch | — | 14.64 ±0.06 | — | 16.02 ±0.37 | — | 1125.9 ±125.8 | — | 1838.8 ±248.2 |
| **paper** | 11.37 | 12.05 | 11.13 | 11.18 | 267.54 | 841.02 | 273.44 | 926.23 |

FD001/FD003 chi co 1 che do van hanh nen chuan hoa theo che do trung voi Z-score toan cuc — khong chay.

So epoch thuc su chay khi bat dung som: FD001: 22, FD002: 84, FD003: 24, FD004: 72 (Table 2 ghi 200 cho FD001/FD003 va 600 cho FD002/FD004).
