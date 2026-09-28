"""So lieu cong bo trong paper — de doi chieu voi ket qua tai lap."""

# Table 3 — RMSE / Score cua tung phuong phap
TABLE3 = {
    "BiLSTM":           dict(rmse=[17.31, 25.82, 17.76, 28.17], score=[395.33, 6117.25, 936.04, 7618.01]),
    "Transformer [37]": dict(rmse=[13.33, 13.37, 13.28, 13.02], score=[293.70, 1628.34, 302.10, 3108.32]),
    "Informer [38]":    dict(rmse=[13.13, 13.20, 12.58, 14.16], score=[263, 715, 228, 1023]),
    "AutoFormer [39]":  dict(rmse=[23.04, 16.51, 25.40, 20.31], score=[1063, 1248, 2034, 2291]),
    "CATA-TCN [40]":    dict(rmse=[12.80, 17.61, 13.16, 21.04], score=[234.31, 1361.12, 290.63, 2303.42]),
    "ICL4RUL [41]":     dict(rmse=[10.26, 14.21, 10.11, 16.38], score=[185.29, 915.97, 146.56, 1244.58]),
    "SBi-Transformer":  dict(rmse=[11.37, 12.05, 11.13, 11.18], score=[267.54, 841.02, 273.44, 926.23]),
}

# Table 4 / Table 5 — ablation (thu tu FD001..FD004)
ABL_RMSE = {
    "no_no_yes":   [13.76, 23.40, 13.32, 23.45],
    "yes_no_no":   [13.33, 13.37, 13.28, 13.02],
    "yes_no_yes":  [12.98, 13.15, 13.11, 12.71],
    "yes_yes_no":  [12.75, 12.95, 12.55, 12.37],
    "yes_yes_yes": [11.37, 12.05, 11.13, 11.18],
}
ABL_SCORE = {
    "no_no_yes":   [345.35, 1997.86, 319.63, 3510.77],
    "yes_no_no":   [336.70, 1754.00, 309.17, 2180.65],
    "yes_no_yes":  [298.00, 1627.34, 407.00, 1747.81],
    "yes_yes_no":  [271.70, 1077.41, 305.10, 1211.32],
    "yes_yes_yes": [267.54, 841.02, 273.44, 926.23],
}

# Table 6 — thoi gian moi epoch (giay)
TABLE6 = {
    "CATA-TCN [42]":   [6.91, 6.95, 7.84, 7.79],
    "LSTM [43]":       [6.2, 16.4, 7.5, 19.5],
    "SBi-Transformer": [4.8, 4.5, 4.2, 5.2],
}

ABL_LABEL = {
    "no_no_yes":   "No / No / Yes",
    "yes_no_no":   "Yes / No / No",
    "yes_no_yes":  "Yes / No / Yes",
    "yes_yes_no":  "Yes / Yes / No",
    "yes_yes_yes": "Yes / Yes / Yes",
}
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


# --- #7: dinh vi so voi van lieu -------------------------------------------
# Bang so sanh trich tu Table 2 cua STA-HPINN (Spatio-temporal Attention-based Hidden
# Physics-informed NN), arXiv:2405.12377, muc 4. Day la bang SOTA gan nhat tim duoc
# co day du ca RMSE va Score cho 4 bo con.
SOTA_RMSE = {
    "DCFA (2023)":              [11.74, 16.81, 10.71, 17.77],
    "3D Attention (2023)":      [13.12, 13.93, 12.15, 20.34],
    "MTSTAN (2023)":            [10.97, 16.81, 10.90, 18.85],
    "DVGTformer (2024)":        [11.33, 14.28, 11.89, 15.50],
    "Optim. Distrib. (2024)":   [11.96, 13.51, 11.40, 17.58],
    "STA-HPINN (2024)":         [11.27, 13.21,  8.30, 13.31],
}
SOTA_SCORE = {
    "DCFA (2023)":              [190.0, 1076.0, 198.0, 1571.0],
    "3D Attention (2023)":      [231.01, 759.84, 195.56, 1710.29],
    "MTSTAN (2023)":            [175.36, 1154.36, 188.22, 1446.29],
    "DVGTformer (2024)":        [179.55, 797.26, 254.55, 1107.50],
    "Optim. Distrib. (2024)":   [233.40, 902.13, 255.60, 1704.59],
    "STA-HPINN (2024)":         [211.03, 764.42, 119.93, 849.98],
}
