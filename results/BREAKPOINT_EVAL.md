# Breakpoint evaluation (MathWorks similarity-RUL protocol)

Held-out run-to-failure validation engines, cut at 50 / 70 / 90% of their life. Error = estimated − true RUL (cycles); estimate = mean over 5 seeds. True RUL: remaining cycles, as in the example (models predict RUL capped at 125, so early-life errors on long engines are negative by construction).

## FD001

| method | mean 50% | mean 70% | mean 90% | median 50% | median 70% | median 90% | s.d. 50% | s.d. 70% | s.d. 90% | engines |
|---|---|---|---|---|---|---|---|---|---|---|
| Memora (Miras) | 3.94 | 0.70 | -1.41 | 12.50 | 2.74 | -0.09 | 22.32 | 12.18 | 3.68 | 20 |
| DCNN | 4.06 | 2.80 | -0.56 | 10.93 | 4.78 | 0.00 | 20.94 | 11.58 | 3.03 | 20 |
| STA-HPINN, no physics | 0.86 | 2.05 | 0.27 | 8.41 | 1.74 | 1.44 | 21.24 | 9.13 | 4.74 | 20 |
| Titans (Miras) | 3.34 | 1.59 | -1.48 | 11.26 | 3.13 | 0.21 | 23.05 | 11.49 | 3.84 | 20 |

## FD002

| method | mean 50% | mean 70% | mean 90% | median 50% | median 70% | median 90% | s.d. 50% | s.d. 70% | s.d. 90% | engines |
|---|---|---|---|---|---|---|---|---|---|---|
| Memora (Miras) | 3.95 | 5.04 | -0.89 | 4.59 | 4.45 | -0.47 | 19.44 | 13.30 | 3.55 | 52 |
| DCNN | 1.88 | 4.51 | -0.67 | 4.27 | 4.91 | -0.08 | 18.59 | 12.15 | 3.34 | 52 |
| STA-HPINN, no physics | 2.51 | 4.56 | 0.25 | 4.67 | 2.07 | -0.00 | 16.62 | 10.15 | 3.49 | 52 |
| Titans (Miras) | 4.88 | 4.82 | -1.07 | 4.56 | 3.93 | -1.67 | 20.20 | 13.95 | 3.79 | 52 |

## FD003

| method | mean 50% | mean 70% | mean 90% | median 50% | median 70% | median 90% | s.d. 50% | s.d. 70% | s.d. 90% | engines |
|---|---|---|---|---|---|---|---|---|---|---|
| Memora (Miras) | -24.54 | -6.81 | -1.96 | -2.52 | 2.53 | -1.38 | 53.43 | 19.06 | 6.28 | 20 |
| DCNN | -26.16 | -3.19 | -2.08 | -6.30 | 5.73 | -2.76 | 51.89 | 17.68 | 3.98 | 20 |
| STA-HPINN, no physics | -28.47 | -8.05 | 0.10 | -14.42 | -3.15 | 0.86 | 50.27 | 11.65 | 3.46 | 20 |
| Titans (Miras) | -25.81 | -6.40 | -2.91 | -5.97 | 3.01 | -1.64 | 53.66 | 20.93 | 5.55 | 20 |

## FD004

| method | mean 50% | mean 70% | mean 90% | median 50% | median 70% | median 90% | s.d. 50% | s.d. 70% | s.d. 90% | engines |
|---|---|---|---|---|---|---|---|---|---|---|
| Memora (Miras) | -19.89 | 3.62 | 0.31 | -16.67 | 5.52 | 0.07 | 39.71 | 15.14 | 5.91 | 50 |
| DCNN | -20.57 | 4.72 | 1.57 | -22.57 | 5.77 | 0.83 | 39.68 | 13.82 | 6.04 | 50 |
| STA-HPINN, no physics | -21.51 | 3.50 | 2.24 | -23.05 | 3.54 | 2.75 | 37.38 | 11.52 | 5.26 | 50 |
| Titans (Miras) | -18.95 | 4.86 | 0.26 | -18.31 | 2.15 | -0.36 | 40.33 | 16.84 | 5.65 | 50 |
