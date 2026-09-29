# Autoresearch — searching for a strong RUL method on C-MAPSS

3-stage successive-halving loop, **selection done entirely on the validation set**; the test set is opened only once, at the final step, for the top-3.

## Search stages (ranked by val)

| stage | # configs | budget | best val | median val |
|---|---|---|---|---|
| 1 | 42 | 30 epochs | 12.568 | 14.998 |
| 2 | 16 | 70 epochs | 12.365 | 12.880 |
| 3 | 8 | 150 epochs, 2 seeds | 11.775 | 11.950 |

## Design trends — last-stage survivors vs. the paper's configuration

| parameter | paper (Table 2) | most common among survivors |
|---|---|---|
| `arch` | sbi:yes_yes_yes | `sbi:yes_yes_yes` (2/4), `sbi:yes_yes_no` (1/4) |
| `seq_len` | 45 | `60` (2/4), `45` (2/4) |
| `rul_cap` | 125 | `125` (4/4) |
| `feature_mode` | paper | `paper` (4/4) |
| `cond_norm` | not stated | `True` (4/4) |
| `num_hidden` | 16 | `16` (3/4), `32` (1/4) |
| `ffn_hidden` | 32 | `128` (2/4), `32` (1/4) |
| `encoder_layers` | 3 / 2 | `4` (3/4), `2` (1/4) |
| `bilstm_size` | 32 | `32` (3/4), `64` (1/4) |
| `n_heads` | 2 | `2` (3/4), `4` (1/4) |
| `lr` | 5e-4 / 1e-4 | `0.0005` (2/4), `0.0001` (1/4) |
| `dropout` | 0.2 / 0.3 | `0.2` (3/4), `0.3` (1/4) |
| `weight_decay` | 1e-05 | `0.0001` (4/4) |
| `batch_size` | 256 | `256` (3/4), `128` (1/4) |

## Final TEST results (3 seeds per cell)

| config | FD001 | FD002 | FD003 | FD004 | mean | params |
|---|---|---|---|---|---|---|
| #1 `h0e17f870b0` | 14.14 ±0.81 | 12.63 ±0.17 | 12.63 ±0.41 | 13.70 ±0.20 | **13.27** | 70497 |
| #2 `h04b5304516` | 14.26 ±0.28 | 12.35 ±0.31 | 13.76 ±0.60 | 13.77 ±0.50 | **13.53** | 142689 |
| #3 `h2bee3d8f59` | 14.50 ±0.17 | 13.05 ±0.12 | 13.94 ±0.44 | 13.90 ±0.47 | **13.85** | 52337 |
| **published paper** | 11.37 | 12.05 | 11.13 | 11.18 | 11.43 | — |

**#5 — search subsets vs. held-out subsets.** The loop ran only on `FD001` + `FD004` (via their own val sets), so `FD002` + `FD003` are truly held out. A large gap between the two groups would signal that the configs are over-specialized to the pair used for the search.

| config | search (FD001, FD004) | held-out (FD002, FD003) | gap |
|---|---|---|---|
| #1 `h0e17f870b0` | 13.92 | 12.63 | -1.29 |
| #2 `h04b5304516` | 14.02 | 13.05 | -0.96 |
| #3 `h2bee3d8f59` | 14.20 | 13.50 | -0.71 |
| ICL4RUL [41] (cited in paper) | 10.26 | 14.21 | 10.11 | 16.38 | 12.74 | — |

### Score

| config | FD001 | FD002 | FD003 | FD004 | mean |
|---|---|---|---|---|---|
| #1 `h0e17f870b0` | 370.3 | 729.5 | 296.0 | 1217.2 | **653.3** |
| #2 `h04b5304516` | 352.5 | 697.0 | 531.7 | 1052.5 | **658.4** |
| #3 `h2bee3d8f59` | 386.0 | 882.3 | 390.3 | 1065.0 | **680.9** |
| **published paper** | 267.54 | 841.02 | 273.44 | 926.23 | 577.1 |

### Configuration details

**#1 `h0e17f870b0`** — val (stage 3) = 11.775

```
arch=sbi:yes_yes_no  seq_len=60  rul_cap=125  feature_mode=paper  cond_norm=True
num_hidden=32  ffn_hidden=128  encoder_layers=4  bilstm_size=32  n_heads=2
lr=0.0005  dropout=0.3  weight_decay=0.0001  batch_size=256
```

**#2 `h04b5304516`** — val (stage 3) = 11.844

```
arch=sbi:no_no_yes  seq_len=60  rul_cap=125  feature_mode=paper  cond_norm=True
num_hidden=16  ffn_hidden=128  encoder_layers=4  bilstm_size=64  n_heads=4
lr=0.0001  dropout=0.2  weight_decay=0.0001  batch_size=128
```

**#3 `h2bee3d8f59`** — val (stage 3) = 11.882

```
arch=sbi:yes_yes_yes  seq_len=45  rul_cap=125  feature_mode=paper  cond_norm=True
num_hidden=16  ffn_hidden=32  encoder_layers=4  bilstm_size=32  n_heads=2
lr=0.0005  dropout=0.2  weight_decay=0.0001  batch_size=256
```
