# Autoresearch — searching for a strong RUL method on C-MAPSS

> **Superseded — do not use these numbers.** This first search put `rul_cap` in the search
> space, and the cap also changed the TEST labels (at 110, 28–85 engines per subset got
> lowered labels), so the RMSE values below are not comparable to the literature. The
> corrected run, with the training cap separated from the evaluation cap, is
> [`AUTORESEARCH2.md`](AUTORESEARCH2.md).

3-stage successive-halving loop, **selection done entirely on the validation set**; the test set is opened only once, at the final step, for the top-3.

## Search stages (ranked by val)

| stage | # configs | budget | best val | median val |
|---|---|---|---|---|
| 1 | 42 | 30 epochs | 10.224 | 14.310 |
| 2 | 16 | 70 epochs | 10.145 | 10.691 |
| 3 | 8 | 150 epochs, 2 seeds | 9.782 | 10.287 |

## Design trends — last-stage survivors vs. the paper's configuration

| parameter | paper (Table 2) | most common among survivors |
|---|---|---|
| `arch` | sbi:yes_yes_yes | `titans+bilstm` (2/4), `sbi:yes_yes_no` (1/4) |
| `seq_len` | 45 | `60` (3/4), `45` (1/4) |
| `rul_cap` | 125 | `110` (4/4) |
| `feature_mode` | paper | `classic14` (4/4) |
| `cond_norm` | not stated | `True` (3/4), `False` (1/4) |
| `num_hidden` | 16 | `64` (3/4), `16` (1/4) |
| `ffn_hidden` | 32 | `32` (3/4), `128` (1/4) |
| `encoder_layers` | 3 / 2 | `1` (3/4), `3` (1/4) |
| `bilstm_size` | 32 | `64` (4/4) |
| `n_heads` | 2 | `4` (4/4) |
| `lr` | 5e-4 / 1e-4 | `0.0005` (3/4), `0.0003` (1/4) |
| `dropout` | 0.2 / 0.3 | `0.1` (4/4) |
| `weight_decay` | 1e-05 | `1e-05` (3/4), `0.0001` (1/4) |
| `batch_size` | 256 | `128` (3/4), `256` (1/4) |

## Final TEST results (3 seeds per cell)

| config | FD001 | FD002 | FD003 | FD004 | mean | params |
|---|---|---|---|---|---|---|
| #1 `h52af25d33b` | 10.40 ±0.07 | 10.35 ±0.31 | 10.63 ±0.39 | 11.15 ±0.26 | **10.63** | 197348 |
| #2 `hd30913ad70` | 10.69 ±0.60 | 10.36 ±0.47 | 11.03 ±0.15 | 11.10 ±0.18 | **10.79** | 209732 |
| #3 `h17468788cd` | 10.46 ±0.71 | 12.13 ±0.58 | 9.83 ±0.27 | 11.38 ±0.15 | **10.95** | 11297 |
| **published paper** | 11.37 | 12.05 | 11.13 | 11.18 | 11.43 | — |

**#5 — search subsets vs. held-out subsets.** The loop ran only on `FD001` + `FD004` (via their own val sets), so `FD002` + `FD003` are truly held out. A large gap between the two groups would signal that the configs are over-specialized to the pair used for the search.

| config | search (FD001, FD004) | held-out (FD002, FD003) | gap |
|---|---|---|---|
| #1 `h52af25d33b` | 10.77 | 10.49 | -0.28 |
| #2 `hd30913ad70` | 10.89 | 10.69 | -0.20 |
| #3 `h17468788cd` | 10.92 | 10.98 | +0.06 |
| ICL4RUL [41] (cited in paper) | 10.26 | 14.21 | 10.11 | 16.38 | 12.74 | — |

### Score

| config | FD001 | FD002 | FD003 | FD004 | mean |
|---|---|---|---|---|---|
| #1 `h52af25d33b` | 160.1 | 445.2 | 230.7 | 593.0 | **357.2** |
| #2 `hd30913ad70` | 175.1 | 466.4 | 294.1 | 625.9 | **390.4** |
| #3 `h17468788cd` | 171.8 | 772.1 | 176.0 | 582.7 | **425.6** |
| **published paper** | 267.54 | 841.02 | 273.44 | 926.23 | 577.1 |

### Configuration details

**#1 `h52af25d33b`** — val (stage 3) = 9.994

```
arch=titans+bilstm  seq_len=60  rul_cap=110  feature_mode=classic14  cond_norm=True
num_hidden=64  ffn_hidden=32  encoder_layers=1  bilstm_size=64  n_heads=4
lr=0.0005  dropout=0.1  weight_decay=1e-05  batch_size=128
```

**#2 `hd30913ad70`** — val (stage 3) = 10.007

```
arch=titans+bilstm  seq_len=60  rul_cap=110  feature_mode=classic14  cond_norm=True
num_hidden=64  ffn_hidden=128  encoder_layers=1  bilstm_size=64  n_heads=4
lr=0.0005  dropout=0.1  weight_decay=1e-05  batch_size=256
```

**#3 `h17468788cd`** — val (stage 3) = 9.782

```
arch=sbi:yes_yes_no  seq_len=60  rul_cap=110  feature_mode=classic14  cond_norm=False
num_hidden=16  ffn_hidden=32  encoder_layers=3  bilstm_size=64  n_heads=4
lr=0.0003  dropout=0.1  weight_decay=0.0001  batch_size=128
```

> Methodological note: the config with the BEST val (`h17468788cd`) is not the config with the best test (`h52af25d33b`). Val does not predict test perfectly — that is exactly why top-K must be fixed by val before opening test, instead of re-ranking by test.
