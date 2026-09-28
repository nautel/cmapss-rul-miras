# Autoresearch — tim phuong phap RUL manh tren C-MAPSS

Vong lap successive halving 3 tang, **chon loc hoan toan tren tap validation**; tap test chi duoc mo mot lan o buoc cuoi cho top-3.

## Cac tang tim kiem (xep hang theo val)

| tang | so cau hinh | ngan sach | val tot nhat | val trung vi |
|---|---|---|---|---|
| 1 | 42 | 30 epoch | 10.224 | 14.310 |
| 2 | 16 | 70 epoch | 10.145 | 10.691 |
| 3 | 8 | 150 epoch, 2 seed | 9.782 | 10.287 |

## Xu huong thiet ke — nhom song sot o tang cuoi so voi cau hinh cua paper

| tham so | paper (Table 2) | pho bien nhat trong nhom song sot |
|---|---|---|
| `arch` | sbi:yes_yes_yes | `titans+bilstm` (2/4), `sbi:yes_yes_no` (1/4) |
| `seq_len` | 45 | `60` (3/4), `45` (1/4) |
| `rul_cap` | 125 | `110` (4/4) |
| `feature_mode` | paper | `classic14` (4/4) |
| `cond_norm` | khong neu | `True` (3/4), `False` (1/4) |
| `num_hidden` | 16 | `64` (3/4), `16` (1/4) |
| `ffn_hidden` | 32 | `32` (3/4), `128` (1/4) |
| `encoder_layers` | 3 / 2 | `1` (3/4), `3` (1/4) |
| `bilstm_size` | 32 | `64` (4/4) |
| `n_heads` | 2 | `4` (4/4) |
| `lr` | 5e-4 / 1e-4 | `0.0005` (3/4), `0.0003` (1/4) |
| `dropout` | 0.2 / 0.3 | `0.1` (4/4) |
| `weight_decay` | 1e-05 | `1e-05` (3/4), `0.0001` (1/4) |
| `batch_size` | 256 | `128` (3/4), `256` (1/4) |

## Ket qua TEST cuoi cung (3 seed moi o)

| cau hinh | FD001 | FD002 | FD003 | FD004 | TB | tham so |
|---|---|---|---|---|---|---|
| #1 `h52af25d33b` | 10.40 ±0.07 | 10.35 ±0.31 | 10.63 ±0.39 | 11.15 ±0.26 | **10.63** | 197348 |
| #2 `hd30913ad70` | 10.69 ±0.60 | 10.36 ±0.47 | 11.03 ±0.15 | 11.10 ±0.18 | **10.79** | 209732 |
| #3 `h17468788cd` | 10.46 ±0.71 | 12.13 ±0.58 | 9.83 ±0.27 | 11.38 ±0.15 | **10.95** | 11297 |
| **paper cong bo** | 11.37 | 12.05 | 11.13 | 11.18 | 11.43 | — |
| ICL4RUL [41] (paper trich) | 10.26 | 14.21 | 10.11 | 16.38 | 12.74 | — |

### Score

| cau hinh | FD001 | FD002 | FD003 | FD004 | TB |
|---|---|---|---|---|---|
| #1 `h52af25d33b` | 160.1 | 445.2 | 230.7 | 593.0 | **357.2** |
| #2 `hd30913ad70` | 175.1 | 466.4 | 294.1 | 625.9 | **390.4** |
| #3 `h17468788cd` | 171.8 | 772.1 | 176.0 | 582.7 | **425.6** |
| **paper cong bo** | 267.54 | 841.02 | 273.44 | 926.23 | 577.1 |

### Cau hinh chi tiet

**#1 `h52af25d33b`** — val (tang 3) = 9.994

```
arch=titans+bilstm  seq_len=60  rul_cap=110  feature_mode=classic14  cond_norm=True
num_hidden=64  ffn_hidden=32  encoder_layers=1  bilstm_size=64  n_heads=4
lr=0.0005  dropout=0.1  weight_decay=1e-05  batch_size=128
```

**#2 `hd30913ad70`** — val (tang 3) = 10.007

```
arch=titans+bilstm  seq_len=60  rul_cap=110  feature_mode=classic14  cond_norm=True
num_hidden=64  ffn_hidden=128  encoder_layers=1  bilstm_size=64  n_heads=4
lr=0.0005  dropout=0.1  weight_decay=1e-05  batch_size=256
```

**#3 `h17468788cd`** — val (tang 3) = 9.782

```
arch=sbi:yes_yes_no  seq_len=60  rul_cap=110  feature_mode=classic14  cond_norm=False
num_hidden=16  ffn_hidden=32  encoder_layers=3  bilstm_size=64  n_heads=4
lr=0.0003  dropout=0.1  weight_decay=0.0001  batch_size=128
```

> Luu y ve phuong phap: cau hinh co val TOT NHAT (`h17468788cd`) khong phai cau hinh co test tot nhat (`h52af25d33b`). Val khong du bao hoan hao test — do chinh la ly do phai chot top-K theo val roi moi mo test, thay vi xep hang lai theo test.
