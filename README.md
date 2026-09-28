# cmapss-rul-miras

Ap dung khung **Miras** (Behrouz et al., *It's All Connected*, arXiv:2504.13173, Google
Research 2025) cho bai toan du doan tuoi tho con lai (RUL) dong co turbofan tren
**NASA C-MAPSS**, va so sanh voi SOTA trong **cung mot pipeline**.

## Ket qua chinh

RMSE, 10 seed moi o, CI 95% bootstrap tren engine o [`results/MIRAS_FIX.md`](results/MIRAS_FIX.md).

| mo hinh | FD001 | FD002 | FD003 | FD004 | TB |
|---|---|---|---|---|---|
| DCNN (Li 2018) — moc | 12.58 | 12.25 | 12.12 | 13.07 | **12.50** |
| STA-HPINN bo physics — SOTA tai tao | 11.69 | 13.94 | **10.27** | 15.88 | 12.95 |
| **Miras / Memora** (KL retention) | 13.64 | 12.30 | 12.99 | 13.88 | **13.20** |
| Miras / Titans | 13.67 | 13.25 | 13.46 | 14.69 | 13.77 |
| Miras / Titans, bo nho hang 4 | 14.17 | 12.77 | 13.73 | 14.40 | 13.77 |
| Miras / DeltaNet, Gated DeltaNet | 13.8 | 13.3–13.4 | 14.0 | 14.8 | 13.98–13.99 |
| Miras / Linear Attn, Mamba2, Moneta | 13.7–14.0 | 13.7–14.5 | 13.6–14.3 | 15.3–15.7 | 14.31–14.35 |
| Miras / Yaad | 13.88 | 13.45 | 16.18 | 15.52 | 14.75 |
| Titans + nhanh cam bien | **12.62** | 19.45 | 12.19 | 17.63 | 15.47 |
| *STA-HPINN — so cong bo (khong tai lap duoc)* | *11.27* | *13.21* | *8.30* | *13.31* | *11.52* |

**Ket luan**
- Miras tot nhat (**Memora**) kem DCNN 0,7 RMSE TB, nhung **khong khac co y nghia** tren
  FD001–FD003 (bootstrap ghep cap + Holm); chi FD004 kem co y nghia (+0,81).
- Tren bo nhieu che do van hanh (FD002, FD004) Memora nhinh hon STA-HPINN tai tao
  (−1,64 / −2,00 RMSE, CI 95% khong chua 0 nhung het y nghia sau Holm); tren bo don
  che do (FD001, FD003) thi kem co y nghia (+1,95 / +2,71).
- **Xap xi chunk khong phai nguyen nhan**: truy hoi chinh xac (chunk 1) khong tot hon
  chunk 5 hay 20 (chenh ≤ 0,6).
- Y tuong hoc tu TSHAE/STA-HPINN: bo nho hang thap ≈ khong doi; nut that + triplet
  **lam te di** (+1,2); nhanh Miras tren truc cam bien **tot nhat FD001/FD003** (ngang DCNN)
  nhung **sup tren FD002/FD004** — hong khi du lieu da che do.
- Dau BiLSTM cua paper 1 **lam te** Titans (13,77 → 14,58).

## Trien khai Miras

`code/miras.py` — mot lop chung, cac bien the la lua chon cua 4 truc Miras:

| bien the | attentional bias | retention gate |
|---|---|---|
| `linear_attn` / `mamba2` | dot product (Hebbian) | khong / alpha_t |
| `deltanet` / `gated_deltanet` | l2 (delta rule) | khong / alpha_t |
| `titans` | l2 + momentum | alpha_t |
| `moneta` | l_p (p=3) | chuan hoa l_q (q=4) |
| `yaad` | Huber | alpha_t |
| `memora` | l2 | KL / softmax |
| `elastic`, `robust`, `retnet`, `titans_mlp` | xem docstring | |

Tuy chon them (09-2026): `rank` (bo nho hang thap), `sensor_branch` (nhanh tren truc
cam bien, khong nhan qua), `bottleneck` + triplet. Huan luyen song song theo chunk
(§5.4); `chunk=1` = truy hoi chinh xac.

**Loi da sua 09-2026** (ket qua cu truoc ngay nay cua Memora, Titans, nhanh cam bien
khong con gia tri): Memora di len gradient (sai dau); Titans mat momentum qua ranh gioi
chunk; nhanh cam bien dung mask nhan qua tren truc khong thu tu; bo `rank_bias`.

## Pipeline so sanh

14 sensor kinh dien, cua so 40 (FD001) / 60, min-max (FD001/FD003), chuan hoa theo 6 che
do van hanh (FD002/FD004), nhan cat 125 (**tach** nguong huan luyen khoi nguong danh gia),
20% engine train lam val, dung som tren val, test chi mo mot lan.
Kiem dinh: bootstrap tren **engine** (don vi lay mau dung), khong phai Wilcoxon tren
tung mau.

## Chay

```bash
bash scripts/fetch_cmapss.sh                    # -> data/CMAPSSData
cd code
python train_miras_fix.py --root ../data/CMAPSSData --outdir ./out \
    --subsets FD001 --ideas memora,titans,dcnn --chunks 5 --seeds 0,1,2
python report_mfix.py --dir ./out
```

Tren cum GPU: `code/run_miras_fix.sh` (3 nhom song song, ~640 run, ~10 h tren V100 dung chung).

## Boi canh (cac giai doan truoc)

Du an bat dau bang viec tai lap SBi-Transformer (Ren et al., *Results in Engineering*
2026) — **khong tai lap duoc**, so cong bo nam ngoai CI 95% ca 4 bo; tai tao STA-HPINN
(arXiv:2405.12377) — thanh phan physics-informed lam te di; 7 baseline van lieu;
autoresearch 3 tang. Tong cong ~2 900 lan huan luyen. Xem [`SUMMARY.md`](SUMMARY.md),
`results/*.md`, toan bo run trong [`results/all_runs.csv`](results/all_runs.csv).

| file | noi dung |
|---|---|
| `code/data.py` | doc C-MAPSS, chuan hoa, cua so, nhan |
| `code/miras.py` | ho Miras |
| `code/sta_hpinn.py`, `code/rve.py`, `code/baselines.py`, `code/model.py` | SOTA, TSHAE, baseline, SBi-Transformer |
| `code/stats.py` | CI bootstrap tren engine, so sanh ghep cap, Holm |
| `code/autoresearch.py`, `code/ablate.py` | tim kiem tu dong, ablation co lap |
