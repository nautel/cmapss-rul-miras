# Tong ket — tai lap 2 paper, tai tao SOTA, va tim huong cho bai toan RUL

**1152 lan huan luyen** tren `cassio.gi.utc` (Tesla V100 dung chung). Bao cao truc quan:
https://claude.ai/code/artifact/99d6257f-6de9-4c26-8f97-72ca7a88e941

Chi tiet tung phan: [`results/REPRODUCTION.md`](results/REPRODUCTION.md) ·
[`results/MIRAS.md`](results/MIRAS.md) · [`results/AUTORESEARCH2.md`](results/AUTORESEARCH2.md) ·
[`results/ALL_METHODS.md`](results/ALL_METHODS.md) · [`results/STATS_BASELINES.md`](results/STATS_BASELINES.md)

---

## 0. Cap nhat 09-2026 — Miras sau khi sua loi

Chi tiet: [`results/MIRAS_FIX.md`](results/MIRAS_FIX.md) (640 run, cung pipeline, 10 seed).
Sua 3 loi trong `code/miras.py` (Memora sai dau, Titans mat momentum qua chunk, nhanh cam
bien nhan qua) — cac so Miras o muc 1 ben duoi la TRUOC khi sua.

| | TB RMSE |
|---|---|
| DCNN (moc) | **12,50** |
| STA-HPINN bo physics | 12,95 |
| Miras tot nhat — Memora | 13,20 (khong khac DCNN co y nghia tren FD001–FD003) |
| Titans / Titans hang 4 | 13,77 |
| 5 bien the Miras khac | 13,98 – 14,75 |

Xap xi chunk khong phai nguyen nhan (chunk 1 / 5 / 20 chenh <= 0,6). Nut that + triplet
lam te; nhanh cam bien tot tren FD001/FD003 nhung sup tren FD002/FD004.

---

## 1. Bang xep hang (RMSE, CI 95% bootstrap tren engine)

| phuong phap | nguon | FD001 | FD002 | FD003 | FD004 | TB | seed |
|---|---|---|---|---|---|---|---|
| STA-HPINN, **bo** physics loss | SOTA 2024, sua | 12,22 | 13,72 | 10,29 | 14,05 | **12,57** | 5 |
| DCNN | Li 2018 | 13,12 | 12,71 | 12,01 | 13,60 | **12,86** | 10 |
| Autoresearch (tim duoc) | nghien cuu nay | 14,14 | 12,63 | 12,63 | 13,70 | **13,28** | 5 |
| GRU | van lieu | 14,08 | 13,52 | 12,80 | 14,15 | 13,64 | 10 |
| Transformer + sparse attn | paper 1, ablation | 13,59 | 12,96 | 14,11 | 14,53 | 13,80 | 10 |
| STA-HPINN (day du) | SOTA 2024 | 13,40 | 14,56 | 11,69 | 15,91 | 13,89 | 10 |
| SBi-Transformer | paper 1 | 14,64 | 13,06 | 14,51 | 14,62 | 14,21 | 10 |
| MLP duoi phang (can duoi) | — | 14,43 | 14,77 | 12,86 | 15,27 | 14,33 | 10 |
| 10 bien the Miras | paper 2 | — | — | — | — | 14,87–15,50 | 3 |
| *SBi-Transformer — **cong bo*** | paper 1 | *11,37* | *12,05* | *11,13* | *11,18* | *11,43* | — |
| *STA-HPINN — **cong bo*** | SOTA 2024 | *11,27* | *13,21* | *8,30* | *13,31* | *11,52* | — |

Ca 29 phuong phap nam trong dai 3,5 RMSE, trong khi CI cua **mot o** da rong **±2,4**.

## 2. Bon ket qua

**a. Paper 1 khong tai lap duoc** — so cong bo nam NGOAI CI 95% tren ca 4 bo con.
Ba gia thuyet ve nguyen nhan da kiem chung bang thuc nghiem: dung som *khong* phai
(600 epoch khong hon 80); cua so 30 theo Table 1 *te hon* 45 theo Table 2; thieu chuan
hoa theo che do van hanh la that nhung chi va duoc FD002 (−24%) va FD004 (−22%).

**b. Ablation cua paper 1 dao nguoc** — paper cho day giam don dieu 18,48 → 11,43;
chay lai du epoch cho ca 5 cau hinh nam trong 14,94–15,65 va mo hinh **day du te nhat**.
Vong tim kiem tu dong, doc lap, cung chon cau hinh `yes_yes_no` — tuc **bo BiLSTM**.

**c. Physics loss cua SOTA lam hai** — bo han nhanh vat ly cua STA-HPINN (dong gop
mang ten bai bao) cho ket qua **tot hon 1,32 RMSE** va nhanh gap doi. Ban day du con
te hon DCNN 2018 mot cach co y nghia tren FD002 (+1,85, p = 0,002).

**d. Tien xu ly quan trong hon kien truc, gap 3,4 lan** — ablation co lap tren cau hinh
tot nhat, doi tung yeu to ve gia tri paper:

| yeu to tra ve gia tri paper | RMSE xau di |
|---|---|
| chuan hoa theo che do van hanh | **+1,37** |
| kien truc (bo BiLSTM) | +0,40 |
| cua so 60 → 45 | +0,39 |
| weight decay | +0,20 |
| chieu FFN / chieu an | +0,09 |
| dropout / so lop encoder | −0,02 / −0,03 |

## 3. Ba cay bay ve phuong phap (deu la loi cua chinh quy trinh nay)

1. **Thuoc do bi dua vao khong gian tim kiem.** `rul_cap` doi ca NHAN TEST: o nguong 110,
   28–85 engine moi bo bi ha nhan — dung nhung engine RUL cao, kho nhat. Vong tim kiem
   khai thac ngay: bao 10,63 nhung do lai bang thuoc do chuan la 13,70. Sau khi tach
   nguong huan luyen khoi nguong danh gia, thuat toan **tu chon lai** dung 125.
2. **Wilcoxon tren sai so tung mau** cho p = 1e-04 den 1e-59 cho MOI cap, ke ca cap ma
   bootstrap noi khong phan biet duoc — vi coi 100 engine × 10 seed la doc lap. Don vi
   lay mau dung la ENGINE.
3. **Ket luan tu luoi chay do.** Hai lan ket luan o 48/96 run, ca hai lan deu sai khi du.

## 4. Con thieu gi de cong bo

Phan tai lap paper 1 gan du cho mot *reproducibility study* — **4 mau thuan noi tai** doc
duoc tu chinh bang so cua ho, khong can tin thi nghiem nao: Table 1 nghich Table 2 (ba do
dai cua so khac nhau), `epochs=600` nghich `patience=10` (do duoc: dung o epoch 22–24),
Eq. (14) cho CI do phu ~0, va Table 6 nghich kich thuoc du lieu (FD002 48.819 mau chay
*nhanh hon* FD001 17.731 mau).

Con thieu: lien he tac gia truoc khi cong bo; mo rong sang N-CMAPSS hoac du lieu that;
them seed cho cac o hien moi co 3–5. Rieng phan "phuong phap moi" thi chua co gi —
sau khi sua thuoc do, cau hinh tim duoc chi ngang ban tai lap va thua DCNN 2018.
