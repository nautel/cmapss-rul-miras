"""C-MAPSS loading + preprocessing theo muc 3.1 / 4.2 cua paper.

Paper: Ren et al., "A transformer-based method for aircraft engine RUL prediction
integrating dual-layer attention with BiLSTM", Results in Engineering 29 (2026) 109187.

Pipeline: Z-score -> PCA/loc tuong quan -> 14 sensor + 3 op setting = 17 feature
(khop `input_size = 17` o Table 2), cua so truot seq_len=45, nhan RUL cat nguong 125.
"""
import os
import numpy as np

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
# 14 sensor "co lien he manh voi suy giam" (muc 4.2). Doc TRUC TIEP tu chu giai
# Fig. 4(b) (engine #30, FD001) va Fig. 4(d) (engine #51, FD002) cua paper:
#   (b) 12 13 7 21 2 20 9 | 11 4 14 17 15 3 6
#   (d)  9 11 12 2 3 21 13 | 4 15 7 20 17 8 6
# Hai tap khac nhau: FD001 co s14 khong co s8; FD002 co s8 khong co s14. Ca hai deu co s6.
PAPER14 = {
    "FD001": [2, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 17, 20, 21],
    "FD002": [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21],
}
PAPER14["FD003"] = PAPER14["FD001"]   # paper gop FD001+FD003 (Table 2 cot "FD0013")
PAPER14["FD004"] = PAPER14["FD002"]   # va FD002+FD004 (cot "FD0024")
# Tap 14 sensor kinh dien trong tai lieu C-MAPSS — de doi chieu (--feature-mode classic14)
CLASSIC14 = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
OPS = ["op1", "op2", "op3"]
RUL_CAP = 125.0


def load_raw(root, subset):
    tr = np.loadtxt(os.path.join(root, f"train_{subset}.txt"))
    te = np.loadtxt(os.path.join(root, f"test_{subset}.txt"))
    rul = np.loadtxt(os.path.join(root, f"RUL_{subset}.txt")).reshape(-1)
    return tr, te, rul


def piecewise_rul(n_cycles, cap=RUL_CAP, offset=0.0):
    """RUL tuyen tinh tung khuc: hang so `cap` luc dau, giam tuyen tinh ve cuoi."""
    r = np.arange(n_cycles - 1, -1, -1, dtype=np.float64) + offset
    return np.minimum(r, cap)


def _col(name):
    return COLS.index(name)


def select_features(tr, mode="paper", subset="FD001", n_pca=5, thr=0.35):
    """Tra ve danh sach ten cot dung lam input.

    mode="paper"     : 3 op setting + 14 sensor doc tu chu giai Fig. 4 (input_size=17).
    mode="classic14" : 3 op setting + tap 14 sensor kinh dien.
    mode="auto"      : lam lai buoc cua paper — Z-score, PCA, roi giu sensor co
                       |rho| lon voi quy dao suy giam (Eq. 3).
    """
    if mode == "paper":
        return OPS + [f"s{i}" for i in PAPER14[subset]]
    if mode == "classic14":
        return OPS + [f"s{i}" for i in CLASSIC14]
    units = tr[:, 0]
    sens = [c for c in COLS if c.startswith("s")]
    X = tr[:, [_col(c) for c in sens]]
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-8
    Xn = np.zeros_like(X)
    Xn[:, keep] = (X[:, keep] - mu[keep]) / sd[keep]
    # PCA (Eq. 2) tren phan bien thien — dung de xac nhan chieu chinh, khong dung de chieu input
    S = Xn[:, keep].T @ Xn[:, keep] / len(Xn)
    w, Q = np.linalg.eigh(S)
    order = np.argsort(w)[::-1][:n_pca]
    _ = Xn[:, keep] @ Q[:, order]
    # y = RUL that cua tung dong trong train
    y = np.concatenate([piecewise_rul(int((units == u).sum())) for u in np.unique(units)])
    rho = np.zeros(len(sens))
    for j in range(len(sens)):
        if not keep[j]:
            continue
        rho[j] = np.corrcoef(Xn[:, j], y)[0, 1]
    sel = [sens[j] for j in range(len(sens)) if keep[j] and abs(rho[j]) >= thr]
    return OPS + sel


def condition_id(ops):
    """6 che do van hanh cua FD002/FD004: op1,op2 lam tron la du de tach."""
    key = np.round(ops[:, 0], 0) * 1000 + np.round(ops[:, 1], 2) * 10 + np.round(ops[:, 2], 0)
    uniq = np.unique(key)
    return np.searchsorted(uniq, key), len(uniq)


def build(root, subset, seq_len=45, feature_mode="paper", cond_norm=False,
          val_frac=0.1, seed=0, cap=RUL_CAP, eval_cap=RUL_CAP, norm="zscore"):
    """Tra ve dict cac tensor numpy da san sang cho training.

    `cap`      : nguong cat RUL cho NHAN HUAN LUYEN — la sieu tham so hop le.
    `eval_cap` : nguong cat cho NHAN DANH GIA (val + test) — phai CO DINH 125.

    Hai thu nay bat buoc phai tach roi. Neu de chung mot `cap`, thay doi nguong se
    thay doi luon nhan test, tuc thay doi THUOC DO: engine co RUL that cao (kho nhat)
    bi ha nhan xuong sat vung mo hinh doan tot. Do 22-08: cap 110 ha nhan cua 28-85
    engine moi bo (trung binh 2,6-4,6 RUL) va lam RMSE giam gia tao ~3 diem.
    """
    tr, te, rul_te = load_raw(root, subset)
    feats = select_features(tr, mode=feature_mode, subset=subset)
    fidx = [_col(c) for c in feats]

    Xtr_raw, Xte_raw = tr[:, fidx], te[:, fidx]
    if cond_norm:
        ctr, ncond = condition_id(tr[:, 2:5])
        cte, _ = condition_id(te[:, 2:5])
        Xtr, Xte = np.zeros_like(Xtr_raw), np.zeros_like(Xte_raw)
        for c in range(ncond):
            m = ctr == c
            mu, sd = Xtr_raw[m].mean(0), Xtr_raw[m].std(0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            Xtr[m] = (Xtr_raw[m] - mu) / sd
            mt = cte == c
            if mt.any():
                Xte[mt] = (Xte_raw[mt] - mu) / sd
    elif norm == "minmax":
        # min-max ve [0,1] — cach cua STA-HPINN (arXiv:2405.12377) va nhieu bai C-MAPSS
        lo, hi = Xtr_raw.min(0), Xtr_raw.max(0)
        rng_ = np.where(hi - lo < 1e-8, 1.0, hi - lo)
        Xtr = (Xtr_raw - lo) / rng_
        Xte = np.clip((Xte_raw - lo) / rng_, -0.5, 1.5)
    else:
        mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)   # Eq. 1, thong ke chi tu tap train
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xtr = (Xtr_raw - mu) / sd
        Xte = (Xte_raw - mu) / sd

    def windows(X, units, targets, only_last=False):
        xs, ys, us, ts = [], [], [], []
        for u in np.unique(units):
            m = units == u
            xu, yu = X[m], targets[m]
            n = len(xu)
            if n < seq_len:   # engine test ngan hon cua so -> lap dong dau (left-pad)
                pad = np.repeat(xu[:1], seq_len - n, axis=0)
                xu = np.concatenate([pad, xu], 0)
                yu = np.concatenate([np.repeat(yu[:1], seq_len - n), yu])
                n = seq_len
            starts = [n - seq_len] if only_last else range(n - seq_len + 1)
            for s in starts:
                xs.append(xu[s:s + seq_len])
                ys.append(yu[s + seq_len - 1])
                us.append(u)
                ts.append(s + seq_len)          # so chu ky da troi qua o cuoi cua so
        return (np.asarray(xs, np.float32), np.asarray(ys, np.float32),
                np.asarray(us, np.int64), np.asarray(ts, np.float32))

    utr = tr[:, 0]
    ytr_row = np.concatenate([piecewise_rul(int((utr == u).sum()), cap)
                              for u in np.unique(utr)])
    yev_row = np.concatenate([piecewise_rul(int((utr == u).sum()), eval_cap)
                              for u in np.unique(utr)])
    Xw, yw, uw, tw = windows(Xtr, utr, ytr_row)
    _, yw_ev, _, _ = windows(Xtr, utr, yev_row)       # nhan val: luon eval_cap

    ute = te[:, 0]
    # nhan test: RUL con lai o chu ky cuoi = rul_te[i]; nguoc ve dau chuoi thi cong don
    yte_row = np.concatenate([piecewise_rul(int((ute == u).sum()), eval_cap,
                                            offset=rul_te[i])
                              for i, u in enumerate(np.unique(ute))])
    Xt, yt, ut, tt = windows(Xte, ute, yte_row, only_last=True)

    # tach validation theo ENGINE (khong tron cua so cua cung engine qua 2 phia)
    rng = np.random.RandomState(seed)
    eng = np.unique(utr)
    perm = rng.permutation(len(eng))
    n_val = max(1, int(round(val_frac * len(eng))))
    val_eng = set(eng[perm[:n_val]].tolist())
    vm = np.array([u in val_eng for u in uw])

    return dict(
        subset=subset, features=feats, seq_len=seq_len,
        Xtr=Xw[~vm], ytr=yw[~vm], Xval=Xw[vm], yval=yw_ev[vm],
        Xte=Xt, yte=yt, ute=ut,
        # chu ky (chuan hoa) — input `t` cua AHPINN; thang chia lay tu tap train
        ttr=tw[~vm] / tw.max(), tval=tw[vm] / tw.max(), tte=tt / tw.max(),
        # toan bo cua so cua tung engine test — dung cho hinh 5 (quy dao RUL)
        raw_test=(Xte, ute, yte_row),
        cap=cap, eval_cap=eval_cap,
    )


def test_trajectory(d, unit):
    """Moi cua so truot cua 1 engine test -> (X, y_true) de ve quy dao Fig. 5."""
    Xte, ute, yte_row = d["raw_test"]
    L = d["seq_len"]
    m = ute == unit
    xu, yu = Xte[m], yte_row[m]
    n = len(xu)
    if n < L:
        xu = np.concatenate([np.repeat(xu[:1], L - n, 0), xu], 0)
        yu = np.concatenate([np.repeat(yu[:1], L - n), yu])
        n = L
    xs = np.stack([xu[s:s + L] for s in range(n - L + 1)]).astype(np.float32)
    ys = np.array([yu[s + L - 1] for s in range(n - L + 1)], np.float32)
    return xs, ys
