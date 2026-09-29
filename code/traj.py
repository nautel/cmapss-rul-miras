"""Export RUL predictions along whole engine lives, for qualitative plots.

Two sets per trained model:
  test : every sliding window of every test engine (engines are truncated before
         failure; true RUL at each cycle comes from the RUL_FDxxx offset)
  val  : every window of the held-out validation engines — run-to-failure and never
         seen in training, so the curve goes all the way down to RUL 0
"""
import numpy as np

import data as D


def windows_along_life(d):
    L = d["seq_len"]
    Xte, ute, _ = d["raw_test"]
    X, y, u, cyc, t = [], [], [], [], []
    for unit in np.unique(ute):
        n = int((ute == unit).sum())
        xs, ys = D.test_trajectory(d, unit)
        k = len(xs)
        X.append(xs)
        y.append(ys)
        u.append(np.full(k, unit))
        # cycle at the end of each window; engines shorter than L give one padded window
        cyc.append(np.arange(k) + L if n >= L else np.array([n]))
        t.append(np.arange(k, dtype=np.float32) + L)          # same convention as training
    test = dict(X=np.concatenate(X), y=np.concatenate(y), unit=np.concatenate(u),
                cycle=np.concatenate(cyc), t=np.concatenate(t) / d["tmax"])
    val = dict(X=d["Xval"], y=d["yval"], unit=d["uval"], cycle=d["cval"], t=d["tval"])
    return test, val


def export(path, d, predict_fn):
    """`predict_fn(X, t) -> RUL in cycles`. Writes one compressed npz."""
    out = {}
    for name, s in zip(("test", "val"), windows_along_life(d)):
        out[f"{name}_pred"] = predict_fn(s["X"], s["t"]).astype(np.float32)
        out[f"{name}_true"] = s["y"].astype(np.float32)
        out[f"{name}_unit"] = s["unit"].astype(np.int32)
        out[f"{name}_cycle"] = s["cycle"].astype(np.int32)
    np.savez_compressed(path, **out)
