"""C-MAPSS evaluation metrics (numpy only, so report scripts run without torch)."""
import numpy as np


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def score(pred, true):
    """PHM08 asymmetric score: late predictions (pred > true) are penalized harder."""
    r = pred - true
    return float(np.sum(np.where(r > 0, np.exp(r / 10.0) - 1.0, np.exp(-r / 13.0) - 1.0)))
