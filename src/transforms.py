"""Per-dataset preprocessing transforms, with inverses.

Networks train in a transformed space (log, log-difference) but every reported
error is computed in the original units, so RMSE stays interpretable and
comparable to the persistence baseline.

Index convention: `t` indexes the *work* (transformed) series and identifies
the target of one window. Each transform maps that back to the raw series.
"""
from __future__ import annotations

import numpy as np


class Identity:
    """No transform. Train and score on the raw series."""
    name = "none"

    def to_work(self, x):          return np.asarray(x, dtype=float)
    def true_values(self, x, t):   return x[t]
    def anchors(self, x, t, h):    return x[t - h]      # last value in the input window
    def invert(self, pred, anchor): return pred


class Log1p(Identity):
    """log(1+x) for strictly positive, heavily skewed counts."""
    name = "log1p"

    def to_work(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x < 0):
            raise ValueError("log1p transform requires non-negative values")
        return np.log1p(x)

    def invert(self, pred, anchor): return np.expm1(pred)


class LogDiff:
    """First difference of log levels, i.e. log returns.

    Removes a unit root. A prediction of the return is converted back to a
    level with the last observed price as the anchor:  p_hat = p_prev * exp(r).
    """
    name = "logdiff"

    def to_work(self, x):
        x = np.asarray(x, dtype=float)
        if np.any(x <= 0):
            raise ValueError("logdiff transform requires strictly positive values")
        return np.diff(np.log(x))

    # work index t corresponds to the step from raw x[t] to raw x[t+1]
    def true_values(self, x, t):   return x[t + 1]
    def anchors(self, x, t, h):    return x[t]
    def invert(self, pred, anchor): return anchor * np.exp(pred)


TRANSFORMS = {t.name: t for t in (Identity(), Log1p(), LogDiff())}

# Chosen per dataset from the stationarity tests; see the report's
# data pre-processing subsection for the justification of each.
DATASET_TRANSFORM = {
    "mackey_glass": "none",      # bounded, stationary, no skew
    "temperature":  "none",      # stationary in level; seasonality left to the network
    "electricity":  "none",      # stationary in level; daily and weekly cycles
    "pageviews":    "log1p",     # spikes ~80x baseline; stabilises variance
    "sp500":        "logdiff",   # unit root present (ADF p = 0.997)
}


def get(name: str):
    if name not in TRANSFORMS:
        raise KeyError(f"unknown transform {name!r}; have {sorted(TRANSFORMS)}")
    return TRANSFORMS[name]