"""Windowing, leakage-safe cross-validation, training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit

from common import RANDOM_STATE, set_seed
from models import build_model
import transforms


@dataclass
class Config:
    window: int = 20           # lags fed to the network
    horizon: int = 1           # steps ahead to forecast
    n_splits: int = 5          # expanding-window CV folds
    n_hidden: int = 8
    lr: float = 0.01
    max_epochs: int = 300
    patience: int = 20         # early-stopping patience, in epochs
    batch_size: int = 64
    val_frac: float = 0.2      # tail of each training fold, for early stopping
    weight_decay: float = 0.0
    transform: str = "none"    # "none" | "log1p" | "logdiff"


def make_windows(x: np.ndarray, window: int, horizon: int):
    """x -> X (n, window, 1), y (n, 1). Row i predicts x[i+window+horizon-1]."""
    n = len(x) - window - horizon + 1
    if n <= 0:
        raise ValueError("series too short for this window/horizon")
    X = np.lib.stride_tricks.sliding_window_view(x, window)[:n]
    y = x[window + horizon - 1: window + horizon - 1 + n]
    return X[..., None].astype(np.float32), y[:, None].astype(np.float32)

def target_index(row_idx: np.ndarray, cfg: Config) -> np.ndarray:
    """Window row -> index of its target in the work series."""
    return np.asarray(row_idx) + cfg.window + cfg.horizon - 1


def n_windows(raw: np.ndarray, cfg: Config) -> int:
    work = transforms.get(cfg.transform).to_work(raw)
    return len(work) - cfg.window - cfg.horizon + 1

def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> dict:
    """RMSE, MAE and the ratio to a persistence forecast (<1 beats naive)."""
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    rmse_naive = float(np.sqrt(np.mean((y_naive - y_true) ** 2)))
    mae_naive = float(np.mean(np.abs(y_naive - y_true)))
    return {
        "rmse": rmse,
        "mae": mae,
        "rmse_naive": rmse_naive,
        "theil_u": rmse / rmse_naive if rmse_naive > 0 else np.nan,
        "mase": mae / mae_naive if mae_naive > 0 else np.nan,
    }


def _train_one(model, Xtr, ytr, Xva, yva, cfg: Config, device):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    n = len(Xtr)
    best, best_state, bad, best_epoch = np.inf, None, 0, 0

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(Xva), yva))
        if val < best - 1e-6:
            best, best_epoch, bad = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)      # restore best-validation weights
    return {"val_loss": best, "best_epoch": best_epoch, "epochs_run": epoch + 1}


def run_fold(raw: np.ndarray, model_name: str, cfg: Config, seed: int,
             tr_idx: np.ndarray, te_idx: np.ndarray, device="cpu") -> dict:
    """Train and evaluate one model on one CV fold.

    The network trains in the transformed space, but predictions are inverted
    before scoring, so every reported error is in the series' original units.
    Scaling is fitted on the training fold only, and the last val_frac of that
    fold is held out for early stopping, so the test fold is never seen during
    training or model selection.
    """
    set_seed(seed)
    T = transforms.get(cfg.transform)
    work = T.to_work(raw)
    X, y = make_windows(work, cfg.window, cfg.horizon)

    Xtr_w, ytr_w = X[tr_idx], y[tr_idx]
    Xte_w = X[te_idx]

    mu = float(Xtr_w.mean())
    sd = float(Xtr_w.std()) or 1.0
    scale = lambda a: (a - mu) / sd

    Xtr, ytr, Xte = scale(Xtr_w), scale(ytr_w), scale(Xte_w)

    cut = max(1, int(len(Xtr) * (1 - cfg.val_frac)))
    t = lambda a: torch.as_tensor(a, device=device)
    Xtr_t, ytr_t = t(Xtr[:cut]), t(ytr[:cut])
    Xva_t, yva_t = t(Xtr[cut:]), t(ytr[cut:])

    model = build_model(model_name, cfg.n_hidden).to(device)
    hist = _train_one(model, Xtr_t, ytr_t, Xva_t, yva_t, cfg, device)

    model.eval()
    with torch.no_grad():
        pred_w = model(t(Xte)).cpu().numpy()
    pred_w = pred_w * sd + mu                      # undo scaling, still in work space

    # back to original units
    tgt = target_index(te_idx, cfg)
    anchor = T.anchors(raw, tgt, cfg.horizon)[:, None]
    y_true = T.true_values(raw, tgt)[:, None]
    y_pred = T.invert(pred_w, anchor)
    y_naive = anchor                               # persistence, in original units

    out = {"model": model_name, "seed": seed, "transform": cfg.transform,
           "n_train": len(Xtr_t), "n_test": len(te_idx)}
    out.update(hist)
    out.update(metrics(y_true, y_pred, y_naive))
    return out


def cv_splits(n_samples: int, cfg: Config, start_frac: float = 0.0):
    """Expanding-window folds with a gap of `window` samples.

    The gap stops the last training window and the first test window sharing
    lagged values, which otherwise makes test error optimistic.

    start_frac > 0 confines every test fold to the tail of the series, after
    that fraction. Training still expands from the very start, so the held-out
    evaluation never scores a window that was available during tuning.
    """
    if start_frac <= 0.0:
        tscv = TimeSeriesSplit(n_splits=cfg.n_splits, gap=cfg.window)
        return list(tscv.split(np.zeros((n_samples, 1))))

    start = int(n_samples * start_frac)
    chunk = (n_samples - start) // cfg.n_splits
    if chunk < 1:
        raise ValueError("not enough held-out samples for this many folds")
    splits = []
    for f in range(cfg.n_splits):
        te0 = start + f * chunk
        te1 = n_samples if f == cfg.n_splits - 1 else te0 + chunk
        tr_end = te0 - cfg.window
        if tr_end < 1:
            raise ValueError("gap leaves no training data")
        splits.append((np.arange(0, tr_end), np.arange(te0, te1)))
    return splits


def evaluate(raw: np.ndarray, model_name: str, cfg: Config,
             seeds=(0, 1, 2, 3, 4), device="cpu",
             start_frac: float = 0.0) -> list[dict]:
    """All folds x all seeds for one model on one series."""
    raw = np.asarray(raw, dtype=float)
    rows = []
    for fold, (tr, te) in enumerate(cv_splits(n_windows(raw, cfg), cfg, start_frac)):
        for seed in seeds:
            r = run_fold(raw, model_name, cfg, RANDOM_STATE + seed, tr, te, device)
            r["fold"] = fold
            r.update({f"cfg_{k}": v for k, v in asdict(cfg).items()})
            rows.append(r)
    return rows