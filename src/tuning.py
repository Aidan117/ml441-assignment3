"""Hyperparameter search, one study per (dataset, model).

Selection is done on a development prefix of each series; the final runs score
on folds drawn entirely from the held-out remainder. So no observation used to
choose hyperparameters is ever used to report performance.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import optuna
import pandas as pd

import data
import pipeline as P
import transforms
from common import RANDOM_STATE, RESULTS_DIR
from models import MODELS

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEV_FRAC = 0.6          # prefix used for tuning
MAX_TUNE_POINTS = 4000  # cap on the dev series, keeps electricity affordable

# Candidate lag windows, per dataset. Chosen to span the natural periods:
# a day and a week for hourly load, a week/month/year for daily series.
WINDOWS = {
    "mackey_glass": [10, 17, 25, 40],          # tau = 17
    "temperature":  [7, 14, 30, 60, 90, 120],
    # A weekly window (168) was excluded: unrolling a simple RNN over 168 steps
    # is both very slow and prone to vanishing gradients, so it is not a fair
    # test of these architectures. Daily periodicity is within reach.
    "electricity":  [24, 48, 72],              # one, two and three days
    "pageviews":    [7, 14, 30, 60],
    "sp500":        [5, 10, 20, 40],
}
HIDDEN = [2, 4, 8, 12, 16, 24, 32, 48, 64]
DECAY = [0.0, 1e-5, 1e-4, 1e-3]


def dev_series(raw: np.ndarray) -> np.ndarray:
    """The tuning portion: a prefix, capped for cost."""
    n = int(len(raw) * DEV_FRAC)
    s = raw[:n]
    return s[-MAX_TUNE_POINTS:] if len(s) > MAX_TUNE_POINTS else s


def base_config(dataset: str) -> P.Config:
    return P.Config(
        horizon=1, n_splits=3, max_epochs=60, patience=8,
        batch_size=128, val_frac=0.2,
        transform=transforms.DATASET_TRANSFORM[dataset],
    )


def objective_factory(raw_dev: np.ndarray, dataset: str, model: str):
    def objective(trial: optuna.Trial) -> float:
        cfg = replace(
            base_config(dataset),
            window=trial.suggest_categorical("window", WINDOWS[dataset]),
            n_hidden=trial.suggest_categorical("n_hidden", HIDDEN),
            lr=trial.suggest_float("lr", 1e-3, 3e-2, log=True),
            weight_decay=trial.suggest_categorical("weight_decay", DECAY),
        )
        rows = P.evaluate(raw_dev, model, cfg, seeds=(0,))
        u = float(np.mean([r["theil_u"] for r in rows]))
        return u if np.isfinite(u) else 1e6
    return objective


def tune_one(dataset: str, model: str, n_trials: int = 20,
             raw: np.ndarray | None = None) -> tuple[dict, pd.DataFrame]:
    if raw is None:
        raw = data.DATASETS[dataset]().to_numpy()
    raw_dev = dev_series(raw)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name=f"{dataset}-{model}")
    t0 = time.time()
    study.optimize(objective_factory(raw_dev, dataset, model), n_trials=n_trials)
    elapsed = time.time() - t0

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.insert(0, "dataset", dataset)
    df.insert(1, "model", model)

    best = dict(study.best_params)
    best.update({"dataset": dataset, "model": model,
                 "theil_u_dev": study.best_value,
                 "n_trials": n_trials, "seconds": round(elapsed, 1)})
    return best, df


def tune_all(n_trials: int = 20, datasets=None, models=MODELS) -> pd.DataFrame:
    datasets = datasets or list(data.DATASETS)
    best_rows, all_trials = [], []
    for ds in datasets:
        raw = data.DATASETS[ds]().to_numpy()
        for m in models:
            best, df = tune_one(ds, m, n_trials, raw=raw)
            print(f"{ds:14s} {m:7s}  U={best['theil_u_dev']:.4f}  "
                  f"window={best['window']:<4d} hidden={best['n_hidden']:<3d} "
                  f"lr={best['lr']:.4f}  ({best['seconds']:.0f}s)", flush=True)
            best_rows.append(best)
            all_trials.append(df)

    tag = f"_{datasets[0]}" if len(datasets) == 1 else ""
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(RESULTS_DIR / f"best{tag}.csv", index=False)
    pd.concat(all_trials, ignore_index=True).to_csv(
        RESULTS_DIR / f"trials{tag}.csv", index=False)
    return best_df


def tuned_config(dataset: str, model: str, best_df: pd.DataFrame | None = None,
                 **overrides) -> P.Config:
    """Rebuild a Config from the tuning results, for the final runs."""
    if best_df is None:
        best_df = pd.read_csv(RESULTS_DIR / "tuning_best.csv")
    row = best_df[(best_df.dataset == dataset) & (best_df.model == model)].iloc[0]
    cfg = replace(
        base_config(dataset),
        window=int(row.window), n_hidden=int(row.n_hidden),
        lr=float(row.lr), weight_decay=float(row.weight_decay),
    )
    return replace(cfg, **overrides) if overrides else cfg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    tune_all(n_trials=args.trials, datasets=args.datasets)
    print(f"\nWrote {RESULTS_DIR/'tuning_best.csv'} and tuning_trials.csv")