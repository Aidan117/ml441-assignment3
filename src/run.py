"""Final experiment: every dataset x model, on held-out folds only.

Hyperparameters come from tuning, which used the first 60% of each series.
Every test fold here falls in the remaining 40%, so no observation that
influenced a hyperparameter choice is ever scored.
"""
from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch

import data
import pipeline as P
import tuning
from common import RESULTS_DIR
from models import MODELS

HOLDOUT_START = 0.6     # test folds live after this fraction of the series
N_SPLITS = 5
SEEDS = (0, 1, 2, 3, 4)

# Full electricity is 17,520 hourly points; the last year is plenty and keeps
# the run affordable. Stated in the report's empirical process section.
MAX_POINTS = {"electricity": 8760}


def merge_tuning() -> pd.DataFrame:
    """Combine per-dataset best_*.csv into one table."""
    parts = sorted(RESULTS_DIR.glob("best_*.csv"))
    if not parts:
        return pd.read_csv(RESULTS_DIR / "tuning_best.csv")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "model"], keep="last")
    df.to_csv(RESULTS_DIR / "tuning_best.csv", index=False)
    return df


def series_for(dataset: str) -> np.ndarray:
    raw = data.DATASETS[dataset]().to_numpy()
    cap = MAX_POINTS.get(dataset)
    return raw[-cap:] if cap and len(raw) > cap else raw


def final_config(dataset: str, model: str, best: pd.DataFrame) -> P.Config:
    return tuning.tuned_config(
        dataset, model, best_df=best,
        n_splits=N_SPLITS, max_epochs=200, patience=15,
    )


def collect_predictions(raw, dataset, model, cfg, seed=0):
    """Re-run the last held-out fold, keeping the predictions for plotting."""
    import transforms
    T = transforms.get(cfg.transform)
    tr, te = P.cv_splits(P.n_windows(raw, cfg), cfg, HOLDOUT_START)[-1]

    from common import RANDOM_STATE, set_seed
    from models import build_model

    set_seed(RANDOM_STATE + seed)
    work = T.to_work(raw)
    X, y = P.make_windows(work, cfg.window, cfg.horizon)
    Xtr_w, ytr_w, Xte_w = X[tr], y[tr], X[te]
    mu, sd = float(Xtr_w.mean()), float(Xtr_w.std()) or 1.0
    sc = lambda a: (a - mu) / sd
    cut = max(1, int(len(Xtr_w) * (1 - cfg.val_frac)))
    t = torch.as_tensor
    model_obj = build_model(model, cfg.n_hidden)
    P._train_one(model_obj, t(sc(Xtr_w)[:cut]), t(sc(ytr_w)[:cut]),
                 t(sc(Xtr_w)[cut:]), t(sc(ytr_w)[cut:]), cfg, "cpu")
    model_obj.eval()
    with torch.no_grad():
        pred_w = model_obj(t(sc(Xte_w))).numpy() * sd + mu

    tgt = P.target_index(te, cfg)
    anchor = T.anchors(raw, tgt, cfg.horizon)[:, None]
    return pd.DataFrame({
        "dataset": dataset, "model": model, "step": np.arange(len(te)),
        "actual": T.true_values(raw, tgt),
        "predicted": T.invert(pred_w, anchor).ravel(),
        "naive": anchor.ravel(),
    })


def main(datasets=None, models=MODELS, with_predictions=True):
    best = merge_tuning()
    datasets = datasets or sorted(best.dataset.unique())
    rows, preds = [], []

    for ds in datasets:
        raw = series_for(ds)
        for m in models:
            cfg = final_config(ds, m, best)
            t0 = time.time()
            r = P.evaluate(raw, m, cfg, seeds=SEEDS, start_frac=HOLDOUT_START)
            for x in r:
                x["dataset"] = ds
            rows.extend(r)
            u = np.mean([x["theil_u"] for x in r])
            print(f"{ds:14s} {m:7s}  U={u:.4f}  "
                  f"window={cfg.window:<4d} hidden={cfg.n_hidden:<3d} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if with_predictions:
                preds.append(collect_predictions(raw, ds, m, cfg))

        # per-dataset filenames so parallel invocations do not clash
        tag = f"_{datasets[0]}" if len(datasets) == 1 else ""
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"final_runs{tag}.csv", index=False)  # save as we go
        if preds:
            pd.concat(preds, ignore_index=True).to_csv(
                RESULTS_DIR / f"final_predictions{tag}.csv", index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--no-predictions", action="store_true")
    args = ap.parse_args()
    main(datasets=args.datasets, with_predictions=not args.no_predictions)
    print(f"\nWrote {RESULTS_DIR/'final_runs.csv'}")