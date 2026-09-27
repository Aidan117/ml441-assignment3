"""Which past values do the forecasts depend on, and does a direct shortcut help?

Stage 1, sensitivity (diagnosis). For every dataset and architecture, measure
how strongly the forecast responds to each value in the input window: the
mean absolute gradient of the forecast with respect to each lagged input
(output sensitivity analysis). Only the first 60% of each series is used,
which the final runs never used as a test fold.

Stage 2, shortcut (intervention). For each dataset, the value at one lag
fixed from the origin of the data is fed directly into the hidden layer at
the final time step, next to the current observation, so the value reaches
the forecast in one step instead of travelling through the whole recurrence.
Everything else matches the final runs: tuned hyperparameters, folds, seeds.
The lags are listed in PHYSICAL_LAGS. The selection rule in stage 1 was
specified in advance but rejected after it flagged a lag on the control
series, so it no longer drives stage 2; its output is kept for the report.

The physical lags serve as a check on the rule, not as an input to it:
    Mackey-Glass  lag 17  the delay tau of the generating equation
    Electricity   lag 24  the daily cycle of hourly load

Lag j means the value j steps before the forecast target, so lag 1 is the
most recent observation.

Usage:
    python lag_study.py sensitivity --datasets mackey_glass electricity ...
    python lag_study.py select                          # after stage 1
    python lag_study.py shortcut --datasets mackey_glass --models jordan
    python lag_study.py analyse
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

import pipeline as P
import run
import transforms
from common import RANDOM_STATE, RESULTS_DIR, savefig, set_seed
from models import MODELS, build_model

ORDER = ["mackey_glass", "electricity", "pageviews", "temperature", "sp500"]
PHYSICAL_LAGS = {
    "mackey_glass": 17,    # delay tau of the generating equation
    "electricity": 24,     # daily cycle of hourly load
    "temperature": 365,    # annual cycle, far outside every window
    "pageviews": 365,      # the tournament recurs annually
    "sp500": 5,            # one trading week: control, no effect expected
}
SENS_FRAC = 0.6          # never a test fold in the final runs
SENS_TRAIN = 0.8         # of that prefix: train on 80%, measure on the rest
PROMINENCE = 0.5         # log-residual threshold: share at least ~65% above local trend

COLOR = {"elman": "#2a78d6", "jordan": "#eb6834", "mrnn": "#1baf7a"}
STYLE = {"elman": "-", "jordan": "--", "mrnn": ":"}
M_LABEL = {"elman": "Elman", "jordan": "Jordan", "mrnn": "Multi-recurrent"}
DS_PLOT = {"mackey_glass": "Mackey-Glass", "electricity": "Electricity",
           "pageviews": "Page views", "temperature": "Temperature", "sp500": "S&P 500"}
REF_LAGS = {"mackey_glass": [17], "electricity": [24, 48], "sp500": [5]}



# One fit, mirroring pipeline.run_fold, with an optional shortcut input
def lag_values(work: np.ndarray, tgt: np.ndarray, k: int) -> np.ndarray:
    """Value k steps before each target; NaN where the series has no such value."""
    idx = np.asarray(tgt) - k
    out = work[np.clip(idx, 0, None)].astype(float)
    out[idx < 0] = np.nan
    return out


def fit(raw, model_name, cfg, seed, tr_idx, te_idx, k=None, keep_model=False):
    """Train one model on one split and score it in original units.

    With k=None this reproduces pipeline.run_fold exactly (checked in the
    self-test). With k set, a second input channel carries the lag-k value
    at the final time step and zero at every earlier step.
    """
    set_seed(seed)
    T = transforms.get(cfg.transform)
    work = T.to_work(raw)
    X, y = P.make_windows(work, cfg.window, cfg.horizon)
    Xtr_w, ytr_w, Xte_w = X[tr_idx], y[tr_idx], X[te_idx]
    mu = float(Xtr_w.mean())
    sd = float(Xtr_w.std()) or 1.0
    sc = lambda a: (a - mu) / sd
    Xtr, ytr, Xte = sc(Xtr_w), sc(ytr_w), sc(Xte_w)

    n_inputs = 1
    if k is not None:
        n_inputs = 2

        def with_shortcut(Xs, rows):
            s = lag_values(work, P.target_index(rows, cfg), k)
            s = np.where(np.isnan(s), 0.0, (s - mu) / sd)   # missing -> training mean
            channel = np.zeros_like(Xs)
            channel[:, -1, 0] = s.astype(np.float32)
            return np.concatenate([Xs, channel], axis=2)

        Xtr, Xte = with_shortcut(Xtr, tr_idx), with_shortcut(Xte, te_idx)

    cut = max(1, int(len(Xtr) * (1 - cfg.val_frac)))
    t = lambda a: torch.as_tensor(a, device="cpu")
    model = build_model(model_name, cfg.n_hidden, n_inputs=n_inputs)
    hist = P._train_one(model, t(Xtr[:cut]), t(ytr[:cut]),
                        t(Xtr[cut:]), t(ytr[cut:]), cfg, "cpu")
    model.eval()
    with torch.no_grad():
        pred_w = model(t(Xte)).cpu().numpy() * sd + mu

    tgt = P.target_index(te_idx, cfg)
    anchor = T.anchors(raw, tgt, cfg.horizon)[:, None]
    y_true = T.true_values(raw, tgt)[:, None]
    y_pred = T.invert(pred_w, anchor)
    out = {"model": model_name, "seed": seed, "transform": cfg.transform,
           "n_train": cut, "n_test": len(te_idx)}
    out.update(hist)
    out.update(P.metrics(y_true, y_pred, anchor))
    out["weights"] = sum(p.numel() for p in model.parameters())
    return (out, model, Xte) if keep_model else out



# Stage 1: sensitivity profiles
def sensitivity(ds: str) -> pd.DataFrame:
    best = run.merge_tuning()
    raw = run.series_for(ds)
    rows = []
    for m in MODELS:
        cfg = run.final_config(ds, m, best)
        n = P.n_windows(raw, cfg)
        prefix = int(n * SENS_FRAC)
        split = int(prefix * SENS_TRAIN)
        tr = np.arange(0, split - cfg.window)          # gap of one window
        te = np.arange(split, prefix)
        t0 = time.time()
        for s in run.SEEDS:
            _, model, Xte = fit(raw, m, cfg, RANDOM_STATE + s, tr, te, keep_model=True)
            X = torch.as_tensor(Xte).requires_grad_(True)
            model(X).sum().backward()
            g = X.grad.abs().mean(dim=0)[:, 0].numpy()  # per window position
            prof = g[::-1] / g.sum()                    # position -> lag, share
            for lag, share in enumerate(prof, start=1):
                rows.append(dict(dataset=ds, model=m, seed=RANDOM_STATE + s,
                                 window=cfg.window, lag=lag, share=float(share)))
        print(f"{ds:14s} {m:7s} sensitivity done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"sensitivity_{ds}.csv", index=False)
    return df



# Selection rule: the lag that stands out above the fading trend
def select_lag(sens: pd.DataFrame, ds: str, half_width: int = 3) -> dict:
    """Pick the lag the network relies on more than its neighbours.

    Uses the architecture(s) with the longest tuned window on this dataset,
    since they see furthest back. For every lag j >= 2 a straight line is
    fitted to log(share) over the neighbouring lags j-3..j+3, excluding j
    itself, and the residual of lag j above that local trend is recorded.
    The lag with the largest residual is selected. A local trend is used
    because the overall fading of sensitivity is curved: one straight line
    through the whole profile makes the first lags look prominent on any
    smooth profile. Lag 1 is excluded because the most recent observation
    already enters the final step directly.
    """
    d = sens[sens.dataset == ds]
    wmax = d.window.max()
    widest = sorted(d[d.window == wmax].model.unique())
    prof = d[d.model.isin(widest)].groupby("lag").share.mean()
    prof = prof[prof.index >= 2]
    lags = prof.index.to_numpy(float)
    logs = np.log(prof.to_numpy())
    resid = np.full(len(lags), -np.inf)
    for i in range(len(lags)):
        nb = [n for n in range(i - half_width, i + half_width + 1)
              if 0 <= n < len(lags) and n != i]
        if len(nb) < 3:
            continue
        slope, icpt = np.polyfit(lags[nb], logs[nb], 1)
        resid[i] = logs[i] - (slope * lags[i] + icpt)
    k = int(prof.index[int(np.argmax(resid))])
    return {"dataset": ds, "selected_lag": k, "from_models": "+".join(widest),
            "window": int(wmax), "residual": float(np.max(resid)),
            "prominent": bool(np.max(resid) > PROMINENCE),
            "physical_lag": PHYSICAL_LAGS.get(ds)}


def selected_lags() -> pd.DataFrame:
    f = RESULTS_DIR / "selected_lags.csv"
    if not f.exists():
        raise FileNotFoundError("run 'python lag_study.py select' after stage 1")
    return pd.read_csv(f)


def select() -> pd.DataFrame:
    parts = sorted(RESULTS_DIR.glob("sensitivity_*.csv"))
    sens = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    out = pd.DataFrame([select_lag(sens, ds) for ds in ORDER if ds in set(sens.dataset)])
    out.to_csv(RESULTS_DIR / "selected_lags.csv", index=False)
    print(out.to_string(index=False))
    return out



# Stage 2: shortcut runs on the held-out folds
def shortcut(ds: str, models=MODELS) -> pd.DataFrame:
    best = run.merge_tuning()
    raw = run.series_for(ds)
    k = PHYSICAL_LAGS[ds]                  # fixed from the data's origins, not from stage 1
    for m in models:
        cfg = run.final_config(ds, m, best)
        splits = P.cv_splits(P.n_windows(raw, cfg), cfg, run.HOLDOUT_START)
        rows, t0 = [], time.time()
        for fold, (tr, te) in enumerate(splits):
            for s in run.SEEDS:
                r = fit(raw, m, cfg, RANDOM_STATE + s, tr, te, k=k)
                r.update(fold=fold, dataset=ds, shortcut_lag=k, window=cfg.window)
                rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"shortcut_{ds}_{m}.csv", index=False)
        print(f"{ds:14s} {m:7s} lag {k:<3d} U={df.theil_u.mean():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)



# Analysis
def holm(p):
    p = np.asarray(p, float); order = np.argsort(p)
    adj = np.empty_like(p); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def sensitivity_figure(sens: pd.DataFrame):
    present = [d for d in ORDER if d in set(sens.dataset)]
    fig, axes = plt.subplots(1, len(present), figsize=(7.16, 2.1),
                             layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, present):
        d = sens[sens.dataset == ds]
        for m in MODELS:
            prof = d[d.model == m].groupby("lag").share.mean()
            ax.plot(prof.index, prof.values, color=COLOR[m],
                    linestyle=STYLE[m], linewidth=1.1, label=M_LABEL[m])
        for L in REF_LAGS.get(ds, []):           # physical lag: grey dashed
            ax.axvline(L, color="#898781", linewidth=0.6,
                       linestyle="--", zorder=0)
        ax.set_yscale("log")
        ax.set_title(DS_PLOT[ds])
        ax.set_xlabel("Lag")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.5)
    axes[0].set_ylabel("Share of sensitivity")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, frameon=False,
               fontsize=7)
    return savefig(fig, "sensitivity_profiles")


def analyse():
    parts = sorted(RESULTS_DIR.glob("sensitivity_*.csv"))
    if parts:
        sens = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        print("Sensitivity share at the reference lags (mean over seeds):")
        for ds, lags in REF_LAGS.items():
            d = sens[sens.dataset == ds]
            for m in MODELS:
                prof = d[d.model == m].groupby("lag").share.mean()
                if prof.empty:
                    continue
                cells = ", ".join(
                    f"lag {L}: {prof.get(L, np.nan):.3f} (rank {int(prof.rank(ascending=False).get(L, 0)) or '-'})"
                    if L in prof.index else f"lag {L}: outside window {len(prof)}"
                    for L in lags)
                print(f"  {ds:13s} {m:7s} lag 1: {prof.get(1):.3f} | {cells}")
        print("Figure:", sensitivity_figure(sens).name, "\n")

    rows = []
    sel = selected_lags().set_index("dataset") if (RESULTS_DIR / "selected_lags.csv").exists() else None
    for ds in ORDER:
        base_f = RESULTS_DIR / f"final_runs_{ds}.csv"
        if not base_f.exists():
            continue
        base = pd.read_csv(base_f)
        for m in MODELS:
            f = RESULTS_DIR / f"shortcut_{ds}_{m}.csv"
            if not f.exists():
                continue
            new = pd.read_csv(f)
            a = base[base.model == m].set_index(["fold", "seed"]).theil_u
            b = new.set_index(["fold", "seed"]).theil_u
            a, b = a.align(b, join="inner")
            rows.append(dict(dataset=ds, model=m, lag=int(new.shortcut_lag.iloc[0]),
                             window=int(new.window.iloc[0]), n_pairs=len(a),
                             u_base=a.mean(), u_shortcut=b.mean(),
                             median_change=float(np.median(b - a)),
                             p=stats.wilcoxon(b, a).pvalue))
    if rows:
        out = pd.DataFrame(rows)
        out["p_holm"] = holm(out.p)
        out["verdict"] = np.where(out.p_holm >= 0.05, "no significant change",
                         np.where(out.median_change < 0, "shortcut lowers U",
                                  "shortcut raises U"))
        out.to_csv(RESULTS_DIR / "shortcut_tests.csv", index=False)
        print("Shortcut versus final runs (paired over fold x seed, Holm-corrected):")
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sensitivity", "select", "shortcut", "analyse"])
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    if a.stage == "sensitivity":
        for ds in a.datasets:
            sensitivity(ds)
    elif a.stage == "select":
        select()
    elif a.stage == "shortcut":
        for ds in a.datasets:
            shortcut(ds, a.models)
    else:
        analyse()