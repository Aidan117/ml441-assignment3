"""Robustness check for pseudo-replication.

The 25 blocks of a series (5 folds x 5 seeds) are not independent: the seeds
of a fold share the same training and test data. This script averages over
seeds within each fold, leaving five values per architecture, and counts on
how many of the five folds each comparison points the same way.

Writes results/fold_robustness.csv and prints a summary.
Usage:  python fold_robustness.py
"""
from __future__ import annotations

import itertools

import pandas as pd

from common import RESULTS_DIR
from models import MODELS

ORDER = ["mackey_glass", "electricity", "pageviews", "temperature", "sp500"]


def fold_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean Theil U per fold and model (average over seeds)."""
    return df.groupby(["fold", "model"]).theil_u.mean().unstack()


def main() -> pd.DataFrame:
    rows = []
    for ds in ORDER:
        f = RESULTS_DIR / f"final_runs_{ds}.csv"
        if not f.exists():
            continue
        fm = fold_means(pd.read_csv(f))
        n = len(fm)

        # architecture pairs: on how many folds is a lower than b?
        for a, b in itertools.combinations(MODELS, 2):
            a_lower = int((fm[a] < fm[b]).sum())
            rows.append(dict(dataset=ds, comparison=f"{a} vs {b}",
                             folds=n, first_lower=a_lower,
                             mean_diff=float((fm[a] - fm[b]).mean())))

        # each architecture against persistence: on how many folds is U < 1?
        for m in MODELS:
            rows.append(dict(dataset=ds, comparison=f"{m} vs persistence",
                             folds=n, first_lower=int((fm[m] < 1).sum()),
                             mean_diff=float((fm[m] - 1).mean())))

        # shortcut against the paired final runs
        for m in MODELS:
            s = RESULTS_DIR / f"shortcut_{ds}_{m}.csv"
            if not s.exists():
                continue
            sm = pd.read_csv(s).groupby("fold").theil_u.mean()
            base = fm[m].loc[sm.index]
            rows.append(dict(dataset=ds, comparison=f"{m} shortcut vs base",
                             folds=len(sm), first_lower=int((sm < base).sum()),
                             mean_diff=float((sm - base).mean())))

    out = pd.DataFrame(rows)
    out["agreement"] = out.apply(
        lambda r: max(r.first_lower, r.folds - r.first_lower), axis=1)
    out.to_csv(RESULTS_DIR / "fold_robustness.csv", index=False)

    pd.set_option("display.width", 120)
    print("first_lower = folds (of 5) on which the first named option has "
          "the lower U")
    print("agreement   = folds pointing in the majority direction "
          "(5/5 is fully consistent)\n")
    for ds, g in out.groupby("dataset", sort=False):
        print(ds)
        print(g.drop(columns="dataset").round(4).to_string(index=False), "\n")
    return out


if __name__ == "__main__":
    main()