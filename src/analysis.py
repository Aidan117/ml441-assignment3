"""Tables, statistical tests and figures from the final runs.

Follows the procedure in the Topic 11 slides: a Friedman test across the three
architectures per dataset, pairwise Wilcoxon signed-rank tests with the Holm
correction, wins and losses from the pairwise outcomes, and average ranks with
a Nemenyi critical difference across datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from common import ROOT, RESULTS_DIR, savefig
from models import MODELS

TABLE_DIR = ROOT / "tables"
TABLE_DIR.mkdir(exist_ok=True)

ALPHA = 0.05
ORDER = ["mackey_glass", "electricity", "pageviews", "temperature", "sp500"]
DS_LABEL = {"mackey_glass": "Mackey-Glass", "electricity": "Electricity",
            "pageviews": "Page views", "temperature": "Temperature",
            "sp500": "S\\&P 500"}
DS_PLOT = {**DS_LABEL, "sp500": "S&P 500"}
# Full names, not acronyms: WR22c forbids acronyms in headings and captions.
M_LABEL = {"elman": "Elman", "jordan": "Jordan", "mrnn": "Multi-recurrent"}
# Validated categorical palette; line style is the secondary encoding.
COLOR = {"elman": "#2a78d6", "jordan": "#eb6834", "mrnn": "#1baf7a"}
STYLE = {"elman": "-", "jordan": "--", "mrnn": ":"}
INK = "#0b0b0b"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load(prefix: str) -> pd.DataFrame:
    parts = sorted(RESULTS_DIR.glob(f"{prefix}_*.csv"))
    if not parts:
        raise FileNotFoundError(f"no {prefix}_*.csv in {RESULTS_DIR}")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def paired(runs: pd.DataFrame, ds: str, metric="theil_u") -> pd.DataFrame:
    """Rows = (fold, seed) blocks, columns = models. Pairing is by block."""
    d = runs[runs.dataset == ds]
    return d.pivot_table(index=["fold", "seed"], columns="model",
                         values=metric)[list(MODELS)].dropna()


def holm(pvals):
    """Holm step-down adjustment of a list of p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def summary(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["dataset", "model"])
    out = g.agg(u_mean=("theil_u", "mean"), u_std=("theil_u", "std"),
                rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                relmae_mean=("mase", "mean"),     # column is relative MAE, see report
                n=("theil_u", "size")).reset_index()
    out.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return out


def latex_summary(summ: pd.DataFrame) -> str:
    rows = []
    for ds in ORDER:
        d = summ[summ.dataset == ds].set_index("model")
        if d.empty:
            continue
        best = d.u_mean.idxmin()
        cells = []
        for m in MODELS:
            cell = f"{d.loc[m, 'u_mean']:.4f} $\\pm$ {d.loc[m, 'u_std']:.4f}"
            cells.append(f"\\textbf{{{cell}}}" if m == best else cell)
        rows.append(f"{DS_LABEL[ds]} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(M_LABEL[m] for m in MODELS)
    return ("\\begin{table*}[t]\n\\centering\n"
            "\\caption{Mean and Standard Deviation of the Theil U Statistic}\n"
            "\\label{tab:theil}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
            f"Dataset & {head} \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# --------------------------------------------------------------------------
# Statistical tests
# --------------------------------------------------------------------------
def per_dataset_tests(runs: pd.DataFrame) -> pd.DataFrame:
    """Friedman across models, then Holm-corrected pairwise Wilcoxon."""
    pairs = [("elman", "jordan"), ("elman", "mrnn"), ("jordan", "mrnn")]
    rows = []
    for ds in ORDER:
        P = paired(runs, ds)
        if P.empty:
            continue
        chi2, p_f = stats.friedmanchisquare(*[P[m] for m in MODELS])
        raw = [stats.wilcoxon(P[a], P[b]).pvalue for a, b in pairs]
        adj = holm(raw)
        for (a, b), pr, pa in zip(pairs, raw, adj):
            med = float(np.median(P[a] - P[b]))
            if p_f >= ALPHA or pa >= ALPHA:
                winner = "tie"
            else:
                winner = a if med < 0 else b          # lower U is better
            rows.append(dict(dataset=ds, n_blocks=len(P), friedman_chi2=chi2,
                             friedman_p=p_f, a=a, b=b, median_diff=med,
                             p_raw=pr, p_holm=pa, winner=winner))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "tests_pairwise.csv", index=False)
    return out


def wins_losses(tests: pd.DataFrame) -> pd.DataFrame:
    rec = {m: {"wins": 0, "losses": 0, "ties": 0} for m in MODELS}
    for _, r in tests.iterrows():
        if r.winner == "tie":
            rec[r.a]["ties"] += 1; rec[r.b]["ties"] += 1
        else:
            loser = r.b if r.winner == r.a else r.a
            rec[r.winner]["wins"] += 1; rec[loser]["losses"] += 1
    out = pd.DataFrame(rec).T
    out["net"] = out.wins - out.losses
    out = out.sort_values("net", ascending=False)
    out.to_csv(RESULTS_DIR / "wins_losses.csv")
    return out


def vs_persistence(runs: pd.DataFrame) -> pd.DataFrame:
    """One-sided Wilcoxon: is U below 1, i.e. better than persistence?"""
    rows = []
    for ds in ORDER:
        P = paired(runs, ds)
        if P.empty:
            continue
        ps = [stats.wilcoxon(P[m] - 1.0, alternative="less").pvalue for m in MODELS]
        for m, p, pa in zip(MODELS, ps, holm(ps)):
            rows.append(dict(dataset=ds, model=m, median_u=float(P[m].median()),
                             p_raw=p, p_holm=pa, beats_persistence=pa < ALPHA))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "tests_vs_persistence.csv", index=False)
    return out


def across_datasets(runs: pd.DataFrame) -> dict:
    """Average ranks over datasets, Friedman, Nemenyi critical difference."""
    means = runs.groupby(["dataset", "model"]).theil_u.mean().unstack()[list(MODELS)]
    ranks = means.rank(axis=1)                        # 1 = best (lowest U)
    k, n = len(MODELS), len(means)
    chi2, p = stats.friedmanchisquare(*[means[m] for m in MODELS])
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}[k]   # Demsar (2006), alpha 0.05
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"ranks": ranks, "avg_rank": ranks.mean(), "chi2": chi2,
            "p": p, "cd": cd, "n_datasets": n}


def convergence(runs: pd.DataFrame) -> pd.DataFrame:
    """Share of fits that exhausted the epoch budget without early stopping."""
    r = runs.assign(hit_cap=runs.epochs_run >= runs.cfg_max_epochs)
    out = r.groupby(["dataset", "model"]).agg(
        hit_cap_share=("hit_cap", "mean"),
        median_best_epoch=("best_epoch", "median")).reset_index()
    out.to_csv(RESULTS_DIR / "convergence.csv", index=False)
    return out


def sp500_diagnostic(preds: pd.DataFrame) -> pd.DataFrame:
    """Spread of implied predicted returns versus actual returns."""
    s = preds[preds.dataset == "sp500"]
    if s.empty:
        return pd.DataFrame()
    rows = []
    for m, g in s.groupby("model"):
        pred_ret = np.log(g.predicted / g.naive)
        act_ret = np.log(g.actual / g.naive)
        rows.append(dict(model=m, pred_return_std=pred_ret.std(),
                         actual_return_std=act_ret.std(),
                         ratio=pred_ret.std() / act_ret.std(),
                         pred_return_mean=pred_ret.mean()))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "sp500_diagnostic.csv", index=False)
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.5)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)


WINDOWS = {"mackey_glass": 200, "electricity": 168, "pageviews": 60,
           "temperature": 120, "sp500": 60}
XLABEL = {"mackey_glass": "Time step", "electricity": "Hour", "pageviews": "Day",
          "temperature": "Day", "sp500": "Trading day"}
YLABEL = {"mackey_glass": "Series value", "electricity": "Load (MW)",
          "pageviews": "Daily page views", "temperature": "Temperature (°C)",
          "sp500": "Index level"}
THOUSANDS = {"electricity", "pageviews", "sp500"}
ACTUAL = "#a8a69f"          # light, thick, underneath: stays visible as a halo


def forecast_figures(preds: pd.DataFrame):
    """One single-column figure per dataset: actual versus the three models."""
    from matplotlib.ticker import FuncFormatter
    paths = []
    for ds in ORDER:
        d = preds[preds.dataset == ds]
        if d.empty:
            continue
        actual = d[d.model == MODELS[0]].set_index("step").actual
        w = WINDOWS[ds]
        if ds == "pageviews":               # centre the window on the largest spike
            c = int(actual.idxmax())
            lo = max(0, c - w // 3)
        else:                               # otherwise the end of the test fold
            lo = max(0, int(actual.index.max()) - w + 1)
        hi = lo + w

        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        a = actual.loc[lo:hi - 1]
        ax.plot(a.index - lo, a.values, color=ACTUAL, linewidth=2.6,
                solid_capstyle="round", zorder=1, label="Actual")
        for z, m in enumerate(MODELS, start=2):
            p = d[d.model == m].set_index("step").predicted.loc[lo:hi - 1]
            ax.plot(p.index - lo, p.values, color=COLOR[m], linestyle=STYLE[m],
                    linewidth=1.0, zorder=z, label=M_LABEL[m])
        ax.set_xlabel(XLABEL[ds])
        ax.set_ylabel(YLABEL[ds])
        if ds in THOUSANDS:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(frameon=False, ncol=4, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), fontsize=6.5,
                  handlelength=1.8, columnspacing=0.9, borderaxespad=0.2)
        _style(ax)
        paths.append(savefig(fig, f"forecast_{ds}"))
    return paths


def theil_boxplots(runs: pd.DataFrame):
    """Small multiples: each dataset on its own scale (never a shared axis)."""
    from matplotlib.patches import Patch
    present = [ds for ds in ORDER if ds in set(runs.dataset)]
    fig, axes = plt.subplots(1, len(present), figsize=(7.16, 1.8))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, present):
        P = paired(runs, ds)
        bp = ax.boxplot([P[m] for m in MODELS], widths=0.6, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=1.0),
                        flierprops=dict(markersize=3, markeredgewidth=0.5))
        for patch, m in zip(bp["boxes"], MODELS):
            patch.set_facecolor(COLOR[m]); patch.set_alpha(0.35)
            patch.set_edgecolor(COLOR[m]); patch.set_linestyle(STYLE[m])
        ax.set_xticks([])                   # identity comes from the shared legend
        ax.set_title(DS_PLOT[ds])
        if ds == "sp500":
            ax.axhline(1.0, color="#898781", linewidth=0.7, linestyle="--")
        _style(ax)
    axes[0].set_ylabel("Theil U statistic")
    handles = [Patch(facecolor=COLOR[m], edgecolor=COLOR[m], alpha=0.5,
                     linestyle=STYLE[m], label=M_LABEL[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.98), fontsize=7)
    fig.tight_layout(w_pad=0.6)
    return savefig(fig, "theil_boxplots")


# --------------------------------------------------------------------------
def main():
    runs, preds = load("final_runs"), load("final_predictions")
    print(f"{len(runs)} fits across {runs.dataset.nunique()} datasets\n")

    summ = summary(runs)
    (TABLE_DIR / "theil_summary.tex").write_text(latex_summary(summ))
    print("Mean Theil U (std):")
    print(summ.pivot(index="dataset", columns="model", values="u_mean")
          .loc[[d for d in ORDER if d in set(summ.dataset)], list(MODELS)].round(4), "\n")

    tests = per_dataset_tests(runs)
    print("Friedman per dataset:")
    print(tests.groupby("dataset")[["friedman_chi2", "friedman_p"]].first()
          .loc[[d for d in ORDER if d in set(tests.dataset)]].round(4), "\n")
    print("Pairwise Wilcoxon, Holm-corrected:")
    print(tests[["dataset", "a", "b", "median_diff", "p_holm", "winner"]]
          .round(4).to_string(index=False), "\n")
    print("Wins / losses / ties:"); print(wins_losses(tests), "\n")

    vp = vs_persistence(runs)
    print("Better than persistence (one-sided Wilcoxon, Holm within dataset):")
    print(vp.round(4).to_string(index=False), "\n")

    acr = across_datasets(runs)
    print(f"Across datasets: Friedman chi2={acr['chi2']:.3f}, p={acr['p']:.4f}, "
          f"Nemenyi CD={acr['cd']:.3f} over {acr['n_datasets']} datasets")
    print("Average ranks:", acr["avg_rank"].round(2).to_dict(), "\n")

    print("Convergence (share of fits that hit the epoch cap):")
    print(convergence(runs).round(3).to_string(index=False), "\n")

    sp = sp500_diagnostic(preds)
    if not sp.empty:
        print("S&P 500 implied return spread (ratio << 1 means near-persistence):")
        print(sp.round(5).to_string(index=False), "\n")

    figs = forecast_figures(preds) + [theil_boxplots(runs)]
    print("Figures:", ", ".join(p.name for p in figs))
    print("Table:  ", TABLE_DIR / "theil_summary.tex")


if __name__ == "__main__":
    main()