"""Schematic of the data partitioning.

Top rows:    the three expanding-window folds used during tuning, confined
             to the first 60% of the series (the tuning prefix).
Bottom rows: the five expanding-window folds of the final runs, whose test
             folds lie in the last 40% (the test region).
Each row shows the training set, the validation tail used for early
stopping, the gap of d patterns, and the test fold.

Usage:  python cv_diagram.py      -> figures/cv_diagram.pdf
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from common import savefig

TEST_START = 0.6          # test region = last 40% of the series
N_TUNE = 3                # tuning folds (TimeSeriesSplit inside the prefix)
N_FINAL = 5               # final folds in the test region
GAP = 0.015               # gap of d patterns (drawn wider than scale)
VAL_FRAC = 0.2            # validation = last 20% of each training set
H = 0.62                  # bar height
SPACE = 0.5               # vertical space between the two groups

INK = "#3a3935"
MUTED = "#898781"
COLORS = {"train": "#cfd9e8", "val": "#7a9cc6", "test": "#3a3935"}


def fold_rows():
    """(label, training end, test start, test end) for every row, top first."""
    rows = []
    q = TEST_START / (N_TUNE + 1)             # TimeSeriesSplit: equal chunks
    for f in range(N_TUNE):
        te0 = (f + 1) * q
        rows.append((f"Tune {f + 1}", te0 - GAP, te0, te0 + q))
    chunk = (1 - TEST_START) / N_FINAL
    for f in range(N_FINAL):
        te0 = TEST_START + f * chunk
        rows.append((f"Fold {f + 1}", te0 - GAP, te0, te0 + chunk))
    return rows


def main():
    rows = fold_rows()
    n = len(rows)
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    for i, (label, tr_end, te0, te1) in enumerate(rows):
        y = n - 1 - i - (SPACE if i >= N_TUNE else 0)
        val0 = tr_end * (1 - VAL_FRAC)
        ax.add_patch(Rectangle((0, y), val0, H, color=COLORS["train"], lw=0))
        ax.add_patch(Rectangle((val0, y), tr_end - val0, H,
                               color=COLORS["val"], lw=0))
        ax.add_patch(Rectangle((te0, y), te1 - te0, H,
                               color=COLORS["test"], lw=0))
        ax.text(-0.015, y + H / 2, label, ha="right", va="center",
                fontsize=7, color=INK)

    # boundary between the tuning prefix and the test region
    ax.axvline(TEST_START, color=INK, lw=0.6, ls="--", zorder=0)
    ax.text(TEST_START / 2, n + 0.05, "Tuning prefix (60%)",
            ha="center", va="bottom", fontsize=7, color=INK)
    ax.text(TEST_START + (1 - TEST_START) / 2, n + 0.05,
            "Test region (40%)", ha="center", va="bottom", fontsize=7,
            color=INK)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.75, n + 0.55)
    ax.set_xticks([])
    ax.set_xlabel("Time $\\rightarrow$", fontsize=7, color=INK, labelpad=2)
    ax.set_yticks([])
    ax.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.5)

    handles = [Patch(color=COLORS["train"], label="Training"),
               Patch(color=COLORS["val"], label="Validation"),
               Patch(facecolor="white", edgecolor=MUTED, lw=0.5,
                     label="Gap of $d$"),
               Patch(color=COLORS["test"], label="Test fold")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.09), handlelength=1.2,
               columnspacing=1.0)
    print("Figure:", savefig(fig, "cv_diagram"))


if __name__ == "__main__":
    main()