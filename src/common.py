"""Shared config, paths and plotting defaults for Assignment 3."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
for _d in (DATA_DIR, RESULTS_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# IEEE two-column: 3.5in single-column width, 8pt text inside figures.
matplotlib.rcParams.update({
    "figure.figsize": (3.5, 2.2),
    "figure.dpi": 150,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def savefig(fig, name: str) -> Path:
    """Save a figure as vector PDF into figures/."""
    path = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)