"""Sanity checks for the Assignment 3 pipeline. Run: python sanity.py"""
import numpy as np
import torch

import common, data, models, pipeline as P
import transforms

ok = lambda label: print(f"  PASS  {label}")


def check_windows():
    x = np.arange(10, dtype=float)
    X, y = P.make_windows(x, window=3, horizon=1)
    assert X.shape == (7, 3, 1), X.shape
    assert y.shape == (7, 1), y.shape
    assert np.allclose(X[0, :, 0], [0, 1, 2]) and y[0, 0] == 3
    assert np.allclose(X[-1, :, 0], [6, 7, 8]) and y[-1, 0] == 9
    ok("make_windows: shapes and alignment")

    X2, y2 = P.make_windows(x, window=3, horizon=2)
    assert np.allclose(X2[0, :, 0], [0, 1, 2]) and y2[0, 0] == 4
    ok("make_windows: horizon=2 skips a step")


def check_cv():
    cfg = P.Config(window=20, n_splits=5)
    splits = P.cv_splits(1000, cfg)
    assert len(splits) == 5
    prev = -1
    for tr, te in splits:
        assert tr.max() < te.min(), "train must precede test"
        assert te.min() - tr.max() > cfg.window, "gap too small"
        assert len(tr) > prev, "training set must expand"
        prev = len(tr)
    ok("cv_splits: ordered, expanding, gap respected")


def check_models():
    x = torch.randn(4, 20, 1)
    counts = {}
    for name in models.MODELS:
        m = models.build_model(name, n_hidden=8)
        out = m(x)
        assert out.shape == (4, 1), (name, out.shape)
        counts[name] = sum(p.numel() for p in m.parameters())
    ok(f"forward pass: output shape (4,1) for all three")

    e, j, r = counts["elman"], counts["jordan"], counts["mrnn"]
    assert j < e < r, counts
    print(f"        params: jordan={j}  elman={e}  mrnn={r}")
    ok("parameter counts ordered jordan < elman < mrnn")


def check_determinism():
    cfg = P.Config(window=10, n_splits=2, n_hidden=4, max_epochs=3, patience=99)
    s = data.load_mackey_glass(n=300).to_numpy()
    a = P.evaluate(s, "elman", cfg, seeds=(0,))
    b = P.evaluate(s, "elman", cfg, seeds=(0,))
    assert a[0]["rmse"] == b[0]["rmse"], (a[0]["rmse"], b[0]["rmse"])
    ok("same seed reproduces the same RMSE")


def check_no_leakage():
    """Shuffling the future must not change a fold's test score."""
    cfg = P.Config(window=10, n_splits=2, n_hidden=4, max_epochs=3, patience=99)
    s = data.load_mackey_glass(n=400).to_numpy()
    X, _ = P.make_windows(s, cfg.window, cfg.horizon)
    tr, te = P.cv_splits(P.n_windows(s, cfg), cfg)[0]

    base = P.run_fold(s, "elman", cfg, 42, tr, te)
    s2 = s.copy()
    tail = (te.max() + cfg.window + cfg.horizon)
    if tail < len(s2) - 2:
        rng = np.random.default_rng(0)
        s2[tail:] = rng.permutation(s2[tail:])       # scramble beyond this fold
    after = P.run_fold(s2, "elman", cfg, 42, tr, te)
    assert abs(base["rmse"] - after["rmse"]) < 1e-9, (base["rmse"], after["rmse"])
    ok("data after the test fold does not affect its score")


def check_baseline():
    cfg = P.Config(window=10, n_splits=2, n_hidden=4, max_epochs=5, patience=99)
    s = data.load_mackey_glass(n=400).to_numpy()
    r = P.evaluate(s, "elman", cfg, seeds=(0,))[0]
    assert r["rmse_naive"] > 0 and np.isfinite(r["theil_u"])
    ok(f"persistence baseline computed (theil_u={r['theil_u']:.3f})")


def check_smoke():
    cfg = P.Config(window=20, n_splits=3, n_hidden=8, max_epochs=60, patience=10)
    s = data.load_mackey_glass(n=1500).to_numpy()
    res = {}
    for m in models.MODELS:
        rows = P.evaluate(s, m, cfg, seeds=(0, 1))
        res[m] = float(np.mean([r["theil_u"] for r in rows]))
    print("        mean Theil's U:", {k: round(v, 3) for k, v in res.items()})
    assert all(v < 1.0 for v in res.values()), "all models should beat persistence on MG"
    ok("all three beat the naive baseline on Mackey-Glass")

def check_transforms():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 5000, 200).astype(float)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))

    assert np.allclose(transforms.get("none").invert(counts, None), counts)
    ok("identity: invert is a no-op")

    T = transforms.get("log1p")
    assert np.allclose(T.invert(T.to_work(counts), None), counts)
    ok("log1p: round-trips to the original counts")

    T = transforms.get("logdiff")
    work = T.to_work(prices)
    assert len(work) == len(prices) - 1
    t = np.arange(len(work))
    recon = T.invert(work, T.anchors(prices, t, 1))
    assert np.allclose(recon, T.true_values(prices, t))
    ok("logdiff: returns invert back to the exact price levels")

    # a zero-return prediction must reproduce the persistence forecast
    assert np.allclose(T.invert(np.zeros_like(work), T.anchors(prices, t, 1)),
                       prices[:-1])
    ok("logdiff: a zero prediction equals persistence")


def check_transform_scoring():
    """Errors must come back in original units, not transformed ones."""
    s = data.load_mackey_glass(n=400).to_numpy() * 1000.0 + 5000.0
    cfg_a = P.Config(window=10, n_splits=2, n_hidden=4, max_epochs=3,
                     patience=99, transform="none")
    cfg_b = P.Config(**{**vars(cfg_a), "transform": "log1p"})
    tr, te = P.cv_splits(P.n_windows(s, cfg_a), cfg_a)[0]
    a = P.run_fold(s, "elman", cfg_a, 42, tr, te)
    b = P.run_fold(s, "elman", cfg_b, 42, tr, te)
    # same series, same units: the naive baseline must be identical
    assert abs(a["rmse_naive"] - b["rmse_naive"]) < 1e-6, (a["rmse_naive"], b["rmse_naive"])
    assert a["rmse"] > 1.0 and b["rmse"] > 1.0, "errors look like log units, not raw"
    ok(f"scores in original units under both transforms "
       f"(naive RMSE={a['rmse_naive']:.1f})")

if __name__ == "__main__":
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"seed={common.RANDOM_STATE}  data dir={common.DATA_DIR}\n")
    for fn in (check_windows, check_cv, check_models, check_determinism,
               check_no_leakage, check_baseline, check_smoke, check_transforms, check_transform_scoring):
        print(fn.__name__)
        fn()
    print("\nAll checks passed.")