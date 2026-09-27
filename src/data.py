"""Dataset loaders for Assignment 3.

Every loader returns a pandas Series indexed by time, with no missing values.
Raw downloads are cached under data/ so each source is fetched once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_DIR, RANDOM_STATE

TEMPS_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
PJME_URL = "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJME_hourly.csv"
WIKI_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/daily/{start}/{end}"
)
USER_AGENT = "ML441-assignment3/1.0 (student project)"


def _cache(name: str, url: str) -> str:
    """Download url to data/name once; return the local path."""
    import urllib.request
    path = DATA_DIR / name
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    return str(path)


# 1. Mackey-Glass: synthetic chaotic benchmark, generated not downloaded.
def load_mackey_glass(n: int = 3000, tau: int = 17, burn_in: int = 1000,
                      beta: float = 0.2, gamma: float = 0.1, n_exp: int = 10,
                      dt: float = 0.1, seed: int = RANDOM_STATE) -> pd.Series:
    """Integrate the Mackey-Glass delay differential equation (RK4).

    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
    tau=17 gives mildly chaotic dynamics; this is the standard RNN benchmark.
    """
    rng = np.random.default_rng(seed)
    steps_per_unit = int(round(1.0 / dt))
    delay = tau * steps_per_unit
    total = (n + burn_in) * steps_per_unit + delay

    x = np.empty(total)
    x[:delay] = 1.2 + 0.01 * rng.standard_normal(delay)  # history segment

    def f(xt, xd):
        return beta * xd / (1.0 + xd ** n_exp) - gamma * xt

    # RK4 on the non-delayed term; the delayed term is held constant across
    # the four stages, the standard simplification for a fixed-step DDE.
    for i in range(delay, total):
        xt, xd = x[i - 1], x[i - 1 - delay]
        k1 = f(xt, xd)
        k2 = f(xt + 0.5 * dt * k1, xd)
        k3 = f(xt + 0.5 * dt * k2, xd)
        k4 = f(xt + dt * k3, xd)
        x[i] = xt + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    series = x[delay + burn_in * steps_per_unit::steps_per_unit][:n]
    if not np.all(np.isfinite(series)) or series.max() > 5.0:
        raise RuntimeError(f"Mackey-Glass integration diverged "
                           f"(max={np.nanmax(series):.3g}); reduce dt")
    idx = pd.RangeIndex(len(series), name="t")
    return pd.Series(series, index=idx, name="mackey_glass")



# 2. Melbourne daily minimum temperatures, 1981-1990.
def load_temperature() -> pd.Series:
    path = _cache("daily-min-temperatures.csv", TEMPS_URL)
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    s = pd.to_numeric(df["Temp"], errors="coerce")
    s = s.asfreq("D").interpolate(method="time").astype(float)
    s.name = "temperature"
    return s



# 3. PJM East hourly electricity demand (MW).
def load_electricity(years: float = 2.0) -> pd.Series:
    """Hourly system load. Duplicated DST hours averaged, gaps interpolated."""
    path = _cache("PJME_hourly.csv", PJME_URL)
    df = pd.read_csv(path, parse_dates=["Datetime"])
    s = df.groupby("Datetime")["PJME_MW"].mean().sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full).interpolate(method="time")
    s = s.iloc[-int(years * 365 * 24):]
    s.index.name = "Datetime"
    s.name = "electricity"
    return s.astype(float)



# 4. Wikipedia daily pageviews for a golf article.
def load_pageviews(article: str = "Masters_Tournament",
                   start: str = "20160101", end: str = "20250831") -> pd.Series:
    import json
    url = WIKI_URL.format(article=article, start=start + "00", end=end + "00")
    path = _cache(f"pageviews_{article}.json", url)
    with open(path) as f:
        items = json.load(f)["items"]
    s = pd.Series(
        [it["views"] for it in items],
        index=pd.to_datetime([it["timestamp"][:8] for it in items]),
        dtype=float,
    )
    s = s.asfreq("D").interpolate(method="time")
    s.index.name = "Date"
    s.name = "pageviews"
    return s



# 5. S&P 500 daily close.
def load_sp500(start: str = "2010-01-01", end: str = "2025-08-31") -> pd.Series:
    """Daily close. Cached to CSV so yfinance is only hit once."""
    path = DATA_DIR / "sp500.csv"
    if not path.exists():
        import yfinance as yf
        df = yf.download("^GSPC", start=start, end=end,
                         auto_adjust=True, progress=False)
        df["Close"].to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index.name = "Date"
    s.name = "sp500"
    return s.astype(float)


DATASETS = {
    "mackey_glass": load_mackey_glass,
    "temperature": load_temperature,
    "electricity": load_electricity,
    "pageviews": load_pageviews,
    "sp500": load_sp500,
}


def describe(s: pd.Series) -> dict:
    """Summary row plus ADF/KPSS stationarity verdicts."""
    from statsmodels.tsa.stattools import adfuller, kpss
    import warnings

    x = s.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(x, autolag="AIC")[1]
        kpss_p = kpss(x, regression="c", nlags="auto")[1]

    adf_stat = adf_p < 0.05          # rejects unit root -> stationary
    kpss_stat = kpss_p >= 0.05       # fails to reject -> stationary
    if adf_stat and kpss_stat:
        verdict = "stationary"
    elif not adf_stat and not kpss_stat:
        verdict = "non-stationary"
    else:
        verdict = "difference-stationary" if adf_stat else "trend-stationary"

    return {
        "dataset": s.name,
        "n": len(s),
        "start": str(s.index[0]),
        "end": str(s.index[-1]),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "adf_p": round(float(adf_p), 4),
        "kpss_p": round(float(kpss_p), 4),
        "verdict": verdict,
    }