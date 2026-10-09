"""Data: daily ETF returns and the risk-free rate (T-bill), cached on disk."""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "EFA", "EEM", "TLT", "IEF", "TIP", "LQD", "GLD", "DBC"]


def download_returns(tickers=TICKERS, start="2006-03-01"):
    """Daily log-returns (decimal) from adjusted prices, with basic sanity checks."""
    prices = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"][list(tickers)]
    assert all(yf.Ticker(t).fast_info["currency"] == "USD" for t in tickers), "Non-USD asset"
    assert prices.index.is_unique and prices.index.is_monotonic_increasing
    prices = prices.loc[prices.apply(pd.Series.first_valid_index).max():].ffill(limit=3).dropna()
    assert (prices > 0).all().all(), "Non-positive prices"
    return np.log(prices).diff().dropna()


def download_risk_free(index):
    """Daily cash return from the 13-week T-bill (^IRX, annual rate in %).

    The return of day t uses the rate of day t-1, i.e. the rate known when deciding."""
    irx = yf.download("^IRX", start=index[0] - pd.Timedelta(days=10),
                      end=index[-1] + pd.Timedelta(days=1), auto_adjust=True,
                      progress=False)["Close"].squeeze()
    irx = irx.reindex(irx.index.union(index)).ffill().shift(1).reindex(index)
    assert irx.notna().all(), "Gaps in the ^IRX series"
    return (irx / 100 / 252).rename("rf")


def load_data(cache_dir="data"):
    """(rets, rf). Downloaded once and saved as CSV, so every run uses the same numbers
    and results are reproducible."""
    cache = Path(cache_dir)
    f_rets, f_rf = cache / "rets.csv", cache / "rf.csv"
    if not f_rets.exists():
        cache.mkdir(parents=True, exist_ok=True)
        rets = download_returns()
        rets.to_csv(f_rets)
        download_risk_free(rets.index).to_csv(f_rf)
    rets = pd.read_csv(f_rets, index_col=0, parse_dates=True)
    rf = pd.read_csv(f_rf, index_col=0, parse_dates=True).squeeze("columns")
    return rets, rf
