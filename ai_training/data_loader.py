# =============================================================================
# ai_training/data_loader.py
#
# Loads historical OHLCV data from:
#   1. MetaTrader 5 (if mt5 is installed and connected)
#   2. A local CSV file
#   3. Synthetic data (fallback for offline development)
#
# Public API
# ----------
#   load_data(source="auto") → pd.DataFrame
#       Columns: time, open, high, low, close, volume  (all lowercase)
# =============================================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Helper: attempt MetaTrader 5 import (optional dependency)
# ---------------------------------------------------------------------------
def _mt5_available() -> bool:
    try:
        import MetaTrader5 as mt5          # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helper: attempt yfinance import (optional dependency)
# ---------------------------------------------------------------------------
def _yfinance_available() -> bool:
    try:
        import yfinance as yf              # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Source 1b: Yahoo Finance (works on Mac/Linux without MT5)
# ---------------------------------------------------------------------------

# Maps MT5-style timeframes to yfinance interval + max period
# yfinance valid period strings: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
_YF_INTERVAL_MAP = {
    "M1":  ("1m",  "7d"),
    "M5":  ("5m",  "60d"),
    "M15": ("15m", "60d"),
    "M30": ("30m", "60d"),
    "H1":  ("1h",  "2y"),
    "H4":  ("1h",  "2y"),   # fetched as 1h, resampled below
    "D1":  ("1d",  "5y"),
}

# Yahoo Finance ticker symbols for common FX pairs and commodities
_YF_SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
    "XAUUSD": "GC=F",       # Gold (futures — most reliable data source)
}


def _load_from_yahoo() -> pd.DataFrame:
    import yfinance as yf

    symbol   = config.SYMBOL
    yf_ticker = _YF_SYMBOL_MAP.get(symbol, symbol + "=X")
    interval, period = _YF_INTERVAL_MAP.get(config.TIMEFRAME, ("15m", "60d"))

    print(f"[DataLoader] Yahoo Finance: {yf_ticker} | interval={interval} | period={period}")
    raw = yf.download(yf_ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)

    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {yf_ticker}")

    # yfinance returns MultiIndex columns when downloading a single ticker
    # in newer versions — flatten if needed
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw = raw.reset_index()

    # Normalise column names
    raw.columns = [c.lower().strip() for c in raw.columns]

    # yfinance uses "datetime" or "date" as the index name
    for time_col in ("datetime", "date", "index"):
        if time_col in raw.columns:
            raw.rename(columns={time_col: "time"}, inplace=True)
            break

    if "time" not in raw.columns:
        raise ValueError(f"Cannot find time column. Columns: {list(raw.columns)}")

    raw["time"] = pd.to_datetime(raw["time"])
    # Strip timezone so it matches synthetic/CSV data
    if hasattr(raw["time"].dt, "tz") and raw["time"].dt.tz is not None:
        raw["time"] = raw["time"].dt.tz_localize(None)

    if "volume" not in raw.columns:
        raw["volume"] = 0.0

    df = raw[["time", "open", "high", "low", "close", "volume"]].copy()

    # Resample H4 from 1h bars
    if config.TIMEFRAME == "H4":
        df = df.set_index("time").resample("4h").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna().reset_index()

    print(f"[DataLoader] Yahoo: {len(df):,} rows loaded.")
    return df


# ---------------------------------------------------------------------------
# Source 1: MetaTrader 5
# ---------------------------------------------------------------------------
def _load_from_mt5() -> pd.DataFrame:
    import MetaTrader5 as mt5

    TIMEFRAME_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }

    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialize() failed: {mt5.last_error()}")

    tf = TIMEFRAME_MAP.get(config.TIMEFRAME, mt5.TIMEFRAME_M15)
    rates = mt5.copy_rates_from_pos(config.SYMBOL, tf, 0, config.N_BARS)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise ValueError("MT5 returned no data.")

    df = pd.DataFrame(rates)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "volume"]].copy()


# ---------------------------------------------------------------------------
# Source 2: CSV file
# ---------------------------------------------------------------------------
def _load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df[["time", "open", "high", "low", "close", "volume"]].copy()


# ---------------------------------------------------------------------------
# Source 3: Synthetic EURUSD-like data (for offline / CI use)
# ---------------------------------------------------------------------------
def _generate_synthetic(n: int = 15_000, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic-looking EURUSD M15 OHLCV data using a correlated
    random walk with a slight mean-reversion component.
    """
    rng = np.random.default_rng(seed)

    # Geometric Brownian Motion parameters (annualised, scaled to M15)
    mu_annual    = 0.02          # slight upward drift
    sigma_annual = 0.07          # typical FX vol
    bars_per_year = 252 * 24 * 4  # M15 bars in a trading year

    dt = 1 / bars_per_year
    mu_dt    = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    sigma_dt = sigma_annual * np.sqrt(dt)

    # Simulate log-returns with mean reversion
    log_ret = mu_dt + sigma_dt * rng.standard_normal(n)
    prices  = 1.1000 * np.exp(np.cumsum(log_ret))

    # Build candles
    noise = sigma_dt * 0.5
    open_  = prices
    close_ = prices + sigma_dt * rng.standard_normal(n)
    high_  = np.maximum(open_, close_) + np.abs(rng.normal(0, noise, n))
    low_   = np.minimum(open_, close_) - np.abs(rng.normal(0, noise, n))
    vol    = rng.integers(500, 3000, n).astype(float)

    # Timestamps: Mon–Fri trading hours only (simplified continuous)
    start = datetime(2020, 1, 6, 0, 0)   # Monday
    times = [start + timedelta(minutes=15 * i) for i in range(n)]

    df = pd.DataFrame({
        "time":   times,
        "open":   np.round(open_,  5),
        "high":   np.round(high_,  5),
        "low":    np.round(low_,   5),
        "close":  np.round(close_, 5),
        "volume": vol,
    })
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_data(
    source: str = "auto",
    csv_path: str = "data/eurusd_m15.csv",
) -> pd.DataFrame:
    """
    Load historical OHLCV data.

    Parameters
    ----------
    source : str
        "auto"      – try MT5 → Yahoo Finance → CSV → synthetic (default)
        "mt5"       – MetaTrader 5 only
        "yahoo"     – Yahoo Finance (requires yfinance)
        "csv"       – CSV file only
        "synthetic" – always use synthetic generator
    csv_path : str
        Path to CSV file (used when source in {"auto", "csv"}).

    Returns
    -------
    pd.DataFrame with columns: time, open, high, low, close, volume
    """
    df = None

    if source in ("auto", "mt5") and _mt5_available():
        try:
            print("[DataLoader] Loading from MetaTrader 5 …")
            df = _load_from_mt5()
            print(f"[DataLoader] MT5: {len(df):,} rows loaded.")
        except Exception as e:
            print(f"[DataLoader] MT5 failed ({e}). Falling back …")

    if df is None and source in ("auto", "yahoo") and _yfinance_available():
        try:
            df = _load_from_yahoo()
        except Exception as e:
            print(f"[DataLoader] Yahoo Finance failed ({e}). Falling back …")

    if df is None and source in ("auto", "csv"):
        if os.path.exists(csv_path):
            try:
                print(f"[DataLoader] Loading from CSV: {csv_path} …")
                df = _load_from_csv(csv_path)
                print(f"[DataLoader] CSV: {len(df):,} rows loaded.")
            except Exception as e:
                print(f"[DataLoader] CSV failed ({e}). Falling back …")

    if df is None:
        print("[DataLoader] Using synthetic data generator …")
        df = _generate_synthetic(n=config.N_BARS)
        print(f"[DataLoader] Synthetic: {len(df):,} rows generated.")

    df = _clean(df)
    print(f"[DataLoader] Final dataset: {len(df):,} rows | {df['time'].min()} → {df['time'].max()}")
    return df


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop rows with any NaN in price columns
    price_cols = ["open", "high", "low", "close"]
    df.dropna(subset=price_cols, inplace=True)

    # Forward-fill volume if missing
    df["volume"] = df["volume"].fillna(0.0)

    # Sanity: high >= max(open, close), low <= min(open, close)
    df["high"]  = df[["high",  "open", "close"]].max(axis=1)
    df["low"]   = df[["low",   "open", "close"]].min(axis=1)

    # Remove duplicate timestamps
    df.drop_duplicates(subset="time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
# CLI helper: save synthetic data as CSV for inspection
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(source="synthetic")
    out = "data/eurusd_m15_synthetic.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved to {out}")
    print(df.tail())
