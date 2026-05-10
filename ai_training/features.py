# =============================================================================
# ai_training/features.py
#
# Feature engineering for the AI trading model.
#
# Indicators computed
# -------------------
#   EMA          – 10, 50, 200 period
#   RSI          – 14 period
#   MACD         – 12/26/9
#   Bollinger Bands – 20/2
#   Price returns  – 1, 5, 10 bar log returns
#   Candle shape   – body size, upper/lower wick ratios
#   Volume         – normalised rolling z-score
#
# Public API
# ----------
#   create_features(df) → pd.DataFrame   (original df + feature columns)
#   FEATURE_COLS        → list[str]       (names of feature columns)
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ---------------------------------------------------------------------------
# Low-level indicator helpers (vectorised, no external TA-lib dependency)
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, period=20, std_dev=2.0):
    mid  = close.rolling(period).mean()
    std  = close.rolling(period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    # %B: position within bands [0 = lower, 1 = upper]
    pct_b = (close - lower) / (upper - lower + 1e-10)
    # Bandwidth: normalised band width
    bw    = (upper - lower) / (mid + 1e-10)
    return mid, upper, lower, pct_b, bw


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ---------------------------------------------------------------------------
# Main feature creator
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all technical indicator features to the DataFrame in-place (copy).

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume

    Returns
    -------
    pd.DataFrame
        Original df plus all feature columns. Rows with NaN features are
        dropped (first ~200 rows due to slow indicators like EMA-200).
    """
    d = df.copy()
    c = d["close"]
    h = d["high"]
    lo = d["low"]
    o = d["open"]

    # ------------------------------------------------------------------ EMA
    d["ema10"]  = _ema(c, config.EMA_FAST)
    d["ema50"]  = _ema(c, config.EMA_MID)
    d["ema200"] = _ema(c, config.EMA_SLOW)

    # EMA crossover signals (normalised distance)
    d["ema10_50_diff"]  = (d["ema10"]  - d["ema50"])  / c
    d["ema50_200_diff"] = (d["ema50"]  - d["ema200"]) / c
    d["price_ema10"]    = (c - d["ema10"])  / c
    d["price_ema50"]    = (c - d["ema50"])  / c
    d["price_ema200"]   = (c - d["ema200"]) / c

    # ------------------------------------------------------------------ RSI
    d["rsi"] = _rsi(c, config.RSI_PERIOD)
    # Overbought / oversold distance
    d["rsi_ob"] = (d["rsi"] - 70).clip(lower=0)   # above 70
    d["rsi_os"] = (30 - d["rsi"]).clip(lower=0)   # below 30

    # ----------------------------------------------------------------- MACD
    d["macd"], d["macd_signal"], d["macd_hist"] = _macd(
        c, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIG
    )
    # Normalise by price level
    d["macd"]        /= c
    d["macd_signal"] /= c
    d["macd_hist"]   /= c

    # ---------------------------------------------------- Bollinger Bands
    d["bb_mid"], d["bb_upper"], d["bb_lower"], d["bb_pct"], d["bb_width"] = \
        _bollinger(c, config.BB_PERIOD, config.BB_STD)
    # Drop raw mid/upper/lower (use derived instead)
    d.drop(columns=["bb_mid", "bb_upper", "bb_lower"], inplace=True)

    # ------------------------------------------------------- Price returns
    for lag in [1, 3, 5, 10, 20]:
        d[f"ret_{lag}"] = np.log(c / c.shift(lag))

    # --------------------------------------------------- Candle morphology
    range_  = (h - lo).replace(0, np.nan)
    d["body"]        = (c - o).abs() / range_         # body as fraction of range
    d["upper_wick"]  = (h - pd.concat([c, o], axis=1).max(axis=1)) / range_
    d["lower_wick"]  = (pd.concat([c, o], axis=1).min(axis=1) - lo) / range_
    d["bull_candle"]  = (c > o).astype(float)

    # ------------------------------------------------------------- Volume
    vol_roll_mean = d["volume"].rolling(20).mean()
    vol_roll_std  = d["volume"].rolling(20).std(ddof=0) + 1e-10
    d["vol_zscore"] = (d["volume"] - vol_roll_mean) / vol_roll_std

    # ---------------------------------------------------------------- ATR
    d["atr"] = _atr(h, lo, c, 14)
    d["atr_pct"] = d["atr"] / c                      # normalised

    # ------------------------------------------------------ Momentum / ROC
    d["roc5"]  = (c - c.shift(5))  / (c.shift(5)  + 1e-10)
    d["roc10"] = (c - c.shift(10)) / (c.shift(10) + 1e-10)

    # ------------------------------------------- Drop raw price / OHLC cols
    # Keep them in df for labelling and backtesting, just exclude from FEATURE_COLS

    # Drop rows that have NaN in any feature column
    feature_cols = _get_feature_cols(d)
    d.dropna(subset=feature_cols, inplace=True)
    d.reset_index(drop=True, inplace=True)

    return d


# ---------------------------------------------------------------------------
# Feature column list (everything that is NOT an original OHLCV or metadata col)
# ---------------------------------------------------------------------------
_ORIGINAL_COLS = {"time", "open", "high", "low", "close", "volume"}
_EXCLUDE_COLS  = {"target", "future_return", "strat_signal",
                  "ai_pred", "confidence", "buy_prob", "sell_prob",
                  "pred", "filtered"}

def _get_feature_cols(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if c not in _ORIGINAL_COLS
        and c not in _EXCLUDE_COLS
        and not c.startswith("target")
        and not c.startswith("bb_mid")   # raw BB mid is dropped during feature creation
        and not c.startswith("bb_upper")
        and not c.startswith("bb_lower")
    ]

# Convenience module-level constant (populated after first call, or call explicitly)
FEATURE_COLS: list = []   # filled by get_feature_cols()

def get_feature_cols(df: pd.DataFrame) -> list:
    """Return the list of feature column names for a featurised DataFrame."""
    global FEATURE_COLS
    FEATURE_COLS = _get_feature_cols(df)
    return FEATURE_COLS


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def fit_scaler(X: pd.DataFrame, scaler_path: str = config.SCALER_PATH) -> RobustScaler:
    """Fit a RobustScaler on training features and persist it."""
    scaler = RobustScaler()
    scaler.fit(X)
    os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[Features] Scaler saved to {scaler_path}")
    return scaler


def load_scaler(scaler_path: str = config.SCALER_PATH) -> RobustScaler:
    """Load a previously fitted scaler."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Train the model first.")
    return joblib.load(scaler_path)


def scale_features(X: pd.DataFrame, scaler: RobustScaler) -> np.ndarray:
    """Transform feature matrix using a fitted scaler."""
    return scaler.transform(X)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from ai_training.data_loader import load_data

    df = load_data(source="synthetic")
    df = create_features(df)
    cols = get_feature_cols(df)
    print(f"Features created: {len(cols)}")
    print(cols)
    print(df[cols].describe().T[["mean", "std", "min", "max"]])
