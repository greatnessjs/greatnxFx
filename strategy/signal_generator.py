# =============================================================================
# strategy/signal_generator.py
#
# Multi-strategy signal generator. Switch strategies via config.STRATEGY
# or pass strategy_name directly to generate_signals_batch().
#
# Available strategies
# --------------------
#   trend_follow  — EMA cross + RSI + MACD + Bollinger Bands (default)
#   rsi_reversal  — Buy oversold (RSI<30), Sell overbought (RSI>70)
#   ema_cross     — Simple EMA10/EMA50 crossover only
#   breakout      — Price breaks above/below recent N-bar high/low
#   macd_signal   — MACD line crosses above/below signal line
#
# Public API
# ----------
#   generate_signals_batch(df, strategy_name=None) → pd.Series
#   generate_signal(row, prev_row, strategy_name=None) → "BUY"|"SELL"|"HOLD"
#   list_strategies() → list[str]
# =============================================================================

import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from ai_training.features import create_features, _ema


# ---------------------------------------------------------------------------
# Shared indicator builder
# ---------------------------------------------------------------------------

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]

    if "ema10" not in d.columns:
        d["ema10"]  = _ema(c, config.EMA_FAST)
        d["ema50"]  = _ema(c, config.EMA_MID)
        d["ema200"] = _ema(c, config.EMA_SLOW)

    if "rsi" not in d.columns:
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(com=config.RSI_PERIOD - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=config.RSI_PERIOD - 1, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        d["rsi"] = 100 - (100 / (1 + rs))

    if "macd_hist" not in d.columns:
        ema_f        = _ema(c, config.MACD_FAST)
        ema_s        = _ema(c, config.MACD_SLOW)
        macd         = ema_f - ema_s
        signal_line  = _ema(macd, config.MACD_SIG)
        d["macd"]      = macd
        d["macd_sig"]  = signal_line
        d["macd_hist"] = macd - signal_line

    if "bb_mid" not in d.columns:
        d["bb_mid"]   = c.rolling(config.BB_PERIOD).mean()
        rolling_std   = c.rolling(config.BB_PERIOD).std()
        d["bb_upper"] = d["bb_mid"] + config.BB_STD * rolling_std
        d["bb_lower"] = d["bb_mid"] - config.BB_STD * rolling_std

    return d


# ---------------------------------------------------------------------------
# Strategy 1 — Trend Follow (default)
# EMA trend + EMA cross + RSI + MACD + Bollinger Bands
# ---------------------------------------------------------------------------

def _strategy_trend_follow(d: pd.DataFrame) -> pd.Series:
    c = d["close"]

    uptrend   = (c > d["ema50"]) & (d["ema50"] > d["ema200"])
    downtrend = (c < d["ema50"]) & (d["ema50"] < d["ema200"])

    ema10_prev = d["ema10"].shift(1)
    ema50_prev = d["ema50"].shift(1)
    cross_up   = (ema10_prev <= ema50_prev) & (d["ema10"] > d["ema50"])
    cross_down = (ema10_prev >= ema50_prev) & (d["ema10"] < d["ema50"])

    recent_cross_up   = cross_up.rolling(3).max().fillna(0).astype(bool)
    recent_cross_down = cross_down.rolling(3).max().fillna(0).astype(bool)

    rsi_buy  = (d["rsi"] >= 40) & (d["rsi"] <= 65)
    rsi_sell = (d["rsi"] >= 35) & (d["rsi"] <= 60)

    macd_bull = d["macd_hist"] > 0
    macd_bear = d["macd_hist"] < 0

    above_mid = c > d["bb_mid"]
    below_mid = c < d["bb_mid"]

    buy_cond  = uptrend   & recent_cross_up   & rsi_buy  & macd_bull & above_mid
    sell_cond = downtrend & recent_cross_down  & rsi_sell & macd_bear & below_mid

    signals = pd.Series("HOLD", index=d.index)
    signals[buy_cond]  = "BUY"
    signals[sell_cond] = "SELL"
    return signals


# ---------------------------------------------------------------------------
# Strategy 2 — RSI Reversal
# Buy when oversold, Sell when overbought — mean reversion approach
# ---------------------------------------------------------------------------

def _strategy_rsi_reversal(d: pd.DataFrame) -> pd.Series:
    c = d["close"]

    # Buy: RSI deeply oversold + price touching lower Bollinger Band
    buy_cond  = (d["rsi"] < 30) & (c <= d["bb_lower"] * 1.001)

    # Sell: RSI deeply overbought + price touching upper Bollinger Band
    sell_cond = (d["rsi"] > 70) & (c >= d["bb_upper"] * 0.999)

    signals = pd.Series("HOLD", index=d.index)
    signals[buy_cond]  = "BUY"
    signals[sell_cond] = "SELL"
    return signals


# ---------------------------------------------------------------------------
# Strategy 3 — EMA Cross (simple)
# Just EMA10 crossing EMA50 — clean and straightforward
# ---------------------------------------------------------------------------

def _strategy_ema_cross(d: pd.DataFrame) -> pd.Series:
    ema10_prev = d["ema10"].shift(1)
    ema50_prev = d["ema50"].shift(1)

    cross_up   = (ema10_prev <= ema50_prev) & (d["ema10"] > d["ema50"])
    cross_down = (ema10_prev >= ema50_prev) & (d["ema10"] < d["ema50"])

    # Allow signal to persist for 2 bars after the cross
    buy_cond  = cross_up.rolling(2).max().fillna(0).astype(bool)
    sell_cond = cross_down.rolling(2).max().fillna(0).astype(bool)

    signals = pd.Series("HOLD", index=d.index)
    signals[buy_cond]  = "BUY"
    signals[sell_cond] = "SELL"
    return signals


# ---------------------------------------------------------------------------
# Strategy 4 — Breakout
# Price breaks above recent high or below recent low with momentum
# ---------------------------------------------------------------------------

BREAKOUT_PERIOD = 20   # bars to look back for high/low

def _strategy_breakout(d: pd.DataFrame) -> pd.Series:
    c = d["close"]

    prev_high = d["high"].shift(1).rolling(BREAKOUT_PERIOD).max()
    prev_low  = d["low"].shift(1).rolling(BREAKOUT_PERIOD).min()

    # Breakout up: close above previous N-bar high + MACD confirming
    buy_cond  = (c > prev_high) & (d["macd_hist"] > 0)

    # Breakout down: close below previous N-bar low + MACD confirming
    sell_cond = (c < prev_low)  & (d["macd_hist"] < 0)

    signals = pd.Series("HOLD", index=d.index)
    signals[buy_cond]  = "BUY"
    signals[sell_cond] = "SELL"
    return signals


# ---------------------------------------------------------------------------
# Strategy 5 — MACD Signal Cross
# MACD line crosses its signal line — classic momentum entry
# ---------------------------------------------------------------------------

def _strategy_macd_signal(d: pd.DataFrame) -> pd.Series:
    macd_prev    = d["macd"].shift(1)
    sig_prev     = d["macd_sig"].shift(1)

    cross_up   = (macd_prev <= sig_prev) & (d["macd"] > d["macd_sig"])
    cross_down = (macd_prev >= sig_prev) & (d["macd"] < d["macd_sig"])

    # Only trade in direction of EMA200 trend
    uptrend   = d["close"] > d["ema200"]
    downtrend = d["close"] < d["ema200"]

    buy_cond  = cross_up   & uptrend
    sell_cond = cross_down & downtrend

    signals = pd.Series("HOLD", index=d.index)
    signals[buy_cond]  = "BUY"
    signals[sell_cond] = "SELL"
    return signals


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "trend_follow": _strategy_trend_follow,
    "rsi_reversal": _strategy_rsi_reversal,
    "ema_cross":    _strategy_ema_cross,
    "breakout":     _strategy_breakout,
    "macd_signal":  _strategy_macd_signal,
}


def list_strategies() -> list:
    return list(_STRATEGIES.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_signals_batch(
    df: pd.DataFrame,
    strategy_name: str = None,
) -> pd.Series:
    """
    Generate signals for all rows using the selected strategy.

    Parameters
    ----------
    df            : OHLCV DataFrame
    strategy_name : override config.STRATEGY if provided

    Returns
    -------
    pd.Series of "BUY", "SELL", or "HOLD"
    """
    name = strategy_name or getattr(config, "STRATEGY", "trend_follow")

    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(_STRATEGIES.keys())}"
        )

    print(f"[Strategy] Using: {name}")
    d = _compute_indicators(df)
    return _STRATEGIES[name](d)


def generate_signal(
    row: pd.Series,
    prev_row: pd.Series,
    strategy_name: str = None,
) -> str:
    """Single-bar signal for real-time use."""
    name = strategy_name or getattr(config, "STRATEGY", "trend_follow")

    uptrend   = row["close"] > row.get("ema50", 0) > row.get("ema200", 0)
    downtrend = row["close"] < row.get("ema50", 0) < row.get("ema200", 0)

    if name == "rsi_reversal":
        if row.get("rsi", 50) < 30:   return "BUY"
        if row.get("rsi", 50) > 70:   return "SELL"

    elif name == "ema_cross":
        cross_up   = prev_row.get("ema10", 0) <= prev_row.get("ema50", 0) and row.get("ema10", 0) > row.get("ema50", 0)
        cross_down = prev_row.get("ema10", 0) >= prev_row.get("ema50", 0) and row.get("ema10", 0) < row.get("ema50", 0)
        if cross_up:   return "BUY"
        if cross_down: return "SELL"

    elif name == "breakout":
        pass   # batch-only; fall through to HOLD for single bar

    elif name == "macd_signal":
        cross_up   = prev_row.get("macd", 0) <= prev_row.get("macd_sig", 0) and row.get("macd", 0) > row.get("macd_sig", 0)
        cross_down = prev_row.get("macd", 0) >= prev_row.get("macd_sig", 0) and row.get("macd", 0) < row.get("macd_sig", 0)
        if cross_up   and uptrend:   return "BUY"
        if cross_down and downtrend: return "SELL"

    else:  # trend_follow (default)
        cross_up   = prev_row.get("ema10", 0) <= prev_row.get("ema50", 0) and row.get("ema10", 0) > row.get("ema50", 0)
        cross_down = prev_row.get("ema10", 0) >= prev_row.get("ema50", 0) and row.get("ema10", 0) < row.get("ema50", 0)
        rsi_buy    = 40 <= row.get("rsi", 50) <= 65
        rsi_sell   = 35 <= row.get("rsi", 50) <= 60
        macd_bull  = row.get("macd_hist", 0) > 0
        macd_bear  = row.get("macd_hist", 0) < 0
        above_mid  = row["close"] > row.get("bb_mid", 0)
        below_mid  = row["close"] < row.get("bb_mid", 0)

        if uptrend   and cross_up   and rsi_buy  and macd_bull and above_mid: return "BUY"
        if downtrend and cross_down and rsi_sell and macd_bear and below_mid: return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# CLI: list strategies or run a quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="List available strategies")
    p.add_argument("--test", default="trend_follow", help="Strategy to test")
    args = p.parse_args()

    if args.list:
        print("Available strategies:")
        for s in list_strategies():
            print(f"  • {s}")
    else:
        sys.path.insert(0, "..")
        from ai_training.data_loader import load_data
        df   = load_data(source="synthetic")
        sigs = generate_signals_batch(df, strategy_name=args.test)
        print(sigs.value_counts())
