# =============================================================================
# ai_training/labels.py
#
# Label (target variable) generation.
#
# Strategy
# --------
#   The label predicts whether entering a LONG position now is profitable
#   after FORWARD_BARS candles, accounting for spread.
#
#   label = 1 (BUY)  if  future_close > current_close + spread_cost
#   label = 0 (SELL) otherwise
#
#   This avoids look-ahead bias: we only use future data that a trader
#   actually COULD have at bar close.
#
# Public API
# ----------
#   create_labels(df, forward_bars, spread_pips) → pd.DataFrame
# =============================================================================

import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def create_labels(
    df: pd.DataFrame,
    forward_bars: int = config.FORWARD_BARS,
    spread_pips:  float = config.SPREAD_PIPS,
) -> pd.DataFrame:
    """
    Append a 'target' column to df.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'close' column.
    forward_bars : int
        Number of candles ahead to look for the outcome.
    spread_pips : float
        Simulated spread in pips (deducted from return).

    Returns
    -------
    pd.DataFrame
        Copy of df with 'target' column added.
        Rows where the future close is unavailable (last forward_bars rows)
        are dropped.
    """
    d = df.copy()

    spread_price = spread_pips * config.PIP_VALUE

    # Future close price (shifted back by forward_bars)
    future_close = d["close"].shift(-forward_bars)

    # Return after accounting for spread
    future_return = future_close - d["close"] - spread_price

    # Binary label: 1 = profitable long, 0 = not profitable (short or flat)
    d["target"] = (future_return > 0).astype(int)

    # Also store continuous return for analysis
    d["future_return"] = future_return

    # Drop the last forward_bars rows (no label available)
    d = d.iloc[:-forward_bars].copy()
    d.reset_index(drop=True, inplace=True)

    buy_pct = d["target"].mean() * 100
    print(f"[Labels] forward_bars={forward_bars} | BUY%={buy_pct:.1f}% | SELL%={100-buy_pct:.1f}%")

    return d


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, "..")
    from ai_training.data_loader import load_data
    from ai_training.features import create_features

    df = load_data(source="synthetic")
    df = create_features(df)
    df = create_labels(df)
    print(df[["time", "close", "target", "future_return"]].tail(10))
    print(f"\nClass balance:\n{df['target'].value_counts()}")
