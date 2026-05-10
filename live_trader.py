#!/usr/bin/env python3
# =============================================================================
# live_trader.py — Connects to MT5 and trades live using the trained AI model
#
# Run AFTER training:
#   python3 main.py --source mt5        # train on real data first
#   python3 live_trader.py              # then run live
#
# What it does every 15 minutes (on candle close):
#   1. Fetch latest bars from MT5
#   2. Run feature engineering
#   3. Get AI signal + confidence
#   4. Check risk management rules
#   5. Send BUY / SELL order to MT5 if approved
#   6. Manage open position (SL/TP handled by MT5 natively)
# =============================================================================

import time
import sys
from datetime import datetime

import pandas as pd
import numpy as np

import config
from ai_training.model   import predict_signal, load_inference_bundle
from risk.risk_manager   import RiskManager
from notifications.telegram_alert import (
    alert_startup, alert_signal, alert_trade_placed,
    alert_position_open, alert_risk_blocked, alert_stopped,
)

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed.")
    print("Run: pip install MetaTrader5")
    sys.exit(1)


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------

TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
}

def connect_mt5() -> bool:
    """Initialise MT5 connection."""
    if not mt5.initialize():
        print(f"[MT5] initialize() failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    print(f"[MT5] Connected: Account #{info.login} | "
          f"Balance: ${info.balance:,.2f} | Server: {info.server}")
    return True


def get_latest_bars(symbol: str, timeframe: str, n: int = 300) -> pd.DataFrame:
    """Fetch the latest N closed bars from MT5."""
    tf  = TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M15)
    # n+1 to exclude the currently-forming (unclosed) candle
    rates = mt5.copy_rates_from_pos(symbol, tf, 1, n)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["time", "open", "high", "low", "close", "volume"]].copy()


def get_open_position(symbol: str):
    """Return the open position for this symbol, or None."""
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        return positions[0]
    return None


def get_pip_size(symbol: str) -> float:
    """Return pip size for the symbol."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return config.PIP_VALUE
    # For most FX pairs: 1 pip = 10 * point
    return info.point * 10


def send_order(
    symbol:      str,
    signal:      str,
    lot_size:    float,
    stop_loss:   float,
    take_profit: float,
    comment:     str = "AI Bot",
) -> bool:
    """
    Send a market order to MT5.

    Parameters
    ----------
    signal : "BUY" or "SELL"
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"[MT5] Cannot get tick for {symbol}")
        return False

    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    price      = tick.ask             if signal == "BUY" else tick.bid

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       float(round(lot_size, 2)),
        "type":         order_type,
        "price":        price,
        "sl":           round(stop_loss,   5),
        "tp":           round(take_profit, 5),
        "deviation":    10,                    # max slippage in points
        "magic":        202400,                # unique EA identifier
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[MT5] ✓ {signal} order placed | Lot: {lot_size} | "
              f"Price: {price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
        return True
    else:
        print(f"[MT5] ✗ Order failed | retcode={result.retcode} | {result.comment}")
        return False


def close_position(symbol: str) -> bool:
    """Close an open position by sending the opposite market order."""
    pos = get_open_position(symbol)
    if pos is None:
        return False

    tick = mt5.symbol_info_tick(symbol)
    close_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = tick.bid            if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       pos.volume,
        "type":         close_type,
        "position":     pos.ticket,
        "price":        close_price,
        "deviation":    10,
        "magic":        202400,
        "comment":      "AI Bot Close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[MT5] Position closed.")
        return True
    print(f"[MT5] Close failed: {result.retcode} {result.comment}")
    return False


# ---------------------------------------------------------------------------
# Lot size calculator (real MT5 account)
# ---------------------------------------------------------------------------

def calculate_lot_size(
    balance:    float,
    stop_pips:  float,
    risk_pct:   float = config.RISK_PER_TRADE,
    symbol:     str   = config.SYMBOL,
) -> float:
    """
    Calculate lot size based on fixed fractional risk.

    Formula: lots = (balance * risk_pct) / (stop_pips * pip_value_per_lot)
    For EURUSD: 1 standard lot = $10/pip → pip_value_per_lot = $10
    """
    pip_value_per_lot = 10.0   # USD per pip per standard lot (EURUSD)
    risk_amount = balance * risk_pct
    lots = risk_amount / (stop_pips * pip_value_per_lot)
    # Clamp to broker minimum/maximum
    lots = max(0.01, round(lots, 2))
    lots = min(lots, 10.0)     # safety cap at 10 lots
    return lots


# ---------------------------------------------------------------------------
# Main trading loop
# ---------------------------------------------------------------------------

def run_live(dry_run: bool = False):
    """
    Main live trading loop.

    Parameters
    ----------
    dry_run : bool
        If True, log signals but do NOT send orders to MT5.
        Use this to validate before going live.
    """
    print("=" * 60)
    print("  AI FOREX LIVE TRADER")
    print(f"  Symbol: {config.SYMBOL} | TF: {config.TIMEFRAME}")
    print(f"  Mode: {'DRY RUN (no real orders)' if dry_run else '⚠ LIVE TRADING'}")
    print("=" * 60)

    # Load trained model
    print("\n[Live] Loading model …")
    model, scaler, feature_cols = load_inference_bundle()

    rm = RiskManager()

    # Connect to MT5
    if not connect_mt5():
        return

    alert_startup(config.SYMBOL, config.TIMEFRAME,
                  "DRY RUN (no real orders)" if dry_run else "LIVE TRADING")

    # How many seconds per bar (M15 = 900s)
    bar_seconds = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}
    sleep_secs  = bar_seconds.get(config.TIMEFRAME, 900)

    print(f"\n[Live] Checking every {sleep_secs//60} minutes on candle close. Ctrl+C to stop.\n")

    try:
        while True:
            now = datetime.utcnow()
            print(f"\n{'─'*50}")
            print(f"[{now.strftime('%Y-%m-%d %H:%M UTC')}] Checking …")

            # 1. Fetch recent bars
            df = get_latest_bars(config.SYMBOL, config.TIMEFRAME, n=300)
            if df.empty:
                print("[Live] No data received. Retrying next bar …")
                time.sleep(sleep_secs)
                continue

            # 2. AI prediction
            sig_info = predict_signal(df, model, scaler, feature_cols)
            signal     = sig_info["signal"]
            confidence = sig_info["confidence"]
            filtered   = sig_info["filtered"]

            print(f"  Signal: {signal} | Confidence: {confidence:.2%} | "
                  f"{'✓ ABOVE THRESHOLD' if filtered else '✗ below threshold'}")

            alert_signal(signal, confidence, config.SYMBOL, filtered)

            # 3. Check for open position
            open_pos = get_open_position(config.SYMBOL)
            if open_pos:
                pos_type = "BUY" if open_pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                print(f"  Open {pos_type} position | P&L: ${open_pos.profit:+.2f} — MT5 manages SL/TP")
                alert_position_open(config.SYMBOL, pos_type, open_pos.profit)
                time.sleep(sleep_secs)
                continue

            # 4. Risk management approval
            acct     = mt5.account_info()
            balance  = acct.balance
            approval = rm.check_trade(signal, balance, 0, confidence)

            if not approval["approved"]:
                print(f"  Risk check: {approval['reason']}")
                alert_risk_blocked(approval["reason"])
                time.sleep(sleep_secs)
                continue

            # 5. Calculate SL / TP and lot size
            last_price = df["close"].iloc[-1]
            sl, tp     = rm.get_sl_tp(signal, last_price)
            stop_pips  = config.STOP_LOSS_PIPS
            lots       = calculate_lot_size(balance, stop_pips)

            print(f"  Entry: {last_price:.5f} | SL: {sl:.5f} | TP: {tp:.5f} | Lots: {lots}")

            # 6. Send order (or simulate)
            if dry_run:
                print(f"  [DRY RUN] Would send {signal} | {lots} lots")
                alert_trade_placed(signal, config.SYMBOL, lots,
                                   last_price, sl, tp, confidence, dry_run=True)
            else:
                placed = send_order(
                    symbol      = config.SYMBOL,
                    signal      = signal,
                    lot_size    = lots,
                    stop_loss   = sl,
                    take_profit = tp,
                    comment     = f"AI {signal} {confidence:.0%}",
                )
                if placed:
                    alert_trade_placed(signal, config.SYMBOL, lots,
                                       last_price, sl, tp, confidence, dry_run=False)

            # Sleep until next bar close
            print(f"  Sleeping {sleep_secs//60}m until next bar …")
            time.sleep(sleep_secs)

    except KeyboardInterrupt:
        print("\n[Live] Stopped by user.")
        alert_stopped()
    finally:
        mt5.shutdown()
        print("[Live] MT5 connection closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true",
                   help="Send real orders (default: dry run)")
    args = p.parse_args()

    run_live(dry_run=not args.live)
