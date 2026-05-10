#!/usr/bin/env python3
# =============================================================================
# signal_monitor.py — Runs on Mac (no MT5 needed)
#
# Checks EURUSD signal every hour using Yahoo Finance data.
# Sends full trade alerts to Telegram when signal fires.
#
# Usage:
#   python3 signal_monitor.py            # live monitor (checks every hour)
#   python3 signal_monitor.py --now      # check once immediately and exit
# =============================================================================

import time
import argparse
from datetime import datetime

import config
from ai_training.data_loader  import load_data
from ai_training.model        import predict_signal, load_inference_bundle
from risk.risk_manager        import RiskManager
from notifications.telegram_alert import (
    alert_startup, alert_signal, alert_trade_placed,
    alert_risk_blocked, alert_stopped,
)


def check_signal(model, scaler, feature_cols, rm) -> dict:
    """Fetch latest data and return signal info."""
    df = load_data(source="yahoo")
    window = df.tail(300).copy()
    sig = predict_signal(window, model, scaler, feature_cols)

    last_price = df["close"].iloc[-1]
    sl, tp     = rm.get_sl_tp(sig["signal"], last_price)

    balance    = config.INITIAL_BALANCE
    stop_pips  = config.STOP_LOSS_PIPS
    risk_amt   = balance * config.RISK_PER_TRADE
    pip_value  = 10.0
    lots       = max(0.01, round(risk_amt / (stop_pips * pip_value), 2))

    return {
        **sig,
        "entry":      last_price,
        "sl":         sl,
        "tp":         tp,
        "lots":       lots,
        "balance":    balance,
        "timestamp":  datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }


def run(check_now: bool = False):
    print("=" * 55)
    print("  AI FOREX SIGNAL MONITOR (Yahoo Finance / No MT5)")
    print(f"  Symbol: {config.SYMBOL} | TF: {config.TIMEFRAME}")
    print(f"  Threshold: {config.CONFIDENCE_THRESHOLD:.0%}")
    print("=" * 55)

    # Load trained model
    try:
        model, scaler, feature_cols = load_inference_bundle()
    except FileNotFoundError:
        print("\nNo trained model found. Run first:")
        print("  python3 main.py --source yahoo")
        return

    rm = RiskManager()

    alert_startup(
        config.SYMBOL, config.TIMEFRAME,
        "SIGNAL MONITOR (simulated — no real orders)"
    )

    sleep_secs = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}.get(
        config.TIMEFRAME, 3600
    )

    if check_now:
        # Single check and exit
        _process(model, scaler, feature_cols, rm)
        return

    print(f"\nChecking every {sleep_secs // 60} minutes. Press Ctrl+C to stop.\n")

    try:
        while True:
            _process(model, scaler, feature_cols, rm)
            print(f"  Sleeping {sleep_secs // 60} min until next check …\n")
            time.sleep(sleep_secs)
    except KeyboardInterrupt:
        print("\nStopped.")
        alert_stopped()


def _process(model, scaler, feature_cols, rm):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"[{now}] Checking signal …")

    try:
        info = check_signal(model, scaler, feature_cols, rm)
    except Exception as e:
        print(f"  Error fetching data: {e}")
        return

    signal     = info["signal"]
    confidence = info["confidence"]
    filtered   = info["filtered"]

    print(f"  Signal: {signal} | Confidence: {confidence:.2%} | "
          f"{'ABOVE threshold' if filtered else 'below threshold'}")

    # Always send signal alert
    alert_signal(signal, confidence, config.SYMBOL, filtered)

    if not filtered:
        print("  No trade — confidence below threshold.")
        return

    # Risk check
    approval = rm.check_trade(signal, info["balance"], 0, confidence)
    if not approval["approved"]:
        print(f"  Risk check blocked: {approval['reason']}")
        alert_risk_blocked(approval["reason"])
        return

    # Send full trade alert to Telegram
    print(f"  TRADE SIGNAL | Entry: {info['entry']:.5f} | "
          f"SL: {info['sl']:.5f} | TP: {info['tp']:.5f} | Lots: {info['lots']}")

    alert_trade_placed(
        signal        = signal,
        symbol        = config.SYMBOL,
        lots          = info["lots"],
        entry         = info["entry"],
        sl            = info["sl"],
        tp            = info["tp"],
        confidence    = confidence,
        dry_run       = True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--now", action="store_true",
                   help="Check once immediately and exit")
    p.add_argument("--strategy", default=None,
                   choices=["trend_follow", "rsi_reversal", "ema_cross", "breakout", "macd_signal"],
                   help="Override strategy from config.py")
    args = p.parse_args()

    if args.strategy:
        config.STRATEGY = args.strategy

    run(check_now=args.now)
