#!/usr/bin/env python3
# =============================================================================
# main.py — AI Forex Trading Bot: Full Pipeline Runner
#
# Usage
# -----
#   python main.py                    # train + backtest (synthetic data)
#   python main.py --source csv       # use CSV file in data/
#   python main.py --source mt5       # use live MT5 connection
#   python main.py --backtest-only    # skip training, load saved model
#   python main.py --no-plot          # skip chart generation
#
# =============================================================================

import argparse
import os
import sys

import config
from ai_training.data_loader  import load_data
from ai_training.features     import create_features, get_feature_cols
from ai_training.labels       import create_labels
from ai_training.train        import train_model, save_model, load_model
from ai_training.model        import predict_batch, load_inference_bundle
from ai_training.features     import load_scaler
from backtesting.backtester   import Backtester
from backtesting.metrics      import compute_metrics, print_report
from backtesting.visualization import plot_results
from risk.risk_manager        import RiskManager


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AI Forex Trading Bot Pipeline")
    p.add_argument("--source",         default="auto",
                   choices=["auto", "mt5", "yahoo", "csv", "synthetic"],
                   help="Data source (default: auto)")
    p.add_argument("--strategy",       default=None,
                   choices=["trend_follow", "rsi_reversal", "ema_cross", "breakout", "macd_signal"],
                   help="Trading strategy (default: from config.py)")
    p.add_argument("--csv-path",       default="data/eurusd_m15.csv",
                   help="Path to CSV if --source=csv")
    p.add_argument("--backtest-only",  action="store_true",
                   help="Skip training; load existing model from disk")
    p.add_argument("--no-plot",        action="store_true",
                   help="Skip chart generation")
    p.add_argument("--output-dir",     default="output",
                   help="Directory for output charts and reports")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Apply strategy override from CLI
    if args.strategy:
        config.STRATEGY = args.strategy

    print("=" * 60)
    print("  AI FOREX TRADING BOT — Full Pipeline")
    print(f"  Symbol: {config.SYMBOL} | TF: {config.TIMEFRAME}")
    print(f"  Strategy: {config.STRATEGY}")
    print(f"  Initial balance: ${config.INITIAL_BALANCE:,.0f}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # STEP 1: Load data once (shared by training + backtest)
    # ----------------------------------------------------------------
    print("\n[Main] Loading data …")
    df_raw = load_data(source=args.source)

    # ----------------------------------------------------------------
    # STEP 2: Train (or load) model
    # ----------------------------------------------------------------
    if args.backtest_only:
        print("\n[Main] Loading existing model …")
        model, scaler, feature_cols = load_inference_bundle()
    else:
        print("\n[Main] Running training pipeline …")
        model, scaler, train_report, df_test = train_model(data_source=args.source, df=df_raw)

        # Attach feature names to model for future loading
        model._feature_cols = train_report["features"]
        save_model(model)

        feature_cols = train_report["features"]

        print(f"\n[Main] Training complete.")
        print(f"       Accuracy : {train_report['accuracy']:.4f}")
        print(f"       AUC-ROC  : {train_report['auc_roc']:.4f}")
        print(f"       F1-Score : {train_report['f1']:.4f}")

    # ----------------------------------------------------------------
    # STEP 2b: Run backtest
    # ----------------------------------------------------------------
    print("\n[Main] Initialising backtester …")
    rm = RiskManager(
        risk_per_trade       = config.RISK_PER_TRADE,
        stop_loss_pips       = config.STOP_LOSS_PIPS,
        take_profit_pips     = config.TAKE_PROFIT_PIPS,
        confidence_threshold = config.CONFIDENCE_THRESHOLD,
    )
    backtester = Backtester(
        model           = model,
        scaler          = scaler,
        feature_cols    = feature_cols,
        risk_manager    = rm,
        initial_balance = config.INITIAL_BALANCE,
    )

    result = backtester.run(df_raw)

    # ----------------------------------------------------------------
    # STEP 4: Compute and display metrics
    # ----------------------------------------------------------------
    metrics = compute_metrics(result)
    print_report(metrics)

    # ----------------------------------------------------------------
    # STEP 5: Visualisation
    # ----------------------------------------------------------------
    if not args.no_plot:
        chart_path = os.path.join(args.output_dir, "backtest_report.png")
        print(f"\n[Main] Generating charts → {chart_path}")
        plot_results(result, metrics, save_path=chart_path)

    # ----------------------------------------------------------------
    # STEP 6: Demo — single-bar real-time prediction
    # ----------------------------------------------------------------
    print("\n[Main] Demo: real-time signal for latest bar …")
    from ai_training.model import predict_signal
    from notifications.telegram_alert import alert_current_signal
    window = df_raw.tail(300).copy()
    signal_info = predict_signal(window, model, scaler, feature_cols)
    print(f"\n  Signal     : {signal_info['signal']}")
    print(f"  Confidence : {signal_info['confidence']:.2%}")
    print(f"  Buy Prob   : {signal_info['buy_prob']:.2%}")
    print(f"  Sell Prob  : {signal_info['sell_prob']:.2%}")
    print(f"  Filtered   : {'YES — TRADE ALLOWED' if signal_info['filtered'] else 'NO — BELOW THRESHOLD'}")
    alert_current_signal(
        signal_info["signal"], signal_info["confidence"],
        config.SYMBOL, signal_info["filtered"], config.TIMEFRAME,
    )

    print("\n[Main] Pipeline complete.\n")
    return metrics


if __name__ == "__main__":
    main()
