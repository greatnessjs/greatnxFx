# =============================================================================
# backtesting/metrics.py
#
# Performance metrics for backtesting results.
#
# Metrics computed
# ----------------
#   total_pnl          – net profit/loss in USD
#   total_return_pct   – return on initial capital
#   n_trades           – number of closed trades
#   n_wins / n_losses  – breakdown
#   win_rate           – % of winning trades
#   avg_win / avg_loss – average P&L of winning / losing trades
#   profit_factor      – gross_profit / gross_loss
#   max_drawdown       – peak-to-trough in USD
#   max_drawdown_pct   – peak-to-trough as % of peak balance
#   sharpe_ratio       – annualised Sharpe (assumes M15 bars)
#   calmar_ratio       – total_return / max_drawdown_pct
#   expectancy         – average $ per trade
#
# Public API
# ----------
#   compute_metrics(result: BacktestResult) → dict
#   print_report(metrics: dict)
# =============================================================================

import numpy as np
import pandas as pd

# Bars per year for M15 (252 trading days × 24h × 4 bars/h)
BARS_PER_YEAR_M15 = 252 * 24 * 4


def compute_metrics(result) -> dict:  # result: BacktestResult
    """
    Compute all performance metrics from a BacktestResult.

    Returns
    -------
    dict of metric_name → value
    """
    trades = result.trades
    equity = result.equity_curve

    # ----------------------------------------------------------------- P&L
    pnl_series = [t.pnl for t in trades]
    total_pnl  = sum(pnl_series) if pnl_series else 0.0
    total_ret  = (result.final_balance - result.initial_balance) / result.initial_balance * 100

    # ---------------------------------------------------------------- Wins
    n_trades  = len(trades)
    wins      = [p for p in pnl_series if p > 0]
    losses    = [p for p in pnl_series if p <= 0]
    n_wins    = len(wins)
    n_losses  = len(losses)
    win_rate  = (n_wins / n_trades * 100) if n_trades > 0 else 0.0
    avg_win   = np.mean(wins)   if wins   else 0.0
    avg_loss  = np.mean(losses) if losses else 0.0

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    expectancy    = (total_pnl / n_trades) if n_trades > 0 else 0.0

    # ----------------------------------------------- Exit reason breakdown
    tp_count  = sum(1 for t in trades if t.exit_reason == "TP")
    sl_count  = sum(1 for t in trades if t.exit_reason == "SL")
    eod_count = sum(1 for t in trades if t.exit_reason == "EOD")

    # ----------------------------------------------------- Max Drawdown
    equity_vals = equity.values
    peak        = np.maximum.accumulate(equity_vals)
    drawdown    = equity_vals - peak                       # always ≤ 0
    max_dd      = float(np.min(drawdown))                  # worst USD drawdown
    max_dd_pct  = float(np.min(drawdown / (peak + 1e-10))) * 100  # worst %

    # -------------------------------------------------------- Sharpe Ratio
    # Use bar-level equity returns for Sharpe
    eq_returns = pd.Series(equity_vals).pct_change().dropna()
    if eq_returns.std() > 0:
        # Annualise for M15 bars
        sharpe = (eq_returns.mean() / eq_returns.std()) * np.sqrt(BARS_PER_YEAR_M15)
    else:
        sharpe = 0.0

    # -------------------------------------------------------- Calmar Ratio
    calmar = (total_ret / abs(max_dd_pct)) if abs(max_dd_pct) > 0 else 0.0

    # -------------------------------------------- BUY vs SELL breakdown
    n_buy  = sum(1 for t in trades if t.signal == "BUY")
    n_sell = sum(1 for t in trades if t.signal == "SELL")

    # ---------------------------------------------------- Average trade hold
    hold_durations = []
    for t in trades:
        if t.entry_time is not None and t.exit_time is not None:
            dur = pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)
            hold_durations.append(dur.total_seconds() / 3600)  # hours
    avg_hold_h = np.mean(hold_durations) if hold_durations else 0.0

    return {
        "total_pnl":          round(total_pnl,  2),
        "total_return_pct":   round(total_ret,  2),
        "initial_balance":    round(result.initial_balance, 2),
        "final_balance":      round(result.final_balance, 2),
        "n_trades":           n_trades,
        "n_wins":             n_wins,
        "n_losses":           n_losses,
        "win_rate_pct":       round(win_rate, 2),
        "avg_win":            round(avg_win, 2),
        "avg_loss":           round(avg_loss, 2),
        "profit_factor":      round(profit_factor, 4) if profit_factor != float("inf") else "∞",
        "expectancy_per_trade": round(expectancy, 2),
        "gross_profit":       round(gross_profit, 2),
        "gross_loss":         round(gross_loss, 2),
        "max_drawdown_usd":   round(max_dd, 2),
        "max_drawdown_pct":   round(max_dd_pct, 2),
        "sharpe_ratio":       round(sharpe, 4),
        "calmar_ratio":       round(calmar, 4),
        "n_buy":              n_buy,
        "n_sell":             n_sell,
        "tp_exits":           tp_count,
        "sl_exits":           sl_count,
        "eod_exits":          eod_count,
        "avg_hold_hours":     round(avg_hold_h, 2),
    }


def print_report(metrics: dict):
    """Pretty-print the performance report to stdout."""
    sep = "═" * 52
    print(f"\n{sep}")
    print("   BACKTEST PERFORMANCE REPORT")
    print(sep)
    print(f"  Initial Balance   : ${metrics['initial_balance']:>12,.2f}")
    print(f"  Final Balance     : ${metrics['final_balance']:>12,.2f}")
    print(f"  Total P&L         : ${metrics['total_pnl']:>+12,.2f}")
    print(f"  Total Return      : {metrics['total_return_pct']:>+11.2f}%")
    print(f"{'─'*52}")
    print(f"  Trades            : {metrics['n_trades']:>12,}")
    print(f"  Wins / Losses     : {metrics['n_wins']:>5,}  / {metrics['n_losses']:>5,}")
    print(f"  Win Rate          : {metrics['win_rate_pct']:>11.2f}%")
    print(f"  Avg Win           : ${metrics['avg_win']:>+12,.2f}")
    print(f"  Avg Loss          : ${metrics['avg_loss']:>+12,.2f}")
    print(f"  Profit Factor     : {str(metrics['profit_factor']):>12}")
    print(f"  Expectancy/Trade  : ${metrics['expectancy_per_trade']:>+12,.2f}")
    print(f"{'─'*52}")
    print(f"  Max Drawdown      : ${metrics['max_drawdown_usd']:>+12,.2f}")
    print(f"  Max Drawdown %    : {metrics['max_drawdown_pct']:>+11.2f}%")
    print(f"  Sharpe Ratio      : {metrics['sharpe_ratio']:>12.4f}")
    print(f"  Calmar Ratio      : {metrics['calmar_ratio']:>12.4f}")
    print(f"{'─'*52}")
    print(f"  BUY / SELL trades : {metrics['n_buy']:>5,}  / {metrics['n_sell']:>5,}")
    print(f"  TP / SL / EOD     : {metrics['tp_exits']:>4,} / {metrics['sl_exits']:>4,} / {metrics['eod_exits']:>4,}")
    print(f"  Avg Hold          : {metrics['avg_hold_hours']:>11.2f}h")
    print(f"{sep}\n")
