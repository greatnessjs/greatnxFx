# =============================================================================
# backtesting/visualization.py
#
# Visualisation of backtest results.
#
# Charts
# ------
#   1. Equity curve with drawdown shading
#   2. Trade entry / exit overlaid on price
#   3. Monthly P&L bar chart
#   4. Win / Loss distribution histogram
#
# Public API
# ----------
#   plot_results(result, metrics, save_path=None)
#   plot_equity(result, save_path=None)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless / non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BUY_COLOR   = "#26a69a"   # teal
SELL_COLOR  = "#ef5350"   # red
DD_COLOR    = "#ffcccc"   # light red
EQ_COLOR    = "#1976d2"   # blue


def plot_results(result, metrics: dict, save_path: str = "backtest_report.png"):
    """
    Generate a 4-panel backtest report and save to PNG.

    Parameters
    ----------
    result    : BacktestResult
    metrics   : dict returned by compute_metrics()
    save_path : str — output file path (None = show interactively)
    """
    trades = result.trades
    equity = result.equity_curve

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_eq    = fig.add_subplot(gs[0, :])   # equity — full width
    ax_price = fig.add_subplot(gs[1, :])   # price + trades — full width
    ax_monthly = fig.add_subplot(gs[2, 0]) # monthly P&L
    ax_dist    = fig.add_subplot(gs[2, 1]) # P&L distribution

    _style_ax(ax_eq, ax_price, ax_monthly, ax_dist)

    # 1. Equity curve + drawdown
    _plot_equity(ax_eq, equity, result.initial_balance)

    # 2. Price with trade markers
    _plot_trades_on_price(ax_price, equity, trades)

    # 3. Monthly P&L
    _plot_monthly_pnl(ax_monthly, trades)

    # 4. P&L distribution
    _plot_pnl_distribution(ax_dist, trades)

    # Title
    ret_pct = metrics["total_return_pct"]
    sr      = metrics["sharpe_ratio"]
    wr      = metrics["win_rate_pct"]
    fig.suptitle(
        f"AI Forex Backtest  |  Return: {ret_pct:+.2f}%  |  Sharpe: {sr:.2f}  |  Win Rate: {wr:.1f}%",
        color="white", fontsize=15, fontweight="bold", y=0.98,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Viz] Report saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Sub-plots
# ---------------------------------------------------------------------------

def _plot_equity(ax, equity: pd.Series, initial_balance: float):
    eq_vals = equity.values
    times   = pd.to_datetime(equity.index)

    # Drawdown
    peak = np.maximum.accumulate(eq_vals)
    ax.fill_between(times, eq_vals, peak, where=(eq_vals < peak),
                    color=DD_COLOR, alpha=0.4, label="Drawdown")

    # Equity line
    ax.plot(times, eq_vals, color=EQ_COLOR, linewidth=1.5, label="Equity")
    ax.axhline(initial_balance, color="gray", linewidth=0.8, linestyle="--", label="Start")

    ax.set_title("Equity Curve", color="white", pad=8)
    ax.set_ylabel("Balance (USD)", color="#aaaaaa")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=8)


def _plot_trades_on_price(ax, equity: pd.Series, trades):
    times  = pd.to_datetime(equity.index)
    prices = equity.values   # approximate proxy; ideally use close prices

    ax.plot(times, prices, color="#888888", linewidth=0.7, alpha=0.6)
    ax.set_title("Trades on Equity", color="white", pad=8)
    ax.set_ylabel("Balance (USD)", color="#aaaaaa")

    for t in trades:
        if t.entry_time is None or t.exit_time is None:
            continue
        entry_t = pd.Timestamp(t.entry_time)
        exit_t  = pd.Timestamp(t.exit_time)
        color   = BUY_COLOR if t.signal == "BUY" else SELL_COLOR
        marker  = "^" if t.signal == "BUY" else "v"
        # Entry marker on equity at entry time
        try:
            idx_e = equity.index.get_loc(t.entry_time, method="nearest")
            idx_x = equity.index.get_loc(t.exit_time,  method="nearest")
        except Exception:
            continue
        ax.scatter(times[idx_e], prices[idx_e], color=color, marker=marker, s=40, zorder=5)
        if t.pnl and t.pnl > 0:
            ax.scatter(times[idx_x], prices[idx_x], color=BUY_COLOR, marker="o", s=25, zorder=5)
        else:
            ax.scatter(times[idx_x], prices[idx_x], color=SELL_COLOR, marker="o", s=25, zorder=5)

    buy_patch  = Patch(color=BUY_COLOR,  label="BUY entry")
    sell_patch = Patch(color=SELL_COLOR, label="SELL entry")
    ax.legend(handles=[buy_patch, sell_patch], loc="upper left",
              framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=8)


def _plot_monthly_pnl(ax, trades):
    if not trades:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", color="white")
        return

    df = pd.DataFrame([{
        "month": pd.Timestamp(t.entry_time).to_period("M"),
        "pnl":   t.pnl,
    } for t in trades if t.pnl is not None])

    if df.empty:
        return

    monthly = df.groupby("month")["pnl"].sum().reset_index()
    monthly["month_str"] = monthly["month"].astype(str)
    colors = [BUY_COLOR if v > 0 else SELL_COLOR for v in monthly["pnl"]]

    ax.bar(range(len(monthly)), monthly["pnl"], color=colors, alpha=0.85)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["month_str"], rotation=45, ha="right", fontsize=7, color="#aaaaaa")
    ax.axhline(0, color="white", linewidth=0.5)
    ax.set_title("Monthly P&L", color="white", pad=8)
    ax.set_ylabel("P&L (USD)", color="#aaaaaa")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))


def _plot_pnl_distribution(ax, trades):
    pnls = [t.pnl for t in trades if t.pnl is not None]
    if not pnls:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", color="white")
        return

    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    bins = 30
    ax.hist(wins,   bins=bins, color=BUY_COLOR,  alpha=0.7, label="Wins")
    ax.hist(losses, bins=bins, color=SELL_COLOR, alpha=0.7, label="Losses")
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax.axvline(np.mean(pnls), color="yellow", linewidth=1.2, linestyle="--",
               label=f"Mean ${np.mean(pnls):+.0f}")
    ax.set_title("P&L Distribution", color="white", pad=8)
    ax.set_xlabel("P&L (USD)", color="#aaaaaa")
    ax.set_ylabel("Count", color="#aaaaaa")
    ax.legend(framealpha=0.3, facecolor="#1a1a2e", labelcolor="white", fontsize=8)


# ---------------------------------------------------------------------------
# Shared axis styling
# ---------------------------------------------------------------------------

def _style_ax(*axes):
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#aaaaaa")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.title.set_color("white")


# ---------------------------------------------------------------------------
# Simple standalone equity plot
# ---------------------------------------------------------------------------

def plot_equity(result, save_path: str = None):
    """Quick equity curve plot."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    _style_ax(ax)
    _plot_equity(ax, result.equity_curve, result.initial_balance)
    ax.set_title("Equity Curve", color="white")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Viz] Equity curve saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
