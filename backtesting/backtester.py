# =============================================================================
# backtesting/backtester.py
#
# Event-driven backtesting engine.
#
# For each bar in the test set:
#   1. Generate rule-based strategy signal
#   2. Apply AI filter (confidence > threshold)
#   3. Check risk management approval
#   4. Open trade if all conditions pass
#   5. Check SL / TP for open trade
#   6. Record balance and trades
#
# Anti-look-ahead safeguards
# --------------------------
#   • Features computed on df up to current bar only (via rolling window).
#   • Batch predictions stored before the loop (but indexed correctly).
#   • No future prices used for entry / exit decisions.
#
# Public API
# ----------
#   Backtester(model, scaler, feature_cols, risk_manager)
#       .run(df) → BacktestResult
# =============================================================================

import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from ai_training.features   import create_features, get_feature_cols
from ai_training.model      import predict_batch
from strategy.signal_generator import generate_signals_batch, _compute_strategy_cols
from risk.risk_manager      import RiskManager


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time:  object          # datetime
    signal:      str             # "BUY" or "SELL"
    entry_price: float
    stop_loss:   float
    take_profit: float
    units:       float
    confidence:  float
    exit_time:   object = None
    exit_price:  float  = None
    pnl:         float  = None
    exit_reason: str    = None   # "TP", "SL", "EOD"


@dataclass
class BacktestResult:
    trades:        List[Trade]
    equity_curve:  pd.Series      # balance at each bar
    initial_balance: float
    final_balance:   float
    metrics:       dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Walk-forward backtester that integrates strategy signals, AI predictions,
    and risk management.
    """

    def __init__(
        self,
        model,
        scaler,
        feature_cols: list,
        risk_manager: Optional[RiskManager] = None,
        initial_balance: float = config.INITIAL_BALANCE,
        commission:      float = config.COMMISSION_PER_TRADE,
    ):
        self.model        = model
        self.scaler       = scaler
        self.feature_cols = feature_cols
        self.rm           = risk_manager or RiskManager()
        self.balance0     = initial_balance
        self.commission   = commission

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run the backtest over the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV data (the full dataset; engine will skip the
            first 250 rows to allow indicators to warm up).

        Returns
        -------
        BacktestResult
        """
        print("\n[Backtest] Preparing data …")

        # 1. Feature engineering (full batch — no future leakage as labels
        #    are NOT included here; we only use OHLCV → indicators)
        df_feat = create_features(df.copy())

        # Resolve feature columns
        feat_cols = self.feature_cols if self.feature_cols else get_feature_cols(df_feat)

        # 2. Pre-compute AI predictions for all rows (batch — efficient)
        print("[Backtest] Running batch AI predictions …")
        ai_preds = predict_batch(df_feat, self.model, self.scaler, feat_cols)

        # 3. Generate strategy signals for all rows
        print("[Backtest] Generating strategy signals …")
        df_strat = _compute_strategy_cols(df_feat)
        strategy_signals = generate_signals_batch(df_strat)

        # 4. Merge into one working frame
        work = df_feat.copy()
        work["strat_signal"] = strategy_signals.values
        work["ai_pred"]      = ai_preds["pred"].values
        work["confidence"]   = ai_preds["confidence"].values
        work["buy_prob"]     = ai_preds["buy_prob"].values
        work.reset_index(drop=True, inplace=True)

        # 5. Walk-forward simulation
        print("[Backtest] Running simulation …")
        balance      = self.balance0
        equity_curve = []
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None

        for i, row in work.iterrows():
            current_price = row["close"]
            current_time  = row["time"]

            # --- Manage open trade: check SL / TP ---
            if open_trade is not None:
                hit_sl, hit_tp = False, False

                if open_trade.signal == "BUY":
                    hit_sl = row["low"]  <= open_trade.stop_loss
                    hit_tp = row["high"] >= open_trade.take_profit
                else:  # SELL
                    hit_sl = row["high"] >= open_trade.stop_loss
                    hit_tp = row["low"]  <= open_trade.take_profit

                if hit_tp:
                    exit_px = open_trade.take_profit
                    reason  = "TP"
                elif hit_sl:
                    exit_px = open_trade.stop_loss
                    reason  = "SL"
                else:
                    exit_px = None
                    reason  = None

                if exit_px is not None:
                    pnl = self.rm.calculate_pnl(
                        open_trade.signal, open_trade.entry_price, exit_px, open_trade.units
                    ) - self.commission

                    balance += pnl
                    open_trade.exit_time  = current_time
                    open_trade.exit_price = exit_px
                    open_trade.pnl        = pnl
                    open_trade.exit_reason = reason
                    trades.append(open_trade)
                    open_trade = None

            # --- Try to enter a new trade ---
            if open_trade is None:
                strat_sig  = row["strat_signal"]
                confidence = row["confidence"]
                buy_prob   = row["buy_prob"]

                # AI alignment: strategy says BUY and model says BUY (pred=1)
                #               strategy says SELL and model says SELL (pred=0)
                ai_agrees = (
                    (strat_sig == "BUY"  and row["ai_pred"] == 1) or
                    (strat_sig == "SELL" and row["ai_pred"] == 0)
                )

                approval = self.rm.check_trade(
                    signal      = strat_sig,
                    balance     = balance,
                    open_trades = 0,   # max 1 trade at a time
                    confidence  = confidence if ai_agrees else 0.0,
                )

                if approval["approved"] and ai_agrees:
                    sl, tp = self.rm.get_sl_tp(strat_sig, current_price)
                    units  = self.rm.calculate_position(balance, current_price, sl)

                    if units > 0:
                        open_trade = Trade(
                            entry_time  = current_time,
                            signal      = strat_sig,
                            entry_price = current_price,
                            stop_loss   = sl,
                            take_profit = tp,
                            units       = units,
                            confidence  = confidence,
                        )

            equity_curve.append(balance)

        # --- Close any open trade at end of data ---
        if open_trade is not None:
            last_row = work.iloc[-1]
            exit_px  = last_row["close"]
            pnl      = self.rm.calculate_pnl(
                open_trade.signal, open_trade.entry_price, exit_px, open_trade.units
            ) - self.commission
            balance += pnl
            open_trade.exit_time   = last_row["time"]
            open_trade.exit_price  = exit_px
            open_trade.pnl         = pnl
            open_trade.exit_reason = "EOD"
            trades.append(open_trade)
            equity_curve[-1] = balance

        equity = pd.Series(equity_curve, index=work["time"].values)

        print(f"[Backtest] Complete. Trades: {len(trades)} | Final balance: ${balance:,.2f}")

        return BacktestResult(
            trades          = trades,
            equity_curve    = equity,
            initial_balance = self.balance0,
            final_balance   = balance,
        )
