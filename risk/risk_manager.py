# =============================================================================
# risk/risk_manager.py
#
# Risk management module.
#
# Responsibilities
# ----------------
#   • Position sizing  – fixed fractional (% of current balance)
#   • Stop-loss price  – fixed pips below/above entry
#   • Take-profit price – fixed pips above/below entry
#   • Trade approval   – reject if risk criteria not met
#
# Public API
# ----------
#   RiskManager(config)
#       .check_trade(signal, balance, open_trades, confidence) → bool
#       .calculate_position(balance, entry, stop_loss) → lot_size
#       .get_sl_tp(signal, entry) → (stop_loss, take_profit)
# =============================================================================

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


class RiskManager:
    """
    Manages trade risk using fixed fractional position sizing and
    fixed pip-based stop-loss / take-profit levels.
    """

    def __init__(
        self,
        risk_per_trade:       float = cfg.RISK_PER_TRADE,
        stop_loss_pips:       float = cfg.STOP_LOSS_PIPS,
        take_profit_pips:     float = cfg.TAKE_PROFIT_PIPS,
        pip_value:            float = cfg.PIP_VALUE,
        spread_pips:          float = cfg.SPREAD_PIPS,
        confidence_threshold: float = cfg.CONFIDENCE_THRESHOLD,
        max_open_trades:      int   = 3,
    ):
        self.risk_per_trade       = risk_per_trade
        self.stop_loss_pips       = stop_loss_pips
        self.take_profit_pips     = take_profit_pips
        self.pip_value            = pip_value
        self.spread_pips          = spread_pips
        self.confidence_threshold = confidence_threshold
        self.max_open_trades      = max_open_trades

    # ------------------------------------------------------------------
    # Approve or reject a trade signal
    # ------------------------------------------------------------------
    def check_trade(
        self,
        signal:       str,
        balance:      float,
        open_trades:  int,
        confidence:   float,
    ) -> dict:
        """
        Determine whether to enter a trade.

        Returns
        -------
        dict
            {"approved": bool, "reason": str}
        """
        if signal not in ("BUY", "SELL"):
            return {"approved": False, "reason": "Signal is HOLD"}

        if confidence < self.confidence_threshold:
            return {
                "approved": False,
                "reason": f"Confidence {confidence:.2%} < threshold {self.confidence_threshold:.2%}",
            }

        if open_trades >= self.max_open_trades:
            return {
                "approved": False,
                "reason": f"Max open trades ({self.max_open_trades}) reached",
            }

        if balance <= 0:
            return {"approved": False, "reason": "Balance exhausted"}

        return {"approved": True, "reason": "OK"}

    # ------------------------------------------------------------------
    # SL / TP prices
    # ------------------------------------------------------------------
    def get_sl_tp(self, signal: str, entry_price: float) -> tuple:
        """
        Calculate stop-loss and take-profit prices.

        Returns
        -------
        (stop_loss, take_profit) : (float, float)
        """
        sl_dist = self.stop_loss_pips   * self.pip_value
        tp_dist = self.take_profit_pips * self.pip_value

        # Add spread to entry for BUY orders
        if signal == "BUY":
            entry_with_spread = entry_price + self.spread_pips * self.pip_value
            stop_loss   = entry_with_spread - sl_dist
            take_profit = entry_with_spread + tp_dist
        else:  # SELL
            entry_with_spread = entry_price - self.spread_pips * self.pip_value
            stop_loss   = entry_with_spread + sl_dist
            take_profit = entry_with_spread - tp_dist

        return round(stop_loss, 5), round(take_profit, 5)

    # ------------------------------------------------------------------
    # Position sizing: fixed fractional
    # ------------------------------------------------------------------
    def calculate_position(
        self,
        balance:     float,
        entry_price: float,
        stop_loss:   float,
    ) -> float:
        """
        Calculate position size (in units / lots) based on risk per trade.

        Formula
        -------
            risk_amount = balance * risk_per_trade
            pip_risk    = |entry - stop_loss| / pip_value
            lot_size    = risk_amount / (pip_risk * pip_value * lot_size_units)

        For simplicity in backtesting we express P&L in pips × fixed value.

        Returns
        -------
        float : position size in "units" where 1 unit = 1 USD per pip.
        """
        risk_amount = balance * self.risk_per_trade
        price_risk  = abs(entry_price - stop_loss)

        if price_risk < 1e-10:
            return 0.0

        pip_risk = price_risk / self.pip_value
        # 1 standard lot = 100,000 units; pip value ≈ $10/pip for EUR/USD
        # We simplify: units = risk_amount / pip_risk
        units = risk_amount / pip_risk
        return round(units, 2)

    # ------------------------------------------------------------------
    # P&L calculation for a closed trade
    # ------------------------------------------------------------------
    def calculate_pnl(
        self,
        signal:     str,
        entry:      float,
        exit_price: float,
        units:      float,
    ) -> float:
        """
        Calculate realised P&L in USD.

        P&L = (exit - entry) * units   for BUY
        P&L = (entry - exit) * units   for SELL
        """
        if signal == "BUY":
            return (exit_price - entry) / self.pip_value * units
        else:
            return (entry - exit_price) / self.pip_value * units


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rm = RiskManager()
    sl, tp = rm.get_sl_tp("BUY", 1.10000)
    print(f"BUY  SL={sl:.5f}  TP={tp:.5f}")

    sl, tp = rm.get_sl_tp("SELL", 1.10000)
    print(f"SELL SL={sl:.5f}  TP={tp:.5f}")

    size = rm.calculate_position(10_000, 1.10000, 1.09800)
    print(f"Position size: {size} units")

    pnl = rm.calculate_pnl("BUY", 1.10000, 1.10400, size)
    print(f"P&L on 40 pip win: ${pnl:.2f}")
