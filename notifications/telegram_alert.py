# =============================================================================
# notifications/telegram_alert.py
# =============================================================================

import requests
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _is_configured() -> bool:
    token   = getattr(config, "TELEGRAM_TOKEN",   "")
    chat_id = getattr(config, "TELEGRAM_CHAT_ID", "")
    return bool(token and chat_id)


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def send_alert(message: str) -> bool:
    if not _is_configured():
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id":    config.TELEGRAM_CHAT_ID,
                "text":       message,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"[Telegram] Failed to send alert: {e}")
        return False


# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------

def alert_startup(symbol: str, timeframe: str, mode: str):
    send_alert(
        f"🤖 <b>AI Forex Bot — Online</b>\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n\n"
        f"📊 Pair         : <b>{symbol}</b>\n"
        f"⏱ Timeframe   : <b>{timeframe}</b>\n"
        f"⚙️ Mode         : {mode}\n"
        f"🕐 Started      : {_now()}\n\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
        f"Watching the market for high-confidence setups..."
    )


def alert_signal(signal: str, confidence: float, symbol: str, filtered: bool):
    if filtered:
        direction_icon = "🟢" if signal == "BUY" else "🔴"
        send_alert(
            f"{direction_icon} <b>Signal Detected!</b>\n"
            f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n\n"
            f"📌 Pair         : <b>{symbol}</b>\n\n"
            f"📈 Direction    : <b>{signal}</b>\n\n"
            f"🎯 Confidence   : <b>{confidence:.1%}</b>\n\n"
            f"✅ Status       : Above threshold\n\n"
            f"🕐 Time         : {_now()}"
        )
    else:
        send_alert(
            f"👀 <b>Weak Signal — Watching</b>\n"
            f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n\n"
            f"📌 Pair         : {symbol}\n\n"
            f"📉 Direction    : {signal}\n\n"
            f"🎯 Confidence   : {confidence:.1%}\n\n"
            f"⏳ Status       : Below threshold — no trade\n\n"
            f"🕐 Time         : {_now()}"
        )


def alert_trade_placed(signal: str, symbol: str, lots: float,
                       entry: float, sl: float, tp: float,
                       confidence: float, dry_run: bool):
    direction_icon = "🟢" if signal == "BUY" else "🔴"
    mode_tag = "📋 <b>[DEMO / DRY RUN]</b>\n" if dry_run else ""
    risk_reward = "2:1"

    send_alert(
        f"{direction_icon} <b>Trade Alert — {signal} {symbol}</b>\n"
        f"{mode_tag}"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n\n"
        f"🎯 Confidence   : <b>{confidence:.1%}</b>\n\n"
        f"💰 Entry Price  : <b>{entry:.5f}</b>\n\n"
        f"🛑 Stop Loss    : {sl:.5f}\n\n"
        f"✅ Take Profit  : {tp:.5f}\n\n"
        f"📦 Lot Size     : {lots}\n\n"
        f"⚖️ Risk/Reward  : {risk_reward}\n\n"
        f"🕐 Time         : {_now()}\n\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
        f"{'📋 Simulated signal — no real order placed.' if dry_run else '✅ Order sent to MT5.'}"
    )


def alert_position_open(symbol: str, direction: str, pnl: float):
    pnl_icon = "📈" if pnl >= 0 else "📉"
    send_alert(
        f"🔄 <b>Position Already Open</b>\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n\n"
        f"📌 Pair         : {symbol}\n\n"
        f"📊 Direction    : {direction}\n\n"
        f"{pnl_icon} Current P&L  : <b>${pnl:+.2f}</b>\n\n"
        f"⚙️ MT5 managing SL / TP automatically\n\n"
        f"🕐 Time         : {_now()}"
    )


def alert_risk_blocked(reason: str):
    send_alert(
        f"🚫 <b>Trade Blocked — Risk Manager</b>\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n\n"
        f"⚠️ Reason       : {reason}\n\n"
        f"🕐 Time         : {_now()}\n\n"
        f"Waiting for next opportunity..."
    )


def alert_stopped():
    send_alert(
        f"🛑 <b>AI Forex Bot — Offline</b>\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n\n"
        f"Bot has been stopped manually.\n\n"
        f"🕐 Time : {_now()}"
    )


def alert_current_signal(signal: str, confidence: float, symbol: str,
                         filtered: bool, timeframe: str):
    direction_icon = "🟢" if signal == "BUY" else "🔴"
    status_icon    = "✅" if filtered else "⏳"
    status_text    = "Above threshold — trade would fire" if filtered else "Below threshold — no trade"
    send_alert(
        f"{direction_icon} <b>Latest Signal — {symbol} {timeframe}</b>\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n\n"
        f"📈 Direction    : <b>{signal}</b>\n\n"
        f"🎯 Confidence   : <b>{confidence:.1%}</b>\n\n"
        f"{status_icon} Status       : {status_text}\n\n"
        f"🕐 Time         : {_now()}"
    )
