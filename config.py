# =============================================================================
# config.py — Central configuration for the AI Forex Trading Bot
# =============================================================================

# --- Symbol & Timeframe ---
# FX Pairs : EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD
# Commodities: XAUUSD (Gold)
SYMBOL = "EURUSD"
TIMEFRAME = "H1"          # M1, M5, M15, M30, H1, H4, D1
N_BARS = 15_000            # rows to load / generate

# --- Feature Engineering ---
EMA_FAST   = 10
EMA_MID    = 50
EMA_SLOW   = 200
RSI_PERIOD = 14
BB_PERIOD  = 20
BB_STD     = 2.0
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9

# --- Label ---
FORWARD_BARS = 3           # predict N bars ahead (3h window reduces noise)

# --- Model ---
MODEL_PATH        = "models/model.pkl"
SCALER_PATH       = "models/scaler.pkl"
TEST_SIZE         = 0.20
RANDOM_STATE      = 42
# HistGradientBoosting hyperparams
MAX_ITER          = 300
MAX_DEPTH         = 6
MIN_SAMPLES_LEAF  = 30
LEARNING_RATE     = 0.05
# RandomForest fallback (unused when USE_HGB=True)
N_ESTIMATORS      = 300
CLASS_WEIGHT      = "balanced"

# Set True to use HistGradientBoostingClassifier (better on real data)
USE_HGB           = True

# --- Strategy Selector ---
# Options:
#   "trend_follow"  — EMA cross + RSI + MACD + Bollinger Bands (default)
#   "rsi_reversal"  — Buy oversold (RSI<30), Sell overbought (RSI>70)
#   "ema_cross"     — Simple EMA10/EMA50 crossover only
#   "breakout"      — Price breaks above/below recent high/low
#   "macd_signal"   — MACD line crosses signal line
STRATEGY = "trend_follow"

# --- AI Filter ---
CONFIDENCE_THRESHOLD = 0.55   # only trade if model confidence > 55 %

# --- Risk Management ---
INITIAL_BALANCE   = 10_000.0  # USD
RISK_PER_TRADE    = 0.01      # 1 % of balance per trade
STOP_LOSS_PIPS    = 20
TAKE_PROFIT_PIPS  = 40
PIP_VALUE         = 0.0001    # EURUSD 1 pip
SPREAD_PIPS       = 1.5       # simulated spread

# --- Backtest ---
COMMISSION_PER_TRADE = 7.0    # set > 0 to model broker commissions

# --- Telegram Alerts ---
TELEGRAM_TOKEN   = "7652282409:AAENeSr2S3QkovRH0E2BIOD7nKd1MCIvQ8Y"          # from @BotFather  e.g. "123456:ABC-DEF..."
TELEGRAM_CHAT_ID = "8300543038"          # your personal chat ID e.g. "987654321"
