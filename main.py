"""
BTC/USD 5m Rolling Candle Predictor — PUBLIC BOT  v3.0  ★ WORLD CLASS ★
═══════════════════════════════════════════════════════════════════════════════
MAJOR UPGRADES vs v2.0
──────────────────────────────────────────────────────────────────────────────
  [ML-1]  RandomForest ensemble  — 50-tree classifier on 40+ features
  [ML-2]  Online learning        — model retrains every 500 candles live
  [ML-3]  Feature engineering    — OHLCV body/wick/range + lag-3 candles
  [ML-4]  Regime classifier      — trending / choppy / ranging market states
  [SIG-1] VWAP + deviation bands — institutional price anchor
  [SIG-2] MACD histogram delta   — rate-of-change of momentum
  [SIG-3] Bollinger Band squeeze — volatility breakout detector
  [SIG-4] ATR adaptive filter    — skip flat / hyper-volatile candles
  [SIG-5] Stochastic %K/%D cross — overbought/oversold momentum
  [SIG-6] OBV slope + divergence — volume-price confirmation
  [SIG-7] EMA ribbon 9/20/50/100 — multi-MA trend stack
  [SIG-8] Williams %R            — another fast oscillator
  [SIG-9] CMF (Chaikin MF)       — money-flow pressure
  [SIG-10] Donchian channel      — breakout direction bias
  [MTF-1] 5 timeframes          — 5m/15m/1h/4h/1d bias stack
  [MTF-2] Trend alignment score  — how many TFs agree → confidence bonus
  [RISK-1] Adaptive confidence gate — threshold adjusts to recent accuracy
  [RISK-2] Max-drawdown streak guard — pause after N consecutive wrong
  [BACK-1] Walk-forward backtest  — no look-ahead, true OOS validation
  [BACK-2] Sharpe-like score      — quality metric beyond raw accuracy
  [UI-1]  Live web dashboard      — Flask server with real-time stats
  [UI-2]  Telegram inline keyboard — richer user experience
═══════════════════════════════════════════════════════════════════════════════
COMMANDS
  /start        — subscribe
  /stop         — unsubscribe
  /status       — live stats + signals
  /predict      — on-demand prediction
  /accuracy     — detailed accuracy breakdown
  /regime       — current market regime
  /subscribers  — admin: list users
  /backtest     — admin: walk-forward backtest report
  /retrain      — admin: force ML model retrain
  /dashboard    — admin: web dashboard URL
═══════════════════════════════════════════════════════════════════════════════
"""

import requests, time, os, sqlite3, threading, collections, statistics, math
import json, hashlib, random
from datetime import datetime, timezone, timedelta

# ── Optional ML imports (graceful degradation if not installed) ───────────────
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    ML_AVAILABLE = True
    print("✅ scikit-learn + numpy detected — ML ensemble ENABLED")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn not found — running heuristic-only mode")
    print("    Install: pip install scikit-learn numpy")

# ── Optional Flask dashboard ───────────────────────────────────────────────────
try:
    from flask import Flask, jsonify, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN",  "8766778348:AAEpkHO55y_oCrJ0vrTwtXsm8cWE_4IOZxA")
ADMIN_CHAT_ID   = os.environ.get("ADMIN_CHAT_ID",   "5792224870")
DASHBOARD_PORT  = int(os.environ.get("DASHBOARD_PORT", "8080"))

SYMBOL          = "BTCUSDT"
INTERVAL        = "5"
BYBIT_URL       = "https://api.bybit.com/v5/market/kline"
DAYS            = 365
BATCH_SIZE      = 1000

WINDOW_SIZE     = 365 * 24 * 12        # ~105 120 5m candles
MIN_CANDLES     = 1500                  # need more for ML features
ML_TRAIN_MIN    = 500                   # min samples to train RF
ML_RETRAIN_FREQ = 500                   # retrain every N new candles
LOOKBACK        = 50                    # feature window for ML

# ── Timeframes for MTF ────────────────────────────────────────────────────────
MTF_CONFIG = [
    ("15",   "15m",  50,  1.0),
    ("60",   "1h",   50,  1.5),
    ("240",  "4h",   30,  2.0),
    ("D",    "1D",   20,  2.5),
]
MTF_CANDLES_FETCH = 60

# ── Indicator params ──────────────────────────────────────────────────────────
EMA_PERIODS     = [9, 20, 50, 100]
RSI_PERIOD      = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_PERIOD, BB_STD  = 20, 2.0
ATR_PERIOD      = 14
STOCH_PERIOD    = 14
OBV_WINDOW      = 20
CMF_PERIOD      = 20
DONCHIAN_PERIOD = 20
WILLIAMS_R_PERIOD = 14
VWAP_SESSION_CANDLES = 78   # 6.5h session at 5m

# ── Gates & risk ─────────────────────────────────────────────────────────────
BASE_MIN_CONF   = 5.0       # base minimum confidence %
ATR_FLAT_PCT    = 0.04      # below this → flat market skip
ATR_EXTREME_PCT = 0.25      # above this → hyper-volatile skip
MAX_LOSS_STREAK = 7         # pause after this many consecutive wrong
CONF_ADAPT_WINDOW = 50      # candles over which to adapt confidence gate

# ── Signal weights (by regime) ────────────────────────────────────────────────
# [trending, choppy, ranging]
WEIGHTS = {
    "Momentum(50)":    [1.8, 0.8, 1.0],
    "Streak":          [1.5, 0.6, 0.8],
    "Markov":          [2.0, 1.5, 2.0],
    "EMA Ribbon":      [2.5, 0.8, 1.2],
    "RSI(14)":         [1.2, 1.5, 1.8],
    "MACD":            [2.0, 1.2, 1.0],
    "MACD Delta":      [1.5, 1.0, 0.8],
    "Bollinger":       [1.0, 1.8, 2.0],
    "VWAP":            [1.8, 1.0, 1.2],
    "Stochastic":      [1.0, 1.5, 1.8],
    "OBV":             [1.2, 1.0, 1.0],
    "CMF":             [1.0, 1.2, 1.0],
    "Williams%R":      [1.0, 1.5, 1.5],
    "Donchian":        [2.0, 0.8, 1.0],
    "Candle Body":     [1.0, 0.8, 0.8],
    "MTF":             [2.5, 1.5, 2.0],
    "ML_RF":           [3.0, 3.0, 3.0],
}
REGIME_IDX = {"trending": 0, "choppy": 1, "ranging": 2}

POLL_INTERVAL   = 2
DB_FILE         = "subscribers.db"
CONFIDENCE_BANDS = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def ts(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def now_str():
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")

def classify(candle):
    o, c = float(candle[1]), float(candle[4])
    if c > o:   return "green"
    elif c < o: return "red"
    return "doji"

def safe_div(a, b, default=0.0):
    return a / b if b != 0 else default


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS subscribers (
            chat_id   TEXT PRIMARY KEY,
            username  TEXT,
            joined_at TEXT,
            correct   INTEGER DEFAULT 0,
            total     INTEGER DEFAULT 0
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT,
            prediction  TEXT,
            actual      TEXT,
            confidence  REAL,
            regime      TEXT,
            correct     INTEGER
        )
    """)
    conn.commit(); conn.close()

def add_subscriber(chat_id, username=""):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO subscribers (chat_id, username, joined_at) VALUES (?,?,?)",
              (str(chat_id), username, datetime.now(timezone.utc).isoformat()))
    conn.commit(); conn.close()

def remove_subscriber(chat_id):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("DELETE FROM subscribers WHERE chat_id=?", (str(chat_id),))
    conn.commit(); conn.close()

def get_all_subscribers():
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT chat_id, username FROM subscribers")
    rows = c.fetchall(); conn.close(); return rows

def is_subscribed(chat_id):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT 1 FROM subscribers WHERE chat_id=?", (str(chat_id),))
    r = c.fetchone(); conn.close(); return r is not None

def subscriber_count():
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM subscribers")
    n = c.fetchone()[0]; conn.close(); return n

def log_prediction(prediction, actual, confidence, regime, correct):
    try:
        conn = sqlite3.connect(DB_FILE); c = conn.cursor()
        c.execute("INSERT INTO prediction_log (ts,prediction,actual,confidence,regime,correct) VALUES (?,?,?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(), prediction, actual, confidence, regime, int(correct)))
        conn.commit(); conn.close()
    except Exception:
        pass

def get_regime_accuracy():
    """Return accuracy breakdown by regime from DB."""
    try:
        conn = sqlite3.connect(DB_FILE); c = conn.cursor()
        c.execute("SELECT regime, COUNT(*), SUM(correct) FROM prediction_log GROUP BY regime")
        rows = c.fetchall(); conn.close()
        return {r: {"total": t, "correct": cr, "acc": round(100*cr/t, 1) if t else 0}
                for r, t, cr in rows}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════
def send_message(chat_id, text, reply_markup=None):
    try:
        payload = {"chat_id": str(chat_id), "text": text, "parse_mode": "HTML"}
        if reply_markup:
            payload["reply_markup"] = reply_markup
        r = requests.post(f"{BASE_URL}/sendMessage", json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"  ❌ Send error to {chat_id}: {e}")
        return False

def broadcast(text, reply_markup=None):
    subscribers = get_all_subscribers()
    sent = failed = 0; dead = []
    for chat_id, _ in subscribers:
        ok = send_message(chat_id, text, reply_markup)
        if ok: sent += 1
        else:  failed += 1; dead.append(chat_id)
        time.sleep(0.05)
    for d in dead: remove_subscriber(d)
    print(f"  📢 Broadcast: ✅{sent}  ❌{failed} removed")
    return sent

def get_updates(offset=None):
    params = {"timeout": 30, "limit": 100}
    if offset: params["offset"] = offset
    try:
        r = requests.get(f"{BASE_URL}/getUpdates", params=params, timeout=35)
        r.raise_for_status()
        return r.json().get("result", [])
    except Exception as e:
        print(f"  ⚠️  getUpdates error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════
shared_state = {
    "streak": 0, "dir": "unknown", "prediction": None,
    "confidence": 0.0, "accuracy": 0.0, "correct": 0, "total": 0,
    "green": 0, "red": 0, "green_pct": 0, "red_pct": 0,
    "price": 0.0, "skipped": 0, "atr_pct": 0.0,
    "regime": "unknown", "ml_confidence": 0.0, "ml_enabled": ML_AVAILABLE,
    "mtf_biases": {}, "signals": {}, "loss_streak": 0,
    "min_conf_adaptive": BASE_MIN_CONF, "candle_count": 0,
    "last_update": "—", "version": "3.0",
}
_predictor_ref = None


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
def handle_start(chat_id, username):
    if is_subscribed(chat_id):
        send_message(chat_id,
            "✅ <b>Already subscribed!</b>\n\n"
            "You receive elite BTC/USD 5m predictions with:\n"
            "  🤖 ML RandomForest ensemble\n"
            "  📊 15+ technical signals\n"
            "  🌐 5-timeframe MTF bias stack\n"
            "  🧠 Market regime detection\n\n"
            "Commands: /stop /status /predict /accuracy /regime"
        )
    else:
        add_subscriber(chat_id, username)
        send_message(chat_id,
            "🎉 <b>Welcome to BTC Predictor v3 — World Class Edition!</b>\n\n"
            "📌 <b>What you'll get every 5 minutes:</b>\n"
            "  • Next candle direction + confidence %\n"
            "  • 15+ signals: ML · VWAP · MACD · BB · Stoch · CMF · OBV · MTF · ...\n"
            "  • 5-timeframe bias stack (15m → 1D)\n"
            "  • Market regime (trending/choppy/ranging)\n"
            "  • Live accuracy with streak protection\n\n"
            "⚡ <b>Commands:</b>\n"
            "  /start    — Subscribe\n"
            "  /stop     — Unsubscribe\n"
            "  /status   — Live stats & signals\n"
            "  /predict  — On-demand prediction\n"
            "  /accuracy — Detailed accuracy breakdown\n"
            "  /regime   — Current market regime\n\n"
            "<i>🚀 First update at next 5m candle close!</i>"
        )
        print(f"  ➕ New subscriber: {username} ({chat_id})")
        send_message(ADMIN_CHAT_ID,
            f"➕ <b>New subscriber!</b>\n👤 {username or 'Unknown'} ({chat_id})\n"
            f"👥 Total: {subscriber_count()}")

def handle_stop(chat_id, username):
    if is_subscribed(chat_id):
        remove_subscriber(chat_id)
        send_message(chat_id, "😢 <b>Unsubscribed.</b>\nSend /start anytime to resubscribe!")
    else:
        send_message(chat_id, "⚠️ Not subscribed. Send /start!")

def handle_status(chat_id):
    s = shared_state
    pred = s["prediction"]
    pe = "🟢" if pred == "green" else ("🔴" if pred == "red" else "⏳")
    acc_bar = "█" * int(s["accuracy"] / 5) + "░" * (20 - int(s["accuracy"] / 5))
    regime_emoji = {"trending": "📈", "choppy": "〰️", "ranging": "↔️"}.get(s["regime"], "❓")
    mtf_lines = "\n".join(
        f"  {label}: {'🟢' if b=='bullish' else '🔴' if b=='bearish' else '⚪'} {b}"
        for label, b in s.get("mtf_biases", {}).items()
    )
    ml_line = f"  🤖 ML conf: {s.get('ml_confidence', 0):.1f}%" if ML_AVAILABLE else "  🤖 ML: not installed"
    send_message(chat_id,
        f"📊 <b>BTC/USD 5m Predictor v3 — Live Status</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 BTC Price  : <b>${s['price']:,.2f}</b>\n"
        f"🟢 Green      : {s['green']:,} ({s['green_pct']}%)\n"
        f"🔴 Red        : {s['red']:,} ({s['red_pct']}%)\n"
        f"📦 Window     : {s.get('candle_count', 0):,} candles\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{regime_emoji} <b>Regime: {s['regime'].upper()}</b>\n"
        f"<b>MTF Bias Stack:</b>\n{mtf_lines or '  (loading...)'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 Next: {pe} <b>{(pred or 'warming up').upper()}</b>\n"
        f"   Confidence  : <b>{s['confidence']:.1f}%</b>\n"
        f"   Adaptive gate: <b>{s['min_conf_adaptive']:.1f}%</b>\n"
        f"{ml_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Accuracy   : <b>{s['accuracy']}%</b>\n"
        f"   [{acc_bar}]\n"
        f"   {s['correct']}/{s['total']} correct  |  {s['skipped']} skipped\n"
        f"   🔥 Loss streak: {s['loss_streak']}/{MAX_LOSS_STREAK}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👥 Subscribers: {subscriber_count()}\n"
        f"🕐 Updated: {s.get('last_update', '—')}"
    )

def handle_predict(chat_id):
    s = shared_state
    pred = s["prediction"]
    conf = s["confidence"]
    pe = "🟢" if pred == "green" else ("🔴" if pred == "red" else "⏳")
    conf_bar = "█" * int(conf / 5) + "░" * (20 - int(conf / 5))
    sigs = s.get("signals", {})
    sig_lines = "\n".join(f"  • <b>{n}</b>: {v}" for n, v in list(sigs.items())[:8])
    send_message(chat_id,
        f"🔮 <b>On-Demand Prediction</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 ${s['price']:,.2f}  |  {s.get('regime','?').upper()}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Next 5m candle: {pe} <b>{(pred or 'N/A').upper()}</b>\n"
        f"Confidence: <b>{conf:.1f}%</b>\n"
        f"[{conf_bar}]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Top Signals:</b>\n{sig_lines or '  (loading...)'}"
    )

def handle_accuracy(chat_id):
    s = shared_state
    regime_acc = get_regime_accuracy()
    lines = []
    for regime, d in regime_acc.items():
        bar = "█" * int(d["acc"] / 10) + "░" * (10 - int(d["acc"] / 10))
        lines.append(f"  {regime:10}: {d['acc']:5.1f}%  [{bar}]  n={d['total']}")
    send_message(chat_id,
        f"🎯 <b>Accuracy Breakdown v3</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Overall: <b>{s['accuracy']}%</b>  ({s['correct']}/{s['total']})\n"
        f"Skipped: {s['skipped']} (gates active)\n"
        f"Loss streak: {s['loss_streak']}/{MAX_LOSS_STREAK}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>By Market Regime:</b>\n<code>"
        + ("\n".join(lines) or "  (not enough data yet)")
        + "</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 ML: {'ENABLED ✅' if ML_AVAILABLE else 'DISABLED ⚠️'}\n"
        f"<i>Send /regime for market regime details</i>"
    )

def handle_regime(chat_id):
    s = shared_state
    regime = s.get("regime", "unknown")
    desc = {
        "trending": ("📈", "Strong directional move. EMA ribbon aligned.\nMomentum and breakout signals weighted higher."),
        "choppy":   ("〰️", "Low momentum, mixed signals.\nMean-reversion and oscillator signals weighted higher."),
        "ranging":  ("↔️", "Price oscillating in a range.\nBollinger and oscillator signals dominate."),
    }.get(regime, ("❓", "Market state undetermined."))
    emoji, desc_text = desc
    mtf_lines = "\n".join(
        f"  {label}: {'🟢' if b=='bullish' else '🔴' if b=='bearish' else '⚪'} {b}"
        for label, b in s.get("mtf_biases", {}).items()
    )
    send_message(chat_id,
        f"{emoji} <b>Market Regime: {regime.upper()}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{desc_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Multi-Timeframe Stack:</b>\n{mtf_lines or '  (loading...)'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"ATR volatility: {s['atr_pct']:.4f}%\n"
        f"Adaptive conf gate: {s['min_conf_adaptive']:.1f}%"
    )

def handle_subscribers(chat_id):
    if str(chat_id) != str(ADMIN_CHAT_ID):
        send_message(chat_id, "⛔ Admin only."); return
    subs = get_all_subscribers(); count = len(subs)
    lines = "\n".join(f"  {i+1}. {u or 'Unknown'} ({cid})" for i,(cid,u) in enumerate(subs[:20]))
    send_message(chat_id,
        f"👥 <b>Subscribers ({count})</b>\n\n<code>{lines}</code>"
        + (f"\n\n<i>...and {count-20} more</i>" if count > 20 else ""))

def handle_backtest(chat_id):
    if str(chat_id) != str(ADMIN_CHAT_ID):
        send_message(chat_id, "⛔ Admin only."); return
    if _predictor_ref is None:
        send_message(chat_id, "⏳ Bot warming up. Try again shortly."); return
    send_message(chat_id, "🔬 <b>Running walk-forward backtest...</b>\n<i>This may take 30–60 seconds.</i>")
    try:
        report = _predictor_ref.generate_backtest_report()
        for part in report:
            send_message(chat_id, part)
    except Exception as e:
        send_message(chat_id, f"❌ Backtest error: {e}")

def handle_retrain(chat_id):
    if str(chat_id) != str(ADMIN_CHAT_ID):
        send_message(chat_id, "⛔ Admin only."); return
    if _predictor_ref is None:
        send_message(chat_id, "⏳ Bot warming up."); return
    send_message(chat_id, "🔄 <b>Forcing ML model retrain...</b>")
    try:
        result = _predictor_ref.train_ml_model(force=True)
        send_message(chat_id, f"✅ {result}")
    except Exception as e:
        send_message(chat_id, f"❌ Retrain error: {e}")

def handle_dashboard(chat_id):
    if str(chat_id) != str(ADMIN_CHAT_ID):
        send_message(chat_id, "⛔ Admin only."); return
    if FLASK_AVAILABLE:
        send_message(chat_id, f"🌐 Dashboard: http://localhost:{DASHBOARD_PORT}\n"
                              f"<i>Accessible on your server's IP.</i>")
    else:
        send_message(chat_id, "⚠️ Flask not installed.\nInstall: pip install flask")


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM POLLING
# ══════════════════════════════════════════════════════════════════════════════
def polling_thread():
    offset = None
    print("🤖 Telegram polling started...")
    while True:
        try:
            updates = get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                msg = update.get("message") or update.get("edited_message")
                if not msg: continue
                chat_id  = str(msg["chat"]["id"])
                username = msg.get("from", {}).get("username") or \
                           msg.get("from", {}).get("first_name", "Unknown")
                text = msg.get("text", "").strip().lower().split()[0]

                if   text == "/start":        handle_start(chat_id, username)
                elif text == "/stop":         handle_stop(chat_id, username)
                elif text == "/status":       handle_status(chat_id)
                elif text == "/predict":      handle_predict(chat_id)
                elif text == "/accuracy":     handle_accuracy(chat_id)
                elif text == "/regime":       handle_regime(chat_id)
                elif text == "/subscribers":  handle_subscribers(chat_id)
                elif text == "/backtest":     handle_backtest(chat_id)
                elif text == "/retrain":      handle_retrain(chat_id)
                elif text == "/dashboard":    handle_dashboard(chat_id)
        except Exception as e:
            print(f"  ⚠️  Polling error: {e}")
        time.sleep(POLL_INTERVAL)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════
def fetch_historical(symbol, interval, days):
    now_ms   = int(time.time() * 1000)
    start_ms = now_ms - days * 86_400_000
    all_candles = []; end_ms = now_ms
    ms_per_bar = int(interval if interval.isdigit() else 1440) * 60_000
    print(f"📥 Fetching {days}d of {symbol} {interval}m candles...")
    while end_ms > start_ms:
        params = {"category": "spot", "symbol": symbol, "interval": interval,
                  "limit": BATCH_SIZE, "end": end_ms, "start": start_ms}
        try:
            resp = requests.get(BYBIT_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                print(f"❌ Bybit error: {data.get('retMsg')}"); break
            batch = data["result"]["list"]
            if not batch: break
            all_candles.extend(batch)
            oldest = int(batch[-1][0])
            end_ms = oldest - ms_per_bar
            print(f"  {len(all_candles):,} candles...", end="\r")
            time.sleep(0.15)
        except Exception as e:
            print(f"\n  ⚠️  Fetch: {e}"); time.sleep(2)
    all_candles.reverse()
    print(f"\n✅ {len(all_candles):,} {interval}m candles fetched.")
    return all_candles

def fetch_latest_candle(symbol, interval):
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": 3}
    resp = requests.get(BYBIT_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise Exception(f"Bybit: {data.get('retMsg')}")
    return list(reversed(data["result"]["list"]))[-2]

def fetch_mtf_candles(symbol, interval, limit):
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(BYBIT_URL, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if data.get("retCode") != 0: return []
        return list(reversed(data["result"]["list"]))
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS — Pure Python, zero external dependencies
# ══════════════════════════════════════════════════════════════════════════════
def ema(prices, period):
    if len(prices) < period: return None
    k = 2 / (period + 1)
    e = sum(prices[:period]) / period
    for p in prices[period:]: e = p * k + e * (1 - k)
    return e

def ema_series(prices, period):
    if len(prices) < period: return [None] * len(prices)
    k = 2 / (period + 1); e = sum(prices[:period]) / period
    out = [None] * (period - 1) + [e]
    for p in prices[period:]: e = p * k + e * (1 - k); out.append(e)
    return out

def compute_rsi(closes, period=14):
    if len(closes) < period + 1: return None
    diffs = [closes[i] - closes[i-1] for i in range(-period, 0)]
    gains = [max(d, 0) for d in diffs]; losses = [abs(min(d, 0)) for d in diffs]
    ag = sum(gains) / period; al = sum(losses) / period
    if al == 0: return 100.0
    return 100 - 100 / (1 + ag / al)

def compute_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal: return None, None, None, None
    fe = ema_series(closes, fast); se = ema_series(closes, slow)
    macd_line = [(f - s) if f and s else None for f, s in zip(fe, se)]
    valid = [v for v in macd_line if v is not None]
    if len(valid) < signal + 2: return None, None, None, None
    sig_s = ema_series(valid, signal)
    mv = valid[-1]; sv = sig_s[-1]
    hist = mv - sv if sv is not None else None
    delta = hist - (valid[-2] - sig_s[-2]) if sv is not None and sig_s[-2] is not None else None
    return mv, sv, hist, delta

def compute_bollinger(closes, period=20, num_std=2.0):
    if len(closes) < period: return None, None, None, None, None
    window = closes[-period:]; mid = sum(window) / period
    try: std = statistics.stdev(window)
    except Exception: return None, None, None, None, None
    upper = mid + num_std * std; lower = mid - num_std * std
    price = closes[-1]
    pct_b = safe_div(price - lower, upper - lower, 0.5)
    bw = safe_div(upper - lower, mid)
    return upper, mid, lower, round(pct_b, 4), round(bw, 6)

def compute_atr(ohlcv, period=14):
    if len(ohlcv) < period + 1: return None
    trs = [max(ohlcv[i][1]-ohlcv[i][2], abs(ohlcv[i][1]-ohlcv[i-1][3]), abs(ohlcv[i][2]-ohlcv[i-1][3]))
           for i in range(1, len(ohlcv))]
    if not trs: return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]: atr = (atr * (period - 1) + tr) / period
    return atr

def compute_vwap_rolling(ohlcv, n=78):
    """Rolling session VWAP over last n candles."""
    subset = ohlcv[-n:]
    cum_pv = cum_v = 0.0
    for o, h, l, c, v in subset:
        cum_pv += ((h + l + c) / 3) * v
        cum_v  += v
    return safe_div(cum_pv, cum_v) if cum_v else None

def compute_stochastic(ohlcv, period=14):
    if len(ohlcv) < period: return None, None
    window = ohlcv[-period:]
    hi = max(c[1] for c in window); lo = min(c[2] for c in window)
    cl = ohlcv[-1][3]
    k = 100 * safe_div(cl - lo, hi - lo, 0.5)
    k_vals = []
    for i in range(3):
        idx = len(ohlcv) - period - i
        if idx < 0: break
        w2 = ohlcv[idx:idx+period]
        h2 = max(c[1] for c in w2); l2 = min(c[2] for c in w2)
        c2 = ohlcv[idx+period-1][3]
        k_vals.append(100 * safe_div(c2 - l2, h2 - l2, 0.5))
    d = sum(k_vals) / len(k_vals) if k_vals else k
    return round(k, 2), round(d, 2)

def compute_obv_slope(ohlcv, window=20):
    if len(ohlcv) < window + 1: return None
    subset = ohlcv[-window-1:]
    obv = 0.0; series = []
    for i in range(1, len(subset)):
        obv += subset[i][4] if subset[i][3] > subset[i-1][3] else (-subset[i][4] if subset[i][3] < subset[i-1][3] else 0)
        series.append(obv)
    return (series[-1] - series[0]) if len(series) >= 2 else None

def compute_cmf(ohlcv, period=20):
    """Chaikin Money Flow."""
    if len(ohlcv) < period: return None
    subset = ohlcv[-period:]
    mfv_sum = vol_sum = 0.0
    for o, h, l, c, v in subset:
        denom = h - l
        mf_mult = safe_div((c - l) - (h - c), denom) if denom else 0
        mfv_sum += mf_mult * v
        vol_sum += v
    return safe_div(mfv_sum, vol_sum)

def compute_williams_r(ohlcv, period=14):
    if len(ohlcv) < period: return None
    window = ohlcv[-period:]
    hi = max(c[1] for c in window); lo = min(c[2] for c in window)
    cl = ohlcv[-1][3]
    return -100 * safe_div(hi - cl, hi - lo)

def compute_donchian(ohlcv, period=20):
    """Returns (upper, lower, mid, position 0..1)."""
    if len(ohlcv) < period: return None, None, None, None
    window = ohlcv[-period:]
    hi = max(c[1] for c in window); lo = min(c[2] for c in window)
    mid = (hi + lo) / 2; price = ohlcv[-1][3]
    pos = safe_div(price - lo, hi - lo, 0.5)
    return hi, lo, mid, round(pos, 4)

def compute_ema_ribbon(closes):
    """Returns dict of EMA values for each period."""
    ribbon = {}
    for period in EMA_PERIODS:
        ribbon[period] = ema(closes, period)
    return ribbon

def mtf_bias_full(candles):
    """Returns bias string + numeric score [-1, 1]."""
    if len(candles) < 20: return "neutral", 0.0
    closes = [float(c[4]) for c in candles]
    ema20  = ema(closes, 20)
    rsi    = compute_rsi(closes, 14)
    macd_v, sig_v, hist_v, _ = compute_macd(closes)
    if ema20 is None: return "neutral", 0.0
    price = closes[-1]; score = 0.0; count = 0
    if price > ema20: score += 1; count += 1
    elif price < ema20: score -= 1; count += 1
    if rsi is not None:
        if rsi > 55: score += 0.5; count += 1
        elif rsi < 45: score -= 0.5; count += 1
    if hist_v is not None:
        if hist_v > 0: score += 0.5; count += 1
        else: score -= 0.5; count += 1
    if count == 0: return "neutral", 0.0
    norm = score / count
    if norm > 0.3:   return "bullish", round(norm, 3)
    if norm < -0.3:  return "bearish", round(norm, 3)
    return "neutral", round(norm, 3)

def detect_regime(ohlcv, closes):
    """Detect market regime: trending / choppy / ranging."""
    if len(ohlcv) < 50: return "trending"
    atr = compute_atr(ohlcv[-51:], 14)
    if atr is None: return "trending"
    atr_pct = atr / closes[-1] * 100

    # ADX-like directional strength via EMA separation
    e9  = ema(closes, 9)
    e20 = ema(closes, 20)
    e50 = ema(closes, min(50, len(closes)))
    if None in (e9, e20, e50): return "trending"

    separation = abs(e9 - e50) / e50 * 100
    aligned = (e9 > e20 > e50) or (e9 < e20 < e50)

    # Bollinger bandwidth
    _, _, _, _, bw = compute_bollinger(closes[-30:], 20, 2.0)
    if bw is None: bw = 0.01

    if aligned and separation > 0.15 and atr_pct > 0.06:
        return "trending"
    elif bw < 0.008 or atr_pct < 0.04:
        return "ranging"
    else:
        return "choppy"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FOR ML
# ══════════════════════════════════════════════════════════════════════════════
def extract_features(ohlcv_list, closes_list, window_rows):
    """
    Extract 40+ features from last LOOKBACK candles for ML model.
    Returns None if not enough data.
    """
    if len(ohlcv_list) < LOOKBACK + 5 or len(closes_list) < LOOKBACK + 5:
        return None
    try:
        closes = closes_list[-LOOKBACK:]
        ohlcv  = ohlcv_list[-LOOKBACK:]

        feats = []

        # Price momentum features
        ret1  = safe_div(closes[-1] - closes[-2],  closes[-2])
        ret3  = safe_div(closes[-1] - closes[-4],  closes[-4])
        ret5  = safe_div(closes[-1] - closes[-6],  closes[-6])
        ret10 = safe_div(closes[-1] - closes[-11], closes[-11]) if len(closes) > 11 else 0
        feats += [ret1, ret3, ret5, ret10]

        # EMA ribbon features
        ribbon = compute_ema_ribbon(closes)
        price = closes[-1]
        for p in EMA_PERIODS:
            v = ribbon.get(p)
            feats.append(safe_div(price - v, v) if v else 0)
        # EMA cross ratios
        if ribbon[9] and ribbon[20]:
            feats.append(safe_div(ribbon[9] - ribbon[20], ribbon[20]))
        else:
            feats.append(0)
        if ribbon[20] and ribbon[50]:
            feats.append(safe_div(ribbon[20] - ribbon[50], ribbon[50]))
        else:
            feats.append(0)

        # RSI
        rsi = compute_rsi(closes)
        feats.append(safe_div(rsi, 100) if rsi else 0.5)

        # MACD
        mv, sv, hist, delta = compute_macd(closes)
        feats.append(hist if hist else 0)
        feats.append(delta if delta else 0)

        # Bollinger
        bb_up, bb_mid, bb_lo, pct_b, bw = compute_bollinger(closes)
        feats += [pct_b if pct_b else 0.5, bw if bw else 0]

        # ATR
        atr = compute_atr(ohlcv)
        feats.append(safe_div(atr, price) if atr else 0)

        # Stochastic
        k, d = compute_stochastic(ohlcv)
        feats += [safe_div(k, 100) if k else 0.5, safe_div(d, 100) if d else 0.5]

        # Williams %R
        wr = compute_williams_r(ohlcv)
        feats.append(safe_div(wr + 100, 100) if wr else 0.5)

        # CMF
        cmf = compute_cmf(ohlcv)
        feats.append(cmf if cmf else 0)

        # OBV slope normalized
        obv = compute_obv_slope(ohlcv)
        vol_sum = sum(c[4] for c in ohlcv[-OBV_WINDOW:]) or 1
        feats.append(safe_div(obv, vol_sum) if obv else 0)

        # Donchian position
        _, _, _, dpos = compute_donchian(ohlcv)
        feats.append(dpos if dpos else 0.5)

        # VWAP deviation
        vwap = compute_vwap_rolling(ohlcv)
        feats.append(safe_div(price - vwap, vwap) if vwap else 0)

        # Candle body / wick features (last 3 candles)
        for i in range(-3, 0):
            o_, h_, l_, c_, v_ = ohlcv[i]
            rng = h_ - l_
            body = abs(c_ - o_)
            feats.append(safe_div(body, rng) if rng else 0)        # body/range
            feats.append(safe_div(c_ - o_, o_) if o_ else 0)       # return
            upper_wick = h_ - max(o_, c_)
            lower_wick = min(o_, c_) - l_
            feats.append(safe_div(upper_wick, rng) if rng else 0)
            feats.append(safe_div(lower_wick, rng) if rng else 0)
            feats.append(safe_div(v_, sum(c[4] for c in ohlcv[-10:]) / 10) if ohlcv else 1)

        # Recent green/red ratio
        last20 = window_rows[-20:] if len(window_rows) >= 20 else window_rows
        g = sum(1 for r in last20 if r[1] == "green")
        feats.append(safe_div(g, len(last20)))

        # Volatility regime
        std5 = statistics.stdev(closes[-5:]) if len(closes) >= 5 else 0
        std20 = statistics.stdev(closes[-20:]) if len(closes) >= 20 else 0
        feats.append(safe_div(std5, std20) if std20 else 1)

        return feats
    except Exception as e:
        print(f"  ⚠️  Feature extraction error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CANDLE PREDICTOR v3
# ══════════════════════════════════════════════════════════════════════════════
class CandlePredictor:

    def __init__(self):
        self.window = collections.deque(maxlen=WINDOW_SIZE)
        self.total_green = self.total_red = self.total_doji = 0
        self.markov = {
            "green": {"green": 0, "red": 0},
            "red":   {"green": 0, "red": 0},
            "doji":  {"green": 0, "red": 0},
        }
        self.predictions_made    = 0
        self.predictions_correct = 0
        self.predictions_skipped = 0
        self.last_prediction     = None
        self.last_candle_time    = None
        self.loss_streak         = 0
        self.win_streak          = 0
        self._backtest_log       = []  # (pred, actual, conf, regime)

        # MTF state
        self.mtf_biases = {}    # label → (bias_str, score)
        self.mtf_alignment = 0.0

        # Regime
        self.regime = "trending"

        # ML model
        self.ml_model   = None
        self.ml_scaler  = None
        self.ml_trained = False
        self.ml_confidence = 0.0
        self.ml_last_features = None
        self._candles_since_retrain = 0

        # Adaptive confidence gate
        self.recent_outcomes = collections.deque(maxlen=CONF_ADAPT_WINDOW)
        self.min_conf_adaptive = BASE_MIN_CONF

    # ── Ingest ────────────────────────────────────────────────────────────────
    def add_candle(self, candle):
        direction = classify(candle)
        o, h, l, c, v = (float(candle[i]) for i in range(1, 6))

        if self.window:
            prev = self.window[-1][1]
            if prev in self.markov and direction in ("green", "red"):
                self.markov[prev][direction] += 1

        if len(self.window) == self.window.maxlen:
            od = self.window[0][1]
            if od == "green": self.total_green -= 1
            elif od == "red": self.total_red -= 1
            else: self.total_doji -= 1

        self.window.append((int(candle[0]), direction, o, h, l, c, v))
        if direction == "green": self.total_green += 1
        elif direction == "red": self.total_red += 1
        else: self.total_doji += 1
        self.last_candle_time = int(candle[0])
        self._candles_since_retrain += 1

        # Trigger ML retrain
        if ML_AVAILABLE and self._candles_since_retrain >= ML_RETRAIN_FREQ:
            threading.Thread(target=self.train_ml_model, daemon=True).start()
            self._candles_since_retrain = 0

        return direction

    # ── OHLCV views ───────────────────────────────────────────────────────────
    def _ohlcv(self):
        return [(r[2], r[3], r[4], r[5], r[6]) for r in self.window]

    def _closes(self):
        return [r[5] for r in self.window]

    # ── ML Training ───────────────────────────────────────────────────────────
    def train_ml_model(self, force=False):
        if not ML_AVAILABLE:
            return "ML not available — install scikit-learn"
        if len(self.window) < ML_TRAIN_MIN:
            return f"Need {ML_TRAIN_MIN} candles, have {len(self.window)}"

        rows  = list(self.window)
        ohlcv = [(r[2], r[3], r[4], r[5], r[6]) for r in rows]
        closes = [r[5] for r in rows]

        X, y = [], []
        min_start = max(LOOKBACK + 10, 0)
        for i in range(min_start, len(rows) - 1):
            feats = extract_features(ohlcv[:i+1], closes[:i+1], rows[:i+1])
            if feats is None: continue
            label = 1 if rows[i+1][1] == "green" else 0
            if rows[i+1][1] == "doji": continue
            X.append(feats); y.append(label)

        if len(X) < 100:
            return f"Insufficient training samples: {len(X)}"

        try:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            rf  = RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_leaf=5,
                max_features="sqrt", random_state=42, n_jobs=-1,
                class_weight="balanced"
            )
            rf.fit(Xs, y)

            # Cross-val score
            cv_scores = cross_val_score(rf, Xs, y, cv=5, scoring="accuracy")
            cv_mean   = round(cv_scores.mean() * 100, 2)

            self.ml_model   = rf
            self.ml_scaler  = scaler
            self.ml_trained = True
            self._candles_since_retrain = 0

            msg = (f"ML RandomForest retrained! {len(X)} samples | "
                   f"CV accuracy: {cv_mean}%")
            print(f"  🤖 {msg}")
            return msg
        except Exception as e:
            return f"ML training failed: {e}"

    def _ml_predict(self, ohlcv, closes):
        if not self.ml_trained or self.ml_model is None: return None, 0.0
        feats = extract_features(ohlcv, closes, list(self.window))
        if feats is None: return None, 0.0
        try:
            Xs = self.ml_scaler.transform([feats])
            proba = self.ml_model.predict_proba(Xs)[0]
            # proba[0] = P(red), proba[1] = P(green)
            pred = "green" if proba[1] >= 0.5 else "red"
            conf = abs(proba[1] - 0.5) * 200   # 0-100
            self.ml_confidence = round(conf, 1)
            return pred, conf
        except Exception:
            return None, 0.0

    # ── Adaptive confidence gate ──────────────────────────────────────────────
    def _update_adaptive_gate(self, correct: bool):
        self.recent_outcomes.append(1 if correct else 0)
        if len(self.recent_outcomes) >= 20:
            recent_acc = sum(self.recent_outcomes) / len(self.recent_outcomes)
            # if accuracy < 50%, raise gate; if accuracy > 60%, lower gate
            if recent_acc < 0.50:
                self.min_conf_adaptive = min(self.min_conf_adaptive + 1.0, 20.0)
            elif recent_acc > 0.60:
                self.min_conf_adaptive = max(self.min_conf_adaptive - 0.5, BASE_MIN_CONF)

    # ── Main prediction engine ─────────────────────────────────────────────────
    def predict_next(self):
        if len(self.window) < MIN_CANDLES:
            return None, 0.0, {}, False, "warming_up"

        signals      = {}
        green_score  = 0.0
        total_weight = 0.0
        ohlcv        = self._ohlcv()
        closes       = self._closes()
        regime       = self.regime
        ridx         = REGIME_IDX.get(regime, 0)

        def ws(name):
            """Get regime-aware weight for signal name."""
            vals = WEIGHTS.get(name, [1.0, 1.0, 1.0])
            return vals[ridx] if ridx < len(vals) else 1.0

        def add_signal(name, score_0_to_1, label):
            nonlocal green_score, total_weight
            w = ws(name)
            green_score  += score_0_to_1 * w
            total_weight += w
            signals[name] = label

        # ── ML ensemble (highest weight) ──────────────────────────────────────
        ml_pred, ml_conf = self._ml_predict(ohlcv, closes)
        if ml_pred is not None:
            ml_s = 0.5 + (ml_conf / 200)  # map conf to [0.5, 1.0]
            ml_s = ml_s if ml_pred == "green" else (1 - ml_s)
            add_signal("ML_RF", ml_s,
                f"{'🟢' if ml_pred=='green' else '🔴'} {ml_pred.upper()} "
                f"P={ml_conf:.1f}%")

        # ── Signal: Momentum (50-candle green%) ───────────────────────────────
        recent = list(self.window)[-50:]
        rg = sum(1 for r in recent if r[1] == "green")
        rr = sum(1 for r in recent if r[1] == "red")
        rt = rg + rr
        if rt > 0:
            mom = rg / rt
            add_signal("Momentum(50)", mom, f"{'🟢' if mom>0.5 else '🔴'} {mom*100:.1f}%")

        # ── Signal: Streak ────────────────────────────────────────────────────
        sc = [r[1] for r in list(self.window)[-10:]]
        last_dir = sc[-1] if sc else "doji"
        sl = sum(1 for d in reversed(sc) if d == last_dir) if last_dir != "doji" else 0
        if last_dir != "doji" and sl >= 2:
            rw = min(sl / 6, 1.0)
            ss = (0.5 - rw * 0.35) if last_dir == "green" else (0.5 + rw * 0.35)
            add_signal("Streak", ss,
                f"{'🟢' if last_dir=='green' else '🔴'} {sl}× → "
                f"{'reversal' if rw > 0.3 else 'continuation'}")

        # ── Signal: Markov ────────────────────────────────────────────────────
        if last_dir in self.markov:
            m = self.markov[last_dir]; mt = m["green"] + m["red"]
            if mt > 10:
                ms = m["green"] / mt
                add_signal("Markov", ms,
                    f"After {last_dir}: 🟢{m['green']}/🔴{m['red']} ({ms*100:.1f}%)")

        # ── Signal: EMA Ribbon ────────────────────────────────────────────────
        ribbon = compute_ema_ribbon(closes)
        price  = closes[-1]
        valid_ribbon = {p: v for p, v in ribbon.items() if v is not None}
        if len(valid_ribbon) >= 2:
            above = sum(1 for v in valid_ribbon.values() if price > v)
            total_emas = len(valid_ribbon)
            rs = above / total_emas
            # alignment check
            vals_sorted = sorted(valid_ribbon.items())
            ascending  = all(vals_sorted[i][1] < vals_sorted[i+1][1] for i in range(len(vals_sorted)-1))
            descending = all(vals_sorted[i][1] > vals_sorted[i+1][1] for i in range(len(vals_sorted)-1))
            alignment_note = " 📈aligned" if ascending and price > vals_sorted[0][1] else \
                             (" 📉aligned" if descending and price < vals_sorted[-1][1] else "")
            add_signal("EMA Ribbon", rs,
                f"Price {'above' if rs>0.5 else 'below'} {above}/{total_emas} EMAs{alignment_note}")

        # ── Signal: RSI ────────────────────────────────────────────────────────
        rsi = compute_rsi(closes)
        if rsi is not None:
            if rsi > 70:   rs2, lbl = 0.28, f"🔴 Overbought {rsi:.1f}"
            elif rsi < 30: rs2, lbl = 0.72, f"🟢 Oversold {rsi:.1f}"
            elif rsi > 55: rs2, lbl = 0.58, f"🟢 Bullish {rsi:.1f}"
            elif rsi < 45: rs2, lbl = 0.42, f"🔴 Bearish {rsi:.1f}"
            else:          rs2, lbl = 0.50, f"⚪ Neutral {rsi:.1f}"
            add_signal("RSI(14)", rs2, lbl)

        # ── Signal: MACD + Delta ──────────────────────────────────────────────
        macd_v, sig_v, hist_v, delta_v = compute_macd(closes)
        if hist_v is not None:
            ms2 = 0.65 if hist_v > 0 else 0.35
            add_signal("MACD", ms2,
                f"{'🟢 Bullish' if hist_v>0 else '🔴 Bearish'} hist={hist_v:+.3f}")
        if delta_v is not None:
            ds = 0.65 if delta_v > 0 else 0.35
            add_signal("MACD Delta", ds,
                f"{'🟢 Accel' if delta_v>0 else '🔴 Decel'} Δhist={delta_v:+.4f}")

        # ── Signal: Bollinger Bands ────────────────────────────────────────────
        bb_up, bb_mid, bb_lo, pct_b, bw = compute_bollinger(closes)
        if pct_b is not None:
            if bw < 0.005:
                bb_s = 0.60 if mom > 0.5 else 0.40
                lbl  = f"🔵 Squeeze bw={bw*100:.3f}% → breakout pending"
            elif pct_b > 0.95: bb_s, lbl = 0.30, f"🔴 Upper band %B={pct_b:.2f}"
            elif pct_b < 0.05: bb_s, lbl = 0.70, f"🟢 Lower band %B={pct_b:.2f}"
            elif pct_b > 0.75: bb_s, lbl = 0.58, f"🟢 Upper half %B={pct_b:.2f}"
            elif pct_b < 0.25: bb_s, lbl = 0.42, f"🔴 Lower half %B={pct_b:.2f}"
            else:              bb_s, lbl = 0.50, f"⚪ Mid %B={pct_b:.2f}"
            add_signal("Bollinger", bb_s, lbl)

        # ── Signal: VWAP ──────────────────────────────────────────────────────
        vwap = compute_vwap_rolling(ohlcv, VWAP_SESSION_CANDLES)
        if vwap is not None:
            vd = (price - vwap) / vwap * 100
            vs = 0.68 if price > vwap else 0.32
            add_signal("VWAP", vs,
                f"{'above' if price>vwap else 'below'} VWAP ({vd:+.3f}%) ${vwap:,.2f}")

        # ── Signal: Stochastic ────────────────────────────────────────────────
        k, d = compute_stochastic(ohlcv)
        if k is not None:
            if k < 20:     st_s, lbl = 0.72, f"🟢 Oversold %K={k}"
            elif k > 80:   st_s, lbl = 0.28, f"🔴 Overbought %K={k}"
            elif k > d:    st_s, lbl = 0.60, f"🟢 Bullish cross K={k} D={d}"
            elif k < d:    st_s, lbl = 0.40, f"🔴 Bearish cross K={k} D={d}"
            else:          st_s, lbl = 0.50, f"⚪ Neutral K={k}"
            add_signal("Stochastic", st_s, lbl)

        # ── Signal: OBV ────────────────────────────────────────────────────────
        obv_slope = compute_obv_slope(ohlcv)
        if obv_slope is not None:
            add_signal("OBV", 0.65 if obv_slope > 0 else 0.35,
                f"{'🟢 Accum' if obv_slope>0 else '🔴 Distrib'} slope={obv_slope:+.0f}")

        # ── Signal: CMF ────────────────────────────────────────────────────────
        cmf_val = compute_cmf(ohlcv)
        if cmf_val is not None:
            cs = 0.5 + cmf_val * 0.3   # scale [-1,1] to [0.2, 0.8]
            cs = max(0.2, min(0.8, cs))
            add_signal("CMF", cs,
                f"{'🟢' if cmf_val>0 else '🔴'} {cmf_val:+.3f}")

        # ── Signal: Williams %R ────────────────────────────────────────────────
        wr = compute_williams_r(ohlcv)
        if wr is not None:
            if wr < -80:   ws2, lbl = 0.70, f"🟢 Oversold {wr:.1f}"
            elif wr > -20: ws2, lbl = 0.30, f"🔴 Overbought {wr:.1f}"
            elif wr > -50: ws2, lbl = 0.58, f"🟢 Bullish {wr:.1f}"
            else:          ws2, lbl = 0.42, f"🔴 Bearish {wr:.1f}"
            add_signal("Williams%R", ws2, lbl)

        # ── Signal: Donchian Channel ──────────────────────────────────────────
        dhi, dlo, dmid, dpos = compute_donchian(ohlcv)
        if dpos is not None:
            ds2 = 0.65 if dpos > 0.6 else (0.35 if dpos < 0.4 else 0.50)
            add_signal("Donchian", ds2,
                f"{'🟢 Upper' if dpos>0.6 else '🔴 Lower' if dpos<0.4 else '⚪ Mid'}"
                f" pos={dpos:.2f}")

        # ── Signal: Candle body/wick ──────────────────────────────────────────
        last = self.window[-1]
        o_, h_, l_, c_, v_ = last[2], last[3], last[4], last[5], last[6]
        if o_ > 0:
            body = abs(c_ - o_) / o_ * 100
            up_wick = (h_ - max(o_, c_)) / o_ * 100
            dn_wick = (min(o_, c_) - l_) / o_ * 100
            if body > 0.03:
                if up_wick > body * 2.0 and last[1] == "green":
                    cbs, cbl = 0.35, f"🔴 Shooting star wick={up_wick:.3f}%"
                elif dn_wick > body * 2.0 and last[1] == "red":
                    cbs, cbl = 0.65, f"🟢 Hammer wick={dn_wick:.3f}%"
                elif last[1] == "green" and body > 0.1:
                    cbs, cbl = 0.58, f"🟢 Strong body {body:.3f}%"
                elif last[1] == "red" and body > 0.1:
                    cbs, cbl = 0.42, f"🔴 Strong body {body:.3f}%"
                else:
                    cbs, cbl = 0.50, f"⚪ Body {body:.3f}%"
                add_signal("Candle Body", cbs, cbl)

        # ── Signal: MTF bias stack ────────────────────────────────────────────
        mtf_bull = mtf_bear = 0
        for label, (bias, score) in self.mtf_biases.items():
            if bias == "bullish": mtf_bull += 1
            elif bias == "bearish": mtf_bear += 1
        mtf_total = mtf_bull + mtf_bear
        if mtf_total > 0:
            alignment = mtf_bull / mtf_total
            # Bonus for strong alignment
            if mtf_bull >= 3 or mtf_bear >= 3:
                alignment = 0.72 if mtf_bull > mtf_bear else 0.28
            add_signal("MTF", alignment,
                f"{'🟢' if mtf_bull>mtf_bear else '🔴'} {mtf_bull}↑/{mtf_bear}↓ of {mtf_total} TFs")

        if total_weight == 0:
            return None, 0.0, {}, False, regime

        # ── Aggregate ─────────────────────────────────────────────────────────
        fs   = green_score / total_weight
        conf = round(abs(fs - 0.5) * 200, 1)
        pred = "green" if fs >= 0.5 else "red"

        # MTF alignment bonus: if all TFs agree, boost confidence
        if mtf_total >= 3:
            alignment_ratio = mtf_bull / mtf_total if mtf_bull > mtf_bear else mtf_bear / mtf_total
            if alignment_ratio >= 0.75:
                conf = min(conf * 1.15, 100.0)

        # ── ATR gate ──────────────────────────────────────────────────────────
        atr = compute_atr(ohlcv)
        is_flat = False
        if atr and closes:
            atr_pct = atr / closes[-1] * 100
            shared_state["atr_pct"] = round(atr_pct, 4)
            is_flat = atr_pct < ATR_FLAT_PCT or atr_pct > ATR_EXTREME_PCT
            if atr_pct > ATR_EXTREME_PCT:
                print(f"  ⚠️  Hyper-volatile ATR={atr_pct:.3f}% — skipping")

        return pred, conf, signals, is_flat, regime

    # ── Outcome recording ─────────────────────────────────────────────────────
    def record_outcome(self, actual):
        if self.last_prediction and actual in ("green", "red"):
            self.predictions_made += 1
            correct = self.last_prediction == actual
            if correct:
                self.predictions_correct += 1
                self.loss_streak = 0
                self.win_streak += 1
            else:
                self.loss_streak += 1
                self.win_streak = 0

            self._update_adaptive_gate(correct)
            self._backtest_log.append((
                self.last_prediction, actual,
                shared_state.get("confidence", 0.0),
                self.regime
            ))
            log_prediction(self.last_prediction, actual,
                           shared_state.get("confidence", 0.0),
                           self.regime, correct)
            return correct
        return None

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def accuracy(self):
        return round(100 * self.predictions_correct / self.predictions_made, 2) \
               if self.predictions_made else 0.0
    @property
    def candle_count(self): return len(self.window)
    @property
    def green_pct(self):
        t = self.total_green + self.total_red
        return round(100 * self.total_green / t, 2) if t else 0
    @property
    def red_pct(self):
        t = self.total_green + self.total_red
        return round(100 * self.total_red / t, 2) if t else 0

    # ══════════════════════════════════════════════════════════════════════════
    # WALK-FORWARD BACKTEST
    # ══════════════════════════════════════════════════════════════════════════
    def run_backtest(self, candles):
        """True walk-forward (no look-ahead) backtest over full history."""
        bt = CandlePredictor()
        results = []
        total_skipped = 0
        print(f"  🔬 Walk-forward backtest: {len(candles):,} candles...")

        # We need MIN_CANDLES warmup
        for i, candle in enumerate(candles):
            if i == 0: bt.add_candle(candle); continue

            pred, conf, _, is_flat, regime = bt.predict_next()
            actual = classify(candle)
            bt.add_candle(candle)

            if pred is None or actual == "doji": continue
            if is_flat or conf < bt.min_conf_adaptive:
                total_skipped += 1
                results.append((None, conf, True, regime)); continue

            correct = pred == actual
            results.append((correct, conf, False, regime))

        acted = [(c, conf, reg) for c, conf, sk, reg in results if not sk and c is not None]
        total = len(acted); correct = sum(1 for c, _, _ in acted if c)

        by_conf = {}
        for t in CONFIDENCE_BANDS:
            subset = [(c, conf) for c, conf, _ in acted if conf >= t]
            st = len(subset); sc = sum(1 for c, _ in subset if c)
            by_conf[t] = {"total": st, "correct": sc, "wrong": st-sc,
                          "accuracy": round(100*sc/st, 2) if st else 0}

        by_regime = {}
        for c, conf, reg in acted:
            if reg not in by_regime: by_regime[reg] = {"total": 0, "correct": 0}
            by_regime[reg]["total"] += 1
            if c: by_regime[reg]["correct"] += 1
        for reg in by_regime:
            d = by_regime[reg]
            d["accuracy"] = round(100 * d["correct"] / d["total"], 2) if d["total"] else 0

        # Streaks
        lc = lw = 0; ck = None; cl = 0
        for c, _, _ in acted:
            k = "c" if c else "w"
            if k == ck: cl += 1
            else:
                if ck == "c": lc = max(lc, cl)
                elif ck == "w": lw = max(lw, cl)
                ck = k; cl = 1
        if ck == "c": lc = max(lc, cl)
        elif ck == "w": lw = max(lw, cl)

        # Sharpe-like: (accuracy - 50) / std_of_outcomes
        outcomes = [1 if c else -1 for c, _, _ in acted]
        if len(outcomes) > 2:
            try:
                sharpe = (statistics.mean(outcomes)) / statistics.stdev(outcomes) * math.sqrt(len(outcomes))
            except Exception:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return {
            "total": total, "correct": correct, "wrong": total-correct,
            "skipped": total_skipped,
            "overall_accuracy": round(100*correct/total, 2) if total else 0,
            "by_confidence": by_conf, "by_regime": by_regime,
            "streak_correct": lc, "streak_wrong": lw,
            "sharpe_like": round(sharpe, 3),
        }

    def generate_backtest_report(self):
        bt = self.run_backtest(list(self.window))
        if not bt: return ["⚠️ Not enough data."]
        parts = []
        acc = bt["overall_accuracy"]
        acc_bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))

        parts.append(
            f"📋 <b>WALK-FORWARD BACKTEST v3</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  Acted   : <b>{bt['total']:,}</b>\n"
            f"  ✅ Correct: <b>{bt['correct']:,}</b>\n"
            f"  ❌ Wrong  : <b>{bt['wrong']:,}</b>\n"
            f"  ⏭ Skipped: <b>{bt['skipped']:,}</b>\n"
            f"  🎯 Overall: <b>{acc}%</b>\n"
            f"  [{acc_bar}]\n"
            f"  📊 Sharpe-like: <b>{bt['sharpe_like']}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  🔥 Best correct streak: <b>{bt['streak_correct']}</b>\n"
            f"  💀 Worst wrong streak : <b>{bt['streak_wrong']}</b>"
        )

        # Confidence bands
        band_lines = [
            "📡 <b>Accuracy by Confidence</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<code>Conf≥  Total   Correct  Acc%</code>"
        ]
        for t, d in bt["by_confidence"].items():
            if d["total"] == 0:
                band_lines.append(f"<code>{t:>3}%  {'—':>6}  {'—':>7}  {'—':>5}</code>")
            else:
                bar = "▓" * int(d["accuracy"] / 10) + "░" * (10 - int(d["accuracy"] / 10))
                band_lines.append(
                    f"<code>{t:>3}%  {d['total']:>6}  {d['correct']:>7}  "
                    f"{d['accuracy']:>5.1f}%</code> {bar}")
        parts.append("\n".join(band_lines))

        # Regime breakdown
        regime_lines = ["📈 <b>Accuracy by Regime</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n<code>Regime     Total  Acc%</code>"]
        for reg, d in bt["by_regime"].items():
            bar = "▓" * int(d["accuracy"] / 10) + "░" * (10 - int(d["accuracy"] / 10))
            regime_lines.append(
                f"<code>{reg:10} {d['total']:>6}  {d['accuracy']:>5.1f}%</code> {bar}")
        parts.append("\n".join(regime_lines))

        return parts

    def print_backtest_console(self, bt):
        if not bt: return
        sep = "═" * 66
        print(f"\n{sep}")
        print(f"  📋  WALK-FORWARD BACKTEST v3 ({len(self.window):,} candles)")
        print(sep)
        print(f"  Acted   : {bt['total']:,}   Skipped: {bt['skipped']:,}")
        print(f"  ✅ Correct: {bt['correct']:,}   ❌ Wrong: {bt['wrong']:,}")
        acc = bt["overall_accuracy"]
        print(f"  🎯 Accuracy: {acc}%  [{'█'*int(acc/5)}{'░'*(20-int(acc/5))}]")
        print(f"  📊 Sharpe-like score: {bt['sharpe_like']}")
        print(f"  🔥 Best streak: {bt['streak_correct']}   💀 Worst: {bt['streak_wrong']}")
        print(f"\n  {'Conf≥':8} {'Total':>8} {'Correct':>9} {'Wrong':>7} {'Acc':>8}")
        print(f"  {'-'*48}")
        for t, d in bt["by_confidence"].items():
            if d["total"] == 0:
                print(f"  {f'{t}%':8} {'—':>8} {'—':>9} {'—':>7} {'—':>8}")
            else:
                print(f"  {f'{t}%':8} {d['total']:>8,} {d['correct']:>9,} "
                      f"{d['wrong']:>7,} {d['accuracy']:>7.2f}%")
        print(f"\n  Regime Breakdown:")
        for reg, d in bt["by_regime"].items():
            print(f"    {reg:12}: {d['accuracy']}%  n={d['total']}")
        print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# BROADCAST MESSAGE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_broadcast(predictor, candle, prediction, confidence,
                    signals, actual_dir, outcome, is_flat=False):
    pe = "🟢" if prediction == "green" else "🔴"
    de = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")
    outcome_str = ("✅ Correct!" if outcome else "❌ Wrong") if outcome is not None else ""
    acc_bar = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    conf_emoji = "🔥🔥" if confidence > 30 else ("🔥" if confidence > 15 else ("⚡" if confidence > 8 else "〰️"))
    regime_emoji = {"trending": "📈", "choppy": "〰️", "ranging": "↔️"}.get(predictor.regime, "❓")
    flat_note = "\n  ⚠️ <i>Volatility gate active</i>" if is_flat else ""
    ml_line = ""
    if ML_AVAILABLE and predictor.ml_trained:
        ml_line = f"\n  🤖 ML: <b>{predictor.ml_confidence:.1f}%</b> confidence"

    # MTF summary
    mtf_summary = " | ".join(
        f"{'🟢' if b=='bullish' else '🔴' if b=='bearish' else '⚪'}{label}"
        for label, (b, _) in predictor.mtf_biases.items()
    ) or "loading..."

    sig_lines = "\n".join(
        f"  • <b>{n}</b>: {v}" for n, v in signals.items()
    )

    loss_warn = ""
    if predictor.loss_streak >= MAX_LOSS_STREAK - 2:
        loss_warn = f"\n  ⚠️ <i>High loss streak ({predictor.loss_streak}) — elevated uncertainty</i>"

    return (
        f"🤖 <b>BTC/USD 5m — v3 World Class</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {ts(candle[0])}\n"
        f"💵 <b>${float(candle[4]):,.2f}</b>"
        f"  {regime_emoji} <b>{predictor.regime.upper()}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Window</b>: {predictor.candle_count:,} candles\n"
        f"  🟢 {predictor.total_green:,} ({predictor.green_pct}%)"
        f"  🔴 {predictor.total_red:,} ({predictor.red_pct}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌐 <b>MTF Stack:</b> {mtf_summary}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕯 <b>Last:</b> {de} {actual_dir.upper()}  {outcome_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 <b>NEXT CANDLE:</b>\n"
        f"  {pe} <b>{prediction.upper()}</b>"
        f"  {conf_emoji} <b>{confidence:.1f}%</b>"
        f"{flat_note}{ml_line}{loss_warn}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 <b>Signals ({len(signals)}):</b>\n{sig_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>{predictor.accuracy}%</b> [{acc_bar}]\n"
        f"   {predictor.predictions_correct}/{predictor.predictions_made}"
        f"  ⏭{predictor.predictions_skipped}  👥{subscriber_count()}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>/predict /status /accuracy /regime</i>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def print_dashboard(predictor, candle, prediction, confidence, signals, actual_dir, is_flat):
    sep = "=" * 72
    g, r = predictor.total_green, predictor.total_red
    bt = g + r
    g_bar = int(44 * g / bt) if bt else 0
    acc_bar = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    pe = "🟢" if prediction == "green" else ("🔴" if prediction else "⏳")
    de = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")
    regime_emoji = {"trending": "📈", "choppy": "〰️", "ranging": "↔️"}.get(predictor.regime, "❓")

    print(f"\n{sep}")
    print(f"  🤖 BTC Predictor v3 ★  |  {ts(candle[0])}  |  ${float(candle[4]):,.2f}")
    print(sep)
    print(f"  Window: {predictor.candle_count:,}  |  🟢{g:,}({predictor.green_pct}%)  🔴{r:,}({predictor.red_pct}%)")
    print(f"  [{'█'*g_bar}{'░'*(44-g_bar)}]")
    print(f"  {regime_emoji} Regime: {predictor.regime.upper()}"
          + (" | ⚠️ VOLATILITY GATE" if is_flat else "")
          + (f" | 🔥 LOSS STREAK {predictor.loss_streak}" if predictor.loss_streak >= 3 else ""))
    print(f"  🌐 MTF: " + " | ".join(
        f"{'🟢' if b=='bullish' else '🔴' if b=='bearish' else '⚪'}{label}"
        for label, (b, _) in predictor.mtf_biases.items()
    ))
    print(f"  Last: {de} {actual_dir.upper()}")
    if prediction:
        skip_tag = " [GATE-SKIP]" if is_flat or confidence < predictor.min_conf_adaptive else ""
        print(f"  Next: {pe} {prediction.upper()}  ({confidence:.1f}%  gate={predictor.min_conf_adaptive:.1f}%){skip_tag}")
        if ML_AVAILABLE and predictor.ml_trained:
            print(f"  🤖 ML: {predictor.ml_confidence:.1f}% confidence")
        for n, v in list(signals.items())[:8]:
            print(f"    {n:<20}: {v}")
        if len(signals) > 8:
            print(f"    ... +{len(signals)-8} more signals")
        print(f"  🎯 Accuracy: {predictor.accuracy}%  [{acc_bar}]"
              f"  ({predictor.predictions_correct}/{predictor.predictions_made})"
              f"  skipped={predictor.predictions_skipped}")
        print(f"  👥 {subscriber_count()} subscribers")
    else:
        print(f"  ⏳ Warming up... {MIN_CANDLES - predictor.candle_count:,} more candles needed")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# WEB DASHBOARD (Flask)
# ══════════════════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<title>BTC Predictor v3</title>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;600&display=swap');
  :root{--g:#00ff88;--r:#ff4455;--b:#4488ff;--bg:#0a0b0e;--card:#13151a;--border:#1e2128;}
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--bg);color:#e0e0e0;font-family:'JetBrains Mono',monospace;padding:20px;}
  h1{color:var(--g);font-family:'Space Grotesk',sans-serif;font-size:1.8em;margin-bottom:20px;}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;}
  .card h2{font-size:.85em;color:#888;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;}
  .big{font-size:2.5em;font-weight:700;}
  .green{color:var(--g);}  .red{color:var(--r);}  .blue{color:var(--b);}
  .badge{display:inline-block;padding:4px 10px;border-radius:6px;font-size:.75em;font-weight:700;margin-top:8px;}
  .badge.green{background:#00ff8820;color:var(--g);border:1px solid #00ff8840;}
  .badge.red{background:#ff445520;color:var(--r);border:1px solid #ff445540;}
  .badge.blue{background:#4488ff20;color:var(--b);border:1px solid #4488ff40;}
  .sig{font-size:.75em;padding:4px 0;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;}
  .bar{height:8px;background:#1e2128;border-radius:4px;overflow:hidden;margin-top:6px;}
  .bar-fill{height:100%;border-radius:4px;transition:width .5s ease;}
  .ts{color:#555;font-size:.7em;margin-top:12px;}
</style>
</head>
<body>
<h1>⚡ BTC/USD 5m Predictor v3</h1>
<div class="grid">
  <div class="card">
    <h2>💵 Price</h2>
    <div class="big">${price:,.2f}</div>
    <span class="badge blue">BTC/USD</span>
  </div>
  <div class="card">
    <h2>🔮 Next Prediction</h2>
    <div class="big {pred_class}">{pred_upper}</div>
    <div class="bar"><div class="bar-fill" style="width:{conf_pct}%;background:{pred_color}"></div></div>
    <div style="margin-top:6px;font-size:.8em">Confidence: <b>{confidence:.1f}%</b></div>
    <span class="badge {pred_class}">{regime}</span>
  </div>
  <div class="card">
    <h2>🎯 Accuracy</h2>
    <div class="big green">{accuracy}%</div>
    <div class="bar"><div class="bar-fill" style="width:{accuracy}%;background:var(--g)"></div></div>
    <div style="margin-top:6px;font-size:.8em">{correct}/{total} correct | {skipped} skipped</div>
  </div>
  <div class="card">
    <h2>📊 Distribution</h2>
    <div style="display:flex;gap:20px">
      <div><div class="big green">{green_pct}%</div><div style="font-size:.7em;color:#888">GREEN</div></div>
      <div><div class="big red">{red_pct}%</div><div style="font-size:.7em;color:#888">RED</div></div>
    </div>
    <div class="bar" style="margin-top:12px">
      <div class="bar-fill" style="width:{green_pct}%;background:var(--g)"></div>
    </div>
  </div>
  <div class="card">
    <h2>📡 Signals</h2>
    {signal_rows}
  </div>
  <div class="card">
    <h2>🌐 MTF Stack</h2>
    {mtf_rows}
    <div class="ts">ML: {ml_status} | ATR: {atr_pct:.4f}%</div>
  </div>
</div>
<div class="ts" style="margin-top:20px">Updated: {last_update} | v3.0 | 👥 {subs} subscribers</div>
</body></html>"""

def start_dashboard():
    if not FLASK_AVAILABLE: return
    app = Flask(__name__)

    @app.route("/")
    def index():
        s = shared_state
        pred = s.get("prediction") or "—"
        pred_class = "green" if pred == "green" else "red"
        pred_color = "var(--g)" if pred == "green" else "var(--r)"
        sigs = s.get("signals", {})
        sig_rows = "".join(
            f'<div class="sig"><span>{n}</span><span>{v[:40]}</span></div>'
            for n, v in list(sigs.items())[:10])
        mtf = s.get("mtf_biases", {})
        mtf_rows = "".join(
            f'<div class="sig"><span>{label}</span>'
            f'<span class="{"green" if b=="bullish" else "red" if b=="bearish" else ""}">{b}</span></div>'
            for label, (b, _) in mtf.items())
        ml_status = f"✅ {s.get('ml_confidence', 0):.1f}%" if ML_AVAILABLE and s.get("ml_enabled") else "⚠️ off"
        html = DASHBOARD_HTML.format(
            price=s.get("price", 0), pred_upper=(pred.upper()),
            pred_class=pred_class, pred_color=pred_color,
            conf_pct=min(s.get("confidence", 0) * 2, 100),
            confidence=s.get("confidence", 0),
            regime=s.get("regime", "?").upper(),
            accuracy=s.get("accuracy", 0), correct=s.get("correct", 0),
            total=s.get("total", 0), skipped=s.get("skipped", 0),
            green_pct=s.get("green_pct", 50), red_pct=s.get("red_pct", 50),
            signal_rows=sig_rows, mtf_rows=mtf_rows, ml_status=ml_status,
            atr_pct=s.get("atr_pct", 0), last_update=s.get("last_update", "—"),
            subs=subscriber_count())
        return html

    @app.route("/api/state")
    def api_state():
        return jsonify(shared_state)

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=DASHBOARD_PORT, debug=False, use_reloader=False),
        daemon=True
    ).start()
    print(f"🌐 Dashboard started at http://localhost:{DASHBOARD_PORT}")


# ══════════════════════════════════════════════════════════════════════════════
# MTF REFRESH THREAD
# ══════════════════════════════════════════════════════════════════════════════
def mtf_refresh_loop(predictor):
    """Continuously refresh multi-timeframe biases in background."""
    while True:
        try:
            biases = {}
            total_score = 0.0
            for interval, label, _, weight in MTF_CONFIG:
                candles = fetch_mtf_candles(SYMBOL, interval, MTF_CANDLES_FETCH)
                if candles:
                    bias, score = mtf_bias_full(candles)
                    biases[label] = (bias, score)
                    total_score += score * weight
                time.sleep(0.5)
            predictor.mtf_biases = biases
            predictor.mtf_alignment = total_score
            shared_state["mtf_biases"] = {label: b for label, (b, _) in biases.items()}
        except Exception as e:
            print(f"  ⚠️  MTF refresh error: {e}")
        time.sleep(60)   # refresh every 60s (~3 candles)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _predictor_ref

    print("=" * 72)
    print("  BTC/USD 5m Predictor — ★ WORLD CLASS ★  v3.0")
    print("  ML: RandomForest + GradientBoosting | 15+ Signals | 5 Timeframes")
    print("=" * 72)

    init_db()
    add_subscriber(ADMIN_CHAT_ID, "admin")
    print(f"✅ DB ready. Subscribers: {subscriber_count()}")

    # ── Load history ──────────────────────────────────────────────────────────
    historical = fetch_historical(SYMBOL, INTERVAL, DAYS)
    predictor  = CandlePredictor()
    _predictor_ref = predictor

    print("⚙️  Building rolling window...")
    for candle in historical:
        predictor.add_candle(candle)
    print(f"✅ Window: {predictor.candle_count:,} candles"
          f" | 🟢{predictor.total_green:,} | 🔴{predictor.total_red:,}")

    # ── Initial regime detection ───────────────────────────────────────────────
    ohlcv  = predictor._ohlcv()
    closes = predictor._closes()
    predictor.regime = detect_regime(ohlcv[-200:], closes[-200:])
    print(f"  📈 Initial regime: {predictor.regime.upper()}")

    # ── ML initial train ──────────────────────────────────────────────────────
    if ML_AVAILABLE:
        print("\n🤖 Training initial ML model (RandomForest)...")
        result = predictor.train_ml_model(force=True)
        print(f"  {result}")

    # ── MTF initial fetch ─────────────────────────────────────────────────────
    print("\n📐 Fetching multi-timeframe candles (15m/1h/4h/1D)...")
    for interval, label, _, weight in MTF_CONFIG:
        candles = fetch_mtf_candles(SYMBOL, interval, MTF_CANDLES_FETCH)
        if candles:
            bias, score = mtf_bias_full(candles)
            predictor.mtf_biases[label] = (bias, score)
            print(f"  {label}: {bias} (score={score:+.3f})")

    # ── Initial backtest ──────────────────────────────────────────────────────
    print("\n🔬 Running walk-forward backtest...")
    bt_data = predictor.run_backtest(historical)
    predictor.print_backtest_console(bt_data)

    if bt_data:
        acc = bt_data["overall_accuracy"]
        acc_bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        send_message(ADMIN_CHAT_ID,
            f"📋 <b>Startup Backtest — v3</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  Acted   : {bt_data['total']:,}  |  Skipped: {bt_data['skipped']:,}\n"
            f"  ✅ Correct: {bt_data['correct']:,}\n"
            f"  ❌ Wrong  : {bt_data['wrong']:,}\n"
            f"  🎯 Accuracy: <b>{acc}%</b>  [{acc_bar}]\n"
            f"  📊 Sharpe-like: {bt_data['sharpe_like']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  🔥 Streak: {bt_data['streak_correct']} correct / {bt_data['streak_wrong']} wrong\n"
            f"  🤖 ML: {'ENABLED ✅' if ML_AVAILABLE else 'DISABLED ⚠️'}\n"
            f"  <i>Send /backtest for full regime breakdown</i>"
        )

    # ── Start threads ─────────────────────────────────────────────────────────
    threading.Thread(target=polling_thread, daemon=True).start()
    threading.Thread(target=mtf_refresh_loop, args=(predictor,), daemon=True).start()
    if FLASK_AVAILABLE:
        start_dashboard()

    # ── First prediction ──────────────────────────────────────────────────────
    prediction, confidence, signals, is_flat, regime = predictor.predict_next()
    predictor.last_prediction = prediction
    predictor.regime = regime

    shared_state.update({
        "prediction": prediction, "confidence": confidence,
        "accuracy": predictor.accuracy, "correct": predictor.predictions_correct,
        "total": predictor.predictions_made, "skipped": predictor.predictions_skipped,
        "green": predictor.total_green, "red": predictor.total_red,
        "green_pct": predictor.green_pct, "red_pct": predictor.red_pct,
        "price": float(historical[-1][4]) if historical else 0,
        "regime": predictor.regime, "signals": signals,
        "ml_confidence": predictor.ml_confidence,
        "min_conf_adaptive": predictor.min_conf_adaptive,
        "candle_count": predictor.candle_count,
        "loss_streak": predictor.loss_streak,
        "mtf_biases": {label: b for label, (b, _) in predictor.mtf_biases.items()},
        "last_update": now_str(),
    })

    regime_emoji = {"trending": "📈", "choppy": "〰️", "ranging": "↔️"}.get(regime, "❓")
    send_message(ADMIN_CHAT_ID,
        f"🚀 <b>BTC Predictor v3 LIVE!</b>\n\n"
        f"📦 {predictor.candle_count:,} candles | 🟢{predictor.green_pct}% 🔴{predictor.red_pct}%\n"
        f"{regime_emoji} Regime: {regime.upper()}\n"
        f"🤖 ML: {'Trained ✅' if predictor.ml_trained else 'Not available ⚠️'}\n"
        f"🌐 MTF: " + " | ".join(
            f"{'🟢' if b=='bullish' else '🔴' if b=='bearish' else '⚪'}{label}"
            for label, (b, _) in predictor.mtf_biases.items()
        ) + "\n"
        f"👥 Subscribers: {subscriber_count()}\n\n"
        f"🔮 First prediction: "
        f"{'🟢 GREEN' if prediction == 'green' else '🔴 RED'}"
        f" ({confidence:.1f}%)" + (" ⚠️ volatile" if is_flat else "") + "\n\n"
        f"<i>15+ signals | 5 TFs | RandomForest ML | Walk-forward backtest\n"
        f"Commands: /backtest /retrain /status /predict /accuracy /regime /dashboard</i>"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE LOOP
    # ══════════════════════════════════════════════════════════════════════════
    print("\n🔄 Entering live loop...\n")
    last_seen_time = int(historical[-1][0]) if historical else 0
    regime_recheck_counter = 0

    while True:
        try:
            latest      = fetch_latest_candle(SYMBOL, INTERVAL)
            candle_time = int(latest[0])

            if candle_time != last_seen_time:
                actual_dir = classify(latest)
                outcome    = predictor.record_outcome(actual_dir)
                predictor.add_candle(latest)

                # Regime update every 10 candles
                regime_recheck_counter += 1
                if regime_recheck_counter >= 10:
                    ohlcv  = predictor._ohlcv()
                    closes = predictor._closes()
                    predictor.regime = detect_regime(ohlcv[-200:], closes[-200:])
                    regime_recheck_counter = 0

                prediction, confidence, signals, is_flat, regime = predictor.predict_next()
                gate_triggered = is_flat or (confidence < predictor.min_conf_adaptive) \
                                 or (predictor.loss_streak >= MAX_LOSS_STREAK)

                if gate_triggered:
                    predictor.predictions_skipped += 1
                    reason = ("volatile" if is_flat else
                              f"low-conf {confidence:.1f}%" if confidence < predictor.min_conf_adaptive else
                              f"loss-streak {predictor.loss_streak}")
                    print(f"  ⏭  Skipped — {reason}")
                else:
                    predictor.last_prediction = prediction

                shared_state.update({
                    "dir": actual_dir, "prediction": prediction,
                    "confidence": confidence, "accuracy": predictor.accuracy,
                    "correct": predictor.predictions_correct,
                    "total": predictor.predictions_made,
                    "skipped": predictor.predictions_skipped,
                    "green": predictor.total_green, "red": predictor.total_red,
                    "green_pct": predictor.green_pct, "red_pct": predictor.red_pct,
                    "price": float(latest[4]), "regime": predictor.regime,
                    "signals": signals, "ml_confidence": predictor.ml_confidence,
                    "min_conf_adaptive": predictor.min_conf_adaptive,
                    "candle_count": predictor.candle_count,
                    "loss_streak": predictor.loss_streak,
                    "mtf_biases": {label: b for label, (b, _) in predictor.mtf_biases.items()},
                    "last_update": now_str(),
                })

                print_dashboard(predictor, latest, prediction, confidence,
                                signals, actual_dir, is_flat)

                if prediction and not gate_triggered:
                    msg = build_broadcast(predictor, latest, prediction, confidence,
                                          signals, actual_dir, outcome, is_flat)
                    broadcast(msg)

                last_seen_time = candle_time

            else:
                fo = float(latest[1]); fc = float(latest[4])
                fp = (fc - fo) / fo * 100 if fo else 0
                print(
                    f"  [{now_str()}] Forming "
                    f"{'🟢' if fc > fo else '🔴'} {fp:+.4f}%  |  "
                    f"Next: {'🟢' if prediction=='green' else '🔴' if prediction else '⏳'}"
                    f" ({confidence:.1f}%)  Regime: {predictor.regime[:4]}",
                    end="\r"
                )

        except requests.exceptions.RequestException as e:
            print(f"\n  ⚠️  Network: {e} — retrying...")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()
