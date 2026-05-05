"""
BTC/USD 5m Rolling Candle Predictor — PUBLIC BOT  v2.0
─────────────────────────────────────────────────────────────────────────────
ACCURACY IMPROVEMENTS vs v1.0
  ① VWAP          — volume-weighted avg price; institutional bias signal
  ② MACD          — momentum crossover (fast 12 / slow 26 / signal 9)
  ③ Bollinger Bands — volatility squeeze + breakout signal
  ④ ATR Filter    — skip broadcast when market is flat / choppy
  ⑤ Multi-Timeframe — 15m & 1h trend bias imported via separate fetches
  ⑥ OHLCV features  — body size, wick ratio, candle range fed into signals
  ⑦ Confidence gate — suppress predictions below MIN_CONFIDENCE threshold
  ⑧ Stochastic    — overbought/oversold + crossover momentum
  ⑨ OBV           — on-balance volume accumulation/distribution trend
  ⑩ EMA crossover — fast(9) vs slow(20) replaces single EMA

COMMANDS
  /start       — subscribe
  /stop        — unsubscribe
  /status      — live stats
  /subscribers — admin: list users
  /backtest    — admin: full confidence-band backtest report
─────────────────────────────────────────────────────────────────────────────
"""

import requests
import time
import os
import sqlite3
import threading
import collections
import statistics
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",  "8766778348:AAEpkHO55y_oCrJ0vrTwtXsm8cWE_4IOZxA")
ADMIN_CHAT_ID    = os.environ.get("ADMIN_CHAT_ID",   "5792224870")

SYMBOL           = "BTCUSDT"
INTERVAL         = "5"          # primary timeframe
INTERVAL_15      = "15"         # MTF mid
INTERVAL_60      = "60"         # MTF high
BYBIT_URL        = "https://api.bybit.com/v5/market/kline"
DAYS             = 365
BATCH_SIZE       = 1000
MS_PER_5MIN      = 5 * 60 * 1000

WINDOW_SIZE      = 365 * 24 * 12   # ~105 120 candles
MIN_CANDLES      = 1000

# ── Signal parameters ─────────────────────────────────────────────────────────
MOMENTUM_WINDOW  = 50
STREAK_WINDOW    = 10
EMA_FAST         = 9
EMA_SLOW         = 20
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
BB_PERIOD        = 20
BB_STD           = 2.0
ATR_PERIOD       = 14
STOCH_PERIOD     = 14
OBV_WINDOW       = 20
MTF_CANDLES      = 50

# ── Gates ─────────────────────────────────────────────────────────────────────
MIN_CONFIDENCE   = 5.0          # skip broadcast below this % (⑦)
ATR_FLAT_PCT     = 0.05         # ATR/price below this → flat market (⑤)

POLL_INTERVAL    = 2
DB_FILE          = "subscribers.db"
CONFIDENCE_BANDS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
# ─────────────────────────────────────────────────────────────────────────────


def ts(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def now_str():
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


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
            joined_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_subscriber(chat_id, username=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT OR IGNORE INTO subscribers (chat_id, username, joined_at)
        VALUES (?, ?, ?)
    """, (str(chat_id), username, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

def remove_subscriber(chat_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM subscribers WHERE chat_id = ?", (str(chat_id),))
    conn.commit()
    conn.close()

def get_all_subscribers():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT chat_id, username FROM subscribers")
    rows = c.fetchall()
    conn.close()
    return rows

def is_subscribed(chat_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT 1 FROM subscribers WHERE chat_id = ?", (str(chat_id),))
    result = c.fetchone()
    conn.close()
    return result is not None

def subscriber_count():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM subscribers")
    count = c.fetchone()[0]
    conn.close()
    return count


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_message(chat_id, text):
    try:
        r = requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id":    str(chat_id),
            "text":       text,
            "parse_mode": "HTML",
        }, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"  ❌ Send error to {chat_id}: {e}")
        return False

def broadcast(text):
    subscribers = get_all_subscribers()
    sent = failed = 0
    dead = []
    for chat_id, _ in subscribers:
        ok = send_message(chat_id, text)
        if ok:
            sent += 1
        else:
            failed += 1
            dead.append(chat_id)
        time.sleep(0.05)
    for d in dead:
        remove_subscriber(d)
    print(f"  📢 Broadcast: ✅{sent} sent  ❌{failed} removed")
    return sent

def get_updates(offset=None):
    params = {"timeout": 30, "limit": 100}
    if offset:
        params["offset"] = offset
    try:
        r = requests.get(f"{BASE_URL}/getUpdates", params=params, timeout=35)
        r.raise_for_status()
        return r.json().get("result", [])
    except Exception as e:
        print(f"  ⚠️  getUpdates error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE & COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
shared_state = {
    "streak": 0, "dir": "unknown", "prediction": None,
    "confidence": 0, "accuracy": 0.0, "correct": 0, "total": 0,
    "green": 0, "red": 0, "green_pct": 0, "red_pct": 0,
    "price": 0, "skipped": 0, "atr_pct": 0.0,
    "mtf_15_bias": "—", "mtf_1h_bias": "—",
}
_predictor_ref = None


def handle_start(chat_id, username):
    if is_subscribed(chat_id):
        send_message(chat_id,
            "✅ <b>You're already subscribed!</b>\n\n"
            "You receive BTC/USD 5m candle predictions every 5 minutes.\n\n"
            "Commands:\n"
            "/stop   — Unsubscribe\n"
            "/status — See current prediction & stats"
        )
    else:
        add_subscriber(chat_id, username)
        send_message(chat_id,
            "🎉 <b>Welcome! You're now subscribed!</b>\n\n"
            "📌 <b>What you'll get every 5 minutes:</b>\n"
            "  • Last candle result (🟢/🔴)\n"
            "  • Next candle prediction\n"
            "  • Live accuracy score\n"
            "  • 12 signals: VWAP · MACD · BB · Stoch · OBV · MTF + more\n\n"
            "⚡ <b>Commands:</b>\n"
            "  /start  — Subscribe\n"
            "  /stop   — Unsubscribe\n"
            "  /status — Current stats\n\n"
            "<i>First update arrives at the next 5m candle close! 🚀</i>"
        )
        print(f"  ➕ New subscriber: {username} ({chat_id})")
        send_message(ADMIN_CHAT_ID,
            f"➕ <b>New subscriber!</b>\n"
            f"👤 {username or 'Unknown'} ({chat_id})\n"
            f"👥 Total: {subscriber_count()}"
        )

def handle_stop(chat_id, username):
    if is_subscribed(chat_id):
        remove_subscriber(chat_id)
        send_message(chat_id,
            "😢 <b>You've been unsubscribed.</b>\n\n"
            "Send /start anytime to subscribe again!"
        )
        print(f"  ➖ Unsubscribed: {username} ({chat_id})")
    else:
        send_message(chat_id, "⚠️ You're not subscribed.\nSend /start to subscribe!")

def handle_status(chat_id):
    s = shared_state
    pred = s["prediction"]
    pred_emoji = "🟢" if pred == "green" else ("🔴" if pred == "red" else "⏳")
    acc_bar = "█" * int(s["accuracy"] / 5) + "░" * (20 - int(s["accuracy"] / 5))
    flat_warn = "  ⚠️ <i>Market flat — ATR filter active</i>\n" \
                if s["atr_pct"] < ATR_FLAT_PCT else ""
    send_message(chat_id,
        f"📊 <b>BTC/USD 5m Predictor v2 — Live Status</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 BTC Price : ${s['price']:,.2f}\n"
        f"🟢 Green     : {s['green']:,} ({s['green_pct']}%)\n"
        f"🔴 Red       : {s['red']:,} ({s['red_pct']}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📐 15m bias  : {s['mtf_15_bias']}\n"
        f"📐 1h  bias  : {s['mtf_1h_bias']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 Next prediction : {pred_emoji} <b>{(pred or 'warming up').upper()}</b>\n"
        f"   Confidence      : <b>{s['confidence']:.1f}%</b>\n"
        f"{flat_warn}"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Accuracy  : <b>{s['accuracy']}%</b>\n"
        f"   [{acc_bar}]\n"
        f"   {s['correct']}/{s['total']} correct  |  {s['skipped']} skipped\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👥 Subscribers : {subscriber_count()}"
    )

def handle_subscribers(chat_id):
    if str(chat_id) != str(ADMIN_CHAT_ID):
        send_message(chat_id, "⛔ Admin only command.")
        return
    subs = get_all_subscribers()
    count = len(subs)
    lines = "\n".join(
        f"  {i+1}. {u or 'Unknown'} ({cid})"
        for i, (cid, u) in enumerate(subs[:20])
    )
    send_message(chat_id,
        f"👥 <b>Subscribers ({count} total)</b>\n\n"
        f"<code>{lines}</code>"
        + (f"\n\n<i>...and {count-20} more</i>" if count > 20 else "")
    )

def handle_backtest(chat_id):
    if str(chat_id) != str(ADMIN_CHAT_ID):
        send_message(chat_id, "⛔ Admin only command.")
        return
    if _predictor_ref is None:
        send_message(chat_id, "⏳ Bot is still warming up. Try again shortly.")
        return
    send_message(chat_id, "🔬 <b>Running backtest on current window...</b>")
    try:
        report = _predictor_ref.generate_backtest_report()
        for part in report:
            send_message(chat_id, part)
    except Exception as e:
        send_message(chat_id, f"❌ Backtest error: {e}")
        print(f"  ❌ Backtest error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM POLLING THREAD
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
                if not msg:
                    continue
                chat_id  = str(msg["chat"]["id"])
                username = msg.get("from", {}).get("username") or \
                           msg.get("from", {}).get("first_name", "Unknown")
                text     = msg.get("text", "").strip().lower().split()[0]

                if   text == "/start":         handle_start(chat_id, username)
                elif text == "/stop":          handle_stop(chat_id, username)
                elif text == "/status":        handle_status(chat_id)
                elif text == "/subscribers":   handle_subscribers(chat_id)
                elif text == "/backtest":      handle_backtest(chat_id)
        except Exception as e:
            print(f"  ⚠️  Polling error: {e}")
        time.sleep(POLL_INTERVAL)


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════
def fetch_historical(symbol, interval, days):
    now_ms   = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    all_candles = []
    end_ms = now_ms
    ms_per_bar = int(interval) * 60 * 1000

    print(f"📥 Fetching {days}d of {symbol} {interval}m candles from Bybit...")
    while end_ms > start_ms:
        params = {
            "category": "spot", "symbol": symbol, "interval": interval,
            "limit": BATCH_SIZE, "end": end_ms, "start": start_ms,
        }
        try:
            resp = requests.get(BYBIT_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                print(f"❌ Bybit error: {data.get('retMsg')}")
                break
            batch = data["result"]["list"]
            if not batch:
                break
            all_candles.extend(batch)
            oldest = int(batch[-1][0])
            end_ms = oldest - ms_per_bar
            print(f"  {len(all_candles):,} candles fetched...", end="\r")
            time.sleep(0.15)
        except Exception as e:
            print(f"\n  ⚠️  Fetch error: {e}")
            time.sleep(2)

    all_candles.reverse()
    print(f"\n✅ Fetched {len(all_candles):,} {interval}m candles.")
    return all_candles

def fetch_latest_candle(symbol, interval):
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": 3}
    resp = requests.get(BYBIT_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise Exception(f"Bybit error: {data.get('retMsg')}")
    return list(reversed(data["result"]["list"]))[-2]

def fetch_mtf_candles(symbol, interval, limit):
    """Fetch a small batch of higher-timeframe candles for MTF bias."""
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(BYBIT_URL, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if data.get("retCode") != 0:
            return []
        return list(reversed(data["result"]["list"]))
    except Exception:
        return []

def classify(candle):
    o, cl = float(candle[1]), float(candle[4])
    if cl > o:   return "green"
    elif cl < o: return "red"
    return "doji"


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS  (pure-Python, zero external dependencies)
# ══════════════════════════════════════════════════════════════════════════════
def compute_ema(prices: list, period: int):
    if len(prices) < period:
        return None
    k   = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema

def compute_ema_series(prices: list, period: int) -> list:
    if len(prices) < period:
        return [None] * len(prices)
    k   = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    out = [None] * (period - 1) + [ema]
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
        out.append(ema)
    return out

def compute_rsi(closes: list, period: int = 14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[-period - 1 + i] - closes[-period - 1 + i - 1]
        if diff >= 0: gains.append(diff); losses.append(0)
        else:         gains.append(0);    losses.append(abs(diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))

def compute_macd(closes: list, fast=12, slow=26, signal=9):
    """Returns (macd_val, signal_val, histogram) or (None, None, None)."""
    if len(closes) < slow + signal:
        return None, None, None
    fast_ema  = compute_ema_series(closes, fast)
    slow_ema  = compute_ema_series(closes, slow)
    macd_line = [
        (f - s) if f is not None and s is not None else None
        for f, s in zip(fast_ema, slow_ema)
    ]
    valid_macd = [v for v in macd_line if v is not None]
    if len(valid_macd) < signal:
        return None, None, None
    sig_series = compute_ema_series(valid_macd, signal)
    sig_val    = sig_series[-1]
    macd_val   = valid_macd[-1]
    hist       = (macd_val - sig_val) if sig_val is not None else None
    return macd_val, sig_val, hist

def compute_bollinger(closes: list, period=20, num_std=2.0):
    """Returns (upper, mid, lower, pct_b, bandwidth) or Nones."""
    if len(closes) < period:
        return None, None, None, None, None
    window = closes[-period:]
    mid    = sum(window) / period
    try:
        std = statistics.stdev(window)
    except Exception:
        return None, None, None, None, None
    upper = mid + num_std * std
    lower = mid - num_std * std
    price = closes[-1]
    pct_b = (price - lower) / (upper - lower) if upper != lower else 0.5
    bw    = (upper - lower) / mid if mid else 0
    return upper, mid, lower, round(pct_b, 4), round(bw, 6)

def compute_atr(candles_ohlcv: list, period=14):
    """candles_ohlcv: list of (o, h, l, c, v)."""
    if len(candles_ohlcv) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles_ohlcv)):
        h  = candles_ohlcv[i][1]
        l  = candles_ohlcv[i][2]
        pc = candles_ohlcv[i-1][3]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if not trs:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr

def compute_vwap(candles_ohlcv: list):
    """Full-session VWAP using typical price × volume."""
    cum_pv = cum_v = 0.0
    for o, h, l, c, v in candles_ohlcv:
        typical = (h + l + c) / 3
        cum_pv += typical * v
        cum_v  += v
    if cum_v == 0:
        return None
    return cum_pv / cum_v

def compute_stochastic(candles_ohlcv: list, period=14):
    """Returns (%K, %D) or (None, None)."""
    if len(candles_ohlcv) < period:
        return None, None
    window = candles_ohlcv[-period:]
    hi  = max(c[1] for c in window)
    lo  = min(c[2] for c in window)
    cl  = candles_ohlcv[-1][3]
    k   = 100 * (cl - lo) / (hi - lo) if hi != lo else 50.0
    k_vals = []
    for i in range(3):
        idx = len(candles_ohlcv) - period - i
        if idx < 0:
            break
        w2  = candles_ohlcv[idx: idx + period]
        hi2 = max(c[1] for c in w2)
        lo2 = min(c[2] for c in w2)
        cl2 = candles_ohlcv[idx + period - 1][3]
        k_vals.append(100 * (cl2 - lo2) / (hi2 - lo2) if hi2 != lo2 else 50.0)
    d = sum(k_vals) / len(k_vals) if k_vals else k
    return round(k, 2), round(d, 2)

def compute_obv_slope(candles_ohlcv: list, window: int = 20):
    """OBV slope: positive = accumulation, negative = distribution."""
    if len(candles_ohlcv) < window + 1:
        return None
    subset = candles_ohlcv[-window - 1:]
    obv    = 0.0
    series = []
    for i in range(1, len(subset)):
        if subset[i][3] > subset[i-1][3]:
            obv += subset[i][4]
        elif subset[i][3] < subset[i-1][3]:
            obv -= subset[i][4]
        series.append(obv)
    if len(series) < 2:
        return None
    return series[-1] - series[0]

def mtf_bias(candles) -> str:
    """Return 'bullish' / 'bearish' / 'neutral' from a HTF candle list."""
    if len(candles) < 20:
        return "neutral"
    closes = [float(c[4]) for c in candles]
    ema20  = compute_ema(closes, 20)
    rsi    = compute_rsi(closes, 14)
    if ema20 is None or rsi is None:
        return "neutral"
    price = closes[-1]
    if price > ema20 and rsi > 50:
        return "bullish"
    if price < ema20 and rsi < 50:
        return "bearish"
    return "neutral"


# ══════════════════════════════════════════════════════════════════════════════
# CANDLE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
class CandlePredictor:

    def __init__(self):
        # store (timestamp, direction, open, high, low, close, volume)
        self.window  = collections.deque(maxlen=WINDOW_SIZE)
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
        self._backtest_log       = []

        # MTF state — refreshed every few candles in live loop
        self.bias_15m = "neutral"
        self.bias_1h  = "neutral"

    # ── candle ingestion ──────────────────────────────────────────────────────
    def add_candle(self, candle):
        direction = classify(candle)
        o = float(candle[1])
        h = float(candle[2])
        l = float(candle[3])
        c = float(candle[4])
        v = float(candle[5]) if len(candle) > 5 else 0.0

        if self.window:
            prev_dir = self.window[-1][1]
            if prev_dir in self.markov and direction in ("green", "red"):
                self.markov[prev_dir][direction] += 1

        if len(self.window) == self.window.maxlen:
            od = self.window[0][1]
            if od == "green": self.total_green -= 1
            elif od == "red": self.total_red   -= 1
            else:             self.total_doji  -= 1

        self.window.append((int(candle[0]), direction, o, h, l, c, v))
        if direction == "green": self.total_green += 1
        elif direction == "red": self.total_red   += 1
        else:                    self.total_doji  += 1
        self.last_candle_time = int(candle[0])
        return direction

    # ── helper views ─────────────────────────────────────────────────────────
    def _ohlcv(self):
        return [(r[2], r[3], r[4], r[5], r[6]) for r in self.window]

    def _closes(self):
        return [r[5] for r in self.window]

    # ── prediction engine ─────────────────────────────────────────────────────
    def predict_next(self):
        if len(self.window) < MIN_CANDLES:
            return None, 0.0, {}, False

        signals      = {}
        green_score  = 0.0
        total_weight = 0.0
        ohlcv        = self._ohlcv()
        closes       = self._closes()

        # ── Signal ① Momentum (50-candle green%) ───────────────────────────
        recent  = list(self.window)[-MOMENTUM_WINDOW:]
        r_green = sum(1 for r in recent if r[1] == "green")
        r_red   = sum(1 for r in recent if r[1] == "red")
        r_total = r_green + r_red
        mom     = r_green / r_total if r_total > 0 else 0.5
        if r_total > 0:
            green_score  += mom * 1.5
            total_weight += 1.5
            signals["Momentum(50)"] = f"{'🟢' if mom > 0.5 else '🔴'} {mom*100:.1f}% green"

        # ── Signal ② Streak ────────────────────────────────────────────────
        sc       = [r[1] for r in list(self.window)[-STREAK_WINDOW:]]
        last_dir = sc[-1] if sc else "doji"
        streak_len = sum(1 for d in reversed(sc) if d == last_dir) \
                     if last_dir != "doji" else 0
        if last_dir != "doji" and streak_len >= 2:
            rw = min(streak_len / 6, 1.0)
            ss = (0.5 - rw * 0.35) if last_dir == "green" else (0.5 + rw * 0.35)
            green_score  += ss * 1.2
            total_weight += 1.2
            signals["Streak"] = (
                f"{'🟢' if last_dir=='green' else '🔴'} {streak_len}× {last_dir}"
                f" → {'reversal' if rw > 0.3 else 'continuation'}"
            )

        # ── Signal ③ Markov ────────────────────────────────────────────────
        if last_dir in self.markov:
            m  = self.markov[last_dir]
            mt = m["green"] + m["red"]
            if mt > 10:
                ms = m["green"] / mt
                green_score  += ms * 2.0
                total_weight += 2.0
                signals["Markov"] = (
                    f"After {last_dir}: 🟢{m['green']} / 🔴{m['red']}"
                    f" ({ms*100:.1f}% green)"
                )

        # ── Signal ④ EMA crossover (fast 9 / slow 20) ──────────────────────
        ema_fast = compute_ema(closes, EMA_FAST)
        ema_slow = compute_ema(closes, EMA_SLOW)
        if ema_fast is not None and ema_slow is not None:
            cross = ema_fast - ema_slow
            price = closes[-1]
            es    = 0.65 if cross > 0 else 0.35
            green_score  += es * 1.2
            total_weight += 1.2
            signals["EMA 9/20"] = (
                f"EMA9 {'above' if cross > 0 else 'below'} EMA20"
                f" | Δ{cross:+.2f}"
                f" | Price {'+' if price > ema_slow else ''}{((price - ema_slow) / ema_slow)*100:.3f}%"
            )

        # ── Signal ⑤ RSI ───────────────────────────────────────────────────
        rsi = compute_rsi(closes, RSI_PERIOD)
        if rsi is not None:
            rs  = 0.3 if rsi > 70 else (0.7 if rsi < 30 else 0.5)
            green_score  += rs * 1.3
            total_weight += 1.3
            lbl = "overbought🔴" if rsi > 70 else ("oversold🟢" if rsi < 30 else "neutral⚪")
            signals["RSI(14)"] = f"{rsi:.1f} ({lbl})"

        # ── Signal ⑥ MACD (12, 26, 9) ─────────────────────────────────────
        macd_v, sig_v, hist_v = compute_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        if hist_v is not None:
            ms2 = 0.65 if hist_v > 0 else 0.35
            green_score  += ms2 * 1.5
            total_weight += 1.5
            signals["MACD(12,26,9)"] = (
                f"{'🟢 Bullish' if hist_v > 0 else '🔴 Bearish'}"
                f" | hist={hist_v:+.2f}"
            )

        # ── Signal ⑦ Bollinger Bands ────────────────────────────────────────
        bb_up, bb_mid, bb_lo, pct_b, bw = compute_bollinger(closes, BB_PERIOD, BB_STD)
        if pct_b is not None:
            if bw < 0.005:                  # squeeze → follow momentum
                bb_s = 0.6 if mom > 0.5 else 0.4
                sig_label = f"🔵 Squeeze bw={bw*100:.3f}%"
            elif pct_b > 0.95:              # near upper band → overbought
                bb_s = 0.35
                sig_label = f"🔴 Near upper band %B={pct_b:.2f}"
            elif pct_b < 0.05:              # near lower band → oversold
                bb_s = 0.65
                sig_label = f"🟢 Near lower band %B={pct_b:.2f}"
            else:
                bb_s = 0.5 + (pct_b - 0.5) * 0.3
                sig_label = f"⚪ Mid-band %B={pct_b:.2f}"
            green_score  += bb_s * 1.3
            total_weight += 1.3
            signals["Bollinger(20)"] = sig_label

        # ── Signal ⑧ VWAP ──────────────────────────────────────────────────
        vwap = compute_vwap(ohlcv)
        if vwap is not None:
            price    = closes[-1]
            vwap_pct = (price - vwap) / vwap * 100
            vs = 0.65 if price > vwap else 0.35
            green_score  += vs * 1.4
            total_weight += 1.4
            signals["VWAP"] = (
                f"Price {'above' if price > vwap else 'below'} VWAP"
                f" ({vwap_pct:+.3f}%) = ${vwap:,.2f}"
            )

        # ── Signal ⑨ Stochastic ─────────────────────────────────────────────
        stoch_k, stoch_d = compute_stochastic(ohlcv, STOCH_PERIOD)
        if stoch_k is not None:
            if stoch_k < 20:
                st_s, st_lbl = 0.70, f"🟢 Oversold %K={stoch_k}"
            elif stoch_k > 80:
                st_s, st_lbl = 0.30, f"🔴 Overbought %K={stoch_k}"
            elif stoch_k > stoch_d:
                st_s, st_lbl = 0.62, f"🟢 Bullish cross %K={stoch_k} %D={stoch_d}"
            elif stoch_k < stoch_d:
                st_s, st_lbl = 0.38, f"🔴 Bearish cross %K={stoch_k} %D={stoch_d}"
            else:
                st_s, st_lbl = 0.50, f"⚪ Neutral %K={stoch_k}"
            green_score  += st_s * 1.0
            total_weight += 1.0
            signals["Stochastic"] = st_lbl

        # ── Signal ⑩ OBV trend ──────────────────────────────────────────────
        obv_slope = compute_obv_slope(ohlcv, OBV_WINDOW)
        if obv_slope is not None:
            obv_s = 0.65 if obv_slope > 0 else 0.35
            green_score  += obv_s * 1.0
            total_weight += 1.0
            signals["OBV(20)"] = (
                f"{'🟢 Accumulation' if obv_slope > 0 else '🔴 Distribution'}"
                f" slope={obv_slope:+.0f}"
            )

        # ── Signal ⑪ Candle body / wick (OHLCV feature) ─────────────────────
        last = self.window[-1]
        o_, h_, l_, c_, v_ = last[2], last[3], last[4], last[5], last[6]
        if o_ > 0:
            body     = abs(c_ - o_) / o_ * 100
            up_wick  = (h_ - max(o_, c_)) / o_ * 100
            dn_wick  = (min(o_, c_) - l_) / o_ * 100
            if body > 0.05:
                if up_wick > body * 1.5 and last[1] == "green":
                    wick_s   = 0.38
                    wick_lbl = f"🔴 Upper wick rejection ({up_wick:.3f}%)"
                elif dn_wick > body * 1.5 and last[1] == "red":
                    wick_s   = 0.62
                    wick_lbl = f"🟢 Lower wick rejection ({dn_wick:.3f}%)"
                else:
                    wick_s   = 0.55 if last[1] == "green" else 0.45
                    wick_lbl = f"⚪ Body {body:.3f}%"
                green_score  += wick_s * 0.8
                total_weight += 0.8
                signals["Candle Body"] = wick_lbl

        # ── Signal ⑫ Multi-timeframe bias ───────────────────────────────────
        if self.bias_15m != "neutral":
            mtf15_s = 0.65 if self.bias_15m == "bullish" else 0.35
            green_score  += mtf15_s * 1.2
            total_weight += 1.2
            signals["MTF 15m"] = (
                f"{'🟢' if self.bias_15m=='bullish' else '🔴'} {self.bias_15m}"
            )
        if self.bias_1h != "neutral":
            mtf1h_s = 0.65 if self.bias_1h == "bullish" else 0.35
            green_score  += mtf1h_s * 1.5     # higher weight for 1h
            total_weight += 1.5
            signals["MTF 1h"] = (
                f"{'🟢' if self.bias_1h=='bullish' else '🔴'} {self.bias_1h}"
            )

        if total_weight == 0:
            return None, 0.0, {}, False

        fs   = green_score / total_weight
        conf = round(abs(fs - 0.5) * 200, 1)
        pred = "green" if fs >= 0.5 else "red"

        # ── ATR flat-market gate ────────────────────────────────────────────
        atr     = compute_atr(ohlcv, ATR_PERIOD)
        is_flat = False
        if atr is not None and closes:
            atr_pct = atr / closes[-1] * 100
            is_flat = atr_pct < ATR_FLAT_PCT
            shared_state["atr_pct"] = round(atr_pct, 4)

        return pred, conf, signals, is_flat

    # ── outcome recording ─────────────────────────────────────────────────────
    def record_outcome(self, actual):
        if self.last_prediction and actual in ("green", "red"):
            self.predictions_made += 1
            correct = self.last_prediction == actual
            if correct:
                self.predictions_correct += 1
            self._backtest_log.append((
                self.last_prediction, actual,
                shared_state.get("confidence", 0.0)
            ))
            return correct
        return None

    # ── properties ────────────────────────────────────────────────────────────
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
    # BACKTESTING ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    def run_backtest(self, candles: list) -> dict:
        bt            = CandlePredictor()
        results       = []       # (correct: bool | None, confidence: float, skipped: bool)
        total_skipped = 0

        print(f"  🔬 Backtesting over {len(candles):,} candles with v2 signals...")
        for i, candle in enumerate(candles):
            if i == 0:
                bt.add_candle(candle)
                continue

            prediction, confidence, _, is_flat = bt.predict_next()
            actual = classify(candle)
            bt.add_candle(candle)

            if prediction is None or actual == "doji":
                continue

            if is_flat or confidence < MIN_CONFIDENCE:
                total_skipped += 1
                results.append((None, confidence, True))
                continue

            correct = prediction == actual
            results.append((correct, confidence, False))

        acted   = [(c, conf) for c, conf, sk in results if not sk and c is not None]
        total   = len(acted)
        correct = sum(1 for c, _ in acted if c)
        wrong   = total - correct

        by_confidence = {}
        for threshold in CONFIDENCE_BANDS:
            subset = [(c, conf) for c, conf in acted if conf >= threshold]
            st = len(subset)
            sc = sum(1 for c, _ in subset if c)
            by_confidence[threshold] = {
                "total":    st,
                "correct":  sc,
                "wrong":    st - sc,
                "accuracy": round(100 * sc / st, 2) if st else 0.0,
            }

        streak_details  = []
        longest_correct = 0
        longest_wrong   = 0
        cur_kind = None
        cur_len  = 0
        for correct_flag, _ in acted:
            kind = "correct" if correct_flag else "wrong"
            if kind == cur_kind:
                cur_len += 1
            else:
                if cur_kind is not None:
                    streak_details.append((cur_kind, cur_len))
                    if cur_kind == "correct":
                        longest_correct = max(longest_correct, cur_len)
                    else:
                        longest_wrong = max(longest_wrong, cur_len)
                cur_kind = kind
                cur_len  = 1
        if cur_kind:
            streak_details.append((cur_kind, cur_len))
            if cur_kind == "correct":
                longest_correct = max(longest_correct, cur_len)
            else:
                longest_wrong = max(longest_wrong, cur_len)

        return {
            "total":            total,
            "correct":          correct,
            "wrong":            wrong,
            "skipped":          total_skipped,
            "overall_accuracy": round(100 * correct / total, 2) if total else 0.0,
            "by_confidence":    by_confidence,
            "streak_correct":   longest_correct,
            "streak_wrong":     longest_wrong,
            "streak_details":   streak_details,
        }

    def generate_backtest_report(self) -> list:
        bt_data = self.run_backtest(list(self.window))
        if not bt_data:
            return ["⚠️ Not enough data for backtesting yet."]

        parts   = []
        acc     = bt_data["overall_accuracy"]
        acc_bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))

        # Part 1 — overview
        parts.append(
            f"📋 <b>BACKTESTING REPORT v2</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Overall (after gates)</b>\n"
            f"  Total acted : <b>{bt_data['total']:,}</b>\n"
            f"  ✅ Correct   : <b>{bt_data['correct']:,}</b>\n"
            f"  ❌ Wrong     : <b>{bt_data['wrong']:,}</b>\n"
            f"  ⏭ Skipped   : <b>{bt_data['skipped']:,}</b> (flat/low-conf)\n"
            f"  🎯 Accuracy  : <b>{acc}%</b>\n"
            f"  [{acc_bar}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔥 Longest correct streak : <b>{bt_data['streak_correct']}</b>\n"
            f"💀 Longest wrong streak   : <b>{bt_data['streak_wrong']}</b>"
        )

        # Part 2 — confidence band table
        band_lines = [
            "📡 <b>Accuracy by Confidence Threshold</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<code>Conf≥  Total   Correct  Wrong   Acc%</code>\n"
            "<code>────────────────────────────────────</code>"
        ]
        for threshold, d in bt_data["by_confidence"].items():
            if d["total"] == 0:
                band_lines.append(
                    f"<code>{threshold:>3}%   {'—':>6}  {'—':>7}  {'—':>5}  {'—':>5}</code>"
                )
            else:
                bar = "▓" * int(d["accuracy"] / 10) + "░" * (10 - int(d["accuracy"] / 10))
                band_lines.append(
                    f"<code>{threshold:>3}%  {d['total']:>6}  "
                    f"{d['correct']:>7}  {d['wrong']:>5}  "
                    f"{d['accuracy']:>5.1f}%</code> {bar}"
                )
        parts.append("\n".join(band_lines))

        # Part 3 — streak histograms
        correct_streaks = sorted(
            [l for k, l in bt_data["streak_details"] if k == "correct"], reverse=True
        )[:10]
        wrong_streaks = sorted(
            [l for k, l in bt_data["streak_details"] if k == "wrong"], reverse=True
        )[:10]

        def fmt_bars(streaks):
            if not streaks:
                return "  (none)"
            return "\n".join(f"  {'█' * min(s, 20):<20} {s}" for s in streaks)

        parts.append(
            f"📈 <b>Top Correct Streaks</b>\n"
            f"<code>{fmt_bars(correct_streaks)}</code>\n\n"
            f"📉 <b>Top Wrong Streaks</b>\n"
            f"<code>{fmt_bars(wrong_streaks)}</code>"
        )
        return parts

    def print_backtest_report_console(self, bt_data: dict):
        if not bt_data:
            print("  ⚠️  No backtest data.")
            return
        sep = "═" * 66
        print(f"\n{sep}")
        print(f"  📋  BACKTESTING REPORT v2  (12 signals + gates)")
        print(sep)
        print(f"  Acted predictions  : {bt_data['total']:,}")
        print(f"  ✅ Correct          : {bt_data['correct']:,}")
        print(f"  ❌ Wrong            : {bt_data['wrong']:,}")
        print(f"  ⏭  Skipped          : {bt_data['skipped']:,}  (ATR flat / low-confidence)")
        acc     = bt_data["overall_accuracy"]
        acc_bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  🎯 Accuracy         : {acc}%  [{acc_bar}]")
        print(f"  🔥 Longest correct  : {bt_data['streak_correct']}")
        print(f"  💀 Longest wrong    : {bt_data['streak_wrong']}")
        print(f"\n  {'Confidence≥':12} {'Total':>7} {'Correct':>8} {'Wrong':>6} {'Accuracy':>9}")
        print(f"  {'-'*48}")
        for threshold, d in bt_data["by_confidence"].items():
            if d["total"] == 0:
                print(f"  {f'{threshold}%':12} {'—':>7} {'—':>8} {'—':>6} {'—':>9}")
            else:
                print(
                    f"  {f'{threshold}%':12} {d['total']:>7,} "
                    f"{d['correct']:>8,} {d['wrong']:>6,} "
                    f"{d['accuracy']:>8.2f}%"
                )
        print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# BROADCAST MESSAGE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_broadcast(predictor, candle, prediction, confidence,
                    signals, actual_dir, outcome, is_flat=False):
    pred_emoji  = "🟢" if prediction == "green" else "🔴"
    dir_emoji   = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")
    outcome_str = ("✅ Correct!" if outcome else "❌ Wrong") if outcome is not None else ""
    acc_bar     = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    signal_lines = "\n".join(f"  • <b>{n}</b>: {v}" for n, v in signals.items())
    flat_note   = "\n  ⚠️ <i>Low volatility — confidence reduced</i>" if is_flat else ""
    conf_fire   = "🔥" if confidence > 20 else ("⚡" if confidence > 10 else "〰️")

    return (
        f"🤖 <b>BTC/USD 5m Update  v2</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {ts(candle[0])}\n"
        f"💵 Price : <b>${float(candle[4]):,.2f}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Rolling Window</b>\n"
        f"  🟢 Green : {predictor.total_green:,} ({predictor.green_pct}%)\n"
        f"  🔴 Red   : {predictor.total_red:,} ({predictor.red_pct}%)\n"
        f"  ⚪ Doji  : {predictor.total_doji:,}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕯 <b>Last Candle:</b> {dir_emoji} {actual_dir.upper()}  {outcome_str}\n"
        f"  📐 15m: {predictor.bias_15m}  |  1h: {predictor.bias_1h}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 <b>NEXT CANDLE PREDICTION</b>\n"
        f"  {pred_emoji} <b>{prediction.upper()}</b>"
        f"  |  Confidence: <b>{confidence:.1f}%</b> {conf_fire}"
        f"{flat_note}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 <b>Signals ({len(signals)}):</b>\n{signal_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Accuracy: <b>{predictor.accuracy}%</b>  [{acc_bar}]\n"
        f"   {predictor.predictions_correct}/{predictor.predictions_made} correct"
        f"  |  {predictor.predictions_skipped} skipped"
        f"  |  👥 {subscriber_count()}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>/start to subscribe • /stop to unsubscribe</i>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def print_dashboard(predictor, candle, prediction, confidence,
                    signals, actual_dir, is_flat):
    sep        = "=" * 64
    g, r       = predictor.total_green, predictor.total_red
    bt         = g + r
    g_bar      = int(40 * g / bt) if bt else 0
    acc_bar    = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    pred_emoji = "🟢" if prediction == "green" else ("🔴" if prediction else "⏳")
    dir_emoji  = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")

    print(f"\n{sep}")
    print(f"  🤖 BTC Predictor v2  |  {ts(candle[0])}  |  ${float(candle[4]):,.2f}")
    print(sep)
    print(f"  Window : {predictor.candle_count:,}"
          f"  |  🟢{g:,}({predictor.green_pct}%)"
          f"  🔴{r:,}({predictor.red_pct}%)")
    print(f"  [{'█'*g_bar}{'▓'*(40-g_bar)}]")
    print(f"  MTF    : 15m={predictor.bias_15m}  |  1h={predictor.bias_1h}")
    flat_tag = "  ⚠️ FLAT MARKET (ATR gate)" if is_flat else ""
    print(f"  Last   : {dir_emoji} {actual_dir.upper()}{flat_tag}")
    if prediction:
        skip_tag = " [SKIPPED]" if is_flat or confidence < MIN_CONFIDENCE else ""
        print(f"  Next   : {pred_emoji} {prediction.upper()}"
              f"  ({confidence:.1f}% conf){skip_tag}")
        for n, v in signals.items():
            print(f"    {n:<18}: {v}")
        print(f"  Acc    : {predictor.accuracy}%  [{acc_bar}]"
              f"  ({predictor.predictions_correct}/{predictor.predictions_made})"
              f"  skipped={predictor.predictions_skipped}")
        print(f"  Subs   : {subscriber_count()}")
    else:
        print(f"  ⏳ Warming up... "
              f"{MIN_CANDLES - predictor.candle_count:,} more candles needed")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _predictor_ref

    print("=" * 64)
    print("  BTC/USD 5m Predictor — PUBLIC BOT  v2.0")
    print("  Signals: VWAP · MACD · BB · ATR · MTF · OBV · Stoch · Body")
    print("=" * 64)

    init_db()
    add_subscriber(ADMIN_CHAT_ID, "admin")
    print(f"✅ DB ready. Subscribers: {subscriber_count()}")

    # ── Load 5m history ───────────────────────────────────────────────────────
    historical     = fetch_historical(SYMBOL, INTERVAL, DAYS)
    predictor      = CandlePredictor()
    _predictor_ref = predictor

    print("⚙️  Building rolling window...")
    for candle in historical:
        predictor.add_candle(candle)
    print(f"✅ Window: {predictor.candle_count:,} candles"
          f" | 🟢{predictor.total_green:,} | 🔴{predictor.total_red:,}\n")

    # ── Initial MTF fetch ─────────────────────────────────────────────────────
    print("📐 Fetching multi-timeframe candles (15m & 1h)...")
    candles_15m        = fetch_mtf_candles(SYMBOL, INTERVAL_15, MTF_CANDLES)
    candles_1h         = fetch_mtf_candles(SYMBOL, INTERVAL_60, MTF_CANDLES)
    predictor.bias_15m = mtf_bias(candles_15m)
    predictor.bias_1h  = mtf_bias(candles_1h)
    print(f"  15m bias: {predictor.bias_15m}  |  1h bias: {predictor.bias_1h}")

    # ── Initial backtest ──────────────────────────────────────────────────────
    print("\n🔬 Running initial backtest on historical data...")
    bt_data = predictor.run_backtest(historical)
    predictor.print_backtest_report_console(bt_data)

    if bt_data:
        acc     = bt_data["overall_accuracy"]
        acc_bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        send_message(ADMIN_CHAT_ID,
            f"📋 <b>Startup Backtest — v2 (12 signals)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  Acted   : {bt_data['total']:,}  |  Skipped: {bt_data['skipped']:,}\n"
            f"  ✅ Correct : {bt_data['correct']:,}\n"
            f"  ❌ Wrong   : {bt_data['wrong']:,}\n"
            f"  🎯 Accuracy: <b>{acc}%</b>  [{acc_bar}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  🔥 Longest correct : {bt_data['streak_correct']}\n"
            f"  💀 Longest wrong   : {bt_data['streak_wrong']}\n"
            f"  <i>Send /backtest for full confidence-band table</i>"
        )

    # ── First live prediction ─────────────────────────────────────────────────
    prediction, confidence, signals, is_flat = predictor.predict_next()
    predictor.last_prediction = prediction

    shared_state.update({
        "prediction":  prediction,   "confidence":  confidence,
        "accuracy":    predictor.accuracy,
        "correct":     predictor.predictions_correct,
        "total":       predictor.predictions_made,
        "skipped":     predictor.predictions_skipped,
        "green":       predictor.total_green,
        "red":         predictor.total_red,
        "green_pct":   predictor.green_pct,
        "red_pct":     predictor.red_pct,
        "price":       float(historical[-1][4]) if historical else 0,
        "mtf_15_bias": predictor.bias_15m,
        "mtf_1h_bias": predictor.bias_1h,
    })

    send_message(ADMIN_CHAT_ID,
        f"🚀 <b>BTC Predictor v2 is LIVE!</b>\n\n"
        f"📊 {predictor.candle_count:,} candles loaded\n"
        f"🟢 Green: {predictor.total_green:,} ({predictor.green_pct}%)"
        f"  🔴 Red: {predictor.total_red:,} ({predictor.red_pct}%)\n"
        f"📐 15m: {predictor.bias_15m}  |  1h: {predictor.bias_1h}\n"
        f"👥 Subscribers: {subscriber_count()}\n\n"
        f"🔮 First prediction: "
        f"{'🟢 GREEN' if prediction == 'green' else '🔴 RED'}"
        f" ({confidence:.1f}%)"
        + (" ⚠️ flat market" if is_flat else "") + "\n\n"
        f"<i>Signals: VWAP · MACD · BB · ATR gate · MTF · OBV"
        f" · Stochastic · Candle body · EMA 9/20\n"
        f"Send /backtest for confidence-band report.</i>"
    )

    # ── Start polling thread ──────────────────────────────────────────────────
    t = threading.Thread(target=polling_thread, daemon=True)
    t.start()

    print("\n🔄 Entering live loop...\n")
    last_seen_time      = int(historical[-1][0]) if historical else 0
    mtf_refresh_counter = 0   # refresh MTF every 3 candles (~15 min)

    while True:
        try:
            latest      = fetch_latest_candle(SYMBOL, INTERVAL)
            candle_time = int(latest[0])

            if candle_time != last_seen_time:
                actual_dir = classify(latest)
                outcome    = predictor.record_outcome(actual_dir)
                predictor.add_candle(latest)

                # Refresh MTF every 3 closed candles
                mtf_refresh_counter += 1
                if mtf_refresh_counter >= 3:
                    try:
                        c15 = fetch_mtf_candles(SYMBOL, INTERVAL_15, MTF_CANDLES)
                        c1h = fetch_mtf_candles(SYMBOL, INTERVAL_60, MTF_CANDLES)
                        predictor.bias_15m         = mtf_bias(c15)
                        predictor.bias_1h           = mtf_bias(c1h)
                        shared_state["mtf_15_bias"] = predictor.bias_15m
                        shared_state["mtf_1h_bias"] = predictor.bias_1h
                    except Exception:
                        pass
                    mtf_refresh_counter = 0

                prediction, confidence, signals, is_flat = predictor.predict_next()
                gate_triggered = is_flat or (confidence < MIN_CONFIDENCE)

                if gate_triggered:
                    predictor.predictions_skipped += 1
                else:
                    predictor.last_prediction = prediction

                shared_state.update({
                    "dir":         actual_dir,
                    "prediction":  prediction,
                    "confidence":  confidence,
                    "accuracy":    predictor.accuracy,
                    "correct":     predictor.predictions_correct,
                    "total":       predictor.predictions_made,
                    "skipped":     predictor.predictions_skipped,
                    "green":       predictor.total_green,
                    "red":         predictor.total_red,
                    "green_pct":   predictor.green_pct,
                    "red_pct":     predictor.red_pct,
                    "price":       float(latest[4]),
                    "mtf_15_bias": predictor.bias_15m,
                    "mtf_1h_bias": predictor.bias_1h,
                })

                print_dashboard(predictor, latest, prediction, confidence,
                                signals, actual_dir, is_flat)

                if prediction and not gate_triggered:
                    msg = build_broadcast(
                        predictor, latest, prediction, confidence,
                        signals, actual_dir, outcome, is_flat
                    )
                    broadcast(msg)
                elif gate_triggered:
                    reason = "flat market" if is_flat else f"low conf ({confidence:.1f}%)"
                    print(f"  ⏭  Broadcast skipped — {reason}")

                last_seen_time = candle_time

            else:
                fo = float(latest[1]); fc = float(latest[4])
                fp = ((fc - fo) / fo * 100) if fo else 0
                gate_note = " [FLAT]" \
                    if shared_state.get("atr_pct", 1.0) < ATR_FLAT_PCT else ""
                print(
                    f"  [{now_str()}] Forming... "
                    f"{'🟢' if fc > fo else '🔴'} {fp:+.3f}%  |  "
                    f"Next: {'🟢' if prediction=='green' else '🔴' if prediction else '⏳'}"
                    f" {prediction or 'warming up'} ({confidence:.1f}%){gate_note}",
                    end="\r"
                )

        except requests.exceptions.RequestException as e:
            print(f"\n  ⚠️  Network error: {e} — retrying...")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()
