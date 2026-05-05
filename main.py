"""
BTC/USD 5m Rolling Candle Predictor — PUBLIC BOT
─────────────────────────────────────────────────────────────
- Anyone can subscribe via /start
- Broadcasts prediction to ALL subscribers every new 5m candle
- SQLite DB stores all subscribers
- Commands: /start, /stop, /status, /subscribers (admin only), /backtest (admin only)
- 5 ML signals: Momentum, Streak, Markov, EMA, RSI
- Tracks prediction accuracy live
- Full backtesting report with confidence-band breakdown & streak analysis
"""

import requests
import time
import os
import sqlite3
import threading
import collections
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",  "8766778348:AAEpkHO55y_oCrJ0vrTwtXsm8cWE_4IOZxA")
ADMIN_CHAT_ID    = os.environ.get("ADMIN_CHAT_ID",   "5792224870")

SYMBOL           = "BTCUSDT"
INTERVAL         = "5"
BYBIT_URL        = "https://api.bybit.com/v5/market/kline"
DAYS             = 365
BATCH_SIZE       = 1000
MS_PER_5MIN      = 5 * 60 * 1000

WINDOW_SIZE      = 365 * 24 * 12   # ~105,120 candles
MIN_CANDLES      = 1000
MOMENTUM_WINDOW  = 50
STREAK_WINDOW    = 10
EMA_PERIOD       = 20
RSI_PERIOD       = 14

POLL_INTERVAL    = 2    # seconds between Telegram polling
DB_FILE          = "subscribers.db"

# Confidence bands to analyse in backtesting (%)
CONFIDENCE_BANDS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# ─────────────────────────────────────────────────────────────────────────────


def ts(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def now_str():
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


# ── DATABASE ──────────────────────────────────────────────────────────────────
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


# ── TELEGRAM ──────────────────────────────────────────────────────────────────
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


# ── COMMAND HANDLERS ──────────────────────────────────────────────────────────
# Shared state updated by monitor thread
shared_state = {
    "streak":     0,
    "dir":        "unknown",
    "prediction": None,
    "confidence": 0,
    "accuracy":   0.0,
    "correct":    0,
    "total":      0,
    "green":      0,
    "red":        0,
    "green_pct":  0,
    "red_pct":    0,
    "price":      0,
}

# Global predictor reference (set in main)
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
            f"🎉 <b>Welcome! You're now subscribed!</b>\n\n"
            f"📌 <b>What you'll get every 5 minutes:</b>\n"
            f"  • Last candle result (🟢/🔴)\n"
            f"  • Next candle prediction\n"
            f"  • Live accuracy score\n"
            f"  • 5 ML signal breakdown\n\n"
            f"⚡ <b>Commands:</b>\n"
            f"  /start  — Subscribe\n"
            f"  /stop   — Unsubscribe\n"
            f"  /status — Current stats\n\n"
            f"<i>First update arrives at the next 5m candle close! 🚀</i>"
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
    send_message(chat_id,
        f"📊 <b>BTC/USD 5m Predictor — Live Status</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 BTC Price : ${s['price']:,.2f}\n"
        f"🟢 Green     : {s['green']:,} ({s['green_pct']}%)\n"
        f"🔴 Red       : {s['red']:,} ({s['red_pct']}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 Next prediction : {pred_emoji} <b>{(pred or 'warming up').upper()}</b>\n"
        f"   Confidence      : <b>{s['confidence']:.1f}%</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Accuracy  : <b>{s['accuracy']}%</b>\n"
        f"   [{acc_bar}]\n"
        f"   {s['correct']}/{s['total']} correct\n"
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
    """Admin-only: trigger backtesting report and send results."""
    if str(chat_id) != str(ADMIN_CHAT_ID):
        send_message(chat_id, "⛔ Admin only command.")
        return
    if _predictor_ref is None:
        send_message(chat_id, "⏳ Bot is still warming up. Try again shortly.")
        return
    send_message(chat_id, "🔬 <b>Running backtest...</b> This may take a moment.")
    try:
        report = _predictor_ref.generate_backtest_report()
        for part in report:
            send_message(chat_id, part)
    except Exception as e:
        send_message(chat_id, f"❌ Backtest error: {e}")
        print(f"  ❌ Backtest error: {e}")


# ── TELEGRAM POLLING THREAD ───────────────────────────────────────────────────
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

                if text == "/start":         handle_start(chat_id, username)
                elif text == "/stop":        handle_stop(chat_id, username)
                elif text == "/status":      handle_status(chat_id)
                elif text == "/subscribers": handle_subscribers(chat_id)
                elif text == "/backtest":    handle_backtest(chat_id)

        except Exception as e:
            print(f"  ⚠️  Polling error: {e}")

        time.sleep(POLL_INTERVAL)


# ── FETCH ─────────────────────────────────────────────────────────────────────
def fetch_historical(symbol, interval, days):
    now_ms   = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    all_candles = []
    end_ms = now_ms

    print(f"📥 Fetching {days} days of {symbol} {interval}m candles from Bybit...")
    while end_ms > start_ms:
        params = {
            "category": "spot",
            "symbol":   symbol,
            "interval": interval,
            "limit":    BATCH_SIZE,
            "end":      end_ms,
            "start":    start_ms,
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
            end_ms = oldest - MS_PER_5MIN
            print(f"  {len(all_candles):,} candles fetched...", end="\r")
            time.sleep(0.15)
        except Exception as e:
            print(f"\n  ⚠️  Fetch error: {e}")
            time.sleep(2)

    all_candles.reverse()
    print(f"\n✅ Fetched {len(all_candles):,} historical candles.")
    return all_candles

def fetch_latest_candle(symbol, interval):
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": 3}
    resp = requests.get(BYBIT_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise Exception(f"Bybit error: {data.get('retMsg')}")
    return list(reversed(data["result"]["list"]))[-2]

def classify(candle):
    o, cl = float(candle[1]), float(candle[4])
    if cl > o:   return "green"
    elif cl < o: return "red"
    return "doji"


# ── INDICATORS ────────────────────────────────────────────────────────────────
def compute_ema(prices, period):
    if len(prices) < period:
        return None
    k   = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema

def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[-period - 1 + i] - closes[-period - 1 + i - 1]
        if diff >= 0: gains.append(diff); losses.append(0)
        else:         gains.append(0);    losses.append(abs(diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0: return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


# ── PREDICTOR ─────────────────────────────────────────────────────────────────
class CandlePredictor:
    def __init__(self):
        self.window      = collections.deque(maxlen=WINDOW_SIZE)
        self.closes      = collections.deque(maxlen=WINDOW_SIZE)
        self.total_green = self.total_red = self.total_doji = 0
        self.markov = {
            "green": {"green": 0, "red": 0},
            "red":   {"green": 0, "red": 0},
            "doji":  {"green": 0, "red": 0},
        }
        self.predictions_made = self.predictions_correct = 0
        self.last_prediction  = None
        self.last_candle_time = None

        # ── Backtest tracking ────────────────────────────────────────────────
        # Each entry: (prediction, actual_direction, confidence)
        self._backtest_log: list[tuple[str, str, float]] = []

    def add_candle(self, candle):
        direction   = classify(candle)
        close_price = float(candle[4])
        if self.window:
            prev_dir = self.window[-1][1]
            if prev_dir in self.markov and direction in ("green", "red"):
                self.markov[prev_dir][direction] += 1
        if len(self.window) == self.window.maxlen:
            od = self.window[0][1]
            if od == "green": self.total_green -= 1
            elif od == "red": self.total_red   -= 1
            else:             self.total_doji  -= 1
        self.window.append((int(candle[0]), direction, close_price))
        self.closes.append(close_price)
        if direction == "green": self.total_green += 1
        elif direction == "red": self.total_red   += 1
        else:                    self.total_doji  += 1
        self.last_candle_time = int(candle[0])
        return direction

    def predict_next(self):
        if len(self.window) < MIN_CANDLES:
            return None, 0, {}
        signals = {}
        green_score = total_weight = 0.0

        # Momentum
        recent  = list(self.window)[-MOMENTUM_WINDOW:]
        r_green = sum(1 for _, d, _ in recent if d == "green")
        r_red   = sum(1 for _, d, _ in recent if d == "red")
        r_total = r_green + r_red
        if r_total > 0:
            mom = r_green / r_total
            green_score += mom * 1.5; total_weight += 1.5
            signals["Momentum(50)"] = f"{'🟢' if mom > 0.5 else '🔴'} {mom*100:.1f}% green"

        # Streak
        sc = [d for _, d, _ in list(self.window)[-STREAK_WINDOW:]]
        last_dir = sc[-1] if sc else "doji"
        streak_len = sum(1 for d in reversed(sc) if d == last_dir) if last_dir != "doji" else 0
        if last_dir != "doji" and streak_len >= 2:
            rw = min(streak_len / 6, 1.0)
            ss = (0.5 - rw * 0.35) if last_dir == "green" else (0.5 + rw * 0.35)
            green_score += ss * 1.2; total_weight += 1.2
            signals["Streak"] = f"{'🟢' if last_dir=='green' else '🔴'} {streak_len}x {last_dir} → {'reversal bias' if rw > 0.3 else 'continuation'}"

        # Markov
        if last_dir in self.markov:
            m = self.markov[last_dir]
            mt = m["green"] + m["red"]
            if mt > 10:
                ms = m["green"] / mt
                green_score += ms * 2.0; total_weight += 2.0
                signals["Markov"] = f"After {last_dir}: 🟢{m['green']} / 🔴{m['red']} ({ms*100:.1f}% green)"

        # EMA
        cl = list(self.closes)
        ema = compute_ema(cl, EMA_PERIOD)
        if ema and cl:
            cp = cl[-1]
            es = 0.6 if cp > ema else 0.4
            green_score += es * 1.0; total_weight += 1.0
            signals["EMA(20)"] = f"Price {'above' if cp > ema else 'below'} EMA ({((cp-ema)/ema)*100:+.3f}%)"

        # RSI
        rsi = compute_rsi(cl, RSI_PERIOD)
        if rsi is not None:
            rs = 0.3 if rsi > 70 else (0.7 if rsi < 30 else 0.5)
            green_score += rs * 1.3; total_weight += 1.3
            signals["RSI(14)"] = f"{rsi:.1f} ({'overbought🔴' if rsi > 70 else 'oversold🟢' if rsi < 30 else 'neutral⚪'})"

        if total_weight == 0:
            return None, 0, {}
        fs = green_score / total_weight
        return ("green" if fs >= 0.5 else "red"), round(abs(fs - 0.5) * 200, 1), signals

    def record_outcome(self, actual):
        if self.last_prediction and actual in ("green", "red"):
            self.predictions_made += 1
            correct = self.last_prediction == actual
            if correct: self.predictions_correct += 1
            # Log for backtesting
            self._backtest_log.append((
                self.last_prediction,
                actual,
                shared_state.get("confidence", 0.0)
            ))
            return correct
        return None

    @property
    def accuracy(self):
        return round(100 * self.predictions_correct / self.predictions_made, 2) if self.predictions_made else 0.0
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

    # ── BACKTESTING ENGINE ───────────────────────────────────────────────────
    def run_backtest(self, candles: list) -> dict:
        """
        Simulate predictions over a list of candles (not yet seen by the live
        predictor) and return a detailed results dictionary.

        Parameters
        ----------
        candles : list
            Raw Bybit kline candles in chronological order.

        Returns
        -------
        dict with keys:
            total, correct, wrong, overall_accuracy,
            by_confidence  → dict[threshold → {total, correct, wrong, accuracy}]
            streak_correct → longest correct streak
            streak_wrong   → longest wrong streak
            streak_details → list of streak records
        """
        bt = CandlePredictor()          # fresh isolated predictor
        results = []                    # list of (correct: bool, confidence: float)

        print(f"  🔬 Backtesting over {len(candles):,} candles...")
        for i, candle in enumerate(candles):
            if i == 0:
                bt.add_candle(candle)
                continue

            # Predict before adding the new candle
            prediction, confidence, _ = bt.predict_next()
            actual = classify(candle)
            bt.add_candle(candle)

            if prediction is None or actual == "doji":
                continue

            correct = prediction == actual
            results.append((correct, confidence))

        if not results:
            return {}

        total   = len(results)
        correct = sum(1 for c, _ in results if c)
        wrong   = total - correct

        # ── Confidence-band breakdown ────────────────────────────────────────
        by_confidence = {}
        for threshold in CONFIDENCE_BANDS:
            subset = [(c, conf) for c, conf in results if conf >= threshold]
            st = len(subset)
            sc = sum(1 for c, _ in subset if c)
            by_confidence[threshold] = {
                "total":    st,
                "correct":  sc,
                "wrong":    st - sc,
                "accuracy": round(100 * sc / st, 2) if st else 0.0,
            }

        # ── Streak analysis ──────────────────────────────────────────────────
        streak_details = []          # (kind, length)
        longest_correct = 0
        longest_wrong   = 0

        cur_kind   = None
        cur_length = 0
        for correct_flag, _ in results:
            kind = "correct" if correct_flag else "wrong"
            if kind == cur_kind:
                cur_length += 1
            else:
                if cur_kind is not None:
                    streak_details.append((cur_kind, cur_length))
                    if cur_kind == "correct":
                        longest_correct = max(longest_correct, cur_length)
                    else:
                        longest_wrong = max(longest_wrong, cur_length)
                cur_kind   = kind
                cur_length = 1
        # flush last streak
        if cur_kind is not None:
            streak_details.append((cur_kind, cur_length))
            if cur_kind == "correct":
                longest_correct = max(longest_correct, cur_length)
            else:
                longest_wrong = max(longest_wrong, cur_length)

        return {
            "total":           total,
            "correct":         correct,
            "wrong":           wrong,
            "overall_accuracy": round(100 * correct / total, 2),
            "by_confidence":   by_confidence,
            "streak_correct":  longest_correct,
            "streak_wrong":    longest_wrong,
            "streak_details":  streak_details,
        }

    def generate_backtest_report(self, candles: list | None = None) -> list[str]:
        """
        Run backtest on the candles currently stored in the rolling window
        (or a provided list) and return a list of Telegram-ready message
        strings (split to stay under the 4096-char limit).
        """
        if candles is None:
            candles = [
                # reconstruct minimal candle tuples from window: (time, ?, close)
                # We need open as well — store a dummy equal to close (no signal distortion)
                (t, c, c, c, c, "0")
                for t, _, c in self.window
            ]

        bt_data = self.run_backtest(list(self.window))   # pass deque directly

        if not bt_data:
            return ["⚠️ Not enough data for backtesting yet."]

        parts = []

        # ── Part 1 : Overview ────────────────────────────────────────────────
        acc = bt_data["overall_accuracy"]
        acc_bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        overview = (
            f"📋 <b>BACKTESTING REPORT</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Overall Performance</b>\n"
            f"  Total predictions : <b>{bt_data['total']:,}</b>\n"
            f"  ✅ Correct         : <b>{bt_data['correct']:,}</b>\n"
            f"  ❌ Wrong           : <b>{bt_data['wrong']:,}</b>\n"
            f"  🎯 Accuracy        : <b>{acc}%</b>\n"
            f"  [{acc_bar}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔥 <b>Streak Records</b>\n"
            f"  ✅ Longest correct : <b>{bt_data['streak_correct']}</b> in a row\n"
            f"  ❌ Longest wrong   : <b>{bt_data['streak_wrong']}</b> in a row\n"
        )
        parts.append(overview)

        # ── Part 2 : Confidence-band table ───────────────────────────────────
        band_lines = [
            "📡 <b>Accuracy by Confidence Threshold</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<code>Conf≥  Total  Correct  Wrong   Acc%</code>\n"
            "<code>─────────────────────────────────</code>"
        ]
        for threshold, d in bt_data["by_confidence"].items():
            if d["total"] == 0:
                row = f"<code>{threshold:>3}%   {'—':>5}  {'—':>7}  {'—':>5}  {'—':>6}</code>"
            else:
                bar_len = int(d["accuracy"] / 10)
                bar = "▓" * bar_len + "░" * (10 - bar_len)
                row = (
                    f"<code>{threshold:>3}%  {d['total']:>6}  "
                    f"{d['correct']:>7}  {d['wrong']:>5}  "
                    f"{d['accuracy']:>5.1f}%</code> {bar}"
                )
            band_lines.append(row)

        parts.append("\n".join(band_lines))

        # ── Part 3 : Streak histogram (top 10 longest of each kind) ──────────
        correct_streaks = sorted(
            [l for k, l in bt_data["streak_details"] if k == "correct"],
            reverse=True
        )[:10]
        wrong_streaks = sorted(
            [l for k, l in bt_data["streak_details"] if k == "wrong"],
            reverse=True
        )[:10]

        def fmt_bars(streaks):
            if not streaks:
                return "  (none)"
            return "\n".join(
                f"  {'█' * min(s, 20):<20} {s}"
                for s in streaks
            )

        streak_report = (
            f"📈 <b>Top Correct Streaks</b>\n"
            f"<code>{fmt_bars(correct_streaks)}</code>\n\n"
            f"📉 <b>Top Wrong Streaks</b>\n"
            f"<code>{fmt_bars(wrong_streaks)}</code>"
        )
        parts.append(streak_report)

        return parts

    def print_backtest_report_console(self, bt_data: dict):
        """Pretty-print backtest results to the console."""
        if not bt_data:
            print("  ⚠️  No backtest data.")
            return
        sep = "═" * 62
        print(f"\n{sep}")
        print(f"  📋  BACKTESTING REPORT")
        print(sep)
        print(f"  Total Predictions : {bt_data['total']:,}")
        print(f"  ✅ Correct         : {bt_data['correct']:,}")
        print(f"  ❌ Wrong           : {bt_data['wrong']:,}")
        acc = bt_data["overall_accuracy"]
        acc_bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  🎯 Accuracy        : {acc}%  [{acc_bar}]")
        print(f"  🔥 Longest correct streak : {bt_data['streak_correct']}")
        print(f"  💀 Longest wrong streak   : {bt_data['streak_wrong']}")
        print(f"\n  {'Confidence≥':12} {'Total':>7} {'Correct':>8} {'Wrong':>6} {'Accuracy':>9}")
        print(f"  {'-'*46}")
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


# ── BROADCAST MESSAGE BUILDER ─────────────────────────────────────────────────
def build_broadcast(predictor, candle, prediction, confidence, signals, actual_dir, outcome):
    pred_emoji = "🟢" if prediction == "green" else "🔴"
    dir_emoji  = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")
    outcome_str = ("✅ Correct!" if outcome else "❌ Wrong") if outcome is not None else ""
    acc_bar = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    signal_lines = "\n".join(f"  • <b>{n}</b>: {v}" for n, v in signals.items())

    return (
        f"🤖 <b>BTC/USD 5m Update</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {ts(candle[0])}\n"
        f"💵 Price : <b>${float(candle[4]):,.2f}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Rolling 365-Day Window</b>\n"
        f"  🟢 Green : {predictor.total_green:,} ({predictor.green_pct}%)\n"
        f"  🔴 Red   : {predictor.total_red:,} ({predictor.red_pct}%)\n"
        f"  ⚪ Doji  : {predictor.total_doji:,}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕯 <b>Last Candle :</b> {dir_emoji} {actual_dir.upper()}  {outcome_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 <b>NEXT CANDLE PREDICTION</b>\n"
        f"  {pred_emoji} <b>{prediction.upper()}</b>  |  Confidence: <b>{confidence:.1f}%</b> {'🔥' if confidence > 15 else '〰️'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 <b>Signals:</b>\n{signal_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Accuracy: <b>{predictor.accuracy}%</b>  [{acc_bar}]\n"
        f"   {predictor.predictions_correct}/{predictor.predictions_made} correct  |  👥 {subscriber_count()} subscribers\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Send /start to subscribe • /stop to unsubscribe</i>"
    )


# ── CONSOLE DASHBOARD ─────────────────────────────────────────────────────────
def print_dashboard(predictor, candle, prediction, confidence, signals, actual_dir):
    sep = "=" * 58
    g, r = predictor.total_green, predictor.total_red
    bt   = g + r
    g_bar = int(40 * g / bt) if bt else 0
    acc_bar = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    pred_emoji = "🟢" if prediction == "green" else ("🔴" if prediction else "⏳")
    dir_emoji  = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")

    print(f"\n{sep}")
    print(f"  🤖 BTC Predictor  |  {ts(candle[0])}  |  ${float(candle[4]):,.2f}")
    print(sep)
    print(f"  Window : {predictor.candle_count:,}  |  🟢{g:,}({predictor.green_pct}%)  🔴{r:,}({predictor.red_pct}%)")
    print(f"  [{'█'*g_bar}{'▓'*(40-g_bar)}]")
    print(f"  Last   : {dir_emoji} {actual_dir.upper()}")
    if prediction:
        print(f"  Next   : {pred_emoji} {prediction.upper()}  ({confidence:.1f}% confidence)")
        for n, v in signals.items():
            print(f"    {n:<16}: {v}")
        print(f"  Acc    : {predictor.accuracy}%  [{acc_bar}]  ({predictor.predictions_correct}/{predictor.predictions_made})")
        print(f"  Subs   : {subscriber_count()}")
    else:
        print(f"  ⏳ Warming up... {MIN_CANDLES - predictor.candle_count:,} more candles needed")
    print(sep)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    global _predictor_ref

    print("=" * 58)
    print("  BTC/USD 5m Predictor — PUBLIC BOT")
    print("=" * 58)

    init_db()
    add_subscriber(ADMIN_CHAT_ID, "admin")
    print(f"✅ DB ready. Subscribers: {subscriber_count()}")

    # Load history
    historical = fetch_historical(SYMBOL, INTERVAL, DAYS)
    predictor  = CandlePredictor()
    _predictor_ref = predictor   # expose to command handlers

    print("⚙️  Building rolling window...")
    for candle in historical:
        predictor.add_candle(candle)

    print(f"✅ Window built: {predictor.candle_count:,} candles | "
          f"🟢{predictor.total_green:,} | 🔴{predictor.total_red:,}\n")

    # ── Run initial backtest on historical data ───────────────────────────────
    print("🔬 Running initial backtest on historical data...")
    bt_data = predictor.run_backtest(historical)
    predictor.print_backtest_report_console(bt_data)

    # Send backtest summary to admin on startup
    if bt_data:
        acc = bt_data["overall_accuracy"]
        acc_bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        bt_summary = (
            f"📋 <b>Startup Backtest Complete</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  Total   : {bt_data['total']:,} predictions\n"
            f"  ✅ Correct : {bt_data['correct']:,}\n"
            f"  ❌ Wrong   : {bt_data['wrong']:,}\n"
            f"  🎯 Accuracy: <b>{acc}%</b>  [{acc_bar}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  🔥 Longest correct streak : {bt_data['streak_correct']}\n"
            f"  💀 Longest wrong streak   : {bt_data['streak_wrong']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  <i>Send /backtest anytime for the full confidence-band report</i>"
        )
        send_message(ADMIN_CHAT_ID, bt_summary)

    # ── Live prediction kick-off ──────────────────────────────────────────────
    prediction, confidence, signals = predictor.predict_next()
    predictor.last_prediction = prediction

    # Update shared state
    shared_state.update({
        "prediction": prediction, "confidence": confidence,
        "accuracy":   predictor.accuracy,
        "correct":    predictor.predictions_correct,
        "total":      predictor.predictions_made,
        "green":      predictor.total_green,
        "red":        predictor.total_red,
        "green_pct":  predictor.green_pct,
        "red_pct":    predictor.red_pct,
        "price":      float(historical[-1][4]) if historical else 0,
    })

    # Startup broadcast
    send_message(ADMIN_CHAT_ID,
        f"🚀 <b>BTC Predictor Bot is LIVE!</b>\n\n"
        f"📊 Loaded {predictor.candle_count:,} candles\n"
        f"🟢 Green : {predictor.total_green:,} ({predictor.green_pct}%)\n"
        f"🔴 Red   : {predictor.total_red:,} ({predictor.red_pct}%)\n"
        f"👥 Subscribers : {subscriber_count()}\n\n"
        f"🔮 First prediction: {'🟢 GREEN' if prediction == 'green' else '🔴 RED'} ({confidence:.1f}%)\n\n"
        f"<i>Share your bot so others can /start!\n"
        f"Send /backtest for the full confidence-band report.</i>"
    )

    # Start polling thread
    t = threading.Thread(target=polling_thread, daemon=True)
    t.start()

    print("🔄 Entering live loop...\n")
    last_seen_time = int(historical[-1][0]) if historical else 0

    while True:
        try:
            latest      = fetch_latest_candle(SYMBOL, INTERVAL)
            candle_time = int(latest[0])

            if candle_time != last_seen_time:
                actual_dir = classify(latest)
                outcome    = predictor.record_outcome(actual_dir)
                predictor.add_candle(latest)
                prediction, confidence, signals = predictor.predict_next()
                predictor.last_prediction = prediction

                # Update shared state for /status command
                shared_state.update({
                    "streak":     0,
                    "dir":        actual_dir,
                    "prediction": prediction,
                    "confidence": confidence,
                    "accuracy":   predictor.accuracy,
                    "correct":    predictor.predictions_correct,
                    "total":      predictor.predictions_made,
                    "green":      predictor.total_green,
                    "red":        predictor.total_red,
                    "green_pct":  predictor.green_pct,
                    "red_pct":    predictor.red_pct,
                    "price":      float(latest[4]),
                })

                print_dashboard(predictor, latest, prediction, confidence, signals, actual_dir)

                # Broadcast to all subscribers
                if prediction:
                    msg = build_broadcast(predictor, latest, prediction, confidence, signals, actual_dir, outcome)
                    broadcast(msg)

                last_seen_time = candle_time

            else:
                fo = float(latest[1]); fc = float(latest[4])
                fp = ((fc - fo) / fo * 100) if fo else 0
                print(
                    f"  [{now_str()}] Forming... {'🟢' if fc>fo else '🔴'} {fp:+.3f}%  |  "
                    f"Next: {'🟢' if prediction=='green' else '🔴' if prediction else '⏳'} "
                    f"{prediction or 'warming up'} ({confidence:.1f}%)",
                    end="\r"
                )

        except requests.exceptions.RequestException as e:
            print(f"\n  ⚠️  Network error: {e} — retrying...")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()
