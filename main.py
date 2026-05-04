"""
BTC/USD 5m Rolling Candle Predictor — PUBLIC BOT v2
─────────────────────────────────────────────────────────────
META-LEVEL ML UPGRADES:
  1. Online Learning     — SGDClassifier with partial_fit() updates every candle
  2. Calibrated Probs    — Platt scaling via CalibratedClassifierCV
  3. Concept Drift       — ADWIN algorithm triggers automatic retraining
  4. Feature Engineering — Lag features, rolling stats, sin/cos time encoding

Original signals (Momentum, Streak, Markov, EMA, RSI) are now *features*
fed into the ML model instead of hand-weighted heuristics.
"""

import requests
import time
import os
import sqlite3
import threading
import collections
import math
import pickle
import numpy as np
from datetime import datetime, timezone

# ── OPTIONAL IMPORTS (graceful fallback if not installed) ─────────────────────
try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not found. pip install scikit-learn — falling back to legacy scoring.")

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

# ── ML CONFIG ─────────────────────────────────────────────────────────────────
ML_WARMUP_CANDLES   = 200     # candles needed before ML model activates
ML_RETRAIN_EVERY    = 288     # retrain full model every N candles (~1 day)
ADWIN_DELTA         = 0.002   # ADWIN sensitivity (lower = more sensitive)
CALIBRATION_WINDOW  = 500     # candles used for Platt scaling calibration
LAG_STEPS           = [1, 2, 3, 5, 10, 20]   # lag feature offsets
ROLLING_WINDOWS     = [10, 20, 50]            # rolling stat windows

POLL_INTERVAL    = 2
DB_FILE          = "subscribers.db"
MODEL_FILE       = "btc_model.pkl"
# ─────────────────────────────────────────────────────────────────────────────


def ts(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def now_str():
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 3: ADWIN CONCEPT DRIFT DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
class ADWIN:
    """
    Adaptive Windowing (ADWIN) drift detector.
    Monitors a binary stream (correct/incorrect predictions).
    Raises drift_detected=True when the error rate has statistically shifted.

    Reference: Bifet & Gavalda, 2007.
    """
    def __init__(self, delta=0.002):
        self.delta   = delta
        self.window  = []          # list of 0/1 (0=correct, 1=error)
        self.total   = 0.0
        self.n       = 0
        self.drift_detected = False
        self.drift_count    = 0

    def add_element(self, is_error: bool):
        """Add a new observation. Returns True if drift detected."""
        val = 1.0 if is_error else 0.0
        self.window.append(val)
        self.total += val
        self.n     += 1
        self.drift_detected = False

        if self.n < 30:
            return False

        # Check all possible split points for statistical difference
        n0    = 0
        sum0  = 0.0
        mu    = self.total / self.n
        drift = False

        for i in range(self.n - 1, 0, -1):
            n0   += 1
            sum0 += self.window[-(n0)]
            n1    = self.n - n0
            sum1  = self.total - sum0
            if n0 < 5 or n1 < 5:
                continue
            mu0 = sum0 / n0
            mu1 = sum1 / n1
            if abs(mu0 - mu1) < self._epsilon_cut(n0, n1):
                break
            if abs(mu0 - mu1) >= self._epsilon_cut(n0, n1):
                # Drift found — drop the older portion
                self.window  = self.window[-n0:]
                self.total   = sum0
                self.n       = n0
                self.drift_detected = True
                self.drift_count   += 1
                drift = True
                break

        return drift

    def _epsilon_cut(self, n0, n1):
        n = n0 + n1
        m = 1 / n0 + 1 / n1
        return math.sqrt(m * math.log(2 * n / self.delta) / 2)

    def reset(self):
        self.window  = []
        self.total   = 0.0
        self.n       = 0
        self.drift_detected = False


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 4: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
class FeatureEngineer:
    """
    Transforms raw candle history into a rich feature vector.

    Features produced:
      - Original 5 signals (Momentum, Streak, Markov, EMA, RSI) as floats
      - Lag features: was candle[t-k] green? (binary) for k in LAG_STEPS
      - Rolling green rate over multiple windows
      - Rolling volatility (std of close-to-close returns)
      - Candle body/wick ratio
      - MACD histogram (fast EMA - slow EMA)
      - Time encoding: sin/cos of hour-of-day and day-of-week
    """

    @staticmethod
    def compute_ema(prices, period):
        if len(prices) < period:
            return None
        k   = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for p in prices[period:]:
            ema = p * k + ema * (1 - k)
        return ema

    @staticmethod
    def compute_rsi(closes, period=14):
        if len(closes) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, period + 1):
            diff = closes[-period - 1 + i] - closes[-period - 1 + i - 1]
            if diff >= 0: gains.append(diff); losses.append(0)
            else:         gains.append(0);    losses.append(abs(diff))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0: return 100.0
        return 100 - (100 / (1 + avg_gain / avg_loss))

    @staticmethod
    def extract(window: list, candle_times: list) -> np.ndarray | None:
        """
        window      : list of (timestamp_ms, direction_str, close_price)
        candle_times: list of int timestamps (ms) matching window

        Returns a 1D numpy feature vector, or None if not enough data.
        """
        if len(window) < max(LAG_STEPS) + max(ROLLING_WINDOWS) + RSI_PERIOD + 2:
            return None

        dirs   = [1 if d == "green" else 0 for _, d, _ in window]
        closes = [c for _, _, c in window]
        opens  = []  # filled from candle raw data if available

        features = []

        # ── 1. MOMENTUM (50-candle green rate) ────────────────────────────
        recent = dirs[-MOMENTUM_WINDOW:]
        r_total = len(recent)
        features.append(sum(recent) / r_total if r_total else 0.5)

        # ── 2. STREAK ─────────────────────────────────────────────────────
        sc       = dirs[-STREAK_WINDOW:]
        last_dir = sc[-1] if sc else 0
        streak   = sum(1 for d in reversed(sc) if d == last_dir)
        # encode: positive = green streak, negative = red streak
        features.append(streak if last_dir == 1 else -streak)

        # ── 3. MARKOV (transition probability to green) ───────────────────
        # P(green | last was green) and P(green | last was red)
        gg = gr = rg = rr = 0
        for i in range(1, min(len(dirs), 500)):
            prev, curr = dirs[-(i+1)], dirs[-i]
            if prev == 1 and curr == 1: gg += 1
            elif prev == 1 and curr == 0: gr += 1
            elif prev == 0 and curr == 1: rg += 1
            else: rr += 1
        if last_dir == 1:
            t = gg + gr
            features.append(gg / t if t > 0 else 0.5)
        else:
            t = rg + rr
            features.append(rg / t if t > 0 else 0.5)

        # ── 4. EMA deviation ──────────────────────────────────────────────
        ema = FeatureEngineer.compute_ema(closes, EMA_PERIOD)
        cp  = closes[-1]
        features.append(((cp - ema) / ema) if ema else 0.0)

        # ── 5. RSI ────────────────────────────────────────────────────────
        rsi = FeatureEngineer.compute_rsi(closes, RSI_PERIOD)
        features.append((rsi - 50) / 50)   # normalise to [-1, 1]

        # ── 6. LAG FEATURES (binary: was candle green?) ───────────────────
        for k in LAG_STEPS:
            features.append(dirs[-k] if k <= len(dirs) else 0.5)

        # ── 7. ROLLING GREEN RATE ─────────────────────────────────────────
        for w in ROLLING_WINDOWS:
            chunk = dirs[-w:]
            features.append(sum(chunk) / len(chunk) if chunk else 0.5)

        # ── 8. ROLLING VOLATILITY (std of log returns) ───────────────────
        for w in ROLLING_WINDOWS:
            c = closes[-w-1:]
            if len(c) >= 2:
                log_rets = [math.log(c[i]/c[i-1]) for i in range(1, len(c)) if c[i-1] > 0]
                if log_rets:
                    mean_r = sum(log_rets) / len(log_rets)
                    std_r  = math.sqrt(sum((r - mean_r)**2 for r in log_rets) / len(log_rets))
                    features.append(std_r)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)

        # ── 9. MACD HISTOGRAM (12 EMA - 26 EMA) ──────────────────────────
        ema12 = FeatureEngineer.compute_ema(closes, 12)
        ema26 = FeatureEngineer.compute_ema(closes, 26)
        if ema12 and ema26 and closes[-1] > 0:
            features.append((ema12 - ema26) / closes[-1])
        else:
            features.append(0.0)

        # ── 10. TIME ENCODING (sin/cos of hour and weekday) ───────────────
        if candle_times:
            last_ts = candle_times[-1]
            dt      = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            hour    = dt.hour + dt.minute / 60
            weekday = dt.weekday()
            features.append(math.sin(2 * math.pi * hour / 24))
            features.append(math.cos(2 * math.pi * hour / 24))
            features.append(math.sin(2 * math.pi * weekday / 7))
            features.append(math.cos(2 * math.pi * weekday / 7))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1 & 2: ONLINE LEARNING + CALIBRATED PROBABILITIES
# ══════════════════════════════════════════════════════════════════════════════
class OnlineMLPredictor:
    """
    SGDClassifier updated incrementally via partial_fit() after every candle.
    Probability calibration via isotonic regression (Platt scaling) on a
    rolling buffer of recent predictions and outcomes.

    Falls back to legacy heuristic scoring if sklearn is unavailable or
    the model hasn't warmed up yet.
    """

    def __init__(self):
        self.is_ready      = False
        self.candles_seen  = 0

        # Feature + label buffers for calibration and periodic full retrain
        self.feature_buf   = collections.deque(maxlen=CALIBRATION_WINDOW)
        self.label_buf     = collections.deque(maxlen=CALIBRATION_WINDOW)

        # Raw SGD model (online updates)
        # FIX: class_weight must be a dict (not 'balanced') for partial_fit compatibility.
        # We start with equal weights and recompute dynamically in partial_update().
        if SKLEARN_AVAILABLE:
            self.sgd = SGDClassifier(
                loss="log_loss",
                alpha=0.0001,
                max_iter=1,
                warm_start=True,
                class_weight={0: 1.0, 1: 1.0},  # FIX: plain dict instead of 'balanced'
                random_state=42,
            )
            self.scaler         = StandardScaler()
            self.scaler_fitted  = False
            self.calibrated_clf = None   # set after enough calibration data
        else:
            self.sgd = None

        # IMPROVEMENT 2: calibration metrics
        self.prob_buf    = collections.deque(maxlen=200)   # (predicted_prob, actual_label)
        self.cal_correct = 0
        self.cal_total   = 0

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if not self.scaler_fitted:
            self.scaler.partial_fit(X)
            self.scaler_fitted = True
        return self.scaler.transform(X)

    def _compute_class_weights(self) -> dict:
        """
        Compute balanced class weights from the rolling label buffer.
        Mirrors sklearn's 'balanced' formula: w_c = n_samples / (n_classes * n_c)
        """
        labels = np.array(self.label_buf)
        n_total = len(labels)
        n0 = int(np.sum(labels == 0))
        n1 = int(np.sum(labels == 1))
        w0 = n_total / (2 * n0) if n0 > 0 else 1.0
        w1 = n_total / (2 * n1) if n1 > 0 else 1.0
        return {0: w0, 1: w1}

    def partial_update(self, feature_vec: np.ndarray, label: int):
        """
        Called after each candle close with the feature vector that was
        computed BEFORE that candle and the label (1=green, 0=red) of that candle.
        """
        if not SKLEARN_AVAILABLE or feature_vec is None:
            return

        self.feature_buf.append(feature_vec)
        self.label_buf.append(label)
        self.candles_seen += 1

        X = feature_vec.reshape(1, -1)
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        self.scaler_fitted = True

        # partial_fit needs both classes to have been seen at least once
        if len(set(self.label_buf)) == 2:
            # FIX: dynamically compute and apply balanced class weights
            self.sgd.class_weight = self._compute_class_weights()
            self.sgd.partial_fit(X_scaled, [label], classes=[0, 1])

        if self.candles_seen >= ML_WARMUP_CANDLES:
            self.is_ready = True

    def full_retrain(self):
        """Periodic full refit on the rolling buffer — resets SGD internal state."""
        if not SKLEARN_AVAILABLE or len(self.label_buf) < ML_WARMUP_CANDLES:
            return
        X = np.array(self.feature_buf)
        y = np.array(self.label_buf)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Fresh SGD retrain — full .fit() supports 'balanced' string directly
        self.sgd = SGDClassifier(
            loss="log_loss", alpha=0.0001, max_iter=200,
            class_weight="balanced", random_state=42,
        )
        self.sgd.fit(X_scaled, y)

        # IMPROVEMENT 2: fit calibrated wrapper on same data
        try:
            base = SGDClassifier(
                loss="log_loss", alpha=0.0001, max_iter=200,
                class_weight="balanced", random_state=42,
            )
            self.calibrated_clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
            self.calibrated_clf.fit(X_scaled, y)
            print(f"  🎯 Calibrated model retrained on {len(y)} samples")
        except Exception as e:
            print(f"  ⚠️  Calibration failed: {e}")
            self.calibrated_clf = None

        # After full_retrain, reset online SGD to dict-based weights so
        # subsequent partial_fit() calls keep working correctly
        self.sgd = SGDClassifier(
            loss="log_loss", alpha=0.0001, max_iter=1,
            warm_start=True,
            class_weight={0: 1.0, 1: 1.0},  # FIX: dict for partial_fit compatibility
            random_state=42,
        )
        self.sgd.fit(X_scaled, y)   # seed warm_start with current data

        self.scaler_fitted = True
        print(f"  🔄 Full retrain complete. Samples: {len(y)}")

    def predict(self, feature_vec: np.ndarray):
        """
        Returns (prediction: str, confidence: float, source: str)
        source is 'ml_calibrated', 'ml_sgd', or 'legacy'
        """
        if not SKLEARN_AVAILABLE or not self.is_ready or feature_vec is None:
            return None, 0.0, "legacy"

        X_scaled = self.scaler.transform(feature_vec.reshape(1, -1))

        # Prefer calibrated model
        clf = self.calibrated_clf if self.calibrated_clf is not None else self.sgd
        source = "ml_calibrated" if self.calibrated_clf is not None else "ml_sgd"

        try:
            proba = clf.predict_proba(X_scaled)[0]   # [P(red), P(green)]
            p_green = float(proba[1])
            prediction = "green" if p_green >= 0.5 else "red"
            confidence = abs(p_green - 0.5) * 200    # 0–100 scale
            return prediction, round(confidence, 1), source
        except Exception as e:
            print(f"  ⚠️  ML predict error: {e}")
            return None, 0.0, "legacy"

    def record_prob_outcome(self, predicted_prob: float, actual_label: int):
        """Track calibration quality."""
        self.prob_buf.append((predicted_prob, actual_label))
        predicted_label = 1 if predicted_prob >= 0.5 else 0
        if predicted_label == actual_label:
            self.cal_correct += 1
        self.cal_total += 1

    def calibration_summary(self) -> str:
        """Human-readable calibration stats."""
        if self.cal_total == 0:
            return "no data"
        acc = 100 * self.cal_correct / self.cal_total
        return f"{acc:.1f}% ({self.cal_correct}/{self.cal_total})"


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
        if ok: sent += 1
        else:  failed += 1; dead.append(chat_id)
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


# ── SHARED STATE ──────────────────────────────────────────────────────────────
shared_state = {
    "prediction": None, "confidence": 0, "accuracy": 0.0,
    "correct": 0, "total": 0, "green": 0, "red": 0,
    "green_pct": 0, "red_pct": 0, "price": 0,
    "ml_source": "warming up", "drift_count": 0,
    "cal_accuracy": "—",
}

# ── COMMAND HANDLERS ──────────────────────────────────────────────────────────
def handle_start(chat_id, username):
    if is_subscribed(chat_id):
        send_message(chat_id,
            "✅ <b>You're already subscribed!</b>\n\n"
            "Commands:\n/stop — Unsubscribe\n/status — Live stats"
        )
    else:
        add_subscriber(chat_id, username)
        send_message(chat_id,
            f"🎉 <b>Welcome! You're now subscribed!</b>\n\n"
            f"📌 Every 5 minutes you'll get:\n"
            f"  • Last candle result (🟢/🔴)\n"
            f"  • ML-powered next candle prediction\n"
            f"  • Calibrated confidence score\n"
            f"  • Drift detection status\n\n"
            f"⚡ Commands: /start  /stop  /status\n\n"
            f"<i>First update at the next 5m candle close! 🚀</i>"
        )
        print(f"  ➕ New subscriber: {username} ({chat_id})")
        send_message(ADMIN_CHAT_ID,
            f"➕ <b>New subscriber!</b>\n👤 {username or 'Unknown'} ({chat_id})\n"
            f"👥 Total: {subscriber_count()}"
        )

def handle_stop(chat_id, username):
    if is_subscribed(chat_id):
        remove_subscriber(chat_id)
        send_message(chat_id, "😢 <b>Unsubscribed.</b>\nSend /start anytime!")
        print(f"  ➖ Unsubscribed: {username} ({chat_id})")
    else:
        send_message(chat_id, "⚠️ You're not subscribed.\nSend /start!")

def handle_status(chat_id):
    s   = shared_state
    pe  = "🟢" if s["prediction"] == "green" else ("🔴" if s["prediction"] == "red" else "⏳")
    bar = "█" * int(s["accuracy"] / 5) + "░" * (20 - int(s["accuracy"] / 5))
    send_message(chat_id,
        f"📊 <b>BTC/USD 5m Predictor — Live Status</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 Price        : ${s['price']:,.2f}\n"
        f"🟢 Green        : {s['green']:,} ({s['green_pct']}%)\n"
        f"🔴 Red          : {s['red']:,} ({s['red_pct']}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 Prediction   : {pe} <b>{(s['prediction'] or 'warming up').upper()}</b>\n"
        f"   Confidence   : <b>{s['confidence']:.1f}%</b>\n"
        f"   Model        : <i>{s['ml_source']}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Raw accuracy : <b>{s['accuracy']}%</b>\n"
        f"   [{bar}] {s['correct']}/{s['total']}\n"
        f"🧪 Cal accuracy : <b>{s['cal_accuracy']}</b>\n"
        f"🌊 Drift events : <b>{s['drift_count']}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👥 Subscribers  : {subscriber_count()}"
    )

def handle_subscribers(chat_id):
    if str(chat_id) != str(ADMIN_CHAT_ID):
        send_message(chat_id, "⛔ Admin only.")
        return
    subs  = get_all_subscribers()
    lines = "\n".join(f"  {i+1}. {u or 'Unknown'} ({cid})" for i, (cid, u) in enumerate(subs[:20]))
    send_message(chat_id,
        f"👥 <b>Subscribers ({len(subs)} total)</b>\n\n<code>{lines}</code>"
        + (f"\n\n<i>...and {len(subs)-20} more</i>" if len(subs) > 20 else "")
    )

def polling_thread():
    offset = None
    print("🤖 Telegram polling started...")
    while True:
        try:
            updates = get_updates(offset)
            for update in updates:
                offset   = update["update_id"] + 1
                msg      = update.get("message") or update.get("edited_message")
                if not msg: continue
                chat_id  = str(msg["chat"]["id"])
                username = msg.get("from", {}).get("username") or \
                           msg.get("from", {}).get("first_name", "Unknown")
                text     = msg.get("text", "").strip().lower().split()[0]
                if text == "/start":         handle_start(chat_id, username)
                elif text == "/stop":        handle_stop(chat_id, username)
                elif text == "/status":      handle_status(chat_id)
                elif text == "/subscribers": handle_subscribers(chat_id)
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
            "category": "spot", "symbol": symbol, "interval": interval,
            "limit": BATCH_SIZE, "end": end_ms, "start": start_ms,
        }
        try:
            resp = requests.get(BYBIT_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                print(f"❌ Bybit error: {data.get('retMsg')}"); break
            batch = data["result"]["list"]
            if not batch: break
            all_candles.extend(batch)
            end_ms = int(batch[-1][0]) - MS_PER_5MIN
            print(f"  {len(all_candles):,} candles fetched...", end="\r")
            time.sleep(0.15)
        except Exception as e:
            print(f"\n  ⚠️  Fetch error: {e}"); time.sleep(2)
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


# ── LEGACY INDICATORS (kept as feature sources) ───────────────────────────────
def compute_ema(prices, period):
    if len(prices) < period: return None
    k   = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for p in prices[period:]: ema = p * k + ema * (1 - k)
    return ema

def compute_rsi(closes, period=14):
    if len(closes) < period + 1: return None
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[-period - 1 + i] - closes[-period - 1 + i - 1]
        if diff >= 0: gains.append(diff); losses.append(0)
        else:         gains.append(0);   losses.append(abs(diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0: return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


# ── LEGACY PREDICTOR (fallback + feature source) ──────────────────────────────
class CandlePredictor:
    def __init__(self):
        self.window      = collections.deque(maxlen=WINDOW_SIZE)
        self.closes      = collections.deque(maxlen=WINDOW_SIZE)
        self.times       = collections.deque(maxlen=WINDOW_SIZE)
        self.total_green = self.total_red = self.total_doji = 0
        self.markov = {
            "green": {"green": 0, "red": 0},
            "red":   {"green": 0, "red": 0},
            "doji":  {"green": 0, "red": 0},
        }
        self.predictions_made = self.predictions_correct = 0
        self.last_prediction  = None
        self.last_candle_time = None

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
        self.times.append(int(candle[0]))
        if direction == "green": self.total_green += 1
        elif direction == "red": self.total_red   += 1
        else:                    self.total_doji  += 1
        self.last_candle_time = int(candle[0])
        return direction

    def legacy_predict(self):
        """Original heuristic scoring — used as fallback."""
        if len(self.window) < MIN_CANDLES:
            return None, 0, {}
        signals = {}
        green_score = total_weight = 0.0
        recent  = list(self.window)[-MOMENTUM_WINDOW:]
        r_green = sum(1 for _, d, _ in recent if d == "green")
        r_red   = sum(1 for _, d, _ in recent if d == "red")
        r_total = r_green + r_red
        if r_total > 0:
            mom = r_green / r_total
            green_score += mom * 1.5; total_weight += 1.5
            signals["Momentum(50)"] = f"{'🟢' if mom > 0.5 else '🔴'} {mom*100:.1f}% green"
        sc       = [d for _, d, _ in list(self.window)[-STREAK_WINDOW:]]
        last_dir = sc[-1] if sc else "doji"
        streak_len = sum(1 for d in reversed(sc) if d == last_dir) if last_dir != "doji" else 0
        if last_dir != "doji" and streak_len >= 2:
            rw = min(streak_len / 6, 1.0)
            ss = (0.5 - rw * 0.35) if last_dir == "green" else (0.5 + rw * 0.35)
            green_score += ss * 1.2; total_weight += 1.2
            signals["Streak"] = f"{'🟢' if last_dir=='green' else '🔴'} {streak_len}x {last_dir}"
        if last_dir in self.markov:
            m  = self.markov[last_dir]
            mt = m["green"] + m["red"]
            if mt > 10:
                ms = m["green"] / mt
                green_score += ms * 2.0; total_weight += 2.0
                signals["Markov"] = f"After {last_dir}: 🟢{m['green']} / 🔴{m['red']} ({ms*100:.1f}%)"
        cl  = list(self.closes)
        ema = compute_ema(cl, EMA_PERIOD)
        if ema and cl:
            es = 0.6 if cl[-1] > ema else 0.4
            green_score += es * 1.0; total_weight += 1.0
            signals["EMA(20)"] = f"Price {'above' if cl[-1] > ema else 'below'} EMA ({((cl[-1]-ema)/ema)*100:+.3f}%)"
        rsi = compute_rsi(cl, RSI_PERIOD)
        if rsi is not None:
            rs = 0.3 if rsi > 70 else (0.7 if rsi < 30 else 0.5)
            green_score += rs * 1.3; total_weight += 1.3
            signals["RSI(14)"] = f"{rsi:.1f} ({'overbought🔴' if rsi > 70 else 'oversold🟢' if rsi < 30 else 'neutral⚪'})"
        if total_weight == 0: return None, 0, {}
        fs = green_score / total_weight
        return ("green" if fs >= 0.5 else "red"), round(abs(fs - 0.5) * 200, 1), signals

    def get_feature_vector(self):
        return FeatureEngineer.extract(list(self.window), list(self.times))

    def record_outcome(self, actual):
        if self.last_prediction and actual in ("green", "red"):
            self.predictions_made += 1
            if self.last_prediction == actual:
                self.predictions_correct += 1
                return True
            return False
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


# ── BROADCAST MESSAGE BUILDER ─────────────────────────────────────────────────
def build_broadcast(predictor, ml_pred, candle, prediction, confidence, signals,
                    actual_dir, outcome, ml_source, drift_count, cal_acc):
    pred_emoji = "🟢" if prediction == "green" else "🔴"
    dir_emoji  = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")
    outcome_str = ("✅ Correct!" if outcome else "❌ Wrong") if outcome is not None else ""
    acc_bar    = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    signal_lines = "\n".join(f"  • <b>{n}</b>: {v}" for n, v in signals.items())

    source_label = {
        "ml_calibrated": "🧠 ML (calibrated)",
        "ml_sgd":        "🤖 ML (online SGD)",
        "legacy":        "📐 Heuristic signals",
    }.get(ml_source, ml_source)

    drift_note = f"\n⚡ <i>Drift detected! Model retrained. ({drift_count} total)</i>" if drift_count > 0 else ""

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
        f"  Model: {source_label}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 <b>Signals (feature inputs):</b>\n{signal_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Raw accuracy : <b>{predictor.accuracy}%</b>  [{acc_bar}]\n"
        f"   {predictor.predictions_correct}/{predictor.predictions_made} correct\n"
        f"🧪 Cal accuracy : <b>{cal_acc}</b>\n"
        f"🌊 Drift events : {drift_count}{drift_note}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"👥 {subscriber_count()} subscribers  •  /start /stop /status"
    )


# ── CONSOLE DASHBOARD ─────────────────────────────────────────────────────────
def print_dashboard(predictor, ml_pred, candle, prediction, confidence,
                    signals, actual_dir, ml_source, drift_count):
    sep = "=" * 62
    g, r = predictor.total_green, predictor.total_red
    bt   = g + r
    g_bar = int(40 * g / bt) if bt else 0
    acc_bar = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    pred_e = "🟢" if prediction == "green" else ("🔴" if prediction else "⏳")
    dir_e  = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")
    source = {"ml_calibrated": "ML-Cal", "ml_sgd": "ML-SGD", "legacy": "Heuristic"}.get(ml_source, ml_source)

    print(f"\n{sep}")
    print(f"  🤖 BTC Predictor v2  |  {ts(candle[0])}  |  ${float(candle[4]):,.2f}")
    print(sep)
    print(f"  Window : {predictor.candle_count:,}  |  🟢{g:,}({predictor.green_pct}%)  🔴{r:,}({predictor.red_pct}%)")
    print(f"  [{'█'*g_bar}{'▓'*(40-g_bar)}]")
    print(f"  Last   : {dir_e} {actual_dir.upper()}")
    if prediction:
        print(f"  Next   : {pred_e} {prediction.upper()}  ({confidence:.1f}% conf)  [{source}]")
        for n, v in signals.items():
            print(f"    {n:<16}: {v}")
        print(f"  Acc    : {predictor.accuracy}%  [{acc_bar}]  ({predictor.predictions_correct}/{predictor.predictions_made})")
        print(f"  Drift  : {drift_count} events  |  Subs: {subscriber_count()}")
    else:
        print(f"  ⏳ Warming up... {MIN_CANDLES - predictor.candle_count:,} more candles needed")
    if drift_count > 0:
        print(f"  ⚡ DRIFT DETECTED — model retrained!")
    print(sep)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  BTC/USD 5m Predictor v2 — ML UPGRADED")
    print("  Improvements: Online Learning | Calibration | ADWIN Drift | Feature Eng")
    print("=" * 62)

    if not SKLEARN_AVAILABLE:
        print("⚠️  Install scikit-learn for full ML: pip install scikit-learn numpy")

    init_db()
    add_subscriber(ADMIN_CHAT_ID, "admin")
    print(f"✅ DB ready. Subscribers: {subscriber_count()}")

    # Initialise all components
    predictor  = CandlePredictor()
    ml_pred    = OnlineMLPredictor()
    drift_det  = ADWIN(delta=ADWIN_DELTA)          # IMPROVEMENT 3
    candles_since_retrain = 0

    # Load history
    historical = fetch_historical(SYMBOL, INTERVAL, DAYS)

    print("⚙️  Building rolling window + pre-training ML model...")
    for i, candle in enumerate(historical):
        # Get feature vector BEFORE adding this candle (label = what this candle will be)
        feat = predictor.get_feature_vector() if len(predictor.window) >= ML_WARMUP_CANDLES else None

        direction = predictor.add_candle(candle)
        label = 1 if direction == "green" else 0

        # Online update
        if feat is not None and SKLEARN_AVAILABLE:
            ml_pred.partial_update(feat, label)

        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,}/{len(historical):,} candles processed...", end="\r")

    # Full retrain on historical data
    print("\n⚙️  Running initial full retrain...")
    ml_pred.full_retrain()

    print(f"✅ Window built: {predictor.candle_count:,} candles | "
          f"🟢{predictor.total_green:,} | 🔴{predictor.total_red:,} | "
          f"ML ready: {ml_pred.is_ready}\n")

    # Initial prediction
    feat = predictor.get_feature_vector()
    ml_prediction, ml_confidence, ml_source = ml_pred.predict(feat)
    if ml_prediction:
        prediction, confidence = ml_prediction, ml_confidence
        source = ml_source
    else:
        prediction, confidence, signals = predictor.legacy_predict()
        source = "legacy"

    _, _, signals = predictor.legacy_predict()
    predictor.last_prediction = prediction

    shared_state.update({
        "prediction": prediction, "confidence": confidence,
        "accuracy": predictor.accuracy, "correct": predictor.predictions_correct,
        "total": predictor.predictions_made, "green": predictor.total_green,
        "red": predictor.total_red, "green_pct": predictor.green_pct,
        "red_pct": predictor.red_pct,
        "price": float(historical[-1][4]) if historical else 0,
        "ml_source": source, "drift_count": 0,
        "cal_accuracy": ml_pred.calibration_summary(),
    })

    send_message(ADMIN_CHAT_ID,
        f"🚀 <b>BTC Predictor v2 is LIVE!</b>\n\n"
        f"📊 {predictor.candle_count:,} candles loaded\n"
        f"🧠 ML model ready: {ml_pred.is_ready}\n"
        f"🌊 ADWIN drift detector: active\n"
        f"📐 Features per candle: {feat.shape[0] if feat is not None else 'N/A'}\n\n"
        f"🔮 First prediction: {'🟢 GREEN' if prediction == 'green' else '🔴 RED'} "
        f"({confidence:.1f}%) via {source}"
    )

    # Start polling thread
    t = threading.Thread(target=polling_thread, daemon=True)
    t.start()

    print("🔄 Entering live loop...\n")
    last_seen_time = int(historical[-1][0]) if historical else 0
    last_feature_vec = feat   # store feature vec used for current prediction
    drift_count = 0

    while True:
        try:
            latest      = fetch_latest_candle(SYMBOL, INTERVAL)
            candle_time = int(latest[0])

            if candle_time != last_seen_time:
                actual_dir = classify(latest)
                label      = 1 if actual_dir == "green" else 0

                # ── Record outcome & update drift detector ────────────────
                outcome = predictor.record_outcome(actual_dir)
                is_error = (outcome is False)

                # IMPROVEMENT 3: feed outcome to ADWIN
                if outcome is not None:
                    drift = drift_det.add_element(is_error)
                    if drift:
                        drift_count += 1
                        print(f"\n  🌊 DRIFT DETECTED (event #{drift_count}) — triggering retrain...")
                        ml_pred.full_retrain()

                # IMPROVEMENT 2: record for calibration tracking
                if last_feature_vec is not None and ml_pred.is_ready:
                    _, raw_p, _ = ml_pred.predict(last_feature_vec)
                    p_val = 0.5 + raw_p / 200
                    ml_pred.record_prob_outcome(p_val, label)

                # IMPROVEMENT 1: online update with last feature vec
                if last_feature_vec is not None and SKLEARN_AVAILABLE:
                    ml_pred.partial_update(last_feature_vec, label)

                # ── Add candle to window ───────────────────────────────────
                predictor.add_candle(latest)
                candles_since_retrain += 1

                # ── Periodic full retrain ─────────────────────────────────
                if candles_since_retrain >= ML_RETRAIN_EVERY:
                    print(f"\n  🔄 Periodic retrain (every {ML_RETRAIN_EVERY} candles)...")
                    ml_pred.full_retrain()
                    candles_since_retrain = 0

                # ── Build features for NEXT candle prediction ─────────────
                # IMPROVEMENT 4: rich feature vector
                feat = predictor.get_feature_vector()
                last_feature_vec = feat

                # ── Predict next candle ───────────────────────────────────
                ml_prediction, ml_confidence, ml_source = ml_pred.predict(feat)
                if ml_prediction:
                    prediction, confidence, source = ml_prediction, ml_confidence, ml_source
                else:
                    prediction, confidence, signals = predictor.legacy_predict()
                    source = "legacy"

                _, _, signals = predictor.legacy_predict()   # always show signal breakdown
                predictor.last_prediction = prediction

                # ── Update shared state ───────────────────────────────────
                shared_state.update({
                    "prediction": prediction, "confidence": confidence,
                    "accuracy":  predictor.accuracy,
                    "correct":   predictor.predictions_correct,
                    "total":     predictor.predictions_made,
                    "green":     predictor.total_green,
                    "red":       predictor.total_red,
                    "green_pct": predictor.green_pct,
                    "red_pct":   predictor.red_pct,
                    "price":     float(latest[4]),
                    "ml_source": source,
                    "drift_count": drift_count,
                    "cal_accuracy": ml_pred.calibration_summary(),
                })

                print_dashboard(predictor, ml_pred, latest, prediction, confidence,
                                signals, actual_dir, source, drift_count)

                if prediction:
                    msg = build_broadcast(
                        predictor, ml_pred, latest, prediction, confidence,
                        signals, actual_dir, outcome, source, drift_count,
                        ml_pred.calibration_summary()
                    )
                    broadcast(msg)

                last_seen_time = candle_time
                drift_count = 0   # reset per-candle drift flag for broadcast

            else:
                fo = float(latest[1]); fc = float(latest[4])
                fp = ((fc - fo) / fo * 100) if fo else 0
                print(
                    f"  [{now_str()}] Forming... {'🟢' if fc>fo else '🔴'} {fp:+.3f}%  |  "
                    f"Next: {'🟢' if prediction=='green' else '🔴' if prediction else '⏳'} "
                    f"{prediction or 'warming up'} ({confidence:.1f}%)  [{source[:6]}]",
                    end="\r"
                )

        except requests.exceptions.RequestException as e:
            print(f"\n  ⚠️  Network error: {e} — retrying...")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()
