"""
BTC/USD 5m ML Candle Predictor — PUBLIC BOT (v2 — Enhanced Accuracy)
─────────────────────────────────────────────────────────────────────
Fixes:
  • IndexError in rsi() — guard now requires idx >= 60 + LOOKBACK

New accuracy improvements:
  • XGBoost classifier (falls back to RF if not installed)
  • Stochastic RSI — more sensitive than plain RSI
  • Williams %R — momentum oscillator
  • CCI (Commodity Channel Index) — trend/cycle detection
  • OBV (On-Balance Volume) — volume-price divergence signal
  • Candle-pattern features: doji, hammer, shooting-star, engulfing
  • Calibrated probabilities via CalibratedClassifierCV
  • Feature importance pruning — noisy features removed after training
  • Ensemble now: XGB + RF + LR (or RF + LR if no XGB)

Commands: /start, /stop, /status, /subscribers (admin only)
"""

import requests
import time
import os
import sqlite3
import threading
import collections
import math
from datetime import datetime, timezone

# ── scikit-learn + optional XGBoost ──────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score
    from sklearn.feature_selection import SelectFromModel
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn not found. Install: pip install scikit-learn numpy")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    print("✅ XGBoost found — will use as primary classifier.")
except ImportError:
    XGB_AVAILABLE = False
    print("ℹ️  XGBoost not installed. Using RF+LR ensemble. "
          "Install with: pip install xgboost   (recommended for better accuracy)")

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
MIN_TRAIN        = 2000
RETRAIN_EVERY    = 500
POLL_INTERVAL    = 2
DB_FILE          = "subscribers.db"

# Feature engineering windows
RSI_PERIOD       = 14
STOCH_RSI_PERIOD = 14
EMA_SHORT        = 9
EMA_LONG         = 21
EMA_TREND        = 50          # new: longer-term trend EMA
MACD_SIGNAL      = 9
BB_PERIOD        = 20
ATR_PERIOD       = 14
CCI_PERIOD       = 20
WILLIAMS_PERIOD  = 14
OBV_LOOKBACK     = 20
LOOKBACK         = 10

# Minimum idx in dataset before compute_features is called
# Must be >= 60 (our fixed history slice) + LOOKBACK
MIN_FEATURE_IDX  = 60 + LOOKBACK   # = 70
# ─────────────────────────────────────────────────────────────────────────────

def ts(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def now_str():
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


# ── DATABASE ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS subscribers (
        chat_id TEXT PRIMARY KEY, username TEXT, joined_at TEXT)""")
    conn.commit(); conn.close()

def add_subscriber(chat_id, username=""):
    conn = sqlite3.connect(DB_FILE)
    conn.cursor().execute(
        "INSERT OR IGNORE INTO subscribers VALUES (?,?,?)",
        (str(chat_id), username, datetime.now(timezone.utc).isoformat()))
    conn.commit(); conn.close()

def remove_subscriber(chat_id):
    conn = sqlite3.connect(DB_FILE)
    conn.cursor().execute("DELETE FROM subscribers WHERE chat_id=?", (str(chat_id),))
    conn.commit(); conn.close()

def get_all_subscribers():
    conn = sqlite3.connect(DB_FILE)
    rows = conn.cursor().execute("SELECT chat_id, username FROM subscribers").fetchall()
    conn.close(); return rows

def is_subscribed(chat_id):
    conn = sqlite3.connect(DB_FILE)
    r = conn.cursor().execute("SELECT 1 FROM subscribers WHERE chat_id=?", (str(chat_id),)).fetchone()
    conn.close(); return r is not None

def subscriber_count():
    conn = sqlite3.connect(DB_FILE)
    n = conn.cursor().execute("SELECT COUNT(*) FROM subscribers").fetchone()[0]
    conn.close(); return n


# ── TELEGRAM ──────────────────────────────────────────────────────────────────
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

def send_message(chat_id, text):
    try:
        r = requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": str(chat_id), "text": text, "parse_mode": "HTML"
        }, timeout=10)
        r.raise_for_status(); return True
    except Exception as e:
        print(f"  ❌ Send error {chat_id}: {e}"); return False

def broadcast(text):
    subs = get_all_subscribers()
    sent = failed = 0; dead = []
    for cid, _ in subs:
        if send_message(cid, text): sent += 1
        else: failed += 1; dead.append(cid)
        time.sleep(0.05)
    for d in dead: remove_subscriber(d)
    print(f"  📢 Broadcast: ✅{sent} ❌{failed}")
    return sent

def get_updates(offset=None):
    params = {"timeout": 30, "limit": 100}
    if offset: params["offset"] = offset
    try:
        r = requests.get(f"{BASE_URL}/getUpdates", params=params, timeout=35)
        r.raise_for_status(); return r.json().get("result", [])
    except Exception as e:
        print(f"  ⚠️  getUpdates: {e}"); return []


# ── FETCH ─────────────────────────────────────────────────────────────────────
def fetch_historical(days):
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    all_candles = []; end_ms = now_ms
    print(f"📥 Fetching {days} days of BTCUSDT 5m candles...")
    while end_ms > start_ms:
        params = {"category":"spot","symbol":SYMBOL,"interval":INTERVAL,
                  "limit":BATCH_SIZE,"end":end_ms,"start":start_ms}
        try:
            resp = requests.get(BYBIT_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0: break
            batch = data["result"]["list"]
            if not batch: break
            all_candles.extend(batch)
            end_ms = int(batch[-1][0]) - MS_PER_5MIN
            print(f"  {len(all_candles):,} candles...", end="\r")
            time.sleep(0.15)
        except Exception as e:
            print(f"\n  ⚠️ {e}"); time.sleep(2)
    all_candles.reverse()
    print(f"\n✅ Fetched {len(all_candles):,} candles.")
    return all_candles

def fetch_latest_candle():
    params = {"category":"spot","symbol":SYMBOL,"interval":INTERVAL,"limit":3}
    resp = requests.get(BYBIT_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0: raise Exception(f"Bybit: {data.get('retMsg')}")
    return list(reversed(data["result"]["list"]))[-2]

def classify(candle):
    o, cl = float(candle[1]), float(candle[4])
    if cl > o: return 1    # green
    if cl < o: return 0    # red
    return -1              # doji


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def _ema(vals, p):
    """Exponential moving average."""
    k = 2 / (p + 1)
    e = vals[0]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
    return e

def _rsi(vals, p=14):
    """Standard RSI. vals must have at least p+1 elements."""
    gains = losses = 0.0
    for i in range(1, p + 1):
        d = vals[i] - vals[i - 1]
        if d > 0: gains += d
        else:     losses += abs(d)
    ag = gains / p
    al = losses / p
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)

def _stoch_rsi(closes, rsi_p=14, stoch_p=14):
    """
    Stochastic RSI: measures RSI relative to its own high/low range.
    Returns a value in [0, 100].
    """
    # Need enough closes to compute rsi_p+1 values of RSI, then stoch_p of those
    needed = rsi_p + 1 + stoch_p
    if len(closes) < needed:
        return 50.0
    rsi_series = []
    for i in range(stoch_p):
        window = closes[i: i + rsi_p + 1]
        rsi_series.append(_rsi(window, rsi_p))
    rsi_now = _rsi(closes[stoch_p: stoch_p + rsi_p + 1], rsi_p)
    rsi_series.append(rsi_now)
    lo = min(rsi_series)
    hi = max(rsi_series)
    if hi == lo:
        return 50.0
    return 100 * (rsi_now - lo) / (hi - lo)

def _atr(highs, lows, closes, p=14):
    """Average True Range."""
    trs = [max(highs[i] - lows[i],
               abs(highs[i] - closes[i - 1]),
               abs(lows[i]  - closes[i - 1]))
           for i in range(1, p + 1)]
    return sum(trs) / p

def _cci(highs, lows, closes, p=20):
    """Commodity Channel Index."""
    tp = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(highs))]
    tp_window = tp[-p:]
    tp_mean = sum(tp_window) / p
    mean_dev = sum(abs(x - tp_mean) for x in tp_window) / p
    if mean_dev == 0:
        return 0.0
    return (tp_window[-1] - tp_mean) / (0.015 * mean_dev)

def _williams_r(highs, lows, closes, p=14):
    """Williams %R. Returns value in [-100, 0]."""
    h = max(highs[-p:])
    l = min(lows[-p:])
    if h == l:
        return -50.0
    return -100 * (h - closes[-1]) / (h - l)

def _obv_slope(closes, volumes, p=20):
    """
    On-Balance Volume: cumulative volume signed by price direction.
    Returns normalised slope of the last p OBV values.
    """
    if len(closes) < p + 1:
        return 0.0
    obv = 0.0
    obv_series = []
    for i in range(1, p + 1):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
        obv_series.append(obv)
    # Linear slope via simple regression
    n = len(obv_series)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(obv_series) / n
    num = sum((xs[i] - x_mean) * (obv_series[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    # Normalise by average volume so it's scale-invariant
    avg_vol = sum(volumes[-p:]) / p if p else 1
    return slope / avg_vol if avg_vol else 0.0

def _candle_patterns(opens, highs, lows, closes):
    """
    Encode common single/multi-candle patterns as numeric features.
    Returns a list of floats (all roughly in [-1, 1] range).
    """
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    body      = abs(c - o)
    full_rng  = h - l if h != l else 1e-9
    body_pct  = body / full_rng               # 0 = doji, 1 = marubozu
    upper_wick = (h - max(o, c)) / full_rng
    lower_wick = (min(o, c) - l) / full_rng
    direction  = 1 if c > o else -1

    # Doji: tiny body
    is_doji = 1.0 if body_pct < 0.1 else 0.0

    # Hammer / inverted hammer (bullish reversal)
    is_hammer = 1.0 if (lower_wick > 2 * body_pct and upper_wick < 0.1 and c > o) else 0.0

    # Shooting star (bearish reversal)
    is_shooting_star = 1.0 if (upper_wick > 2 * body_pct and lower_wick < 0.1 and c < o) else 0.0

    # Engulfing: current body completely covers previous body
    o_prev, c_prev = opens[-2], closes[-2]
    prev_dir = 1 if c_prev > o_prev else -1
    bull_engulf = 1.0 if (direction == 1 and prev_dir == -1
                          and o < min(o_prev, c_prev) and c > max(o_prev, c_prev)) else 0.0
    bear_engulf = 1.0 if (direction == -1 and prev_dir == 1
                          and o > max(o_prev, c_prev) and c < min(o_prev, c_prev)) else 0.0

    return [body_pct * direction, upper_wick, lower_wick,
            is_doji, is_hammer, is_shooting_star,
            bull_engulf, bear_engulf]


def compute_features(candles, idx):
    """
    Build feature vector for candle at idx.
    *** FIX: requires idx >= MIN_FEATURE_IDX (= 60 + LOOKBACK = 70) ***
    to guarantee all history slices are fully populated.
    """
    if idx < MIN_FEATURE_IDX:
        return None

    # Pull a 65-candle history window (enough for all indicators)
    hist = candles[idx - 64: idx + 1]   # 65 elements
    opens   = [float(c[1]) for c in hist]
    closes  = [float(c[4]) for c in hist]
    highs   = [float(c[2]) for c in hist]
    lows    = [float(c[3]) for c in hist]
    volumes = [float(c[5]) for c in hist]

    # ── Existing indicators ──────────────────────────────────────────────────

    # RSI (pass a slice of exactly rsi_p+1 elements — no IndexError possible)
    rsi_window = closes[-(RSI_PERIOD + 1):]
    rsi_val = _rsi(rsi_window, RSI_PERIOD)

    # Stochastic RSI
    stoch_rsi_val = _stoch_rsi(closes, STOCH_RSI_PERIOD, STOCH_RSI_PERIOD)

    # EMAs
    ema9   = _ema(closes[-EMA_SHORT - 5:], EMA_SHORT)
    ema21  = _ema(closes[-EMA_LONG  - 5:], EMA_LONG)
    ema50  = _ema(closes[-EMA_TREND - 5:], EMA_TREND)
    ema_diff_short = (ema9  - ema21) / ema21 * 100
    ema_diff_long  = (ema21 - ema50) / ema50 * 100   # new: medium vs trend

    # MACD
    ema12     = _ema(closes[-20:], 12)
    ema26     = _ema(closes[-35:], 26)
    macd_line = ema12 - ema26
    macd_pct  = macd_line / closes[-1] * 100

    # Bollinger Bands
    bb_cls   = closes[-BB_PERIOD:]
    bb_mean  = sum(bb_cls) / BB_PERIOD
    bb_std   = math.sqrt(sum((x - bb_mean) ** 2 for x in bb_cls) / BB_PERIOD)
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    bb_pos   = ((closes[-1] - bb_lower) / (bb_upper - bb_lower)
                if bb_upper != bb_lower else 0.5)
    bb_width = (bb_upper - bb_lower) / bb_mean * 100

    # ATR
    atr_val = _atr(highs[-(ATR_PERIOD + 1):], lows[-(ATR_PERIOD + 1):],
                   closes[-(ATR_PERIOD + 1):], ATR_PERIOD)
    atr_pct = atr_val / closes[-1] * 100

    # Returns
    ret1  = (closes[-1] - closes[-2])  / closes[-2]  * 100 if closes[-2]  else 0
    ret3  = (closes[-1] - closes[-4])  / closes[-4]  * 100 if len(closes) >= 4  and closes[-4]  else 0
    ret5  = (closes[-1] - closes[-6])  / closes[-6]  * 100 if len(closes) >= 6  and closes[-6]  else 0
    ret10 = (closes[-1] - closes[-11]) / closes[-11] * 100 if len(closes) >= 11 and closes[-11] else 0

    # Volume ratio
    vol_now = volumes[-1]
    vol_avg = sum(volumes[-20:-1]) / 19 if len(volumes) >= 20 else vol_now
    vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0

    # Past N candle directions
    past_dirs = [1 if float(candles[idx - i][4]) > float(candles[idx - i][1]) else 0
                 for i in range(LOOKBACK, 0, -1)]

    # Streak
    streak = 0; last_d = past_dirs[-1]
    for d in reversed(past_dirs):
        if d == last_d: streak += 1
        else: break
    streak_signed = streak if last_d == 1 else -streak

    # Price position vs recent high/low
    recent_h = max(highs[-20:])
    recent_l = min(lows[-20:])
    price_pos = ((closes[-1] - recent_l) / (recent_h - recent_l)
                 if recent_h != recent_l else 0.5)

    # ── New indicators ───────────────────────────────────────────────────────

    # CCI
    cci_val = _cci(highs[-CCI_PERIOD:], lows[-CCI_PERIOD:], closes[-CCI_PERIOD:], CCI_PERIOD)
    cci_norm = max(-3.0, min(3.0, cci_val / 100))   # clamp to ~[-3, 3]

    # Williams %R
    w_r = _williams_r(highs[-WILLIAMS_PERIOD:], lows[-WILLIAMS_PERIOD:],
                      closes[-WILLIAMS_PERIOD:], WILLIAMS_PERIOD)
    w_r_norm = (w_r + 50) / 50    # map [-100,0] → [-1, 1]

    # OBV slope (normalised)
    obv_slope = _obv_slope(closes[-(OBV_LOOKBACK + 1):],
                           volumes[-(OBV_LOOKBACK + 1):], OBV_LOOKBACK)
    obv_slope_clamped = max(-5.0, min(5.0, obv_slope))

    # Candle patterns
    pattern_feats = _candle_patterns(opens, highs, lows, closes)

    # ── Assemble ─────────────────────────────────────────────────────────────
    features = [
        # Original
        rsi_val,
        ema_diff_short,
        macd_pct,
        bb_pos,
        bb_width,
        atr_pct,
        ret1, ret3, ret5, ret10,
        vol_ratio,
        streak_signed,
        price_pos,
        # New
        stoch_rsi_val,
        ema_diff_long,
        cci_norm,
        w_r_norm,
        obv_slope_clamped,
    ] + pattern_feats + past_dirs   # 8 pattern + 10 past dirs

    return features


def feature_names():
    base = [
        "RSI", "EMA_diff_short", "MACD%", "BB_pos", "BB_width", "ATR%",
        "ret1", "ret3", "ret5", "ret10", "vol_ratio", "streak", "price_pos",
        "StochRSI", "EMA_diff_long", "CCI", "Williams%R", "OBV_slope",
        "body_dir", "upper_wick", "lower_wick",
        "is_doji", "is_hammer", "is_shooting_star", "bull_engulf", "bear_engulf",
    ]
    return base + [f"dir_t-{i}" for i in range(LOOKBACK, 0, -1)]


# ── ML MODEL ──────────────────────────────────────────────────────────────────
class MLPredictor:
    def __init__(self):
        self.candles               = collections.deque(maxlen=WINDOW_SIZE)
        self.scaler                = StandardScaler() if ML_AVAILABLE else None
        self.is_trained            = False
        self.candles_since_retrain = 0
        self.predictions_made      = 0
        self.predictions_correct   = 0
        self.last_prediction       = None
        self.last_confidence       = 0
        self.selected_features     = None   # indices kept after pruning
        self._models               = []     # list of (weight, calibrated_model)
        self.train_acc_rf          = 0
        self.train_acc_lr          = 0
        self.train_acc_xgb         = 0

    def add_candle(self, candle):
        self.candles.append(candle)
        self.candles_since_retrain += 1
        if (ML_AVAILABLE and
                len(self.candles) >= MIN_TRAIN and
                self.candles_since_retrain >= RETRAIN_EVERY):
            self._train()
            self.candles_since_retrain = 0

    def _build_dataset(self):
        candles = list(self.candles)
        X, y = [], []
        for i in range(len(candles) - 1):
            feats = compute_features(candles, i)
            label = classify(candles[i + 1])
            if feats is None or label == -1:
                continue
            X.append(feats)
            y.append(label)
        return X, y

    def _select_features(self, X_train_s, y_train):
        """
        Use Random Forest importance to drop the bottom 20% of features.
        Returns a boolean mask.
        """
        selector_rf = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=0, n_jobs=-1)
        selector_rf.fit(X_train_s, y_train)
        importances = selector_rf.feature_importances_
        threshold = sorted(importances)[int(len(importances) * 0.20)]
        mask = importances >= threshold
        kept = sum(mask)
        print(f"  🔬 Feature pruning: keeping {kept}/{len(mask)} features")
        return mask

    def _train(self):
        print(f"  🔄 Training ML model on {len(self.candles):,} candles...")
        X, y = self._build_dataset()
        if len(X) < 500:
            print("  ⚠️  Not enough clean samples yet."); return

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        # Feature selection (prune noisy features)
        self.selected_features = self._select_features(X_train_s, y_train)
        X_train_s = X_train_s[:, self.selected_features]
        X_test_s  = X_test_s[:,  self.selected_features]

        models = []

        # ── Random Forest ────────────────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
        rf_cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
        rf_cal.fit(X_train_s, y_train)
        rf_acc = accuracy_score(y_test, rf_cal.predict(X_test_s)) * 100
        self.train_acc_rf = round(rf_acc, 2)
        models.append((max(rf_acc - 50, 0.1), rf_cal))
        print(f"    RF  accuracy: {rf_acc:.2f}%")

        # ── Logistic Regression ───────────────────────────────────────────────
        lr = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        lr_cal = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
        lr_cal.fit(X_train_s, y_train)
        lr_acc = accuracy_score(y_test, lr_cal.predict(X_test_s)) * 100
        self.train_acc_lr = round(lr_acc, 2)
        models.append((max(lr_acc - 50, 0.1), lr_cal))
        print(f"    LR  accuracy: {lr_acc:.2f}%")

        # ── XGBoost (if available) ─────────────────────────────────────────
        if XGB_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            xgb_cal = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
            xgb_cal.fit(X_train_s, y_train)
            xgb_acc = accuracy_score(y_test, xgb_cal.predict(X_test_s)) * 100
            self.train_acc_xgb = round(xgb_acc, 2)
            models.append((max(xgb_acc - 50, 0.1), xgb_cal))
            print(f"    XGB accuracy: {xgb_acc:.2f}%")

        # ── Gradient Boosting (sklearn, always available) ─────────────────
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42
        )
        gb_cal = CalibratedClassifierCV(gb, method="isotonic", cv=3)
        gb_cal.fit(X_train_s, y_train)
        gb_acc = accuracy_score(y_test, gb_cal.predict(X_test_s)) * 100
        models.append((max(gb_acc - 50, 0.1), gb_cal))
        print(f"    GB  accuracy: {gb_acc:.2f}%")

        self._models   = models
        self.is_trained = True
        print(f"  ✅ Ensemble ready — {len(models)} models")

    def predict(self):
        if not self.is_trained or not ML_AVAILABLE:
            return None, 0
        candles = list(self.candles)
        feats = compute_features(candles, len(candles) - 1)
        if feats is None:
            return None, 0
        try:
            X = self.scaler.transform([feats])
            if self.selected_features is not None:
                X = X[:, self.selected_features]

            total_w    = sum(w for w, _ in self._models)
            green_prob = 0.0
            for w, m in self._models:
                prob = m.predict_proba(X)[0]
                green_prob += (prob[1] if len(prob) > 1 else 0.5) * w / total_w

            pred       = "green" if green_prob >= 0.5 else "red"
            confidence = abs(green_prob - 0.5) * 200
            return pred, round(confidence, 1)
        except Exception as e:
            print(f"  ❌ Predict error: {e}")
            return None, 0

    def record_outcome(self, actual_dir):
        if self.last_prediction and actual_dir in ("green", "red"):
            self.predictions_made += 1
            correct = self.last_prediction == actual_dir
            if correct: self.predictions_correct += 1
            return correct
        return None

    @property
    def accuracy(self):
        if not self.predictions_made: return 0.0
        return round(100 * self.predictions_correct / self.predictions_made, 2)

    @property
    def candle_count(self): return len(self.candles)

    @property
    def warmup_left(self): return max(0, MIN_TRAIN - len(self.candles))

    def quick_classify_stats(self):
        green = red = doji = 0
        for c in self.candles:
            d = classify(c)
            if d == 1:   green += 1
            elif d == 0: red   += 1
            else:        doji  += 1
        total = green + red
        return (green, red, doji,
                round(100 * green / total, 2) if total else 0,
                round(100 * red   / total, 2) if total else 0)

    def model_summary(self):
        parts = [f"RF:{self.train_acc_rf}%", f"LR:{self.train_acc_lr}%"]
        if XGB_AVAILABLE: parts.append(f"XGB:{self.train_acc_xgb}%")
        return "  ".join(parts)


# ── BACKTEST ──────────────────────────────────────────────────────────────────
def run_backtest(candles, _model):
    print("📊 Running walk-forward backtest...")
    n = len(candles)
    train_end = int(n * 0.7)

    bt_model = MLPredictor()
    for c in candles[:train_end]:
        bt_model.candles.append(c)
    bt_model._train()
    if not bt_model.is_trained:
        return None

    correct = wrong = skipped = 0
    wins_conf = {">15": [0, 0], ">20": [0, 0], ">25": [0, 0],
                 ">30": [0, 0], ">40": [0, 0], ">50": [0, 0]}
    pnl = 0.0
    trade_stake = 10.0
    win_payout  = 0.785

    for i in range(train_end, n - 1):
        bt_model.candles.append(candles[i])
        pred, conf = bt_model.predict()
        if pred is None:
            skipped += 1
            continue
        actual_dir = "green" if classify(candles[i + 1]) == 1 else "red"
        hit = pred == actual_dir

        if hit: correct += 1; pnl += trade_stake * win_payout
        else:   wrong   += 1; pnl -= trade_stake

        for thresh, bucket in wins_conf.items():
            t = float(thresh[1:])
            if conf > t:
                bucket[1] += 1
                if hit: bucket[0] += 1

    total_tested = correct + wrong
    acc = round(100 * correct / total_tested, 2) if total_tested else 0

    return {
        "train_candles": train_end,
        "test_candles":  n - train_end,
        "correct":       correct,
        "wrong":         wrong,
        "skipped":       skipped,
        "accuracy":      acc,
        "pnl":           round(pnl, 2),
        "rf_train_acc":  getattr(bt_model, "train_acc_rf",  0),
        "lr_train_acc":  getattr(bt_model, "train_acc_lr",  0),
        "xgb_train_acc": getattr(bt_model, "train_acc_xgb", 0),
        "conf_buckets":  wins_conf,
    }


def build_backtest_telegram(r, candle_count):
    if not r:
        return "❌ Backtest failed — not enough data."

    conf_lines = "\n".join(
        f"  >{thresh[1:]}%  |  {v[1]:>6,}  |  "
        f"{round(100*v[0]/v[1], 2) if v[1] else 0:.2f}%  "
        f"{'✅' if v[1] and 100*v[0]/v[1] > 51 else '❌'}"
        for thresh, v in r["conf_buckets"].items()
    )
    pnl_emoji = "🟢" if r["pnl"] >= 0 else "🔴"
    xgb_line  = f"  XGBoost       : {r['xgb_train_acc']}%\n" if XGB_AVAILABLE else ""

    return (
        f"📊 <b>ML Backtest Report — BTC/USD 5m (v2)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Total candles   : {candle_count:,}\n"
        f"🏋️  Train set       : {r['train_candles']:,} (70%)\n"
        f"🧪 Test set        : {r['test_candles']:,} (30%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 <b>Model Performance (Train)</b>\n"
        f"  Random Forest  : {r['rf_train_acc']}%\n"
        f"  Logistic Reg.  : {r['lr_train_acc']}%\n"
        f"{xgb_line}"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Backtest Results (Test Set)</b>\n"
        f"  Correct   : {r['correct']:,}\n"
        f"  Wrong     : {r['wrong']:,}\n"
        f"  Skipped   : {r['skipped']:,}\n"
        f"  <b>Accuracy  : {r['accuracy']}%</b>\n"
        f"  {pnl_emoji} P&L ($10/trade, 78.5% win): ${r['pnl']:+,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔥 <b>Confidence Bucket Accuracy</b>\n"
        f"  Thresh | Trades | Acc\n"
        f"<code>{conf_lines}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>✅ = above 51% accuracy threshold</i>"
    )


# ── COMMAND HANDLERS ──────────────────────────────────────────────────────────
shared_state = {
    "prediction": None, "confidence": 0,
    "accuracy": 0.0, "correct": 0, "total": 0,
    "green": 0, "red": 0, "green_pct": 0, "red_pct": 0,
    "price": 0, "trained": False,
    "rf_acc": 0, "lr_acc": 0, "xgb_acc": 0,
}

def handle_start(chat_id, username):
    if is_subscribed(chat_id):
        send_message(chat_id,
            "✅ <b>Already subscribed!</b>\n\nCommands: /stop  /status")
    else:
        add_subscriber(chat_id, username)
        send_message(chat_id,
            f"🎉 <b>Welcome! Subscribed to BTC ML Predictor v2!</b>\n\n"
            f"📌 Every 5 minutes you'll receive:\n"
            f"  • Last candle result ✅/❌\n"
            f"  • ML prediction (XGB + RF + LR ensemble)\n"
            f"  • Confidence score\n"
            f"  • Live accuracy tracking\n\n"
            f"⚡ Commands:\n"
            f"  /start  — Subscribe\n"
            f"  /stop   — Unsubscribe\n"
            f"  /status — Current stats\n\n"
            f"<i>Next update at the next 5m candle close 🚀</i>"
        )
        print(f"  ➕ {username} ({chat_id})")
        send_message(ADMIN_CHAT_ID,
            f"➕ <b>New subscriber:</b> {username or 'Unknown'} ({chat_id})\n"
            f"👥 Total: {subscriber_count()}")

def handle_stop(chat_id, username):
    if is_subscribed(chat_id):
        remove_subscriber(chat_id)
        send_message(chat_id, "😢 Unsubscribed. Send /start to resubscribe!")
    else:
        send_message(chat_id, "⚠️ Not subscribed. Send /start!")

def handle_status(chat_id):
    s = shared_state
    pred = s["prediction"]
    pe = "🟢" if pred == "green" else ("🔴" if pred == "red" else "⏳")
    ab = "█" * int(s["accuracy"] / 5) + "░" * (20 - int(s["accuracy"] / 5))
    if s["trained"]:
        xgb_str = f"  XGB: {s['xgb_acc']}%\n" if XGB_AVAILABLE else ""
        trained_str = f"RF: {s['rf_acc']}%  LR: {s['lr_acc']}%\n{xgb_str}"
    else:
        trained_str = "⏳ Training..."
    send_message(chat_id,
        f"📊 <b>BTC ML Predictor v2 — Live Status</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 BTC   : ${s['price']:,.2f}\n"
        f"🟢 Green : {s['green']:,} ({s['green_pct']}%)\n"
        f"🔴 Red   : {s['red']:,} ({s['red_pct']}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 Models:\n{trained_str}"
        f"🔮 Next  : {pe} <b>{(pred or 'warming up').upper()}</b> ({s['confidence']:.1f}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Accuracy: <b>{s['accuracy']}%</b>\n"
        f"[{ab}]\n"
        f"{s['correct']}/{s['total']} correct  |  👥 {subscriber_count()}"
    )

def handle_subscribers(chat_id):
    if str(chat_id) != str(ADMIN_CHAT_ID):
        send_message(chat_id, "⛔ Admin only."); return
    subs = get_all_subscribers(); count = len(subs)
    lines = "\n".join(f"  {i+1}. {u or 'Unknown'} ({cid})"
                      for i, (cid, u) in enumerate(subs[:20]))
    send_message(chat_id,
        f"👥 <b>Subscribers ({count})</b>\n<code>{lines}</code>"
        + (f"\n<i>...and {count-20} more</i>" if count > 20 else ""))

def polling_thread():
    offset = None
    print("🤖 Polling started...")
    while True:
        try:
            updates = get_updates(offset)
            for u in updates:
                offset = u["update_id"] + 1
                msg = u.get("message") or u.get("edited_message")
                if not msg: continue
                cid  = str(msg["chat"]["id"])
                name = (msg.get("from", {}).get("username") or
                        msg.get("from", {}).get("first_name", "Unknown"))
                txt  = msg.get("text", "").strip().lower().split()[0]
                if txt == "/start":         handle_start(cid, name)
                elif txt == "/stop":        handle_stop(cid, name)
                elif txt == "/status":      handle_status(cid)
                elif txt == "/subscribers": handle_subscribers(cid)
        except Exception as e:
            print(f"  ⚠️  Poll error: {e}")
        time.sleep(POLL_INTERVAL)


# ── BROADCAST MESSAGE ─────────────────────────────────────────────────────────
def build_broadcast_msg(predictor, candle, pred, conf, actual_dir, outcome):
    pe = "🟢" if pred == "green" else "🔴"
    de = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")
    oc = ("✅ Correct!" if outcome else "❌ Wrong") if outcome is not None else ""
    ab = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    g, r, doji, gp, rp = predictor.quick_classify_stats()

    return (
        f"🤖 <b>BTC/USD 5m ML Update (v2)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {ts(candle[0])}\n"
        f"💵 <b>${float(candle[4]):,.2f}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Rolling 365-Day Window\n"
        f"  🟢 Green : {g:,} ({gp}%)\n"
        f"  🔴 Red   : {r:,} ({rp}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕯 Last candle : {de} {actual_dir.upper()}  {oc}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 <b>NEXT PREDICTION</b>\n"
        f"  {pe} <b>{pred.upper()}</b>  |  Confidence: <b>{conf:.1f}%</b> "
        f"{'🔥' if conf > 15 else '〰️'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 {predictor.model_summary()}\n"
        f"🎯 Live Accuracy: <b>{predictor.accuracy}%</b>\n"
        f"[{ab}]  {predictor.predictions_correct}/{predictor.predictions_made}\n"
        f"👥 {subscriber_count()} subscribers\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>/start to subscribe • /stop to unsubscribe</i>"
    )


# ── CONSOLE DASHBOARD ─────────────────────────────────────────────────────────
def print_dashboard(predictor, candle, pred, conf, actual_dir):
    sep = "=" * 58
    g, r, doji, gp, rp = predictor.quick_classify_stats()
    bt = g + r; gb = int(40 * g / bt) if bt else 0
    ab = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    pe = "🟢" if pred == "green" else ("🔴" if pred == "red" else "⏳")
    de = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")

    print(f"\n{sep}")
    print(f"  🤖 BTC ML Predictor v2  |  {ts(candle[0])}  |  ${float(candle[4]):,.2f}")
    print(sep)
    print(f"  Window : {predictor.candle_count:,}  |  🟢{g:,}({gp}%)  🔴{r:,}({rp}%)")
    print(f"  [{'█'*gb}{'▓'*(40-gb)}]")
    trained_str = (predictor.model_summary()
                   if predictor.is_trained
                   else f"⏳ warmup {predictor.warmup_left} left")
    print(f"  Models : {trained_str}")
    print(f"  Last   : {de} {actual_dir.upper()}")
    if pred:
        print(f"  Next   : {pe} {pred.upper()}  ({conf:.1f}% confidence)")
        print(f"  Acc    : {predictor.accuracy}%  [{ab}]  ({predictor.predictions_correct}/{predictor.predictions_made})")
        print(f"  Subs   : {subscriber_count()}")
    else:
        print(f"  ⏳ Warming up... {predictor.warmup_left} more candles needed")
    print(sep)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if not ML_AVAILABLE:
        print("❌ scikit-learn missing. Run: pip install scikit-learn numpy"); return

    print("=" * 58)
    print("  BTC/USD 5m ML Predictor v2 — PUBLIC BOT")
    print("=" * 58)
    if XGB_AVAILABLE:
        print("  🚀 XGBoost enabled — full 4-model ensemble")
    else:
        print("  ℹ️  Running 3-model ensemble (RF + LR + GB)")
        print("     pip install xgboost  for best accuracy")
    print("=" * 58)

    init_db()
    add_subscriber(ADMIN_CHAT_ID, "admin")
    print(f"✅ DB ready. Subscribers: {subscriber_count()}")

    candles = fetch_historical(DAYS)

    predictor = MLPredictor()
    print("⚙️  Loading candles into model...")
    for c in candles:
        predictor.candles.append(c)

    print("🏋️  Training initial ML model (this takes ~2-3 min)...")
    predictor._train()

    send_message(ADMIN_CHAT_ID, "⏳ Running backtest... please wait ~2 min.")
    backtest_result = run_backtest(candles, predictor)
    bt_msg = build_backtest_telegram(backtest_result, len(candles))
    send_message(ADMIN_CHAT_ID, bt_msg)
    print("✅ Backtest complete & sent to Telegram.")

    pred, conf = predictor.predict()
    predictor.last_prediction = pred
    predictor.last_confidence = conf

    g, r, doji, gp, rp = predictor.quick_classify_stats()
    shared_state.update({
        "prediction": pred, "confidence": conf,
        "green": g, "red": r, "green_pct": gp, "red_pct": rp,
        "trained": predictor.is_trained,
        "rf_acc":  predictor.train_acc_rf,
        "lr_acc":  predictor.train_acc_lr,
        "xgb_acc": predictor.train_acc_xgb,
        "price": float(candles[-1][4]) if candles else 0,
    })

    send_message(ADMIN_CHAT_ID,
        f"🚀 <b>ML Predictor v2 Bot is LIVE!</b>\n\n"
        f"📊 Candles loaded : {predictor.candle_count:,}\n"
        f"🟢 Green : {g:,} ({gp}%)\n"
        f"🔴 Red   : {r:,} ({rp}%)\n"
        f"🤖 {predictor.model_summary()}\n"
        f"🔮 First prediction: {'🟢 GREEN' if pred=='green' else '🔴 RED'} ({conf:.1f}%)\n\n"
        f"👥 Subscribers: {subscriber_count()}\n"
        f"<i>Share your bot so others can /start!</i>"
    )

    threading.Thread(target=polling_thread, daemon=True).start()

    print("🔄 Entering live loop...\n")
    last_seen_time = int(candles[-1][0]) if candles else 0

    while True:
        try:
            latest      = fetch_latest_candle()
            candle_time = int(latest[0])

            if candle_time != last_seen_time:
                actual_dir = ("green" if classify(latest) == 1
                              else ("red" if classify(latest) == 0 else "doji"))
                outcome = predictor.record_outcome(actual_dir)
                predictor.add_candle(latest)
                pred, conf = predictor.predict()
                predictor.last_prediction = pred
                predictor.last_confidence = conf

                g, r, doji, gp, rp = predictor.quick_classify_stats()
                shared_state.update({
                    "prediction": pred, "confidence": conf,
                    "accuracy":   predictor.accuracy,
                    "correct":    predictor.predictions_correct,
                    "total":      predictor.predictions_made,
                    "green": g, "red": r, "green_pct": gp, "red_pct": rp,
                    "trained":  predictor.is_trained,
                    "rf_acc":   predictor.train_acc_rf,
                    "lr_acc":   predictor.train_acc_lr,
                    "xgb_acc":  predictor.train_acc_xgb,
                    "price":    float(latest[4]),
                })

                print_dashboard(predictor, latest, pred, conf, actual_dir)

                if pred:
                    msg = build_broadcast_msg(predictor, latest, pred, conf, actual_dir, outcome)
                    broadcast(msg)

                last_seen_time = candle_time

            else:
                fo = float(latest[1]); fc = float(latest[4])
                fp = ((fc - fo) / fo * 100) if fo else 0
                print(
                    f"  [{now_str()}] Forming {'🟢' if fc>fo else '🔴'} {fp:+.3f}%  |  "
                    f"Next: {'🟢' if pred=='green' else '🔴' if pred else '⏳'} "
                    f"{pred or 'warming up'} ({conf:.1f}%)",
                    end="\r"
                )

        except requests.exceptions.RequestException as e:
            print(f"\n  ⚠️  Network error: {e}")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()
