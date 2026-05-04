"""
BTC/USD 5m Rolling Candle Predictor + Telegram Alerts
─────────────────────────────────────────────────────────────
- Fetches last 365 days of BTC/USD 5m candles (Bybit API)
- Maintains a rolling window: adds newest, drops oldest each candle
- After 1000 candles observed, starts predicting next candle direction
- Uses multiple ML signals that update dynamically:
    1. Momentum     — recent green/red ratio
    2. Streak bias  — consecutive candle patterns
    3. Markov chain — transition probabilities (green→?, red→?)
    4. EMA ratio    — price above/below exponential moving average
    5. RSI          — overbought/oversold signal
- Tracks prediction accuracy live
- Sends Telegram alert every new candle with prediction
"""

import requests
import time
import os
import collections
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "8766778348:AAEpkHO55y_oCrJ0vrTwtXsm8cWE_4IOZxA")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "5792224870")

SYMBOL          = "BTCUSDT"
INTERVAL        = "5"
BYBIT_URL       = "https://api.bybit.com/v5/market/kline"
DAYS            = 365
BATCH_SIZE      = 1000
MS_PER_5MIN     = 5 * 60 * 1000

WINDOW_SIZE     = 365 * 24 * 12
MIN_CANDLES     = 1000
MOMENTUM_WINDOW = 50
STREAK_WINDOW   = 10
EMA_PERIOD      = 20
RSI_PERIOD      = 14
# ─────────────────────────────────────────────────────────────────────────────


def ts(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def now_str():
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


# ── TELEGRAM ──────────────────────────────────────────────────────────────────
def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       text,
            "parse_mode": "HTML",
        }, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"  ❌ Telegram error: {e}")


def build_telegram_message(predictor, candle, prediction, confidence, signals, actual_dir, outcome):
    pred_emoji = "🟢" if prediction == "green" else "🔴"
    dir_emoji  = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")
    outcome_str = ""
    if outcome is not None:
        outcome_str = "✅ Correct!" if outcome else "❌ Wrong"

    signal_lines = "\n".join(
        f"  • <b>{name}</b>: {val}" for name, val in signals.items()
    )

    acc_bar_filled = int(predictor.accuracy / 5)
    acc_bar = "█" * acc_bar_filled + "░" * (20 - acc_bar_filled)

    return (
        f"🤖 <b>BTC/USD 5m Candle Update</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 <b>Time  :</b> {ts(candle[0])}\n"
        f"💵 <b>Price :</b> ${float(candle[4]):,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Rolling Window ({predictor.candle_count:,} candles)</b>\n"
        f"  🟢 Green : {predictor.total_green:,} ({predictor.green_pct}%)\n"
        f"  🔴 Red   : {predictor.total_red:,} ({predictor.red_pct}%)\n"
        f"  ⚪ Doji  : {predictor.total_doji:,}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🕯 <b>Last Candle :</b> {dir_emoji} {actual_dir.upper()}  {outcome_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 <b>NEXT CANDLE PREDICTION</b>\n"
        f"  Direction  : {pred_emoji} <b>{prediction.upper()}</b>\n"
        f"  Confidence : <b>{confidence:.1f}%</b> {'🔥' if confidence > 15 else '〰️'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📡 <b>Signals:</b>\n"
        f"{signal_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Accuracy : {predictor.accuracy}%</b>  [{acc_bar}]\n"
        f"   Correct/Total : {predictor.predictions_correct}/{predictor.predictions_made}"
    )


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
    params = {
        "category": "spot",
        "symbol":   symbol,
        "interval": interval,
        "limit":    3,
    }
    resp = requests.get(BYBIT_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise Exception(f"Bybit error: {data.get('retMsg')}")
    candles = list(reversed(data["result"]["list"]))
    return candles[-2]


# ── CLASSIFY ──────────────────────────────────────────────────────────────────
def classify(candle):
    o  = float(candle[1])
    cl = float(candle[4])
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
        if diff >= 0:
            gains.append(diff); losses.append(0)
        else:
            gains.append(0); losses.append(abs(diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


# ── PREDICTOR ─────────────────────────────────────────────────────────────────
class CandlePredictor:
    def __init__(self):
        self.window      = collections.deque(maxlen=WINDOW_SIZE)
        self.closes      = collections.deque(maxlen=WINDOW_SIZE)
        self.total_green = 0
        self.total_red   = 0
        self.total_doji  = 0
        self.markov = {
            "green": {"green": 0, "red": 0},
            "red":   {"green": 0, "red": 0},
            "doji":  {"green": 0, "red": 0},
        }
        self.predictions_made    = 0
        self.predictions_correct = 0
        self.last_prediction     = None
        self.last_candle_time    = None

    def add_candle(self, candle):
        direction   = classify(candle)
        open_time   = int(candle[0])
        close_price = float(candle[4])

        if self.window:
            prev_dir = self.window[-1][1]
            if prev_dir in self.markov and direction in ("green", "red"):
                self.markov[prev_dir][direction] += 1

        if len(self.window) == self.window.maxlen:
            oldest_dir = self.window[0][1]
            if oldest_dir == "green": self.total_green -= 1
            elif oldest_dir == "red": self.total_red   -= 1
            else:                     self.total_doji  -= 1

        self.window.append((open_time, direction, close_price))
        self.closes.append(close_price)

        if direction == "green": self.total_green += 1
        elif direction == "red": self.total_red   += 1
        else:                    self.total_doji  += 1

        self.last_candle_time = open_time
        return direction

    def predict_next(self):
        if len(self.window) < MIN_CANDLES:
            return None, 0, {}

        signals      = {}
        green_score  = 0.0
        total_weight = 0.0

        # Signal 1: Momentum
        recent  = list(self.window)[-MOMENTUM_WINDOW:]
        r_green = sum(1 for _, d, _ in recent if d == "green")
        r_red   = sum(1 for _, d, _ in recent if d == "red")
        r_total = r_green + r_red
        if r_total > 0:
            mom_score = r_green / r_total
            weight    = 1.5
            green_score  += mom_score * weight
            total_weight += weight
            signals["Momentum(50)"] = f"{'🟢' if mom_score > 0.5 else '🔴'} {mom_score*100:.1f}% green"

        # Signal 2: Streak bias
        streak_candles = [d for _, d, _ in list(self.window)[-STREAK_WINDOW:]]
        last_dir   = streak_candles[-1] if streak_candles else "doji"
        streak_len = 0
        for d in reversed(streak_candles):
            if d == last_dir: streak_len += 1
            else: break

        if last_dir != "doji" and streak_len >= 2:
            reversal_weight = min(streak_len / 6, 1.0)
            streak_score = (0.5 - reversal_weight * 0.35) if last_dir == "green" else (0.5 + reversal_weight * 0.35)
            weight = 1.2
            green_score  += streak_score * weight
            total_weight += weight
            signals["Streak"] = f"{'🟢' if last_dir == 'green' else '🔴'} {streak_len}x {last_dir} → {'reversal bias' if reversal_weight > 0.3 else 'continuation'}"

        # Signal 3: Markov chain
        if last_dir in self.markov:
            m = self.markov[last_dir]
            m_total = m["green"] + m["red"]
            if m_total > 10:
                markov_score = m["green"] / m_total
                weight       = 2.0
                green_score  += markov_score * weight
                total_weight += weight
                signals["Markov"] = f"After {last_dir}: 🟢{m['green']} / 🔴{m['red']} ({markov_score*100:.1f}% green)"

        # Signal 4: EMA
        closes_list = list(self.closes)
        ema = compute_ema(closes_list, EMA_PERIOD)
        if ema and closes_list:
            current_price = closes_list[-1]
            ema_score     = 0.6 if current_price > ema else 0.4
            weight        = 1.0
            green_score  += ema_score * weight
            total_weight += weight
            diff_pct = ((current_price - ema) / ema) * 100
            signals["EMA(20)"] = f"Price {'above' if current_price > ema else 'below'} EMA ({diff_pct:+.3f}%)"

        # Signal 5: RSI
        rsi = compute_rsi(closes_list, RSI_PERIOD)
        if rsi is not None:
            rsi_score = 0.3 if rsi > 70 else (0.7 if rsi < 30 else 0.5)
            weight    = 1.3
            green_score  += rsi_score * weight
            total_weight += weight
            signals["RSI(14)"] = f"{rsi:.1f} ({'overbought🔴' if rsi > 70 else 'oversold🟢' if rsi < 30 else 'neutral⚪'})"

        if total_weight == 0:
            return None, 0, {}

        final_score = green_score / total_weight
        prediction  = "green" if final_score >= 0.5 else "red"
        confidence  = abs(final_score - 0.5) * 200

        return prediction, round(confidence, 1), signals

    def record_outcome(self, actual_direction):
        if self.last_prediction and actual_direction in ("green", "red"):
            self.predictions_made += 1
            correct = self.last_prediction == actual_direction
            if correct:
                self.predictions_correct += 1
            return correct
        return None

    @property
    def accuracy(self):
        if self.predictions_made == 0: return 0.0
        return round(100 * self.predictions_correct / self.predictions_made, 2)

    @property
    def candle_count(self): return len(self.window)

    @property
    def green_pct(self):
        total = self.total_green + self.total_red
        return round(100 * self.total_green / total, 2) if total else 0

    @property
    def red_pct(self):
        total = self.total_green + self.total_red
        return round(100 * self.total_red / total, 2) if total else 0


# ── CONSOLE DASHBOARD ─────────────────────────────────────────────────────────
def print_dashboard(predictor, candle, prediction, confidence, signals, actual_dir):
    sep  = "=" * 58
    sep2 = "-" * 58
    pred_emoji = "🟢" if prediction == "green" else ("🔴" if prediction == "red" else "⏳")
    dir_emoji  = "🟢" if actual_dir == "green" else ("🔴" if actual_dir == "red" else "⚪")
    acc_bar = "█" * int(predictor.accuracy / 5) + "░" * (20 - int(predictor.accuracy / 5))
    g, r = predictor.total_green, predictor.total_red
    bar_total = g + r
    g_bar = int(40 * g / bar_total) if bar_total else 0

    print(f"\n{sep}")
    print(f"  🤖 BTC/USD 5m Rolling Candle Predictor")
    print(f"  🕐 {ts(candle[0])}   |   BTC: ${float(candle[4]):,.2f}")
    print(sep)
    print(f"  📊 Window  : {predictor.candle_count:,} candles")
    print(f"  🟢 Green   : {g:,} ({predictor.green_pct}%)")
    print(f"  🔴 Red     : {r:,} ({predictor.red_pct}%)")
    print(f"  ⚪ Doji    : {predictor.total_doji:,}")
    print(f"  [{'█' * g_bar}{'▓' * (40-g_bar)}]")
    print(sep2)
    print(f"  Last candle : {dir_emoji} {actual_dir.upper()}")
    if prediction:
        print(sep2)
        print(f"  🔮 NEXT: {pred_emoji} {prediction.upper()}  |  Confidence: {confidence:.1f}%")
        for name, val in signals.items():
            print(f"     {name:<16}: {val}")
        print(sep2)
        print(f"  🎯 Accuracy: {predictor.accuracy}%  [{acc_bar}]  ({predictor.predictions_correct}/{predictor.predictions_made})")
    else:
        print(f"\n  ⏳ Warming up… {MIN_CANDLES - predictor.candle_count:,} more candles needed.")
    print(sep)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  BTC/USD 5m Rolling Candle Predictor + Telegram")
    print("=" * 58)

    historical = fetch_historical(SYMBOL, INTERVAL, DAYS)

    predictor = CandlePredictor()
    print("⚙️  Building rolling window...")
    for candle in historical:
        predictor.add_candle(candle)

    print(f"✅ Window built: {predictor.candle_count:,} candles | "
          f"🟢 {predictor.total_green:,} | 🔴 {predictor.total_red:,}\n")

    prediction, confidence, signals = predictor.predict_next()
    predictor.last_prediction = prediction

    # Send startup message
    send_telegram(
        f"🚀 <b>BTC Candle Predictor is LIVE!</b>\n\n"
        f"📊 Loaded <b>{predictor.candle_count:,}</b> candles (365 days)\n"
        f"🟢 Green: {predictor.total_green:,} ({predictor.green_pct}%)\n"
        f"🔴 Red  : {predictor.total_red:,} ({predictor.red_pct}%)\n\n"
        f"🔮 First prediction: {'🟢 GREEN' if prediction == 'green' else '🔴 RED'} "
        f"({confidence:.1f}% confidence)\n\n"
        f"<i>You'll receive an update every new 5m candle!</i>"
    )

    print("🔄 Entering live loop...\n")
    last_seen_time = int(historical[-1][0]) if historical else 0

    while True:
        try:
            latest      = fetch_latest_candle(SYMBOL, INTERVAL)
            candle_time = int(latest[0])

            if candle_time != last_seen_time:
                actual_dir = classify(latest)

                # Score previous prediction
                outcome = predictor.record_outcome(actual_dir)

                # Update rolling window
                predictor.add_candle(latest)

                # New prediction
                prediction, confidence, signals = predictor.predict_next()
                predictor.last_prediction = prediction

                # Console dashboard
                print_dashboard(predictor, latest, prediction, confidence, signals, actual_dir)

                # Telegram alert (only when predictions are active)
                if prediction:
                    msg = build_telegram_message(
                        predictor, latest, prediction, confidence,
                        signals, actual_dir, outcome
                    )
                    send_telegram(msg)

                last_seen_time = candle_time

            else:
                forming_open  = float(latest[1])
                forming_close = float(latest[4])
                forming_pct   = ((forming_close - forming_open) / forming_open * 100) if forming_open else 0
                forming_dir   = "🟢" if forming_close > forming_open else "🔴"
                print(
                    f"  [{now_str()}] Forming... {forming_dir} {forming_pct:+.3f}%  |  "
                    f"Prediction: {'🟢' if prediction == 'green' else '🔴' if prediction else '⏳'} "
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