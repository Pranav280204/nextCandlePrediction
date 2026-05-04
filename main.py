import requests
import time
import math
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import HistGradientBoostingClassifier

# ── CONFIG ─────────────────────────────────────────
SYMBOL = "BTCUSDT"
INTERVAL = "5"
BYBIT_URL = "https://api.bybit.com/v5/market/kline"

DAYS = 365
BATCH_SIZE = 1000
MS_PER_5MIN = 5 * 60 * 1000

CONF_THRESHOLD = 55
MIN_MOVE = 0.0005
FEE = 0.0006

# ── FETCH DATA ─────────────────────────────────────
def fetch_historical():
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - DAYS * 24 * 60 * 60 * 1000

    all_candles = []
    end_ms = now_ms

    print("📥 Fetching data...")

    while end_ms > start_ms:
        params = {
            "category": "spot",
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "limit": BATCH_SIZE,
            "end": end_ms,
            "start": start_ms,
        }

        r = requests.get(BYBIT_URL, params=params, timeout=10)
        data = r.json()

        batch = data["result"]["list"]
        if not batch:
            break

        all_candles.extend(batch)
        end_ms = int(batch[-1][0]) - MS_PER_5MIN

        print(f"{len(all_candles)} candles...", end="\r")
        time.sleep(0.1)

    all_candles.reverse()
    print(f"\n✅ Done: {len(all_candles)} candles")
    return all_candles


# ── FEATURE ENGINEERING ────────────────────────────
def extract_features(closes):
    if len(closes) < 50:
        return None

    returns = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
    vol = np.std(returns[-20:]) if len(returns) >= 20 else 0

    momentum = sum(1 if r > 0 else 0 for r in returns[-10:]) / 10

    return np.array([
        momentum,
        vol,
        returns[-1],
        returns[-2],
        returns[-3],
    ], dtype=np.float32)


# ── BACKTEST ENGINE ────────────────────────────────
def backtest():
    candles = fetch_historical()

    closes = []
    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05)

    X, y = [], []

    balance = 1000
    equity_curve = [balance]

    trades = wins = losses = 0

    filtered_total = 0
    filtered_correct = 0

    last_feat = None
    last_price = None
    last_pred = None
    last_conf = 0

    for i, c in enumerate(candles):
        close = float(c[4])

        # ── Prediction ──
        if last_feat is not None and len(y) > 200:
            proba = model.predict_proba([last_feat])[0]
            p_up = proba[1]

            last_pred = "green" if p_up > 0.5 else "red"
            last_conf = abs(p_up - 0.5) * 200

        # ── Trade Execution ──
        if last_pred and last_price:
            ret = (close - last_price) / last_price

            if abs(ret) > MIN_MOVE and last_conf > CONF_THRESHOLD:
                trades += 1

                pnl = ret if last_pred == "green" else -ret
                pnl -= FEE

                balance *= (1 + pnl)

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                # accuracy tracking
                filtered_total += 1
                if (last_pred == "green" and ret > 0) or (last_pred == "red" and ret < 0):
                    filtered_correct += 1

        # ── Build dataset ──
        closes.append(close)
        feat = extract_features(closes)

        if feat is not None:
            if len(closes) > 1:
                ret = (closes[-1] - closes[-2]) / closes[-2]

                # skip noise labels
                if abs(ret) > MIN_MOVE:
                    label = 1 if ret > 0 else 0
                    X.append(last_feat if last_feat is not None else feat)
                    y.append(label)

        # ── Train model ──
        if len(y) > 200:
            model.fit(X, y)

        last_feat = feat
        last_price = close
        equity_curve.append(balance)

        if i % 10000 == 0:
            print(f"{i}/{len(candles)} processed...")

    # ── RESULTS ───────────────────────────────────
    winrate = 100 * wins / trades if trades else 0
    roi = 100 * (balance - 1000) / 1000
    acc = 100 * filtered_correct / filtered_total if filtered_total else 0

    # drawdown
    peak = equity_curve[0]
    max_dd = 0
    for x in equity_curve:
        peak = max(peak, x)
        dd = (peak - x) / peak
        max_dd = max(max_dd, dd)

    print("\n💰 RESULTS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Trades        : {trades}")
    print(f"Winrate       : {winrate:.2f}%")
    print(f"Filtered Acc  : {acc:.2f}%")
    print(f"Balance       : ${balance:.2f}")
    print(f"ROI           : {roi:.2f}%")
    print(f"Max Drawdown  : {max_dd*100:.2f}%")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ── RUN ────────────────────────────────────────────
if __name__ == "__main__":
    backtest()
