# XGBoost Crypto Market Trend Prediction

A Python project that uses **XGBoost** to predict the probability of a price
rise in the next **4 hours** for 10 major cryptocurrency pairs traded on Binance.
The training target is whether the close price 4 candles ahead is above the
current close.

## Supported Symbols

| Symbol   | Description                |
|----------|----------------------------|
| BTCUSDT  | Bitcoin / Tether           |
| ETHUSDT  | Ethereum / Tether          |
| SOLUSDT  | Solana / Tether            |
| BNBUSDT  | BNB / Tether               |
| XRPUSDT  | XRP / Tether               |
| ADAUSDT  | Cardano / Tether           |
| LINKUSDT | Chainlink / Tether         |
| AVAXUSDT | Avalanche / Tether         |
| DOGEUSDT | Dogecoin / Tether          |
| SUIUSDT  | Sui / Tether               |

---

## Project Structure

```
.
├── requirements.txt          # Python dependencies
├── models/                   # Saved model files (created after training)
└── src/
    ├── __init__.py
    ├── data_fetcher.py       # Fetches OHLCV data from the Binance public API
    ├── feature_engineering.py# Computes technical indicators / features
    ├── model.py              # XGBoost wrapper (train, save, load, predict)
    ├── train.py              # CLI training script
    └── predict.py            # CLI prediction script
```

---

## Installation

```bash
pip install -r requirements.txt
```

> No Binance API key is needed — only public market-data endpoints are used.

---

## Usage

### 1. Train the models

```bash
python -m src.train
```

This fetches ~5 000 hourly candles for each symbol (via backward pagination),
engineers features, trains an XGBoost classifier per symbol with
walk-forward cross-validation, a broader regularization grid search, feature
importance pruning, and simple baselines for comparison. One model file per
symbol is saved to `models/`.

During training you will see the mean out-of-fold AUC, accuracy, precision,
recall, F1 on the chronological training window versus a held-out test window,
baseline scores, and which parameter set won the grid search — making it easy
to spot overfitting.

Optional arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--symbols` | all 10 | Space-separated list of symbols to train |
| `--interval` | `1h` | Binance kline interval |
| `--limit` | `5000` | Number of historical candles (paginated) |
| `--splits` | `5` | Walk-forward CV folds |

### 2. Make predictions

```bash
python -m src.predict
```

Fetches the latest candles, applies feature engineering and outputs the
**probability of an upward price move** over the next 4 hours for each symbol.
The saved model reflects the primitive target (future close > current close).

```
=== 4-Hour Uptrend Probability ===
  BTCUSDT      0.6134  [████████████████████████░░░░░░░░░░░░░░░░]
  ETHUSDT      0.5821  [███████████████████████░░░░░░░░░░░░░░░░░]
  ...
```

Optional arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--symbols` | all 10 | Space-separated list of symbols |
| `--interval` | `1h` | Binance kline interval |
| `--json` | off | Output results as JSON |
| `--limit` | `200` | Candles to fetch for feature calculation |

JSON output example:

```json
{
  "predictions": {
    "BTCUSDT": 0.613412,
    "ETHUSDT": 0.582109,
    ...
  }
}
```

---

### Model focus

The repository now concentrates solely on producing robust probability
predictions. Training uses strict chronological splits, cross-validation, and
early stopping on a held-out validation window (AUC). Feature space is
regularized by correlation-based selection and rolling, stationary transforms
to reduce noise.

---

## Features

FEATURE_COLUMNS now cover **seasonality, cross-asset context, lagged momentum and
volatility-adjusted signals** alongside the stationary ratios:

- **EMA cross ratios**: ema_7 / ema_21 - 1, ema_21 / ema_50 - 1, price / ema_200
- **Momentum**: RSI(14)+slope(3), MACD histogram+slope(3), ROC(6, 24),
  lagged returns (1–12), daily change/RSI
- **Volatility**: ATR% of price, Bollinger %B/width, realized vol (24),
  24h return volatility, returns / ATR
- **Volume**: relative volume (vs. 14), volume z-score (24), taker-buy ratio
  (smoothed), OBV ROC(6), volume breakout flag, BTC volume ratio
- **Price action**: 1h/24h returns, candle body ratio, high-low spread,
  return z-score (24), close percentile rank (24)
- **Trend strength / regimes**: ADX(14), +DI / -DI difference, price over ema_200
- **Cross-asset & cycles**: BTC dominance and rolling correlation (24/48),
  hour/day-of-week plus Fourier (24h/168h)

---

## Model Details

- **Algorithm**: XGBoost binary classifier (`XGBClassifier`)
- **Target**: close price 4 candles ahead > current close
- **Regularization**: broader grid across depth/learning-rate/regularization,
  early stopping, feature-importance pruning after correlation pre-filtering
- **Validation**: Walk-forward cross-validation (5 folds) + early stopping on a
  chronological hold-out slice using AUC
- **Baselines**: positive 24h return, ROC(6) momentum, EMA(7/21) crossover ratio
- **Output**: Probability in **[0, 1]** — higher values indicate a stronger
  uptrend signal for the next 4 hours

---

## Disclaimer

This project is for **educational and research purposes only**.
Cryptocurrency markets are highly volatile and unpredictable.
Model predictions should **not** be used as financial advice or as the sole
basis for trading decisions.
