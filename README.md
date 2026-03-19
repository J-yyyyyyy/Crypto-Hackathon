# XGBoost Crypto Market Trend Prediction

A Python project that uses **XGBoost** to predict the probability of a price
rise in the next **4 hours** for 10 major cryptocurrency pairs traded on Binance.
Two label definitions are supported during training and prediction:

- **Primitive**: future close > current close
- **1% threshold**: future close is at least **1%** higher than current close

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

This fetches the last 1 000 hourly candles for each symbol, engineers features,
trains two XGBoost classifiers per symbol (primitive vs. 1% return targets) with
time-series cross-validation, a small regularization grid search, feature
importance pruning, and simple baselines for comparison. One model file per
target variant is saved to `models/`.

During training you will see the mean out-of-fold AUC, the accuracy on the
chronological training window versus a held-out test window, baseline scores,
and which parameter set won the grid search — making it easy to spot
overfitting.

Optional arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--symbols` | all 10 | Space-separated list of symbols to train |
| `--interval` | `1h` | Binance kline interval |
| `--limit` | `1000` | Number of historical candles (max 1,000) |
| `--splits` | `5` | Time-series CV folds |

### 2. Make predictions

```bash
python -m src.predict
```

Fetches the latest candles, applies feature engineering and outputs the
**probability of an upward price move** over the next 4 hours for each symbol.
Use `--target-type` to pick which trained label to load (`primitive` or
`return_1pct`).

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
| `--target-type` | `primitive` | Which trained target variant to load |

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

FEATURE_COLUMNS contains **46 total indicators** derived from OHLCV candlestick
data before any feature selection:

- **Moving averages**: SMA & EMA (7, 14, 21, 50 periods) + EMA slopes
- **RSI** (14-period) + short-term RSI slope
- **MACD** (12/26/9): line, signal, histogram + histogram slope
- **Trend-strength**: ADX (+DI / -DI)
- **Bollinger Bands**: upper/lower bands, %B, bandwidth
- **Volatility context**: ATR, ATR as % of price, realized volatility (6, 24)
- **Stochastic Oscillator** (%K, %D)
- **Volume trends**: OBV, Volume Price Trend (VPT) + VPT MA(14)
- **Price features**: 1h/4h/24h returns, candle body/wick ratios, H-L spread
- **Volume features**: volume MAs, relative volume, taker-buy ratio
- **Rate of Change** (3, 6, 12, 24 periods)
- **Stationarity helpers**: rolling z-scores for returns/volume, 24-period
  percentile rank of close

---

## Model Details

- **Algorithm**: XGBoost binary classifier (`XGBClassifier`)
- **Targets**:
  - Primitive: close price 4 candles ahead > current close
  - Threshold: forward return > **1%**
- **Regularization**: small grid search across stronger regularization options,
  early stopping, feature-importance pruning after correlation pre-filtering
- **Validation**: Time-series cross-validation (`TimeSeriesSplit`, 5 folds) +
  early stopping on a chronological hold-out slice using AUC
- **Baselines**: positive 4h return, ROC(6) momentum, EMA(7/21) crossover
- **Output**: Probability in **[0, 1]** — higher values indicate a stronger
  uptrend signal for the next 4 hours

---

## Disclaimer

This project is for **educational and research purposes only**.
Cryptocurrency markets are highly volatile and unpredictable.
Model predictions should **not** be used as financial advice or as the sole
basis for trading decisions.
