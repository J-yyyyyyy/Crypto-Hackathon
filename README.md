# XGBoost Crypto Market Trend Prediction

A Python project that uses **XGBoost** to predict the probability of a price
rise in the next **4 hours** for 10 major cryptocurrency pairs traded on Binance.

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
    ├── predict.py            # CLI prediction script
    └── backtest.py           # CLI backtest / evaluation script
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
trains an XGBoost classifier with time-series cross-validation and saves one
model file per symbol to `models/`.

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

## Backtesting

The backtest module evaluates prediction effectiveness by training on the
earlier portion of the historical data and testing on the later, unseen
portion — with **no look-ahead bias**.

### 3. Run a backtest

```bash
python -m src.backtest
```

For each symbol the backtest:

1. Fetches 1 000 hourly candles.
2. Splits the data chronologically (default 70 % train / 30 % test).
3. Trains an XGBoost model on the training window using time-series CV.
4. Predicts on the held-out test window.
5. Reports:
   - **Training improvement**: side-by-side OOF AUC (training) vs. test AUC.
     A test AUC consistently above 0.50 confirms the model generalises to
     unseen data rather than memorising the training set.
   - **Signal quality**: accuracy, precision, recall, F1.
   - **Trading simulation**: a simple long-only strategy that enters at the
     close price when the predicted probability exceeds the threshold and
     holds for 4 hours.  Reports win rate, cumulative return vs. buy-and-hold,
     maximum drawdown and annualised Sharpe ratio.

Example output:

```
====================================================
  Backtest: BTCUSDT
====================================================
  Period      : train 659 bars / test 283 bars
  Train OOF AUC : 0.5812   │   Test AUC : 0.5634

  Signal Quality (test period, threshold=0.55):
    Accuracy  : 0.5406
    Precision : 0.5714
    Recall    : 0.4512
    F1-Score  : 0.5040

  Trading Simulation (long-only, threshold=0.55, hold=4h, no fees):
    Trades       : 31
    Win Rate     : 58.1%
    Total Return : +4.92%
    Buy & Hold   : +3.17%
    Max Drawdown : -5.43%
    Sharpe Ratio : 1.18
```

Optional arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--symbols` | all 10 | Space-separated list of symbols |
| `--interval` | `1h` | Binance kline interval |
| `--limit` | `1000` | Number of historical candles (max 1,000) |
| `--train-ratio` | `0.7` | Fraction of data used for training |
| `--threshold` | `0.55` | Minimum probability to trigger a trade |
| `--splits` | `5` | Time-series CV folds during training |
| `--json` | off | Output all results as JSON |

---

## Features

The model uses **35 technical indicators** derived from OHLCV candlestick data:

- **Moving averages**: SMA & EMA (7, 14, 21, 50 periods)
- **RSI** (14-period)
- **MACD** (12/26/9): line, signal, histogram
- **Bollinger Bands**: upper/lower bands, %B, bandwidth
- **ATR** (14-period) — volatility measure
- **Stochastic Oscillator** (%K, %D)
- **On-Balance Volume** (OBV)
- **Price features**: 1h/4h/24h returns, candle body/wick ratios, H-L spread
- **Volume features**: volume MAs, relative volume, taker-buy ratio
- **Rate of Change** (3, 6, 12, 24 periods)

---

## Model Details

- **Algorithm**: XGBoost binary classifier (`XGBClassifier`)
- **Target**: 1 if the close price 4 candles ahead > current close; 0 otherwise
- **Validation**: Time-series cross-validation (`TimeSeriesSplit`, 5 folds)
- **Output**: Probability in **[0, 1]** — higher values indicate a stronger
  uptrend signal for the next 4 hours

---

## Disclaimer

This project is for **educational and research purposes only**.
Cryptocurrency markets are highly volatile and unpredictable.
Model predictions should **not** be used as financial advice or as the sole
basis for trading decisions.
