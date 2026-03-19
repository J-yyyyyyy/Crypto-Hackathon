"""
Prediction script: fetches the latest market data from Binance and outputs
the probability of a price rise over the next 4 hours for each symbol.

Usage
-----
    # After training (python -m src.train):
    python -m src.predict

    # Optionally limit to specific symbols:
    python -m src.predict --symbols BTCUSDT ETHUSDT
"""

from __future__ import annotations

import argparse
import json
import sys

from .data_fetcher import SYMBOLS, fetch_klines
from .feature_engineering import build_features
from .model import CryptoTrendModel


def predict_symbol(symbol: str, interval: str = "1h", limit: int = 200) -> float:
    """
    Load the saved model for *symbol* and return the probability of a price
    rise in the next 4 candles (≈ 4 hours with 1-h candles).

    Parameters
    ----------
    symbol : str
        Trading-pair symbol (e.g. 'BTCUSDT').
    interval : str
        Binance kline interval.
    limit : int
        Number of recent candles to fetch (only the last row is used for
        prediction; enough candles are needed for the longest rolling window).

    Returns
    -------
    float
        Probability in [0, 1].
    """
    df_raw = fetch_klines(symbol, interval=interval, limit=limit)
    btc_ref = None
    if symbol.upper() != "BTCUSDT":
        btc_ref = fetch_klines("BTCUSDT", interval=interval, limit=limit)
    df_feat = build_features(
        df_raw,
        horizon=4,
        symbol=symbol,
        interval=interval,
        limit=limit,
        btc_df=btc_ref,
    )

    model = CryptoTrendModel(symbol=symbol)
    model.load()

    return model.predict_latest(df_feat)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict the probability of a crypto price rise over the next 4 hours."
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        help="Symbols to predict (default: all 10).",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Binance kline interval (default: 1h).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of candles to fetch for feature calculation (default: 200).",
    )
    args = parser.parse_args()

    predictions: dict[str, float] = {}
    errors: dict[str, str] = {}

    for symbol in args.symbols:
        try:
            prob = predict_symbol(symbol, interval=args.interval, limit=args.limit)
            predictions[symbol] = round(prob, 6)
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            errors[symbol] = str(exc)
            print(f"  ERROR [{symbol}]: {exc}", file=sys.stderr)

    if args.json:
        output = {"predictions": predictions}
        if errors:
            output["errors"] = errors
        print(json.dumps(output, indent=2))
    else:
        print("\n=== 4-Hour Uptrend Probability ===")
        for symbol, prob in predictions.items():
            bar_len = int(prob * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  {symbol:<12} {prob:.4f}  [{bar}]")
        if errors:
            print(f"\n  {len(errors)} symbol(s) failed — see stderr for details.")


if __name__ == "__main__":
    main()
