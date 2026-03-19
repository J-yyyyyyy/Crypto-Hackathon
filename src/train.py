"""
Training script for the XGBoost crypto trend prediction models.

Usage
-----
    python -m src.train

One model per symbol is trained using historical 1-hour Binance klines and
saved to the ``models/`` directory at the project root.
"""

from __future__ import annotations

import argparse
import sys

from .data_fetcher import SYMBOLS, fetch_klines
from .feature_engineering import build_features
from .model import CryptoTrendModel


def train_symbol(
    symbol: str,
    interval: str = "1h",
    limit: int = 1000,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Fetch data, build features, train and save a model for *symbol*.

    Parameters
    ----------
    symbol : str
        Trading-pair symbol (e.g. 'BTCUSDT').
    interval : str
        Binance kline interval.
    limit : int
        Number of historical candles to fetch (max 1000).
    n_splits : int
        Time-series cross-validation folds.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict
        Training metrics, e.g. ``{"symbol": ..., "oof_auc": ..., "train_accuracy": ..., "test_accuracy": ...}``.
    """
    if verbose:
        print(f"\nTraining model for {symbol} …")

    df_raw = fetch_klines(symbol, interval=interval, limit=limit)
    df_feat = build_features(df_raw, horizon=4)

    if len(df_feat) < 100:
        print(
            f"  WARNING: only {len(df_feat)} samples for {symbol}. "
            "Skipping to avoid unreliable training.",
            file=sys.stderr,
        )
        return {"symbol": symbol, "oof_auc": None, "status": "skipped"}

    model = CryptoTrendModel(symbol=symbol)
    metrics = model.train(df_feat, n_splits=n_splits, verbose=verbose)
    path = model.save()

    if verbose:
        train_acc = metrics.get("train_accuracy")
        test_acc = metrics.get("test_accuracy")
        train_acc_fmt = f"{train_acc:.4f}" if train_acc is not None else "N/A"
        test_acc_fmt = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        print(f"  Accuracy — train: {train_acc_fmt}  test: {test_acc_fmt}")

    if verbose:
        print(f"  Model saved to {path}")

    return {"symbol": symbol, **metrics, "status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for crypto trend prediction."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        help="Symbols to train (default: all 10).",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Binance kline interval (default: 1h).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of historical candles per symbol (max 1000).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Time-series CV folds (default: 5).",
    )
    args = parser.parse_args()

    results = []
    for symbol in args.symbols:
        result = train_symbol(
            symbol,
            interval=args.interval,
            limit=args.limit,
            n_splits=args.splits,
        )
        results.append(result)

    print("\n=== Training Summary ===")
    for r in results:
        auc = f"{r['oof_auc']:.4f}" if r.get("oof_auc") is not None else "N/A"
        train_acc = r.get("train_accuracy")
        test_acc = r.get("test_accuracy")
        train_acc_fmt = f"{train_acc:.4f}" if train_acc is not None else "N/A"
        test_acc_fmt = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        print(
            f"  {r['symbol']:<12} OOF AUC={auc}  "
            f"train_acc={train_acc_fmt}  test_acc={test_acc_fmt}  "
            f"status={r['status']}"
        )


if __name__ == "__main__":
    main()
