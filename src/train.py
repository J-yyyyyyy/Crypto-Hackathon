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

from sklearn.metrics import accuracy_score, roc_auc_score

from .data_fetcher import SYMBOLS, fetch_klines
from .feature_engineering import build_features
from .model import CryptoTrendModel, XGBOOST_PARAMS


TARGET_CONFIGS = [
    {"name": "primitive", "target_col": "target", "variant": None},
    {"name": "return_1pct", "target_col": "target_return_1pct", "variant": "return1pct"},
]

MIN_TRAIN_SAMPLES = 100

REGULARIZATION_GRID = [
    XGBOOST_PARAMS.copy(),
    {
        **XGBOOST_PARAMS,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "min_child_weight": 6,
        "gamma": 1.2,
        "n_estimators": 600,
    },
    {
        **XGBOOST_PARAMS,
        "learning_rate": 0.03,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.8,
        "reg_lambda": 3.5,
        "max_depth": 4,
        "gamma": 1.0,
    },
]


def _safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return None


def evaluate_baselines(df_feat, target_col: str) -> dict:
    """Compare simple baselines (persistence, momentum, crossover)."""
    y_true = df_feat[target_col].values
    required_cols = {"return_4h", "roc_6", "ema_7", "ema_21"}
    if missing := required_cols - set(df_feat.columns):
        return {
            "missing_features": {
                "columns": sorted(missing),
                "accuracy": None,
                "auc": None,
            }
        }
    baselines = {
        "positive_return_4h": (df_feat["return_4h"] > 0).astype(int).values,
        "momentum_roc6": (df_feat["roc_6"] > 0).astype(int).values,
        "ema7_gt_ema21": (df_feat["ema_7"] > df_feat["ema_21"]).astype(int).values,
    }
    results: dict[str, dict[str, float | None]] = {}
    for name, preds in baselines.items():
        acc = float(accuracy_score(y_true, preds))
        auc = _safe_auc(y_true, preds)
        results[name] = {"accuracy": acc, "auc": auc}
    return results


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

    if len(df_feat) < MIN_TRAIN_SAMPLES:
        msg = (
            f"  WARNING: only {len(df_feat)} samples for {symbol}. "
            "Skipping to avoid unreliable training."
        )
        print(msg, file=sys.stderr)
        return [
            {
                "symbol": symbol,
                "target": cfg["name"],
                "status": "skipped",
                "reason": msg,
            }
            for cfg in TARGET_CONFIGS
        ]

    results = []
    for cfg in TARGET_CONFIGS:
        target_col = cfg["target_col"]
        variant = cfg["variant"]
        model = CryptoTrendModel(
            symbol=symbol,
            variant=variant,
            target_column=target_col,
            importance_threshold=0.0,
        )
        metrics = model.train(
            df_feat,
            n_splits=n_splits,
            verbose=verbose,
            target_column=target_col,
            param_grid=REGULARIZATION_GRID,
        )
        path = model.save()
        baselines = evaluate_baselines(df_feat, target_col=target_col)

        if verbose:
            train_acc = metrics.get("train_accuracy")
            test_acc = metrics.get("test_accuracy")
            train_acc_fmt = f"{train_acc:.4f}" if train_acc is not None else "N/A"
            test_acc_fmt = f"{test_acc:.4f}" if test_acc is not None else "N/A"
            print(
                f"  Accuracy ({cfg['name']}) — train: {train_acc_fmt}  test: {test_acc_fmt}"
            )
            print(f"  Feature columns used ({cfg['name']}): {len(metrics['features'])}")

        if verbose:
            print(f"  Model saved to {path}")

        results.append(
            {
                "symbol": symbol,
                "target": cfg["name"],
                "status": "ok",
                "model_path": path,
                "baselines": baselines,
                **metrics,
            }
        )

    return results


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
        results.extend(
            train_symbol(
                symbol,
                interval=args.interval,
                limit=args.limit,
                n_splits=args.splits,
            )
        )

    print("\n=== Training Summary ===")
    for r in results:
        auc = f"{r['oof_auc']:.4f}" if r.get("oof_auc") is not None else "N/A"
        train_acc = r.get("train_accuracy")
        test_acc = r.get("test_accuracy")
        train_acc_fmt = f"{train_acc:.4f}" if train_acc is not None else "N/A"
        test_acc_fmt = f"{test_acc:.4f}" if test_acc is not None else "N/A"
        print(
            f"  {r['symbol']:<10} [{r['target']:<11}] "
            f"OOF AUC={auc}  train_acc={train_acc_fmt}  test_acc={test_acc_fmt}  "
            f"status={r['status']}"
        )
        if r.get("baselines"):
            for name, stats in r["baselines"].items():
                auc_val = stats.get("auc")
                auc_fmt = f"{auc_val:.4f}" if auc_val is not None else "N/A"
                acc_val = stats.get("accuracy")
                acc_fmt = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                extra = ""
                if name == "missing_features":
                    cols = ",".join(stats.get("columns", []))
                    if cols:
                        extra = f" (missing: {cols})"
                print(
                    f"      baseline {name:<16} acc={acc_fmt} auc={auc_fmt}{extra}"
                )


if __name__ == "__main__":
    main()
