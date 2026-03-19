"""
Backtest module for the XGBoost crypto trend predictor.

Performs an out-of-sample evaluation by:

  1. Splitting historical data chronologically into a training window and a
     test window (no look-ahead bias).
  2. Training an XGBoost model on the training window with time-series CV.
  3. Generating predictions on the held-out test window.
  4. Reporting two complementary views of model effectiveness:

     * **Signal quality** — AUC, accuracy, precision, recall, F1.  These show
       how often the predicted direction is correct independent of any trading
       strategy.

     * **Trading simulation** — a simple long-only strategy that enters at the
       close price when the predicted probability exceeds *threshold* and exits
       *horizon* candles later.  Metrics include win rate, total return,
       buy-and-hold return, maximum drawdown and annualised Sharpe ratio.

  The side-by-side display of Training OOF AUC vs. Test AUC quantifies how
  well the model generalises from the training window to unseen data, making
  over-fitting visible at a glance.

Usage
-----
    python -m src.backtest
    python -m src.backtest --symbols BTCUSDT ETHUSDT --threshold 0.55
    python -m src.backtest --train-ratio 0.7 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data_fetcher import SYMBOLS, fetch_klines
from .feature_engineering import build_features
from .model import CryptoTrendModel


# ---------------------------------------------------------------------------
# Trading simulation helpers
# ---------------------------------------------------------------------------

def _simulate_trades(
    close: pd.Series,
    probas: np.ndarray,
    threshold: float = 0.55,
    horizon: int = 4,
) -> pd.DataFrame:
    """
    Long-only, signal-based strategy (no leverage, no short selling).

    Enter a position at ``close[i]`` when ``probas[i] >= threshold``; exit at
    ``close[i + horizon]``.  Trades are non-overlapping: after entering, the
    cursor advances by *horizon* bars so no two trades share the same candles.

    Parameters
    ----------
    close : pd.Series
        Close prices for the evaluation window, indexed by datetime.
    probas : np.ndarray
        Model-predicted probability of an upward move for each bar in *close*.
    threshold : float
        Minimum probability required to trigger a trade.
    horizon : int
        Holding period in candles.

    Returns
    -------
    pd.DataFrame
        One row per trade with columns: entry_time, exit_time, entry_price,
        exit_price, pct_return, correct.
    """
    n = len(close)
    records: list[dict[str, Any]] = []
    i = 0
    while i < n - horizon:
        if probas[i] >= threshold:
            entry_price = close.iloc[i]
            exit_price = close.iloc[i + horizon]
            pct = (exit_price - entry_price) / entry_price
            records.append(
                {
                    "entry_time": close.index[i],
                    "exit_time": close.index[i + horizon],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pct_return": pct,
                    "correct": bool(pct > 0),
                }
            )
            i += horizon  # skip to after exit to avoid overlapping trades
        else:
            i += 1
    return pd.DataFrame(records)


def _equity_curve(trades: pd.DataFrame) -> np.ndarray:
    """Cumulative equity (starting at 1.0) updated after each closed trade."""
    capital = 1.0
    curve = [capital]
    for pct in trades["pct_return"]:
        capital *= 1.0 + pct
        curve.append(capital)
    return np.array(curve)


def _max_drawdown(equity: np.ndarray) -> float:
    """Worst peak-to-trough decline as a fraction (e.g. -0.15 means –15 %)."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def _sharpe_ratio(
    returns: np.ndarray,
    horizon: int = 4,
    candles_per_year: int = 8766,
) -> float:
    """
    Annualised Sharpe ratio from per-trade percentage returns.

    With 1-h candles there are ~8 766 candles per year; each trade covers
    *horizon* candles, giving ``trades_per_year ≈ 8766 / horizon``.
    """
    if len(returns) < 2:
        return float("nan")
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return float("nan")
    mean = float(np.mean(returns))
    trades_per_year = candles_per_year / horizon
    return float((mean / std) * np.sqrt(trades_per_year))


# ---------------------------------------------------------------------------
# Metric aggregators
# ---------------------------------------------------------------------------

def compute_signal_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.55,
) -> dict[str, float]:
    """
    Classification / signal-quality metrics for the test period.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (1 = price rose, 0 = did not).
    y_proba : np.ndarray
        Model-predicted probability of a price rise for each bar.
    threshold : float
        Decision boundary used to convert probabilities to class labels.

    Returns
    -------
    dict
        Keys: auc, accuracy, precision, recall, f1.
    """
    y_pred = (y_proba >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_trading_metrics(
    trades: pd.DataFrame,
    buy_hold_return: float,
    horizon: int = 4,
) -> dict[str, float]:
    """
    Aggregate trading-simulation performance over the test period.

    Parameters
    ----------
    trades : pd.DataFrame
        Output of :func:`_simulate_trades`.
    buy_hold_return : float
        Percentage return from holding the asset over the full test window.
    horizon : int
        Holding period used in the simulation (needed for Sharpe calculation).

    Returns
    -------
    dict
        Keys: num_trades, win_rate, total_return, buy_hold_return,
        max_drawdown, sharpe_ratio.
    """
    if trades.empty:
        return {
            "num_trades": 0,
            "win_rate": float("nan"),
            "total_return": float("nan"),
            "buy_hold_return": float(buy_hold_return),
            "max_drawdown": float("nan"),
            "sharpe_ratio": float("nan"),
        }
    equity = _equity_curve(trades)
    return {
        "num_trades": len(trades),
        "win_rate": float(trades["correct"].mean()),
        "total_return": float(equity[-1] - 1.0),
        "buy_hold_return": float(buy_hold_return),
        "max_drawdown": float(_max_drawdown(equity)),
        "sharpe_ratio": _sharpe_ratio(trades["pct_return"].values, horizon=horizon),
    }


# ---------------------------------------------------------------------------
# Main backtest function
# ---------------------------------------------------------------------------

def backtest_symbol(
    symbol: str,
    df_raw: pd.DataFrame | None = None,
    interval: str = "1h",
    limit: int = 1000,
    train_ratio: float = 0.7,
    threshold: float = 0.55,
    horizon: int = 4,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Run a chronological train/test backtest for *symbol*.

    The function splits the available data at ``train_ratio``, trains an
    XGBoost model on the earlier portion (reporting out-of-fold AUC via
    time-series cross-validation) and then evaluates signal quality and a
    simple trading strategy on the later, unseen portion.

    Parameters
    ----------
    symbol : str
        Trading-pair symbol (e.g. ``'BTCUSDT'``).
    df_raw : pd.DataFrame, optional
        Pre-fetched raw OHLCV DataFrame.  Fetched from Binance when ``None``.
    interval : str
        Binance kline interval (default ``'1h'``).
    limit : int
        Number of candles to fetch (max 1,000).
    train_ratio : float
        Fraction of rows allocated to training (default 0.70).
    threshold : float
        Minimum predicted probability to trigger a long trade (default 0.55).
    horizon : int
        Prediction and holding horizon in candles (default 4).
    n_splits : int
        Time-series CV folds used during training (default 5).
    verbose : bool
        Print fold-level progress messages.

    Returns
    -------
    dict
        symbol, train_samples, test_samples, train_auc,
        signal_metrics, trading_metrics, trades_df.
    """
    if df_raw is None:
        if verbose:
            print(f"  Fetching {limit} {interval} candles for {symbol} …")
        df_raw = fetch_klines(symbol, interval=interval, limit=limit)

    # Preserve close prices before build_features drops the OHLCV columns.
    close_all = df_raw["close"].copy()

    df_feat = build_features(df_raw, horizon=horizon)

    if len(df_feat) < 100:
        raise ValueError(
            f"{symbol}: only {len(df_feat)} rows after feature engineering; "
            "backtest requires at least 100 rows."
        )

    # Chronological split — no shuffling to avoid look-ahead bias.
    split_idx = int(len(df_feat) * train_ratio)
    train_feat = df_feat.iloc[:split_idx]
    test_feat = df_feat.iloc[split_idx:]

    if len(train_feat) < 50 or len(test_feat) < 20:
        raise ValueError(
            f"{symbol}: train ({len(train_feat)}) or test ({len(test_feat)}) "
            "split is too small — adjust train_ratio or fetch more data."
        )

    # Close prices aligned with the test window (for trading simulation).
    test_close = close_all.loc[test_feat.index]

    # ----- Training -----
    model = CryptoTrendModel(symbol)
    train_result = model.train(train_feat, n_splits=n_splits, verbose=verbose)
    train_auc = train_result["oof_auc"]

    # ----- Out-of-sample prediction -----
    test_proba = model.predict_proba(test_feat)
    y_test = test_feat["target"].values

    # ----- Signal quality -----
    signal_metrics = compute_signal_metrics(y_test, test_proba, threshold=threshold)

    # ----- Trading simulation -----
    buy_hold_return = float(
        (test_close.iloc[-1] - test_close.iloc[0]) / test_close.iloc[0]
    )
    trades = _simulate_trades(
        test_close, test_proba, threshold=threshold, horizon=horizon
    )
    trading_metrics = compute_trading_metrics(
        trades, buy_hold_return, horizon=horizon
    )

    return {
        "symbol": symbol,
        "train_samples": len(train_feat),
        "test_samples": len(test_feat),
        "train_auc": train_auc,
        "signal_metrics": signal_metrics,
        "trading_metrics": trading_metrics,
        "trades_df": trades,
    }


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    try:
        return "N/A" if np.isnan(v) else f"{v * 100:+.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_f4(v: float) -> str:
    try:
        return "N/A" if np.isnan(v) else f"{v:.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_f2(v: float) -> str:
    try:
        return "N/A" if np.isnan(v) else f"{v:.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _print_result(result: dict, threshold: float, horizon: int) -> None:
    sym = result["symbol"]
    sm = result["signal_metrics"]
    tm = result["trading_metrics"]

    win_str = (
        f"{tm['win_rate'] * 100:.1f}%"
        if not np.isnan(tm["win_rate"])
        else "N/A"
    )

    print(f"\n{'=' * 52}")
    print(f"  Backtest: {sym}")
    print(f"{'=' * 52}")
    print(
        f"  Period      : train {result['train_samples']} bars / "
        f"test {result['test_samples']} bars"
    )
    print(
        f"  Train OOF AUC : {result['train_auc']:.4f}"
        f"   │   Test AUC : {_fmt_f4(sm['auc'])}"
    )
    print()
    print(f"  Signal Quality (test period, threshold={threshold}):")
    print(f"    Accuracy  : {_fmt_f4(sm['accuracy'])}")
    print(f"    Precision : {_fmt_f4(sm['precision'])}")
    print(f"    Recall    : {_fmt_f4(sm['recall'])}")
    print(f"    F1-Score  : {_fmt_f4(sm['f1'])}")
    print()
    print(
        f"  Trading Simulation "
        f"(long-only, threshold={threshold}, hold={horizon}h, no fees):"
    )
    print(f"    Trades       : {tm['num_trades']}")
    print(f"    Win Rate     : {win_str}")
    print(f"    Total Return : {_fmt_pct(tm['total_return'])}")
    print(f"    Buy & Hold   : {_fmt_pct(tm['buy_hold_return'])}")
    print(f"    Max Drawdown : {_fmt_pct(tm['max_drawdown'])}")
    print(f"    Sharpe Ratio : {_fmt_f2(tm['sharpe_ratio'])}")


def _print_summary(results: list[dict]) -> None:
    print("\n\n=== Backtest Summary ===")
    header = (
        f"  {'Symbol':<12} {'TrainAUC':>9} {'TestAUC':>8} "
        f"{'#Trades':>8} {'Win%':>6} {'Return':>8} {'B&H':>8} {'Sharpe':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        sm = r["signal_metrics"]
        tm = r["trading_metrics"]
        win = (
            f"{tm['win_rate'] * 100:.1f}"
            if not np.isnan(tm["win_rate"])
            else "N/A"
        )
        ret = (
            f"{tm['total_return'] * 100:+.2f}"
            if not np.isnan(tm["total_return"])
            else "N/A"
        )
        bh = (
            f"{tm['buy_hold_return'] * 100:+.2f}"
            if not np.isnan(tm["buy_hold_return"])
            else "N/A"
        )
        sr = _fmt_f2(tm["sharpe_ratio"])
        print(
            f"  {r['symbol']:<12} {r['train_auc']:>9.4f} {sm['auc']:>8.4f} "
            f"{tm['num_trades']:>8} {win:>6} {ret:>8} {bh:>8} {sr:>7}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest the XGBoost crypto trend model on historical data. "
            "Trains on the first train-ratio fraction of candles and evaluates "
            "signal quality + a simulated long-only trading strategy on the rest."
        )
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        help="Symbols to backtest (default: all 10).",
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
        help="Historical candles per symbol, max 1,000 (default: 1000).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data used for training (default: 0.7).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Probability threshold to trigger a long trade (default: 0.55).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Time-series CV folds during training (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output all results as a single JSON document.",
    )
    args = parser.parse_args()

    all_results: list[dict] = []
    errors: dict[str, str] = {}

    for symbol in args.symbols:
        try:
            if not args.json:
                print(f"\nBacktesting {symbol} …")
            result = backtest_symbol(
                symbol,
                interval=args.interval,
                limit=args.limit,
                train_ratio=args.train_ratio,
                threshold=args.threshold,
                n_splits=args.splits,
                verbose=not args.json,
            )
            all_results.append(result)
            if not args.json:
                _print_result(result, threshold=args.threshold, horizon=4)
        except (ValueError, RuntimeError, KeyError) as exc:
            errors[symbol] = str(exc)
            print(f"  ERROR [{symbol}]: {exc}", file=sys.stderr)

    if args.json:
        output: list[dict] = []
        for r in all_results:
            # Exclude non-serialisable trades_df from JSON output.
            output.append(
                {
                    "symbol": r["symbol"],
                    "train_samples": r["train_samples"],
                    "test_samples": r["test_samples"],
                    "train_auc": r["train_auc"],
                    "signal_metrics": r["signal_metrics"],
                    "trading_metrics": {
                        k: (None if isinstance(v, float) and np.isnan(v) else v)
                        for k, v in r["trading_metrics"].items()
                    },
                }
            )
        print(json.dumps({"results": output, "errors": errors}, indent=2))
    else:
        if all_results:
            _print_summary(all_results)
        if errors:
            print(f"\n  {len(errors)} symbol(s) failed — see stderr for details.")


if __name__ == "__main__":
    main()
