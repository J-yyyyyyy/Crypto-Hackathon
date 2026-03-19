"""
Feature engineering for the XGBoost crypto trend prediction model.

All features are computed from OHLCV candlestick data and represent
standard technical analysis indicators. The target label indicates
whether the closing price 4 candles ahead (≈ 4 hours when using 1-h
candles) is higher than the current closing price.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual indicator helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Simple and exponential moving averages (7, 14, 21, 50 periods)."""
    for period in [7, 14, 21, 50]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"ema_{period}"] = _ema(df["close"], period)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = _ema(df["macd_line"], 9)
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Bollinger Bands and the position of close within the bands."""
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    band_width = df["bb_upper"] - df["bb_lower"]
    df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / band_width.replace(0, np.nan)
    df["bb_width"] = band_width / sma.replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range — a measure of volatility."""
    high_low = df["high"] - df["low"]
    high_pc = (df["high"] - df["close"].shift(1)).abs()
    low_pc = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    df["atr_14"] = true_range.ewm(com=period - 1, adjust=False).mean()
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D)."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low_min) / denom
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return, candle body/wick ratios, and high-low spread."""
    df["return_1h"] = df["close"].pct_change(1)
    df["return_4h"] = df["close"].pct_change(4)
    df["return_24h"] = df["close"].pct_change(24)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / candle_range
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / candle_range
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / candle_range
    df["hl_spread"] = candle_range / df["close"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume moving averages and relative volume."""
    df["volume_ma_7"] = df["volume"].rolling(7).mean()
    df["volume_ma_14"] = df["volume"].rolling(14).mean()
    df["rel_volume"] = df["volume"] / df["volume_ma_14"].replace(0, np.nan)
    df["taker_buy_ratio"] = (
        df["taker_buy_base_volume"] / df["volume"].replace(0, np.nan)
    )
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Rate-of-change momentum over several look-back windows."""
    for period in [3, 6, 12, 24]:
        df[f"roc_{period}"] = df["close"].pct_change(period) * 100
    return df


def add_stationary_transforms(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Rolling z-scores / percentile ranks to stabilise feature scales over time."""
    returns = df["close"].pct_change()
    ret_mean = returns.rolling(window).mean()
    ret_std = returns.rolling(window).std()
    df["return_zscore_24"] = (returns - ret_mean) / ret_std.replace(0, np.nan)

    vol_mean = df["volume"].rolling(window).mean()
    vol_std = df["volume"].rolling(window).std()
    df["volume_zscore_24"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    def _percentile_rank(arr: np.ndarray) -> float:
        sorted_arr = np.sort(arr)
        return float(np.searchsorted(sorted_arr, arr[-1]) / (len(arr) - 1))

    df["close_percentile_24"] = df["close"].rolling(window).apply(
        _percentile_rank, raw=True
    )
    return df


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """
    Compute all features and create the binary target column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame as returned by ``data_fetcher.fetch_klines``.
    horizon : int
        Number of candles ahead to define the prediction target.
        With 1-h candles the default of 4 corresponds to 4 hours.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with a ``target`` column (1 = price rises, 0 = not).
        Rows with NaN values (warm-up period) are dropped.
    """
    df = df.copy()

    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_stochastic(df)
    df = add_obv(df)
    df = add_price_features(df)
    df = add_volume_features(df)
    df = add_momentum(df)
    df = add_stationary_transforms(df)

    # Target: 1 if close price `horizon` candles ahead is higher than current
    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)

    # Drop raw OHLCV columns that are not features
    drop_cols = [
        "open", "high", "low", "close",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
        "num_trades", "volume",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Remove warm-up rows (NaNs from rolling windows) and the final `horizon`
    # rows whose target would look beyond the available data.
    df = df.dropna()

    return df


FEATURE_COLUMNS = [
    "sma_7", "sma_14", "sma_21", "sma_50",
    "ema_7", "ema_14", "ema_21", "ema_50",
    "rsi_14",
    "macd_line", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_pct_b", "bb_width",
    "atr_14",
    "stoch_k", "stoch_d",
    "obv",
    "return_1h", "return_4h", "return_24h",
    "body_ratio", "upper_wick", "lower_wick", "hl_spread",
    "volume_ma_7", "volume_ma_14", "rel_volume", "taker_buy_ratio",
    "roc_3", "roc_6", "roc_12", "roc_24",
    "return_zscore_24", "volume_zscore_24", "close_percentile_24",
]
