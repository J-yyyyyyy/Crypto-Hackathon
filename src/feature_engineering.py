"""
Feature engineering for the XGBoost crypto trend prediction model.

All features are computed from OHLCV candlestick data and represent
standard technical analysis indicators. Two targets are produced:
the original binary rise/fall label and a thresholded label that
requires >1% forward return for a positive class.
"""

import numpy as np
import pandas as pd

RETURN_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Individual indicator helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _slope(series: pd.Series, window: int = 3) -> pd.Series:
    return series.diff(window) / window


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    std_safe = std.where(std != 0, np.nan)
    return (series - mean) / std_safe


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


def add_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Trend-strength cues: slopes of RSI/MACD and short/medium EMA slopes."""
    df["rsi_slope_3"] = _slope(df["rsi_14"], window=3)
    df["macd_hist_slope_3"] = _slope(df["macd_hist"], window=3)
    df["ema_7_slope"] = _slope(df["ema_7"], window=1)
    df["ema_21_slope"] = _slope(df["ema_21"], window=1)
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


def add_volatility_context(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility context via realized volatility and ATR as a % of price."""
    pct_returns = df["close"].pct_change()
    for window in [6, 24]:
        df[f"realized_volatility_{window}"] = (
            # Scale by sqrt(window) so volatility is comparable across horizons
            pct_returns.rolling(window).std() * np.sqrt(window)
        )
    df["atr_pct"] = df["atr_14"] / df["close"].replace(0, np.nan)
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


def add_volume_price_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Price Trend — cumulative volume weighted by price change."""
    pct_change = df["close"].pct_change().fillna(0)
    volume_price_trend = (pct_change * df["volume"]).cumsum()
    df["vpt"] = volume_price_trend
    df["vpt_ma_14"] = volume_price_trend.rolling(14).mean()
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
    """Rolling z-scores / percentile ranks to stabilize feature scales over time."""
    returns = df["close"].pct_change()
    df["return_zscore_24"] = _rolling_zscore(returns, window)
    df["volume_zscore_24"] = _rolling_zscore(df["volume"], window)

    def _percentile_rank(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return np.nan
        sorted_arr = np.sort(arr)
        denom = len(arr) - 1
        return float(np.searchsorted(sorted_arr, arr[-1]) / denom)

    df["close_percentile_24"] = df["close"].rolling(window).apply(
        _percentile_rank, raw=True
    )
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index plus directional indicators."""
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    high_low = df["high"] - df["low"]
    high_pc = (df["high"] - df["close"].shift(1)).abs()
    low_pc = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    atr = true_range.ewm(com=period - 1, adjust=False).mean()

    plus_di = 100 * _ema(pd.Series(plus_dm, index=df.index), period) / atr.replace(
        0, np.nan
    )
    minus_di = 100 * _ema(pd.Series(minus_dm, index=df.index), period) / atr.replace(
        0, np.nan
    )
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["plus_di_14"] = plus_di
    df["minus_di_14"] = minus_di
    df["adx_14"] = (dx * 100).ewm(com=period - 1, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """
    Compute all features and create the binary target columns.

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
        Feature DataFrame with ``target`` and ``target_return_1pct`` columns
        (1 = price rises). Rows with NaN values (warm-up period) are dropped.
    """
    df = df.copy()

    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_trend_strength(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volatility_context(df)
    df = add_adx(df)
    df = add_stochastic(df)
    df = add_obv(df)
    df = add_volume_price_trend(df)
    df = add_price_features(df)
    df = add_volume_features(df)
    df = add_momentum(df)
    df = add_stationary_transforms(df)

    # Target: 1 if close price `horizon` candles ahead is higher than current
    future_close = df["close"].shift(-horizon)
    current_close = df["close"]
    df["target"] = (future_close > current_close).astype(int)
    df["target_return_1pct"] = (
        (future_close - current_close) / current_close.replace(0, np.nan)
        > RETURN_THRESHOLD
    ).astype(int)

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
    "rsi_14", "rsi_slope_3",
    "macd_line", "macd_signal", "macd_hist", "macd_hist_slope_3",
    "ema_7_slope", "ema_21_slope",
    "bb_upper", "bb_lower", "bb_pct_b", "bb_width",
    "atr_14", "atr_pct", "realized_volatility_6", "realized_volatility_24",
    "adx_14", "plus_di_14", "minus_di_14",
    "stoch_k", "stoch_d",
    "obv", "vpt", "vpt_ma_14",
    "return_1h", "return_4h", "return_24h",
    "body_ratio", "upper_wick", "lower_wick", "hl_spread",
    "volume_ma_7", "volume_ma_14", "rel_volume", "taker_buy_ratio",
    "roc_3", "roc_6", "roc_12", "roc_24",
    "return_zscore_24", "volume_zscore_24", "close_percentile_24",
]
