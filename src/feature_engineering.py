"""
Feature engineering for the XGBoost crypto trend prediction model.

All features are computed from OHLCV candlestick data and represent
standard technical analysis indicators. A single binary target is
produced: whether the close price ``horizon`` candles ahead exceeds
the current close.
"""

import numpy as np
import pandas as pd
import requests
from typing import Optional

from .data_fetcher import fetch_klines

_BTC_DOMINANCE_CACHE: float | None = None


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


def _winsorize(series: pd.Series, z: float = 5.0) -> pd.Series:
    """Clamp extreme values to reduce outlier impact."""
    mean = series.mean()
    std = series.std()
    upper = mean + z * std
    lower = mean - z * std
    return series.clip(lower=lower, upper=upper)


def _get_cached_dominance() -> float:
    """Fetch BTC dominance once and cache it."""
    global _BTC_DOMINANCE_CACHE
    if _BTC_DOMINANCE_CACHE is not None:
        return _BTC_DOMINANCE_CACHE
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        resp.raise_for_status()
        _BTC_DOMINANCE_CACHE = (
            resp.json()
            .get("data", {})
            .get("market_cap_percentage", {})
            .get("btc", np.nan)
        )
    except Exception:
        _BTC_DOMINANCE_CACHE = np.nan
    return _BTC_DOMINANCE_CACHE


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Exponential moving averages used for cross ratios."""
    for period in [7, 21, 50, 200]:
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
    """Trend-strength cues: slopes of RSI/MACD and EMA cross ratios."""
    df["rsi_slope_3"] = _slope(df["rsi_14"], window=3)
    df["macd_hist_slope_3"] = _slope(df["macd_hist"], window=3)
    df["ema_7_21_cross"] = df["ema_7"] / df["ema_21"].replace(0, np.nan) - 1
    df["ema_21_50_cross"] = df["ema_21"] / df["ema_50"].replace(0, np.nan) - 1
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
    df["realized_volatility_24"] = (
        pct_returns.rolling(24).std() * np.sqrt(24)
    )
    df["return_volatility_24h"] = pct_returns.rolling(24).std()
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
    """On-Balance Volume momentum (rate-of-change)."""
    direction = np.sign(df["close"].diff()).fillna(0)
    obv = (direction * df["volume"]).cumsum()
    df["obv_roc_6"] = obv.pct_change(6)
    return df


def add_volume_price_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Price Trend — use momentum instead of cumulative level."""
    pct_change = df["close"].pct_change().fillna(0)
    volume_price_trend = (pct_change * df["volume"]).cumsum()
    df["vpt_roc_6"] = volume_price_trend.pct_change(6)
    df["vpt_ratio_14"] = volume_price_trend / volume_price_trend.rolling(14).mean()
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return-based price action plus candle body metrics."""
    df["return_1h"] = df["close"].pct_change(1)
    df["return_24h"] = df["close"].pct_change(24)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / candle_range
    df["hl_spread"] = candle_range / df["close"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume moving averages and relative volume."""
    volume_ma_14 = df["volume"].rolling(14).mean()
    df["rel_volume"] = df["volume"] / volume_ma_14.replace(0, np.nan)
    df["volume_ma_14"] = volume_ma_14
    df["taker_buy_ratio"] = (
        df["taker_buy_base_volume"] / df["volume"].replace(0, np.nan)
    )
    df["taker_buy_ratio_smooth"] = df["taker_buy_ratio"].rolling(5).mean()
    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Rate-of-change momentum over several look-back windows."""
    for period in [6, 24]:
        df[f"roc_{period}"] = df["close"].pct_change(period)
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
    df["plus_di_minus_di"] = plus_di - minus_di
    df["adx_14"] = (dx * 100).ewm(com=period - 1, adjust=False).mean()
    return df


def add_lagged_returns(df: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    """Lagged hourly returns to capture short memory effects."""
    for lag in range(1, max_lag + 1):
        df[f"return_lag_{lag}"] = df["close"].pct_change(lag)
    return df


def add_volatility_adjusted_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize returns by recent volatility."""
    df["return_over_atr"] = df["return_1h"] / df["atr_14"].replace(0, np.nan)
    return df


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour/day cycles and Fourier terms."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    # Fourier cycles: 24h and 168h (weekly) using row index (keeps alignment even with gaps)
    for period, name in [(24, "24h"), (168, "168h")]:
        angle = 2 * np.pi * (np.arange(len(df)) % period) / period
        df[f"sin_{name}"] = np.sin(angle)
        df[f"cos_{name}"] = np.cos(angle)
    return df


def add_volume_breakout(df: pd.DataFrame, mult: float = 2.0) -> pd.DataFrame:
    """Flag large volume surges relative to a moving average."""
    df["volume_breakout"] = (df["volume"] > mult * df["volume_ma_14"]).astype(int)
    return df


def _fetch_btc_reference(interval: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        return fetch_klines("BTCUSDT", interval=interval, limit=limit)
    except Exception:
        return None


def add_btc_context(
    df: pd.DataFrame,
    symbol: Optional[str],
    interval: str,
    limit: int,
    btc_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Cross-asset context: BTC dominance and rolling correlations."""
    if symbol and symbol.upper().startswith("BTC"):
        df["btc_dominance"] = 1.0
        df["btc_corr_24"] = 1.0
        df["btc_corr_48"] = 1.0
        df["btc_volume_ratio"] = 1.0
        return df
    btc_data = btc_df if btc_df is not None else _fetch_btc_reference(interval, limit)
    if btc_data is None or btc_data.empty:
        df["btc_dominance"] = np.nan
        df["btc_corr_24"] = np.nan
        df["btc_corr_48"] = np.nan
        df["btc_volume_ratio"] = np.nan
        return df
    btc_returns = btc_data["close"].pct_change()
    btc_returns.name = "btc_return"
    combined = df.join(btc_returns, how="left")
    combined["btc_corr_24"] = (
        combined["return_1h"].rolling(24).corr(combined["btc_return"])
    )
    combined["btc_corr_48"] = (
        combined["return_1h"].rolling(48).corr(combined["btc_return"])
    )
    btc_volume_aligned = btc_data["volume"].reindex(df.index).replace(0, np.nan)
    combined["btc_volume_ratio"] = df["volume"] / btc_volume_aligned

    # BTC dominance from CoinGecko global endpoint (latest snapshot)
    # Dominance is cached per process (static snapshot for the training run)
    dominance = _get_cached_dominance()
    combined["btc_dominance"] = dominance
    return combined


def add_higher_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily features aligned back to hourly index. Requires DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("add_higher_timeframe_features expects a DatetimeIndex.")
    daily = df["close"].resample("1D").last()
    daily_change = daily.pct_change()
    gain = daily_change.clip(lower=0)
    loss = -daily_change.clip(upper=0)
    avg_gain = gain.ewm(com=14 - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=14 - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    daily_rsi = 100 - (100 / (1 + rs))
    daily_rsi = daily_rsi.fillna(100.0)
    daily_df = pd.DataFrame(
        {"close_daily": daily, "close_daily_change": daily_change, "rsi_daily": daily_rsi}
    )
    aligned = daily_df.reindex(df.index, method="ffill")
    df["close_daily_change"] = aligned["close_daily_change"]
    df["rsi_daily"] = aligned["rsi_daily"]
    return df


def add_target_smoothing(df: pd.DataFrame, horizon: int, window: int = 3) -> pd.Series:
    """
Smooth close price before computing the target (no look-ahead).

The rolling median is applied only to historical closes; the smoothed
series is then shifted forward to define the label. ``min_periods=1`` keeps
the warm-up rows valid without introducing future information.
    """
    smoothed = df["close"].rolling(window, center=False, min_periods=1).median()
    future_close = smoothed.shift(-horizon)
    current_close = smoothed
    return (future_close > current_close).astype(int)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Regime cues using long-term trend."""
    df["price_over_ema_200"] = df["close"] / df["ema_200"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    horizon: int = 4,
    symbol: Optional[str] = None,
    interval: str = "1h",
    limit: int = 5000,
    btc_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
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
        Feature DataFrame with a ``target`` column (1 = price rises). Rows with
        NaN values (warm-up period) are dropped. Optionally uses BTC reference
        data for dominance / correlation features.
    """
    df = df.copy()
    # Outlier handling on price to reduce extreme jumps
    df["close"] = _winsorize(df["close"])
    df["open"] = _winsorize(df["open"])
    df["high"] = _winsorize(df["high"])
    df["low"] = _winsorize(df["low"])

    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volatility_context(df)
    df = add_adx(df)
    df = add_obv(df)
    df = add_price_features(df)
    df = add_volume_features(df)
    df = add_momentum(df)
    df = add_lagged_returns(df)
    df = add_higher_timeframe_features(df)
    df = add_seasonality_features(df)
    df = add_volume_breakout(df)
    df = add_stationary_transforms(df)
    df = add_trend_strength(df)
    df = add_volatility_adjusted_returns(df)
    df = add_regime_features(df)
    df = add_btc_context(df, symbol=symbol, interval=interval, limit=limit, btc_df=btc_df)

    # Target: 1 if smoothed close price `horizon` candles ahead is higher than current
    df["target"] = add_target_smoothing(df, horizon=horizon)

    # Drop raw OHLCV columns that are not features
    drop_cols = [
        "open",
        "high",
        "low",
        "close",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "num_trades",
        "volume",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Remove warm-up rows (NaNs from rolling windows) and the final `horizon`
    # rows whose target would look beyond the available data.
    df = df.dropna()

    # Keep only the configured feature set plus the target to avoid redundancy.
    keep_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if "target" in df.columns:
        keep_cols.append("target")
    df = df[keep_cols]

    return df


FEATURE_COLUMNS = [
    # 均线关系（比值）
    "ema_7_21_cross",
    "ema_21_50_cross",
    "price_over_ema_200",
    # 动量
    "rsi_14",
    "rsi_slope_3",
    "macd_hist",
    "macd_hist_slope_3",
    "roc_6",
    "roc_24",
    # 波动率
    "atr_pct",
    "bb_pct_b",
    "bb_width",
    "realized_volatility_24",
    "return_volatility_24h",
    # 成交量
    "rel_volume",
    "volume_zscore_24",
    "taker_buy_ratio",
    "taker_buy_ratio_smooth",
    "obv_roc_6",
    "volume_breakout",
    "btc_volume_ratio",
    # K线形态
    "body_ratio",
    "hl_spread",
    # 趋势强度
    "adx_14",
    "plus_di_minus_di",
    # 收益率
    "return_1h",
    "return_24h",
    "return_over_atr",
    "close_daily_change",
    "rsi_daily",
    "return_zscore_24",
    "close_percentile_24",
    "btc_corr_24",
    "btc_corr_48",
    "btc_dominance",
    # 滞后收益
    *[f"return_lag_{i}" for i in range(1, 13)],
    # 季节性
    "hour",
    "dayofweek",
    "sin_24h",
    "cos_24h",
    "sin_168h",
    "cos_168h",
]
