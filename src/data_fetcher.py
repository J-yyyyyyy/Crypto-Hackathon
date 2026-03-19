"""
Fetches historical OHLCV candlestick data from the Binance public REST API.
No API key is required for market data endpoints.
"""

import time
import requests
import pandas as pd

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "DOGEUSDT",
    "SUIUSDT",
]

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def fetch_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 20000,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch candlestick data for *symbol* from Binance.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g. 'BTCUSDT').
    interval : str
        Candlestick interval understood by Binance (e.g. '1h', '4h').
    limit : int
        Number of candles to retrieve. Values above 1000 are fetched in
        batches using backward pagination.
    retries : int
        Number of retry attempts on network errors.
    backoff : float
        Initial back-off delay in seconds (doubles on each retry).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: open_time, open, high, low, close, volume, …
        Numeric columns are cast to float64; open_time is a UTC datetime index.
    """
    url = f"{BINANCE_BASE_URL}{KLINES_ENDPOINT}"

    def _request_batch(params: dict) -> list:
        nonlocal backoff
        delay = backoff
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Failed to fetch klines for {symbol} after {retries} attempts: {exc}"
                    ) from exc
                time.sleep(delay)
                delay *= 2
        return []

    remaining = limit
    end_time: int | None = None
    all_rows: list = []

    while remaining > 0:
        batch_size = min(1000, remaining)
        params = {"symbol": symbol, "interval": interval, "limit": batch_size}
        if end_time is not None:
            params["endTime"] = end_time

        raw = _request_batch(params)
        if not raw:
            break

        all_rows.extend(raw)
        remaining -= len(raw)

        # Prepare next batch to fetch older candles. Binance returns data in
        # ascending order by open time.
        first_open_time = int(raw[0][0])
        end_time = first_open_time - 1

        # Stop early if fewer rows than requested were returned (no more history).
        if len(raw) < batch_size:
            break

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)

    df = df.sort_values("open_time")

    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").drop(columns=["close_time", "ignore"])

    return df


def fetch_all_symbols(interval: str = "1h", limit: int = 5000) -> dict[str, pd.DataFrame]:
    """
    Fetch klines for every symbol in ``SYMBOLS``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from symbol name to its candlestick DataFrame.
    """
    data = {}
    for symbol in SYMBOLS:
        data[symbol] = fetch_klines(symbol, interval=interval, limit=limit)
    return data
