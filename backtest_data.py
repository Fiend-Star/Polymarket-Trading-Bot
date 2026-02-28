"""
Backtest data loading — Binance klines, CSV candles, funding rates.

SRP: All external data fetching for backtesting.
"""
import time
from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass

import httpx
from loguru import logger

from backtest_models import Candle
from funding_rate_filter import (
    classify_funding, BINANCE_FUNDING_RATE_URL,
)

BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"


def _parse_binance_kline(k) -> Candle:
    return Candle(timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                  open=float(k[1]), high=float(k[2]), low=float(k[3]),
                  close=float(k[4]), volume=float(k[5]))


def fetch_binance_klines(start_dt: datetime, end_dt: datetime,
                         symbol: str = "BTCUSDT", interval: str = "1m") -> List[Candle]:
    """Fetch 1-minute BTC/USDT candles from Binance public API."""
    candles: List[Candle] = []
    start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)
    logger.info(f"Fetching {symbol} {interval}: {start_dt.date()} to {end_dt.date()}")
    with httpx.Client(timeout=30.0) as client:
        cursor = start_ms
        while cursor < end_ms:
            params = {"symbol": symbol, "interval": interval,
                      "startTime": cursor, "endTime": end_ms, "limit": 1000}
            data = client.get(BINANCE_KLINE_URL, params=params).raise_for_status().json()
            if not data: break
            candles.extend(_parse_binance_kline(k) for k in data)
            cursor = int(data[-1][0]) + 60_000
            time.sleep(0.15)
    logger.info(f"Fetched {len(candles)} candles")
    return candles


def load_csv_candles(csv_path: str) -> List[Candle]:
    """Load candles from CSV."""
    import csv
    candles = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_raw = row["timestamp"]
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                ts_val = float(ts_raw)
                if ts_val > 1e12: ts_val /= 1000
                ts = datetime.fromtimestamp(ts_val, tz=timezone.utc)
            candles.append(Candle(
                timestamp=ts, open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row["volume"])))
    candles.sort(key=lambda c: c.timestamp)
    logger.info(f"Loaded {len(candles)} candles from {csv_path}")
    return candles


# ── Funding rates ────────────────────────────────────────────────────

@dataclass
class FundingSnapshot:
    timestamp: datetime
    rate: float
    rate_pct: float
    classification: str
    mean_reversion_bias: float


def _parse_funding_entry(entry) -> FundingSnapshot:
    ts = datetime.fromtimestamp(entry["fundingTime"] / 1000, tz=timezone.utc)
    rate = float(entry["fundingRate"])
    cls, bias = classify_funding(rate)
    return FundingSnapshot(timestamp=ts, rate=rate, rate_pct=rate * 100,
                           classification=cls, mean_reversion_bias=bias)


def fetch_funding_rates(start_dt: datetime, end_dt: datetime) -> List[FundingSnapshot]:
    """Fetch historical BTCUSDT funding rates from Binance Futures."""
    snapshots: List[FundingSnapshot] = []
    start_ms, end_ms = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)
    logger.info(f"Fetching funding rates: {start_dt.date()} to {end_dt.date()}")
    with httpx.Client(timeout=30.0) as client:
        cursor = start_ms
        while cursor < end_ms:
            params = {"symbol": "BTCUSDT", "startTime": cursor, "endTime": end_ms, "limit": 1000}
            try:
                data = client.get(BINANCE_FUNDING_RATE_URL, params=params).raise_for_status().json()
            except Exception as e:
                logger.warning(f"Funding fetch failed: {e}"); break
            if not data: break
            snapshots.extend(_parse_funding_entry(e) for e in data)
            cursor = int(data[-1]["fundingTime"]) + 1
            time.sleep(0.15)
    logger.info(f"Fetched {len(snapshots)} funding snapshots")
    return snapshots


def get_funding_bias_at(snapshots: List[FundingSnapshot], ts: datetime) -> float:
    """Get the most recent funding bias for a given timestamp."""
    if not snapshots: return 0.0
    best = None
    for s in snapshots:
        if s.timestamp <= ts: best = s
        else: break
    return best.mean_reversion_bias if best else 0.0

