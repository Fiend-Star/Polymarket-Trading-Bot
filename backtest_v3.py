"""
Backtest Engine V4.1 — REAL Polymarket Data (Synced with Live Bot March 2026)
==============================================================================

The critical upgrade over V3: replaces the synthetic btc_to_probability()
function with ACTUAL historical Polymarket market prices.

Data pipeline:
  1. Generate slug timestamps for every 15-min window in the date range
     Slug format: btc-updown-15m-{unix_start}
  2. Query Gamma API for each slug → get token IDs + resolution outcome
  3. Query CLOB /prices-history for YES token → real minute-by-minute prices
  4. Fetch Binance 1m candles (same as V3) → feed VolEstimator
  5. Replay V3.1 quant strategy against REAL market prices

What this validates that V3 cannot:
  - Model edge vs real Polymarket crowd (not a toy function)
  - Actual bid-ask conditions and price paths for gamma exits
  - Real resolution outcomes (not BTC close > open proxy)

Usage:
    python backtest_v4.py                          # Last 24 hours
    python backtest_v4.py --days 7                 # Last 7 days
    python backtest_v4.py --start 2026-02-21 --end 2026-02-28
    python backtest_v4.py --verbose                # Print each trade
    python backtest_v4.py --output results.json    # Export
    python backtest_v4.py --cache-dir ./cache      # Cache API responses
    python backtest_v4.py --no-confirmation        # Skip fusion confirmation
    python backtest_v4.py --taker                  # Simulate taker fills

Requirements:
    - binary_pricer.py, vol_estimator.py, mispricing_detector.py (V2)
    - Network access to gamma-api.polymarket.com and clob.polymarket.com
    - Network access to api.binance.com (for BTC candles + funding rates)
    - No Polymarket credentials needed (all public endpoints)
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import httpx
from loguru import logger

# Suppress noisy logs by default
logger.remove()
logger.add(sys.stderr, level="WARNING")

# ── V3 Quant modules ────────────────────────────────────────────────────────
from binary_pricer import BinaryOptionPricer
from vol_estimator import VolEstimator
from mispricing_detector import MispricingDetector, polymarket_taker_fee

# ── Optional: Fusion processors for confirmation ────────────────────────────
# [PATCH-6] Removed SentimentProcessor import — dropped from live bot March 2026
try:
    from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
    from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
    from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False

# =============================================================================
# Configuration (same defaults as V3.1 / bot.py)
# =============================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MIN_EDGE_CENTS = float(os.getenv("MIN_EDGE_CENTS", "0.02"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.30"))
CUT_LOSS_PCT = float(os.getenv("CUT_LOSS_PCT", "-0.50"))
VOL_METHOD = os.getenv("VOL_METHOD", "ewma")
DEFAULT_VOL = float(os.getenv("DEFAULT_VOL", "0.65"))
BANKROLL_USD = float(os.getenv("BANKROLL_USD", "25.0"))  # [PATCH-1] synced with live .env.example
MARKET_BUY_USD = float(os.getenv("MARKET_BUY_USD", "1.00"))
WINDOW_MINUTES = 15

# [PATCH-2] Trade window enforcement (synced with live bot)
TRADE_WINDOW_START_SEC = int(os.getenv("TRADE_WINDOW_START", "180"))   # 3 minutes into window
TRADE_WINDOW_END_SEC = int(os.getenv("TRADE_WINDOW_END", "600"))       # 10 minutes into window
QUOTE_STABILITY_REQUIRED = int(os.getenv("QUOTE_STABILITY_REQUIRED", "5"))  # ticks before trading

GAMMA_EXIT_PROFIT_PCT = float(os.getenv("GAMMA_EXIT_PROFIT_PCT", "0.04"))
GAMMA_EXIT_TIME_MINS = float(os.getenv("GAMMA_EXIT_TIME_MINS", "3.0"))


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class PolymarketWindow:
    """One resolved 15-minute market with real price data."""
    slug: str
    window_start: datetime
    window_end: datetime
    yes_token_id: str
    no_token_id: str
    resolved_up: bool                    # True = YES won, False = NO won
    # Minute-by-minute YES price from CLOB /prices-history
    # Key = minute offset (0..14), Value = YES price (0.0 - 1.0)
    yes_prices: Dict[int, float]
    # Raw timeseries for debugging
    raw_timeseries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestTrade:
    window_start: datetime
    slug: str
    direction: str                  # "long" (buy YES) or "short" (buy NO)
    entry_price: float              # REAL Polymarket YES/NO price at decision minute
    exit_price: float               # REAL price at gamma exit or binary resolution
    resolved_early: bool            # gamma exit triggered?
    position_size: float
    resolved_up: bool               # actual market resolution
    outcome: str                    # "WIN" or "LOSS"
    pnl: float
    pnl_before_fees: float
    fee_paid: float
    # Model outputs
    model_yes_price: float
    model_no_price: float
    edge: float
    edge_pct: float
    kelly_fraction: float
    realized_vol: float
    implied_vol: float
    vrp: float
    pricing_method: str
    # Confirmation
    confirming_signals: int
    contradicting_signals: int
    # V3.1 overlays
    vol_skew: float
    funding_bias: float
    funding_regime: str
    time_remaining_min: float
    # Data quality
    price_points_available: int     # how many minutes had real price data


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    strategy_mode: str
    total_windows: int
    windows_with_data: int
    windows_skipped_no_data: int
    trades_taken: int
    trades_skipped: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    total_fees: float
    max_drawdown: float
    profit_factor: float
    avg_pnl_per_trade: float
    avg_edge_per_trade: float
    avg_kelly: float
    best_trade: float
    worst_trade: float
    sharpe_ratio: float
    bankroll_final: float
    bankroll_growth: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


# =============================================================================
# Polymarket Data Fetcher
# =============================================================================
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


class PolymarketDataFetcher:
    """
    Fetches real historical Polymarket BTC 15-min market data.

    Two-step process:
    1. Gamma API → enumerate resolved markets, get token IDs + outcomes
    2. CLOB /prices-history → minute-by-minute YES token prices
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.client = httpx.Client(timeout=20.0)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_count = 0

    def close(self):
        self.client.close()

    def _rate_limit(self):
        """Polymarket Gamma: ~300 req/10s. CLOB similar. Stay safe."""
        self._request_count += 1
        if self._request_count % 50 == 0:
            time.sleep(1.0)
        else:
            time.sleep(0.12)

    def _cache_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        safe_key = key.replace("/", "_").replace("?", "_").replace("&", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _get_cached(self, key: str) -> Optional[Any]:
        path = self._cache_path(key)
        if path and path.exists():
            return json.loads(path.read_text())
        return None

    def _set_cached(self, key: str, data: Any):
        path = self._cache_path(key)
        if path:
            path.write_text(json.dumps(data, default=str))

    # ── Step 1: Enumerate resolved markets ───────────────────────────────

    def _generate_slugs(self, start_dt: datetime, end_dt: datetime) -> List[str]:
        """
        Generate all possible btc-updown-15m slug timestamps in the range.

        Windows are every 15 minutes aligned to :00/:15/:30/:45.
        This is the most reliable discovery method — doesn't depend on
        Gamma API supporting prefix/partial slug matching.
        """
        slugs = []
        # Align start to nearest 15-min boundary
        start_ts = int(start_dt.timestamp())
        start_ts = start_ts - (start_ts % 900)
        end_ts = int(end_dt.timestamp())

        cursor = start_ts
        while cursor < end_ts:
            slugs.append(f"btc-updown-15m-{cursor}")
            cursor += 900

        return slugs

    def fetch_resolved_markets(
        self, start_dt: datetime, end_dt: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all resolved BTC 15-min markets in the date range.

        Uses the dedicated /markets/slug/{slug} endpoint which returns
        a single market object per exact slug. We generate all possible
        15-min-aligned slug timestamps and query each one.

        ~96 slugs/day, ~12s with rate limiting. Cached for re-runs.
        """
        cache_key = f"markets_{start_dt.date()}_{end_dt.date()}"
        cached = self._get_cached(cache_key)
        if cached:
            logger.info(f"Using cached market list: {len(cached)} markets")
            return cached

        slugs = self._generate_slugs(start_dt, end_dt)
        all_markets = []
        errors = 0

        logger.info(f"Fetching {len(slugs)} BTC 15-min markets via /markets/slug/...")
        first_response_logged = False

        for i, slug in enumerate(slugs):
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{len(slugs)} ({len(all_markets)} found)...")
                print(f"  Progress: {i + 1}/{len(slugs)} ({len(all_markets)} found)...")

            # Try dedicated slug endpoint first: GET /markets/slug/{slug}
            m = self._fetch_single_slug(slug)

            # Log first successful response for debugging
            if m and not first_response_logged:
                first_response_logged = True
                print(f"  ✓ First market found: {slug}")
                print(f"    Keys: {list(m.keys())[:15]}...")
                outcomes = m.get("outcomes", "?")
                prices = m.get("outcomePrices", "?")
                print(f"    Outcomes: {outcomes}")
                print(f"    OutcomePrices: {prices}")
                print(f"    clobTokenIds: {str(m.get('clobTokenIds', '?'))[:80]}...")

            if m is None:
                errors += 1
                # If too many consecutive errors, API might be down
                if errors > 10:
                    logger.warning(f"Too many errors ({errors}), stopping slug queries")
                    break
                continue

            errors = 0  # reset on success

            # Validate it has token IDs
            token_ids_raw = m.get("clobTokenIds")
            if not token_ids_raw:
                continue
            if isinstance(token_ids_raw, str):
                try:
                    token_ids = json.loads(token_ids_raw)
                except Exception:
                    continue
            else:
                token_ids = token_ids_raw
            if len(token_ids) < 2:
                continue

            all_markets.append(m)

        self._set_cached(cache_key, all_markets)
        logger.info(f"Fetched {len(all_markets)} resolved markets (of {len(slugs)} slugs)")
        return all_markets

    def _fetch_single_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single market by exact slug.

        Tries two endpoints:
        1. GET /markets/slug/{slug}  → returns a single market object
        2. GET /markets?slug={slug}  → returns array (fallback)
        """
        self._rate_limit()

        # Primary: dedicated slug endpoint
        try:
            resp = self.client.get(f"{GAMMA_API}/markets/slug/{slug}")
            if resp.status_code == 200:
                data = resp.json()
                # This endpoint returns a single market object (not an array)
                if isinstance(data, dict) and data.get("slug") == slug:
                    return data
                # Some APIs wrap in array
                if isinstance(data, list) and data:
                    return data[0]
        except Exception as e:
            logger.debug(f"Slug endpoint failed for {slug}: {e}")

        # Fallback: query param
        try:
            resp = self.client.get(f"{GAMMA_API}/markets", params={
                "slug": slug,
                "limit": 1,
            })
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    m = data[0]
                    if m.get("slug") == slug:
                        return m
        except Exception as e:
            logger.debug(f"Markets query fallback failed for {slug}: {e}")

        return None

    # ── Step 2: Fetch price history for a token ──────────────────────────

    def fetch_price_history(
        self, token_id: str, start_ts: int, end_ts: int, fidelity: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Fetch minute-by-minute price history from CLOB /prices-history.

        Returns list of {"t": unix_timestamp, "p": price} dicts.
        fidelity=1 means 1-minute resolution.
        """
        cache_key = f"prices_{token_id}_{start_ts}_{end_ts}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self._rate_limit()
        try:
            resp = self.client.get(f"{CLOB_API}/prices-history", params={
                "market": token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": fidelity,
            })
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"CLOB price history error for {token_id}: {e}")
            return []

        history = data.get("history", [])
        self._set_cached(cache_key, history)
        return history

    # ── Step 3: Build PolymarketWindow objects ───────────────────────────

    def build_windows(
        self, start_dt: datetime, end_dt: datetime,
    ) -> List[PolymarketWindow]:
        """
        Full pipeline: enumerate markets → fetch prices → build windows.
        """
        raw_markets = self.fetch_resolved_markets(start_dt, end_dt)
        windows: List[PolymarketWindow] = []
        skipped = 0

        logger.info(f"Building price windows for {len(raw_markets)} markets...")
        first_price_logged = False

        for i, m in enumerate(raw_markets):
            if (i + 1) % 50 == 0:
                logger.info(f"  Processing market {i + 1}/{len(raw_markets)}...")
                print(f"  Fetching prices: {i + 1}/{len(raw_markets)} "
                      f"({len(windows)} with data, {skipped} skipped)...")

            slug = m.get("slug", "")

            # Parse token IDs
            token_ids_raw = m.get("clobTokenIds")
            if isinstance(token_ids_raw, str):
                token_ids = json.loads(token_ids_raw)
            else:
                token_ids = token_ids_raw

            # Parse outcomes to identify YES/NO token mapping
            outcomes_raw = m.get("outcomes")
            if isinstance(outcomes_raw, str):
                outcomes = json.loads(outcomes_raw)
            else:
                outcomes = outcomes_raw or ["Up", "Down"]

            # Determine which token is YES (Up) and which is NO (Down)
            # Polymarket convention: outcomes[0] = first option, outcomes[1] = second
            # For BTC up/down: "Up" is YES, "Down" is NO
            yes_idx = 0
            no_idx = 1
            for idx, out in enumerate(outcomes):
                if out.lower() in ("up", "yes"):
                    yes_idx = idx
                elif out.lower() in ("down", "no"):
                    no_idx = idx

            yes_token_id = token_ids[yes_idx]
            no_token_id = token_ids[no_idx]

            # Parse resolution outcome
            # Check outcomePrices (resolved markets show 1.00/0.00)
            outcome_prices_raw = m.get("outcomePrices")
            if isinstance(outcome_prices_raw, str):
                try:
                    outcome_prices = json.loads(outcome_prices_raw)
                except Exception:
                    outcome_prices = None
            else:
                outcome_prices = outcome_prices_raw

            resolved_up = None
            if outcome_prices and len(outcome_prices) >= 2:
                yes_final = float(outcome_prices[yes_idx])
                no_final = float(outcome_prices[no_idx])
                if yes_final > 0.9:
                    resolved_up = True
                elif no_final > 0.9:
                    resolved_up = False

            # Fallback: check "outcome" field
            if resolved_up is None:
                outcome_str = str(m.get("outcome", "")).lower()
                if outcome_str in ("up", "yes"):
                    resolved_up = True
                elif outcome_str in ("down", "no"):
                    resolved_up = False

            if resolved_up is None:
                skipped += 1
                continue

            # Parse window times from slug
            # Slug format: btc-updown-15m-{unix_start}
            slug_parts = slug.rsplit("-", 1)
            try:
                window_start_ts = int(slug_parts[-1])
            except (ValueError, IndexError):
                # Fallback: parse from endDate - 15 minutes
                end_date_str = m.get("endDate") or m.get("end_date_min") or ""
                try:
                    window_end = datetime.fromisoformat(
                        end_date_str.replace("Z", "+00:00")
                    )
                    window_start_ts = int(window_end.timestamp()) - 900
                except Exception:
                    skipped += 1
                    continue

            window_start = datetime.fromtimestamp(window_start_ts, tz=timezone.utc)
            window_end = window_start + timedelta(minutes=15)

            # Fetch YES token price history
            history = self.fetch_price_history(
                yes_token_id,
                start_ts=window_start_ts,
                end_ts=int(window_end.timestamp()),
                fidelity=1,
            )

            if not history:
                skipped += 1
                continue

            # Log first price history for debugging
            if not first_price_logged and history:
                first_price_logged = True
                print(f"  ✓ First price history: {slug}")
                print(f"    Points: {len(history)}")
                if history:
                    print(f"    Sample: t={history[0].get('t')}, p={history[0].get('p')}")
                    if len(history) > 1:
                        print(f"    Last:   t={history[-1].get('t')}, p={history[-1].get('p')}")

            # Map timestamps to minute offsets within the window
            yes_prices: Dict[int, float] = {}
            for point in history:
                t = point.get("t", 0)
                p = point.get("p", 0.5)
                offset_sec = t - window_start_ts
                minute = offset_sec // 60
                if 0 <= minute < 15:
                    # If multiple points in same minute, use the latest
                    yes_prices[minute] = float(p)

            if len(yes_prices) < 2:
                skipped += 1
                continue

            windows.append(PolymarketWindow(
                slug=slug,
                window_start=window_start,
                window_end=window_end,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
                resolved_up=resolved_up,
                yes_prices=yes_prices,
                raw_timeseries=history,
            ))

        logger.info(
            f"Built {len(windows)} windows with price data "
            f"({skipped} skipped: no resolution or no price data)"
        )
        return windows


# =============================================================================
# Binance kline + funding rate fetchers (same as V3)
# =============================================================================
def fetch_binance_klines(
    start_dt: datetime, end_dt: datetime,
    symbol: str = "BTCUSDT", interval: str = "1m",
) -> List[Candle]:
    """Fetch 1-minute BTC/USDT candles from Binance public API."""
    candles = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    logger.info(f"Fetching Binance klines: {start_dt.date()} to {end_dt.date()}")

    with httpx.Client(timeout=30.0) as client:
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": symbol, "interval": interval,
                "startTime": cursor, "endTime": end_ms, "limit": 1000,
            }
            resp = client.get("https://api.binance.com/api/v3/klines", params=params)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for k in data:
                ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
                candles.append(Candle(
                    timestamp=ts, open=float(k[1]), high=float(k[2]),
                    low=float(k[3]), close=float(k[4]), volume=float(k[5]),
                ))
            cursor = int(data[-1][0]) + 60_000
            time.sleep(0.15)

    logger.info(f"Fetched {len(candles)} Binance candles")
    return candles


@dataclass
class FundingSnapshot:
    timestamp: datetime
    rate: float
    classification: str
    mean_reversion_bias: float


def classify_funding(rate: float) -> Tuple[str, float]:
    EXTREME = 0.0005
    HIGH = 0.0002
    MAX_BIAS = 0.02
    if rate >= EXTREME:
        return "EXTREME_POSITIVE", -MAX_BIAS
    elif rate >= HIGH:
        scale = (rate - HIGH) / (EXTREME - HIGH)
        return "HIGH_POSITIVE", -(0.005 + scale * 0.015)
    elif rate <= -EXTREME:
        return "EXTREME_NEGATIVE", MAX_BIAS
    elif rate <= -HIGH:
        scale = (-rate - HIGH) / (EXTREME - HIGH)
        return "HIGH_NEGATIVE", (0.005 + scale * 0.015)
    else:
        return "NEUTRAL", 0.0


def fetch_funding_rates(start_dt: datetime, end_dt: datetime) -> List[FundingSnapshot]:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    snapshots = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    with httpx.Client(timeout=30.0) as client:
        cursor = start_ms
        while cursor < end_ms:
            try:
                resp = client.get(url, params={
                    "symbol": "BTCUSDT",
                    "startTime": cursor, "endTime": end_ms, "limit": 1000,
                })
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"Funding rate fetch failed: {e}")
                break
            if not data:
                break
            for entry in data:
                ts = datetime.fromtimestamp(entry["fundingTime"] / 1000, tz=timezone.utc)
                rate = float(entry["fundingRate"])
                classification, bias = classify_funding(rate)
                snapshots.append(FundingSnapshot(
                    timestamp=ts, rate=rate,
                    classification=classification,
                    mean_reversion_bias=bias,
                ))
            cursor = int(data[-1]["fundingTime"]) + 1
            time.sleep(0.15)

    return snapshots


def get_funding_bias_at(snapshots: List[FundingSnapshot], ts: datetime) -> Tuple[float, str]:
    if not snapshots:
        return 0.0, "DISABLED"
    best = None
    for s in snapshots:
        if s.timestamp <= ts:
            best = s
        else:
            break
    if best:
        return best.mean_reversion_bias, best.classification
    return 0.0, "NEUTRAL"


# =============================================================================
# RSI helper (same as V3)
# =============================================================================
def compute_rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent = deltas[-period:]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0001
    avg_loss = sum(losses) / period if losses else 0.0001
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# =============================================================================
# Fusion confirmation (same as V3)
# =============================================================================
def create_fusion_processors() -> Optional[Dict]:
    # [PATCH-4] Synced with live bot (March 2026):
    #   - Removed SentimentProcessor (dropped from live)
    #   - Removed DeribitPCR (dropped from live)
    #   - Live weights: OrderBook=0.40, TickVelocity=0.30, Divergence=0.20, Spike=0.10
    #   - OrderBook cannot run in backtest (no historical orderbook data)
    #   - So backtest confirmation uses: spike, divergence, tick_velocity only
    if not FUSION_AVAILABLE:
        return None
    return {
        "spike": SpikeDetectionProcessor(
            spike_threshold=float(os.getenv("SPIKE_THRESHOLD", "0.05")),
            lookback_periods=20,
        ),
        "divergence": PriceDivergenceProcessor(
            divergence_threshold=float(os.getenv("DIVERGENCE_THRESHOLD", "0.05")),
        ),
        "tick_velocity": TickVelocityProcessor(
            velocity_threshold_60s=0.015, velocity_threshold_30s=0.010,
        ),
    }


def run_confirmation(
    processors, current_price, price_history, metadata, model_bullish,
) -> Tuple[int, int]:
    if processors is None:
        return 0, 0
    signals = []
    pm = {}
    for k, v in metadata.items():
        pm[k] = Decimal(str(v)) if isinstance(v, float) else v
    for name in ["spike", "divergence", "tick_velocity"]:  # [PATCH-5] sentiment removed (live bot dropped it)
        proc = processors.get(name)
        if not proc:
            continue
        try:
            sig = proc.process(
                current_price=current_price,
                historical_prices=price_history,
                metadata=pm,
            )
            if sig:
                signals.append(sig)
        except Exception:
            pass
    confirming = contradicting = 0
    for sig in signals:
        sig_bullish = "BULLISH" in str(sig.direction).upper()
        if sig_bullish == model_bullish:
            confirming += 1
        else:
            contradicting += 1
    return confirming, contradicting


# =============================================================================
# Main backtest loop — V4 REAL DATA
# =============================================================================
def run_backtest_v4(
    windows: List[PolymarketWindow],
    candles: List[Candle],
    decision_minute: int = 2,
    verbose: bool = False,
    use_confirmation: bool = True,
    use_maker: bool = True,
    use_funding: bool = True,
    funding_snapshots: Optional[List[FundingSnapshot]] = None,
) -> BacktestResult:
    """
    Backtest against REAL Polymarket price data.

    For each resolved 15-min market:
      1. Get the REAL YES price at 'decision_minute' → this is the market price
      2. Run the quant model (same as V3.1) → get model fair value
      3. Compare model vs REAL market → edge
      4. Simulate gamma exits using REAL intra-window prices
      5. Resolution from REAL market outcome (not BTC close > open)
    """

    # ── Build a minute-indexed candle lookup ─────────────────────────────
    # Key = unix timestamp floored to minute → Candle
    candle_by_minute: Dict[int, Candle] = {}
    for c in candles:
        key = int(c.timestamp.timestamp()) // 60 * 60
        candle_by_minute[key] = c

    # ── Initialize quant modules ─────────────────────────────────────────
    pricer = BinaryOptionPricer()
    vol_est = VolEstimator(
        window_minutes=60.0,
        resample_interval_sec=60.0,
        ewma_halflife_samples=20,
        min_samples=5,
        default_vol=DEFAULT_VOL,
    )
    detector = MispricingDetector(
        pricer=pricer,
        vol_estimator=vol_est,
        maker_fee=0.00,
        min_edge_cents=MIN_EDGE_CENTS,
        min_edge_after_fees=0.005,
        take_profit_pct=TAKE_PROFIT_PCT,
        cut_loss_pct=CUT_LOSS_PCT,
        vol_method=VOL_METHOD,
        bankroll=BANKROLL_USD,
        use_half_kelly=True,
        max_kelly_fraction=0.05,
    )

    fusion_processors = create_fusion_processors() if use_confirmation else None

    # ── Pre-seed vol estimator with candles before first window ─────────
    first_window_ts = int(windows[0].window_start.timestamp()) if windows else 0
    warmup_candles = [c for c in candles if c.timestamp.timestamp() < first_window_ts]
    if not warmup_candles:
        warmup_candles = candles[:60]
    for c in warmup_candles:
        vol_est.add_price(c.close, c.timestamp.timestamp())

    # Track the last candle timestamp we've fed to vol estimator
    last_fed_ts = int(warmup_candles[-1].timestamp.timestamp()) if warmup_candles else 0

    # ── State tracking ───────────────────────────────────────────────────
    historical_btc_closes: List[float] = [c.close for c in warmup_candles]
    poly_price_history: List[Decimal] = []
    trades: List[BacktestTrade] = []
    equity_curve: List[float] = [0.0]
    bankroll = BANKROLL_USD
    total_fees = 0.0
    skipped = 0
    windows_with_data = 0

    skip_reasons: Dict[str, int] = {
        "no_price_at_decision": 0,
        "no_btc_data": 0,
        "warmup": 0,
        "no_edge": 0,
        "confirmation_veto": 0,
    }

    # ── Process each real market window ──────────────────────────────────
    total_windows = len(windows)

    for wi, w in enumerate(windows):

        # ── Feed BTC candles up to decision minute to vol estimator ───────
        window_start_ts = int(w.window_start.timestamp())
        decision_ts = window_start_ts + decision_minute * 60

        # Feed all candles between last fed timestamp and this decision point
        # This keeps the vol estimator current even when we skip windows
        for ts_key in sorted(candle_by_minute.keys()):
            if ts_key <= last_fed_ts:
                continue
            if ts_key > decision_ts:
                break
            c = candle_by_minute[ts_key]
            vol_est.add_price(c.close, c.timestamp.timestamp())
            historical_btc_closes.append(c.close)
            last_fed_ts = ts_key

        # Set simulated time for vol estimator
        vol_est.set_simulated_time(float(decision_ts))

        if len(historical_btc_closes) < 30:
            skip_reasons["warmup"] += 1
            skipped += 1
            continue

        # ── Get REAL Polymarket YES price at decision minute ─────────────
        yes_at_decision = w.yes_prices.get(decision_minute)

        # If exact minute not available, try nearest available
        if yes_at_decision is None:
            for offset in [1, -1, 2, -2]:
                yes_at_decision = w.yes_prices.get(decision_minute + offset)
                if yes_at_decision is not None:
                    break

        if yes_at_decision is None:
            skip_reasons["no_price_at_decision"] += 1
            skipped += 1
            continue

        windows_with_data += 1

        # Real market prices
        yes_market = yes_at_decision
        no_market = 1.0 - yes_at_decision

        # Track poly price history for confirmation processors
        poly_price_history.append(Decimal(str(round(yes_market, 4))))
        if len(poly_price_history) > 100:
            poly_price_history.pop(0)

        # ── BTC spot and strike from Binance candles ─────────────────────
        btc_spot_candle = candle_by_minute.get(decision_ts)
        btc_strike_candle = candle_by_minute.get(window_start_ts)

        if not btc_spot_candle or not btc_strike_candle:
            skip_reasons["no_btc_data"] += 1
            skipped += 1
            continue

        btc_spot = btc_spot_candle.close
        btc_strike = btc_strike_candle.open
        time_remaining_min = float(WINDOW_MINUTES - decision_minute)

        # ── V3.1: Vol skew correction ────────────────────────────────────
        vol_estimate = vol_est.get_vol(VOL_METHOD)
        vol_skew = BinaryOptionPricer.estimate_btc_vol_skew(
            spot=btc_spot,
            strike=btc_strike,
            vol=vol_estimate.annualized_vol,
            time_remaining_min=time_remaining_min,
        )

        # ── V3.1: Funding rate bias ─────────────────────────────────────
        if use_funding and funding_snapshots:
            funding_bias, funding_regime = get_funding_bias_at(
                funding_snapshots, w.window_start
            )
        else:
            funding_bias, funding_regime = 0.0, "DISABLED"

        # ── Run mispricing detector against REAL market price ────────────
        signal = detector.detect(
            yes_market_price=yes_market,
            no_market_price=no_market,
            btc_spot=btc_spot,
            btc_strike=btc_strike,
            time_remaining_min=time_remaining_min,
            position_size_usd=MARKET_BUY_USD,
            use_maker=use_maker,
            vol_skew=vol_skew,
            funding_bias=funding_bias,
        )

        if not signal.is_tradeable:
            skip_reasons["no_edge"] += 1
            skipped += 1
            continue

        # ── Fusion confirmation ──────────────────────────────────────────
        model_bullish = signal.direction == "BUY_YES"
        confirming, contradicting = 0, 0

        if use_confirmation and fusion_processors and len(poly_price_history) >= 5:
            rsi = compute_rsi(historical_btc_closes[-50:], period=14)
            recent_closes = historical_btc_closes[-20:]
            sma_20 = sum(recent_closes) / len(recent_closes) if recent_closes else btc_spot
            metadata = {
                "deviation": (btc_spot - sma_20) / sma_20 if sma_20 else 0.0,
                "momentum": (btc_spot - historical_btc_closes[-5]) / historical_btc_closes[-5]
                            if len(historical_btc_closes) >= 5 else 0.0,
                "volatility": (sum((p - sma_20) ** 2 for p in recent_closes) / len(recent_closes)) ** 0.5
                              if recent_closes else 0.0,
                "sentiment_score": rsi,
                "sentiment_classification": (
                    "Extreme Fear" if rsi < 25 else
                    "Fear" if rsi < 45 else
                    "Neutral" if rsi < 55 else
                    "Greed" if rsi < 75 else
                    "Extreme Greed"
                ),
                "spot_price": btc_spot,
                "tick_buffer": [],
                "yes_token_id": None,
            }

            confirming, contradicting = run_confirmation(
                fusion_processors,
                Decimal(str(round(yes_market, 4))),
                list(poly_price_history),
                metadata,
                model_bullish,
            )

            if contradicting > confirming and signal.confidence < 0.6:
                skip_reasons["confirmation_veto"] += 1
                skipped += 1
                continue

        # ── Determine direction and entry price ──────────────────────────
        if signal.direction == "BUY_YES":
            direction = "long"
            entry_price = yes_market
        else:
            direction = "short"
            entry_price = no_market

        # ── Position sizing (same as V3.1) ───────────────────────────────
        base_risk = bankroll * 0.05
        score_mult = (85.0 if signal.confidence > 0.6 else 50.0) / 100.0
        strength = signal.confidence * score_mult * getattr(signal, 'kelly_fraction', 1.0)
        time_factor = max(0.1, time_remaining_min / 5.0) if time_remaining_min < 5 else 1.0
        position_size = max(1.0, min(base_risk * strength * time_factor, MARKET_BUY_USD))

        # ── Gamma scalping: use REAL intra-window prices ─────────────────
        exit_price = None
        resolved_early = False

        for m in range(decision_minute + 1, 15):
            m_yes = w.yes_prices.get(m)
            if m_yes is None:
                continue

            if direction == "long":
                m_price = m_yes
            else:
                m_price = 1.0 - m_yes

            pnl_pct = (m_price - entry_price) / entry_price if entry_price > 0 else 0
            m_remaining = 15 - m

            if pnl_pct >= TAKE_PROFIT_PCT or (
                m_remaining < GAMMA_EXIT_TIME_MINS and pnl_pct > GAMMA_EXIT_PROFIT_PCT
            ):
                exit_price = m_price
                resolved_early = True
                break

        # If no gamma exit, hold to binary resolution
        if not resolved_early:
            if direction == "long":
                won = w.resolved_up
            else:
                won = not w.resolved_up
            exit_price = 1.0 if won else 0.0

        outcome = "WIN" if exit_price > entry_price else "LOSS"

        # ── P&L with correct fees ────────────────────────────────────────
        num_tokens = position_size / entry_price if entry_price > 0 else 0

        entry_fee_rate = 0.0 if use_maker else polymarket_taker_fee(entry_price)
        exit_fee_rate = 0.0 if not resolved_early else polymarket_taker_fee(exit_price)

        fee = (entry_fee_rate * num_tokens * entry_price) + \
              (exit_fee_rate * num_tokens * exit_price)

        pnl_before_fees = (exit_price - entry_price) * num_tokens
        pnl = pnl_before_fees - fee
        total_fees += fee

        bankroll += pnl
        bankroll = max(1.0, bankroll)
        detector.update_bankroll(bankroll)

        cum_pnl = equity_curve[-1] + pnl
        equity_curve.append(cum_pnl)

        trade = BacktestTrade(
            window_start=w.window_start,
            slug=w.slug,
            direction=direction,
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            resolved_early=resolved_early,
            position_size=round(position_size, 2),
            resolved_up=w.resolved_up,
            outcome=outcome,
            pnl=round(pnl, 4),
            pnl_before_fees=round(pnl_before_fees, 4),
            fee_paid=round(fee, 4),
            model_yes_price=round(signal.yes_model, 4),
            model_no_price=round(signal.no_model, 4),
            edge=round(signal.edge, 4),
            edge_pct=round(signal.edge_pct, 4),
            kelly_fraction=round(signal.kelly_fraction, 4),
            realized_vol=round(signal.realized_vol, 4),
            implied_vol=round(signal.implied_vol, 4),
            vrp=round(signal.vrp, 4),
            pricing_method=signal.pricing_method,
            confirming_signals=confirming,
            contradicting_signals=contradicting,
            vol_skew=round(vol_skew, 8),
            funding_bias=round(funding_bias, 6),
            funding_regime=funding_regime,
            time_remaining_min=time_remaining_min,
            price_points_available=len(w.yes_prices),
        )
        trades.append(trade)

        if verbose:
            arrow = "↑" if w.resolved_up else "↓"
            emoji = "✅" if outcome == "WIN" else "❌"
            exit_tag = f"gamma@{exit_price:.2f}" if resolved_early else f"resolve→{'$1' if exit_price == 1.0 else '$0'}"
            logger.warning(
                f"W{wi + 1:>4}/{total_windows} | "
                f"{w.window_start.strftime('%m-%d %H:%M')} {arrow} | "
                f"{'long' if direction == 'long' else 'short':>5} @{entry_price:.2f} "
                f"(model={signal.yes_model:.2f}/{signal.no_model:.2f}) | "
                f"edge={signal.edge:+.3f} kelly={signal.kelly_fraction:.1%} "
                f"${position_size:.2f} | "
                f"{exit_tag} {emoji} ${pnl:+.4f} | "
                f"cum=${cum_pnl:+.2f} | "
                f"mkt_pts={len(w.yes_prices)} "
                f"[{signal.pricing_method}]"
            )

    # ── Statistics ────────────────────────────────────────────────────────
    wins = sum(1 for t in trades if t.outcome == "WIN")
    losses = sum(1 for t in trades if t.outcome == "LOSS")
    total_pnl = sum(t.pnl for t in trades)
    win_rate = (wins / len(trades) * 100) if trades else 0.0
    avg_pnl = (total_pnl / len(trades)) if trades else 0.0
    avg_edge = (sum(abs(t.edge) for t in trades) / len(trades)) if trades else 0.0
    avg_kelly = (sum(t.kelly_fraction for t in trades) / len(trades)) if trades else 0.0
    best_trade = max((t.pnl for t in trades), default=0.0)
    worst_trade = min((t.pnl for t in trades), default=0.0)

    peak = max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    if len(trades) > 1:
        pnls = [t.pnl for t in trades]
        mean_pnl = sum(pnls) / len(pnls)
        std_pnl = math.sqrt(sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1))
        trades_per_year = 96 * 365
        sharpe = (mean_pnl / std_pnl * math.sqrt(trades_per_year)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    bankroll_growth = ((bankroll - BANKROLL_USD) / BANKROLL_USD * 100) if BANKROLL_USD > 0 else 0.0

    return BacktestResult(
        start_date=windows[0].window_start.strftime("%Y-%m-%d") if windows else "",
        end_date=windows[-1].window_end.strftime("%Y-%m-%d") if windows else "",
        strategy_mode="quant_v4_real_data",
        total_windows=total_windows,
        windows_with_data=windows_with_data,
        windows_skipped_no_data=total_windows - windows_with_data,
        trades_taken=len(trades),
        trades_skipped=skipped,
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 2),
        total_pnl=round(total_pnl, 4),
        total_fees=round(total_fees, 4),
        max_drawdown=round(max_dd, 4),
        profit_factor=round(profit_factor, 2),
        avg_pnl_per_trade=round(avg_pnl, 4),
        avg_edge_per_trade=round(avg_edge, 4),
        avg_kelly=round(avg_kelly, 4),
        best_trade=round(best_trade, 4),
        worst_trade=round(worst_trade, 4),
        sharpe_ratio=round(sharpe, 2),
        bankroll_final=round(bankroll, 2),
        bankroll_growth=round(bankroll_growth, 2),
        trades=trades,
        equity_curve=equity_curve,
    )


# =============================================================================
# Pretty printer
# =============================================================================
def print_results(result: BacktestResult):
    print()
    print("=" * 90)
    print(f"  BACKTEST RESULTS — {result.strategy_mode.upper()} (REAL POLYMARKET DATA)")
    print("=" * 90)
    print(f"  Period:              {result.start_date} → {result.end_date}")
    print(f"  Total windows:       {result.total_windows}")
    print(f"  Windows with data:   {result.windows_with_data}")
    print(f"  Skipped (no data):   {result.windows_skipped_no_data}")
    print(f"  Trades taken:        {result.trades_taken}")
    print(f"  Trades skipped:      {result.trades_skipped}")
    print(f"  Trade rate:          {result.trades_taken / max(result.windows_with_data, 1) * 100:.1f}%")
    print("-" * 90)
    print(f"  Wins:                {result.wins}")
    print(f"  Losses:              {result.losses}")
    print(f"  Win Rate:            {result.win_rate:.1f}%")
    print("-" * 90)
    print(f"  Total P&L:           ${result.total_pnl:+.4f}")
    print(f"  Total Fees Paid:     ${result.total_fees:.4f}")
    print(f"  Avg P&L / Trade:     ${result.avg_pnl_per_trade:+.4f}")
    print(f"  Avg Edge / Trade:    ${result.avg_edge_per_trade:.4f}")
    print(f"  Avg Kelly Fraction:  {result.avg_kelly:.1%}")
    print(f"  Best Trade:          ${result.best_trade:+.4f}")
    print(f"  Worst Trade:         ${result.worst_trade:+.4f}")
    print(f"  Max Drawdown:        ${result.max_drawdown:.4f}")
    print(f"  Profit Factor:       {result.profit_factor:.2f}")
    print(f"  Sharpe Ratio (ann):  {result.sharpe_ratio:.2f}")
    print("-" * 90)
    print(f"  Starting Bankroll:   ${BANKROLL_USD:.2f}")
    print(f"  Ending Bankroll:     ${result.bankroll_final:.2f}")
    print(f"  Bankroll Growth:     {result.bankroll_growth:+.1f}%")
    print("=" * 90)

    if not result.trades:
        print("\n  No trades taken.")
        return

    # Direction breakdown
    longs = [t for t in result.trades if t.direction == "long"]
    shorts = [t for t in result.trades if t.direction == "short"]
    long_wins = sum(1 for t in longs if t.outcome == "WIN")
    short_wins = sum(1 for t in shorts if t.outcome == "WIN")
    print(f"\n  Long trades:  {len(longs):>4} | Wins: {long_wins:>4} | "
          f"WR: {(long_wins / len(longs) * 100) if longs else 0:.1f}%")
    print(f"  Short trades: {len(shorts):>4} | Wins: {short_wins:>4} | "
          f"WR: {(short_wins / len(shorts) * 100) if shorts else 0:.1f}%")

    # Gamma exit analysis
    gamma_exits = [t for t in result.trades if t.resolved_early]
    binary_resolves = [t for t in result.trades if not t.resolved_early]
    print(f"\n  Gamma exits:       {len(gamma_exits):>4} "
          f"({len(gamma_exits) / len(result.trades) * 100:.0f}%)")
    if gamma_exits:
        ge_wins = sum(1 for t in gamma_exits if t.outcome == "WIN")
        print(f"    Win rate:        {ge_wins / len(gamma_exits) * 100:.1f}%")
        print(f"    Avg P&L:         ${sum(t.pnl for t in gamma_exits) / len(gamma_exits):+.4f}")
    print(f"  Binary resolves:   {len(binary_resolves):>4} "
          f"({len(binary_resolves) / len(result.trades) * 100:.0f}%)")
    if binary_resolves:
        br_wins = sum(1 for t in binary_resolves if t.outcome == "WIN")
        print(f"    Win rate:        {br_wins / len(binary_resolves) * 100:.1f}%")
        print(f"    Avg P&L:         ${sum(t.pnl for t in binary_resolves) / len(binary_resolves):+.4f}")

    # Vol analysis
    avg_rv = sum(t.realized_vol for t in result.trades) / len(result.trades)
    avg_iv = sum(t.implied_vol for t in result.trades) / len(result.trades)
    avg_vrp = sum(t.vrp for t in result.trades) / len(result.trades)
    print(f"\n  Avg Realized Vol:    {avg_rv:.0%}")
    print(f"  Avg Implied Vol:     {avg_iv:.0%}")
    print(f"  Avg VRP (IV - RV):   {avg_vrp:+.0%}")

    # Pricing methods
    methods: Dict[str, int] = {}
    for t in result.trades:
        methods[t.pricing_method] = methods.get(t.pricing_method, 0) + 1
    print(f"\n  Pricing methods:")
    for m, count in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"    {m:<20} {count:>4} ({count / len(result.trades) * 100:.0f}%)")

    # Funding regimes
    regimes: Dict[str, List] = {}
    for t in result.trades:
        regimes.setdefault(t.funding_regime, []).append(t)
    if any(t.funding_regime != "DISABLED" for t in result.trades):
        print(f"\n  Funding regime breakdown:")
        for regime, rt in sorted(regimes.items()):
            rw = sum(1 for t in rt if t.outcome == "WIN")
            rpnl = sum(t.pnl for t in rt)
            wr = (rw / len(rt) * 100) if rt else 0
            print(f"    {regime:<20} {len(rt):>4} trades, "
                  f"{wr:.1f}% WR, PnL ${rpnl:+.4f}")

    # Edge distribution
    edges = sorted([abs(t.edge) for t in result.trades])
    if edges:
        print(f"\n  Edge distribution (vs REAL market):")
        print(f"    Min:    ${edges[0]:.4f}")
        print(f"    25th:   ${edges[len(edges) // 4]:.4f}")
        print(f"    Median: ${edges[len(edges) // 2]:.4f}")
        print(f"    75th:   ${edges[3 * len(edges) // 4]:.4f}")
        print(f"    Max:    ${edges[-1]:.4f}")

    # Win rate by edge bucket
    print(f"\n  Win rate by edge size (vs REAL market):")
    for lo, hi in [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.00)]:
        bucket = [t for t in result.trades if lo <= abs(t.edge) < hi]
        if bucket:
            bw = sum(1 for t in bucket if t.outcome == "WIN")
            print(f"    ${lo:.2f}-${hi:.2f}: {len(bucket):>4} trades, "
                  f"{bw / len(bucket) * 100:.1f}% WR, "
                  f"avg PnL ${sum(t.pnl for t in bucket) / len(bucket):+.4f}")

    # Data quality
    avg_pts = sum(t.price_points_available for t in result.trades) / len(result.trades)
    print(f"\n  Avg price points/window: {avg_pts:.1f} (of 15 possible)")

    # Last 15 trades
    print(f"\n  Last 15 trades:")
    print(f"  {'Time':<13} {'Dir':<6} {'Entry':<6} {'Model':<6} "
          f"{'Edge':<7} {'Exit':<8} {'Result':<5} {'P&L':<9} {'Pts':<4} {'Method'}")
    print(f"  {'-' * 95}")
    for t in result.trades[-15:]:
        model_val = t.model_yes_price if t.direction == "long" else t.model_no_price
        exit_tag = f"γ{t.exit_price:.2f}" if t.resolved_early else ("$1.00" if t.exit_price == 1.0 else "$0.00")
        print(
            f"  {t.window_start.strftime('%m-%d %H:%M'):<13} "
            f"{t.direction:<6} "
            f"{t.entry_price:<6.2f} "
            f"{model_val:<6.2f} "
            f"{t.edge:+6.3f} "
            f"{exit_tag:<8} "
            f"{'W' if t.outcome == 'WIN' else 'L':<5} "
            f"${t.pnl:+8.4f} "
            f"{t.price_points_available:>3} "
            f"{t.pricing_method}"
        )

    print()
    print("  ★ ALL PRICES ARE REAL POLYMARKET MARKET DATA ★")
    print("  • Entry prices from CLOB /prices-history at decision minute")
    print("  • Gamma exits use real intra-window price path")
    print("  • Resolution outcomes from actual market settlement")
    print("  • Fees use correct Polymarket nonlinear formula")
    print("  • BTC spot/vol from Binance 1-min candles (same as live bot)")
    print()


# =============================================================================
# Export
# =============================================================================
def export_results(result: BacktestResult, output_path: str):
    data = {
        "summary": {
            "strategy_mode": result.strategy_mode,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_windows": result.total_windows,
            "windows_with_data": result.windows_with_data,
            "trades_taken": result.trades_taken,
            "trades_skipped": result.trades_skipped,
            "wins": result.wins,
            "losses": result.losses,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "total_fees": result.total_fees,
            "max_drawdown": result.max_drawdown,
            "profit_factor": result.profit_factor,
            "avg_pnl_per_trade": result.avg_pnl_per_trade,
            "avg_edge_per_trade": result.avg_edge_per_trade,
            "avg_kelly": result.avg_kelly,
            "sharpe_ratio": result.sharpe_ratio,
            "bankroll_final": result.bankroll_final,
            "bankroll_growth": result.bankroll_growth,
        },
        "trades": [
            {
                "window_start": t.window_start.isoformat(),
                "slug": t.slug,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "resolved_early": t.resolved_early,
                "position_size": t.position_size,
                "resolved_up": t.resolved_up,
                "outcome": t.outcome,
                "pnl": t.pnl,
                "fee_paid": t.fee_paid,
                "model_yes_price": t.model_yes_price,
                "model_no_price": t.model_no_price,
                "edge": t.edge,
                "kelly_fraction": t.kelly_fraction,
                "realized_vol": t.realized_vol,
                "implied_vol": t.implied_vol,
                "vrp": t.vrp,
                "pricing_method": t.pricing_method,
                "vol_skew": t.vol_skew,
                "funding_bias": t.funding_bias,
                "funding_regime": t.funding_regime,
                "price_points_available": t.price_points_available,
            }
            for t in result.trades
        ],
        "equity_curve": result.equity_curve,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results exported to {output_path}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Backtest V4 — Real Polymarket Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_v4.py                           # Last 24 hours
  python backtest_v4.py --days 7                  # Last 7 days
  python backtest_v4.py --start 2026-02-21 --end 2026-02-28
  python backtest_v4.py --verbose                 # Print each trade
  python backtest_v4.py --cache-dir ./cache       # Cache API responses
  python backtest_v4.py --no-confirmation         # Pure quant model
  python backtest_v4.py --taker                   # Taker fills (1.56% max)
  python backtest_v4.py --output results.json     # Export to JSON
        """,
    )
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1, help="Days to backtest (default: 1)")
    parser.add_argument("--decision-minute", type=int, default=3,
                        help="Minute within window to decide (default: 3, matches live TRADE_WINDOW_START=180s)")  # [PATCH-3]
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", type=str, help="Export results to JSON")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache API responses to disk (recommended for re-runs)")
    parser.add_argument("--no-confirmation", action="store_true")
    parser.add_argument("--taker", action="store_true")
    parser.add_argument("--no-funding", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    use_maker = not args.taker
    use_funding = not args.no_funding

    # ── Date range ───────────────────────────────────────────────────────
    if args.start and args.end:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=args.days)

    print("=" * 90)
    print("  POLYMARKET BTC 15-MIN — V4 REAL DATA BACKTESTER")
    print("=" * 90)
    print(f"  Period:            {start_dt.date()} → {end_dt.date()}")
    print(f"  Model:             Merton Jump-Diffusion + Mean Reversion + Seasonality")
    print(f"  Sizing:            Half-Kelly, capped at 5% of ${BANKROLL_USD:.0f} bankroll")
    print(f"  Fees:              {'Maker (0%)' if use_maker else 'Taker (nonlinear, max 1.56%)'}")
    print(f"  Min edge:          ${MIN_EDGE_CENTS:.2f}")
    print(f"  Decision minute:   {args.decision_minute}")
    print(f"  Trade window:      {TRADE_WINDOW_START_SEC}s–{TRADE_WINDOW_END_SEC}s (synced with live)")  # [PATCH-8]
    print(f"  Quote stability:   {QUOTE_STABILITY_REQUIRED} ticks required")
    print(f"  Confirmation:      {'ON' if not args.no_confirmation else 'OFF'}")
    print(f"  Funding filter:    {'ON' if use_funding else 'OFF'}")
    print(f"  Cache:             {args.cache_dir or 'disabled'}")
    print(f"  Data source:       ★ REAL Polymarket CLOB prices ★")
    print()

    # ── Fetch real Polymarket data ───────────────────────────────────────
    total_slugs = ((end_dt - start_dt).total_seconds()) // 900
    print(f"Phase 1: Fetching resolved Polymarket markets (~{int(total_slugs)} slugs)...")
    print(f"  Using: GET /markets/slug/btc-updown-15m-{{timestamp}}")
    print(f"  ETA: ~{int(total_slugs * 0.13)}s with rate limiting (cached on re-run)")
    fetcher = PolymarketDataFetcher(cache_dir=args.cache_dir)
    try:
        windows = fetcher.build_windows(start_dt, end_dt)
    finally:
        fetcher.close()

    if not windows:
        print("ERROR: No Polymarket windows with price data found")
        print("  Check: are the Polymarket APIs accessible from this machine?")
        print("  Try:   curl 'https://gamma-api.polymarket.com/markets?slug=btc-updown-15m&closed=true&limit=1'")
        sys.exit(1)

    print(f"  ✓ {len(windows)} windows with real price data")

    # Price data quality summary
    pts_counts = [len(w.yes_prices) for w in windows]
    print(f"  Price points/window: min={min(pts_counts)}, "
          f"median={sorted(pts_counts)[len(pts_counts) // 2]}, "
          f"max={max(pts_counts)}")

    resolutions = sum(1 for w in windows if w.resolved_up)
    print(f"  Resolved UP: {resolutions}/{len(windows)} "
          f"({resolutions / len(windows) * 100:.1f}%)")
    print()

    # ── Fetch Binance candles (for vol estimator) ────────────────────────
    print("Phase 2: Fetching Binance 1m candles (for vol estimator)...")
    candles = fetch_binance_klines(start_dt - timedelta(hours=2), end_dt)
    print(f"  ✓ {len(candles)} candles")
    print()

    # ── Fetch funding rates ──────────────────────────────────────────────
    funding_snapshots = None
    if use_funding:
        print("Phase 3: Fetching Binance funding rates...")
        try:
            funding_snapshots = fetch_funding_rates(start_dt, end_dt)
            print(f"  ✓ {len(funding_snapshots)} snapshots")
        except Exception as e:
            print(f"  WARNING: {e} — proceeding without funding")
            use_funding = False
    print()

    # ── Run backtest ─────────────────────────────────────────────────────
    print("Phase 4: Running strategy against real market data...")
    print()
    result = run_backtest_v4(
        windows=windows,
        candles=candles,
        decision_minute=args.decision_minute,
        verbose=args.verbose,
        use_confirmation=not args.no_confirmation,
        use_maker=use_maker,
        use_funding=use_funding,
        funding_snapshots=funding_snapshots,
    )

    print_results(result)

    if args.output:
        export_results(result, args.output)


if __name__ == "__main__":
    main()