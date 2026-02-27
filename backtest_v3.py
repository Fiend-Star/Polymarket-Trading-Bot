"""
Backtest Engine V3.1 — BTC 15-Minute Polymarket Trading Strategy
================================================================

Tests the SAME V3.1 quant strategy the live bot runs:
  1. VolEstimator builds realized vol + jump params from 1-min candles
  2. BinaryOptionPricer (Merton JD + overlays) computes model fair value
  3. MispricingDetector compares model vs synthetic market price
     V3.1: + vol skew correction + funding rate bias
  4. Kelly criterion sizes the bet
  5. Old fusion processors provide CONFIRMATION (same as live bot)

V3.1 additions over V3:
  - Vol skew correction (BTC crash risk premium per moneyness)
  - Historical Binance funding rate filter (crowding bias)
  - Funding regime breakdown in results

Usage:
    python backtest_v3.py                          # Last 7 days
    python backtest_v3.py --days 30                # Last 30 days
    python backtest_v3.py --start 2026-02-01 --end 2026-02-27
    python backtest_v3.py --csv btc_1m_candles.csv
    python backtest_v3.py --verbose                # Print each trade
    python backtest_v3.py --output results.json
    python backtest_v3.py --no-confirmation        # Skip fusion confirmation
    python backtest_v3.py --no-funding             # Disable funding rate overlay
    python backtest_v3.py --taker                  # Simulate taker fills

Requirements:
    - binary_pricer.py, vol_estimator.py, mispricing_detector.py (V2)
    - No Polymarket credentials, no Redis, no NautilusTrader
    - Uses free Binance public kline + funding rate APIs (or local CSV)
"""

import argparse
import json
import math
import os
import random
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
from mispricing_detector import (
    MispricingDetector,
    polymarket_taker_fee,
    kelly_fraction,
)

# ── V3.1: Funding rate filter (optional — uses Binance historical) ──────────
try:
    from funding_rate_filter import FundingRateFilter
    FUNDING_AVAILABLE = True
except ImportError:
    FUNDING_AVAILABLE = False

# ── Old fusion processors (for CONFIRMATION, same as live bot) ───────────────
try:
    from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
    from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
    from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
    from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
    from core.strategy_brain.fusion_engine.signal_fusion import SignalFusionEngine
    FUSION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Fusion processors not available: {e}")
    FUSION_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# V3 quant parameters (from .env)
MIN_EDGE_CENTS      = float(os.getenv("MIN_EDGE_CENTS", "0.02"))
TAKE_PROFIT_PCT     = float(os.getenv("TAKE_PROFIT_PCT", "0.30"))
CUT_LOSS_PCT        = float(os.getenv("CUT_LOSS_PCT", "-0.50"))
VOL_METHOD          = os.getenv("VOL_METHOD", "ewma")
DEFAULT_VOL         = float(os.getenv("DEFAULT_VOL", "0.65"))
BANKROLL_USD        = float(os.getenv("BANKROLL_USD", "20.0"))
MARKET_BUY_USD      = float(os.getenv("MARKET_BUY_USD", "1.00"))
USE_LIMIT_ORDERS    = os.getenv("USE_LIMIT_ORDERS", "true").lower() == "true"

# Old fusion params (for confirmation / heuristic mode)
MIN_FUSION_SIGNALS  = int(os.getenv("MIN_FUSION_SIGNALS", "2"))
MIN_FUSION_SCORE    = float(os.getenv("MIN_FUSION_SCORE", "55.0"))
TREND_UP_THRESHOLD  = float(os.getenv("TREND_UP_THRESHOLD", "0.60"))
TREND_DOWN_THRESHOLD = float(os.getenv("TREND_DOWN_THRESHOLD", "0.40"))

WINDOW_MINUTES = 15


# =============================================================================
# Data structures
# =============================================================================
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
    direction: str              # "long" or "short"
    entry_price: float          # synthetic Polymarket price paid
    position_size: float        # USD size (Kelly-determined)
    actual_btc_open: float
    actual_btc_close: float
    btc_moved_up: bool
    outcome: str                # "WIN" or "LOSS"
    pnl: float                  # dollar P&L (after fees)
    pnl_before_fees: float
    fee_paid: float

    # V3 quant fields
    model_yes_price: float      # Jump-diffusion fair value
    model_no_price: float
    edge: float                 # model - market
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

    # Timing
    time_remaining_min: float


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    strategy_mode: str          # "quant_v3" or "heuristic_v1"
    total_windows: int
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
# Binance kline fetcher
# =============================================================================
BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"


def _parse_binance_kline(k) -> Candle:
    """Parse a single Binance kline array into a Candle."""
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
    """Load candles from CSV (columns: timestamp, open, high, low, close, volume)."""
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
                if ts_val > 1e12:
                    ts_val /= 1000
                ts = datetime.fromtimestamp(ts_val, tz=timezone.utc)

            candles.append(Candle(
                timestamp=ts, open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    candles.sort(key=lambda c: c.timestamp)
    logger.info(f"Loaded {len(candles)} candles from {csv_path}")
    return candles


# =============================================================================
# Historical funding rate (Binance Futures — free, no auth)
# =============================================================================
@dataclass
class FundingSnapshot:
    timestamp: datetime
    rate: float           # e.g. 0.0001 = 0.01%
    rate_pct: float       # e.g. 0.01
    classification: str   # EXTREME_POSITIVE, HIGH_POSITIVE, NEUTRAL, etc.
    mean_reversion_bias: float  # ±0.02 max


FUNDING_EXTREME_THRESHOLD = 0.0005   # 0.05%/8h
FUNDING_HIGH_THRESHOLD    = 0.0002   # 0.02%/8h
FUNDING_MAX_BIAS          = 0.02     # ±2%


def classify_funding(rate: float) -> Tuple[str, float]:
    """Classify funding rate into regime and compute mean-reversion bias."""
    rate_pct = rate * 100
    if rate >= FUNDING_EXTREME_THRESHOLD:
        return "EXTREME_POSITIVE", -FUNDING_MAX_BIAS
    elif rate >= FUNDING_HIGH_THRESHOLD:
        scale = (rate - FUNDING_HIGH_THRESHOLD) / (FUNDING_EXTREME_THRESHOLD - FUNDING_HIGH_THRESHOLD)
        return "HIGH_POSITIVE", -(0.005 + scale * 0.015)
    elif rate <= -FUNDING_EXTREME_THRESHOLD:
        return "EXTREME_NEGATIVE", FUNDING_MAX_BIAS
    elif rate <= -FUNDING_HIGH_THRESHOLD:
        scale = (-rate - FUNDING_HIGH_THRESHOLD) / (FUNDING_EXTREME_THRESHOLD - FUNDING_HIGH_THRESHOLD)
        return "HIGH_NEGATIVE", (0.005 + scale * 0.015)
    else:
        return "NEUTRAL", 0.0


def _parse_funding_entry(entry) -> FundingSnapshot:
    """Parse a single funding rate API entry."""
    ts = datetime.fromtimestamp(entry["fundingTime"] / 1000, tz=timezone.utc)
    rate = float(entry["fundingRate"])
    classification, bias = classify_funding(rate)
    return FundingSnapshot(
        timestamp=ts, rate=rate, rate_pct=rate * 100,
        classification=classification, mean_reversion_bias=bias)


def fetch_funding_rates(start_dt: datetime, end_dt: datetime) -> List[FundingSnapshot]:
    """Fetch historical BTCUSDT funding rates from Binance Futures (every 8h)."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    snapshots: List[FundingSnapshot] = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    logger.info(f"Fetching Binance funding rates: {start_dt.date()} to {end_dt.date()}")

    with httpx.Client(timeout=30.0) as client:
        cursor = start_ms
        while cursor < end_ms:
            params = {"symbol": "BTCUSDT", "startTime": cursor, "endTime": end_ms, "limit": 1000}
            try:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"Funding rate fetch failed: {e}")
                break
            if not data:
                break
            snapshots.extend(_parse_funding_entry(e) for e in data)
            cursor = int(data[-1]["fundingTime"]) + 1
            time.sleep(0.15)

    logger.info(f"Fetched {len(snapshots)} funding rate snapshots")
    return snapshots


def get_funding_bias_at(snapshots: List[FundingSnapshot], ts: datetime) -> float:
    """Get the most recent funding bias for a given timestamp."""
    if not snapshots:
        return 0.0
    # Find the most recent snapshot <= ts
    best = None
    for s in snapshots:
        if s.timestamp <= ts:
            best = s
        else:
            break
    return best.mean_reversion_bias if best else 0.0



# =============================================================================
# Realistic simulation config
# =============================================================================
@dataclass
class RealisticConfig:
    """
    Calibrated from live Polymarket BTC 15-min market observations.

    The default backtest has 3 inflation sources:
    1. Dumb synthetic market (linear sensitivity, no noise)
    2. No bid-ask spread (enter at mid, not ask)
    3. 100% fill rate (every limit order fills)

    And the strategy has 1 leak:
    4. Small-edge trades ($0.02-$0.05) that are 52% WR = coin flips after spread

    Realistic mode fixes all 4, AND adds strategy improvements that
    genuinely increase live performance via better selectivity.
    """
    # ── Market simulation ─────────────────────────────────────
    market_noise_std: float = 0.025     # ±2.5 cent random noise on synthetic prob
    market_efficiency: float = 0.35     # Market already reflects 35% of model's view
    spread_cents: float = 0.03          # 3-cent bid-ask spread (1.5c each side)
    spread_widen_late: float = 0.02     # Spread widens +2c in last 3 minutes

    # ── Execution simulation ──────────────────────────────────
    fill_rate: float = 0.72             # Limit orders at bid fill 72% of time
    fill_rate_boost_edge: float = 0.10  # +10% fill rate per $0.10 edge (bigger edges = more aggressive)

    # ── Strategy improvements ─────────────────────────────────
    min_edge_realistic: float = 0.05    # Raise floor from $0.02 → $0.05 (kills 52% WR noise)
    min_vol_confidence: float = 0.40    # Don't trade when vol estimate is unreliable
    max_consecutive_losses: int = 3     # Skip 1 window after 3 straight losses
    max_trade_rate: float = 0.60        # Cap at 60% of windows traded (selectivity)

    enabled: bool = False


# =============================================================================
# Synthetic Polymarket probability from BTC price action
# =============================================================================
def _intrawindow_vol(window_candles, decision_minute):
    """Compute intra-window realized vol from candle returns."""
    if len(window_candles) <= 1:
        return 0.0
    returns = []
    for i in range(1, min(decision_minute + 1, len(window_candles))):
        prev = window_candles[i - 1].close
        curr = window_candles[i].close
        if prev > 0:
            returns.append((curr - prev) / prev)
    if not returns:
        return 0.0
    return (sum(r ** 2 for r in returns) / len(returns)) ** 0.5


def _realistic_probability(pct_change, realistic, model_fair_value, window_candles, decision_minute):
    """Compute realistic-mode synthetic probability."""
    iwv = _intrawindow_vol(window_candles, decision_minute)
    vol_factor = max(0.5, min(1.2, 1.2 - iwv * 80))
    base_prob = 0.50 + pct_change * 11.0 * vol_factor

    if model_fair_value is not None:
        base_prob = (realistic.market_efficiency * model_fair_value
                     + (1.0 - realistic.market_efficiency) * base_prob)

    return base_prob + random.gauss(0, realistic.market_noise_std)


def btc_to_probability(
    window_candles: List[Candle],
    decision_minute: int = 2,
    realistic: Optional[RealisticConfig] = None,
    model_fair_value: Optional[float] = None,
) -> float:
    """Convert BTC price movement into a synthetic Polymarket 'Up' probability."""
    if len(window_candles) < decision_minute + 1:
        return 0.50
    window_open = window_candles[0].open
    current = window_candles[decision_minute].close
    if window_open == 0:
        return 0.50

    pct_change = (current - window_open) / window_open

    if realistic and realistic.enabled:
        prob = _realistic_probability(pct_change, realistic, model_fair_value, window_candles, decision_minute)
    else:
        prob = 0.50 + pct_change * 15.0

    return max(0.05, min(0.95, prob))


# =============================================================================
# RSI helper for synthetic sentiment
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
# Fusion confirmation (mirrors V3 bot Step 7-8)
# =============================================================================
def create_fusion_processors() -> Optional[Dict]:
    """Create old signal processors for confirmation role."""
    if not FUSION_AVAILABLE:
        return None
    return {
        "spike": SpikeDetectionProcessor(
            spike_threshold=float(os.getenv("SPIKE_THRESHOLD", "0.05")),
            lookback_periods=20,
        ),
        "sentiment": SentimentProcessor(
            extreme_fear_threshold=25, extreme_greed_threshold=75,
        ),
        "divergence": PriceDivergenceProcessor(
            divergence_threshold=float(os.getenv("DIVERGENCE_THRESHOLD", "0.05")),
        ),
        "tick_velocity": TickVelocityProcessor(
            velocity_threshold_60s=0.015, velocity_threshold_30s=0.010,
        ),
    }


def _collect_processor_signals(processors, current_price, price_history, pm):
    """Run all fusion processors and collect signals."""
    signals = []
    for name in ["spike", "sentiment", "divergence", "tick_velocity"]:
        proc = processors.get(name)
        if proc is None:
            continue
        try:
            sig = proc.process(current_price=current_price, historical_prices=price_history, metadata=pm)
            if sig:
                signals.append(sig)
        except Exception:
            pass
    return signals


def _count_confirmations(signals, model_bullish):
    """Count confirming vs contradicting signals."""
    confirming = contradicting = 0
    for sig in signals:
        if ("BULLISH" in str(sig.direction).upper()) == model_bullish:
            confirming += 1
        else:
            contradicting += 1
    return confirming, contradicting


def run_confirmation(processors, current_price, price_history, metadata, model_bullish):
    """Run old signal processors as CONFIRMATION. Returns (confirming, contradicting)."""
    if processors is None:
        return (0, 0)
    pm = {k: (Decimal(str(v)) if isinstance(v, float) else v) for k, v in metadata.items()}
    signals = _collect_processor_signals(processors, current_price, price_history, pm)
    return _count_confirmations(signals, model_bullish)


def _build_tick_buffer(window_candles, decision_minute):
    """Build synthetic tick buffer from candle data."""
    tick_buffer = []
    start_idx = max(0, decision_minute - 2)
    for i in range(start_idx, min(decision_minute + 1, len(window_candles))):
        c = window_candles[i]
        for j, px in enumerate([c.open, (c.open + c.close) / 2, c.close]):
            tick_ts = c.timestamp + timedelta(seconds=j * 20)
            tick_buffer.append({"ts": tick_ts, "price": Decimal(str(round(px, 4)))})
    return tick_buffer


def build_confirmation_metadata(window_candles, decision_minute, historical_btc_closes):
    """Build metadata dict for confirmation signal processors."""
    if len(window_candles) <= decision_minute:
        return {}
    current_close = window_candles[decision_minute].close
    recent = historical_btc_closes[-20:]
    sma_20 = sum(recent) / len(recent) if recent else current_close
    deviation = (current_close - sma_20) / sma_20 if sma_20 != 0 else 0.0
    if len(historical_btc_closes) >= 5:
        p5 = historical_btc_closes[-5]
        momentum = (current_close - p5) / p5 if p5 != 0 else 0.0
    else:
        momentum = 0.0
    variance = sum((p - sma_20) ** 2 for p in recent) / len(recent) if recent else 0.0
    rsi = compute_rsi(historical_btc_closes, period=14)
    return {
        "deviation": deviation, "momentum": momentum,
        "volatility": math.sqrt(variance), "sentiment_score": rsi,
        "sentiment_classification": _classify_sentiment(rsi),
        "spot_price": current_close,
        "tick_buffer": _build_tick_buffer(window_candles, decision_minute),
        "yes_token_id": None,
    }


def _classify_sentiment(score: float) -> str:
    if score < 25: return "Extreme Fear"
    elif score < 45: return "Fear"
    elif score < 55: return "Neutral"
    elif score < 75: return "Greed"
    else: return "Extreme Greed"


# =============================================================================
# Backtest helpers (Rule of 30 decomposition)
# =============================================================================

def _init_backtest_modules():
    """Initialize quant modules for backtesting."""
    pricer = BinaryOptionPricer()
    vol_est = VolEstimator(
        window_minutes=60.0, resample_interval_sec=60.0,
        ewma_halflife_samples=20, min_samples=5, default_vol=DEFAULT_VOL,
    )
    detector = MispricingDetector(
        pricer=pricer, vol_estimator=vol_est, maker_fee=0.00,
        min_edge_cents=MIN_EDGE_CENTS, min_edge_after_fees=0.005,
        take_profit_pct=TAKE_PROFIT_PCT, cut_loss_pct=CUT_LOSS_PCT,
        vol_method=VOL_METHOD, bankroll=BANKROLL_USD,
        use_half_kelly=True, max_kelly_fraction=0.05,
    )
    return vol_est, detector


def _chunk_candles_to_windows(candles: List[Candle]) -> List[List[Candle]]:
    """Chunk 1-min candles into 15-minute windows."""
    windows: List[List[Candle]] = []
    current_window: List[Candle] = []
    window_start: Optional[datetime] = None
    for c in candles:
        minute_slot = c.timestamp.minute // WINDOW_MINUTES
        slot_start = c.timestamp.replace(
            minute=minute_slot * WINDOW_MINUTES, second=0, microsecond=0)
        if window_start is None:
            window_start = slot_start
        if slot_start != window_start:
            if len(current_window) >= WINDOW_MINUTES:
                windows.append(current_window)
            current_window = [c]
            window_start = slot_start
        else:
            current_window.append(c)
    if len(current_window) >= WINDOW_MINUTES:
        windows.append(current_window)
    logger.info(f"Created {len(windows)} complete 15-minute windows")
    return windows


def _get_funding_at_window(use_funding, funding_snapshots, window_ts):
    """Get funding bias and regime for a window timestamp."""
    if use_funding and funding_snapshots:
        bias = get_funding_bias_at(funding_snapshots, window_ts)
        regime = "NEUTRAL"
        for s in funding_snapshots:
            if s.timestamp <= window_ts:
                regime = s.classification
            else:
                break
        return bias, regime
    return 0.0, "DISABLED"


def _apply_realistic_entry(rc, entry_price, time_remaining_min):
    """Apply realistic spread to entry price."""
    if not rc:
        return entry_price
    half_spread = rc.spread_cents / 2.0
    if time_remaining_min <= 3.0:
        half_spread += rc.spread_widen_late / 2.0
    return min(0.95, entry_price + half_spread)


def _check_realistic_fill(rc, signal):
    """Return True if order fills in realistic mode."""
    if not rc:
        return True
    edge_boost = min(0.15, abs(signal.edge) * rc.fill_rate_boost_edge / 0.10)
    eff = min(0.95, rc.fill_rate + edge_boost)
    return random.random() <= eff


def _compute_position_size(rc, signal, bankroll):
    """Compute position size with optional realistic edge-scaling."""
    if rc:
        ae = abs(signal.edge)
        if ae < 0.10:
            scale = 0.6
        elif ae < 0.20:
            scale = 1.0
        else:
            scale = 1.5
        cap = MARKET_BUY_USD * scale
        ps = min(signal.kelly_bet_usd, bankroll * 0.05, cap)
    else:
        ps = min(signal.kelly_bet_usd, bankroll * 0.05, MARKET_BUY_USD)
    return max(ps, 0.10)


def _compute_trade_pnl(direction, entry_price, btc_moved_up, position_size, use_maker):
    """Compute PnL and fees for a single trade."""
    won = btc_moved_up if direction == "long" else not btc_moved_up
    outcome = "WIN" if won else "LOSS"
    num_tokens = position_size / entry_price if entry_price > 0 else 0
    fee_rate = 0.0 if use_maker else polymarket_taker_fee(entry_price)
    fee = fee_rate * num_tokens * entry_price
    pnl_bf = num_tokens * (1.0 - entry_price) if won else -num_tokens * entry_price
    return won, outcome, pnl_bf - fee, pnl_bf, fee


def _check_window_skip(hist_len, window_len, dm, rc, closs, tts, wp, sr):
    """Check if window should be skipped. Returns reason string or None."""
    if hist_len < 30:
        sr["warmup"] += 1
        return "warmup"
    if window_len <= dm:
        sr["no_data"] += 1
        return "no_data"
    if rc and closs >= rc.max_consecutive_losses:
        sr["streak_throttle"] += 1
        return "streak_throttle"
    if rc and wp > 20 and tts / wp > rc.max_trade_rate:
        sr["trade_rate_cap"] += 1
        return "trade_rate_cap"
    return None


def _run_backtest_confirmation(use_conf, fproc, ph, window, dm, hbc, pp, mb):
    """Run fusion confirmation; returns (confirming, contradicting)."""
    if not (use_conf and fproc and len(ph) >= 5):
        return 0, 0
    md = build_confirmation_metadata(window, dm, hbc[-100:])
    if not md:
        return 0, 0
    return run_confirmation(fproc, Decimal(str(round(pp, 4))), list(ph), md, mb)


def _build_backtest_trade(window, direction, entry_price, position_size,
                          btc_open, btc_close, btc_moved_up, outcome,
                          pnl, pnl_bf, fee, signal, conf, contra,
                          vol_skew, funding_bias, funding_regime, trm):
    """Build a BacktestTrade dataclass instance."""
    return BacktestTrade(
        window_start=window[0].timestamp, direction=direction,
        entry_price=round(entry_price, 4), position_size=round(position_size, 2),
        actual_btc_open=btc_open, actual_btc_close=btc_close,
        btc_moved_up=btc_moved_up, outcome=outcome,
        pnl=round(pnl, 4), pnl_before_fees=round(pnl_bf, 4),
        fee_paid=round(fee, 4),
        model_yes_price=round(signal.yes_model, 4),
        model_no_price=round(signal.no_model, 4),
        edge=round(signal.edge, 4), edge_pct=round(signal.edge_pct, 4),
        kelly_fraction=round(signal.kelly_fraction, 4),
        realized_vol=round(signal.realized_vol, 4),
        implied_vol=round(signal.implied_vol, 4),
        vrp=round(signal.vrp, 4), pricing_method=signal.pricing_method,
        confirming_signals=conf, contradicting_signals=contra,
        vol_skew=round(vol_skew, 8), funding_bias=round(funding_bias, 6),
        funding_regime=funding_regime, time_remaining_min=trm,
    )


def _compute_max_drawdown(equity_curve):
    peak = max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_profit_factor(trades):
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl < 0))
    return (gp / gl) if gl > 0 else float("inf")


def _compute_sharpe(trades):
    if len(trades) <= 1:
        return 0.0
    pnls = [t.pnl for t in trades]
    mu = sum(pnls) / len(pnls)
    sd = math.sqrt(sum((p - mu) ** 2 for p in pnls) / (len(pnls) - 1))
    return (mu / sd * math.sqrt(96 * 365)) if sd > 0 else 0.0


def _compute_backtest_stats(trades, eq, tw, skipped, candles, bankroll, tf, rc):
    """Compute all backtest statistics and return BacktestResult."""
    n = len(trades)
    wins = sum(1 for t in trades if t.outcome == "WIN")
    tp = sum(t.pnl for t in trades)
    return BacktestResult(
        start_date=candles[0].timestamp.strftime("%Y-%m-%d") if candles else "",
        end_date=candles[-1].timestamp.strftime("%Y-%m-%d") if candles else "",
        strategy_mode="quant_v3.1_realistic" if (rc and rc.enabled) else "quant_v3.1",
        total_windows=tw, trades_taken=n, trades_skipped=skipped,
        wins=wins, losses=n - wins,
        win_rate=round((wins / n * 100) if n else 0.0, 2),
        total_pnl=round(tp, 4), total_fees=round(tf, 4),
        max_drawdown=round(_compute_max_drawdown(eq), 4),
        profit_factor=round(_compute_profit_factor(trades), 2),
        avg_pnl_per_trade=round((tp / n) if n else 0.0, 4),
        avg_edge_per_trade=round((sum(abs(t.edge) for t in trades) / n) if n else 0.0, 4),
        avg_kelly=round((sum(t.kelly_fraction for t in trades) / n) if n else 0.0, 4),
        best_trade=round(max((t.pnl for t in trades), default=0.0), 4),
        worst_trade=round(min((t.pnl for t in trades), default=0.0), 4),
        sharpe_ratio=round(_compute_sharpe(trades), 2),
        bankroll_final=round(bankroll, 2),
        bankroll_growth=round(((bankroll - BANKROLL_USD) / BANKROLL_USD * 100) if BANKROLL_USD > 0 else 0.0, 2),
        trades=trades, equity_curve=eq,
    )


def _log_backtest_trade(wi, tw, w, bo, bc, bmu, d, ep, sig, ps, won, pnl, fee, cum, br, fr, fb):
    """Log a single backtest trade in verbose mode."""
    arrow = "^" if bmu else "v"
    emoji = "✅" if won else "❌"
    ft = f" fund={fr[:3]}" if fb != 0 else ""
    logger.info(
        f"W{wi+1:>4}/{tw} | {w[0].timestamp.strftime('%m-%d %H:%M')} | "
        f"BTC ${bo:,.0f}{arrow}${bc:,.0f} | {d:>5} @{ep:.2f} "
        f"(model={sig.yes_model:.2f}/{sig.no_model:.2f}) | "
        f"edge={sig.edge:+.3f} kelly={sig.kelly_fraction:.1%} ${ps:.2f} | "
        f"{emoji} ${pnl:+.4f} (fee=${fee:.4f}) | cum=${cum:+.2f} bank=${br:.2f} | "
        f"RV={sig.realized_vol:.0%} IV={sig.implied_vol:.0%} "
        f"VRP={sig.vrp:+.0%} [{sig.pricing_method}]{ft}"
    )


# =============================================================================
# Main backtest loop — V3 QUANT STRATEGY (decomposed)
# =============================================================================

def _get_signal_for_window(vol_est, detector, window, decision_minute, rc, lmy, use_funding, funding_snapshots, use_maker, ph):
    """Compute market price, vol skew, funding, and detect signal for a window."""
    bo = window[0].open
    bs = window[decision_minute].close
    trm = float(WINDOW_MINUTES - decision_minute)

    pp = btc_to_probability(window, decision_minute, realistic=rc, model_fair_value=lmy)
    ph.append(Decimal(str(round(pp, 4))))
    if len(ph) > 100:
        ph.pop(0)

    ve = vol_est.get_vol(VOL_METHOD)
    vs = BinaryOptionPricer.estimate_btc_vol_skew(spot=bs, strike=bo, vol=ve.annualized_vol, time_remaining_min=trm)
    fb, fr = _get_funding_at_window(use_funding, funding_snapshots, window[decision_minute].timestamp)

    sig = detector.detect(
        yes_market_price=pp, no_market_price=1.0 - pp,
        btc_spot=bs, btc_strike=bo, time_remaining_min=trm,
        position_size_usd=MARKET_BUY_USD, use_maker=use_maker, vol_skew=vs, funding_bias=fb)

    return pp, vs, fb, fr, sig, trm


def _should_skip_signal(sig, rc, sr, use_conf, fproc, ph, window, dm, hbc, pp):
    """Check if a tradeable signal should be skipped due to filters. Returns True if skip."""
    if rc and sig.vol_confidence < rc.min_vol_confidence:
        sr["vol_confidence"] += 1; return True
    if rc and abs(sig.edge) < rc.min_edge_realistic:
        sr["edge_too_small"] += 1; return True
    mb = sig.direction == "BUY_YES"
    conf, contra = _run_backtest_confirmation(use_conf, fproc, ph, window, dm, hbc, pp, mb)
    if contra > conf and sig.confidence < 0.6:
        sr["confirmation_veto"] += 1; return True
    return False


def _execute_window_trade(sig, pp, rc, trm, use_maker, bankroll):
    """Compute direction, entry, fill, size, pnl for a single window. Returns None if no fill."""
    d = "long" if sig.direction == "BUY_YES" else "short"
    ep = pp if d == "long" else 1.0 - pp
    ep = _apply_realistic_entry(rc, ep, trm)
    if not _check_realistic_fill(rc, sig):
        return None
    ps = _compute_position_size(rc, sig, bankroll)
    return d, ep, ps


def _init_backtest_state(realistic):
    """Initialize mutable backtest state."""
    return {
        "ph": [], "hbc": [], "trades": [], "eq": [0.0],
        "bankroll": BANKROLL_USD, "tf": 0.0, "skipped": 0,
        "sr": {"warmup": 0, "no_edge": 0, "confirmation_veto": 0, "theta_decay": 0,
               "no_data": 0, "funding_veto": 0, "vol_confidence": 0,
               "streak_throttle": 0, "no_fill": 0, "edge_too_small": 0, "trade_rate_cap": 0},
        "rc": realistic if (realistic and realistic.enabled) else None,
        "closs": 0, "tts": 0, "wp": 0, "lmy": None,
    }


def _record_trade(st, window, d, ep, ps, sig, vs, fb, fr, trm, use_maker, use_conf, fproc, dm, pp, wi, tw, verbose):
    """Record a trade result and update backtest state. Returns pnl."""
    bo, bc = window[0].open, window[-1].close
    won, _, pnl, pnl_bf, fee = _compute_trade_pnl(d, ep, bc > bo, ps, use_maker)
    st["closs"] = 0 if won else st["closs"] + 1; st["tts"] += 1; st["tf"] += fee
    st["bankroll"] = max(1.0, st["bankroll"] + pnl)
    cum = st["eq"][-1] + pnl; st["eq"].append(cum)
    conf, contra = _run_backtest_confirmation(use_conf, fproc, st["ph"], window, dm, st["hbc"], pp, sig.direction == "BUY_YES")
    st["trades"].append(_build_backtest_trade(window, d, ep, ps, bo, bc, bc > bo, "WIN" if won else "LOSS", pnl, pnl_bf, fee, sig, conf, contra, vs, fb, fr, trm))
    if verbose:
        _log_backtest_trade(wi, tw, window, bo, bc, bc > bo, d, ep, sig, ps, won, pnl, fee, cum, st["bankroll"], fr, fb)


def _process_window(wi, window, st, vol_est, detector, fproc, dm, use_conf, use_maker, use_funding, fs, tw, verbose):
    """Process a single backtest window. Modifies state dict in place."""
    for c in window:
        vol_est.add_price(c.close, c.timestamp.timestamp()); st["hbc"].append(c.close)
    vol_est.set_simulated_time(window[min(dm, len(window)-1)].timestamp.timestamp())
    skip = _check_window_skip(len(st["hbc"]), len(window), dm, st["rc"], st["closs"], st["tts"], st["wp"], st["sr"])
    if skip:
        st["skipped"] += 1
        if skip == "streak_throttle": st["closs"] = max(0, st["closs"]-1)
        return
    st["wp"] += 1
    pp, vs, fb, fr, sig, trm = _get_signal_for_window(
        vol_est, detector, window, dm, st["rc"], st["lmy"], use_funding, fs, use_maker, st["ph"])
    if not sig.is_tradeable:
        st["sr"]["no_edge"] += 1; st["skipped"] += 1; st["lmy"] = sig.yes_model; return
    st["lmy"] = sig.yes_model
    if _should_skip_signal(sig, st["rc"], st["sr"], use_conf, fproc, st["ph"], window, dm, st["hbc"], pp):
        st["skipped"] += 1; return
    result = _execute_window_trade(sig, pp, st["rc"], trm, use_maker, st["bankroll"])
    if result is None:
        st["sr"]["no_fill"] += 1; st["skipped"] += 1; return
    d, ep, ps = result
    _record_trade(st, window, d, ep, ps, sig, vs, fb, fr, trm, use_maker, use_conf, fproc, dm, pp, wi, tw, verbose)
    detector.update_bankroll(st["bankroll"])


def run_backtest_v3(
    candles: List[Candle], decision_minute: int = 2, verbose: bool = False,
    use_confirmation: bool = True, use_maker: bool = True, use_funding: bool = True,
    funding_snapshots: Optional[List[FundingSnapshot]] = None,
    realistic: Optional[RealisticConfig] = None,
) -> BacktestResult:
    """Backtest the V3.1 quant strategy."""
    vol_est, detector = _init_backtest_modules()
    fproc = create_fusion_processors() if use_confirmation else None
    windows = _chunk_candles_to_windows(candles)
    st = _init_backtest_state(realistic)
    tw = len(windows)

    for wi, window in enumerate(windows):
        _process_window(wi, window, st, vol_est, detector, fproc, decision_minute,
                        use_confirmation, use_maker, use_funding, funding_snapshots, tw, verbose)

    if verbose:
        logger.info(f"Skip reasons: {st['sr']}")
    return _compute_backtest_stats(
        st["trades"], st["eq"], tw, st["skipped"], candles, st["bankroll"], st["tf"], st["rc"])


# =============================================================================
# Pretty printer (split into sub-functions)
# =============================================================================

def _print_summary_header(r: BacktestResult):
    """Print the top-level summary section."""
    print()
    print("=" * 90)
    print(f"          BACKTEST RESULTS — {r.strategy_mode.upper()}")
    print("=" * 90)
    print(f"  Period:              {r.start_date} → {r.end_date}")
    print(f"  Total 15m windows:   {r.total_windows}")
    print(f"  Trades taken:        {r.trades_taken}")
    print(f"  Trades skipped:      {r.trades_skipped}")
    print(f"  Trade rate:          {r.trades_taken / max(r.total_windows, 1) * 100:.1f}%")
    print("-" * 90)
    print(f"  Wins / Losses:       {r.wins} / {r.losses}")
    print(f"  Win Rate:            {r.win_rate:.1f}%")
    print("-" * 90)
    print(f"  Total P&L:           ${r.total_pnl:+.4f}")
    print(f"  Total Fees Paid:     ${r.total_fees:.4f}")
    print(f"  Avg P&L / Trade:     ${r.avg_pnl_per_trade:+.4f}")
    print(f"  Avg Edge / Trade:    ${r.avg_edge_per_trade:.4f}")
    print(f"  Avg Kelly Fraction:  {r.avg_kelly:.1%}")
    print(f"  Best / Worst Trade:  ${r.best_trade:+.4f} / ${r.worst_trade:+.4f}")
    print(f"  Max Drawdown:        ${r.max_drawdown:.4f}")
    print(f"  Profit Factor:       {r.profit_factor:.2f}")
    print(f"  Sharpe Ratio (ann):  {r.sharpe_ratio:.2f}")
    print("-" * 90)
    print(f"  Bankroll:            ${BANKROLL_USD:.2f} → ${r.bankroll_final:.2f} ({r.bankroll_growth:+.1f}%)")
    print("=" * 90)


def _print_direction_and_vol(r: BacktestResult):
    """Print direction breakdown and vol analysis."""
    longs = [t for t in r.trades if t.direction == "long"]
    shorts = [t for t in r.trades if t.direction == "short"]
    lw = sum(1 for t in longs if t.outcome == "WIN")
    sw = sum(1 for t in shorts if t.outcome == "WIN")
    print(f"\n  Long:  {len(longs):>4} ({(lw / len(longs) * 100) if longs else 0:.1f}% WR) | "
          f"Short: {len(shorts):>4} ({(sw / len(shorts) * 100) if shorts else 0:.1f}% WR)")

    n = len(r.trades)
    arv = sum(t.realized_vol for t in r.trades) / n
    aiv = sum(t.implied_vol for t in r.trades) / n
    avrp = sum(t.vrp for t in r.trades) / n
    print(f"  Avg RV: {arv:.0%} | IV: {aiv:.0%} | VRP: {avrp:+.0%}")

    methods: Dict[str, int] = {}
    for t in r.trades:
        methods[t.pricing_method] = methods.get(t.pricing_method, 0) + 1
    print(f"\n  Pricing methods:")
    for m, c in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"    {m:<20} {c:>4} ({c / n * 100:.0f}%)")


def _print_funding_and_edge(r: BacktestResult):
    """Print funding regime breakdown and edge distribution."""
    regimes: Dict[str, List[BacktestTrade]] = {}
    for t in r.trades:
        regimes.setdefault(t.funding_regime, []).append(t)
    if any(t.funding_regime != "DISABLED" for t in r.trades):
        print(f"\n  Funding regime breakdown:")
        for regime, rt in sorted(regimes.items()):
            rw = sum(1 for t in rt if t.outcome == "WIN")
            rp = sum(t.pnl for t in rt)
            print(f"    {regime:<20} {len(rt):>4} trades, "
                  f"{(rw / len(rt) * 100) if rt else 0:.1f}% WR, PnL ${rp:+.4f}")

    edges = sorted([abs(t.edge) for t in r.trades])
    n = len(edges)
    print(f"\n  Edge dist: min=${edges[0]:.4f} | 25th=${edges[n//4]:.4f} | "
          f"med=${edges[n//2]:.4f} | 75th=${edges[3*n//4]:.4f} | max=${edges[-1]:.4f}")

    print(f"\n  Win rate by edge size:")
    for lo, hi in [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.00)]:
        bt = [t for t in r.trades if lo <= abs(t.edge) < hi]
        if bt:
            bw = sum(1 for t in bt if t.outcome == "WIN")
            print(f"    ${lo:.2f}-${hi:.2f}: {len(bt):>4}, {bw / len(bt) * 100:.1f}% WR, "
                  f"avg PnL ${sum(t.pnl for t in bt) / len(bt):+.4f}")


def _print_recent_trades(r: BacktestResult):
    """Print last 15 trades table."""
    print(f"\n  Last 15 trades:")
    print(f"  {'Time':<13} {'Dir':<6} {'Entry':<6} {'Model':<6} "
          f"{'Edge':<7} {'Kelly':<6} {'Size':<6} {'Result':<5} {'P&L':<9} {'Fee':<6} "
          f"{'RV':<5} {'IV':<5} {'Method'}")
    print(f"  {'-' * 105}")
    for t in r.trades[-15:]:
        mv = t.model_yes_price if t.direction == "long" else t.model_no_price
        print(
            f"  {t.window_start.strftime('%m-%d %H:%M'):<13} "
            f"{t.direction:<6} {t.entry_price:<6.2f} {mv:<6.2f} "
            f"{t.edge:+6.3f} {t.kelly_fraction:5.1%} "
            f"${t.position_size:<5.2f} "
            f"{'W' if t.outcome == 'WIN' else 'L':<5} "
            f"${t.pnl:+8.4f} ${t.fee_paid:<5.4f} "
            f"{t.realized_vol:4.0%} {t.implied_vol:4.0%} {t.pricing_method}")


def print_results(result: BacktestResult):
    """Pretty-print backtest results."""
    _print_summary_header(result)
    if not result.trades:
        print("\n  No trades taken.")
        return
    _print_direction_and_vol(result)
    _print_funding_and_edge(result)
    _print_recent_trades(result)
    print("\n  NOTES:")
    print("  • Market prices are synthetic | Fees: correct nonlinear formula")
    print("  • Sizing: half-Kelly, 5% cap | V3.1: vol skew + funding bias")
    print()


# =============================================================================
# Export
# =============================================================================

def export_results(result: BacktestResult, output_path: str):
    """Export results to JSON."""
    data = {
        "summary": {
            k: getattr(result, k) for k in [
                "strategy_mode", "start_date", "end_date", "total_windows",
                "trades_taken", "trades_skipped", "wins", "losses", "win_rate",
                "total_pnl", "total_fees", "max_drawdown", "profit_factor",
                "avg_pnl_per_trade", "avg_edge_per_trade", "avg_kelly",
                "sharpe_ratio", "bankroll_final", "bankroll_growth",
            ]
        },
        "trades": [
            {k: (getattr(t, k).isoformat() if k == "window_start" else getattr(t, k))
             for k in BacktestTrade.__dataclass_fields__}
            for t in result.trades
        ],
        "equity_curve": result.equity_curve,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results exported to {output_path}")


# =============================================================================
# CLI (split into helpers)
# =============================================================================

def _parse_args():
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Backtest V3.1 — BTC 15-Min Polymarket Quant Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--csv", type=str)
    p.add_argument("--decision-minute", type=int, default=2)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--output", "-o", type=str)
    p.add_argument("--no-confirmation", action="store_true")
    p.add_argument("--taker", action="store_true")
    p.add_argument("--no-funding", action="store_true")
    p.add_argument("--realistic", action="store_true")
    p.add_argument("--spread", type=float, default=None)
    p.add_argument("--fill-rate", type=float, default=None)
    return p.parse_args()


def _build_realistic_config(args):
    """Build RealisticConfig from CLI args."""
    rc = RealisticConfig(enabled=args.realistic)
    if args.spread is not None:
        rc.spread_cents = args.spread
    if args.fill_rate is not None:
        rc.fill_rate = args.fill_rate
    if rc.enabled:
        random.seed(42)
    return rc


def _print_config_banner(args, rc, use_maker, use_funding):
    """Print the configuration banner."""
    label = "V3.1 REALISTIC" if rc.enabled else "V3.1"
    print("=" * 90)
    print(f"  POLYMARKET BTC 15-MIN — {label} QUANT BACKTESTER")
    print("=" * 90)
    print(f"  Fees: {'Maker 0%' if use_maker else 'Taker (max 1.56%)'} | "
          f"Min edge: ${rc.min_edge_realistic if rc.enabled else MIN_EDGE_CENTS:.2f} | "
          f"Vol: {VOL_METHOD}")
    print(f"  Confirmation: {'ON' if not args.no_confirmation else 'OFF'} | "
          f"Funding: {'ON' if use_funding else 'OFF'} | "
          f"Decision min: {args.decision_minute}")
    if rc.enabled:
        print(f"  Realistic: spread={rc.spread_cents*100:.0f}¢ | "
              f"fill={rc.fill_rate:.0%} | noise=±{rc.market_noise_std*100:.1f}¢ | "
              f"eff={rc.market_efficiency:.0%}")
    print()


def _load_candles(args):
    """Load candle data from CSV or Binance API."""
    if args.csv:
        return load_csv_candles(args.csv)
    if args.start and args.end:
        s = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        e = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        e = datetime.now(timezone.utc)
        s = e - timedelta(days=args.days)
    return fetch_binance_klines(s, e)


def _load_funding(candles, use_funding):
    """Load historical funding rates if enabled."""
    if not use_funding:
        return None, use_funding
    try:
        snaps = fetch_funding_rates(candles[0].timestamp, candles[-1].timestamp)
        if snaps:
            rc: Dict[str, int] = {}
            for s in snaps:
                rc[s.classification] = rc.get(s.classification, 0) + 1
            print(f"  Funding: {len(snaps)} snapshots — " +
                  ", ".join(f"{k}: {v}" for k, v in sorted(rc.items())))
            print()
        return snaps, use_funding
    except Exception as ex:
        print(f"  WARNING: Funding rates unavailable: {ex}")
        return None, False


def main():
    args = _parse_args()
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    use_maker = not args.taker
    use_funding = not args.no_funding
    rc = _build_realistic_config(args)
    _print_config_banner(args, rc, use_maker, use_funding)

    candles = _load_candles(args)
    if not candles:
        print("ERROR: No candle data available")
        sys.exit(1)

    funding_snapshots, use_funding = _load_funding(candles, use_funding)
    result = run_backtest_v3(
        candles=candles, decision_minute=args.decision_minute,
        verbose=args.verbose, use_confirmation=not args.no_confirmation,
        use_maker=use_maker, use_funding=use_funding,
        funding_snapshots=funding_snapshots, realistic=rc)
    print_results(result)
    if args.output:
        export_results(result, args.output)


if __name__ == "__main__":
    main()
