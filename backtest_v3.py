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
MIN_EDGE_CENTS = float(os.getenv("MIN_EDGE_CENTS", "0.02"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.30"))
CUT_LOSS_PCT = float(os.getenv("CUT_LOSS_PCT", "-0.50"))
VOL_METHOD = os.getenv("VOL_METHOD", "ewma")
DEFAULT_VOL = float(os.getenv("DEFAULT_VOL", "0.65"))
BANKROLL_USD = float(os.getenv("BANKROLL_USD", "20.0"))
MARKET_BUY_USD = float(os.getenv("MARKET_BUY_USD", "1.00"))
USE_LIMIT_ORDERS = os.getenv("USE_LIMIT_ORDERS", "true").lower() == "true"

# V3 quant parameters continued...
WINDOW_MINUTES = 15

# V3.1: Gamma Scalping Config (matching bot.py)
GAMMA_EXIT_PROFIT_PCT = float(os.getenv("GAMMA_EXIT_PROFIT_PCT", "0.04"))
GAMMA_EXIT_TIME_MINS = float(os.getenv("GAMMA_EXIT_TIME_MINS", "3.0"))

# Old fusion params (for confirmation / heuristic mode)
MIN_FUSION_SIGNALS = int(os.getenv("MIN_FUSION_SIGNALS", "2"))
MIN_FUSION_SCORE = float(os.getenv("MIN_FUSION_SCORE", "55.0"))
TREND_UP_THRESHOLD = float(os.getenv("TREND_UP_THRESHOLD", "0.60"))
TREND_DOWN_THRESHOLD = float(os.getenv("TREND_DOWN_THRESHOLD", "0.40"))


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
    direction: str  # "long" or "short"
    entry_price: float  # synthetic Polymarket price paid
    position_size: float  # USD size (Kelly-determined)
    actual_btc_open: float
    actual_btc_close: float
    btc_moved_up: bool
    outcome: str  # "WIN" or "LOSS"
    pnl: float  # dollar P&L (after fees)
    pnl_before_fees: float
    fee_paid: float

    # V3 quant fields
    model_yes_price: float  # Jump-diffusion fair value
    model_no_price: float
    edge: float  # model - market
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
    strategy_mode: str  # "quant_v3" or "heuristic_v1"
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


def fetch_binance_klines(
        start_dt: datetime, end_dt: datetime,
        symbol: str = "BTCUSDT", interval: str = "1m",
) -> List[Candle]:
    """Fetch 1-minute BTC/USDT candles from Binance public API."""
    candles: List[Candle] = []
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    logger.info(f"Fetching Binance klines: {symbol} {interval} "
                f"from {start_dt.date()} to {end_dt.date()}")

    with httpx.Client(timeout=30.0) as client:
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": symbol, "interval": interval,
                "startTime": cursor, "endTime": end_ms, "limit": 1000,
            }
            resp = client.get(BINANCE_KLINE_URL, params=params)
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

    logger.info(f"Fetched {len(candles)} candles")
    return candles


def load_csv_candles(csv_path):
    """Load candles from CSV (columns: timestamp, open, high, low, close, volume)."""
    import csv
    candles = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_raw = row.get("timestamp")
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
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
    rate: float  # e.g. 0.0001 = 0.01%
    rate_pct: float  # e.g. 0.01
    classification: str  # EXTREME_POSITIVE, HIGH_POSITIVE, NEUTRAL, etc.
    mean_reversion_bias: float  # ±0.02 max


FUNDING_EXTREME_THRESHOLD = 0.0005  # 0.05%/8h
FUNDING_HIGH_THRESHOLD = 0.0002  # 0.02%/8h
FUNDING_MAX_BIAS = 0.02  # ±2%


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
            params = {
                "symbol": "BTCUSDT",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
            try:
                resp = client.get(url, params=params)
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
                    timestamp=ts,
                    rate=rate,
                    rate_pct=rate * 100,
                    classification=classification,
                    mean_reversion_bias=bias,
                ))

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
    market_noise_std: float = 0.025  # ±2.5 cent random noise on synthetic prob
    market_efficiency: float = 0.35  # Market already reflects 35% of model's view
    spread_cents: float = 0.03  # 3-cent bid-ask spread (1.5c each side)
    spread_widen_late: float = 0.02  # Spread widens +2c in last 3 minutes

    # ── Execution simulation ──────────────────────────────────
    fill_rate: float = 0.72  # Limit orders at bid fill 72% of time
    fill_rate_boost_edge: float = 0.10  # +10% fill rate per $0.10 edge (bigger edges = more aggressive)

    # ── Strategy improvements ─────────────────────────────────
    min_edge_realistic: float = 0.05  # Raise floor from $0.02 → $0.05 (kills 52% WR noise)
    min_vol_confidence: float = 0.40  # Don't trade when vol estimate is unreliable
    max_consecutive_losses: int = 3  # Skip 1 window after 3 straight losses
    max_trade_rate: float = 0.60  # Cap at 60% of windows traded (selectivity)

    enabled: bool = False


# =============================================================================
# Synthetic Polymarket probability from BTC price action
# =============================================================================
def btc_to_probability(
        window_candles: List[Candle],
        decision_minute: int = 2,
        realistic: Optional[RealisticConfig] = None,
        model_fair_value: Optional[float] = None,
) -> float:
    """
    Convert BTC price movement into a synthetic Polymarket "Up" probability.

    Default mode: Simple linear mapping (sensitivity=15.0).
    Realistic mode: Vol-aware + noise + partial model efficiency.

    The realistic mode produces tighter edges because:
    - Market makers on Polymarket use similar quant models
    - The crowd is noisy but not dumb
    - Prices cluster around fair value with random dispersion
    """
    if len(window_candles) < decision_minute + 1:
        return 0.50

    window_open = window_candles[0].open
    current = window_candles[decision_minute].close

    if window_open == 0:
        return 0.50

    pct_change = (current - window_open) / window_open

    if realistic and realistic.enabled:
        # ── Vol-aware sensitivity ─────────────────────────────────────
        # High vol → crowd is uncertain → prices stay closer to 0.50
        # Low vol → crowd is confident → prices move further from 0.50
        # Calibrated: ~12x at low vol, ~8x at high vol
        intrawindow_vol = 0.0
        if len(window_candles) > 1:
            returns = []
            for i in range(1, min(decision_minute + 1, len(window_candles))):
                prev = window_candles[i - 1].close
                curr = window_candles[i].close
                if prev > 0:
                    returns.append((curr - prev) / prev)
            if returns:
                intrawindow_vol = (sum(r ** 2 for r in returns) / len(returns)) ** 0.5

        # Scale sensitivity inversely with local vol
        # Low vol (< 0.001) → sensitivity ~14 (market is sure)
        # High vol (> 0.005) → sensitivity ~8 (market is uncertain)
        vol_factor = max(0.5, min(1.2, 1.2 - intrawindow_vol * 80))
        sensitivity = 11.0 * vol_factor

        base_prob = 0.50 + pct_change * sensitivity

        # ── Market efficiency: partial reflection of model view ───────
        if model_fair_value is not None:
            # The Polymarket crowd partially knows what the model knows
            # efficiency=0.35 means the market price is 35% model + 65% base
            base_prob = (
                    realistic.market_efficiency * model_fair_value
                    + (1.0 - realistic.market_efficiency) * base_prob
            )

        # ── Random noise from retail flow / MM repositioning ──────────
        noise = random.gauss(0, realistic.market_noise_std)
        prob = base_prob + noise

    else:
        # Original linear mapping
        sensitivity = 15.0
        prob = 0.50 + pct_change * sensitivity

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


def run_confirmation(
        processors: Optional[Dict],
        current_price: Decimal,
        price_history: List[Decimal],
        metadata: Dict[str, Any],
        model_bullish: bool,
) -> Tuple[int, int]:
    """
    Run old signal processors as CONFIRMATION.
    Returns (confirming_count, contradicting_count).
    Mirrors V3 bot Steps 7-8.
    """
    if processors is None:
        return (0, 0)

    signals = []
    pm = {}
    for key, value in metadata.items():
        pm[key] = Decimal(str(value)) if isinstance(value, float) else value

    for name in ["spike", "sentiment", "divergence", "tick_velocity"]:
        proc = processors.get(name)
        if proc is None:
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

    confirming = 0
    contradicting = 0
    for sig in signals:
        sig_bullish = "BULLISH" in str(sig.direction).upper()
        if sig_bullish == model_bullish:
            confirming += 1
        else:
            contradicting += 1

    return confirming, contradicting


def build_confirmation_metadata(
        window_candles: List[Candle],
        decision_minute: int,
        historical_btc_closes: List[float],
) -> Dict[str, Any]:
    """Build metadata dict for confirmation signal processors."""
    if len(window_candles) <= decision_minute:
        return {}

    current_close = window_candles[decision_minute].close
    window_open = window_candles[0].open

    recent = historical_btc_closes[-20:]
    sma_20 = sum(recent) / len(recent) if recent else current_close
    deviation = (current_close - sma_20) / sma_20 if sma_20 != 0 else 0.0

    if len(historical_btc_closes) >= 5:
        p5 = historical_btc_closes[-5]
        momentum = (current_close - p5) / p5 if p5 != 0 else 0.0
    else:
        momentum = 0.0

    variance = sum((p - sma_20) ** 2 for p in recent) / len(recent) if recent else 0.0
    volatility = math.sqrt(variance)

    rsi = compute_rsi(historical_btc_closes, period=14)

    # Synthetic tick buffer
    tick_buffer = []
    start_idx = max(0, decision_minute - 2)
    for i in range(start_idx, min(decision_minute + 1, len(window_candles))):
        c = window_candles[i]
        for j, px in enumerate([c.open, (c.open + c.close) / 2, c.close]):
            tick_ts = c.timestamp + timedelta(seconds=j * 20)
            tick_buffer.append({"ts": tick_ts, "price": Decimal(str(round(px, 4)))})

    return {
        "deviation": deviation,
        "momentum": momentum,
        "volatility": volatility,
        "sentiment_score": rsi,
        "sentiment_classification": _classify_sentiment(rsi),
        "spot_price": current_close,
        "tick_buffer": tick_buffer,
        "yes_token_id": None,
    }


def _classify_sentiment(score: float) -> str:
    if score < 25:
        return "Extreme Fear"
    elif score < 45:
        return "Fear"
    elif score < 55:
        return "Neutral"
    elif score < 75:
        return "Greed"
    else:
        return "Extreme Greed"


# =============================================================================
# Main backtest loop — V3 QUANT STRATEGY
# =============================================================================
def run_backtest_v3(
        candles: List[Candle],
        decision_minute: int = 2,
        verbose: bool = False,
        use_confirmation: bool = True,
        use_maker: bool = True,
        use_funding: bool = True,
        funding_snapshots: Optional[List[FundingSnapshot]] = None,
        realistic: Optional[RealisticConfig] = None,
) -> BacktestResult:
    """
    Backtest the V3.1 quant strategy:
      1. Feed candle data to VolEstimator → vol + jumps + return stats
      2. BinaryOptionPricer (Merton JD + overlays) → model fair value
      3. MispricingDetector → edge, Kelly sizing, tradeability
         V3.1: + vol_skew correction + funding rate bias
      4. Fusion processors → confirmation (optional)
      5. P&L with correct nonlinear fees

    Realistic mode adds:
      - Smarter synthetic market (vol-aware, noisy, partially efficient)
      - Bid-ask spread on entry
      - Fill rate simulation
      - Higher edge floor (kills noise trades)
      - Vol confidence gate
      - Consecutive loss throttle
    """

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

    # ── Chunk candles into 15-minute windows ─────────────────────────────
    windows: List[List[Candle]] = []
    current_window: List[Candle] = []
    window_start: Optional[datetime] = None

    for c in candles:
        minute_slot = c.timestamp.minute // WINDOW_MINUTES
        slot_start = c.timestamp.replace(
            minute=minute_slot * WINDOW_MINUTES, second=0, microsecond=0
        )
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

    # ── State tracking ───────────────────────────────────────────────────
    price_history: List[Decimal] = []
    historical_btc_closes: List[float] = []
    trades: List[BacktestTrade] = []
    equity_curve: List[float] = [0.0]
    bankroll = BANKROLL_USD
    total_fees = 0.0
    skipped = 0
    total_windows = len(windows)

    skip_reasons: Dict[str, int] = {
        "warmup": 0, "no_edge": 0, "confirmation_veto": 0,
        "theta_decay": 0, "no_data": 0, "funding_veto": 0,
        "vol_confidence": 0, "streak_throttle": 0, "no_fill": 0,
        "edge_too_small": 0, "trade_rate_cap": 0,
    }

    # Realistic mode state
    rc = realistic if (realistic and realistic.enabled) else None
    consecutive_losses = 0
    trades_this_session = 0
    windows_processed = 0

    # ── Process each window ──────────────────────────────────────────────
    last_model_yes: Optional[float] = None  # For realistic market efficiency

    for wi, window in enumerate(windows):

        # Feed candle closes up to the decision minute to vol estimator to avoid lookahead
        for c in window[: decision_minute + 1]:
            vol_est.add_price(c.close, c.timestamp.timestamp())
            historical_btc_closes.append(c.close)

        # Set simulated time to the decision point in this window
        # (so VolEstimator prunes bars relative to backtest time, not wall clock)
        decision_ts = window[min(decision_minute, len(window) - 1)].timestamp.timestamp()
        vol_est.set_simulated_time(decision_ts)

        # Warmup: need enough history for vol estimation
        if len(historical_btc_closes) < 30:
            skip_reasons["warmup"] += 1
            skipped += 1
            continue

        windows_processed += 1

        # Actual outcome
        btc_open = window[0].open
        btc_close = window[-1].close
        btc_moved_up = btc_close > btc_open

        # Strike price = BTC at window open (same as live bot)
        btc_strike = btc_open

        # BTC spot at decision time
        if len(window) <= decision_minute:
            skip_reasons["no_data"] += 1
            skipped += 1
            continue
        btc_spot = window[decision_minute].close

        # Time remaining at decision point
        time_remaining_min = float(WINDOW_MINUTES - decision_minute)

        # ── Realistic: Streak throttle ────────────────────────────────
        if rc and consecutive_losses >= rc.max_consecutive_losses:
            skip_reasons["streak_throttle"] += 1
            skipped += 1
            consecutive_losses = max(0, consecutive_losses - 1)  # Cool down
            continue

        # ── Realistic: Trade rate cap ─────────────────────────────────
        if rc and windows_processed > 20:
            current_trade_rate = trades_this_session / windows_processed
            if current_trade_rate > rc.max_trade_rate:
                skip_reasons["trade_rate_cap"] += 1
                skipped += 1
                continue

        # Synthetic Polymarket market price (what the crowd would price)
        poly_price = btc_to_probability(
            window, decision_minute,
            realistic=rc,
            model_fair_value=last_model_yes,
        )
        yes_market = poly_price
        no_market = 1.0 - poly_price

        # Track price history for confirmation processors
        price_history.append(Decimal(str(round(poly_price, 4))))
        if len(price_history) > 100:
            price_history.pop(0)

        # ── V3.1: Compute vol skew correction ────────────────────────────
        vol_estimate = vol_est.get_vol(VOL_METHOD)
        vol_skew = BinaryOptionPricer.estimate_btc_vol_skew(
            spot=btc_spot,
            strike=btc_strike,
            vol=vol_estimate.annualized_vol,
            time_remaining_min=time_remaining_min,
        )

        # ── V3.1: Get funding rate bias at this timestamp ────────────────
        window_ts = window[decision_minute].timestamp
        if use_funding and funding_snapshots:
            funding_bias = get_funding_bias_at(funding_snapshots, window_ts)
            # Find the regime for logging
            funding_regime = "NEUTRAL"
            for s in funding_snapshots:
                if s.timestamp <= window_ts:
                    funding_regime = s.classification
                else:
                    break
        else:
            funding_bias = 0.0
            funding_regime = "DISABLED"

        # ── Run V3.1 mispricing detector ─────────────────────────────────
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
            last_model_yes = signal.yes_model  # Track even on skip
            continue

        # Update model tracking for realistic market efficiency
        last_model_yes = signal.yes_model

        # ── Realistic: Vol confidence gate ────────────────────────────
        if rc and signal.vol_confidence < rc.min_vol_confidence:
            skip_reasons["vol_confidence"] += 1
            skipped += 1
            continue

        # ── Realistic: Higher edge floor (kills noise trades) ─────────
        if rc and abs(signal.edge) < rc.min_edge_realistic:
            skip_reasons["edge_too_small"] += 1
            skipped += 1
            continue

        # ── Fusion confirmation (mirrors V3 bot Steps 7-8) ──────────────
        model_bullish = signal.direction == "BUY_YES"
        confirming, contradicting = 0, 0

        if use_confirmation and fusion_processors and len(price_history) >= 5:
            metadata = build_confirmation_metadata(
                window, decision_minute, historical_btc_closes[-100:],
            )
            if metadata:
                confirming, contradicting = run_confirmation(
                    fusion_processors,
                    Decimal(str(round(poly_price, 4))),
                    list(price_history),
                    metadata,
                    model_bullish,
                )

                # V3 veto rule: if more processors disagree AND model confidence < 60%
                if contradicting > confirming and signal.confidence < 0.6:
                    skip_reasons["confirmation_veto"] += 1
                    skipped += 1
                    continue

        # ── Determine trade direction + outcome ──────────────────────────
        if signal.direction == "BUY_YES":
            direction = "long"
            entry_price = yes_market
        else:
            direction = "short"
            entry_price = no_market

        # ── Realistic: Bid-ask spread (enter at worse side) ───────────
        if rc:
            half_spread = rc.spread_cents / 2.0
            # Spread widens in last 3 minutes
            if time_remaining_min <= 3.0:
                half_spread += rc.spread_widen_late / 2.0
            # Long = buy at ask (higher), short = buy at ask of NO (higher)
            entry_price = min(0.95, entry_price + half_spread)

        # ── Realistic: Fill rate simulation ───────────────────────────
        if rc:
            edge_boost = min(0.15, abs(signal.edge) * rc.fill_rate_boost_edge / 0.10)
            effective_fill_rate = min(0.95, rc.fill_rate + edge_boost)
            if random.random() > effective_fill_rate:
                skip_reasons["no_fill"] += 1
                skipped += 1
                continue

        # ── Simulate New Risk Engine Sizing (Tier-0 Architecture) ─────
        # 1. Base max risk amount
        base_risk_amount = bankroll * 0.05

        # 2. Signal score proxy (assume 85 for high conf, 50 for low)
        signal_score = 85.0 if signal.confidence > 0.6 else 50.0
        score_multiplier = signal_score / 100.0

        # 3. Strength multiplier (using pure fraction from detector)
        strength_multiplier = signal.confidence * score_multiplier * getattr(signal, 'kelly_fraction', 1.0)

        # 4. Gamma Penalty: Reduce size aggressively in the final 5 minutes
        time_factor = 1.0
        if time_remaining_min < 5.0:
            time_factor = max(0.1, time_remaining_min / 5.0)

        theoretical_size = base_risk_amount * strength_multiplier * time_factor

        if rc:
            # Realistic: Scale max position with edge quality
            abs_edge = abs(signal.edge)
            if abs_edge < 0.10:
                edge_scale = 0.6  # 60% position on small edges
            elif abs_edge < 0.20:
                edge_scale = 1.0  # Full position
            else:
                edge_scale = 1.5  # 150% position on large edges
            max_position = MARKET_BUY_USD * edge_scale
            position_size = min(theoretical_size, max_position)
        else:
            position_size = min(theoretical_size, MARKET_BUY_USD)

        # Floor at $1.00 minimum exchange size
        position_size = max(position_size, 1.0)

        # ── Simulate Intra-Window Gamma Scalping (Early Exits) ────────
        exit_price = None
        resolved_early = False

        # Scan subsequent minutes in the window for an exit trigger
        for m in range(decision_minute + 1, len(window)):
            # Synthesize the market price at minute 'm'
            m_poly_price = btc_to_probability(
                window, m, realistic=rc, model_fair_value=last_model_yes
            )
            m_market_price = m_poly_price if direction == "long" else (1.0 - m_poly_price)

            # Apply taker spread to the exit (selling at the bid)
            if rc:
                m_half_spread = rc.spread_cents / 2.0
                m_remaining = WINDOW_MINUTES - m
                if m_remaining <= 3.0:
                    m_half_spread += rc.spread_widen_late / 2.0
                m_exit_price = max(0.01, m_market_price - m_half_spread)
            else:
                m_exit_price = m_market_price

            pnl_pct = (m_exit_price - entry_price) / entry_price
            m_remaining = WINDOW_MINUTES - m

            # THE GAMMA EXIT TRIGGER
            if pnl_pct >= TAKE_PROFIT_PCT or (m_remaining < GAMMA_EXIT_TIME_MINS and pnl_pct > GAMMA_EXIT_PROFIT_PCT):
                exit_price = m_exit_price
                resolved_early = True
                break

        # If no early exit triggered, hold to expiry
        if not resolved_early:
            if direction == "long":
                won = btc_moved_up
            else:
                won = not btc_moved_up
            exit_price = 1.0 if won else 0.0

        outcome = "WIN" if exit_price > entry_price else "LOSS"

        # Track consecutive losses for streak throttle
        if outcome == "WIN":
            consecutive_losses = 0
        else:
            consecutive_losses += 1
        trades_this_session += 1

        # ── P&L with correct fees ────────────────────────────────────────
        num_tokens = position_size / entry_price if entry_price > 0 else 0

        # Entry Fee
        if use_maker:
            entry_fee_rate = 0.0  # Maker pays 0%
        else:
            entry_fee_rate = polymarket_taker_fee(entry_price)

        # Exit Fee (0% if held to expiry, otherwise nonlinear Taker fee)
        if not resolved_early:
            exit_fee_rate = 0.0
        else:
            exit_fee_rate = polymarket_taker_fee(exit_price)

        fee = (entry_fee_rate * num_tokens * entry_price) + (exit_fee_rate * num_tokens * exit_price)

        pnl_before_fees = (exit_price - entry_price) * num_tokens
        pnl = pnl_before_fees - fee
        total_fees += fee

        # Update bankroll
        bankroll += pnl
        bankroll = max(1.0, bankroll)  # Floor at $1
        detector.update_bankroll(bankroll)

        # Equity curve
        cum_pnl = equity_curve[-1] + pnl
        equity_curve.append(cum_pnl)

        trade = BacktestTrade(
            window_start=window[0].timestamp,
            direction=direction,
            entry_price=round(entry_price, 4),
            position_size=round(position_size, 2),
            actual_btc_open=btc_open,
            actual_btc_close=btc_close,
            btc_moved_up=btc_moved_up,
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
        )
        trades.append(trade)

        if verbose:
            arrow = "^" if btc_moved_up else "v"
            emoji = "✅" if won else "❌"
            fund_tag = f" fund={funding_regime[:3]}" if funding_bias != 0 else ""
            logger.info(
                f"W{wi + 1:>4}/{total_windows} | "
                f"{window[0].timestamp.strftime('%m-%d %H:%M')} | "
                f"BTC ${btc_open:,.0f}{arrow}${btc_close:,.0f} | "
                f"{'long' if direction == 'long' else 'short':>5} @{entry_price:.2f} "
                f"(model={signal.yes_model:.2f}/{signal.no_model:.2f}) | "
                f"edge={signal.edge:+.3f} kelly={signal.kelly_fraction:.1%} "
                f"${position_size:.2f} | "
                f"{emoji} ${pnl:+.4f} (fee=${fee:.4f}) | "
                f"cum=${cum_pnl:+.2f} bank=${bankroll:.2f} | "
                f"RV={signal.realized_vol:.0%} IV={signal.implied_vol:.0%} "
                f"VRP={signal.vrp:+.0%} [{signal.pricing_method}]{fund_tag}"
            )

    # ── Compute statistics ───────────────────────────────────────────────
    wins = sum(1 for t in trades if t.outcome == "WIN")
    losses = sum(1 for t in trades if t.outcome == "LOSS")
    total_pnl = sum(t.pnl for t in trades)
    win_rate = (wins / len(trades) * 100) if trades else 0.0
    avg_pnl = (total_pnl / len(trades)) if trades else 0.0
    avg_edge = (sum(abs(t.edge) for t in trades) / len(trades)) if trades else 0.0
    avg_kelly = (sum(t.kelly_fraction for t in trades) / len(trades)) if trades else 0.0
    best_trade = max((t.pnl for t in trades), default=0.0)
    worst_trade = min((t.pnl for t in trades), default=0.0)

    # Max drawdown
    peak = 0.0
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Sharpe ratio (annualized, using trade PnLs)
    if len(trades) > 1:
        pnls = [t.pnl for t in trades]
        mean_pnl = sum(pnls) / len(pnls)
        std_pnl = math.sqrt(sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1))
        # ~96 windows/day × 365 days
        trades_per_year = 96 * 365
        sharpe = (mean_pnl / std_pnl * math.sqrt(trades_per_year)) if std_pnl > 0 else 0.0
    else:
        sharpe = 0.0

    bankroll_growth = ((bankroll - BANKROLL_USD) / BANKROLL_USD * 100) if BANKROLL_USD > 0 else 0.0

    result = BacktestResult(
        start_date=candles[0].timestamp.strftime("%Y-%m-%d") if candles else "",
        end_date=candles[-1].timestamp.strftime("%Y-%m-%d") if candles else "",
        strategy_mode="quant_v3.1_realistic" if (rc and rc.enabled) else "quant_v3.1",
        total_windows=total_windows,
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

    # Log skip reasons
    if verbose:
        logger.info(f"Skip reasons: {skip_reasons}")

    return result


# =============================================================================
# Pretty printer
# =============================================================================
def print_results(result: BacktestResult):
    print()
    print("=" * 90)
    print(f"          BACKTEST RESULTS — {result.strategy_mode.upper()}")
    print("=" * 90)
    print(f"  Period:              {result.start_date} → {result.end_date}")
    print(f"  Total 15m windows:   {result.total_windows}")
    print(f"  Trades taken:        {result.trades_taken}")
    print(f"  Trades skipped:      {result.trades_skipped}")
    print(f"  Trade rate:          {result.trades_taken / max(result.total_windows, 1) * 100:.1f}%")
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
          f"Win rate: {(long_wins / len(longs) * 100) if longs else 0:.1f}%")
    print(f"  Short trades: {len(shorts):>4} | Wins: {short_wins:>4} | "
          f"Win rate: {(short_wins / len(shorts) * 100) if shorts else 0:.1f}%")

    # Vol / VRP analysis
    avg_rv = sum(t.realized_vol for t in result.trades) / len(result.trades)
    avg_iv = sum(t.implied_vol for t in result.trades) / len(result.trades)
    avg_vrp = sum(t.vrp for t in result.trades) / len(result.trades)
    print(f"\n  Avg Realized Vol:    {avg_rv:.0%}")
    print(f"  Avg Implied Vol:     {avg_iv:.0%}")
    print(f"  Avg VRP (IV - RV):   {avg_vrp:+.0%}")

    # Pricing method breakdown
    methods: Dict[str, int] = {}
    for t in result.trades:
        methods[t.pricing_method] = methods.get(t.pricing_method, 0) + 1
    print(f"\n  Pricing methods used:")
    for m, count in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"    {m:<20} {count:>4} trades ({count / len(result.trades) * 100:.0f}%)")

    # Funding regime breakdown
    regimes: Dict[str, List[BacktestTrade]] = {}
    for t in result.trades:
        r = t.funding_regime
        regimes.setdefault(r, []).append(t)
    if any(t.funding_regime != "DISABLED" for t in result.trades):
        print(f"\n  Funding regime breakdown:")
        for regime, regime_trades in sorted(regimes.items()):
            rw = sum(1 for t in regime_trades if t.outcome == "WIN")
            rpnl = sum(t.pnl for t in regime_trades)
            wr = (rw / len(regime_trades) * 100) if regime_trades else 0
            print(f"    {regime:<20} {len(regime_trades):>4} trades, "
                  f"{wr:.1f}% win rate, PnL ${rpnl:+.4f}")

    # Edge distribution
    edges = sorted([abs(t.edge) for t in result.trades])
    print(f"\n  Edge distribution:")
    print(f"    Min:    ${edges[0]:.4f}")
    print(f"    25th:   ${edges[len(edges) // 4]:.4f}")
    print(f"    Median: ${edges[len(edges) // 2]:.4f}")
    print(f"    75th:   ${edges[3 * len(edges) // 4]:.4f}")
    print(f"    Max:    ${edges[-1]:.4f}")

    # Win rate by edge bucket
    print(f"\n  Win rate by edge size:")
    buckets = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.00)]
    for lo, hi in buckets:
        bucket_trades = [t for t in result.trades if lo <= abs(t.edge) < hi]
        if bucket_trades:
            bw = sum(1 for t in bucket_trades if t.outcome == "WIN")
            print(f"    ${lo:.2f}-${hi:.2f}: {len(bucket_trades):>4} trades, "
                  f"{bw / len(bucket_trades) * 100:.1f}% win rate, "
                  f"avg PnL ${sum(t.pnl for t in bucket_trades) / len(bucket_trades):+.4f}")

    # Last 15 trades
    print(f"\n  Last 15 trades:")
    print(f"  {'Time':<13} {'Dir':<6} {'Entry':<6} {'Model':<6} "
          f"{'Edge':<7} {'Kelly':<6} {'Size':<6} {'Result':<5} {'P&L':<9} {'Fee':<6} "
          f"{'RV':<5} {'IV':<5} {'Method'}")
    print(f"  {'-' * 105}")
    for t in result.trades[-15:]:
        model_val = t.model_yes_price if t.direction == "long" else t.model_no_price
        print(
            f"  {t.window_start.strftime('%m-%d %H:%M'):<13} "
            f"{t.direction:<6} "
            f"{t.entry_price:<6.2f} "
            f"{model_val:<6.2f} "
            f"{t.edge:+6.3f} "
            f"{t.kelly_fraction:5.1%} "
            f"${t.position_size:<5.2f} "
            f"{'W' if t.outcome == 'WIN' else 'L':<5} "
            f"${t.pnl:+8.4f} "
            f"${t.fee_paid:<5.4f} "
            f"{t.realized_vol:4.0%} "
            f"{t.implied_vol:4.0%} "
            f"{t.pricing_method}"
        )

    print()
    print("  NOTES:")
    print("  • Market prices are synthetic (BTC move → probability via sensitivity curve)")
    print("  • Fees use correct Polymarket nonlinear formula (max 1.56% taker, 0% maker)")
    print("  • Position sizing via half-Kelly, capped at 5% of bankroll")
    print("  • V3.1: Vol skew correction + historical Binance funding rate bias")
    print("  • OrderBook/DeribitPCR skipped (require live data)")
    print("  • Sentiment is RSI-based proxy for Fear & Greed Index")
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
                "direction": t.direction,
                "entry_price": t.entry_price,
                "position_size": t.position_size,
                "actual_btc_open": t.actual_btc_open,
                "actual_btc_close": t.actual_btc_close,
                "btc_moved_up": t.btc_moved_up,
                "outcome": t.outcome,
                "pnl": t.pnl,
                "pnl_before_fees": t.pnl_before_fees,
                "fee_paid": t.fee_paid,
                "model_yes_price": t.model_yes_price,
                "model_no_price": t.model_no_price,
                "edge": t.edge,
                "edge_pct": t.edge_pct,
                "kelly_fraction": t.kelly_fraction,
                "realized_vol": t.realized_vol,
                "implied_vol": t.implied_vol,
                "vrp": t.vrp,
                "pricing_method": t.pricing_method,
                "confirming_signals": t.confirming_signals,
                "contradicting_signals": t.contradicting_signals,
                "vol_skew": t.vol_skew,
                "funding_bias": t.funding_bias,
                "funding_regime": t.funding_regime,
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
        description="Backtest V3.1 — BTC 15-Min Polymarket Quant Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_v3.py                           # Last 7 days, V3.1 quant
  python backtest_v3.py --realistic               # Realistic mode (spread + noise + filters)
  python backtest_v3.py --realistic --days 100    # 100-day realistic backtest
  python backtest_v3.py --days 30                 # Last 30 days
  python backtest_v3.py --start 2026-02-01 --end 2026-02-27
  python backtest_v3.py --csv data.csv            # From local CSV
  python backtest_v3.py --verbose                 # Print each trade
  python backtest_v3.py --no-confirmation         # Skip fusion confirmation
  python backtest_v3.py --no-funding              # Disable funding rate overlay
  python backtest_v3.py --taker                   # Simulate taker fills (1.56% max fee)
  python backtest_v3.py --realistic --spread 4    # Override spread to 4 cents
  python backtest_v3.py --realistic --fill-rate 0.65  # Override fill rate
  python backtest_v3.py --output results.json
        """,
    )
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="Days to backtest (default: 7)")
    parser.add_argument("--csv", type=str, help="Path to CSV file with 1m candles")
    parser.add_argument("--decision-minute", type=int, default=2,
                        help="Minute within 15-min window to decide (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each trade")
    parser.add_argument("--output", "-o", type=str, help="Export results to JSON")
    parser.add_argument("--no-confirmation", action="store_true",
                        help="Skip fusion processor confirmation (pure quant model)")
    parser.add_argument("--taker", action="store_true",
                        help="Simulate taker fills with nonlinear fees (default: maker/0%%)")
    parser.add_argument("--no-funding", action="store_true",
                        help="Disable funding rate filter overlay")
    parser.add_argument("--realistic", action="store_true",
                        help="Realistic mode: spread, fill rate, noise, smarter filters")
    parser.add_argument("--spread", type=float, default=None,
                        help="Override spread in cents (default: 3 in realistic mode)")
    parser.add_argument("--fill-rate", type=float, default=None,
                        help="Override fill rate 0.0-1.0 (default: 0.72 in realistic mode)")
    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    use_maker = not args.taker
    use_funding = not args.no_funding

    # ── Build realistic config ───────────────────────────────────────────
    rc = RealisticConfig(enabled=args.realistic)
    if args.spread is not None:
        rc.spread_cents = args.spread
    if args.fill_rate is not None:
        rc.fill_rate = args.fill_rate

    # Set random seed for reproducibility in realistic mode
    if rc.enabled:
        random.seed(42)

    mode_label = "V3.1 REALISTIC" if rc.enabled else "V3.1"

    print("=" * 90)
    print(f"  POLYMARKET BTC 15-MIN STRATEGY — {mode_label} QUANT BACKTESTER")
    print("=" * 90)
    print(f"  Model:             Merton Jump-Diffusion + Mean Reversion + Seasonality")
    print(f"  Sizing:            Half-Kelly, capped at 5% of ${BANKROLL_USD:.0f} bankroll")
    print(f"  Fees:              {'Maker (0%)' if use_maker else 'Taker (nonlinear, max 1.56%)'}")
    print(f"  Min edge:          ${rc.min_edge_realistic if rc.enabled else MIN_EDGE_CENTS:.2f}")
    print(f"  Vol method:        {VOL_METHOD}")
    print(f"  Confirmation:      {'ON' if not args.no_confirmation else 'OFF (pure quant)'}")
    print(f"  Funding filter:    {'ON' if use_funding else 'OFF'}")
    print(f"  Vol skew:          ON (BTC crash risk premium)")
    print(f"  Decision minute:   {args.decision_minute}")
    if rc.enabled:
        print(f"  ── REALISTIC MODE ──")
        print(
            f"  Spread:            {rc.spread_cents * 100:.0f}¢ ({(rc.spread_cents + rc.spread_widen_late) * 100:.0f}¢ late window)")
        print(f"  Fill rate:         {rc.fill_rate:.0%} base (+{rc.fill_rate_boost_edge:.0%}/10¢ edge)")
        print(f"  Market noise:      ±{rc.market_noise_std * 100:.1f}¢")
        print(f"  Market efficiency: {rc.market_efficiency:.0%} (MM partially reflects model)")
        print(f"  Vol conf. gate:    >{rc.min_vol_confidence:.0%}")
        print(f"  Streak throttle:   skip after {rc.max_consecutive_losses} consecutive losses")
    print()

    # ── Load candle data ─────────────────────────────────────────────────
    if args.csv:
        candles = load_csv_candles(args.csv)
    else:
        if args.start and args.end:
            start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=args.days)

        candles = fetch_binance_klines(start_dt, end_dt)

    if not candles:
        print("ERROR: No candle data available")
        sys.exit(1)

    # ── Fetch historical funding rates (V3.1) ────────────────────────────
    funding_snapshots: Optional[List[FundingSnapshot]] = None
    if use_funding:
        candle_start = candles[0].timestamp
        candle_end = candles[-1].timestamp
        try:
            funding_snapshots = fetch_funding_rates(candle_start, candle_end)
            if funding_snapshots:
                regime_counts: Dict[str, int] = {}
                for s in funding_snapshots:
                    regime_counts[s.classification] = regime_counts.get(s.classification, 0) + 1
                print(f"  Funding rates loaded: {len(funding_snapshots)} snapshots")
                for regime, count in sorted(regime_counts.items()):
                    print(f"    {regime}: {count}")
                print()
        except Exception as e:
            print(f"  WARNING: Could not fetch funding rates: {e}")
            print(f"  Proceeding without funding filter")
            funding_snapshots = None
            use_funding = False

    # ── Run backtest ─────────────────────────────────────────────────────
    result = run_backtest_v3(
        candles=candles,
        decision_minute=args.decision_minute,
        verbose=args.verbose,
        use_confirmation=not args.no_confirmation,
        use_maker=use_maker,
        use_funding=use_funding,
        funding_snapshots=funding_snapshots,
        realistic=rc,
    )

    print_results(result)

    if args.output:
        export_results(result, args.output)


if __name__ == "__main__":
    main()
