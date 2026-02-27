import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import math
from decimal import Decimal
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from types import SimpleNamespace
import threading

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from patch_gamma_markets import apply_gamma_markets_patch, verify_patch

    patch_applied = apply_gamma_markets_patch()
    if patch_applied:
        verify_patch()
    else:
        print("ERROR: Failed to apply gamma_market patch")
        sys.exit(1)
except ImportError as e:
    print(f"ERROR: Could not import patch module: {e}")
    print("Make sure patch_gamma_markets.py is in the same directory")
    sys.exit(1)

# Nautilus imports
from nautilus_trader.config import (
    LiveDataEngineConfig,
    LiveExecEngineConfig,
    LiveRiskEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.adapters.polymarket import POLYMARKET
from nautilus_trader.adapters.polymarket import (
    PolymarketDataClientConfig,
    PolymarketExecClientConfig,
)
from nautilus_trader.adapters.polymarket.providers import PolymarketInstrumentProviderConfig
from nautilus_trader.adapters.polymarket.factories import (
    PolymarketLiveDataClientFactory,
    PolymarketLiveExecClientFactory,
)
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.identifiers import InstrumentId, ClientOrderId
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity, Price
from nautilus_trader.model.data import QuoteTick

from dotenv import load_dotenv
from loguru import logger
import redis

# Signal processors (kept as confirmation signals in V3)
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.deribit_pcr_processor import DeribitPCRProcessor
from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine
from execution.risk_engine import get_risk_engine
from monitoring.performance_tracker import get_performance_tracker
from monitoring.grafana_exporter import get_grafana_exporter
from feedback.learning_engine import get_learning_engine

# V3: Quantitative pricing model
from binary_pricer import get_binary_pricer, BinaryOptionPricer
from vol_estimator import get_vol_estimator
from mispricing_detector import get_mispricing_detector

# V3.1: RTDS Chainlink settlement oracle + Binance sub-second prices
from rtds_connector import RTDSConnector
from funding_rate_filter import FundingRateFilter

load_dotenv()
from patch_market_orders import apply_market_order_patch

patch_applied = apply_market_order_patch()
if patch_applied:
    logger.info("Market order patch applied successfully")
else:
    logger.warning("Market order patch failed - orders may be rejected")

# =============================================================================
# CONSTANTS
# =============================================================================
QUOTE_STABILITY_REQUIRED = 3
QUOTE_MIN_SPREAD = 0.001
MARKET_INTERVAL_SECONDS = 900

# Config from .env
TRADE_WINDOW_START_SEC = int(os.getenv("TRADE_WINDOW_START", "60"))
TRADE_WINDOW_END_SEC = int(os.getenv("TRADE_WINDOW_END", "180"))
POSITION_SIZE_USD = Decimal(os.getenv("MARKET_BUY_USD", "1.00"))
USE_LIMIT_ORDERS = os.getenv("USE_LIMIT_ORDERS", "true").lower() == "true"
LIMIT_ORDER_OFFSET = Decimal(os.getenv("LIMIT_ORDER_OFFSET", "0.01"))
MIN_FUSION_SIGNALS = int(os.getenv("MIN_FUSION_SIGNALS", "2"))
MIN_FUSION_SCORE = float(os.getenv("MIN_FUSION_SCORE", "55.0"))
TREND_UP_THRESHOLD = float(os.getenv("TREND_UP_THRESHOLD", "0.60"))
TREND_DOWN_THRESHOLD = float(os.getenv("TREND_DOWN_THRESHOLD", "0.40"))
MAX_RETRIES_PER_WINDOW = int(os.getenv("MAX_RETRIES_PER_WINDOW", "3"))

# V3 Config
MIN_EDGE_CENTS = float(os.getenv("MIN_EDGE_CENTS", "0.02"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.30"))
CUT_LOSS_PCT = float(os.getenv("CUT_LOSS_PCT", "-0.50"))
VOL_METHOD = os.getenv("VOL_METHOD", "ewma")
DEFAULT_VOL = float(os.getenv("DEFAULT_VOL", "0.65"))

# V3.1: RTDS + Funding Rate Config
USE_RTDS = os.getenv("USE_RTDS", "true").lower() == "true"
RTDS_LATE_WINDOW_SEC = float(os.getenv("RTDS_LATE_WINDOW_SEC", "15"))
RTDS_LATE_WINDOW_MIN_BPS = float(os.getenv("RTDS_LATE_WINDOW_MIN_BPS", "3.0"))
RTDS_DIVERGENCE_THRESHOLD_BPS = float(os.getenv("RTDS_DIVERGENCE_THRESHOLD_BPS", "5.0"))
USE_FUNDING_FILTER = os.getenv("USE_FUNDING_FILTER", "true").lower() == "true"
LATE_WINDOW_ENABLED = os.getenv("LATE_WINDOW_ENABLED", "true").lower() == "true"


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class PaperTrade:
    timestamp: datetime
    direction: str
    size_usd: float
    price: float
    signal_score: float
    signal_confidence: float
    outcome: str = "PENDING"

    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'direction': self.direction,
            'size_usd': self.size_usd,
            'price': self.price,
            'signal_score': self.signal_score,
            'signal_confidence': self.signal_confidence,
            'outcome': self.outcome,
        }


@dataclass
class OpenPosition:
    """Track a live position until market resolves."""
    market_slug: str
    direction: str  # "long" (bought YES) or "short" (bought NO)
    entry_price: float  # Price we PAID for the token
    size_usd: float
    entry_time: datetime
    market_end_time: datetime
    instrument_id: object
    order_id: str
    resolved: bool = False
    exit_price: Optional[float] = None
    pnl: Optional[float] = None


# =============================================================================
# Redis
# =============================================================================

def init_redis():
    try:
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 2)),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True
        )
        redis_client.ping()
        logger.info("Redis connection established")
        return redis_client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        return None


# =============================================================================
# V3 Strategy
# =============================================================================

class IntegratedBTCStrategy(Strategy):
    """
    Integrated BTC Strategy - V3.1 (BINARY OPTION PRICING MODEL + RTDS)

    V3.1 ADDITIONS:
    1. RTDS Chainlink settlement oracle — sees exact settlement price in real-time
    2. Late-window Chainlink strategy — 85% win rate at T-10s, 0% maker fees
    3. Funding rate regime filter — fades crowded positioning
    4. Skew-adjusted binary correction — accounts for BTC vol smile
    5. Binance-Chainlink divergence — proxy for order flow direction

    V3 CORE (unchanged):
    1. Binary option model (Merton JD) computes fair value for YES/NO tokens
    2. Trades only when model_price - market_price > fee_cost (guaranteed +EV)
    3. Old signal processors demoted to CONFIRMATION role
    4. Take-profit + cut-loss exit monitoring on every tick
    5. Real-time realized vol from sub-second ticks (EWMA estimator)
    6. Timer loop feeds vol estimator and caches spot/sentiment
    7. Limit orders placed at bid (maker, 0% fee)
    8. P&L formula fixed (all positions are BUYS)

    KEPT FROM V2:
    - Early window (minutes 1-3)
    - Limit orders (maker, 0% fee)
    - Persistent data source connections (in timer loop)
    - Both YES and NO token subscriptions
    - 3-tick stability gate
    - Position resolution tracking
    - Retry rate limiting
    """

    def __init__(self, redis_client=None, enable_grafana=True, test_mode=False):
        super().__init__()

        self.bot_start_time = datetime.now(timezone.utc)
        self.restart_after_minutes = 90

        # Nautilus
        self.instrument_id = None
        self.redis_client = redis_client
        self.current_simulation_mode = False

        # Store ALL BTC instruments
        self.all_btc_instruments: List[Dict] = []
        self.current_instrument_index: int = -1
        self.next_switch_time: Optional[datetime] = None

        # Quote-stability tracking
        self._stable_tick_count = 0
        self._market_stable = False
        self._last_instrument_switch = None

        self.last_trade_time = -1
        self._waiting_for_market_open = False
        self._last_bid_ask = None

        # Retry counter per trade window
        self._retry_count_this_window = 0

        # Tick buffer for TickVelocityProcessor
        from collections import deque
        self._tick_buffer: deque = deque(maxlen=500)

        # YES/NO token ids
        self._yes_token_id: Optional[str] = None
        self._yes_instrument_id = None
        self._no_instrument_id = None

        # Open positions
        self._open_positions: List[OpenPosition] = []

        # =====================================================================
        # V3 FIX: Cached market data (populated by timer loop, read by trading decision)
        # This fixes the "Event loop is closed" bug — timer loop owns the async
        # connections, trading decision just reads from cache.
        # =====================================================================
        self._cached_spot_price: Optional[float] = None
        self._cached_sentiment: Optional[float] = None
        self._cached_sentiment_class: Optional[str] = None
        self._cache_lock = threading.Lock()

        # Persistent data source clients (initialized in timer loop)
        self._news_source = None
        self._coinbase_source = None
        self._data_sources_initialized = False

        # =====================================================================
        # V3: Quantitative pricing model
        # =====================================================================
        self.binary_pricer = get_binary_pricer()
        self.vol_estimator = get_vol_estimator()
        self.mispricing_detector = get_mispricing_detector(
            maker_fee=0.00,
            taker_fee=0.10,
            min_edge_cents=MIN_EDGE_CENTS,
            min_edge_after_fees=0.005,
            take_profit_pct=TAKE_PROFIT_PCT,
            cut_loss_pct=CUT_LOSS_PCT,
            vol_method=VOL_METHOD,
        )

        # Strike price (BTC at market open) — reset each market
        self._btc_strike_price: Optional[float] = None
        self._strike_recorded = False

        # Active entry for exit monitoring
        self._active_entry: Optional[dict] = None

        # Late-window trade tracking (one per market window)
        self._late_window_traded = False

        # =====================================================================
        # V3.1: RTDS Chainlink settlement oracle + Funding rate filter
        # =====================================================================
        if USE_RTDS:
            self.rtds = RTDSConnector(
                vol_estimator=self.vol_estimator,
                divergence_threshold_bps=RTDS_DIVERGENCE_THRESHOLD_BPS,
                late_window_max_sec=RTDS_LATE_WINDOW_SEC,
                late_window_min_bps=RTDS_LATE_WINDOW_MIN_BPS,
            )
        else:
            self.rtds = None

        if USE_FUNDING_FILTER:
            self.funding_filter = FundingRateFilter()
        else:
            self.funding_filter = None

        # =====================================================================
        # Signal Processors (V3: confirmation role only)
        # =====================================================================
        self.spike_detector = SpikeDetectionProcessor(
            spike_threshold=float(os.getenv("SPIKE_THRESHOLD", "0.05")),
            lookback_periods=20,
        )
        self.sentiment_processor = SentimentProcessor(
            extreme_fear_threshold=25,
            extreme_greed_threshold=75,
        )
        self.divergence_processor = PriceDivergenceProcessor(
            divergence_threshold=float(os.getenv("DIVERGENCE_THRESHOLD", "0.05")),
        )
        self.orderbook_processor = OrderBookImbalanceProcessor(
            imbalance_threshold=0.30,
            min_book_volume=50.0,
        )
        self.tick_velocity_processor = TickVelocityProcessor(
            velocity_threshold_60s=0.015,
            velocity_threshold_30s=0.010,
        )
        self.deribit_pcr_processor = DeribitPCRProcessor(
            bullish_pcr_threshold=1.20,
            bearish_pcr_threshold=0.70,
            max_days_to_expiry=2,
            cache_seconds=300,
        )

        # Fusion engine (kept for confirmation and heuristic fallback)
        self.fusion_engine = get_fusion_engine()
        self.fusion_engine.set_weight("OrderBookImbalance", 0.30)
        self.fusion_engine.set_weight("TickVelocity", 0.25)
        self.fusion_engine.set_weight("PriceDivergence", 0.18)
        self.fusion_engine.set_weight("SpikeDetection", 0.12)
        self.fusion_engine.set_weight("DeribitPCR", 0.10)
        self.fusion_engine.set_weight("SentimentAnalysis", 0.05)

        # Risk / Performance / Learning
        self.risk_engine = get_risk_engine()
        self.performance_tracker = get_performance_tracker()
        self.learning_engine = get_learning_engine()

        # Grafana
        if enable_grafana:
            self.grafana_exporter = get_grafana_exporter()
        else:
            self.grafana_exporter = None

        # Price history (Polymarket token prices)
        self.price_history = []
        self.max_history = 100

        # Paper trading
        self.paper_trades: List[PaperTrade] = []
        self.test_mode = test_mode

        if test_mode:
            logger.info("=" * 80)
            logger.info("  TEST MODE ACTIVE")
            logger.info("=" * 80)

        logger.info("=" * 80)
        logger.info("INTEGRATED BTC STRATEGY V3.1 — BINARY OPTION PRICING + RTDS ORACLE")
        logger.info(
            f"  Trade window: {TRADE_WINDOW_START_SEC}s–{TRADE_WINDOW_END_SEC}s ({TRADE_WINDOW_START_SEC / 60:.0f}–{TRADE_WINDOW_END_SEC / 60:.0f} min)")
        logger.info(f"  Order type: {'LIMIT @ BID (maker, 0% fee)' if USE_LIMIT_ORDERS else 'MARKET (taker, 10% fee)'}")
        logger.info(f"  Position size: ${POSITION_SIZE_USD}")
        logger.info(f"  Min edge: ${MIN_EDGE_CENTS:.2f}")
        logger.info(f"  Take profit: {TAKE_PROFIT_PCT:.0%} | Cut loss: {CUT_LOSS_PCT:.0%}")
        logger.info(f"  Vol method: {VOL_METHOD} | Default vol: {DEFAULT_VOL:.0%}")
        logger.info(f"  Confirmation: fusion min {MIN_FUSION_SIGNALS} signals, score >= {MIN_FUSION_SCORE}")
        logger.info(
            f"  RTDS: {'Enabled (Chainlink + Binance sub-second)' if USE_RTDS else 'Disabled (Coinbase fallback)'}")
        logger.info(
            f"  Late-window: {'Enabled (T-{:.0f}s, min {:.0f}bps)'.format(RTDS_LATE_WINDOW_SEC, RTDS_LATE_WINDOW_MIN_BPS) if LATE_WINDOW_ENABLED else 'Disabled'}")
        logger.info(f"  Funding filter: {'Enabled' if USE_FUNDING_FILTER else 'Disabled'}")
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seconds_to_next_15min_boundary(self) -> float:
        now_ts = datetime.now(timezone.utc).timestamp()
        next_boundary = (math.floor(now_ts / MARKET_INTERVAL_SECONDS) + 1) * MARKET_INTERVAL_SECONDS
        return next_boundary - now_ts

    def _is_quote_valid(self, bid, ask) -> bool:
        if bid is None or ask is None:
            return False
        try:
            b = float(bid)
            a = float(ask)
        except (TypeError, ValueError):
            return False
        if b < QUOTE_MIN_SPREAD or a < QUOTE_MIN_SPREAD:
            return False
        if b > 0.999 or a > 0.999:
            return False
        return True

    def _reset_stability(self, reason: str = ""):
        if self._market_stable:
            logger.warning(f"Market stability RESET{' – ' + reason if reason else ''}")
        self._market_stable = False
        self._stable_tick_count = 0

    # ------------------------------------------------------------------
    # V3 FIX: Data sources run in timer loop, cache results thread-safely
    # ------------------------------------------------------------------

    async def _init_data_sources(self):
        """Initialize persistent connections to external data sources."""
        if self._data_sources_initialized:
            return

        try:
            from data_sources.news_social.adapter import NewsSocialDataSource
            self._news_source = NewsSocialDataSource()
            await self._news_source.connect()
            logger.info("✓ Persistent news/social data source connected")
        except Exception as e:
            logger.warning(f"Could not connect news source: {e}")
            self._news_source = None

        try:
            from data_sources.coinbase.adapter import CoinbaseDataSource
            self._coinbase_source = CoinbaseDataSource()
            await self._coinbase_source.connect()
            logger.info("✓ Persistent Coinbase data source connected")
        except Exception as e:
            logger.warning(f"Could not connect Coinbase source: {e}")
            self._coinbase_source = None

        self._data_sources_initialized = True

    async def _teardown_data_sources(self):
        if self._news_source:
            try:
                await self._news_source.disconnect()
            except Exception:
                pass
        if self._coinbase_source:
            try:
                await self._coinbase_source.disconnect()
            except Exception:
                pass
        self._data_sources_initialized = False

    async def _refresh_cached_data(self):
        """
        V3.1: Fetch BTC spot + sentiment, cache thread-safely.
        Priority: RTDS Chainlink (settlement oracle) > RTDS Binance > Coinbase (fallback)
        """
        spot = None
        sentiment = None
        sentiment_class = None

        # V3.1: Prefer RTDS Chainlink (THE settlement oracle) over Coinbase
        if self.rtds and self.rtds.chainlink_btc_price > 0 and self.rtds.chainlink_age_ms < 30000:
            spot = self.rtds.chainlink_btc_price
            # Log divergence if significant
            if self.rtds.binance_btc_price > 0:
                div = self.rtds.get_divergence()
                if div.is_significant:
                    logger.info(
                        f"⚡ Price divergence: Binance=${div.binance_price:,.2f} vs "
                        f"Chainlink=${div.chainlink_price:,.2f} ({div.divergence_bps:+.1f}bps) "
                        f"→ {div.direction}"
                    )
        elif self._coinbase_source:
            try:
                coinbase_spot = await self._coinbase_source.get_current_price()
                if coinbase_spot:
                    spot = float(coinbase_spot)
                    # Feed vol estimator from Coinbase fallback
                    self.vol_estimator.add_price(spot)
            except Exception as e:
                logger.debug(f"Coinbase refresh failed: {e}")

        if self._news_source:
            try:
                fg = await self._news_source.get_fear_greed_index()
                if fg and "value" in fg:
                    sentiment = float(fg["value"])
                    sentiment_class = fg.get("classification", "")
            except Exception as e:
                logger.debug(f"Sentiment refresh failed: {e}")

        # V3.1: Update funding rate periodically
        if self.funding_filter and self.funding_filter.should_update():
            try:
                self.funding_filter.update_sync()
            except Exception as e:
                logger.debug(f"Funding rate update failed: {e}")

        with self._cache_lock:
            if spot is not None:
                self._cached_spot_price = spot
            if sentiment is not None:
                self._cached_sentiment = sentiment
                self._cached_sentiment_class = sentiment_class

    def _get_cached_data(self) -> dict:
        """Read cached market data (thread-safe). Called from trading decision."""
        with self._cache_lock:
            return {
                "spot_price": self._cached_spot_price,
                "sentiment_score": self._cached_sentiment,
                "sentiment_classification": self._cached_sentiment_class,
            }

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------

    async def check_simulation_mode(self) -> bool:
        if not self.redis_client:
            return self.current_simulation_mode
        try:
            sim_mode = self.redis_client.get('btc_trading:simulation_mode')
            if sim_mode is not None:
                redis_simulation = sim_mode == '1'
                if redis_simulation != self.current_simulation_mode:
                    self.current_simulation_mode = redis_simulation
                    mode_text = "SIMULATION" if redis_simulation else "LIVE TRADING"
                    logger.warning(f"Trading mode changed to: {mode_text}")
                return redis_simulation
        except Exception as e:
            logger.warning(f"Failed to check Redis simulation mode: {e}")
        return self.current_simulation_mode

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def on_start(self):
        logger.info("=" * 80)
        logger.info("INTEGRATED BTC STRATEGY V3 STARTED")
        logger.info("=" * 80)

        self._load_all_btc_instruments()

        if self.instrument_id:
            self.subscribe_quote_ticks(self.instrument_id)
            logger.info(f"✓ SUBSCRIBED to YES token: {self.instrument_id}")

            no_id = getattr(self, '_no_instrument_id', None)
            if no_id:
                self.subscribe_quote_ticks(no_id)
                logger.info(f"✓ SUBSCRIBED to NO token: {no_id}")

            try:
                quote = self.cache.quote_tick(self.instrument_id)
                if quote and quote.bid_price and quote.ask_price:
                    current_price = (quote.bid_price + quote.ask_price) / 2
                    self.price_history.append(current_price)
                    logger.info(f"✓ Initial price: ${float(current_price):.4f}")
            except Exception as e:
                logger.debug(f"No initial price yet: {e}")

        # No synthetic history — wait for real ticks
        if len(self.price_history) < 5:
            logger.info(f"Waiting for real price data ({len(self.price_history)}/5 ticks so far)")

        # Start timer loop
        self.run_in_executor(self._start_timer_loop)

        # V3.1: Start RTDS Chainlink settlement oracle
        if self.rtds:
            self.rtds.start_background()
            logger.info("✓ RTDS connector started — streaming Chainlink settlement oracle + Binance")

        # V3.1: Initial funding rate fetch
        if self.funding_filter:
            try:
                regime = self.funding_filter.update_sync()
                if regime:
                    logger.info(
                        f"✓ Funding rate: {regime.funding_rate_pct:+.4f}% → "
                        f"{regime.classification}, bias={regime.mean_reversion_bias:+.3f}"
                    )
            except Exception as e:
                logger.debug(f"Initial funding rate fetch failed: {e}")

        if self.grafana_exporter:
            threading.Thread(target=self._start_grafana_sync, daemon=True).start()

        logger.info("=" * 80)
        logger.info("V3.1 active — binary option pricing + RTDS Chainlink oracle")
        logger.info(f"Price history: {len(self.price_history)} points")
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Load all BTC instruments
    # ------------------------------------------------------------------

    def _load_all_btc_instruments(self):
        instruments = self.cache.instruments()
        logger.info(f"Loading ALL BTC instruments from {len(instruments)} total...")

        now = datetime.now(timezone.utc)
        current_timestamp = int(now.timestamp())

        btc_instruments = []

        for instrument in instruments:
            try:
                if hasattr(instrument, 'info') and instrument.info:
                    question = instrument.info.get('question', '').lower()
                    slug = instrument.info.get('market_slug', '').lower()

                    if ('btc' in question or 'btc' in slug) and '15m' in slug:
                        try:
                            timestamp_part = slug.split('-')[-1]
                            market_timestamp = int(timestamp_part)
                            end_timestamp = market_timestamp + 900
                            time_diff = market_timestamp - current_timestamp

                            if end_timestamp > current_timestamp:
                                raw_id = str(instrument.id)
                                without_suffix = raw_id.split('.')[0] if '.' in raw_id else raw_id
                                yes_token_id = without_suffix.split('-')[
                                    -1] if '-' in without_suffix else without_suffix

                                btc_instruments.append({
                                    'instrument': instrument,
                                    'slug': slug,
                                    'start_time': datetime.fromtimestamp(market_timestamp, tz=timezone.utc),
                                    'end_time': datetime.fromtimestamp(end_timestamp, tz=timezone.utc),
                                    'market_timestamp': market_timestamp,
                                    'end_timestamp': end_timestamp,
                                    'time_diff_minutes': time_diff / 60,
                                    'yes_token_id': yes_token_id,
                                })
                        except (ValueError, IndexError):
                            continue
            except Exception:
                continue

        # Pair YES and NO tokens by slug
        seen_slugs = {}
        deduped = []
        for inst in btc_instruments:
            slug = inst['slug']
            if slug not in seen_slugs:
                inst['yes_instrument_id'] = inst['instrument'].id
                inst['no_instrument_id'] = None
                seen_slugs[slug] = inst
                deduped.append(inst)
            else:
                seen_slugs[slug]['no_instrument_id'] = inst['instrument'].id
        btc_instruments = deduped
        btc_instruments.sort(key=lambda x: x['market_timestamp'])

        logger.info("=" * 80)
        logger.info(f"FOUND {len(btc_instruments)} BTC 15-MIN MARKETS:")
        for i, inst in enumerate(btc_instruments):
            is_active = inst['time_diff_minutes'] <= 0 and inst['end_timestamp'] > current_timestamp
            status = "ACTIVE" if is_active else "FUTURE" if inst['time_diff_minutes'] > 0 else "PAST"
            logger.info(
                f"  [{i}] {inst['slug']}: {status} (starts at {inst['start_time'].strftime('%H:%M:%S')}, ends at {inst['end_time'].strftime('%H:%M:%S')})")
        logger.info("=" * 80)

        self.all_btc_instruments = btc_instruments

        # Find current market
        for i, inst in enumerate(btc_instruments):
            is_active = inst['time_diff_minutes'] <= 0 and inst['end_timestamp'] > current_timestamp
            if is_active:
                self.current_instrument_index = i
                self.instrument_id = inst['instrument'].id
                self.next_switch_time = inst['end_time']
                self._yes_token_id = inst.get('yes_token_id')
                self._yes_instrument_id = inst.get('yes_instrument_id', inst['instrument'].id)
                self._no_instrument_id = inst.get('no_instrument_id')
                logger.info(f"✓ CURRENT MARKET: {inst['slug']} (index {i})")
                logger.info(f"  Next switch at: {self.next_switch_time.strftime('%H:%M:%S')}")

                self.subscribe_quote_ticks(self.instrument_id)
                logger.info(f"  ✓ SUBSCRIBED to current market (YES)")
                if inst.get('no_instrument_id'):
                    self.subscribe_quote_ticks(inst['no_instrument_id'])
                    logger.info(f"  ✓ SUBSCRIBED to current market (NO)")
                break

        if self.current_instrument_index == -1 and btc_instruments:
            future_markets = [inst for inst in btc_instruments if inst['time_diff_minutes'] > 0]
            if future_markets:
                nearest = min(future_markets, key=lambda x: x['time_diff_minutes'])
                nearest_idx = btc_instruments.index(nearest)
            else:
                nearest = btc_instruments[-1]
                nearest_idx = len(btc_instruments) - 1

            self.current_instrument_index = nearest_idx
            inst = nearest
            self.instrument_id = inst['instrument'].id
            self._yes_token_id = inst.get('yes_token_id')
            self._yes_instrument_id = inst.get('yes_instrument_id', inst['instrument'].id)
            self._no_instrument_id = inst.get('no_instrument_id')
            self.next_switch_time = inst['start_time']
            logger.info(f"⚠ NO CURRENT MARKET — WAITING FOR: {inst['slug']}")

            self.subscribe_quote_ticks(self.instrument_id)
            if inst.get('no_instrument_id'):
                self.subscribe_quote_ticks(inst['no_instrument_id'])
            logger.info(f"  ✓ SUBSCRIBED to future market (YES + NO)")
            self._waiting_for_market_open = True

    # ------------------------------------------------------------------
    # Market switching
    # ------------------------------------------------------------------

    def _switch_to_next_market(self):
        if not self.all_btc_instruments:
            logger.error("No instruments loaded!")
            return False

        next_index = self.current_instrument_index + 1
        if next_index >= len(self.all_btc_instruments):
            logger.warning("No more markets available — will restart bot")
            return False

        next_market = self.all_btc_instruments[next_index]
        now = datetime.now(timezone.utc)

        if now < next_market['start_time']:
            logger.info(f"Waiting for next market at {next_market['start_time'].strftime('%H:%M:%S')}")
            return False

        # Resolve positions from previous market
        self._resolve_positions_for_market(self.current_instrument_index)

        self.current_instrument_index = next_index
        self.instrument_id = next_market['instrument'].id
        self.next_switch_time = next_market['end_time']
        self._yes_token_id = next_market.get('yes_token_id')
        self._yes_instrument_id = next_market.get('yes_instrument_id', next_market['instrument'].id)
        self._no_instrument_id = next_market.get('no_instrument_id')

        logger.info("=" * 80)
        logger.info(f"SWITCHING TO NEXT MARKET: {next_market['slug']}")
        logger.info(f"  Current time: {now.strftime('%H:%M:%S')}")
        logger.info(f"  Market ends at: {self.next_switch_time.strftime('%H:%M:%S')}")
        logger.info("=" * 80)

        self._stable_tick_count = 0
        self._market_stable = False
        self._waiting_for_market_open = False
        self.last_trade_time = -1
        self._retry_count_this_window = 0

        # V3: Reset strike for new market
        self._btc_strike_price = None
        self._strike_recorded = False
        self._active_entry = None
        self._late_window_traded = False

        logger.info(f"  Trade timer reset — will trade after {QUOTE_STABILITY_REQUIRED} stable ticks")

        self.subscribe_quote_ticks(self.instrument_id)
        if next_market.get('no_instrument_id'):
            self.subscribe_quote_ticks(next_market['no_instrument_id'])
        logger.info(f"  ✓ SUBSCRIBED to new market (YES + NO)")

        return True

    # ------------------------------------------------------------------
    # V3 FIX: Position resolution (ALL positions are BUYS)
    # ------------------------------------------------------------------

    def _resolve_positions_for_market(self, market_index: int):
        """
        When a market ends, check the final price to determine P&L.

        V3 FIX: Both "long" (bought YES) and "short" (bought NO) are LONG
        positions on their respective tokens. The P&L formula is always:
            pnl = size * (exit_price - entry_price) / entry_price

        V3.1 FIX: Also resolves paper trades (simulation mode) that are
        tracked in _open_positions alongside real trades.
        """
        if market_index < 0 or market_index >= len(self.all_btc_instruments):
            return

        market = self.all_btc_instruments[market_index]
        slug = market['slug']

        for pos in self._open_positions:
            if pos.market_slug == slug and not pos.resolved:
                try:
                    quote = self.cache.quote_tick(pos.instrument_id)
                    if quote:
                        final_price = float((quote.bid_price.as_decimal() + quote.ask_price.as_decimal()) / 2)
                    else:
                        final_price = None
                except Exception:
                    final_price = None

                if final_price is not None:
                    # V3 FIX: Always long formula. We BOUGHT tokens.
                    # "long" = bought YES token, "short" = bought NO token.
                    # Either way, profit = (exit - entry) / entry * size
                    pnl = pos.size_usd * (final_price - pos.entry_price) / pos.entry_price

                    pos.resolved = True
                    pos.exit_price = final_price
                    pos.pnl = pnl

                    outcome = "WIN" if pnl > 0 else "LOSS"
                    is_paper = pos.order_id.startswith("paper_")
                    token_type = "YES" if pos.direction == "long" else "NO"
                    tag = "[PAPER] " if is_paper else ""

                    logger.info(f"📊 {tag}POSITION RESOLVED: {slug} bought {token_type}")
                    logger.info(f"   Entry: ${pos.entry_price:.4f} → Exit: ${final_price:.4f}")
                    logger.info(f"   P&L: ${pnl:+.4f} ({outcome})")

                    self.performance_tracker.record_trade(
                        trade_id=pos.order_id,
                        direction=pos.direction,
                        entry_price=Decimal(str(pos.entry_price)),
                        exit_price=Decimal(str(final_price)),
                        size=Decimal(str(pos.size_usd)),
                        entry_time=pos.entry_time,
                        exit_time=datetime.now(timezone.utc),
                        signal_score=0,
                        signal_confidence=0,
                        metadata={"resolved": True, "market": slug, "paper": is_paper}
                    )

                    # V3.1: Update Grafana metrics on resolution
                    if hasattr(self, 'grafana_exporter') and self.grafana_exporter:
                        self.grafana_exporter.increment_trade_counter(won=(pnl > 0))

                    # V3.1: Update matching paper_trade entry
                    if is_paper:
                        for pt in self.paper_trades:
                            if (pt.outcome == "PENDING"
                                    and pt.direction == pos.direction.upper()
                                    and abs(pt.price - pos.entry_price) < 0.0001):
                                pt.outcome = outcome
                                break
                        self._save_paper_trades()

                else:
                    logger.warning(f"Could not resolve position for {slug} — no final price")

    # ------------------------------------------------------------------
    # Timer loop
    # ------------------------------------------------------------------

    def _start_timer_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._timer_loop())
        finally:
            loop.close()

    async def _timer_loop(self):
        """
        Timer loop: runs every 10s.
        V3: Also refreshes Coinbase/sentiment cache and feeds vol estimator.
        """
        await self._init_data_sources()

        while True:
            # Auto-restart check
            uptime_minutes = (datetime.now(timezone.utc) - self.bot_start_time).total_seconds() / 60
            if uptime_minutes >= self.restart_after_minutes:
                logger.warning("AUTO-RESTART TIME — Loading fresh filters")
                await self._teardown_data_sources()
                import signal as _signal
                os.kill(os.getpid(), _signal.SIGTERM)
                return

            now = datetime.now(timezone.utc)

            if self.next_switch_time and now >= self.next_switch_time:
                if self._waiting_for_market_open:
                    logger.info("=" * 80)
                    logger.info(f"⏰ WAITING MARKET NOW OPEN: {now.strftime('%H:%M:%S')} UTC")
                    logger.info("=" * 80)
                    if (self.current_instrument_index >= 0 and
                            self.current_instrument_index < len(self.all_btc_instruments)):
                        current_market = self.all_btc_instruments[self.current_instrument_index]
                        self.next_switch_time = current_market['end_time']
                    self._waiting_for_market_open = False
                    self._stable_tick_count = 0
                    self._market_stable = False
                    self.last_trade_time = -1
                    self._retry_count_this_window = 0
                    # V3: Reset strike
                    self._btc_strike_price = None
                    self._strike_recorded = False
                    self._active_entry = None
                    self._late_window_traded = False
                    logger.info("  ✓ MARKET OPEN — waiting for stable quotes")
                else:
                    self._switch_to_next_market()

            # =========================================================
            # V3: Refresh cached data (Coinbase spot + sentiment + vol)
            # This runs in the timer loop's event loop where the async
            # connections are alive. Results cached thread-safely.
            # =========================================================
            try:
                await self._refresh_cached_data()

                # Record strike on first successful fetch of each market
                if not self._strike_recorded and self._cached_spot_price:
                    self._btc_strike_price = self._cached_spot_price
                    self._strike_recorded = True
                    logger.info(f"📌 BTC Strike recorded: ${self._btc_strike_price:,.2f}")

            except Exception as e:
                logger.debug(f"Cache refresh error: {e}")

            await asyncio.sleep(10)

    # ------------------------------------------------------------------
    # Quote tick handler
    # ------------------------------------------------------------------

    def on_quote_tick(self, tick: QuoteTick):
        try:
            if self.instrument_id is None or tick.instrument_id != self.instrument_id:
                return

            now = datetime.now(timezone.utc)
            bid = tick.bid_price
            ask = tick.ask_price

            if bid is None or ask is None:
                return

            try:
                bid_decimal = bid.as_decimal()
                ask_decimal = ask.as_decimal()
            except:
                return

            mid_price = (bid_decimal + ask_decimal) / 2
            self.price_history.append(mid_price)
            if len(self.price_history) > self.max_history:
                self.price_history.pop(0)

            self._last_bid_ask = (bid_decimal, ask_decimal)
            self._tick_buffer.append({'ts': now, 'price': mid_price})

            # Stability gate
            if not self._market_stable:
                if self._is_quote_valid(bid_decimal, ask_decimal):
                    self._stable_tick_count += 1
                    if self._stable_tick_count >= QUOTE_STABILITY_REQUIRED:
                        self._market_stable = True
                        logger.info(f"✓ Market STABLE after {QUOTE_STABILITY_REQUIRED} valid ticks")
                    else:
                        return
                else:
                    self._stable_tick_count = 0
                    return

            if self._waiting_for_market_open:
                return

            if (self.current_instrument_index < 0 or
                    self.current_instrument_index >= len(self.all_btc_instruments)):
                return

            current_market = self.all_btc_instruments[self.current_instrument_index]
            market_start_ts = current_market['market_timestamp']

            elapsed_secs = now.timestamp() - market_start_ts
            if elapsed_secs < 0:
                return

            sub_interval = int(elapsed_secs // MARKET_INTERVAL_SECONDS)
            trade_key = (market_start_ts, sub_interval)

            seconds_into_sub_interval = elapsed_secs % MARKET_INTERVAL_SECONDS

            # Early window trade trigger
            if (TRADE_WINDOW_START_SEC <= seconds_into_sub_interval < TRADE_WINDOW_END_SEC
                    and trade_key != self.last_trade_time):
                self.last_trade_time = trade_key
                self._retry_count_this_window = 0

                logger.info("=" * 80)
                logger.info(f"🎯 EARLY-WINDOW TRADE: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                logger.info(f"   Market: {current_market['slug']}")
                logger.info(
                    f"   Sub-interval #{sub_interval} ({seconds_into_sub_interval:.1f}s in = {seconds_into_sub_interval / 60:.1f} min)")
                logger.info(
                    f"   YES Price: ${float(mid_price):,.4f} | Bid: ${float(bid_decimal):,.4f} | Ask: ${float(ask_decimal):,.4f}")
                logger.info(f"   Price history: {len(self.price_history)} points")
                logger.info("=" * 80)

                self.run_in_executor(lambda: self._make_trading_decision_sync(float(mid_price)))

            # V3.1: Late-window Chainlink strategy
            # At T-15s through T-0s, use Chainlink settlement oracle for high-confidence trades
            time_remaining_sec = MARKET_INTERVAL_SECONDS - seconds_into_sub_interval
            if (LATE_WINDOW_ENABLED and self.rtds and self._btc_strike_price
                    and 0 < time_remaining_sec <= RTDS_LATE_WINDOW_SEC
                    and not self._late_window_traded):

                late_signal = self.rtds.get_late_window_signal(
                    strike=self._btc_strike_price,
                    time_remaining_sec=time_remaining_sec,
                )

                if late_signal.direction != "NO_SIGNAL" and late_signal.confidence >= 0.70:
                    self._late_window_traded = True

                    logger.info("=" * 80)
                    logger.info(f"🎯 LATE-WINDOW CHAINLINK TRADE: T-{time_remaining_sec:.0f}s")
                    logger.info(f"   {late_signal.reason}")
                    logger.info(
                        f"   Chainlink: ${late_signal.chainlink_price:,.2f} vs Strike: ${late_signal.strike:,.2f}")
                    logger.info(f"   Delta: {late_signal.delta_bps:+.1f}bps | Confidence: {late_signal.confidence:.0%}")
                    logger.info(f"   Direction: {late_signal.direction}")
                    logger.info("=" * 80)

                    self.run_in_executor(
                        lambda: self._execute_late_window_trade(late_signal, float(mid_price))
                    )

        except Exception as e:
            logger.error(f"Error processing quote tick: {e}")

    # ------------------------------------------------------------------
    # Trading decision (V3: Binary Option Model)
    # ------------------------------------------------------------------

    def _make_trading_decision_sync(self, current_price):
        """Sync wrapper — runs in executor thread."""
        price_decimal = Decimal(str(current_price))
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._make_trading_decision(price_decimal))
        finally:
            loop.close()

    def _execute_late_window_trade(self, late_signal, current_yes_price):
        """
        V3.1: Execute late-window Chainlink trade.

        At T-10s, Chainlink shows clear direction → place postOnly limit
        at 0.90-0.93 on the winning side for 0% fees + rebates.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._execute_late_window_trade_async(late_signal, current_yes_price)
            )
        finally:
            loop.close()

    async def _execute_late_window_trade_async(self, late_signal, current_yes_price):
        """Async late-window execution."""
        is_simulation = await self.check_simulation_mode()

        if late_signal.direction == "BUY_YES":
            direction = "long"
            # Place limit order at high price — we expect YES to settle at 1.0
            limit_price = Decimal(str(min(0.93, 0.88 + 0.05 * late_signal.confidence)))
        else:
            direction = "short"
            # Buy NO token — we expect NO to settle at 1.0
            limit_price = Decimal(str(min(0.93, 0.88 + 0.05 * late_signal.confidence)))

        logger.info(
            f"  Late-window: {'BUY YES' if direction == 'long' else 'BUY NO'} "
            f"LIMIT @ ${float(limit_price):.2f} (maker, 0% fee + rebates)"
        )

        # Risk check
        is_valid, error = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD,
            direction=direction,
            current_price=Decimal(str(current_yes_price)),
        )
        if not is_valid:
            logger.warning(f"Late-window risk check failed: {error}")
            return

        mock_signal = SimpleNamespace(
            direction="BULLISH" if direction == "long" else "BEARISH",
            score=late_signal.confidence * 100,
            confidence=late_signal.confidence,
        )

        if is_simulation:
            await self._record_paper_trade(mock_signal, POSITION_SIZE_USD,
                                           limit_price, direction)
        else:
            # Use limit order execution
            await self._place_limit_order(mock_signal, POSITION_SIZE_USD,
                                          limit_price, direction)

    async def _make_trading_decision(self, current_price: Decimal):
        """
        V3: Quantitative trading decision using binary option pricing.

        Decision hierarchy:
        1. Read cached BTC spot + sentiment (populated by timer loop)
        2. Binary option model computes fair value from (spot, strike, vol, T)
        3. Compare model fair value to Polymarket market price
        4. If edge > fees → PRIMARY BUY signal
        5. Old signal processors provide CONFIRMATION (boost/reduce confidence)
        6. Execute if model says yes AND confirmations don't strongly disagree
        """
        is_simulation = await self.check_simulation_mode()
        logger.info(f"Mode: {'SIMULATION' if is_simulation else 'LIVE TRADING'}")

        if len(self.price_history) < 5:
            logger.warning(f"Not enough price history ({len(self.price_history)}/5)")
            return

        logger.info(f"Current YES price: ${float(current_price):,.4f}")

        # Step 1: Read cached market data (from timer loop — no async calls needed)
        cached = self._get_cached_data()
        btc_spot = cached.get("spot_price")

        # V3.1: Show data source
        if btc_spot:
            source = "RTDS Chainlink" if (self.rtds and self.rtds.chainlink_btc_price > 0) else "Coinbase"
            logger.info(f"BTC spot ({source}, cached): ${btc_spot:,.2f}")
        if cached.get("sentiment_score") is not None:
            logger.info(
                f"Fear & Greed (cached): {cached['sentiment_score']:.0f} ({cached.get('sentiment_classification', '')})")

        # V3.1: Show funding rate regime
        funding_bias = 0.0
        if self.funding_filter:
            regime = self.funding_filter.get_regime()
            funding_bias = regime.mean_reversion_bias
            if regime.classification != "NEUTRAL":
                logger.info(
                    f"Funding regime: {regime.classification} ({regime.funding_rate_pct:+.4f}%), bias={funding_bias:+.3f}")

        # Step 2: Check if we can run the quant model
        if btc_spot is None or self._btc_strike_price is None:
            logger.warning(f"No BTC spot or strike — falling back to heuristic signals")
            await self._make_trading_decision_heuristic(current_price, cached, is_simulation)
            return

        # Step 3: Calculate time remaining
        current_market = self.all_btc_instruments[self.current_instrument_index]
        now_ts = datetime.now(timezone.utc).timestamp()
        end_ts = current_market['end_timestamp']
        time_remaining_min = max(0, (end_ts - now_ts) / 60.0)

        logger.info(f"BTC: spot=${btc_spot:,.2f}, strike=${self._btc_strike_price:,.2f}, T={time_remaining_min:.1f}min")

        # Step 4: Get YES/NO market prices
        yes_price = float(current_price)
        no_price = 1.0 - yes_price

        if self._no_instrument_id:
            try:
                no_quote = self.cache.quote_tick(self._no_instrument_id)
                if no_quote:
                    no_price = float((no_quote.bid_price.as_decimal() + no_quote.ask_price.as_decimal()) / 2)
            except Exception:
                pass

        # Step 5: Run mispricing detector (THE CORE QUANT SIGNAL)
        # V3.1: Compute vol skew + pass funding bias
        vol_skew = BinaryOptionPricer.estimate_btc_vol_skew(
            btc_spot, self._btc_strike_price,
            self.vol_estimator.get_vol(VOL_METHOD).annualized_vol,
            time_remaining_min,
        )

        signal = self.mispricing_detector.detect(
            yes_market_price=yes_price,
            no_market_price=no_price,
            btc_spot=btc_spot,
            btc_strike=self._btc_strike_price,
            time_remaining_min=time_remaining_min,
            position_size_usd=float(POSITION_SIZE_USD),
            use_maker=USE_LIMIT_ORDERS,
            # V3.1: Additional overlays
            vol_skew=vol_skew,
            funding_bias=funding_bias,
        )

        if not signal.is_tradeable:
            logger.info(f"Model says NO TRADE: edge=${signal.edge:+.4f}, net_EV=${signal.net_expected_pnl:+.4f}")
            return

        # Step 6: Build metadata for confirmation signals
        metadata = self._build_metadata(cached, current_price)

        # Step 7: Run old signal processors as CONFIRMATION
        old_signals = self._process_signals(current_price, metadata)

        model_bullish = signal.direction == "BUY_YES"
        confirming = 0
        contradicting = 0
        for sig in old_signals:
            sig_bullish = "BULLISH" in str(sig.direction).upper()
            if sig_bullish == model_bullish:
                confirming += 1
            else:
                contradicting += 1

        logger.info(
            f"Confirmation: {confirming} agree, {contradicting} disagree "
            f"(of {len(old_signals)} signals)"
        )

        # Step 8: Decision — trade if model + confirmations align
        if contradicting > confirming and signal.confidence < 0.6:
            logger.info(
                f"⏭ SKIP: Model says {signal.direction} but "
                f"{contradicting}/{len(old_signals)} processors disagree "
                f"and model confidence is only {signal.confidence:.0%}"
            )
            return

        # Step 9: Map to direction
        if signal.direction == "BUY_YES":
            direction = "long"
            logger.info(f"📈 MODEL SAYS: BUY YES (edge=${signal.edge:+.4f}, net_EV=${signal.net_expected_pnl:+.4f})")
        else:
            direction = "short"
            logger.info(f"📉 MODEL SAYS: BUY NO (edge=${signal.edge:+.4f}, net_EV=${signal.net_expected_pnl:+.4f})")

        logger.info(
            f"  Vol: RV={signal.realized_vol:.0%}, IV={signal.implied_vol:.0%}, spread={signal.vol_spread:+.0%}")
        logger.info(f"  Greeks: Δ={signal.delta:.6f}, Γ={signal.gamma:.6f}, Θ={signal.theta_per_min:.6f}/min")

        # Step 10: Risk engine
        is_valid, error = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD,
            direction=direction,
            current_price=current_price,
        )
        if not is_valid:
            logger.warning(f"Risk engine blocked: {error}")
            return

        # Step 11: Liquidity guard
        last_tick = getattr(self, '_last_bid_ask', None)
        if last_tick:
            last_bid, last_ask = last_tick
            MIN_LIQUIDITY = Decimal("0.02")
            if direction == "long" and last_ask <= MIN_LIQUIDITY:
                logger.warning(f"⚠ No liquidity for BUY YES: ask=${float(last_ask):.4f}")
                self.last_trade_time = -1
                return
            if direction == "short" and last_bid <= MIN_LIQUIDITY:
                logger.warning(f"⚠ No liquidity for BUY NO: bid=${float(last_bid):.4f}")
                self.last_trade_time = -1
                return

        # Step 12: Execute
        logger.info(f"Position size: ${POSITION_SIZE_USD} | Direction: {direction.upper()}")

        mock_signal = SimpleNamespace(
            direction="BULLISH" if direction == "long" else "BEARISH",
            score=signal.confidence * 100,
            confidence=signal.confidence,
        )

        if is_simulation:
            await self._record_paper_trade(mock_signal, POSITION_SIZE_USD, current_price, direction)
        else:
            if USE_LIMIT_ORDERS:
                await self._place_limit_order(mock_signal, POSITION_SIZE_USD, current_price, direction)
            else:
                await self._place_real_order(mock_signal, POSITION_SIZE_USD, current_price, direction)

        # Track entry for exit monitoring
        self._active_entry = {
            "direction": signal.direction,
            "entry_price": float(current_price) if direction == "long" else no_price,
            "entry_time": datetime.now(timezone.utc),
            "btc_strike": self._btc_strike_price,
        }

    def _build_metadata(self, cached: dict, current_price: Decimal) -> dict:
        """Build metadata dict for signal processors from cached data."""
        current_price_float = float(current_price)
        recent_prices = [float(p) for p in self.price_history[-20:]]

        metadata = {
            "tick_buffer": list(self._tick_buffer),
            "yes_token_id": self._yes_token_id,
        }

        if len(recent_prices) >= 5:
            sma_20 = sum(recent_prices) / len(recent_prices)
            metadata["deviation"] = (current_price_float - sma_20) / sma_20 if sma_20 != 0 else 0.0
            metadata["momentum"] = (
                (current_price_float - float(self.price_history[-5])) / float(self.price_history[-5])
                if len(self.price_history) >= 5 else 0.0
            )
            variance = sum((p - sma_20) ** 2 for p in recent_prices) / len(recent_prices)
            metadata["volatility"] = math.sqrt(variance)

        if cached.get("spot_price"):
            metadata["spot_price"] = cached["spot_price"]
        if cached.get("sentiment_score") is not None:
            metadata["sentiment_score"] = cached["sentiment_score"]
            metadata["sentiment_classification"] = cached.get("sentiment_classification", "")

        return metadata

    async def _make_trading_decision_heuristic(self, current_price, cached, is_simulation):
        """Fallback to V2 fusion-based logic when Coinbase data is unavailable."""
        metadata = self._build_metadata(cached, current_price)

        signals = self._process_signals(current_price, metadata)
        if not signals:
            logger.info("No signals generated — no trade")
            return

        logger.info(f"Generated {len(signals)} signal(s):")
        for sig in signals:
            logger.info(f"  [{sig.source}] {sig.direction.value}: score={sig.score:.1f}, conf={sig.confidence:.2%}")

        fused = self.fusion_engine.fuse_signals(
            signals,
            min_signals=MIN_FUSION_SIGNALS,
            min_score=MIN_FUSION_SCORE,
        )
        if not fused:
            logger.info(f"Fusion produced no actionable signal (need {MIN_FUSION_SIGNALS}+ signals)")
            return

        logger.info(f"FUSED SIGNAL: {fused.direction.value} (score={fused.score:.1f}, conf={fused.confidence:.2%})")

        price_float = float(current_price)
        if TREND_DOWN_THRESHOLD <= price_float <= TREND_UP_THRESHOLD:
            if fused.confidence < 0.70:
                logger.info(f"⏭ SKIP: coin flip zone, weak confidence")
                return

        if "BULLISH" in str(fused.direction).upper():
            direction = "long"
        else:
            direction = "short"

        is_valid, error = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD, direction=direction, current_price=current_price,
        )
        if not is_valid:
            logger.warning(f"Risk engine blocked: {error}")
            return

        if is_simulation:
            await self._record_paper_trade(fused, POSITION_SIZE_USD, current_price, direction)
        else:
            if USE_LIMIT_ORDERS:
                await self._place_limit_order(fused, POSITION_SIZE_USD, current_price, direction)
            else:
                await self._place_real_order(fused, POSITION_SIZE_USD, current_price, direction)

    # ------------------------------------------------------------------
    # Signal processing (confirmation role in V3)
    # ------------------------------------------------------------------

    def _process_signals(self, current_price, metadata=None):
        signals = []
        if metadata is None:
            metadata = {}

        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, float):
                processed_metadata[key] = Decimal(str(value))
            else:
                processed_metadata[key] = value

        spike_signal = self.spike_detector.process(
            current_price=current_price,
            historical_prices=self.price_history,
            metadata=processed_metadata,
        )
        if spike_signal:
            signals.append(spike_signal)

        if 'sentiment_score' in processed_metadata:
            sentiment_signal = self.sentiment_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if sentiment_signal:
                signals.append(sentiment_signal)

        if 'spot_price' in processed_metadata:
            divergence_signal = self.divergence_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if divergence_signal:
                signals.append(divergence_signal)

        if processed_metadata.get('yes_token_id'):
            ob_signal = self.orderbook_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if ob_signal:
                signals.append(ob_signal)

        if processed_metadata.get('tick_buffer'):
            tv_signal = self.tick_velocity_processor.process(
                current_price=current_price,
                historical_prices=self.price_history,
                metadata=processed_metadata,
            )
            if tv_signal:
                signals.append(tv_signal)

        pcr_signal = self.deribit_pcr_processor.process(
            current_price=current_price,
            historical_prices=self.price_history,
            metadata=processed_metadata,
        )
        if pcr_signal:
            signals.append(pcr_signal)

        return signals

    # ------------------------------------------------------------------
    # Paper trade
    # ------------------------------------------------------------------

    async def _record_paper_trade(self, signal, position_size, current_price, direction):
        """
        V3.1 FIX: Track paper trades as real positions in _open_positions.
        They get resolved by _resolve_positions_for_market() when the 15-min
        window closes, using ACTUAL final Polymarket prices — not random dice.

        Old behavior: random.uniform(-0.02, 0.08) = meaningless noise.
        New behavior: wait for market resolution = real P&L.
        """
        if (self.current_instrument_index < 0 or
                self.current_instrument_index >= len(self.all_btc_instruments)):
            logger.warning("No active market — cannot record paper trade")
            return

        current_market = self.all_btc_instruments[self.current_instrument_index]
        timestamp_ms = int(time.time() * 1000)
        order_id = f"paper_{timestamp_ms}"

        # Determine which instrument we're "buying"
        if direction == "long":
            trade_instrument_id = getattr(self, '_yes_instrument_id', self.instrument_id)
            token_label = "YES (UP)"
        else:
            trade_instrument_id = getattr(self, '_no_instrument_id', None) or self.instrument_id
            token_label = "NO (DOWN)"

        # Track as a real position — resolved at market end
        self._open_positions.append(OpenPosition(
            market_slug=current_market['slug'],
            direction=direction,
            entry_price=float(current_price),
            size_usd=float(position_size),
            entry_time=datetime.now(timezone.utc),
            market_end_time=current_market['end_time'],
            instrument_id=trade_instrument_id,
            order_id=order_id,
        ))

        # Also track in paper_trades list (outcome = PENDING until resolved)
        paper_trade = PaperTrade(
            timestamp=datetime.now(timezone.utc),
            direction=direction.upper(),
            size_usd=float(position_size),
            price=float(current_price),
            signal_score=signal.score,
            signal_confidence=signal.confidence,
            outcome="PENDING",
        )
        self.paper_trades.append(paper_trade)

        logger.info("=" * 80)
        logger.info("[SIMULATION] PAPER TRADE OPENED — awaiting market resolution")
        logger.info(f"  Direction: {direction.upper()} → {token_label}")
        logger.info(f"  Size: ${float(position_size):.2f}")
        logger.info(f"  Entry Price: ${float(current_price):,.4f}")
        logger.info(f"  Market: {current_market['slug']}")
        logger.info(f"  Resolves at: {current_market['end_time'].strftime('%H:%M:%S')} UTC")
        logger.info(f"  Order ID: {order_id}")
        logger.info("=" * 80)

        self._save_paper_trades()

    def _save_paper_trades(self):
        import json
        try:
            trades_data = [t.to_dict() for t in self.paper_trades]
            with open('paper_trades.json', 'w') as f:
                json.dump(trades_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save paper trades: {e}")

    # ------------------------------------------------------------------
    # V3 FIX: Limit order at BID (guarantees maker, 0% fee)
    # ------------------------------------------------------------------

    async def _place_limit_order(self, signal, position_size, current_price, direction):
        """
        Place a LIMIT order at the best BID to guarantee maker status.

        V3 FIX: V2 placed inside the spread (at mid or ask-offset), which
        crossed the spread and filled as taker (10% fee). Placing at the
        bid means our order can only rest in the book = true maker (0% fee).
        Tradeoff: lower fill rate, but zero fees when filled.
        """
        if not self.instrument_id:
            logger.error("No instrument available")
            return

        try:
            side = OrderSide.BUY

            if direction == "long":
                trade_instrument_id = getattr(self, '_yes_instrument_id', self.instrument_id)
                trade_label = "YES (UP)"
            else:
                no_id = getattr(self, '_no_instrument_id', None)
                if no_id is None:
                    logger.warning("NO token not found — cannot bet DOWN. Skipping.")
                    return
                trade_instrument_id = no_id
                trade_label = "NO (DOWN)"

            instrument = self.cache.instrument(trade_instrument_id)
            if not instrument:
                logger.error(f"Instrument not in cache: {trade_instrument_id}")
                return

            # Get current best bid/ask for the token we're buying
            try:
                token_quote = self.cache.quote_tick(trade_instrument_id)
                if token_quote:
                    token_bid = token_quote.bid_price.as_decimal()
                    token_ask = token_quote.ask_price.as_decimal()
                else:
                    token_bid = current_price
                    token_ask = current_price + Decimal("0.02")
            except Exception:
                token_bid = current_price
                token_ask = current_price + Decimal("0.02")

            spread = token_ask - token_bid

            # V3 FIX: Place at best bid to GUARANTEE maker.
            # At the bid, our order rests in the book = maker (0% fee).
            # V2 placed at mid-spread, which crossed to the ask = taker (10%).
            limit_price = token_bid

            # Clamp
            limit_price = max(Decimal("0.01"), min(Decimal("0.99"), limit_price))

            # Calculate token quantity
            token_qty = float(position_size / limit_price)
            precision = instrument.size_precision
            token_qty = round(token_qty, precision)
            token_qty = max(token_qty, 5.0)

            logger.info("=" * 80)
            logger.info(f"LIMIT ORDER @ BID (MAKER, 0% FEE)")
            logger.info(f"  Buying {trade_label}: {token_qty:.2f} tokens @ ${float(limit_price):.4f}")
            logger.info(f"  Book: bid=${float(token_bid):.4f} / ask=${float(token_ask):.4f}")
            logger.info(f"  Spread: ${float(spread):.4f}")
            logger.info(f"  Strategy: resting at bid — waits for seller to hit us")
            logger.info("=" * 80)

            qty = Quantity(token_qty, precision=precision)
            price = Price(float(limit_price), precision=instrument.price_precision)
            timestamp_ms = int(time.time() * 1000)
            unique_id = f"BTC-15MIN-V3-{timestamp_ms}"

            # V3 FIX: No post_only (adapter ignores it). Bid-pricing guarantees maker.
            order = self.order_factory.limit(
                instrument_id=trade_instrument_id,
                order_side=side,
                quantity=qty,
                price=price,
                client_order_id=ClientOrderId(unique_id),
                time_in_force=TimeInForce.GTC,
            )

            self.submit_order(order)

            logger.info(f"LIMIT ORDER SUBMITTED!")
            logger.info(f"  Order ID: {unique_id}")
            logger.info(f"  Direction: {trade_label}")
            logger.info(f"  Limit Price: ${float(limit_price):.4f} (at bid)")
            logger.info(f"  Quantity: {token_qty:.2f}")
            logger.info(f"  Estimated Cost: ~${float(position_size):.2f}")
            logger.info(f"  Fee: 0% (maker — resting at bid)")
            logger.info("=" * 80)

            # Track position
            current_market = self.all_btc_instruments[self.current_instrument_index]
            self._open_positions.append(OpenPosition(
                market_slug=current_market['slug'],
                direction=direction,
                entry_price=float(limit_price),
                size_usd=float(position_size),
                entry_time=datetime.now(timezone.utc),
                market_end_time=current_market['end_time'],
                instrument_id=trade_instrument_id,
                order_id=unique_id,
            ))

            self._track_order_event("placed")

        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            import traceback
            traceback.print_exc()
            self._track_order_event("rejected")

    # ------------------------------------------------------------------
    # Market order (fallback)
    # ------------------------------------------------------------------

    async def _place_real_order(self, signal, position_size, current_price, direction):
        if not self.instrument_id:
            logger.error("No instrument available")
            return

        try:
            logger.info("=" * 80)
            logger.info("LIVE MODE — MARKET ORDER (10% TAKER FEE!)")
            logger.info("⚠️  Consider setting USE_LIMIT_ORDERS=true for 0% fee")
            logger.info("=" * 80)

            side = OrderSide.BUY

            if direction == "long":
                trade_instrument_id = getattr(self, '_yes_instrument_id', self.instrument_id)
                trade_label = "YES (UP)"
            else:
                no_id = getattr(self, '_no_instrument_id', None)
                if no_id is None:
                    logger.warning("NO token not found — cannot bet DOWN. Skipping.")
                    return
                trade_instrument_id = no_id
                trade_label = "NO (DOWN)"

            instrument = self.cache.instrument(trade_instrument_id)
            if not instrument:
                logger.error(f"Instrument not in cache: {trade_instrument_id}")
                return

            trade_price = float(current_price)
            max_usd_amount = float(position_size)
            precision = instrument.size_precision
            min_qty_val = float(getattr(instrument, 'min_quantity', None) or 5.0)
            token_qty = max(min_qty_val, 5.0)
            token_qty = round(token_qty, precision)

            qty = Quantity(token_qty, precision=precision)
            timestamp_ms = int(time.time() * 1000)
            unique_id = f"BTC-15MIN-MKT-{timestamp_ms}"

            order = self.order_factory.market(
                instrument_id=trade_instrument_id,
                order_side=side,
                quantity=qty,
                client_order_id=ClientOrderId(unique_id),
                quote_quantity=False,
                time_in_force=TimeInForce.IOC,
            )

            self.submit_order(order)

            logger.info(f"MARKET ORDER SUBMITTED!")
            logger.info(f"  Order ID: {unique_id}")
            logger.info(f"  Direction: {trade_label}")
            logger.info(f"  Estimated Cost: ~${max_usd_amount:.2f}")
            logger.info(f"  Fee: ~10% (taker)")
            logger.info("=" * 80)

            current_market = self.all_btc_instruments[self.current_instrument_index]
            self._open_positions.append(OpenPosition(
                market_slug=current_market['slug'],
                direction=direction,
                entry_price=trade_price,
                size_usd=max_usd_amount,
                entry_time=datetime.now(timezone.utc),
                market_end_time=current_market['end_time'],
                instrument_id=trade_instrument_id,
                order_id=unique_id,
            ))

            self._track_order_event("placed")

        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            import traceback
            traceback.print_exc()
            self._track_order_event("rejected")

    # ------------------------------------------------------------------
    # Order events
    # ------------------------------------------------------------------

    def _track_order_event(self, event_type: str) -> None:
        try:
            pt = self.performance_tracker
            if hasattr(pt, 'record_order_event'):
                pt.record_order_event(event_type)
            elif hasattr(pt, 'increment_counter'):
                pt.increment_counter(event_type)
            else:
                logger.debug(f"PerformanceTracker has no order-counter method; ignoring '{event_type}'")
        except Exception as e:
            logger.warning(f"Failed to track order event '{event_type}': {e}")

    def on_order_filled(self, event):
        logger.info("=" * 80)
        logger.info(f"ORDER FILLED!")
        logger.info(f"  Order: {event.client_order_id}")
        logger.info(f"  Fill Price: ${float(event.last_px):.4f}")
        logger.info(f"  Quantity: {float(event.last_qty):.6f}")
        logger.info("=" * 80)
        self._track_order_event("filled")

        # Update position entry price with actual fill
        order_id = str(event.client_order_id)
        for pos in self._open_positions:
            if pos.order_id == order_id:
                pos.entry_price = float(event.last_px)
                logger.info(f"  ✓ Position entry updated to actual fill: ${pos.entry_price:.4f}")
                # Also update active entry for exit monitoring
                if self._active_entry:
                    self._active_entry["entry_price"] = pos.entry_price
                break

    def on_order_denied(self, event):
        logger.error("=" * 80)
        logger.error(f"ORDER DENIED!")
        logger.error(f"  Order: {event.client_order_id}")
        logger.error(f"  Reason: {event.reason}")
        logger.error("=" * 80)
        self._track_order_event("rejected")

    def on_order_rejected(self, event):
        reason = str(getattr(event, 'reason', ''))
        reason_lower = reason.lower()
        if 'no orders found' in reason_lower or 'fak' in reason_lower or 'no match' in reason_lower:
            self._retry_count_this_window += 1
            if self._retry_count_this_window <= MAX_RETRIES_PER_WINDOW:
                logger.warning(
                    f"⚠ FAK rejected — retry {self._retry_count_this_window}/{MAX_RETRIES_PER_WINDOW}\n"
                    f"  Reason: {reason}"
                )
                self.last_trade_time = -1
            else:
                logger.warning(
                    f"⚠ FAK rejected — max retries ({MAX_RETRIES_PER_WINDOW}) reached\n"
                    f"  Reason: {reason}"
                )
        else:
            logger.warning(f"Order rejected: {reason}")

    # ------------------------------------------------------------------
    # Grafana / stop
    # ------------------------------------------------------------------

    def _start_grafana_sync(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.grafana_exporter.start())
            logger.info("Grafana metrics started on port 8000")
        except Exception as e:
            logger.error(f"Failed to start Grafana: {e}")

    def on_stop(self):
        logger.info("Integrated BTC strategy V3.1 stopped")
        logger.info(f"Total paper trades: {len(self.paper_trades)}")

        for pos in self._open_positions:
            if not pos.resolved:
                logger.info(f"Unresolved position: {pos.market_slug} {pos.direction} @ ${pos.entry_price:.4f}")

        # Log mispricing detector stats
        stats = self.mispricing_detector.get_stats()
        logger.info(
            f"Mispricing detector: {stats['tradeable_detections']}/{stats['total_detections']} tradeable ({stats['hit_rate']:.0%})")
        logger.info(f"Vol estimator: {self.vol_estimator.get_stats()}")

        # V3.1: RTDS stats + cleanup
        if self.rtds:
            rtds_stats = self.rtds.get_stats()
            logger.info(
                f"RTDS: {rtds_stats['chainlink_ticks']} Chainlink ticks, "
                f"{rtds_stats['binance_ticks']} Binance ticks, "
                f"avg latency={rtds_stats['avg_chainlink_latency_ms']}ms"
            )
            try:
                _loop = asyncio.new_event_loop()
                _loop.run_until_complete(self.rtds.disconnect())
                _loop.close()
            except Exception:
                pass

        if self.funding_filter:
            logger.info(f"Funding filter: {self.funding_filter.get_stats()}")

        if self.grafana_exporter:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.grafana_exporter.stop())
            except Exception:
                pass


# ===========================================================================
# Runner
# ===========================================================================

def run_integrated_bot(simulation: bool = False, enable_grafana: bool = True, test_mode: bool = False):
    print("=" * 80)
    print("INTEGRATED POLYMARKET BTC 15-MIN TRADING BOT V3.1")
    print("BINARY OPTION PRICING + RTDS CHAINLINK ORACLE")
    print("=" * 80)

    redis_client = init_redis()

    if redis_client:
        try:
            mode_value = '1' if simulation else '0'
            redis_client.set('btc_trading:simulation_mode', mode_value)
            mode_label = 'SIMULATION' if simulation else 'LIVE'
            logger.info(f"Redis simulation_mode forced to: {mode_label} ({mode_value})")
        except Exception as e:
            logger.warning(f"Could not set Redis simulation mode: {e}")

    print(f"\nConfiguration:")
    print(f"  Initial Mode: {'SIMULATION' if simulation else 'LIVE TRADING'}")
    print(f"  Redis Control: {'Enabled' if redis_client else 'Disabled'}")
    print(f"  Grafana: {'Enabled' if enable_grafana else 'Disabled'}")
    print(f"  Trade Size: ${POSITION_SIZE_USD}")
    print(f"  Order Type: {'LIMIT @ BID (maker, 0% fee)' if USE_LIMIT_ORDERS else 'MARKET (taker, 10% fee)'}")
    print(f"  Trade Window: {TRADE_WINDOW_START_SEC}s–{TRADE_WINDOW_END_SEC}s")
    print(f"  Min Edge: ${MIN_EDGE_CENTS:.2f}")
    print(f"  Take Profit: {TAKE_PROFIT_PCT:.0%} | Cut Loss: {CUT_LOSS_PCT:.0%}")
    print(f"  Vol: {VOL_METHOD} (default {DEFAULT_VOL:.0%})")
    print(f"  Confirmation: fusion min {MIN_FUSION_SIGNALS} signals, score >= {MIN_FUSION_SCORE}")
    print(f"  Stability gate: {QUOTE_STABILITY_REQUIRED} valid ticks")
    print(f"  RTDS Oracle: {'Enabled (Chainlink + Binance sub-second)' if USE_RTDS else 'Disabled'}")
    print(f"  Late-window: {'Enabled (T-{:.0f}s)'.format(RTDS_LATE_WINDOW_SEC) if LATE_WINDOW_ENABLED else 'Disabled'}")
    print(f"  Funding filter: {'Enabled' if USE_FUNDING_FILTER else 'Disabled'}")
    print()

    logger.info("=" * 80)
    logger.info("LOADING BTC 15-MIN MARKETS VIA EVENT SLUG BUILDER")
    logger.info("=" * 80)

    instrument_cfg = PolymarketInstrumentProviderConfig(
        event_slug_builder="slug_builders:build_btc_15min_slugs",
    )

    poly_data_cfg = PolymarketDataClientConfig(
        private_key=os.getenv("POLYMARKET_PK"),
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
        signature_type=1,
        instrument_config=instrument_cfg,
    )

    poly_exec_cfg = PolymarketExecClientConfig(
        private_key=os.getenv("POLYMARKET_PK"),
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
        signature_type=1,
        instrument_config=instrument_cfg,
    )

    config = TradingNodeConfig(
        environment="live",
        trader_id="BTC-15MIN-V31-001",
        logging=LoggingConfig(
            log_level="ERROR",  # Silences the "Reconciling NET position" spam
            log_directory="./logs/nautilus",
            log_component_levels={
                "IntegratedBTCStrategy": "INFO",  # Keeps the stuff you actually care about
            },
        ),
        data_engine=LiveDataEngineConfig(qsize=6000),
        exec_engine=LiveExecEngineConfig(qsize=6000),
        risk_engine=LiveRiskEngineConfig(bypass=False),
        data_clients={POLYMARKET: poly_data_cfg},
        exec_clients={POLYMARKET: poly_exec_cfg},
    )

    strategy = IntegratedBTCStrategy(
        redis_client=redis_client,
        enable_grafana=enable_grafana,
        test_mode=test_mode,
    )

    print("\nBuilding Nautilus node...")
    node = TradingNode(config=config)
    node.add_data_client_factory(POLYMARKET, PolymarketLiveDataClientFactory)
    node.add_exec_client_factory(POLYMARKET, PolymarketLiveExecClientFactory)
    node.trader.add_strategy(strategy)
    node.build()
    logger.info("Nautilus node built successfully")

    print()
    print("=" * 80)
    print("BOT V3 STARTING")
    print("=" * 80)

    try:
        node.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.dispose()
        logger.info("Bot stopped")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Integrated BTC 15-Min Trading Bot V3")
    parser.add_argument("--live", action="store_true",
                        help="Run in LIVE mode (real money at risk!)")
    parser.add_argument("--no-grafana", action="store_true", help="Disable Grafana metrics")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in TEST MODE (simulation, fast clock)")

    args = parser.parse_args()
    enable_grafana = not args.no_grafana
    test_mode = args.test_mode

    if args.test_mode:
        simulation = True
    else:
        simulation = not args.live

    if not simulation:
        logger.warning("=" * 80)
        logger.warning("LIVE TRADING MODE — REAL MONEY AT RISK!")
        logger.warning(f"Order type: {'LIMIT @ BID (0% fee)' if USE_LIMIT_ORDERS else 'MARKET (10% fee)'}")
        logger.warning("=" * 80)
    else:
        logger.info("=" * 80)
        logger.info(f"SIMULATION MODE — {'TEST MODE' if test_mode else 'paper trading only'}")
        logger.info("=" * 80)

    run_integrated_bot(simulation=simulation, enable_grafana=enable_grafana, test_mode=test_mode)


if __name__ == "__main__":
    main()