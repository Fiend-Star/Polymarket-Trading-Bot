import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import math
from decimal import Decimal
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import random

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

# Now import Nautilus
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

# Import our phases
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
QUOTE_STABILITY_REQUIRED = 3      # Need 3 valid ticks to be stable
QUOTE_MIN_SPREAD = 0.001          # Both bid AND ask must be at least this
MARKET_INTERVAL_SECONDS = 900     # 15-minute markets

# =============================================================================
# CHANGED: Read config from .env with sensible defaults
# These were previously hardcoded or ignored.
# =============================================================================
TRADE_WINDOW_START_SEC = int(os.getenv("TRADE_WINDOW_START", "60"))   # CHANGED: was 780 (min 13). Now min 1.
TRADE_WINDOW_END_SEC = int(os.getenv("TRADE_WINDOW_END", "180"))      # CHANGED: was 840 (min 14). Now min 3.
POSITION_SIZE_USD = Decimal(os.getenv("MARKET_BUY_USD", "1.00"))      # CHANGED: reads from .env
USE_LIMIT_ORDERS = os.getenv("USE_LIMIT_ORDERS", "true").lower() == "true"  # NEW: maker orders
LIMIT_ORDER_OFFSET = Decimal(os.getenv("LIMIT_ORDER_OFFSET", "0.01"))       # NEW: how far inside spread
MIN_FUSION_SIGNALS = int(os.getenv("MIN_FUSION_SIGNALS", "2"))         # CHANGED: was 1. Now requires 2+ agreeing signals.
MIN_FUSION_SCORE = float(os.getenv("MIN_FUSION_SCORE", "55.0"))       # CHANGED: was 40. Raised threshold.
TREND_UP_THRESHOLD = float(os.getenv("TREND_UP_THRESHOLD", "0.60"))    # Used as a SKIP gate only, not direction override
TREND_DOWN_THRESHOLD = float(os.getenv("TREND_DOWN_THRESHOLD", "0.40"))
MAX_RETRIES_PER_WINDOW = int(os.getenv("MAX_RETRIES_PER_WINDOW", "3"))  # NEW: cap FAK retry storms


@dataclass
class PaperTrade:
    """Track paper/simulation trades"""
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


# =============================================================================
# NEW: Track open positions for resolution
# =============================================================================
@dataclass
class OpenPosition:
    """Track a live position until market resolves"""
    market_slug: str
    direction: str  # "long" or "short"
    entry_price: float
    size_usd: float
    entry_time: datetime
    market_end_time: datetime
    instrument_id: object  # InstrumentId
    order_id: str
    resolved: bool = False
    exit_price: Optional[float] = None
    pnl: Optional[float] = None


def init_redis():
    """Initialize Redis connection for simulation mode control."""
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
        logger.warning("Simulation mode will be static (from .env)")
        return None


class IntegratedBTCStrategy(Strategy):
    """
    Integrated BTC Strategy - V2 (EARLY WINDOW + MAKER ORDERS)

    KEY CHANGES FROM V1:
    1. Trades at minutes 1-3 (signal-driven prediction) instead of 13-14 (trend-following)
    2. Uses limit orders (0% maker fee) instead of market orders (10% taker fee)
    3. Fusion engine DRIVES direction (min 2 signals required), no trend override
    4. Persistent data source connections (no reconnect per decision)
    5. Subscribes to BOTH YES and NO token quotes (risk engine works)
    6. No synthetic price history poisoning
    7. Position resolution tracking
    8. Retry rate limiting on FAK rejections
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

        self.last_trade_time = -1  # Force first trade immediately
        self._waiting_for_market_open = False
        self._last_bid_ask = None

        # CHANGED: Retry counter per trade window to prevent FAK storms
        self._retry_count_this_window = 0

        # Tick buffer: rolling 90s of ticks for TickVelocityProcessor
        from collections import deque
        self._tick_buffer: deque = deque(maxlen=500)

        # YES/NO token ids for the current market
        self._yes_token_id: Optional[str] = None

        # NEW: Track open positions
        self._open_positions: List[OpenPosition] = []

        # =============================================================================
        # CHANGED: Persistent data source clients (initialized once, not per-decision)
        # =============================================================================
        self._news_source = None
        self._coinbase_source = None
        self._data_sources_initialized = False

        # Phase 4: Signal Processors
        self.spike_detector = SpikeDetectionProcessor(
            spike_threshold=float(os.getenv("SPIKE_THRESHOLD", "0.05")),  # CHANGED: reads .env
            lookback_periods=20,
        )
        self.sentiment_processor = SentimentProcessor(
            extreme_fear_threshold=25,
            extreme_greed_threshold=75,
        )
        self.divergence_processor = PriceDivergenceProcessor(
            divergence_threshold=float(os.getenv("DIVERGENCE_THRESHOLD", "0.05")),  # CHANGED: reads .env
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

        # Phase 4: Signal Fusion
        self.fusion_engine = get_fusion_engine()
        self.fusion_engine.set_weight("OrderBookImbalance", 0.30)
        self.fusion_engine.set_weight("TickVelocity",       0.25)
        self.fusion_engine.set_weight("PriceDivergence",    0.18)
        self.fusion_engine.set_weight("SpikeDetection",     0.12)
        self.fusion_engine.set_weight("DeribitPCR",         0.10)
        self.fusion_engine.set_weight("SentimentAnalysis",  0.05)

        # Phase 5: Risk Management
        self.risk_engine = get_risk_engine()

        # Phase 6: Performance Tracking
        self.performance_tracker = get_performance_tracker()

        # Phase 7: Learning Engine
        self.learning_engine = get_learning_engine()

        # Phase 6: Grafana (optional)
        if enable_grafana:
            self.grafana_exporter = get_grafana_exporter()
        else:
            self.grafana_exporter = None

        # Price history
        self.price_history = []
        self.max_history = 100

        # Paper trading tracker
        self.paper_trades: List[PaperTrade] = []

        self.test_mode = test_mode

        if test_mode:
            logger.info("=" * 80)
            logger.info("  TEST MODE ACTIVE - Trading every minute!")
            logger.info("=" * 80)

        logger.info("=" * 80)
        logger.info("INTEGRATED BTC STRATEGY V2 - EARLY WINDOW + MAKER ORDERS")
        logger.info(f"  Trade window: {TRADE_WINDOW_START_SEC}s–{TRADE_WINDOW_END_SEC}s ({TRADE_WINDOW_START_SEC/60:.0f}–{TRADE_WINDOW_END_SEC/60:.0f} min)")
        logger.info(f"  Order type: {'LIMIT (maker, 0% fee)' if USE_LIMIT_ORDERS else 'MARKET (taker, 10% fee)'}")
        logger.info(f"  Position size: ${POSITION_SIZE_USD}")
        logger.info(f"  Fusion: min {MIN_FUSION_SIGNALS} signals, min score {MIN_FUSION_SCORE}")
        logger.info(f"  Skip zone: {TREND_DOWN_THRESHOLD:.0%}–{TREND_UP_THRESHOLD:.0%} (coin flip)")
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seconds_to_next_15min_boundary(self) -> float:
        """Return seconds until the next 15-minute UTC boundary."""
        now_ts = datetime.now(timezone.utc).timestamp()
        next_boundary = (math.floor(now_ts / MARKET_INTERVAL_SECONDS) + 1) * MARKET_INTERVAL_SECONDS
        return next_boundary - now_ts

    def _is_quote_valid(self, bid, ask) -> bool:
        """Return True only when BOTH bid and ask are present and make sense."""
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
        """Mark the market as unstable and reset the counter."""
        if self._market_stable:
            logger.warning(f"Market stability RESET{' – ' + reason if reason else ''}")
        self._market_stable = False
        self._stable_tick_count = 0

    # ------------------------------------------------------------------
    # CHANGED: Persistent data source connections
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
        """Disconnect persistent data sources."""
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

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------

    async def check_simulation_mode(self) -> bool:
        """Check Redis for current simulation mode."""
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
                    if not redis_simulation:
                        logger.warning("LIVE TRADING ACTIVE - Real money at risk!")
                return redis_simulation
        except Exception as e:
            logger.warning(f"Failed to check Redis simulation mode: {e}")
        return self.current_simulation_mode

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def on_start(self):
        """Called when strategy starts - LOAD ALL MARKETS AND SUBSCRIBE IMMEDIATELY"""
        logger.info("=" * 80)
        logger.info("INTEGRATED BTC STRATEGY V2 STARTED")
        logger.info("=" * 80)

        # Load ALL BTC instruments at startup
        self._load_all_btc_instruments()

        # Subscribe to current market IMMEDIATELY
        if self.instrument_id:
            self.subscribe_quote_ticks(self.instrument_id)
            logger.info(f"✓ SUBSCRIBED to YES token: {self.instrument_id}")

            # CHANGED: Also subscribe to NO token so risk engine has price data
            no_id = getattr(self, '_no_instrument_id', None)
            if no_id:
                self.subscribe_quote_ticks(no_id)
                logger.info(f"✓ SUBSCRIBED to NO token: {no_id}")

            # Try to get current price from cache
            try:
                quote = self.cache.quote_tick(self.instrument_id)
                if quote and quote.bid_price and quote.ask_price:
                    current_price = (quote.bid_price + quote.ask_price) / 2
                    self.price_history.append(current_price)
                    logger.info(f"✓ Initial price: ${float(current_price):.4f}")
            except Exception as e:
                logger.debug(f"No initial price yet: {e}")

        # =============================================================================
        # CHANGED: No synthetic price history. Wait for real ticks.
        # Synthetic data was poisoning the spike detector MA calculation.
        # We just need 5 ticks to start (not 20), and spike detector
        # will return no signal until it has enough data.
        # =============================================================================
        if len(self.price_history) < 5:
            logger.info(f"Waiting for real price data ({len(self.price_history)}/5 ticks so far)")

        # Start the timer loop
        self.run_in_executor(self._start_timer_loop)

        if self.grafana_exporter:
            import threading
            threading.Thread(target=self._start_grafana_sync, daemon=True).start()

        logger.info("=" * 80)
        logger.info("Strategy active — trading in early window (minutes 1-3)")
        logger.info(f"Price history: {len(self.price_history)} points")
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Load all BTC instruments at once
    # ------------------------------------------------------------------

    def _load_all_btc_instruments(self):
        """Load ALL BTC instruments from cache and sort by start time"""
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

                            real_start_ts = market_timestamp
                            end_timestamp = market_timestamp + 900
                            time_diff = real_start_ts - current_timestamp

                            if end_timestamp > current_timestamp:
                                raw_id = str(instrument.id)
                                without_suffix = raw_id.split('.')[0] if '.' in raw_id else raw_id
                                yes_token_id = without_suffix.split('-')[-1] if '-' in without_suffix else without_suffix

                                btc_instruments.append({
                                    'instrument': instrument,
                                    'slug': slug,
                                    'start_time': datetime.fromtimestamp(real_start_ts, tz=timezone.utc),
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
            logger.info(f"  [{i}] {inst['slug']}: {status} (starts at {inst['start_time'].strftime('%H:%M:%S')}, ends at {inst['end_time'].strftime('%H:%M:%S')})")
        logger.info("=" * 80)

        self.all_btc_instruments = btc_instruments

        # Find current market and SUBSCRIBE IMMEDIATELY
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
                logger.info(f"  YES token: {self._yes_token_id[:16]}…" if self._yes_token_id else "  YES token: unknown")

                self.subscribe_quote_ticks(self.instrument_id)
                logger.info(f"  ✓ SUBSCRIBED to current market (YES)")

                # CHANGED: Subscribe to NO token too
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
            logger.info(f"⚠ NO CURRENT MARKET - WAITING FOR NEAREST FUTURE: {inst['slug']}")
            logger.info(f"  Starts in {inst['time_diff_minutes']:.1f} min at {self.next_switch_time.strftime('%H:%M:%S')} UTC")

            self.subscribe_quote_ticks(self.instrument_id)
            # CHANGED: Subscribe NO token for future market too
            if inst.get('no_instrument_id'):
                self.subscribe_quote_ticks(inst['no_instrument_id'])
            logger.info(f"  ✓ SUBSCRIBED to future market (YES + NO)")
            self._waiting_for_market_open = True

    def _switch_to_next_market(self):
        """Switch to the next market in the pre-loaded list"""
        if not self.all_btc_instruments:
            logger.error("No instruments loaded!")
            return False

        next_index = self.current_instrument_index + 1
        if next_index >= len(self.all_btc_instruments):
            logger.warning("No more markets available - will restart bot")
            return False

        next_market = self.all_btc_instruments[next_index]
        now = datetime.now(timezone.utc)

        if now < next_market['start_time']:
            logger.info(f"Waiting for next market at {next_market['start_time'].strftime('%H:%M:%S')}")
            return False

        # =============================================================================
        # CHANGED: Check resolution of positions from the PREVIOUS market before switching
        # =============================================================================
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

        # CHANGED: Use actual QUOTE_STABILITY_REQUIRED constant for the gate
        self._stable_tick_count = 0
        self._market_stable = False
        self._waiting_for_market_open = False

        # Reset trade timer and retry counter
        self.last_trade_time = -1
        self._retry_count_this_window = 0  # CHANGED: reset retry counter
        logger.info(f"  Trade timer reset — will trade after {QUOTE_STABILITY_REQUIRED} stable ticks")

        self.subscribe_quote_ticks(self.instrument_id)
        # CHANGED: subscribe to NO token
        if next_market.get('no_instrument_id'):
            self.subscribe_quote_ticks(next_market['no_instrument_id'])
        logger.info(f"  ✓ SUBSCRIBED to new market (YES + NO)")

        return True

    # ------------------------------------------------------------------
    # NEW: Position resolution tracking
    # ------------------------------------------------------------------

    def _resolve_positions_for_market(self, market_index: int):
        """
        When a market ends, check the final price to determine P&L
        of any positions we had in that market.
        """
        if market_index < 0 or market_index >= len(self.all_btc_instruments):
            return

        market = self.all_btc_instruments[market_index]
        slug = market['slug']

        for pos in self._open_positions:
            if pos.market_slug == slug and not pos.resolved:
                # Try to get the last known price for this instrument
                try:
                    quote = self.cache.quote_tick(pos.instrument_id)
                    if quote:
                        final_price = float((quote.bid_price.as_decimal() + quote.ask_price.as_decimal()) / 2)
                    else:
                        final_price = None
                except Exception:
                    final_price = None

                if final_price is not None:
                    if pos.direction == "long":
                        # Bought YES: profit if final > entry
                        pnl = pos.size_usd * (final_price - pos.entry_price) / pos.entry_price
                    else:
                        # Bought NO: profit if final < entry (YES price dropped)
                        pnl = pos.size_usd * (pos.entry_price - final_price) / pos.entry_price

                    pos.resolved = True
                    pos.exit_price = final_price
                    pos.pnl = pnl

                    outcome = "WIN" if pnl > 0 else "LOSS"
                    logger.info(f"📊 POSITION RESOLVED: {slug} {pos.direction.upper()}")
                    logger.info(f"   Entry: ${pos.entry_price:.4f} → Exit: ${final_price:.4f}")
                    logger.info(f"   P&L: ${pnl:+.4f} ({outcome})")

                    # Record in performance tracker
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
                        metadata={"resolved": True, "market": slug}
                    )
                else:
                    logger.warning(f"Could not resolve position for {slug} — no final price")

    # ------------------------------------------------------------------
    # Timer loop
    # ------------------------------------------------------------------

    def _start_timer_loop(self):
        """Start timer loop in executor"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._timer_loop())
        finally:
            loop.close()

    async def _timer_loop(self):
        """Timer loop: checks every 10 seconds if it's time to switch markets."""
        # CHANGED: Initialize persistent data sources in timer loop (has its own event loop)
        await self._init_data_sources()

        while True:
            # --- auto-restart check ---
            uptime_minutes = (datetime.now(timezone.utc) - self.bot_start_time).total_seconds() / 60
            if uptime_minutes >= self.restart_after_minutes:
                logger.warning("AUTO-RESTART TIME - Loading fresh filters")
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
                        logger.info(f"  Market ends at {self.next_switch_time.strftime('%H:%M:%S')} UTC")
                    self._waiting_for_market_open = False
                    # CHANGED: Don't force stable — let real ticks prove stability
                    self._stable_tick_count = 0
                    self._market_stable = False
                    self.last_trade_time = -1
                    self._retry_count_this_window = 0
                    logger.info("  ✓ MARKET OPEN — waiting for stable quotes before trading")
                else:
                    self._switch_to_next_market()

            await asyncio.sleep(10)

    # ------------------------------------------------------------------
    # Quote tick handler
    # ------------------------------------------------------------------

    def on_quote_tick(self, tick: QuoteTick):
        """Handle quote tick - TRADE in early window (minutes 1-3) using signal-driven prediction"""
        try:
            # Only process ticks from current YES instrument
            # (We also subscribe to NO, but only for risk engine pricing — we don't need to trade off NO ticks)
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

            # Always store price history
            mid_price = (bid_decimal + ask_decimal) / 2
            self.price_history.append(mid_price)
            if len(self.price_history) > self.max_history:
                self.price_history.pop(0)

            self._last_bid_ask = (bid_decimal, ask_decimal)

            # Tick buffer for TickVelocityProcessor
            self._tick_buffer.append({'ts': now, 'price': mid_price})

            # =============================================================================
            # CHANGED: Stability gate actually uses QUOTE_STABILITY_REQUIRED
            # =============================================================================
            if not self._market_stable:
                if self._is_quote_valid(bid_decimal, ask_decimal):
                    self._stable_tick_count += 1
                    if self._stable_tick_count >= QUOTE_STABILITY_REQUIRED:
                        self._market_stable = True
                        logger.info(f"✓ Market STABLE after {QUOTE_STABILITY_REQUIRED} valid ticks")
                    else:
                        return  # Still waiting for stability
                else:
                    self._stable_tick_count = 0  # Reset on invalid tick
                    return

            # Block trading if waiting for a future market to open
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

            # =============================================================================
            # CHANGED: Early trade window (configurable via .env, default minutes 1-3)
            #
            # WHY EARLY:
            #   At minutes 1-3, the market price is near 0.50. Signal processors
            #   (orderbook, Deribit PCR, tick velocity, divergence) have actual
            #   PREDICTIVE edge here — they detect developing moves before price
            #   confirms them. This is where alpha lives.
            #
            # WHY NOT LATE (minutes 13-14, the old approach):
            #   At minute 13, price is 0.80 or 0.20 — the move already happened.
            #   You're paying $0.77+$0.10 fee for something worth $0.77.
            #   Zero edge, negative EV from fees.
            #
            # SAFETY: Fusion engine requires 2+ agreeing signals (no single-signal
            #   pass-through). Coin-flip zone (0.40-0.60) skip still applies as a
            #   sanity check, but it rarely triggers at minutes 1-3 since prices
            #   start near 0.50.
            # =============================================================================
            seconds_into_sub_interval = elapsed_secs % MARKET_INTERVAL_SECONDS

            if (TRADE_WINDOW_START_SEC <= seconds_into_sub_interval < TRADE_WINDOW_END_SEC
                    and trade_key != self.last_trade_time):
                self.last_trade_time = trade_key
                self._retry_count_this_window = 0  # Reset retry counter for new window

                logger.info("=" * 80)
                logger.info(f"🎯 EARLY-WINDOW TRADE: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                logger.info(f"   Market: {current_market['slug']}")
                logger.info(f"   Sub-interval #{sub_interval} ({seconds_into_sub_interval:.1f}s in = {seconds_into_sub_interval/60:.1f} min)")
                logger.info(f"   Price: ${float(mid_price):,.4f} | Bid: ${float(bid_decimal):,.4f} | Ask: ${float(ask_decimal):,.4f}")
                logger.info(f"   Price history: {len(self.price_history)} points")
                logger.info("=" * 80)

                self.run_in_executor(lambda: self._make_trading_decision_sync(float(mid_price)))

        except Exception as e:
            logger.error(f"Error processing quote tick: {e}")

    # ------------------------------------------------------------------
    # Trading decision
    # ------------------------------------------------------------------

    def _make_trading_decision_sync(self, current_price):
        """Synchronous wrapper for trading decision (called from executor)."""
        from decimal import Decimal
        price_decimal = Decimal(str(current_price))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._make_trading_decision(price_decimal))
        finally:
            loop.close()

    async def _fetch_market_context(self, current_price: Decimal) -> dict:
        """
        Fetch REAL external data to populate signal processor metadata.

        CHANGED: Uses persistent connections instead of connect/disconnect per call.
        """
        current_price_float = float(current_price)

        recent_prices = [float(p) for p in self.price_history[-20:]]
        if len(recent_prices) < 5:
            # Not enough history yet — return minimal metadata
            return {
                "deviation": 0.0,
                "momentum": 0.0,
                "volatility": 0.0,
                "tick_buffer": list(self._tick_buffer),
                "yes_token_id": self._yes_token_id,
            }

        sma_20 = sum(recent_prices) / len(recent_prices)
        deviation = (current_price_float - sma_20) / sma_20 if sma_20 != 0 else 0.0
        momentum = (
            (current_price_float - float(self.price_history[-5])) / float(self.price_history[-5])
            if len(self.price_history) >= 5 else 0.0
        )
        variance = sum((p - sma_20) ** 2 for p in recent_prices) / len(recent_prices)
        volatility = math.sqrt(variance)

        metadata = {
            "deviation": deviation,
            "momentum": momentum,
            "volatility": volatility,
            "tick_buffer": list(self._tick_buffer),
            "yes_token_id": self._yes_token_id,
        }

        # CHANGED: Use persistent connections
        if self._news_source:
            try:
                fg = await self._news_source.get_fear_greed_index()
                if fg and "value" in fg:
                    metadata["sentiment_score"] = float(fg["value"])
                    metadata["sentiment_classification"] = fg.get("classification", "")
                    logger.info(f"Fear & Greed: {metadata['sentiment_score']:.0f} ({metadata['sentiment_classification']})")
            except Exception as e:
                logger.warning(f"Fear & Greed fetch failed: {e}")

        if self._coinbase_source:
            try:
                spot = await self._coinbase_source.get_current_price()
                if spot:
                    metadata["spot_price"] = float(spot)
                    logger.info(f"Coinbase spot price: ${float(spot):,.2f}")
            except Exception as e:
                logger.warning(f"Coinbase fetch failed: {e}")

        logger.info(
            f"Market context — deviation={deviation:.2%}, "
            f"momentum={momentum:.2%}, volatility={volatility:.4f}, "
            f"sentiment={'%.0f' % metadata['sentiment_score'] if 'sentiment_score' in metadata else 'N/A'}, "
            f"spot=${'%.2f' % metadata['spot_price'] if 'spot_price' in metadata else 'N/A'}"
        )
        return metadata

    async def _make_trading_decision(self, current_price: Decimal):
        """
        Make trading decision using signal fusion.

        CHANGED FROM V1:
        - Fusion engine DRIVES direction (no trend override)
        - Requires min 2 signals to agree (no single-signal pass-through)
        - Coin-flip zone skip still applies as sanity check
        - Uses limit orders for 0% maker fee
        - Tracks open positions for resolution
        """
        is_simulation = await self.check_simulation_mode()
        logger.info(f"Mode: {'SIMULATION' if is_simulation else 'LIVE TRADING'}")

        # CHANGED: Lower minimum history requirement (no synthetic data to pad it)
        if len(self.price_history) < 5:
            logger.warning(f"Not enough price history ({len(self.price_history)}/5) — waiting for real data")
            return

        logger.info(f"Current price: ${float(current_price):,.4f}")

        # Phase 4a: Build real metadata for processors
        metadata = await self._fetch_market_context(current_price)

        # Phase 4b: Run all signal processors
        signals = self._process_signals(current_price, metadata)

        if not signals:
            logger.info("No signals generated — no trade this interval")
            return

        logger.info(f"Generated {len(signals)} signal(s):")
        for sig in signals:
            logger.info(
                f"  [{sig.source}] {sig.direction.value}: "
                f"score={sig.score:.1f}, confidence={sig.confidence:.2%}"
            )

        # =============================================================================
        # CHANGED: Fusion engine drives direction. Require 2+ signals, score >= 55.
        # This fixes the single-signal pass-through bug where consensus_score was
        # always 100 with one signal (dominant == total_contrib).
        # =============================================================================
        fused = self.fusion_engine.fuse_signals(
            signals,
            min_signals=MIN_FUSION_SIGNALS,
            min_score=MIN_FUSION_SCORE,
        )
        if not fused:
            logger.info(
                f"Fusion produced no actionable signal "
                f"(need {MIN_FUSION_SIGNALS}+ signals with score >= {MIN_FUSION_SCORE}) "
                f"— no trade this interval"
            )
            return

        logger.info(
            f"FUSED SIGNAL: {fused.direction.value} "
            f"(score={fused.score:.1f}, confidence={fused.confidence:.2%})"
        )

        # =============================================================================
        # CHANGED: Fusion direction drives the trade. No trend override.
        # The coin-flip skip zone remains as a safety net only.
        # At the early window (min 1-3), prices are usually near 0.50,
        # so this gate is mainly relevant if we somehow trade late.
        # =============================================================================
        price_float = float(current_price)

        # Coin-flip safety: if price is deep in no-man's land AND fusion is weak, skip
        if TREND_DOWN_THRESHOLD <= price_float <= TREND_UP_THRESHOLD:
            if fused.confidence < 0.70:
                logger.info(
                    f"⏭ SKIP: price ${price_float:.4f} in neutral zone "
                    f"({TREND_DOWN_THRESHOLD:.0%}–{TREND_UP_THRESHOLD:.0%}) "
                    f"AND fusion confidence {fused.confidence:.0%} < 70% — coin flip territory"
                )
                return

        # Direction from fusion engine
        if "BULLISH" in str(fused.direction).upper():
            direction = "long"
            logger.info(f"📈 FUSION SAYS: BUY YES (bullish, score={fused.score:.1f})")
        else:
            direction = "short"
            logger.info(f"📉 FUSION SAYS: BUY NO (bearish, score={fused.score:.1f})")

        # Risk engine: check position-count / exposure limits
        is_valid, error = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD,
            direction=direction,
            current_price=current_price,
        )
        if not is_valid:
            logger.warning(f"Risk engine blocked trade: {error}")
            return

        logger.info(f"Position size: ${POSITION_SIZE_USD} (fixed) | Direction: {direction.upper()}")

        # Liquidity guard
        last_tick = getattr(self, '_last_bid_ask', None)
        if last_tick:
            last_bid, last_ask = last_tick
            MIN_LIQUIDITY = Decimal("0.02")
            if direction == "long" and last_ask <= MIN_LIQUIDITY:
                logger.warning(f"⚠ No liquidity for BUY: ask=${float(last_ask):.4f} — skipping")
                self.last_trade_time = -1
                return
            if direction == "short" and last_bid <= MIN_LIQUIDITY:
                logger.warning(f"⚠ No liquidity for SELL: bid=${float(last_bid):.4f} — skipping")
                self.last_trade_time = -1
                return

        # Execute
        if is_simulation:
            await self._record_paper_trade(fused, POSITION_SIZE_USD, current_price, direction)
        else:
            if USE_LIMIT_ORDERS:
                await self._place_limit_order(fused, POSITION_SIZE_USD, current_price, direction)
            else:
                await self._place_real_order(fused, POSITION_SIZE_USD, current_price, direction)

    async def _record_paper_trade(self, signal, position_size, current_price, direction):
        exit_delta = timedelta(minutes=1) if self.test_mode else timedelta(minutes=15)
        exit_time = datetime.now(timezone.utc) + exit_delta

        if "BULLISH" in str(signal.direction):
            movement = random.uniform(-0.02, 0.08)
        else:
            movement = random.uniform(-0.08, 0.02)

        exit_price = current_price * (Decimal("1.0") + Decimal(str(movement)))
        exit_price = max(Decimal("0.01"), min(Decimal("0.99"), exit_price))

        if direction == "long":
            pnl = position_size * (exit_price - current_price) / current_price
        else:
            pnl = position_size * (current_price - exit_price) / current_price

        outcome = "WIN" if pnl > 0 else "LOSS"
        paper_trade = PaperTrade(
            timestamp=datetime.now(timezone.utc),
            direction=direction.upper(),
            size_usd=float(position_size),
            price=float(current_price),
            signal_score=signal.score,
            signal_confidence=signal.confidence,
            outcome=outcome,
        )
        self.paper_trades.append(paper_trade)

        self.performance_tracker.record_trade(
            trade_id=f"paper_{int(datetime.now().timestamp())}",
            direction=direction,
            entry_price=current_price,
            exit_price=exit_price,
            size=position_size,
            entry_time=datetime.now(timezone.utc),
            exit_time=exit_time,
            signal_score=signal.score,
            signal_confidence=signal.confidence,
            metadata={
                "simulated": True,
                "num_signals": signal.num_signals if hasattr(signal, 'num_signals') else 1,
                "fusion_score": signal.score,
            }
        )

        if hasattr(self, 'grafana_exporter') and self.grafana_exporter:
            self.grafana_exporter.increment_trade_counter(won=(pnl > 0))
            self.grafana_exporter.record_trade_duration(exit_delta.total_seconds())

        logger.info("=" * 80)
        logger.info("[SIMULATION] PAPER TRADE RECORDED")
        logger.info(f"  Direction: {direction.upper()}")
        logger.info(f"  Size: ${float(position_size):.2f}")
        logger.info(f"  Entry Price: ${float(current_price):,.4f}")
        logger.info(f"  Simulated Exit: ${float(exit_price):,.4f}")
        logger.info(f"  Simulated P&L: ${float(pnl):+.2f} ({movement*100:+.2f}%)")
        logger.info(f"  Outcome: {outcome}")
        logger.info(f"  Total Paper Trades: {len(self.paper_trades)}")
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
    # NEW: Limit order placement (maker, 0% fee)
    # ------------------------------------------------------------------

    async def _place_limit_order(self, signal, position_size, current_price, direction):
        """
        Place a LIMIT order instead of a MARKET order.

        Maker orders on Polymarket have 0% fee (vs 10% taker fee).
        We place slightly inside the spread to maximize fill probability
        while avoiding the taker penalty.

        For BUY YES: place limit at best_bid + offset (or best_ask - offset)
        For BUY NO:  same logic on the NO token's book
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
                    # Fall back to YES token price if NO has no quotes yet
                    token_bid = current_price
                    token_ask = current_price + Decimal("0.02")
            except Exception:
                token_bid = current_price
                token_ask = current_price + Decimal("0.02")

            # Place limit price just inside the spread (near the ask but not at it)
            # This makes us a maker (0% fee) rather than taker (10% fee)
            spread = token_ask - token_bid
            if spread > LIMIT_ORDER_OFFSET * 2:
                # Wide spread: place at ask - offset (aggressive maker)
                limit_price = token_ask - LIMIT_ORDER_OFFSET
            else:
                # Tight spread: place at midpoint
                limit_price = (token_bid + token_ask) / 2

            # Clamp to valid range
            limit_price = max(Decimal("0.01"), min(Decimal("0.99"), limit_price))

            # Calculate token quantity from USD amount and limit price
            token_qty = float(position_size / limit_price)
            precision = instrument.size_precision
            token_qty = round(token_qty, precision)
            token_qty = max(token_qty, 5.0)  # Polymarket minimum for limit orders

            logger.info("=" * 80)
            logger.info(f"LIMIT ORDER (MAKER, 0% FEE)")
            logger.info(f"  Buying {trade_label}: {token_qty:.2f} tokens @ ${float(limit_price):.4f}")
            logger.info(f"  Book: bid=${float(token_bid):.4f} / ask=${float(token_ask):.4f}")
            logger.info(f"  Spread: ${float(spread):.4f}")
            logger.info("=" * 80)

            qty = Quantity(token_qty, precision=precision)
            price = Price(float(limit_price), precision=instrument.price_precision)
            timestamp_ms = int(time.time() * 1000)
            unique_id = f"BTC-15MIN-LMT-{timestamp_ms}"

            # CHANGED: Use limit order instead of market
            order = self.order_factory.limit(
                instrument_id=trade_instrument_id,
                order_side=side,
                quantity=qty,
                price=price,
                client_order_id=ClientOrderId(unique_id),
                time_in_force=TimeInForce.GTC,  # Good till cancelled (or market end)
                post_only=True,  # Ensure maker-only (rejected if would cross spread)
            )

            self.submit_order(order)

            logger.info(f"LIMIT ORDER SUBMITTED!")
            logger.info(f"  Order ID: {unique_id}")
            logger.info(f"  Direction: {trade_label}")
            logger.info(f"  Limit Price: ${float(limit_price):.4f}")
            logger.info(f"  Quantity: {token_qty:.2f}")
            logger.info(f"  Estimated Cost: ~${float(position_size):.2f}")
            logger.info(f"  Fee: 0% (maker)")
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
    # Market order (kept as fallback, USE_LIMIT_ORDERS=false)
    # ------------------------------------------------------------------

    async def _place_real_order(self, signal, position_size, current_price, direction):
        if not self.instrument_id:
            logger.error("No instrument available")
            return

        try:
            logger.info("=" * 80)
            logger.info("LIVE MODE - PLACING MARKET ORDER (10% TAKER FEE!)")
            logger.info("⚠️  Consider setting USE_LIMIT_ORDERS=true for 0% maker fee")
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

            logger.info(f"Buying {trade_label} token: {trade_instrument_id}")

            trade_price = float(current_price)
            max_usd_amount = float(position_size)

            precision = instrument.size_precision
            min_qty_val = float(getattr(instrument, 'min_quantity', None) or 5.0)
            token_qty = max(min_qty_val, 5.0)
            token_qty = round(token_qty, precision)
            logger.info(
                f"BUY {trade_label}: dummy qty={token_qty:.6f} "
                f"(patch converts to ${max_usd_amount:.2f} USD)"
            )

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

            # Track position
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
    # Signal processing
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
    # Order events
    # ------------------------------------------------------------------

    def _track_order_event(self, event_type: str) -> None:
        try:
            pt = self.performance_tracker
            if hasattr(pt, 'record_order_event'):
                pt.record_order_event(event_type)
            elif hasattr(pt, 'increment_counter'):
                pt.increment_counter(event_type)
            elif hasattr(pt, 'increment_order_counter'):
                pt.increment_order_counter(event_type)
            else:
                logger.debug(f"PerformanceTracker has no order-counter method; ignoring event '{event_type}'")
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
                break

    def on_order_denied(self, event):
        logger.error("=" * 80)
        logger.error(f"ORDER DENIED!")
        logger.error(f"  Order: {event.client_order_id}")
        logger.error(f"  Reason: {event.reason}")
        logger.error("=" * 80)
        self._track_order_event("rejected")

    def on_order_rejected(self, event):
        """
        Handle order rejection.
        CHANGED: Rate-limit retries to prevent FAK storms.
        """
        reason = str(getattr(event, 'reason', ''))
        reason_lower = reason.lower()
        if 'no orders found' in reason_lower or 'fak' in reason_lower or 'no match' in reason_lower:
            self._retry_count_this_window += 1
            if self._retry_count_this_window <= MAX_RETRIES_PER_WINDOW:
                logger.warning(
                    f"⚠ FAK rejected (no liquidity) — retry {self._retry_count_this_window}/{MAX_RETRIES_PER_WINDOW}\n"
                    f"  Reason: {reason}"
                )
                self.last_trade_time = -1  # Allow retry
            else:
                logger.warning(
                    f"⚠ FAK rejected — max retries ({MAX_RETRIES_PER_WINDOW}) reached, giving up this window\n"
                    f"  Reason: {reason}"
                )
                # Do NOT reset last_trade_time — skip this window
        else:
            logger.warning(f"Order rejected: {reason}")

    # ------------------------------------------------------------------
    # Grafana / stop
    # ------------------------------------------------------------------

    def _start_grafana_sync(self):
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.grafana_exporter.start())
            logger.info("Grafana metrics started on port 8000")
        except Exception as e:
            logger.error(f"Failed to start Grafana: {e}")

    def on_stop(self):
        logger.info("Integrated BTC strategy V2 stopped")
        logger.info(f"Total paper trades recorded: {len(self.paper_trades)}")

        # Resolve any remaining open positions
        for pos in self._open_positions:
            if not pos.resolved:
                logger.info(f"Unresolved position: {pos.market_slug} {pos.direction} @ ${pos.entry_price:.4f}")

        if self.grafana_exporter:
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.grafana_exporter.stop())
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_integrated_bot(simulation: bool = False, enable_grafana: bool = True, test_mode: bool = False):
    """Run the integrated BTC 15-min trading bot V2"""

    print("=" * 80)
    print("INTEGRATED POLYMARKET BTC 15-MIN TRADING BOT V2")
    print("EARLY WINDOW + MAKER ORDERS + SIGNAL-DRIVEN")
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
    print(f"  Order Type: {'LIMIT (maker, 0% fee)' if USE_LIMIT_ORDERS else 'MARKET (taker, 10% fee)'}")
    print(f"  Trade Window: {TRADE_WINDOW_START_SEC}s–{TRADE_WINDOW_END_SEC}s ({TRADE_WINDOW_START_SEC/60:.0f}–{TRADE_WINDOW_END_SEC/60:.0f} min)")
    print(f"  Fusion: min {MIN_FUSION_SIGNALS} signals, score >= {MIN_FUSION_SCORE}")
    print(f"  Stability gate: {QUOTE_STABILITY_REQUIRED} valid ticks")
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
        trader_id="BTC-15MIN-INTEGRATED-001",
        logging=LoggingConfig(
            log_level="ERROR",
            log_directory="./logs/nautilus",
            log_component_levels={
                "IntegratedBTCStrategy": "INFO",
            },
        ),
        data_engine=LiveDataEngineConfig(qsize=6000),
        exec_engine=LiveExecEngineConfig(qsize=6000),
        # CHANGED: Don't bypass risk engine even in simulation
        # (we want to validate order sizing regardless of mode)
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
    print("BOT V2 STARTING")
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

    parser = argparse.ArgumentParser(description="Integrated BTC 15-Min Trading Bot V2")
    parser.add_argument("--live", action="store_true",
                        help="Run in LIVE mode (real money at risk!). Default is simulation.")
    parser.add_argument("--no-grafana", action="store_true", help="Disable Grafana metrics")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in TEST MODE (trade every minute for faster testing)")

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
        logger.warning(f"Order type: {'LIMIT (0% fee)' if USE_LIMIT_ORDERS else 'MARKET (10% fee)'}")
        logger.warning("=" * 80)
    else:
        logger.info("=" * 80)
        logger.info(f"SIMULATION MODE — {'TEST MODE (fast clock)' if test_mode else 'paper trading only'}")
        logger.info("No real orders will be placed.")
        logger.info("=" * 80)

    run_integrated_bot(simulation=simulation, enable_grafana=enable_grafana, test_mode=test_mode)


if __name__ == "__main__":
    main()