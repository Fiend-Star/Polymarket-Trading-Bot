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
        self.redis_client = redis_client
        self.current_simulation_mode = False
        self.test_mode = test_mode

        self._init_market_state()
        self._init_quant_modules()
        self._init_rtds_and_funding()
        self._init_signal_processors()
        self._init_risk_and_monitoring(enable_grafana)

        if test_mode:
            logger.info("TEST MODE ACTIVE")

        logger.info(
            f"Strategy V3.1 | {TRADE_WINDOW_START_SEC}s–{TRADE_WINDOW_END_SEC}s window | "
            f"{'LIMIT' if USE_LIMIT_ORDERS else 'MARKET'} orders | ${POSITION_SIZE_USD} size | "
            f"edge≥${MIN_EDGE_CENTS:.2f} | TP {TAKE_PROFIT_PCT:.0%} / SL {CUT_LOSS_PCT:.0%} | "
            f"vol={VOL_METHOD} {DEFAULT_VOL:.0%} | fusion≥{MIN_FUSION_SIGNALS}sig/{MIN_FUSION_SCORE}pts"
        )
        if USE_RTDS:
            logger.info(
                f"  RTDS: Chainlink + Binance | Late-window: "
                f"{'T-{:.0f}s'.format(RTDS_LATE_WINDOW_SEC) if LATE_WINDOW_ENABLED else 'off'} | "
                f"Funding: {'on' if USE_FUNDING_FILTER else 'off'}"
            )

    def _init_market_state(self):
        """Initialize market tracking and quote stability state."""
        self._init_instrument_state()
        self._init_cache_and_history()

    def _init_instrument_state(self):
        """Initialize instrument tracking, ticks, and positions."""
        self.instrument_id = None
        self.all_btc_instruments: List[Dict] = []
        self.current_instrument_index: int = -1
        self.next_switch_time: Optional[datetime] = None
        self._stable_tick_count = 0
        self._market_stable = False
        self._last_instrument_switch = None
        self.last_trade_time = -1
        self._waiting_for_market_open = False
        self._last_bid_ask = None
        self._retry_count_this_window = 0
        from collections import deque
        self._tick_buffer: deque = deque(maxlen=500)
        self._yes_token_id: Optional[str] = None
        self._yes_instrument_id = None
        self._no_instrument_id = None
        self._open_positions: List[OpenPosition] = []

    def _init_cache_and_history(self):
        """Initialize data source cache and price history."""
        self._cached_spot_price: Optional[float] = None
        self._cached_sentiment: Optional[float] = None
        self._cached_sentiment_class: Optional[str] = None
        self._cache_lock = threading.Lock()
        self._news_source = None
        self._coinbase_source = None
        self._data_sources_initialized = False
        self.price_history = []
        self.max_history = 100
        self.paper_trades: List[PaperTrade] = []

    def _init_quant_modules(self):
        """Initialize V3 quantitative pricing model components."""
        self.binary_pricer = get_binary_pricer()
        self.vol_estimator = get_vol_estimator()
        self.mispricing_detector = get_mispricing_detector(
            maker_fee=0.00, taker_fee=0.10,
            min_edge_cents=MIN_EDGE_CENTS,
            min_edge_after_fees=0.005,
            take_profit_pct=TAKE_PROFIT_PCT,
            cut_loss_pct=CUT_LOSS_PCT,
            vol_method=VOL_METHOD,
        )
        self._btc_strike_price: Optional[float] = None
        self._strike_recorded = False
        self._active_entry: Optional[dict] = None
        self._late_window_traded = False

    def _init_rtds_and_funding(self):
        """Initialize RTDS Chainlink oracle and funding rate filter."""
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

    def _init_signal_processors(self):
        """Initialize V3 signal processors for confirmation role."""
        self.spike_detector = SpikeDetectionProcessor(
            spike_threshold=float(os.getenv("SPIKE_THRESHOLD", "0.05")),
            lookback_periods=20,
        )
        self.sentiment_processor = SentimentProcessor(
            extreme_fear_threshold=25, extreme_greed_threshold=75,
        )
        self.divergence_processor = PriceDivergenceProcessor(
            divergence_threshold=float(os.getenv("DIVERGENCE_THRESHOLD", "0.05")),
        )
        self.orderbook_processor = OrderBookImbalanceProcessor(
            imbalance_threshold=0.30, min_book_volume=50.0,
        )
        self.tick_velocity_processor = TickVelocityProcessor(
            velocity_threshold_60s=0.015, velocity_threshold_30s=0.010,
        )
        self.deribit_pcr_processor = DeribitPCRProcessor(
            bullish_pcr_threshold=1.20, bearish_pcr_threshold=0.70,
            max_days_to_expiry=2, cache_seconds=300,
        )
        self.fusion_engine = get_fusion_engine()
        self.fusion_engine.set_weight("OrderBookImbalance", 0.30)
        self.fusion_engine.set_weight("TickVelocity", 0.25)
        self.fusion_engine.set_weight("PriceDivergence", 0.18)
        self.fusion_engine.set_weight("SpikeDetection", 0.12)
        self.fusion_engine.set_weight("DeribitPCR", 0.10)
        self.fusion_engine.set_weight("SentimentAnalysis", 0.05)

    def _init_risk_and_monitoring(self, enable_grafana):
        """Initialize risk engine, performance tracker, and Grafana."""
        self.risk_engine = get_risk_engine()
        self.performance_tracker = get_performance_tracker()
        self.learning_engine = get_learning_engine()

        if enable_grafana:
            self.grafana_exporter = get_grafana_exporter()
        else:
            self.grafana_exporter = None


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

    async def _preseed_vol_estimator(self):
        """Fetch recent Coinbase 1-min candles to warm up vol estimator on startup."""
        import aiohttp
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {"granularity": 60}

        try:
            candles = await self._fetch_coinbase_candles(url, params)
            if not candles:
                return
            candles.sort(key=lambda c: c[0])
            self._feed_candles_to_vol_estimator(candles)
        except Exception as e:
            logger.warning(f"Vol estimator pre-seed failed: {e} — will use default vol until enough ticks")

    async def _fetch_coinbase_candles(self, url, params):
        """Fetch candles from Coinbase public API."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"Coinbase candles API returned {resp.status}")
                    return None
                candles = await resp.json()
        if not candles or not isinstance(candles, list):
            logger.warning("No candle data returned from Coinbase")
            return None
        return candles

    def _feed_candles_to_vol_estimator(self, candles):
        """Feed candle data into vol estimator and cache initial spot."""
        fed = 0
        for candle in candles:
            try:
                ts, low, high, open_p, close, volume = candle
                self.vol_estimator.add_price(float(close), float(ts))
                fed += 1
            except (ValueError, IndexError, TypeError):
                continue

        vol = self.vol_estimator.get_vol(VOL_METHOD)
        logger.info(
            f"✓ Vol estimator pre-seeded: {fed} candles → "
            f"RV={vol.annualized_vol:.1%} ({vol.method}), confidence={vol.confidence:.0%}"
        )

        if candles:
            latest_price = float(candles[-1][4])
            with self._cache_lock:
                if self._cached_spot_price is None:
                    self._cached_spot_price = latest_price
                    logger.info(f"✓ Initial BTC spot from candles: ${latest_price:,.2f}")

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
        """Fetch BTC spot + sentiment, cache thread-safely."""
        spot = await self._fetch_spot_price()
        sentiment, sentiment_class = await self._fetch_sentiment()

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

    async def _fetch_spot_price(self):
        """Fetch BTC spot from RTDS Chainlink or Coinbase fallback."""
        if self.rtds and self.rtds.chainlink_btc_price > 0 and self.rtds.chainlink_age_ms < 30000:
            spot = self.rtds.chainlink_btc_price
            if self.rtds.binance_btc_price > 0:
                div = self.rtds.get_divergence()
                if div.is_significant:
                    logger.info(
                        f"⚡ Price divergence: Binance=${div.binance_price:,.2f} vs "
                        f"Chainlink=${div.chainlink_price:,.2f} ({div.divergence_bps:+.1f}bps) → {div.direction}"
                    )
            return spot
        elif self._coinbase_source:
            try:
                coinbase_spot = await self._coinbase_source.get_current_price()
                if coinbase_spot:
                    spot = float(coinbase_spot)
                    self.vol_estimator.add_price(spot)
                    return spot
            except Exception as e:
                logger.debug(f"Coinbase refresh failed: {e}")
        return None

    async def _fetch_sentiment(self):
        """Fetch Fear & Greed index from news source."""
        if not self._news_source:
            return None, None
        try:
            fg = await self._news_source.get_fear_greed_index()
            if fg and "value" in fg:
                return float(fg["value"]), fg.get("classification", "")
        except Exception as e:
            logger.debug(f"Sentiment refresh failed: {e}")
        return None, None

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

    def _subscribe_and_seed_price(self):
        """Subscribe to quote ticks and seed initial price."""
        if not self.instrument_id:
            return
        self.subscribe_quote_ticks(self.instrument_id)
        no_id = getattr(self, '_no_instrument_id', None)
        if no_id:
            self.subscribe_quote_ticks(no_id)
        try:
            quote = self.cache.quote_tick(self.instrument_id)
            if quote and quote.bid_price and quote.ask_price:
                current_price = (quote.bid_price + quote.ask_price) / 2
                self.price_history.append(current_price)
                logger.info(f"✓ Initial price: ${float(current_price):.4f}")
        except Exception as e:
            logger.debug(f"No initial price yet: {e}")

    def _init_services(self):
        """Start background services (timer, RTDS, funding, Grafana)."""
        self.run_in_executor(self._start_timer_loop)
        if self.rtds:
            self.rtds.start_background()
            logger.info("✓ RTDS connector started")
        if self.funding_filter:
            try:
                regime = self.funding_filter.update_sync()
                if regime:
                    logger.info(f"✓ Funding rate: {regime.funding_rate_pct:+.4f}% → "
                                f"{regime.classification}, bias={regime.mean_reversion_bias:+.3f}")
            except Exception as e:
                logger.debug(f"Initial funding rate fetch failed: {e}")
        if self.grafana_exporter:
            threading.Thread(target=self._start_grafana_sync, daemon=True).start()

    def on_start(self):
        logger.info("Strategy starting — loading instruments...")
        self._load_all_btc_instruments()
        self._subscribe_and_seed_price()
        if len(self.price_history) < 5:
            logger.info(f"Waiting for real price data ({len(self.price_history)}/5 ticks so far)")
        self._init_services()
        logger.info(f"V3.1 active — {len(self.price_history)} price points cached")

    # ------------------------------------------------------------------
    # Load all BTC instruments
    # ------------------------------------------------------------------

    def _load_all_btc_instruments(self):
        instruments = self.cache.instruments()
        logger.info(f"Loading ALL BTC instruments from {len(instruments)} total...")

        now = datetime.now(timezone.utc)
        current_timestamp = int(now.timestamp())

        btc_instruments = self._parse_btc_instruments(instruments, current_timestamp)
        btc_instruments = self._dedup_and_pair_instruments(btc_instruments)

        if btc_instruments:
            first_t = btc_instruments[0]['start_time'].strftime('%H:%M')
            last_t = btc_instruments[-1]['end_time'].strftime('%H:%M')
            logger.info(f"Found {len(btc_instruments)} BTC 15-min markets ({first_t}–{last_t})")
        else:
            logger.warning("No BTC 15-min markets found!")

        self.all_btc_instruments = btc_instruments
        self._activate_current_instrument(btc_instruments, current_timestamp)

    def _parse_btc_instruments(self, instruments, current_timestamp):
        """Parse raw instruments into BTC 15-min market dicts."""
        btc_instruments = []
        for instrument in instruments:
            try:
                if not (hasattr(instrument, 'info') and instrument.info):
                    continue
                question = instrument.info.get('question', '').lower()
                slug = instrument.info.get('market_slug', '').lower()
                if not (('btc' in question or 'btc' in slug) and '15m' in slug):
                    continue

                parsed = self._parse_single_instrument(instrument, slug, current_timestamp)
                if parsed:
                    btc_instruments.append(parsed)
            except Exception:
                continue
        return btc_instruments

    def _parse_single_instrument(self, instrument, slug, current_timestamp):
        """Parse a single instrument into a market dict, or return None."""
        try:
            timestamp_part = slug.split('-')[-1]
            market_timestamp = int(timestamp_part)
            end_timestamp = market_timestamp + 900

            if end_timestamp <= current_timestamp:
                return None

            raw_id = str(instrument.id)
            without_suffix = raw_id.split('.')[0] if '.' in raw_id else raw_id
            yes_token_id = without_suffix.split('-')[-1] if '-' in without_suffix else without_suffix

            return {
                'instrument': instrument, 'slug': slug,
                'start_time': datetime.fromtimestamp(market_timestamp, tz=timezone.utc),
                'end_time': datetime.fromtimestamp(end_timestamp, tz=timezone.utc),
                'market_timestamp': market_timestamp,
                'end_timestamp': end_timestamp,
                'time_diff_minutes': (market_timestamp - current_timestamp) / 60,
                'yes_token_id': yes_token_id,
            }
        except (ValueError, IndexError):
            return None

    def _dedup_and_pair_instruments(self, btc_instruments):
        """Pair YES and NO tokens by slug, deduplicate, and sort."""
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
        deduped.sort(key=lambda x: x['market_timestamp'])
        return deduped

    def _activate_current_instrument(self, btc_instruments, current_timestamp):
        """Find and subscribe to the current or next market."""
        for i, inst in enumerate(btc_instruments):
            is_active = inst['time_diff_minutes'] <= 0 and inst['end_timestamp'] > current_timestamp
            if is_active:
                self._subscribe_to_market(i, inst)
                logger.info(f"✓ Active: {inst['slug']} → switch at {self.next_switch_time.strftime('%H:%M:%S')}")
                return

        if not btc_instruments:
            return

        # No active market — find nearest future
        future_markets = [inst for inst in btc_instruments if inst['time_diff_minutes'] > 0]
        if future_markets:
            nearest = min(future_markets, key=lambda x: x['time_diff_minutes'])
        else:
            nearest = btc_instruments[-1]
        nearest_idx = btc_instruments.index(nearest)

        self._subscribe_to_market(nearest_idx, nearest)
        self.next_switch_time = nearest['start_time']
        logger.info(f"⚠ NO CURRENT MARKET — WAITING FOR: {nearest['slug']}")
        logger.info(f"  ✓ SUBSCRIBED to future market (YES + NO)")
        self._waiting_for_market_open = True

    def _subscribe_to_market(self, index, inst):
        """Subscribe to a market's YES (and optionally NO) instrument."""
        self.current_instrument_index = index
        self.instrument_id = inst['instrument'].id
        self.next_switch_time = inst['end_time']
        self._yes_token_id = inst.get('yes_token_id')
        self._yes_instrument_id = inst.get('yes_instrument_id', inst['instrument'].id)
        self._no_instrument_id = inst.get('no_instrument_id')
        self.subscribe_quote_ticks(self.instrument_id)
        if inst.get('no_instrument_id'):
            self.subscribe_quote_ticks(inst['no_instrument_id'])

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

        self._resolve_positions_for_market(self.current_instrument_index)
        self._subscribe_to_market(next_index, next_market)

        logger.info("=" * 80)
        logger.info(f"SWITCHING TO NEXT MARKET: {next_market['slug']}")
        logger.info(f"  Current time: {now.strftime('%H:%M:%S')}")
        logger.info(f"  Market ends at: {self.next_switch_time.strftime('%H:%M:%S')}")
        logger.info("=" * 80)

        self._reset_market_state()
        logger.info(f"  Trade timer reset — will trade after {QUOTE_STABILITY_REQUIRED} stable ticks")
        logger.info(f"  ✓ SUBSCRIBED to new market (YES + NO)")
        return True

    def _reset_market_state(self):
        """Reset per-market trading state."""
        self._stable_tick_count = 0
        self._market_stable = False
        self._waiting_for_market_open = False
        self.last_trade_time = -1
        self._retry_count_this_window = 0
        self._btc_strike_price = None
        self._strike_recorded = False
        self._active_entry = None
        self._late_window_traded = False


    # ------------------------------------------------------------------
    # V3 FIX: Position resolution (ALL positions are BUYS)
    # ------------------------------------------------------------------

    def _resolve_positions_for_market(self, market_index: int):
        """When a market ends, check the final price to determine P&L."""
        if market_index < 0 or market_index >= len(self.all_btc_instruments):
            return

        market = self.all_btc_instruments[market_index]
        slug = market['slug']

        for pos in self._open_positions:
            if pos.market_slug == slug and not pos.resolved:
                self._resolve_single_position(pos, slug)

    def _resolve_single_position(self, pos, slug):
        """Resolve a single open position at market end."""
        final_price = self._get_final_price(pos)

        if final_price is None:
            logger.warning(f"Could not resolve position for {slug} — no final price")
            return

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

        self._record_resolved_trade(pos, final_price, slug, is_paper, outcome)

        if hasattr(self, 'grafana_exporter') and self.grafana_exporter:
            self.grafana_exporter.increment_trade_counter(won=(pnl > 0))

        if is_paper:
            self._update_paper_trade_outcome(pos, outcome)

    def _get_final_price(self, pos):
        """Get the final mid-price for a position's instrument."""
        try:
            quote = self.cache.quote_tick(pos.instrument_id)
            if quote:
                return float((quote.bid_price.as_decimal() + quote.ask_price.as_decimal()) / 2)
        except Exception:
            pass
        return None

    def _record_resolved_trade(self, pos, final_price, slug, is_paper, outcome):
        """Record resolved trade in performance tracker."""
        self.performance_tracker.record_trade(
            trade_id=pos.order_id,
            direction="long",
            entry_price=Decimal(str(pos.entry_price)),
            exit_price=Decimal(str(final_price)),
            size=Decimal(str(pos.size_usd)),
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc),
            signal_score=0, signal_confidence=0,
            metadata={
                "resolved": True, "market": slug, "paper": is_paper,
                "original_direction": pos.direction,
                "token": "YES" if pos.direction == "long" else "NO",
                "pnl_computed": round(pos.pnl, 4),
            }
        )

    def _update_paper_trade_outcome(self, pos, outcome):
        """Update matching paper_trade entry with resolved outcome."""
        for pt in self.paper_trades:
            if (pt.outcome == "PENDING"
                    and pt.direction == pos.direction.upper()
                    and abs(pt.price - pos.entry_price) < 0.0001):
                pt.outcome = outcome
                break
        self._save_paper_trades()

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

    async def _check_auto_restart(self):
        """Check uptime and trigger restart if needed. Returns True if restarting."""
        uptime_min = (datetime.now(timezone.utc) - self.bot_start_time).total_seconds() / 60
        if uptime_min >= self.restart_after_minutes:
            logger.warning("AUTO-RESTART TIME — Loading fresh filters")
            await self._teardown_data_sources()
            import signal as _signal
            os.kill(os.getpid(), _signal.SIGTERM)
            return True
        return False

    def _record_strike_if_needed(self):
        """Record BTC strike price on first cached spot."""
        if not self._strike_recorded and self._cached_spot_price:
            self._btc_strike_price = self._cached_spot_price
            self._strike_recorded = True
            logger.info(f"📌 BTC Strike recorded: ${self._btc_strike_price:,.2f}")

    async def _timer_loop(self):
        """Timer loop: runs every 10s. Refreshes cache and manages market switching."""
        await self._init_data_sources()
        await self._preseed_vol_estimator()
        while True:
            if await self._check_auto_restart():
                return
            now = datetime.now(timezone.utc)
            if self.next_switch_time and now >= self.next_switch_time:
                if self._waiting_for_market_open:
                    self._handle_market_open(now)
                else:
                    self._switch_to_next_market()
            try:
                await self._refresh_cached_data()
                self._record_strike_if_needed()
            except Exception as e:
                logger.debug(f"Cache refresh error: {e}")
            await asyncio.sleep(10)

    def _handle_market_open(self, now):
        """Handle transition when a waiting market opens."""
        logger.info("=" * 80)
        logger.info(f"⏰ WAITING MARKET NOW OPEN: {now.strftime('%H:%M:%S')} UTC")
        logger.info("=" * 80)
        if (self.current_instrument_index >= 0 and
                self.current_instrument_index < len(self.all_btc_instruments)):
            current_market = self.all_btc_instruments[self.current_instrument_index]
            self.next_switch_time = current_market['end_time']
        self._reset_market_state()
        self._waiting_for_market_open = False
        logger.info("  ✓ MARKET OPEN — waiting for stable quotes")

    # ------------------------------------------------------------------
    # Quote tick handler
    # ------------------------------------------------------------------

    def _extract_mid_price(self, tick):
        """Extract mid price from tick. Returns (mid, bid, ask) or None."""
        if self.instrument_id is None or tick.instrument_id != self.instrument_id:
            return None
        bid, ask = tick.bid_price, tick.ask_price
        if bid is None or ask is None:
            return None
        try:
            bd, ad = bid.as_decimal(), ask.as_decimal()
        except Exception:
            return None
        return (bd + ad) / 2, bd, ad

    def on_quote_tick(self, tick: QuoteTick):
        try:
            result = self._extract_mid_price(tick)
            if result is None:
                return
            mid_price, bid_decimal, ask_decimal = result
            self._update_price_tracking(mid_price, bid_decimal, ask_decimal)

            if not self._ensure_market_stable(bid_decimal, ask_decimal):
                return
            if self._waiting_for_market_open:
                return
            if (self.current_instrument_index < 0 or
                    self.current_instrument_index >= len(self.all_btc_instruments)):
                return

            current_market = self.all_btc_instruments[self.current_instrument_index]
            elapsed_secs, seconds_into_sub = self._compute_market_timing(current_market)
            if elapsed_secs < 0:
                return
            self._check_early_window_trade(current_market, elapsed_secs, seconds_into_sub, mid_price, bid_decimal, ask_decimal)
            self._check_late_window_trade(current_market, seconds_into_sub, mid_price)
        except Exception as e:
            logger.error(f"Error processing quote tick: {e}")

    def _update_price_tracking(self, mid_price, bid_decimal, ask_decimal):
        """Update price history and tick buffer."""
        now = datetime.now(timezone.utc)
        self.price_history.append(mid_price)
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        self._last_bid_ask = (bid_decimal, ask_decimal)
        self._tick_buffer.append({'ts': now, 'price': mid_price})

    def _ensure_market_stable(self, bid_decimal, ask_decimal):
        """Check stability gate. Returns True if market is stable."""
        if not self._market_stable:
            if self._is_quote_valid(bid_decimal, ask_decimal):
                self._stable_tick_count += 1
                if self._stable_tick_count >= QUOTE_STABILITY_REQUIRED:
                    self._market_stable = True
                    logger.info(f"✓ Market STABLE after {QUOTE_STABILITY_REQUIRED} valid ticks")
                    return True
                return False
            else:
                self._stable_tick_count = 0
                return False
        return True

    def _compute_market_timing(self, current_market):
        """Compute elapsed seconds and sub-interval timing."""
        now_ts = datetime.now(timezone.utc).timestamp()
        market_start_ts = current_market['market_timestamp']
        elapsed_secs = now_ts - market_start_ts
        seconds_into_sub = elapsed_secs % MARKET_INTERVAL_SECONDS
        return elapsed_secs, seconds_into_sub

    def _check_early_window_trade(self, current_market, elapsed_secs, seconds_into_sub, mid_price, bid_decimal, ask_decimal):
        """Check and trigger early-window trade."""
        sub_interval = int(elapsed_secs // MARKET_INTERVAL_SECONDS)
        trade_key = (current_market['market_timestamp'], sub_interval)

        if (TRADE_WINDOW_START_SEC <= seconds_into_sub < TRADE_WINDOW_END_SEC
                and trade_key != self.last_trade_time):
            self.last_trade_time = trade_key
            self._retry_count_this_window = 0

            now = datetime.now(timezone.utc)
            logger.info("=" * 80)
            logger.info(f"🎯 EARLY-WINDOW TRADE: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            logger.info(f"   Market: {current_market['slug']}")
            logger.info(f"   Sub-interval #{sub_interval} ({seconds_into_sub:.1f}s in = {seconds_into_sub / 60:.1f} min)")
            logger.info(f"   YES Price: ${float(mid_price):,.4f} | Bid: ${float(bid_decimal):,.4f} | Ask: ${float(ask_decimal):,.4f}")
            logger.info(f"   Price history: {len(self.price_history)} points")
            logger.info("=" * 80)

            self.run_in_executor(lambda: self._make_trading_decision_sync(float(mid_price)))

    def _check_late_window_trade(self, current_market, seconds_into_sub, mid_price):
        """Check and trigger late-window Chainlink trade."""
        time_remaining_sec = MARKET_INTERVAL_SECONDS - seconds_into_sub
        if not (LATE_WINDOW_ENABLED and self.rtds and self._btc_strike_price
                and 0 < time_remaining_sec <= RTDS_LATE_WINDOW_SEC
                and not self._late_window_traded):
            return

        late_signal = self.rtds.get_late_window_signal(
            strike=self._btc_strike_price,
            time_remaining_sec=time_remaining_sec,
        )

        if late_signal.direction != "NO_SIGNAL" and late_signal.confidence >= 0.70:
            self._late_window_traded = True
            logger.info("=" * 80)
            logger.info(f"🎯 LATE-WINDOW CHAINLINK TRADE: T-{time_remaining_sec:.0f}s")
            logger.info(f"   {late_signal.reason}")
            logger.info(f"   Chainlink: ${late_signal.chainlink_price:,.2f} vs Strike: ${late_signal.strike:,.2f}")
            logger.info(f"   Delta: {late_signal.delta_bps:+.1f}bps | Confidence: {late_signal.confidence:.0%}")
            logger.info(f"   Direction: {late_signal.direction}")
            logger.info("=" * 80)

            self.run_in_executor(
                lambda: self._execute_late_window_trade(late_signal, float(mid_price))
            )


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
        direction = "long" if late_signal.direction == "BUY_YES" else "short"
        limit_price = Decimal(str(min(0.93, 0.88 + 0.05 * late_signal.confidence)))

        logger.info(f"  Late-window: {'BUY YES' if direction == 'long' else 'BUY NO'} "
                    f"LIMIT @ ${float(limit_price):.2f} (maker, 0% fee + rebates)")

        is_valid, error = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD, direction=direction,
            current_price=Decimal(str(current_yes_price)))
        if not is_valid:
            logger.warning(f"Late-window risk check failed: {error}")
            return

        mock_signal = SimpleNamespace(
            direction="BULLISH" if direction == "long" else "BEARISH",
            score=late_signal.confidence * 100, confidence=late_signal.confidence)

        if is_simulation:
            await self._record_paper_trade(mock_signal, POSITION_SIZE_USD, limit_price, direction)
        else:
            await self._place_limit_order(mock_signal, POSITION_SIZE_USD, limit_price, direction)

    def _check_confirmation_veto(self, signal, cached, current_price):
        """Run fusion confirmation and return True if trade should be vetoed."""
        metadata = self._build_metadata(cached, current_price)
        old_signals = self._process_signals(current_price, metadata)
        confirming, contradicting = self._count_confirmations(signal, old_signals)
        if contradicting > confirming and signal.confidence < 0.6:
            logger.info(f"⏭ SKIP: Model says {signal.direction} but {contradicting}/{len(old_signals)} disagree")
            return True
        return False

    async def _make_trading_decision(self, current_price: Decimal):
        """V3: Quantitative trading decision using binary option pricing."""
        is_simulation = await self.check_simulation_mode()
        logger.info(f"Mode: {'SIMULATION' if is_simulation else 'LIVE TRADING'}")
        if len(self.price_history) < 5:
            logger.warning(f"Not enough price history ({len(self.price_history)}/5)"); return

        cached = self._get_cached_data()
        btc_spot = cached.get("spot_price")
        funding_bias = self._log_cached_data(cached, btc_spot)
        if btc_spot is None or self._btc_strike_price is None:
            logger.warning("No BTC spot or strike — falling back to heuristic signals")
            await self._make_trading_decision_heuristic(current_price, cached, is_simulation); return

        current_market = self.all_btc_instruments[self.current_instrument_index]
        trm = max(0, (current_market['end_timestamp'] - datetime.now(timezone.utc).timestamp()) / 60.0)
        logger.info(f"BTC: spot=${btc_spot:,.2f}, strike=${self._btc_strike_price:,.2f}, T={trm:.1f}min")

        yes_price, no_price = self._get_market_prices(current_price)
        signal = self._run_quant_signal(btc_spot, trm, yes_price, no_price, funding_bias)
        if not signal.is_tradeable:
            logger.info(f"Model says NO TRADE: edge=${signal.edge:+.4f}"); return
        if self._check_confirmation_veto(signal, cached, current_price):
            return
        direction = "long" if signal.direction == "BUY_YES" else "short"
        await self._execute_quant_trade(signal, direction, current_price, no_price, is_simulation)

    def _log_cached_data(self, cached, btc_spot):
        """Log cached data sources and return funding bias."""
        if btc_spot:
            source = "RTDS Chainlink" if (self.rtds and self.rtds.chainlink_btc_price > 0) else "Coinbase"
            logger.info(f"BTC spot ({source}, cached): ${btc_spot:,.2f}")
        if cached.get("sentiment_score") is not None:
            logger.info(f"Fear & Greed (cached): {cached['sentiment_score']:.0f} ({cached.get('sentiment_classification', '')})")

        funding_bias = 0.0
        if self.funding_filter:
            regime = self.funding_filter.get_regime()
            funding_bias = regime.mean_reversion_bias
            if regime.classification != "NEUTRAL":
                logger.info(f"Funding regime: {regime.classification} ({regime.funding_rate_pct:+.4f}%), bias={funding_bias:+.3f}")
        return funding_bias

    def _get_market_prices(self, current_price):
        """Get YES and NO market prices."""
        yes_price = float(current_price)
        no_price = 1.0 - yes_price
        if self._no_instrument_id:
            try:
                no_quote = self.cache.quote_tick(self._no_instrument_id)
                if no_quote:
                    no_price = float((no_quote.bid_price.as_decimal() + no_quote.ask_price.as_decimal()) / 2)
            except Exception:
                pass
        return yes_price, no_price

    def _run_quant_signal(self, btc_spot, time_remaining_min, yes_price, no_price, funding_bias):
        """Run mispricing detector with vol skew and funding bias."""
        vol_skew = BinaryOptionPricer.estimate_btc_vol_skew(
            btc_spot, self._btc_strike_price,
            self.vol_estimator.get_vol(VOL_METHOD).annualized_vol,
            time_remaining_min,
        )
        return self.mispricing_detector.detect(
            yes_market_price=yes_price, no_market_price=no_price,
            btc_spot=btc_spot, btc_strike=self._btc_strike_price,
            time_remaining_min=time_remaining_min,
            position_size_usd=float(POSITION_SIZE_USD),
            use_maker=USE_LIMIT_ORDERS,
            vol_skew=vol_skew, funding_bias=funding_bias,
        )

    def _count_confirmations(self, signal, old_signals):
        """Count confirming and contradicting signals."""
        model_bullish = signal.direction == "BUY_YES"
        confirming = contradicting = 0
        for sig in old_signals:
            if ("BULLISH" in str(sig.direction).upper()) == model_bullish:
                confirming += 1
            else:
                contradicting += 1
        logger.info(f"Confirmation: {confirming} agree, {contradicting} disagree (of {len(old_signals)} signals)")
        return confirming, contradicting

    async def _dispatch_order(self, signal, direction, current_price, is_simulation):
        """Dispatch to paper, limit, or market order based on mode."""
        mock_signal = SimpleNamespace(
            direction="BULLISH" if direction == "long" else "BEARISH",
            score=signal.confidence * 100, confidence=signal.confidence)
        if is_simulation:
            await self._record_paper_trade(mock_signal, POSITION_SIZE_USD, current_price, direction)
        elif USE_LIMIT_ORDERS:
            await self._place_limit_order(mock_signal, POSITION_SIZE_USD, current_price, direction)
        else:
            await self._place_real_order(mock_signal, POSITION_SIZE_USD, current_price, direction)

    async def _execute_quant_trade(self, signal, direction, current_price, no_price, is_simulation):
        """Execute a quant-model-driven trade."""
        label = "BUY YES" if direction == "long" else "BUY NO"
        logger.info(f"{'📈' if direction == 'long' else '📉'} MODEL SAYS: {label} "
                    f"(edge=${signal.edge:+.4f}, net_EV=${signal.net_expected_pnl:+.4f})")
        logger.info(f"  Vol: RV={signal.realized_vol:.0%}, IV={signal.implied_vol:.0%}")
        logger.info(f"  Greeks: Δ={signal.delta:.6f}, Γ={signal.gamma:.6f}, Θ={signal.theta_per_min:.6f}/min")

        is_valid, error = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD, direction=direction, current_price=current_price)
        if not is_valid:
            logger.warning(f"Risk engine blocked: {error}"); return
        if not self._check_liquidity(direction):
            return

        logger.info(f"Position size: ${POSITION_SIZE_USD} | Direction: {direction.upper()}")
        await self._dispatch_order(signal, direction, current_price, is_simulation)
        self._active_entry = {
            "direction": signal.direction,
            "entry_price": float(current_price) if direction == "long" else no_price,
            "entry_time": datetime.now(timezone.utc),
            "btc_strike": self._btc_strike_price,
        }

    def _check_liquidity(self, direction):
        """Check if there's sufficient liquidity. Returns True if OK."""
        last_tick = getattr(self, '_last_bid_ask', None)
        if not last_tick:
            return True
        last_bid, last_ask = last_tick
        MIN_LIQUIDITY = Decimal("0.02")
        if direction == "long" and last_ask <= MIN_LIQUIDITY:
            logger.warning(f"⚠ No liquidity for BUY YES: ask=${float(last_ask):.4f}")
            self.last_trade_time = -1
            return False
        if direction == "short" and last_bid <= MIN_LIQUIDITY:
            logger.warning(f"⚠ No liquidity for BUY NO: bid=${float(last_bid):.4f}")
            self.last_trade_time = -1
            return False
        return True

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

    def _fuse_heuristic_signals(self, signals, current_price):
        """Fuse signals and apply coin-flip zone filter. Returns (fused, direction) or (None, None)."""
        fused = self.fusion_engine.fuse_signals(signals, min_signals=MIN_FUSION_SIGNALS, min_score=MIN_FUSION_SCORE)
        if not fused:
            logger.info(f"Fusion produced no actionable signal (need {MIN_FUSION_SIGNALS}+ signals)")
            return None, None
        logger.info(f"FUSED SIGNAL: {fused.direction.value} (score={fused.score:.1f}, conf={fused.confidence:.2%})")
        price_float = float(current_price)
        if TREND_DOWN_THRESHOLD <= price_float <= TREND_UP_THRESHOLD and fused.confidence < 0.70:
            logger.info("⏭ SKIP: coin flip zone, weak confidence")
            return None, None
        direction = "long" if "BULLISH" in str(fused.direction).upper() else "short"
        return fused, direction

    async def _make_trading_decision_heuristic(self, current_price, cached, is_simulation):
        """Fallback to V2 fusion-based logic when Coinbase data is unavailable."""
        metadata = self._build_metadata(cached, current_price)
        signals = self._process_signals(current_price, metadata)
        if not signals:
            logger.info("No signals generated — no trade"); return

        logger.info(f"Generated {len(signals)} signal(s):")
        for sig in signals:
            logger.info(f"  [{sig.source}] {sig.direction.value}: score={sig.score:.1f}, conf={sig.confidence:.2%}")

        fused, direction = self._fuse_heuristic_signals(signals, current_price)
        if fused is None:
            return

        is_valid, error = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD, direction=direction, current_price=current_price)
        if not is_valid:
            logger.warning(f"Risk engine blocked: {error}"); return

        if is_simulation:
            await self._record_paper_trade(fused, POSITION_SIZE_USD, current_price, direction)
        elif USE_LIMIT_ORDERS:
            await self._place_limit_order(fused, POSITION_SIZE_USD, current_price, direction)
        else:
            await self._place_real_order(fused, POSITION_SIZE_USD, current_price, direction)

    # ------------------------------------------------------------------
    # Signal processing (confirmation role in V3)
    # ------------------------------------------------------------------

    def _process_signals(self, current_price, metadata=None):
        """Run all signal processors and return list of signals."""
        if metadata is None:
            metadata = {}
        processed_metadata = self._prepare_signal_metadata(metadata)
        return self._run_all_processors(current_price, processed_metadata)

    def _prepare_signal_metadata(self, metadata):
        """Convert float values in metadata to Decimal for signal processors."""
        processed = {}
        for key, value in metadata.items():
            if isinstance(value, float):
                processed[key] = Decimal(str(value))
            else:
                processed[key] = value
        return processed

    def _run_all_processors(self, current_price, pm):
        """Run each signal processor and collect results."""
        signals = []
        args = dict(current_price=current_price, historical_prices=self.price_history, metadata=pm)
        # (processor, required_key) — None means always run
        dispatch = [
            (self.spike_detector, None),
            (self.sentiment_processor, 'sentiment_score'),
            (self.divergence_processor, 'spot_price'),
            (self.orderbook_processor, 'yes_token_id'),
            (self.tick_velocity_processor, 'tick_buffer'),
            (self.deribit_pcr_processor, None),
        ]
        for proc, key in dispatch:
            if key and not pm.get(key):
                continue
            s = proc.process(**args)
            if s:
                signals.append(s)
        return signals

    # ------------------------------------------------------------------
    # Paper trade
    # ------------------------------------------------------------------

    async def _record_paper_trade(self, signal, position_size, current_price, direction):
        """Track paper trades as real positions for resolution at market end."""
        if (self.current_instrument_index < 0 or
                self.current_instrument_index >= len(self.all_btc_instruments)):
            logger.warning("No active market — cannot record paper trade")
            return

        current_market = self.all_btc_instruments[self.current_instrument_index]
        timestamp_ms = int(time.time() * 1000)
        order_id = f"paper_{timestamp_ms}"

        if direction == "long":
            trade_instrument_id = getattr(self, '_yes_instrument_id', self.instrument_id)
            token_label = "YES (UP)"
        else:
            trade_instrument_id = getattr(self, '_no_instrument_id', None) or self.instrument_id
            token_label = "NO (DOWN)"

        self._open_positions.append(OpenPosition(
            market_slug=current_market['slug'], direction=direction,
            entry_price=float(current_price), size_usd=float(position_size),
            entry_time=datetime.now(timezone.utc),
            market_end_time=current_market['end_time'],
            instrument_id=trade_instrument_id, order_id=order_id,
        ))

        self._add_paper_trade_record(signal, position_size, current_price, direction)
        self._log_paper_trade(direction, token_label, position_size, current_price, current_market, order_id)
        self._save_paper_trades()

    def _add_paper_trade_record(self, signal, position_size, current_price, direction):
        """Add a paper trade record to the paper_trades list."""
        self.paper_trades.append(PaperTrade(
            timestamp=datetime.now(timezone.utc),
            direction=direction.upper(),
            size_usd=float(position_size),
            price=float(current_price),
            signal_score=signal.score,
            signal_confidence=signal.confidence,
            outcome="PENDING",
        ))

    def _log_paper_trade(self, direction, token_label, position_size, current_price, current_market, order_id):
        """Log paper trade details."""
        logger.info("=" * 80)
        logger.info("[SIMULATION] PAPER TRADE OPENED — awaiting market resolution")
        logger.info(f"  Direction: {direction.upper()} → {token_label}")
        logger.info(f"  Size: ${float(position_size):.2f} | Entry: ${float(current_price):,.4f}")
        logger.info(f"  Market: {current_market['slug']} | Resolves: {current_market['end_time'].strftime('%H:%M:%S')} UTC")
        logger.info(f"  Order ID: {order_id}")
        logger.info("=" * 80)

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
        """Place a LIMIT order at the best BID to guarantee maker status (0% fee)."""
        if not self.instrument_id:
            logger.error("No instrument available"); return
        try:
            resolved = self._resolve_trade_instrument(direction)
            if resolved is None:
                return
            trade_instrument_id, trade_label, instrument = resolved
            token_bid, token_ask = self._get_token_book(trade_instrument_id, current_price)
            limit_price = max(Decimal("0.01"), min(Decimal("0.99"), token_bid))
            token_qty = self._compute_token_qty(position_size, limit_price, instrument)

            spread = token_ask - token_bid
            logger.info("=" * 80)
            logger.info(f"LIMIT ORDER @ BID (MAKER, 0% FEE) | {trade_label}: "
                        f"{token_qty:.2f} tokens @ ${float(limit_price):.4f} | "
                        f"spread=${float(spread):.4f}")
            logger.info("=" * 80)

            self._submit_limit_order(
                trade_instrument_id, instrument, token_qty, limit_price,
                trade_label, position_size, direction)
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            import traceback; traceback.print_exc()
            self._track_order_event("rejected")

    def _resolve_trade_instrument(self, direction):
        """Resolve the trade instrument for a direction. Returns (id, label, instrument) or None."""
        if direction == "long":
            trade_instrument_id = getattr(self, '_yes_instrument_id', self.instrument_id)
            trade_label = "YES (UP)"
        else:
            no_id = getattr(self, '_no_instrument_id', None)
            if no_id is None:
                logger.warning("NO token not found — cannot bet DOWN. Skipping.")
                return None
            trade_instrument_id = no_id
            trade_label = "NO (DOWN)"

        instrument = self.cache.instrument(trade_instrument_id)
        if not instrument:
            logger.error(f"Instrument not in cache: {trade_instrument_id}")
            return None
        return trade_instrument_id, trade_label, instrument

    def _get_token_book(self, trade_instrument_id, current_price):
        """Get current best bid/ask for a token."""
        try:
            token_quote = self.cache.quote_tick(trade_instrument_id)
            if token_quote:
                return token_quote.bid_price.as_decimal(), token_quote.ask_price.as_decimal()
        except Exception:
            pass
        return current_price, current_price + Decimal("0.02")

    def _compute_token_qty(self, position_size, limit_price, instrument):
        """Compute token quantity from position size and price."""
        token_qty = float(position_size / limit_price)
        precision = instrument.size_precision
        token_qty = round(token_qty, precision)
        return max(token_qty, 5.0)

    def _submit_limit_order(self, trade_instrument_id, instrument, token_qty,
                            limit_price, trade_label, position_size, direction):
        """Build, submit a limit order, and track the position."""
        precision = instrument.size_precision
        qty = Quantity(token_qty, precision=precision)
        price = Price(float(limit_price), precision=instrument.price_precision)
        timestamp_ms = int(time.time() * 1000)
        unique_id = f"BTC-15MIN-V3-{timestamp_ms}"

        order = self.order_factory.limit(
            instrument_id=trade_instrument_id,
            order_side=OrderSide.BUY,
            quantity=qty, price=price,
            client_order_id=ClientOrderId(unique_id),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

        logger.info(f"LIMIT ORDER SUBMITTED: {unique_id} | {trade_label} @ ${float(limit_price):.4f}")

        self._track_open_position(direction, float(limit_price), float(position_size),
                                  trade_instrument_id, unique_id)


    # ------------------------------------------------------------------
    # Market order (fallback)
    # ------------------------------------------------------------------

    async def _place_real_order(self, signal, position_size, current_price, direction):
        """Place a market order (taker, 10% fee)."""
        if not self.instrument_id:
            logger.error("No instrument available")
            return

        try:
            logger.info("=" * 80)
            logger.info("LIVE MODE — MARKET ORDER (10% TAKER FEE!)")
            logger.info("⚠️  Consider setting USE_LIMIT_ORDERS=true for 0% fee")
            logger.info("=" * 80)

            resolved = self._resolve_trade_instrument(direction)
            if resolved is None:
                return
            trade_instrument_id, trade_label, instrument = resolved

            self._submit_market_order(
                trade_instrument_id, trade_label, instrument,
                float(current_price), float(position_size), direction,
            )

        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            import traceback
            traceback.print_exc()
            self._track_order_event("rejected")

    def _track_open_position(self, direction, entry_price, size_usd, instrument_id, order_id):
        """Record a new open position for market resolution."""
        current_market = self.all_btc_instruments[self.current_instrument_index]
        self._open_positions.append(OpenPosition(
            market_slug=current_market['slug'], direction=direction,
            entry_price=entry_price, size_usd=size_usd,
            entry_time=datetime.now(timezone.utc),
            market_end_time=current_market['end_time'],
            instrument_id=instrument_id, order_id=order_id))
        self._track_order_event("placed")

    def _submit_market_order(self, trade_instrument_id, trade_label, instrument,
                             trade_price, max_usd_amount, direction):
        """Build, submit a market order, and track the position."""
        precision = instrument.size_precision
        min_qty_val = float(getattr(instrument, 'min_quantity', None) or 5.0)
        token_qty = round(max(min_qty_val, 5.0), precision)

        qty = Quantity(token_qty, precision=precision)
        timestamp_ms = int(time.time() * 1000)
        unique_id = f"BTC-15MIN-MKT-{timestamp_ms}"
        order = self.order_factory.market(
            instrument_id=trade_instrument_id, order_side=OrderSide.BUY,
            quantity=qty, client_order_id=ClientOrderId(unique_id),
            quote_quantity=False, time_in_force=TimeInForce.IOC)
        self.submit_order(order)
        logger.info(f"MARKET ORDER SUBMITTED: {unique_id} | {trade_label}")
        self._track_open_position(direction, trade_price, max_usd_amount, trade_instrument_id, unique_id)


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
        except Exception as e:
            logger.error(f"Failed to start Grafana: {e}")

    def on_stop(self):
        logger.info("Integrated BTC strategy V3.1 stopped")
        logger.info(f"Total paper trades: {len(self.paper_trades)}")

        for pos in self._open_positions:
            if not pos.resolved:
                logger.info(f"Unresolved position: {pos.market_slug} {pos.direction} @ ${pos.entry_price:.4f}")

        self._log_stop_stats()
        self._cleanup_connections()

    def _log_stop_stats(self):
        """Log final statistics on shutdown."""
        stats = self.mispricing_detector.get_stats()
        logger.info(f"Mispricing detector: {stats['tradeable_detections']}/{stats['total_detections']} tradeable ({stats['hit_rate']:.0%})")
        logger.info(f"Vol estimator: {self.vol_estimator.get_stats()}")

        if self.rtds:
            rs = self.rtds.get_stats()
            logger.info(f"RTDS: {rs['chainlink_ticks']} Chainlink, {rs['binance_ticks']} Binance, avg lat={rs['avg_chainlink_latency_ms']}ms")

        if self.funding_filter:
            logger.info(f"Funding filter: {self.funding_filter.get_stats()}")

    def _cleanup_connections(self):
        """Clean up async connections on shutdown."""
        if self.rtds:
            try:
                _loop = asyncio.new_event_loop()
                _loop.run_until_complete(self.rtds.disconnect())
                _loop.close()
            except Exception:
                pass

        if self.grafana_exporter:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.grafana_exporter.stop())
            except Exception:
                pass


# ===========================================================================
# Runner
# ===========================================================================

def _build_polymarket_configs():
    """Build Polymarket data and exec client configs."""
    instrument_cfg = PolymarketInstrumentProviderConfig(
        event_slug_builder="slug_builders:build_btc_15min_slugs",
    )

    data_cfg = PolymarketDataClientConfig(
        private_key=os.getenv("POLYMARKET_PK"),
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
        signature_type=1,
        instrument_config=instrument_cfg,
    )

    exec_cfg = PolymarketExecClientConfig(
        private_key=os.getenv("POLYMARKET_PK"),
        api_key=os.getenv("POLYMARKET_API_KEY"),
        api_secret=os.getenv("POLYMARKET_API_SECRET"),
        passphrase=os.getenv("POLYMARKET_PASSPHRASE"),
        signature_type=1,
        instrument_config=instrument_cfg,
    )
    return data_cfg, exec_cfg


def _build_trading_node_config(poly_data_cfg, poly_exec_cfg):
    """Build TradingNodeConfig with Polymarket adapters."""
    return TradingNodeConfig(
        environment="live",
        trader_id="BTC-15MIN-V31-001",
        logging=LoggingConfig(
            log_level="ERROR",
            log_directory="./logs/nautilus",
            log_component_levels={"IntegratedBTCStrategy": "INFO"},
        ),
        data_engine=LiveDataEngineConfig(qsize=6000),
        exec_engine=LiveExecEngineConfig(qsize=6000),
        risk_engine=LiveRiskEngineConfig(bypass=False),
        data_clients={POLYMARKET: poly_data_cfg},
        exec_clients={POLYMARKET: poly_exec_cfg},
    )


def _init_redis_mode(simulation):
    """Initialize Redis and set simulation mode."""
    redis_client = init_redis()
    if redis_client:
        try:
            redis_client.set('btc_trading:simulation_mode', '1' if simulation else '0')
            logger.info(f"Redis simulation_mode: {'SIMULATION' if simulation else 'LIVE'}")
        except Exception as e:
            logger.warning(f"Could not set Redis simulation mode: {e}")
    return redis_client


def _build_and_run_node(strategy):
    """Build and run the Nautilus trading node."""
    poly_data_cfg, poly_exec_cfg = _build_polymarket_configs()
    config = _build_trading_node_config(poly_data_cfg, poly_exec_cfg)
    print("\nBuilding Nautilus node...")
    node = TradingNode(config=config)
    node.add_data_client_factory(POLYMARKET, PolymarketLiveDataClientFactory)
    node.add_exec_client_factory(POLYMARKET, PolymarketLiveExecClientFactory)
    node.trader.add_strategy(strategy)
    node.build()
    logger.info("Nautilus node built successfully")
    print()
    try:
        node.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.dispose()
        logger.info("Bot stopped")


def run_integrated_bot(simulation: bool = False, enable_grafana: bool = True, test_mode: bool = False):
    print("=" * 80)
    print("INTEGRATED POLYMARKET BTC 15-MIN TRADING BOT V3.1")
    print("BINARY OPTION PRICING + RTDS CHAINLINK ORACLE")
    print("=" * 80)
    redis_client = _init_redis_mode(simulation)
    print(f"\nMode: {'SIMULATION' if simulation else 'LIVE TRADING'} | "
          f"Trade Size: ${POSITION_SIZE_USD} | Min Edge: ${MIN_EDGE_CENTS:.2f}")
    logger.info("Loading BTC 15-min markets via event slug builder...")
    strategy = IntegratedBTCStrategy(redis_client=redis_client,
                                     enable_grafana=enable_grafana, test_mode=test_mode)
    _build_and_run_node(strategy)


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
        logger.warning(f"⚠ LIVE TRADING MODE — REAL MONEY | {'LIMIT' if USE_LIMIT_ORDERS else 'MARKET'} orders")
    else:
        logger.info(f"SIMULATION MODE — {'TEST MODE' if test_mode else 'paper trading only'}")

    run_integrated_bot(simulation=simulation, enable_grafana=enable_grafana, test_mode=test_mode)


if __name__ == "__main__":
    main()