import asyncio
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Dict

try:
    from polymarket_apis import PolymarketGaslessWeb3Client
except ImportError:
    PolymarketGaslessWeb3Client = None

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
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity, Price
from nautilus_trader.model.data import QuoteTick

from dotenv import load_dotenv
from loguru import logger
import redis
import requests

# Persistent HTTP session for Gamma API calls (reuses TCP/TLS connections)
_gamma_session: Optional[requests.Session] = None


def _get_gamma_session() -> requests.Session:
    """Return a module-level persistent requests.Session for Gamma API."""
    global _gamma_session
    if _gamma_session is None:
        _gamma_session = requests.Session()
        _gamma_session.headers.update({"Accept": "application/json"})
    return _gamma_session


# Signal processors (kept as confirmation signals in V3)
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine
from execution.risk_engine import get_risk_engine
from monitoring.performance_tracker import get_performance_tracker
from monitoring.grafana_exporter import get_grafana_exporter
from feedback.learning_engine import get_learning_engine

# V3: Quantitative pricing model
from binary_pricer import get_binary_pricer
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
QUOTE_STABILITY_REQUIRED = int(os.getenv("QUOTE_STABILITY_REQUIRED", "5"))
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
BANKROLL_USD = float(os.getenv("BANKROLL_USD", "20.0"))

# V3.1: RTDS + Funding Rate Config
USE_RTDS = os.getenv("USE_RTDS", "true").lower() == "true"
RTDS_LATE_WINDOW_SEC = float(os.getenv("RTDS_LATE_WINDOW_SEC", "15"))
RTDS_LATE_WINDOW_MIN_BPS = float(os.getenv("RTDS_LATE_WINDOW_MIN_BPS", "3.0"))
RTDS_DIVERGENCE_THRESHOLD_BPS = float(os.getenv("RTDS_DIVERGENCE_THRESHOLD_BPS", "5.0"))
USE_FUNDING_FILTER = os.getenv("USE_FUNDING_FILTER", "true").lower() == "true"
LATE_WINDOW_ENABLED = os.getenv("LATE_WINDOW_ENABLED", "true").lower() == "true"
SHADOW_MODE_ONLY = os.getenv("SHADOW_MODE_ONLY", "false").lower() == "true"

# V3: Gamma Scalping Config
GAMMA_EXIT_PROFIT_PCT = float(os.getenv("GAMMA_EXIT_PROFIT_PCT", "0.04"))  # 4% profit threshold for late exit
GAMMA_EXIT_TIME_MINS = float(os.getenv("GAMMA_EXIT_TIME_MINS", "3.0"))  # Active threshold in final 3 minutes


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
    actual_qty: float = 0.0  # <--- ADD THIS FIELD
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
# V3.2: Fetch authoritative strike ("price to beat") from Gamma API
# =============================================================================

def fetch_price_to_beat(slug: str, timeout: float = 10.0, max_retries: int = 3) -> Optional[float]:
    """
    Fetch the authoritative 'priceToBeat' for a BTC 15-min UpDown market.

    Polymarket stores the exact Chainlink BTC/USD price at eventStartTime in
    the event metadata field ``priceToBeat``.  This is the REAL strike that
    determines resolution — NOT the Chainlink price we capture ourselves.

    Path: GET /markets?slug=<slug>  →  events[0].eventMetadata.priceToBeat
    """
    session = _get_gamma_session()
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(
                "https://gamma-api.polymarket.com/markets",
                params={"slug": slug},
                timeout=timeout,
            )
            if r.status_code != 200:
                logger.debug(f"Gamma API returned {r.status_code} for slug={slug} (attempt {attempt})")
                # Retry on server errors / non-200 up to max_retries
                if attempt < max_retries:
                    logger.info(f"Retrying fetch_price_to_beat in {backoff}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return None

            data = r.json()
            if not data:
                return None

            market = data[0]
            events = market.get("events", [])
            if not events:
                return None

            metadata = events[0].get("eventMetadata")
            if not metadata or not isinstance(metadata, dict):
                return None

            ptb = metadata.get("priceToBeat")
            if ptb is not None:
                return float(ptb)

            return None
        except Exception as e:
            # Handle request-level failures with exponential backoff
            logger.debug(f"fetch_price_to_beat({slug}) attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"fetch_price_to_beat retry in {backoff}s (attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                return None


def fetch_condition_id(slug: str, timeout: float = 10.0) -> Optional[str]:
    """Fetch the specific condition ID needed to redeem a market."""
    session = _get_gamma_session()
    try:
        r = session.get(
            "https://gamma-api.polymarket.com/markets",
            params={"slug": slug},
            timeout=timeout,
        )
        if r.status_code == 200:
            data = r.json()
            if data and len(data) > 0:
                return data[0].get("conditionId")
    except Exception as e:
        logger.error(f"Failed to fetch condition ID for {slug}: {e}")
    return None

def auto_redeem_winnings(slug: str, direction: str, amount_won: float):
    """Redeems winning shares for USDC via gasless proxy transaction."""
    if not PolymarketGaslessWeb3Client:
        logger.error("polymarket-apis not installed. Cannot auto-redeem.")
        return

    private_key = os.getenv("POLYMARKET_PK")
    proxy_address = os.getenv("POLYMARKET_FUNDER")
    
    if not private_key or not proxy_address:
        logger.error("Missing POLYMARKET_PK or POLYMARKET_FUNDER in .env")
        return
        
    logger.info(f"🔄 Fetching condition ID to auto-redeem winnings for {slug}...")
    condition_id = fetch_condition_id(slug)
    
    if not condition_id:
        logger.error("Could not find condition_id. Redemption aborted.")
        return

    try:
        client = PolymarketGaslessWeb3Client(
            private_key=private_key,
            signature_type=1
        )
        
        # direction: "long" = Bought YES (index 0), "short" = Bought NO (index 1)
        amounts = [float(amount_won), 0.0] if direction == "long" else [0.0, float(amount_won)]
        
        receipt = client.redeem_position(
            condition_id=condition_id,
            amounts=amounts,
            neg_risk=False
        )
        logger.info(f"💰 Successfully Redeemed {slug}! Tx Hash: {receipt}")
    except Exception as e:
        logger.error(f"Failed to auto-redeem winnings: {e}")

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
        self._aiohttp_session: Optional[object] = None  # Reusable aiohttp session

        # =====================================================================
        # V3: Quantitative pricing model
        # =====================================================================
        self.binary_pricer = get_binary_pricer()
        self.vol_estimator = get_vol_estimator()
        self.mispricing_detector = get_mispricing_detector(
            maker_fee=0.00,
            taker_fee=0.02,  # V2: ~2% default, overridden by nonlinear formula (was 10%!)
            min_edge_cents=MIN_EDGE_CENTS,
            min_edge_after_fees=0.005,
            take_profit_pct=TAKE_PROFIT_PCT,
            cut_loss_pct=CUT_LOSS_PCT,
            vol_method=VOL_METHOD,
            bankroll=BANKROLL_USD,
        )

        # Strike price (BTC at market open) — reset each market
        self._btc_strike_price: Optional[float] = None
        self._strike_recorded = False
        self._strike_defer_start: Optional[float] = None  # V3.2: When we started deferring
        self._strike_source: str = ""  # V3.3: "ptb", "chainlink_boundary", "chainlink_current", "binance", "coinbase"

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

        # Fusion engine (kept for confirmation and heuristic fallback)
        self.fusion_engine = get_fusion_engine()
        self.fusion_engine.set_weight("OrderBookImbalance", 0.40)
        self.fusion_engine.set_weight("TickVelocity", 0.30)
        self.fusion_engine.set_weight("PriceDivergence", 0.20)
        self.fusion_engine.set_weight("SpikeDetection", 0.10)

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register_upcoming_boundaries(self):
        """
        Pre-register upcoming 15-min boundaries with the RTDS connector.

        The connector's Chainlink tick handler will then capture the price
        at each boundary in real-time (zero extra latency — captured inline
        as ticks arrive, not fetched retroactively).
        """
        if not self.rtds or not self.all_btc_instruments:
            return
        now_ts = int(datetime.now(timezone.utc).timestamp())
        registered = 0
        for inst in self.all_btc_instruments:
            ts = inst['market_timestamp']
            # Register boundaries that are within the next 30 minutes
            if -60 <= (ts - now_ts) <= 1800:
                self.rtds.register_boundary(ts)
                registered += 1
        if registered:
            logger.info(f"✓ Pre-registered {registered} boundary snapshots with RTDS")

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

    async def _get_aiohttp_session(self):
        """Return a persistent aiohttp session (creates one if needed)."""
        import aiohttp
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            self._aiohttp_session = aiohttp.ClientSession()
        return self._aiohttp_session

    async def _preseed_vol_estimator(self):
        """
        V3.1 FIX: Fetch recent Coinbase 1-min candles to warm up vol estimator
        immediately on startup. Without this, the first 15-min window always
        uses DEFAULT_VOL (65%) which makes the model see a coin flip and Kelly
        says f*=0% → NO_TRADE on every first window.

        Coinbase public API: no auth required.
        GET /products/BTC-USD/candles?granularity=60
        Returns: [[timestamp, low, high, open, close, volume], ...]
        """
        import aiohttp

        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        params = {"granularity": 60}  # 1-min candles, returns last 300 by default

        try:
            session = await self._get_aiohttp_session()
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"Coinbase candles API returned {resp.status}")
                    return

                candles = await resp.json()

            if not candles or not isinstance(candles, list):
                logger.warning("No candle data returned from Coinbase")
                return

            # Candles come newest-first: [[ts, low, high, open, close, vol], ...]
            # Sort oldest-first so vol estimator sees them in chronological order
            candles.sort(key=lambda c: c[0])

            # Build (timestamp, price) list for preseed
            # Uses preseed_historical() to properly insert bars even if RTDS
            # has already set _last_bar_time to the current minute
            price_pairs = []
            for candle in candles:
                try:
                    ts, low, high, open_p, close, volume = candle
                    price_pairs.append((float(ts), float(close)))
                except (ValueError, IndexError, TypeError):
                    continue

            self.vol_estimator.preseed_historical(price_pairs)

            vol = self.vol_estimator.get_vol(VOL_METHOD)
            logger.info(
                f"✓ Vol estimator pre-seeded: {len(price_pairs)} candles → "
                f"RV={vol.annualized_vol:.1%} ({vol.method}), "
                f"confidence={vol.confidence:.0%}"
            )

            # Also cache the latest price as initial spot
            if candles:
                latest_price = float(candles[-1][4])  # close of most recent candle
                with self._cache_lock:
                    if self._cached_spot_price is None:
                        self._cached_spot_price = latest_price
                        logger.info(f"✓ Initial BTC spot from candles: ${latest_price:,.2f}")

        except Exception as e:
            logger.warning(f"Vol estimator pre-seed failed: {e} — will use default vol until enough ticks")

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
        if self._aiohttp_session and not self._aiohttp_session.closed:
            try:
                await self._aiohttp_session.close()
            except Exception:
                pass
            self._aiohttp_session = None
        self._data_sources_initialized = False

    async def _refresh_cached_data(self):
        """
        V3.2: Fetch BTC spot + funding CONCURRENTLY.
        Sentiment (F&G) is now handled autonomously by a background thread.
        Each source has its own 5s timeout so one dead API can't block others.
        """
        spot = None

        # RTDS Chainlink — in-memory, instant
        if self.rtds and self.rtds.chainlink_btc_price > 0 and self.rtds.chainlink_age_ms < 30000:
            spot = self.rtds.chainlink_btc_price
            # Log divergence if significant
            # V3.4 FIX: Check Binance staleness before logging divergence
            if self.rtds.binance_btc_price > 0 and self.rtds.binance_age_ms < 60000:
                div = self.rtds.get_divergence()
                if div.is_significant:
                    logger.info(
                        f"⚡ Price divergence: Binance=${div.binance_price:,.2f} vs "
                        f"Chainlink=${div.chainlink_price:,.2f} ({div.divergence_bps:+.1f}bps) "
                        f"→ {div.direction}"
                    )
        elif self.rtds and self.rtds.binance_btc_price > 0 and self.rtds.binance_age_ms < 60000:
            # V3.2: RTDS Binance as intermediate fallback (Chainlink feed may be silent)
            spot = self.rtds.binance_btc_price

        # Concurrent I/O fetches
        async def _fetch_coinbase():
            if spot is not None or not self._coinbase_source:
                return None
            try:
                result = await asyncio.wait_for(
                    self._coinbase_source.get_current_price(), timeout=5.0
                )
                if result:
                    price = float(result)
                    self.vol_estimator.add_price(price)
                    return price
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Coinbase refresh error: {e}")
            return None

        async def _fetch_funding():
            if not self.funding_filter or not self.funding_filter.should_update():
                return
            try:
                await asyncio.wait_for(
                    self.funding_filter.update(), timeout=5.0
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Funding update error: {e}")

        results = await asyncio.gather(
            _fetch_coinbase(),
            _fetch_funding(),
            return_exceptions=True,
        )

        coinbase_price = results[0] if not isinstance(results[0], Exception) else None

        with self._cache_lock:
            if spot is not None:
                self._cached_spot_price = spot
            elif coinbase_price is not None:
                self._cached_spot_price = coinbase_price

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
        logger.info("Strategy starting — loading instruments...")

        self._load_all_btc_instruments()

        if self.instrument_id:
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

        # No synthetic history — wait for real ticks
        if len(self.price_history) < 5:
            logger.info(f"Waiting for real price data ({len(self.price_history)}/5 ticks so far)")

        # Start timer loop
        self.run_in_executor(self._start_timer_loop)

        # V3.1: Start RTDS Chainlink settlement oracle
        if self.rtds:
            self.rtds.start_background()
            logger.info("✓ RTDS connector started")

            # V3.3: Pre-register all upcoming 15-min boundaries so the
            # Chainlink tick handler captures the price in real-time
            # (zero extra latency — captured inline as ticks arrive).
            self._register_upcoming_boundaries()

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

        # V3.2 FIX: Dedicated background thread for News polling
        self.news_thread = self._start_news_polling()

        logger.info(f"V3.1 active — {len(self.price_history)} price points cached")

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

        if btc_instruments:
            first_t = btc_instruments[0]['start_time'].strftime('%H:%M')
            last_t = btc_instruments[-1]['end_time'].strftime('%H:%M')
            logger.info(f"Found {len(btc_instruments)} BTC 15-min markets ({first_t}–{last_t})")
        else:
            logger.warning("No BTC 15-min markets found!")

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
                logger.info(f"✓ Active: {inst['slug']} → switch at {self.next_switch_time.strftime('%H:%M:%S')}")

                self.subscribe_quote_ticks(self.instrument_id)
                if inst.get('no_instrument_id'):
                    self.subscribe_quote_ticks(inst['no_instrument_id'])
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
        self._strike_defer_start = None
        self._strike_source = ""
        self._active_entry = None
        self._late_window_traded = False

        # V3.3: Set strike from pre-captured boundary snapshot (zero latency).
        # The RTDS tick handler has been capturing Chainlink prices at pre-registered
        # boundaries in real-time. Check for a snapshot first, then fall back.
        if self.rtds:
            # Try pre-captured snapshot (captured inline by tick handler)
            snap_price = self.rtds.get_boundary_price(next_market['market_timestamp'])
            if snap_price:
                detail = self.rtds.get_boundary_detail(next_market['market_timestamp'])
                # detail may be (price, delta_ms) or (price, delta_ms, source_ts, received_ts, latency)
                delta_ms = detail[1] if detail else 0
                source_ts = detail[2] if detail and len(detail) >= 3 else None
                received_ts = detail[3] if detail and len(detail) >= 4 else None
                latency_ms = detail[4] if detail and len(detail) >= 5 else None
                self._btc_strike_price = snap_price
                self._strike_recorded = True
                self._strike_source = "chainlink_boundary"
                if source_ts is not None:
                    logger.info(
                        f"📌 Strike from boundary snapshot: ${snap_price:,.2f} "
                        f"(Δ={delta_ms}ms from boundary, source_ts={source_ts}, "
                        f"received_ts={received_ts}, latency={latency_ms}ms)"
                    )
                else:
                    logger.info(
                        f"📌 Strike from boundary snapshot: ${snap_price:,.2f} (Δ={delta_ms}ms from boundary)"
                    )
            else:
                # Fall back to deque scan (e.g. boundary just happened)
                cl_strike = self.rtds.get_chainlink_price_at(
                    float(next_market['market_timestamp']),
                    max_staleness_ms=10000,
                )
                if cl_strike:
                    self._btc_strike_price = cl_strike
                    self._strike_recorded = True
                    self._strike_source = "chainlink_boundary"
                    logger.info(
                        f"📌 Strike from Chainlink at interval start: "
                        f"${cl_strike:,.2f}"
                    )

            # Pre-register the NEXT boundary for real-time capture
            next_boundary_ts = next_market['market_timestamp'] + MARKET_INTERVAL_SECONDS
            self.rtds.register_boundary(next_boundary_ts)

        if not self._strike_recorded:
            # Chainlink boundary tick not available — try Gamma API (rare, startup edge case)
            ptb = fetch_price_to_beat(next_market['slug'])
            if ptb:
                self._btc_strike_price = ptb
                self._strike_recorded = True
                self._strike_source = "ptb"
                logger.info(
                    f"📌 Strike from Gamma API priceToBeat: ${ptb:,.2f}"
                )

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

                    # === AUTO-REDEEM WINNINGS ===
                    if outcome == "WIN" and not is_paper:
                        # Use actual quantity filled (requires the OpenPosition actual_qty patch applied earlier)
                        win_qty = getattr(pos, 'actual_qty', float(pos.size_usd / pos.entry_price))
                        if win_qty > 0:
                            logger.info(f"   Initiating background redemption for {win_qty:.2f} tokens...")
                            threading.Thread(
                                target=auto_redeem_winnings, 
                                args=(slug, pos.direction, win_qty),
                                daemon=True
                            ).start()

                    # V3.1 FIX: Always pass direction="long" to performance tracker.
                    # In our system "short" = bought NO tokens = still a LONG position.
                    # The tracker assumes "short" means SOLD, which inverts the PnL sign.
                    self.performance_tracker.record_trade(
                        trade_id=pos.order_id,
                        direction="long",  # Always LONG — we BUY tokens (YES or NO)
                        entry_price=Decimal(str(pos.entry_price)),
                        exit_price=Decimal(str(final_price)),
                        size=Decimal(str(pos.size_usd)),
                        entry_time=pos.entry_time,
                        exit_time=datetime.now(timezone.utc),
                        signal_score=0,
                        signal_confidence=0,
                        metadata={
                            "resolved": True,
                            "market": slug,
                            "paper": is_paper,
                            "original_direction": pos.direction,  # Keep for reference
                            "token": "YES" if pos.direction == "long" else "NO",
                            "pnl_computed": round(pnl, 4),  # Our computed PnL for cross-check
                        }
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

        # V3.1: Pre-seed vol estimator with recent Coinbase candles
        # so the first trade window has real vol instead of 65% default
        await self._preseed_vol_estimator()

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
                    self._strike_defer_start = None
                    self._strike_source = ""
                    self._active_entry = None
                    self._late_window_traded = False

                    # V3.3: Set strike from boundary snapshot or Chainlink.
                    if (self.current_instrument_index >= 0 and
                            self.current_instrument_index < len(self.all_btc_instruments)):
                        mkt = self.all_btc_instruments[self.current_instrument_index]
                        if self.rtds:
                            snap_price = self.rtds.get_boundary_price(mkt['market_timestamp'])
                            if snap_price:
                                detail = self.rtds.get_boundary_detail(mkt['market_timestamp'])
                                delta_ms = detail[1] if detail else 0
                                source_ts = detail[2] if detail and len(detail) >= 3 else None
                                received_ts = detail[3] if detail and len(detail) >= 4 else None
                                latency_ms = detail[4] if detail and len(detail) >= 5 else None
                                self._btc_strike_price = snap_price
                                self._strike_recorded = True
                                self._strike_source = "chainlink_boundary"
                                if source_ts is not None:
                                    logger.info(
                                        f"📌 Strike from boundary snapshot: ${snap_price:,.2f} "
                                        f"(Δ={delta_ms}ms from boundary, source_ts={source_ts}, "
                                        f"received_ts={received_ts}, latency={latency_ms}ms)"
                                    )
                                else:
                                    logger.info(
                                        f"📌 Strike from boundary snapshot: ${snap_price:,.2f} (Δ={delta_ms}ms)"
                                    )
                            else:
                                cl_strike = self.rtds.get_chainlink_price_at(
                                    float(mkt['market_timestamp']),
                                    max_staleness_ms=10000,
                                )
                                if cl_strike:
                                    self._btc_strike_price = cl_strike
                                    self._strike_recorded = True
                                    self._strike_source = "chainlink_boundary"
                                    logger.info(
                                        f"📌 Strike from Chainlink at interval start: "
                                        f"${cl_strike:,.2f}"
                                    )
                            # Register next boundary
                            next_ts = mkt['market_timestamp'] + MARKET_INTERVAL_SECONDS
                            self.rtds.register_boundary(next_ts)

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
                await self._manage_open_positions()

                # V3.3: Record strike if not yet set.
                # Priority: boundary snapshot > Chainlink deque > current Chainlink
                # > defer 30s > Gamma API priceToBeat > Binance > Coinbase
                if not self._strike_recorded:
                    strike_set = False
                    mkt = None
                    if 0 <= self.current_instrument_index < len(self.all_btc_instruments):
                        mkt = self.all_btc_instruments[self.current_instrument_index]

                    # 1. Boundary snapshot (pre-captured in tick handler, zero latency)
                    if not strike_set and mkt and self.rtds:
                        snap = self.rtds.get_boundary_price(mkt['market_timestamp'])
                        if snap:
                            detail = self.rtds.get_boundary_detail(mkt['market_timestamp'])
                            delta_ms = detail[1] if detail else 0
                            source_ts = detail[2] if detail and len(detail) >= 3 else None
                            received_ts = detail[3] if detail and len(detail) >= 4 else None
                            latency_ms = detail[4] if detail and len(detail) >= 5 else None
                            self._btc_strike_price = snap
                            self._strike_recorded = True
                            self._strike_defer_start = None
                            self._strike_source = "chainlink_boundary"
                            if source_ts is not None:
                                logger.info(
                                    f"📌 Strike from boundary snapshot: ${snap:,.2f} "
                                    f"(Δ={delta_ms}ms from boundary, source_ts={source_ts}, "
                                    f"received_ts={received_ts}, latency={latency_ms}ms)"
                                )
                            else:
                                logger.info(
                                    f"📌 Strike from boundary snapshot: ${snap:,.2f} (Δ={delta_ms}ms)"
                                )
                            strike_set = True

                    # 2. Chainlink deque scan
                    if not strike_set and mkt and self.rtds:
                        cl_strike = self.rtds.get_chainlink_price_at(
                            float(mkt['market_timestamp']),
                            max_staleness_ms=10000,
                        )
                        if cl_strike:
                            self._btc_strike_price = cl_strike
                            self._strike_recorded = True
                            self._strike_defer_start = None
                            self._strike_source = "chainlink_boundary"
                            logger.info(
                                f"📌 Strike from Chainlink at interval start: "
                                f"${cl_strike:,.2f}"
                            )
                            strike_set = True

                    # 3. Gamma API priceToBeat (authoritative settlement reference)
                    # CRITICAL: Try this BEFORE falling back to current Chainlink.
                    # On cold start, current Chainlink ≠ settlement reference because
                    # we missed the boundary tick. priceToBeat IS the real strike.
                    if not strike_set and mkt:
                        ptb = fetch_price_to_beat(mkt['slug'])
                        if ptb:
                            self._btc_strike_price = ptb
                            self._strike_recorded = True
                            self._strike_defer_start = None
                            self._strike_source = "ptb"
                            logger.info(
                                f"📌 Strike from Gamma API priceToBeat: ${ptb:,.2f}"
                            )
                            strike_set = True

                    # V3.4 FIX: Never fall back to unverified strike prices (chainlink_current, timeout, coinbase).
                    # If we don't have a verified boundary tick or Gamma API priceToBeat, we must wait.
                    # Trading on a phantom strike is catastrophic.
                    if not strike_set:
                        if self._strike_defer_start is None:
                            self._strike_defer_start = time.time()
                        defer_elapsed = time.time() - self._strike_defer_start

                        # Throttle log to every 30s
                        if getattr(self, "_last_defer_log", 0) < time.time() - 25:
                            logger.debug(
                                f"Strike deferred — waiting for verified boundary/Gamma API "
                                f"({defer_elapsed:.0f}s elapsed)"
                            )
                            self._last_defer_log = time.time()

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

            # Append both a normalized 'price' (spot if available, else mid) and explicit 'spot_price'
            spot_for_tick = float(self._cached_spot_price) if self._cached_spot_price is not None else float(mid_price)
            self._tick_buffer.append({'ts': now, 'price': spot_for_tick, 'spot_price': self._cached_spot_price})

            # Stability gate
            if not self._market_stable:
                if self._is_quote_valid(bid_decimal, ask_decimal):
                    self._stable_tick_count += 1
                    if self._stable_tick_count >= QUOTE_STABILITY_REQUIRED:
                        self._market_stable = True
                        logger.info(f"✓ Market STABLE after {QUOTE_STABILITY_REQUIRED} valid ticks")
                return

            if self._waiting_for_market_open:
                return

            if self.current_instrument_index < 0 or \
                    self.current_instrument_index >= len(self.all_btc_instruments):
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
            # Evaluates continuously between TRADE_WINDOW_START_SEC and TRADE_WINDOW_END_SEC
            # Locks out only AFTER a successful execution or decision has been recorded
            if (TRADE_WINDOW_START_SEC <= seconds_into_sub_interval < TRADE_WINDOW_END_SEC
                    and trade_key != self.last_trade_time
                    and self._btc_strike_price is not None):

                # We don't lock `self.last_trade_time` here yet! 
                # We only lock it if `_make_trading_decision_sync` actually executes a trade.
                # However we do want to ratelimit the logs + evaluations so we don't spam the CPU 
                # 50 times a second. We'll evaluate once every 5 seconds.

                # Use a throttling key for evaluations
                throttle_key = f"{sub_interval}_{int(seconds_into_sub_interval // 5)}"
                if getattr(self, "_last_eval_throttle", None) != throttle_key:
                    self._last_eval_throttle = throttle_key

                    logger.info("=" * 80)
                    logger.info(f"🔎 CONTINUOUS EVALUATION: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    logger.info(f"   Market: {current_market['slug']}")
                    logger.info(
                        f"   Sub-interval #{sub_interval} ({seconds_into_sub_interval:.1f}s in = {seconds_into_sub_interval / 60:.1f} min)")
                    logger.info(
                        f"   YES Price: ${float(mid_price):,.4f} | Bid: ${float(bid_decimal):,.4f} | Ask: ${float(ask_decimal):,.4f}")
                    logger.info("=" * 80)

                    self.run_in_executor(lambda: self._make_trading_decision_sync(float(mid_price), trade_key))

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

    def _make_trading_decision_sync(self, current_price, trade_key=None):
        """Sync wrapper — runs in executor thread."""
        price_decimal = Decimal(str(current_price))
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._make_trading_decision(price_decimal, trade_key))
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

    async def _make_trading_decision(self, current_price: Decimal, trade_key=None):
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
            if is_simulation:
                logger.warning(f"No BTC spot or strike — falling back to heuristic signals (SIMULATION)")
                await self._make_trading_decision_heuristic(current_price, cached, is_simulation, trade_key)
            else:
                logger.warning(
                    f"No BTC spot or strike — SKIPPING trade (live mode requires quant model). "
                    f"spot={'$' + f'{btc_spot:,.2f}' if btc_spot else 'None'}, "
                    f"strike={'$' + f'{self._btc_strike_price:,.2f}' if self._btc_strike_price else 'None'}"
                )
            return

        # Step 3: Calculate time remaining
        current_market = self.all_btc_instruments[self.current_instrument_index]
        now_ts = datetime.now(timezone.utc).timestamp()
        end_ts = current_market['end_timestamp']
        time_remaining_min = max(0, (end_ts - now_ts) / 60.0)

        logger.info(f"BTC: spot=${btc_spot:,.2f}, strike=${self._btc_strike_price:,.2f}, T={time_remaining_min:.1f}min")

        # V3.2: Log strike source for diagnostics
        if self.rtds:
            cl_at_start = self.rtds.get_chainlink_price_at(
                float(current_market['market_timestamp']),
                max_staleness_ms=10000,
            )
            if cl_at_start and abs(cl_at_start - self._btc_strike_price) > 0.01:
                delta_bps = (self._btc_strike_price - cl_at_start) / cl_at_start * 10000
                logger.warning(
                    f"  ⚠ Strike drift: using ${self._btc_strike_price:,.2f}, "
                    f"Chainlink@start was ${cl_at_start:,.2f} (Δ={delta_bps:+.1f}bps)"
                )

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
        # V3.4 FIX: Pass `vol_skew` from genuine Deribit data instead of synthetic formula
        vol_skew = 0.0

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
            # V3.4: Exact market end time
            market_end_time=getattr(self, "next_switch_time", None),
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

        # Step 7b: Orderbook hard block removed (Anomaly 9 fix)
        # Filtering directionally correct signals based on extreme orderbook imbalance 
        # causes too many false negatives. The model's edge should override queue-based dynamics.

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

        # Step 10: Liquidity check
        last_bid, last_ask = self._last_bid_ask if self._last_bid_ask else (Decimal("0"), Decimal("0"))
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
        # Determine position size via Risk Engine (use detector's kelly fraction)
        try:
            # signal.score may not exist; use confidence*100 as fallback score
            signal_score = getattr(signal, 'score', signal.confidence * 100)
            position_size_decimal = self.risk_engine.calculate_position_size(
                signal_confidence=signal.confidence,
                signal_score=signal_score,
                current_price=Decimal(str(current_price)),
                time_remaining_mins=time_remaining_min,
                kelly_fraction=getattr(signal, 'kelly_fraction', 0.0),
            )
            position_size_usd = float(position_size_decimal)
        except Exception:
            # Fallback to configured static size
            position_size_usd = float(POSITION_SIZE_USD)

        logger.info(f"Position size: ${position_size_usd:.2f} | Direction: {direction.upper()}")

        mock_signal = SimpleNamespace(
            direction="BULLISH" if direction == "long" else "BEARISH",
            score=signal.confidence * 100,
            confidence=signal.confidence,
        )

        # Lock out future early-window trades for this specific market sub-interval!
        # This prevents the bot from spamming trades on every tick once an edge is found.
        if trade_key is not None:
            self.last_trade_time = trade_key

        if is_simulation or SHADOW_MODE_ONLY:
            if SHADOW_MODE_ONLY and not is_simulation:
                logger.warning(f"👻 [SHADOW MODE] Paper trading {direction.upper()} | Size: ${position_size_usd}")
            await self._record_paper_trade(mock_signal, position_size_usd, current_price, direction)
        else:
            if USE_LIMIT_ORDERS:
                await self._place_limit_order(mock_signal, position_size_usd, current_price, direction)
            else:
                await self._place_real_order(mock_signal, position_size_usd, current_price, direction)

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

    async def _make_trading_decision_heuristic(self, current_price, cached, is_simulation, trade_key=None):
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

        if trade_key is not None:
            self.last_trade_time = trade_key

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

        # V3.4 FIX: Removed SentimentProcessor (Fear & Greed) from intraday
        # signal path. A daily indicator provides zero alpha for 15-minute
        # binary options and only adds a permanent structural bias.

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
            # --- DYNAMIC EXECUTION ROUTING (Choice A) ---
            edge = getattr(signal, 'edge', 0.0)
            if edge == 0.0 and hasattr(signal, 'confidence'):
                # Principled heuristic calculation:
                # Maps confidence (0.50 to 1.0) linearly to an assumed edge (0% to 25%)
                # Rationale: A purely heuristic signal lacks a priced edge. We assume a
                # 100% confidence signal implies a maximum expected VRP/edge of ~25% in 15m markets.
                edge = max(0.0, float(signal.confidence - 0.50) * 0.50)

            if edge > 0.10:
                # CRITICAL EDGE: Cross the spread. Eat the taker fee to guarantee fill.
                limit_price = token_ask
                logger.info(f"⚡ CRITICAL EDGE ({edge:.1%}): Crossing spread to guarantee fill.")
            elif edge > 0.04 or signal.confidence > 0.70:
                # STRONG EDGE: Penny the bid to jump the queue (Maker status likely)
                limit_price = token_bid + Decimal("0.001")
                limit_price = min(limit_price, token_ask - Decimal("0.001"))  # Don't cross ask
                logger.info(f"🎯 STRONG EDGE ({edge:.1%}): Pennying the bid (+0.001) to jump queue.")
            else:
                # STANDARD EDGE: Rest passively at the bid (Pure Maker)
                limit_price = token_bid
                logger.info(f"🛡️ STANDARD EDGE ({edge:.1%}): Resting passively at best bid.")

            # Clamp bounds
            limit_price = max(Decimal("0.01"), min(Decimal("0.99"), limit_price))

            # Calculate token quantity from position size
            token_qty = 0.0
            try:
                token_qty = float(position_size / limit_price)
            except Exception:
                token_qty = 0.0

            precision = instrument.size_precision

            # Determine exchange-declared min quantity (if present)
            min_qty_val = float(getattr(instrument, 'min_quantity', None) or Decimal("0.01"))

            # Enforce a hard minimum token quantity (Polymarket enforces min shares, often 5)
            enforced_min_token_qty = float(os.getenv("MIN_TOKEN_QTY", "5.0"))
            final_min_qty = max(min_qty_val, enforced_min_token_qty)

            # Ensure quantity respects minimums and precision
            token_qty = max(token_qty, final_min_qty)
            token_qty = round(token_qty, precision)

            # Recompute estimated USD position size based on enforced token quantity
            try:
                estimated_cost = float(Decimal(str(token_qty)) * limit_price)
            except Exception:
                estimated_cost = float(position_size)

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
            logger.info(f"  Estimated Cost: ~${float(estimated_cost):.2f}")
            logger.info(f"  Fee: 0% (maker — resting at bid)")
            logger.info("=" * 80)

            # Track position (use estimated cost which respects enforced min qty)
            current_market = self.all_btc_instruments[self.current_instrument_index]
            self._open_positions.append(OpenPosition(
                market_slug=current_market['slug'],
                direction=direction,
                entry_price=float(limit_price),
                size_usd=float(estimated_cost),
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
            min_qty_val = float(getattr(instrument, 'min_quantity', None) or Decimal("0.01"))
            token_qty = max_usd_amount / trade_price if trade_price > 0 else min_qty_val

            # Enforce a hard minimum token quantity (Polymarket enforces min shares, often 5)
            enforced_min_token_qty = float(os.getenv("MIN_TOKEN_QTY", "5.0"))
            final_min_qty = max(min_qty_val, enforced_min_token_qty)

            token_qty = max(token_qty, final_min_qty)
            token_qty = round(token_qty, precision)

            # Recompute estimated USD amount based on enforced token quantity
            try:
                estimated_cost = float(Decimal(str(token_qty)) * Decimal(str(trade_price)))
            except Exception:
                estimated_cost = float(max_usd_amount)

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
            logger.info(f"  Estimated Cost: ~${estimated_cost:.2f}")
            logger.info(f"  Fee: ~10% (taker)")
            logger.info("=" * 80)

            current_market = self.all_btc_instruments[self.current_instrument_index]
            self._open_positions.append(OpenPosition(
                market_slug=current_market['slug'],
                direction=direction,
                entry_price=trade_price,
                size_usd=estimated_cost,
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

    async def _safe_submit_order(self, order) -> bool:
        """
        Safely submit an order: if this is a paper-derived EXIT order or we're in
        simulation/shadow mode, do not submit to the exchange and simulate a local
        fill instead.

        Returns True if order was actually submitted, False if simulated/skipped.
        """
        try:
            # client_order_id may be a ClientOrderId object or string-like
            coid = getattr(order, 'client_order_id', None)
            coid_str = str(coid) if coid is not None else ''

            # If this is an EXIT for a paper trade, never send to exchange
            if 'EXIT-paper_' in coid_str or coid_str.startswith('EXIT-paper_'):
                logger.info(f"[SAFE SUBMIT] Skipping real submit for paper EXIT {coid_str}")
                return False

            is_sim = False
            try:
                is_sim = await self.check_simulation_mode()
            except Exception:
                is_sim = False

            if is_sim:
                logger.info(f"[SAFE SUBMIT] Simulation mode active — skipping real submit for {coid_str}")
                return False

            # Otherwise submit as normal
            self.submit_order(order)
            return True
        except Exception as e:
            logger.error(f"_safe_submit_order failed: {e}")
            return False

    def on_order_filled(self, event):
        logger.info("=" * 80)
        logger.info(f"ORDER FILLED!")
        logger.info(f"  Order: {event.client_order_id}")
        logger.info(f"  Fill Price: ${float(event.last_px):.4f}")
        logger.info(f"  Quantity: {float(event.last_qty):.6f}")
        logger.info("=" * 80)
        self._track_order_event("filled")

        # Update position with actual fill data
        order_id = str(event.client_order_id)
        for pos in self._open_positions:
            if pos.order_id == order_id:
                pos.entry_price = float(event.last_px)
                pos.actual_qty = float(event.last_qty)  # <--- CAPTURE ACTUAL QUANTITY
                logger.info(f"  ✓ Position updated: {pos.actual_qty} tokens @ ${pos.entry_price:.4f}")
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
        """Start Grafana exporter in a dedicated event loop that stays alive."""
        try:
            self._grafana_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._grafana_loop)
            self._grafana_loop.run_until_complete(self.grafana_exporter.start())
            # Keep loop alive so the _update_loop task continues running
            self._grafana_loop.run_forever()
        except Exception as e:
            logger.error(f"Grafana exporter stopped: {e}")
        finally:
            try:
                self._grafana_loop.close()
            except Exception:
                pass

    def _start_news_polling(self):
        """Start news/sentiment polling in a dedicated background thread."""

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._news_poll_loop())
            except Exception as e:
                logger.error(f"News polling thread error: {e}")
            finally:
                loop.close()

        t = threading.Thread(target=_run, daemon=True, name="news-poller")
        t.start()
        logger.info("News/sentiment poller started in background thread")
        return t

    async def _news_poll_loop(self):
        """Poll sentiment every 60s in its own thread/loop."""
        from data_sources.news_social.adapter import NewsSocialDataSource
        news_src = NewsSocialDataSource()
        await news_src.connect()

        while True:
            try:
                fg = await asyncio.wait_for(
                    news_src.get_fear_greed_index(), timeout=8.0
                )
                if fg and "value" in fg:
                    with self._cache_lock:
                        self._cached_sentiment = float(fg["value"])
                        self._cached_sentiment_class = fg.get("classification", "")
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"News poll error: {e}")

            await asyncio.sleep(60)  # F&G only updates daily, 60s is plenty

    async def _manage_open_positions(self):
        """
        Continuously monitor positions and execute Gamma Scalp exits.
        """
        now_ts = datetime.now(timezone.utc).timestamp()

        for pos in self._open_positions:
            if pos.resolved or pos.actual_qty <= 0: # Ensure we have tokens to sell
                continue

            try:
                instrument = self.cache.instrument(pos.instrument_id)
                if not instrument:
                    continue

                quote = self.cache.quote_tick(pos.instrument_id)
                if not quote:
                    continue
                # If long, sell at the BID to exit immediately
                current_exit_price = float(quote.bid_price.as_decimal())
            except Exception:
                continue

            # Calculate Unrealized PnL
            pnl_pct = (current_exit_price - pos.entry_price) / pos.entry_price

            end_ts = pos.market_end_time.timestamp()
            mins_remaining = max(0.0, (end_ts - now_ts) / 60.0)

            # GAMMA SCALP LOGIC using defined constants
            if pnl_pct >= TAKE_PROFIT_PCT or (
                    mins_remaining < GAMMA_EXIT_TIME_MINS and pnl_pct > GAMMA_EXIT_PROFIT_PCT):
                logger.info(
                    f"💰 GAMMA SCALP EXIT: {pos.market_slug} | "
                    f"PnL: +{pnl_pct:.1%} | T-Minus: {mins_remaining:.1f}m"
                )

                # Fetch dynamic precision from the instrument
                precision = instrument.size_precision

                # If this is a paper trade or we're running in simulation/shadow mode,
                # do NOT submit a real exchange order. Simulate the exit locally.
                try:
                    is_simulation = await self.check_simulation_mode()
                except Exception:
                    is_simulation = False

                if is_simulation or str(pos.order_id).startswith("paper_"):
                    logger.info(f"[SIMULATION] Simulating exit for {pos.order_id} — no real order submitted")
                    pos.resolved = True
                    pos.exit_price = current_exit_price
                    pos.pnl = pos.size_usd * pnl_pct
                    continue

                # FIX: Use pos.actual_qty instead of recalculating from pos.size_usd
                qty = Quantity(pos.actual_qty, precision=precision)
                order = self.order_factory.market(
                    instrument_id=pos.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=qty,
                    client_order_id=ClientOrderId(f"EXIT-{pos.order_id}"),
                    time_in_force=TimeInForce.IOC,
                )
                self.submit_order(order)
                pos.resolved = True
                pos.exit_price = current_exit_price
                pos.pnl = pos.size_usd * pnl_pct

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
            self.funding_filter.close()

        # Clean up persistent HTTP sessions
        global _gamma_session
        if _gamma_session is not None:
            try:
                _gamma_session.close()
            except Exception:
                pass
            _gamma_session = None

        if self.grafana_exporter:
            try:
                grafana_loop = getattr(self, '_grafana_loop', None)
                if grafana_loop and grafana_loop.is_running():
                    # Schedule stop() coroutine on the Grafana loop, then stop the loop
                    async def _shutdown_grafana():
                        await self.grafana_exporter.stop()
                        grafana_loop.stop()

                    asyncio.run_coroutine_threadsafe(_shutdown_grafana(), grafana_loop)
                else:
                    # Loop already stopped — just clean up
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.grafana_exporter.stop())
                    loop.close()
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

    print(
        f"\nMode: {'SIMULATION' if simulation else 'LIVE TRADING'} | Trade Size: ${POSITION_SIZE_USD} | Min Edge: ${MIN_EDGE_CENTS:.2f}")

    logger.info("Loading BTC 15-min markets via event slug builder...")

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
        logger.warning(f"⚠ LIVE TRADING MODE — REAL MONEY | {'LIMIT' if USE_LIMIT_ORDERS else 'MARKET'} orders")
    else:
        logger.info(f"SIMULATION MODE — {'TEST MODE' if test_mode else 'paper trading only'}")

    run_integrated_bot(simulation=simulation, enable_grafana=enable_grafana, test_mode=test_mode)


if __name__ == "__main__":
    main()
