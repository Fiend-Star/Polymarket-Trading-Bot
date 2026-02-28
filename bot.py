"""
Integrated BTC Strategy V3.1 — Polymarket Binary Option Trading Bot.

This is the thin composition shell. All behavior lives in bot_mixins/:
  market_mixin.py   — instrument loading, pairing, switching
  quote_mixin.py    — quote tick processing, timer loop
  trading_mixin.py  — quant model + heuristic trading decisions
  orders_mixin.py   — order execution, paper trades, signal processing
  events_mixin.py   — order events, lifecycle, shutdown
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Dict
import threading

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from patch_gamma_markets import apply_gamma_markets_patch, verify_patch
    if not apply_gamma_markets_patch():
        print("ERROR: Failed to apply gamma_market patch"); sys.exit(1)
    verify_patch()
except ImportError as e:
    print(f"ERROR: Could not import patch module: {e}"); sys.exit(1)

from nautilus_trader.config import (
    LiveDataEngineConfig, LiveExecEngineConfig, LiveRiskEngineConfig,
    LoggingConfig, TradingNodeConfig)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.adapters.polymarket import POLYMARKET
from nautilus_trader.adapters.polymarket import PolymarketDataClientConfig, PolymarketExecClientConfig
from nautilus_trader.adapters.polymarket.providers import PolymarketInstrumentProviderConfig
from nautilus_trader.adapters.polymarket.factories import (
    PolymarketLiveDataClientFactory, PolymarketLiveExecClientFactory)
from nautilus_trader.trading.strategy import Strategy
from dotenv import load_dotenv
from loguru import logger
import redis
from config import get_config
from container import get_container, ServiceContainer
from data_source_manager import DataSourceManager
from position_tracker import PositionTracker
from order_dispatcher import PaperTrade
from rtds_connector import RTDSConnector
from funding_rate_filter import FundingRateFilter
from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.signal_processors.orderbook_processor import OrderBookImbalanceProcessor
from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
from core.strategy_brain.signal_processors.deribit_pcr_processor import DeribitPCRProcessor
from bot_mixins.market_mixin import MarketLifecycleMixin
from bot_mixins.quote_mixin import QuoteHandlerMixin
from bot_mixins.trading_mixin import TradingDecisionMixin
from bot_mixins.orders_mixin import OrdersMixin
from bot_mixins.events_mixin import EventsMixin
from bot_mixins._constants import (
    QUOTE_MIN_SPREAD, POSITION_SIZE_USD, MIN_EDGE_CENTS, VOL_METHOD,
    USE_LIMIT_ORDERS, USE_RTDS, USE_FUNDING_FILTER,
    RTDS_LATE_WINDOW_SEC, RTDS_DIVERGENCE_THRESHOLD_BPS,
    RTDS_LATE_WINDOW_MIN_BPS, LATE_WINDOW_ENABLED)

load_dotenv()
from patch_market_orders import apply_market_order_patch
if apply_market_order_patch():
    logger.info("Market order patch applied")
else:
    logger.warning("Market order patch failed")

_CFG = get_config()


# ═══════════════════════════════════════════════════════════════════════
# Redis
# ═══════════════════════════════════════════════════════════════════════

def init_redis():
    rc = _CFG.redis
    try:
        c = redis.Redis(host=rc.host, port=rc.port, db=rc.db,
                        decode_responses=True, socket_connect_timeout=5)
        c.ping(); logger.info("Redis connected"); return c
    except Exception as e:
        logger.warning(f"Redis failed: {e}"); return None


# ═══════════════════════════════════════════════════════════════════════
# Strategy (composition of mixins)
# ═══════════════════════════════════════════════════════════════════════

class IntegratedBTCStrategy(
    MarketLifecycleMixin, QuoteHandlerMixin, TradingDecisionMixin,
    OrdersMixin, EventsMixin, Strategy,
):
    """V3.1 BTC 15-min binary option strategy — composed from focused mixins."""

    def __init__(self, redis_client=None, enable_grafana=True, test_mode=False,
                 container: Optional[ServiceContainer] = None):
        super().__init__()
        self.bot_start_time = datetime.now(timezone.utc)
        self.restart_after_minutes = _CFG.restart_after_minutes
        self.redis_client = redis_client
        self.current_simulation_mode = False
        self.test_mode = test_mode
        self._svc = container or get_container()
        self._init_market_state()
        self._init_quant_modules()
        self._init_rtds_and_funding()
        self._init_signal_processors()
        self._init_risk_and_monitoring(enable_grafana)
        if test_mode:
            logger.info("TEST MODE")

    def _init_market_state(self):
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
        self._data_mgr: Optional[DataSourceManager] = None
        self.price_history = []
        self.max_history = 100
        self.paper_trades: List[PaperTrade] = []

    def _init_quant_modules(self):
        self.binary_pricer = self._svc.pricer
        self.vol_estimator = self._svc.vol_estimator
        self.mispricing_detector = self._svc.mispricing_detector
        self._btc_strike_price: Optional[float] = None
        self._strike_recorded = False
        self._active_entry: Optional[dict] = None
        self._late_window_traded = False

    def _init_rtds_and_funding(self):
        self.rtds = RTDSConnector(
            vol_estimator=self.vol_estimator,
            divergence_threshold_bps=RTDS_DIVERGENCE_THRESHOLD_BPS,
            late_window_max_sec=RTDS_LATE_WINDOW_SEC,
            late_window_min_bps=RTDS_LATE_WINDOW_MIN_BPS,
        ) if USE_RTDS else None
        self.funding_filter = FundingRateFilter() if USE_FUNDING_FILTER else None
        self._data_mgr = DataSourceManager(
            vol_estimator=self.vol_estimator, rtds=self.rtds,
            funding_filter=self.funding_filter)

    def _init_signal_processors(self):
        fc = _CFG.fusion
        self.spike_detector = SpikeDetectionProcessor(spike_threshold=fc.spike_threshold, lookback_periods=20)
        self.sentiment_processor = SentimentProcessor(extreme_fear_threshold=25, extreme_greed_threshold=75)
        self.divergence_processor = PriceDivergenceProcessor(divergence_threshold=fc.divergence_threshold)
        self.orderbook_processor = OrderBookImbalanceProcessor(imbalance_threshold=0.30, min_book_volume=50.0)
        self.tick_velocity_processor = TickVelocityProcessor(velocity_threshold_60s=0.015, velocity_threshold_30s=0.010)
        self.deribit_pcr_processor = DeribitPCRProcessor(
            bullish_pcr_threshold=1.20, bearish_pcr_threshold=0.70,
            max_days_to_expiry=2, cache_seconds=300)
        self.fusion_engine = self._svc.fusion_engine
        for name, w in [("OrderBookImbalance", 0.30), ("TickVelocity", 0.25),
                        ("PriceDivergence", 0.18), ("SpikeDetection", 0.12),
                        ("DeribitPCR", 0.10), ("SentimentAnalysis", 0.05)]:
            self.fusion_engine.set_weight(name, w)

    def _init_risk_and_monitoring(self, enable_grafana):
        self.risk_engine = self._svc.risk_engine
        self.performance_tracker = self._svc.performance_tracker
        self.learning_engine = self._svc.learning_engine
        if enable_grafana:
            from monitoring.grafana_exporter import get_grafana_exporter
            self.grafana_exporter = get_grafana_exporter()
        else:
            self.grafana_exporter = None
        self._pos_tracker = PositionTracker(
            cache_ref=self.cache, performance=self.performance_tracker,
            metrics=self.grafana_exporter)

    # ── Thin delegates ───────────────────────────────────────────────

    async def _init_data_sources(self):
        await self._data_mgr.init_sources()

    async def _preseed_vol_estimator(self):
        await self._data_mgr.preseed_vol(VOL_METHOD)

    async def _teardown_data_sources(self):
        await self._data_mgr.teardown()

    async def _refresh_cached_data(self):
        await self._data_mgr.refresh()

    def _get_cached_data(self) -> dict:
        return self._data_mgr.get_cached()

    async def check_simulation_mode(self) -> bool:
        if not self.redis_client:
            return self.current_simulation_mode
        try:
            v = self.redis_client.get('btc_trading:simulation_mode')
            if v is not None:
                s = v == '1'
                if s != self.current_simulation_mode:
                    self.current_simulation_mode = s
                    logger.warning(f"Mode → {'SIM' if s else 'LIVE'}")
                return s
        except Exception:
            pass
        return self.current_simulation_mode

    def _is_quote_valid(self, bid, ask) -> bool:
        try:
            b, a = float(bid), float(ask)
        except (TypeError, ValueError):
            return False
        return QUOTE_MIN_SPREAD < b < 0.999 and QUOTE_MIN_SPREAD < a < 0.999

    def _reset_stability(self, reason=""):
        if self._market_stable:
            logger.warning(f"Stability RESET{' — '+reason if reason else ''}")
        self._market_stable = False
        self._stable_tick_count = 0


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def _build_polymarket_configs():
    creds = _CFG.creds
    ic = PolymarketInstrumentProviderConfig(event_slug_builder="slug_builders:build_btc_15min_slugs")
    kw = dict(private_key=creds.private_key, api_key=creds.api_key,
              api_secret=creds.api_secret, passphrase=creds.passphrase,
              signature_type=creds.signature_type, instrument_config=ic)
    return PolymarketDataClientConfig(**kw), PolymarketExecClientConfig(**kw)


def _build_node_config(dc, ec):
    return TradingNodeConfig(
        environment="live", trader_id="BTC-15MIN-V31-001",
        logging=LoggingConfig(log_level="ERROR", log_directory="./logs/nautilus",
                              log_component_levels={"IntegratedBTCStrategy": "INFO"}),
        data_engine=LiveDataEngineConfig(qsize=6000),
        exec_engine=LiveExecEngineConfig(qsize=6000),
        risk_engine=LiveRiskEngineConfig(bypass=False),
        data_clients={POLYMARKET: dc}, exec_clients={POLYMARKET: ec})


def run_integrated_bot(simulation=False, enable_grafana=True, test_mode=False):
    print("=" * 60)
    print("BTC 15-MIN TRADING BOT V3.1")
    print("=" * 60)
    rc = init_redis()
    if rc:
        try:
            rc.set('btc_trading:simulation_mode', '1' if simulation else '0')
        except Exception:
            pass
    strategy = IntegratedBTCStrategy(redis_client=rc, enable_grafana=enable_grafana, test_mode=test_mode)
    dc, ec = _build_polymarket_configs()
    node = TradingNode(config=_build_node_config(dc, ec))
    node.add_data_client_factory(POLYMARKET, PolymarketLiveDataClientFactory)
    node.add_exec_client_factory(POLYMARKET, PolymarketLiveExecClientFactory)
    node.trader.add_strategy(strategy)
    node.build()
    try:
        node.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.dispose()


def main():
    import argparse
    p = argparse.ArgumentParser(description="BTC 15-Min Trading Bot V3.1")
    p.add_argument("--live", action="store_true")
    p.add_argument("--no-grafana", action="store_true")
    p.add_argument("--test-mode", action="store_true")
    a = p.parse_args()
    sim = not a.live if not a.test_mode else True
    run_integrated_bot(simulation=sim, enable_grafana=not a.no_grafana, test_mode=a.test_mode)


if __name__ == "__main__":
    main()


