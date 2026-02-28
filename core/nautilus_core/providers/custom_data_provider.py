"""
Custom Data Provider for NautilusTrader
Bridges our ingestion layer to Nautilus data engine
"""
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from collections import defaultdict

from nautilus_trader.model.data import QuoteTick, TradeTick, Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, TradeId
from nautilus_trader.model.enums import AggressorSide, BarAggregation, PriceType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.common.component import LiveClock, Logger
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.data.engine import DataEngine
from loguru import logger as loguru_logger
import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.ingestion.adapters.unified_adapter import UnifiedDataAdapter, MarketData
from core.nautilus_core.instruments.btc_instruments import get_instrument_registry


class CustomDataProvider:
    """
    Custom data provider that feeds market data from our ingestion layer
    into NautilusTrader's data engine.
    
    Converts:
    - MarketData → QuoteTick
    - MarketData → TradeTick (synthetic)
    - Aggregates → Bar data
    """
    
    def __init__(self, data_engine: DataEngine, clock: LiveClock, logger: Logger):
        """Initialize custom data provider."""
        self.data_engine = data_engine
        self.clock = clock
        self.logger = logger
        self.adapter: Optional[UnifiedDataAdapter] = None
        self.instruments = get_instrument_registry()
        self._last_prices: dict = {}
        self._bar_aggregators: dict = defaultdict(list)
        loguru_logger.info("Initialized Custom Data Provider")
    
    async def connect(self) -> None:
        """Connect to data sources."""
        loguru_logger.info("Connecting custom data provider...")
        
        # Create and connect unified adapter
        self.adapter = UnifiedDataAdapter()
        
        # Set callbacks
        self.adapter.on_price_update = self._on_price_update
        self.adapter.on_sentiment_update = self._on_sentiment_update
        
        # Connect all sources
        results = await self.adapter.connect_all()
        
        connected = sum(results.values())
        loguru_logger.info(f"Connected {connected}/{len(results)} data sources")
        
        # Register instruments with data engine
        self._register_instruments()
        
        # Start streaming
        await self.adapter.start_streaming()
        
        loguru_logger.info("Custom data provider connected and streaming")
    
    async def disconnect(self) -> None:
        """Disconnect from data sources."""
        if self.adapter:
            await self.adapter.disconnect_all()
        
        loguru_logger.info("Custom data provider disconnected")
    
    def _register_instruments(self) -> None:
        """Register instruments with data engine."""
        for instrument in self.instruments.get_all():
            # Add instrument to data engine's cache
            # This makes it available to strategies
            loguru_logger.info(f"Registered instrument: {instrument.id}")
    
    async def _on_price_update(self, data: MarketData) -> None:
        """Handle price update - convert to Nautilus ticks."""
        try:
            iid = self._get_instrument_id(data.source)
            if not iid: return
            qt = self._create_quote_tick(data, iid)
            if qt:
                self.data_engine.process(qt)
                tt = self._create_trade_tick(data, iid)
                if tt: self.data_engine.process(tt)
            self._last_prices[data.source] = data.price
        except Exception as e:
            loguru_logger.error(f"Price update error: {e}")

    async def _on_sentiment_update(self, data) -> None:
        """
        Handle sentiment update.
        
        Args:
            data: Sentiment data
        """
        # Store sentiment for strategies to access
        loguru_logger.debug(f"Sentiment update: {data.score}/100 - {data.classification}")
    
    def _get_instrument_id(self, source: str) -> Optional[InstrumentId]:
        """
        Map data source to instrument ID.
        
        Args:
            source: Data source name
            
        Returns:
            InstrumentId or None
        """
        mapping = {
            "coinbase": "BTC-USD.COINBASE",
            "binance": "BTCUSDT.BINANCE",
        }
        
        instrument_id_str = mapping.get(source)
        if not instrument_id_str:
            return None
        
        instrument = self.instruments.get(instrument_id_str)
        return instrument.id if instrument else None
    
    @staticmethod
    def _format_price_str(val):
        """Format a float/Decimal to a valid Price string (max 9 decimals)."""
        s = f"{float(val):.9f}".rstrip('0').rstrip('.')
        return s if '.' in s else f"{s}.0"

    def _create_quote_tick(
        self,
        data: MarketData,
        instrument_id: InstrumentId,
    ) -> Optional[QuoteTick]:
        """Create QuoteTick from market data."""
        try:
            if data.bid and data.ask:
                bid_p = Price.from_str(self._format_price_str(data.bid))
                ask_p = Price.from_str(self._format_price_str(data.ask))
            else:
                spread = data.price * Decimal("0.001")
                bid_p = Price.from_str(self._format_price_str(data.price - spread))
                ask_p = Price.from_str(self._format_price_str(data.price + spread))
            return QuoteTick(
                instrument_id=instrument_id, bid_price=bid_p, ask_price=ask_p,
                bid_size=Quantity.from_str("1.0"), ask_size=Quantity.from_str("1.0"),
                ts_event=self._to_nanoseconds(data.timestamp), ts_init=self.clock.timestamp_ns())
        except Exception as e:
            loguru_logger.error(f"Quote tick error: {e}"); return None

    def _create_trade_tick(
        self,
        data: MarketData,
        instrument_id: InstrumentId,
    ) -> Optional[TradeTick]:
        """Create synthetic TradeTick from market data."""
        try:
            last = self._last_prices.get(data.source)
            if last and last == data.price: return None
            side = AggressorSide.BUYER if (not last or data.price > last) else AggressorSide.SELLER
            return TradeTick(
                instrument_id=instrument_id, price=Price.from_str(str(data.price)),
                size=Quantity.from_str("1.0"), aggressor_side=side,
                trade_id=TradeId(f"{data.source}_{data.timestamp.timestamp()}"),
                ts_event=self._to_nanoseconds(data.timestamp), ts_init=self.clock.timestamp_ns())
        except Exception as e:
            loguru_logger.error(f"Trade tick error: {e}"); return None

    @staticmethod
    def _to_nanoseconds(dt: datetime) -> int:
        """Convert datetime to nanoseconds since epoch."""
        return int(dt.timestamp() * 1_000_000_000)
    
    def get_latest_price(self, source: str) -> Optional[Decimal]:
        """Get latest price from a source."""
        return self._last_prices.get(source)
    
    def get_price_consensus(self) -> Optional[dict]:
        """Get price consensus across all sources."""
        if self.adapter:
            return self.adapter.get_price_consensus()
        return None