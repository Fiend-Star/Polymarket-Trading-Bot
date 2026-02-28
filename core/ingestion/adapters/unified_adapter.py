"""
Unified Data Adapter
Provides single interface to all external data sources
"""
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from loguru import logger

import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from data_sources.coinbase.adapter import CoinbaseDataSource
from data_sources.binance.websocket import BinanceWebSocketSource
from data_sources.news_social.adapter import NewsSocialDataSource
from data_sources.solana.rpc import SolanaRPCDataSource


@dataclass
class MarketData:
    """Normalized market data structure."""
    timestamp: datetime
    source: str  # "coinbase", "binance", etc.
    symbol: str  # "BTC-USD", "BTCUSDT", etc.
    price: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    high_24h: Optional[Decimal] = None
    low_24h: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentimentData:
    """Normalized sentiment data structure."""
    timestamp: datetime
    source: str
    score: float  # 0-100
    classification: str  # "extreme_fear", "fear", "neutral", "greed", "extreme_greed"
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedDataAdapter:
    """
    Unified adapter that aggregates all data sources.
    
    Provides single interface for:
    - Market data (price, orderbook, trades)
    - Sentiment data (fear/greed, news)
    - On-chain data (network stats)
    """
    
    def __init__(self):
        """Initialize unified adapter."""
        # Data sources
        self.coinbase: Optional[CoinbaseDataSource] = None
        self.binance: Optional[BinanceWebSocketSource] = None
        self.news_social: Optional[NewsSocialDataSource] = None
        self.solana: Optional[SolanaRPCDataSource] = None
        
        # Callbacks
        self.on_price_update: Optional[Callable[[MarketData], None]] = None
        self.on_sentiment_update: Optional[Callable[[SentimentData], None]] = None
        
        # State
        self._is_running = False
        self._update_tasks: List[asyncio.Task] = []
        
        # Cache latest data from each source
        self._latest_data: Dict[str, MarketData] = {}
        self._latest_sentiment: Optional[SentimentData] = None
        
        logger.info("Initialized Unified Data Adapter")
    
    async def _connect_source(self, name, factory, connect_args=()):
        """Connect a single data source. Returns (name, success)."""
        try:
            src = factory()
            setattr(self, name, src)
            return await (src.connect(*connect_args) if connect_args else src.connect())
        except Exception as e:
            logger.error(f"Failed to connect {name}: {e}"); return False

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all data sources. Returns {name: connected}."""
        logger.info("Connecting to all data sources...")
        results = {
            "coinbase": await self._connect_source("coinbase", CoinbaseDataSource),
            "binance": await self._connect_source("binance", BinanceWebSocketSource, ("ticker",)),
            "news_social": await self._connect_source("news_social", NewsSocialDataSource),
            "solana": await self._connect_source("solana", SolanaRPCDataSource),
        }
        logger.info(f"Connected {sum(results.values())}/{len(results)} sources")
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect from all data sources."""
        logger.info("Disconnecting from all data sources...")
        
        self._is_running = False
        
        # Cancel all update tasks
        for task in self._update_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._update_tasks.clear()
        
        # Disconnect sources
        if self.coinbase:
            await self.coinbase.disconnect()
        
        if self.binance:
            await self.binance.disconnect()
        
        if self.news_social:
            await self.news_social.disconnect()
        
        if self.solana:
            await self.solana.disconnect()
        
        logger.info("Disconnected from all data sources")
    
    async def start_streaming(self) -> None:
        """Start streaming data from all sources."""
        self._is_running = True
        logger.info("Starting data streams...")
        
        # Start Coinbase polling (REST API)
        if self.coinbase:
            task = asyncio.create_task(self._poll_coinbase())
            self._update_tasks.append(task)
        
        # Start Binance WebSocket streaming
        if self.binance:
            task = asyncio.create_task(self._stream_binance())
            self._update_tasks.append(task)
        
        # Start sentiment polling
        if self.news_social:
            task = asyncio.create_task(self._poll_sentiment())
            self._update_tasks.append(task)
        
        logger.info(f"Started {len(self._update_tasks)} data streams")
    
    async def _poll_coinbase(self, interval: int = 5) -> None:
        """Poll Coinbase API for price updates."""
        while self._is_running:
            try:
                price = await self.coinbase.get_current_price()
                if price:
                    book = await self.coinbase.get_order_book(level=1)
                    stats = await self.coinbase.get_24h_stats()
                    md = MarketData(
                        timestamp=datetime.now(), source="coinbase", symbol="BTC-USD", price=price,
                        bid=book["bids"][0]["price"] if book and book.get("bids") else None,
                        ask=book["asks"][0]["price"] if book and book.get("asks") else None,
                        volume_24h=stats["volume"] if stats else None,
                        high_24h=stats["high"] if stats else None,
                        low_24h=stats["low"] if stats else None)
                    self._latest_data["coinbase"] = md
                    if self.on_price_update: await self.on_price_update(md)
                await asyncio.sleep(interval)
            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"Coinbase poll error: {e}"); await asyncio.sleep(interval)

    async def _stream_binance(self) -> None:
        """Stream real-time data from Binance WebSocket."""
        async def on_ticker(td: Dict[str, Any]) -> None:
            try:
                md = MarketData(
                    timestamp=td["timestamp"], source="binance", symbol="BTCUSDT",
                    price=td["price"], volume_24h=td["volume"], high_24h=td["high"],
                    low_24h=td["low"], metadata={"price_change": td["price_change"],
                    "price_change_percent": td["price_change_percent"]})
                self._latest_data["binance"] = md
                if self.on_price_update: await self.on_price_update(md)
            except Exception as e:
                logger.error(f"Binance ticker error: {e}")
        self.binance.on_price_update = on_ticker
        try: await self.binance.stream_ticker()
        except asyncio.CancelledError: pass
        except Exception as e: logger.error(f"Binance stream error: {e}")

    async def _poll_sentiment(self, interval: int = 300) -> None:
        """Poll sentiment data every `interval` seconds."""
        while self._is_running:
            try:
                fg = await self.news_social.get_fear_greed_index()
                if fg:
                    sd = SentimentData(
                        timestamp=fg["timestamp"], source="fear_greed_index",
                        score=float(fg["value"]),
                        classification=fg["classification"].lower().replace(" ", "_"),
                        metadata=fg)
                    self._latest_sentiment = sd
                    if self.on_sentiment_update: await self.on_sentiment_update(sd)
                await asyncio.sleep(interval)
            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"Sentiment poll error: {e}"); await asyncio.sleep(interval)

    def get_latest_price(self, source: Optional[str] = None) -> Optional[Decimal]:
        """
        Get latest price from a specific source or average of all.
        
        Args:
            source: Specific source ("coinbase", "binance") or None for average
            
        Returns:
            Latest price
        """
        if source:
            data = self._latest_data.get(source)
            return data.price if data else None
        else:
            # Return average of all sources
            prices = [data.price for data in self._latest_data.values()]
            if prices:
                return sum(prices) / len(prices)
            return None
    
    def get_price_consensus(self) -> Optional[Dict[str, Any]]:
        """
        Get price consensus across all sources.
        
        Returns:
            Dict with average, min, max, spread, sources
        """
        if not self._latest_data:
            return None
        
        prices = [data.price for data in self._latest_data.values()]
        
        return {
            "timestamp": datetime.now(),
            "average": sum(prices) / len(prices),
            "min": min(prices),
            "max": max(prices),
            "spread": max(prices) - min(prices),
            "spread_percent": ((max(prices) - min(prices)) / min(prices)) * 100,
            "num_sources": len(prices),
            "sources": {
                source: float(data.price)
                for source, data in self._latest_data.items()
            }
        }
    
    def get_latest_sentiment(self) -> Optional[SentimentData]:
        """Get latest sentiment data."""
        return self._latest_sentiment
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all data sources.
        
        Returns:
            Dict of source -> is_healthy
        """
        health = {}
        
        if self.coinbase:
            health["coinbase"] = await self.coinbase.health_check()
        
        if self.binance:
            health["binance"] = await self.binance.health_check()
        
        if self.news_social:
            health["news_social"] = await self.news_social.health_check()
        
        if self.solana:
            health["solana"] = await self.solana.health_check()
        
        return health


# Singleton instance
_adapter_instance: Optional[UnifiedDataAdapter] = None

def get_unified_adapter() -> UnifiedDataAdapter:
    """Get singleton instance of unified adapter."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = UnifiedDataAdapter()
    return _adapter_instance