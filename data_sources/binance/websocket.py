"""
Binance WebSocket Data Source
Real-time BTC price streaming from Binance
"""
import asyncio
import json
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable, Dict, Any
import websockets
from loguru import logger


class BinanceWebSocketSource:
    """
    Binance WebSocket data source for real-time BTC data.
    
    Provides:
    - Real-time ticker updates
    - Trade stream
    - Order book updates
    - Kline (candlestick) data
    """
    
    def __init__(
        self,
        symbol: str = "btcusdt",
        ws_url: str = "wss://stream.binance.com:9443/ws",
    ):
        """
        Initialize Binance WebSocket source.
        
        Args:
            symbol: Trading pair (lowercase)
            ws_url: WebSocket endpoint URL
        """
        self.symbol = symbol.lower()
        self.ws_url = ws_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        
        # Callbacks
        self.on_price_update: Optional[Callable] = None
        self.on_trade: Optional[Callable] = None
        self.on_orderbook: Optional[Callable] = None
        
        # State
        self._is_running = False
        self._last_price: Optional[Decimal] = None
        self._last_update: Optional[datetime] = None
        
        logger.info(f"Initialized Binance WebSocket for {symbol.upper()}")
    
    async def connect(self, stream_type: str = "ticker") -> bool:
        """
        Connect to Binance WebSocket.
        
        Args:
            stream_type: "ticker", "trade", "depth", "kline_1m", etc.
            
        Returns:
            True if connected successfully
        """
        try:
            # Build stream URL
            stream = f"{self.symbol}@{stream_type}"
            url = f"{self.ws_url}/{stream}"
            
            self.websocket = await websockets.connect(url)
            self._is_running = True
            
            logger.info(f"✓ Connected to Binance WebSocket: {stream}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance WebSocket: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._is_running = False
        
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from Binance WebSocket")
    
    async def _stream_loop(self, stream_name, parser, callback_attr):
        """Generic streaming loop: connect, parse messages, invoke callback."""
        await self.connect(stream_name)
        try:
            while self._is_running and self.websocket:
                msg = await self.websocket.recv()
                data = json.loads(msg)
                parsed = parser(data)
                if "price" in parsed:
                    self._last_price = parsed["price"]
                if "timestamp" in parsed:
                    self._last_update = parsed["timestamp"]
                cb = getattr(self, callback_attr, None)
                if cb:
                    await cb(parsed)
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Binance {stream_name} connection closed")
        except Exception as e:
            logger.error(f"Error in Binance {stream_name}: {e}")
        finally:
            await self.disconnect()

    def _parse_ticker(self, data):
        """Parse 24hr ticker message."""
        return {
            "timestamp": datetime.fromtimestamp(data["E"] / 1000), "symbol": data["s"],
            "price": Decimal(data["c"]), "open": Decimal(data["o"]),
            "high": Decimal(data["h"]), "low": Decimal(data["l"]),
            "volume": Decimal(data["v"]), "quote_volume": Decimal(data["q"]),
            "price_change": Decimal(data["p"]), "price_change_percent": Decimal(data["P"])}

    def _parse_trade(self, data):
        """Parse individual trade message."""
        return {
            "timestamp": datetime.fromtimestamp(data["T"] / 1000), "trade_id": data["t"],
            "price": Decimal(data["p"]), "quantity": Decimal(data["q"]),
            "buyer_is_maker": data["m"], "side": "sell" if data["m"] else "buy"}

    def _parse_orderbook(self, data):
        """Parse order book depth message."""
        parse_levels = lambda levels: [{"price": Decimal(l[0]), "quantity": Decimal(l[1])} for l in levels]
        return {"timestamp": datetime.now(), "last_update_id": data.get("lastUpdateId"),
                "bids": parse_levels(data.get("bids", [])), "asks": parse_levels(data.get("asks", []))}

    def _parse_kline(self, data):
        """Parse kline/candlestick message."""
        k = data["k"]
        return {"timestamp": datetime.fromtimestamp(k["t"] / 1000),
                "open": Decimal(k["o"]), "high": Decimal(k["h"]),
                "low": Decimal(k["l"]), "close": Decimal(k["c"]),
                "volume": Decimal(k["v"]), "is_closed": k["x"]}

    async def stream_ticker(self) -> None:
        """Stream 24hr ticker updates."""
        await self._stream_loop("ticker", self._parse_ticker, "on_price_update")

    async def stream_trades(self) -> None:
        """Stream individual trades."""
        await self._stream_loop("trade", self._parse_trade, "on_trade")

    async def stream_orderbook(self, depth: str = "5") -> None:
        """Stream order book depth updates."""
        await self._stream_loop(f"depth{depth}", self._parse_orderbook, "on_orderbook")

    async def stream_klines(self, interval: str = "1m") -> None:
        """Stream candlestick data."""
        await self._stream_loop(f"kline_{interval}", self._parse_kline, None)

    @property
    def last_price(self) -> Optional[Decimal]:
        """Get last received price."""
        return self._last_price
    
    @property
    def last_update(self) -> Optional[datetime]:
        """Get timestamp of last update."""
        return self._last_update
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._is_running and self.websocket is not None
    
    async def health_check(self) -> bool:
        """Check if WebSocket is healthy."""
        return self.is_connected and self._last_price is not None


# Singleton instance
_binance_instance: Optional[BinanceWebSocketSource] = None

def get_binance_source() -> BinanceWebSocketSource:
    """Get singleton instance of Binance WebSocket source."""
    global _binance_instance
    if _binance_instance is None:
        _binance_instance = BinanceWebSocketSource()
    return _binance_instance