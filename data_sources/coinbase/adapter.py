"""
Coinbase Data Source Adapter
Fetches BTC price data from Coinbase Pro API
"""
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
import httpx
from loguru import logger


class CoinbaseDataSource:
    """
    Coinbase Pro API data source for BTC price data.
    
    Provides:
    - Real-time BTC price
    - Order book data
    - Recent trades
    - 24h statistics
    """
    
    def __init__(
        self,
        base_url: str = "https://api.exchange.coinbase.com",
        product_id: str = "BTC-USD",
    ):
        """
        Initialize Coinbase data source.
        
        Args:
            base_url: Coinbase Pro API base URL
            product_id: Trading pair (default: BTC-USD)
        """
        self.base_url = base_url
        self.product_id = product_id
        self.session: Optional[httpx.AsyncClient] = None
        
        # Cache
        self._last_price: Optional[Decimal] = None
        self._last_update: Optional[datetime] = None
        self._last_logged_price: Optional[Decimal] = None

        logger.info(f"Initialized Coinbase data source for {product_id}")
    
    async def connect(self) -> bool:
        """
        Connect to Coinbase API.
        
        Returns:
            True if connection successful
        """
        try:
            self.session = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                headers={
                    "User-Agent": "PolymarketBot/1.0",
                    "Accept": "application/json",
                }
            )
            
            # Test connection
            response = await self.session.get(f"/products/{self.product_id}")
            response.raise_for_status()
            
            logger.info("✓ Connected to Coinbase API")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Coinbase: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection."""
        if self.session:
            await self.session.aclose()
            logger.info("Disconnected from Coinbase API")
    
    async def get_current_price(self) -> Optional[Decimal]:
        """
        Get current BTC price.
        
        Returns:
            Current price or None if error
        """
        try:
            response = await self.session.get(f"/products/{self.product_id}/ticker")
            response.raise_for_status()
            
            data = response.json()
            price = Decimal(str(data["price"]))
            
            self._last_price = price
            self._last_update = datetime.now()
            
            if price != self._last_logged_price:
                logger.trace(f"Coinbase BTC price: ${price:,.2f}")
                self._last_logged_price = price
            return price
            
        except Exception as e:
            logger.error(f"Error fetching Coinbase price: {e}")
            return None
    
    async def get_order_book(self, level: int = 2) -> Optional[Dict[str, Any]]:
        """Get order book data. Level: 1=best, 2=top50, 3=full."""
        try:
            resp = await self.session.get(f"/products/{self.product_id}/book", params={"level": level})
            resp.raise_for_status()
            data = resp.json()
            parse = lambda levels: [{"price": Decimal(l[0]), "size": Decimal(l[1])} for l in levels]
            return {"timestamp": datetime.now(), "bids": parse(data.get("bids", [])),
                    "asks": parse(data.get("asks", []))}
        except Exception as e:
            logger.error(f"Coinbase order book error: {e}"); return None

    async def get_24h_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get 24-hour statistics.
        
        Returns:
            Dict with open, high, low, volume, etc.
        """
        try:
            response = await self.session.get(f"/products/{self.product_id}/stats")
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "timestamp": datetime.now(),
                "open": Decimal(str(data["open"])),
                "high": Decimal(str(data["high"])),
                "low": Decimal(str(data["low"])),
                "volume": Decimal(str(data["volume"])),
                "last": Decimal(str(data["last"])),
            }
            
        except Exception as e:
            logger.error(f"Error fetching Coinbase 24h stats: {e}")
            return None
    
    async def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        try:
            resp = await self.session.get(f"/products/{self.product_id}/trades", params={"limit": limit})
            resp.raise_for_status()
            return [{"timestamp": datetime.fromisoformat(t["time"].replace("Z", "+00:00")),
                     "trade_id": t["trade_id"], "price": Decimal(str(t["price"])),
                     "size": Decimal(str(t["size"])), "side": t["side"]}
                    for t in resp.json()[:limit]]
        except Exception as e:
            logger.error(f"Coinbase trades error: {e}"); return []

    async def get_candles(self, granularity: int = 300, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical candles (OHLCV). Granularity in seconds."""
        try:
            resp = await self.session.get(f"/products/{self.product_id}/candles",
                                          params={"granularity": granularity})
            resp.raise_for_status()
            return [{"timestamp": datetime.fromtimestamp(c[0]), "open": Decimal(str(c[3])),
                     "high": Decimal(str(c[2])), "low": Decimal(str(c[1])),
                     "close": Decimal(str(c[4])), "volume": Decimal(str(c[5]))}
                    for c in resp.json()[:limit]]
        except Exception as e:
            logger.error(f"Coinbase candles error: {e}"); return []

    @property
    def last_price(self) -> Optional[Decimal]:
        """Get cached last price."""
        return self._last_price
    
    @property
    def last_update(self) -> Optional[datetime]:
        """Get time of last price update."""
        return self._last_update
    
    async def health_check(self) -> bool:
        """
        Check if data source is healthy.
        
        Returns:
            True if healthy
        """
        try:
            price = await self.get_current_price()
            return price is not None
        except:
            return False


# Singleton instance for easy import
_coinbase_instance: Optional[CoinbaseDataSource] = None

def get_coinbase_source() -> CoinbaseDataSource:
    """Get singleton instance of Coinbase data source."""
    global _coinbase_instance
    if _coinbase_instance is None:
        _coinbase_instance = CoinbaseDataSource()
    return _coinbase_instance