"""
Polymarket Client - Production Implementation
Real API integration with Polymarket CLOB
"""
import os
import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any, List
from loguru import logger

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType as PolyOrderType
from py_clob_client.order_builder.constants import BUY, SELL
POLYMARKET_AVAILABLE = True


class PolymarketClient:
    """
    Production Polymarket API client.
    
    Features:
    - Real order placement
    - Live market data
    - Position tracking
    - Balance management
    """
    
    def __init__(self, private_key: Optional[str] = None, api_key: Optional[str] = None,
                 api_secret: Optional[str] = None, api_passphrase: Optional[str] = None,
                 chain_id: int = 137, testnet: bool = False):
        """Initialize Polymarket client from args or environment."""
        self.private_key = private_key or os.getenv("POLYMARKET_PK")
        self.api_key = api_key or os.getenv("POLYMARKET_API_KEY")
        self.api_secret = api_secret or os.getenv("POLYMARKET_API_SECRET")
        self.api_passphrase = api_passphrase or os.getenv("POLYMARKET_PASSPHRASE")
        self.chain_id = chain_id
        self.testnet = testnet
        self.client: Optional[ClobClient] = None
        self._connected = False
        self._markets_cache: Dict[str, Any] = {}
        if not POLYMARKET_AVAILABLE:
            logger.error("Polymarket SDK not available"); return
        if not self.private_key: logger.error("POLYMARKET_PK not found")
        if not self.api_key: logger.error("POLYMARKET_API_KEY not found")
        logger.info(f"Polymarket Client [{'TESTNET' if testnet else 'MAINNET'}] chain={chain_id}")

    def _init_clob_client(self):
        """Initialize the CLOB client with credentials."""
        host = "https://clob-testnet.polymarket.com" if self.testnet else "https://clob.polymarket.com"
        self.client = ClobClient(
            host=host, key=self.private_key, chain_id=self.chain_id,
            signature_type=1, funder=os.getenv("POLYMARKET_FUNDER"))
        self.client.set_api_creds(
            api_key=self.api_key, api_secret=self.api_secret,
            api_passphrase=self.api_passphrase)

    async def connect(self) -> bool:
        """Connect to Polymarket API. Returns True if successful."""
        if not POLYMARKET_AVAILABLE or not self.private_key or not self.api_key:
            logger.error("Cannot connect: missing SDK or credentials"); return False
        try:
            self._init_clob_client()
            balance = await self._get_balance_internal()
            if balance is not None:
                self._connected = True
                logger.info(f"✓ Connected (${balance.get('USDC', 0):.2f} USDC)")
                return True
            logger.error("Failed to verify connection"); return False
        except Exception as e:
            logger.error(f"Connection failed: {e}"); return False

    async def disconnect(self) -> None:
        """Disconnect from API."""
        self._connected = False
        self.client = None
        logger.info("Disconnected from Polymarket")
    
    async def get_btc_market(self) -> Optional[Dict[str, Any]]:
        """Get BTC prediction market details."""
        if not self.client:
            return None
        try:
            # TODO: Implement actual market search
            logger.warning("BTC market lookup not fully implemented")
            return {"condition_id": "BTC_PRICE_PREDICTION", "market_id": "btc_market",
                    "question": "Will BTC be above $65000?", "end_date": "2026-03-01"}
        except Exception as e:
            logger.error(f"Error fetching BTC market: {e}"); return None

    async def get_market_price(self, token_id: str) -> Optional[Decimal]:
        """
        Get current market price for a token.
        
        Args:
            token_id: Token ID (outcome token)
            
        Returns:
            Current price (0-1 for binary markets)
        """
        if not self.client:
            return None
        
        try:
            # Get order book
            book = self.client.get_order_book(token_id)
            
            if book and "bids" in book and len(book["bids"]) > 0:
                # Best bid price
                best_bid = Decimal(str(book["bids"][0]["price"]))
                return best_bid
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching market price: {e}")
            return None
    
    async def get_orderbook(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Get order book for token."""
        if not self.client:
            return None
        try:
            book = self.client.get_order_book(token_id)
            parse = lambda levels: [{"price": Decimal(str(l["price"])), "size": Decimal(str(l["size"]))} for l in levels]
            return {"timestamp": datetime.now(), "token_id": token_id,
                    "bids": parse(book.get("bids", [])), "asks": parse(book.get("asks", []))}
        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}"); return None

    async def _resolve_price(self, token_id, side, price):
        """Resolve order price - use given or fetch from orderbook."""
        if price is not None:
            return price
        book = await self.get_orderbook(token_id)
        if not book:
            return None
        if side.lower() == "buy":
            return book["asks"][0]["price"] if book["asks"] else Decimal("0.5")
        return book["bids"][0]["price"] if book["bids"] else Decimal("0.5")

    async def place_order(self, token_id: str, side: str, size: Decimal,
                          price: Optional[Decimal] = None,
                          order_type: str = "GTC") -> Optional[str]:
        """Place order on market. Returns order ID if successful."""
        if not self.client:
            logger.error("Client not connected"); return None
        try:
            poly_side = BUY if side.lower() == "buy" else SELL
            resolved_price = await self._resolve_price(token_id, side, price)
            if resolved_price is None:
                logger.error("Cannot resolve price"); return None
            order_args = OrderArgs(token_id=token_id, price=float(resolved_price),
                                   size=float(size), side=poly_side, fee_rate_bps=0)
            signed = self.client.create_order(order_args)
            resp = self.client.post_order(signed, order_type=order_type)
            if resp and "orderID" in resp:
                oid = resp["orderID"]
                logger.info(f"Order: {oid} {side.upper()} {size} @ {resolved_price:.4f}")
                return oid
            logger.error(f"Order failed: {resp}"); return None
        except Exception as e:
            logger.error(f"Error placing order: {e}"); return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        if not self.client:
            return False
        
        try:
            response = self.client.cancel_order(order_id)
            
            if response:
                logger.info(f"Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders."""
        if not self.client: return []
        try:
            return [{"order_id": o["id"], "token_id": o["token_id"], "side": o["side"],
                     "price": Decimal(str(o["price"])), "size": Decimal(str(o["size"])),
                     "filled": Decimal(str(o.get("size_matched", 0))),
                     "timestamp": datetime.fromisoformat(o["created_at"])}
                    for o in self.client.get_orders() if o.get("status") == "live"]
        except Exception as e:
            logger.error(f"Open orders error: {e}"); return []

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of positions
        """
        if not self.client:
            return []
        
        try:
            # Get balance of outcome tokens
            balances = self.client.get_balances()
            
            positions = []
            for token_id, balance in balances.items():
                if token_id != "USDC" and float(balance) > 0:
                    positions.append({
                        "token_id": token_id,
                        "size": Decimal(str(balance)),
                        "timestamp": datetime.now(),
                    })
            
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    async def _get_balance_internal(self) -> Optional[Dict[str, Decimal]]:
        """Internal method to get balance."""
        if not self.client:
            return None
        
        try:
            balances = self.client.get_balances()
            
            return {
                token: Decimal(str(amount))
                for token, amount in balances.items()
            }
            
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None
    
    async def get_balance(self) -> Dict[str, Decimal]:
        """
        Get account balance.
        
        Returns:
            Balance dict with USDC and token balances
        """
        return await self._get_balance_internal() or {}
    
    async def get_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        if not self.client: return []
        try:
            return [{"trade_id": t["id"], "order_id": t["order_id"],
                     "token_id": t["asset_id"], "side": t["side"],
                     "price": Decimal(str(t["price"])), "size": Decimal(str(t["size"])),
                     "timestamp": datetime.fromisoformat(t["timestamp"])}
                    for t in self.client.get_trades()[:limit]]
        except Exception as e:
            logger.error(f"Trades error: {e}"); return []

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self.client is not None


# Singleton instance
_polymarket_client_instance = None

def get_polymarket_client(
    testnet: bool = False,
    force_new: bool = False,
) -> PolymarketClient:
    """
    Get singleton Polymarket client.
    
    Args:
        testnet: Use testnet mode
        force_new: Force creation of new instance
    """
    global _polymarket_client_instance
    
    if _polymarket_client_instance is None or force_new:
        _polymarket_client_instance = PolymarketClient(testnet=testnet)
    
    return _polymarket_client_instance