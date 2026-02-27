import os
import asyncio
import math
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from loguru import logger
from dotenv import load_dotenv

from nautilus_trader.adapters.polymarket import POLYMARKET
from nautilus_trader.adapters.polymarket import (
    PolymarketDataClientConfig,
    PolymarketExecClientConfig,
)
from nautilus_trader.adapters.polymarket.factories import (
    PolymarketLiveDataClientFactory,
    PolymarketLiveExecClientFactory,
)
from nautilus_trader.adapters.polymarket.providers import PolymarketInstrumentProviderConfig
from nautilus_trader.config import (
    LiveDataEngineConfig,
    LiveExecEngineConfig,
    LiveRiskEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.objects import Quantity, Price
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId
from nautilus_trader.trading.strategy import Strategy

load_dotenv()


def current_btc_15m_slug() -> str:
    """
    Get the current BTC 15-minute market slug.
    
    Polymarket BTC 15-min markets follow the pattern:
    btc-updown-15m-{unix_timestamp}
    
    Where unix_timestamp is the start of the 15-minute interval.
    
    Returns:
        Current market slug (e.g., "btc-updown-15m-1739461800")
    """
    now = datetime.now(timezone.utc)
    unix_s = int(now.timestamp())
    interval_start = math.floor(unix_s / 900) * 900  # 900 = 15×60
    slug = f"btc-updown-15m-{interval_start}"
    
    logger.info(f"Current BTC 15-min market slug: {slug}")
    return slug


def get_next_btc_15m_markets(count: int = 3) -> list[str]:
    """
    Get the next N BTC 15-minute market slugs.
    
    Useful for pre-loading markets that will be active soon.
    
    Args:
        count: Number of future markets to include
        
    Returns:
        List of market slugs including current and future markets
    """
    now = datetime.now(timezone.utc)
    unix_s = int(now.timestamp())
    interval_start = math.floor(unix_s / 900) * 900
    
    slugs = []
    for i in range(count):
        timestamp = interval_start + (i * 900)
        slug = f"btc-updown-15m-{timestamp}"
        slugs.append(slug)
    
    logger.info(f"BTC 15-min market slugs (next {count}): {slugs}")
    return slugs


class PolymarketBTCIntegration:
    """
    Integration layer between BTC strategy and Polymarket via Nautilus.
    
    This handles:
    - Nautilus node setup
    - Polymarket client configuration
    - Instrument loading
    - Order routing
    - Position tracking
    """
    
    def __init__(
        self,
        simulation_mode: bool = True,
        btc_market_condition_id: Optional[str] = None,
    ):
        """
        Initialize Polymarket integration.
        
        Args:
            simulation_mode: If True, don't place real orders
            btc_market_condition_id: Polymarket condition ID for BTC market
        """
        self.simulation_mode = simulation_mode
        self.btc_market_condition_id = btc_market_condition_id
        
        # Nautilus components
        self.node: Optional[TradingNode] = None
        self.strategy: Optional[Strategy] = None
        
        # Track Polymarket instruments
        self.btc_instrument_id: Optional[InstrumentId] = None
        
        # Statistics
        self.orders_submitted = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        
        mode = "SIMULATION" if simulation_mode else "LIVE TRADING"
        logger.info(f"Initialized Polymarket BTC Integration [{mode}]")
    
    async def start(self) -> bool:
        """Start the Nautilus trading node with Polymarket."""
        try:
            logger.info("=" * 80 + " STARTING NAUTILUS-POLYMARKET " + "=" * 80)
            config = self._create_nautilus_config()
            self.node = TradingNode(config=config)
            self.node.add_data_client_factory(POLYMARKET, PolymarketLiveDataClientFactory)
            self.node.add_exec_client_factory(POLYMARKET, PolymarketLiveExecClientFactory)
            self.node.build()
            logger.info("✓ Node built, starting...")
            self.node.start()
            await asyncio.sleep(5)
            await self._find_btc_instrument()
            logger.info("✓ Nautilus-Polymarket integration started")
            return True
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            import traceback; traceback.print_exc()
            return False

    def _create_poly_config(self, instrument_cfg):
        """Create Polymarket client configs from environment."""
        env = {k: os.getenv(f"POLYMARKET_{k.upper()}")
               for k in ("pk", "api_key", "api_secret", "passphrase")}
        kwargs = {"private_key": env["pk"], "api_key": env["api_key"],
                  "api_secret": env["api_secret"], "passphrase": env["passphrase"],
                  "instrument_config": instrument_cfg}
        return PolymarketDataClientConfig(**kwargs), PolymarketExecClientConfig(**kwargs)

    def _create_nautilus_config(self) -> TradingNodeConfig:
        """Create Nautilus trading node configuration."""
        inst_cfg = PolymarketInstrumentProviderConfig(
            event_slug_builder="slug_builders:build_btc_15min_slugs")
        data_cfg, exec_cfg = self._create_poly_config(inst_cfg)
        return TradingNodeConfig(
            environment="live", trader_id="BTC-15MIN-BOT-001",
            logging=LoggingConfig(log_level="ERROR", log_directory="./logs/nautilus"),
            data_engine=LiveDataEngineConfig(qsize=6000),
            exec_engine=LiveExecEngineConfig(qsize=6000),
            risk_engine=LiveRiskEngineConfig(bypass=self.simulation_mode),
            data_clients={POLYMARKET: data_cfg},
            exec_clients={POLYMARKET: exec_cfg})

    async def _find_btc_instrument(self) -> bool:
        """Find the BTC 15-minute prediction market instrument."""
        if not self.node:
            return False
        instruments = self.node.cache.instruments()
        logger.info(f"Found {len(instruments)} total instruments")
        btc = [i for i in instruments if '.POLYMARKET' in str(i.id)]
        if not btc:
            logger.error("No BTC 15-min instruments found!"); return False
        self.btc_instrument_id = btc[0].id
        inst = btc[0]
        logger.info(f"✓ Using: {self.btc_instrument_id} "
                    f"(prec={inst.price_precision}/{inst.size_precision}, min={inst.min_quantity})")
        return True

    def _calculate_token_qty(self, size_usd, price, precision):
        """Calculate token quantity from USD amount and price."""
        qty = float(size_usd) / float(price) if price > 0 else float(size_usd) * 2
        return round(qty, precision)

    def _get_current_price(self):
        """Get current mid price from cache, or default 0.5."""
        quote = self.node.cache.quote(self.btc_instrument_id)
        if not quote:
            return Decimal("0.5")
        return (quote.bid_price + quote.ask_price) / 2

    async def place_market_order(self, side: str, size_usd: Decimal,
                                  metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Place market order on Polymarket. Returns order ID."""
        if not self.node or not self.btc_instrument_id:
            logger.error("Integration not ready"); return None
        if self.simulation_mode:
            return f"sim_order_{datetime.now().timestamp()}"
        try:
            instrument = self.node.cache.instrument(self.btc_instrument_id)
            if not instrument: return None
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            price = self._get_current_price()
            token_qty = self._calculate_token_qty(size_usd, price, instrument.size_precision)
            qty = Quantity(token_qty, precision=instrument.size_precision)
            ts = int(datetime.now().timestamp() * 1000)
            oid = f"BTC-15MIN-{side.upper()}-{ts}"
            order = self.node.trader.order_factory.market(
                instrument_id=self.btc_instrument_id, order_side=order_side,
                quantity=qty, client_order_id=ClientOrderId(oid),
                quote_quantity=False, time_in_force=TimeInForce.IOC)
            logger.info(f"Submitting: {order_side.name} {token_qty:.6f} tokens ~${size_usd:.2f}")
            self.node.trader.submit_order(order)
            self.orders_submitted += 1
            return oid
        except Exception as e:
            logger.error(f"Market order failed: {e}")
            self.orders_rejected += 1; return None

    async def place_limit_order(self, side: str, size_usd: Decimal,
                                 limit_price: Decimal,
                                 metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Place limit order on Polymarket. Returns order ID."""
        if not self.node or not self.btc_instrument_id:
            return None
        if self.simulation_mode:
            return f"sim_order_{datetime.now().timestamp()}"
        try:
            instrument = self.node.cache.instrument(self.btc_instrument_id)
            if not instrument: return None
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            token_qty = self._calculate_token_qty(size_usd, limit_price, instrument.size_precision)
            qty = Quantity(token_qty, precision=instrument.size_precision)
            price = Price.from_str(f"{float(limit_price):.4f}")
            ts = int(datetime.now().timestamp() * 1000)
            oid = f"BTC-15MIN-LIMIT-{ts}"
            order = self.node.trader.order_factory.limit(
                instrument_id=self.btc_instrument_id, order_side=order_side,
                quantity=qty, price=price, client_order_id=ClientOrderId(oid),
                quote_quantity=False, time_in_force=TimeInForce.GTC)
            logger.info(f"Limit: {order_side.name} {token_qty:.6f} @ ${limit_price:.4f}")
            self.node.trader.submit_order(order)
            self.orders_submitted += 1
            return oid
        except Exception as e:
            logger.error(f"Limit order failed: {e}")
            self.orders_rejected += 1; return None

    def get_open_positions(self) -> list:
        """Get open positions from Nautilus."""
        if not self.node:
            return []
        
        return list(self.node.cache.positions_open())
    
    def get_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        if not self.node:
            return {"USDC": 0.0}
        
        # Get account state from Nautilus cache
        account = self.node.cache.account(self.node.trader.id.get_tag())
        
        if not account:
            return {"USDC": 0.0}
        
        return {
            "USDC": float(account.balance_total().as_decimal()),
            "free": float(account.balance_free().as_decimal()),
            "locked": float(account.balance_locked().as_decimal()),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics."""
        return {
            "simulation_mode": self.simulation_mode,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "instrument_id": str(self.btc_instrument_id) if self.btc_instrument_id else None,
            "node_running": self.node is not None,
        }
    
    async def stop(self) -> None:
        """Stop the integration."""
        if self.node:
            logger.info("Stopping Nautilus node...")
            await self.node.stop_async()
            await self.node.dispose_async()
            self.node = None
        
        logger.info("Polymarket integration stopped")


# Singleton instance
_integration_instance: Optional[PolymarketBTCIntegration] = None

def get_polymarket_integration(
    simulation_mode: bool = True,
    btc_market_condition_id: Optional[str] = None,
) -> PolymarketBTCIntegration:
    """Get singleton Polymarket integration."""
    global _integration_instance
    
    if _integration_instance is None:
        _integration_instance = PolymarketBTCIntegration(
            simulation_mode=simulation_mode,
            btc_market_condition_id=btc_market_condition_id,
        )
    
    return _integration_instance