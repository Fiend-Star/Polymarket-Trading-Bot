"""
Order Manager — owns order lifecycle (creation, submission, fill tracking).

SRP: This class handles ONLY order construction, ID generation, submission
     (live or dry-run), and fill simulation. It knows nothing about positions.

OCP: New order types (e.g. TWAP, iceberg) are added via new methods;
     existing callers are untouched.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Callable, Dict, Optional, Any
from loguru import logger

from execution.execution_engine import Order, OrderType, OrderSide, OrderStatus


class OrderManager:
    """Creates, submits, and tracks orders."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._orders: Dict[str, Order] = {}
        self._counter = 0
        self.total = self.filled = self.rejected = 0
        self.on_order_filled: Optional[Callable] = None

    def _gen_id(self) -> str:
        self._counter += 1
        return f"order_{self._counter}_{datetime.now().timestamp()}"

    async def place_market(self, side: OrderSide, size: Decimal,
                           stop_loss: Optional[Decimal] = None,
                           take_profit: Optional[Decimal] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Optional[Order]:
        """Place a market order. Returns Order or None."""
        order = Order(
            order_id=self._gen_id(), timestamp=datetime.now(),
            order_type=OrderType.MARKET, side=side, size=size, price=None,
            status=OrderStatus.PENDING, stop_loss=stop_loss,
            take_profit=take_profit, metadata=metadata or {})
        self._orders[order.order_id] = order
        self.total += 1
        logger.info(f"Market order: {order.order_id} {side.value.upper()} ${size:.2f}")
        if not self.dry_run:
            if not await self._submit_live(order, side, size, metadata):
                return None
        else:
            order.status = OrderStatus.SUBMITTED
        return order

    async def simulate_fill(self, order: Order, fill_price: Decimal):
        """Simulate immediate fill (dry-run mode)."""
        order.status = OrderStatus.FILLED
        order.filled_size = order.size
        order.filled_price = fill_price
        order.fills.append({"timestamp": datetime.now(), "price": fill_price, "size": order.size})
        self.filled += 1
        if self.on_order_filled:
            await self.on_order_filled(order)

    def get(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def pending_count(self) -> int:
        return sum(1 for o in self._orders.values() if o.status == OrderStatus.PENDING)

    async def _submit_live(self, order, side, size, metadata) -> bool:
        try:
            from execution.nautilus_polymarket_integration import get_polymarket_integration
            integration = get_polymarket_integration(simulation_mode=False)
            oid = await integration.place_market_order(side=side.value, size_usd=size, metadata=metadata)
            if oid:
                order.status = OrderStatus.SUBMITTED
                order.metadata["polymarket_order_id"] = oid
                return True
            order.status = OrderStatus.REJECTED
            return False
        except Exception as e:
            logger.error(f"Polymarket submission failed: {e}")
            order.status = OrderStatus.REJECTED
            return False


