"""
Execution Engine
Manages order placement, fills, and position lifecycle
"""
import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
from loguru import logger

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from execution.risk_engine import get_risk_engine, RiskEngine
from core.strategy_brain.signal_processors.base_processor import SignalDirection


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Trading order."""
    order_id: str
    timestamp: datetime
    order_type: OrderType
    side: OrderSide
    size: Decimal  # USD amount
    price: Optional[Decimal]  # None for market orders
    status: OrderStatus
    
    # Position management
    position_id: Optional[str] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    
    # Execution details
    filled_size: Decimal = Decimal("0")
    filled_price: Optional[Decimal] = None
    fills: List[Dict[str, Any]] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.fills is None:
            self.fills = []
        if self.metadata is None:
            self.metadata = {}


class ExecutionEngine:
    """
    Execution engine that manages order lifecycle.
    
    Workflow:
    1. Receive trading signal from strategy
    2. Check risk limits
    3. Calculate position size
    4. Place order
    5. Monitor fills
    6. Manage position
    7. Handle exits (stop loss, take profit)
    """
    
    def __init__(self, risk_engine: Optional[RiskEngine] = None, dry_run: bool = True):
        """Initialize execution engine."""
        self.risk_engine = risk_engine or get_risk_engine()
        self.dry_run = dry_run
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        self._positions: Dict[str, Dict[str, Any]] = {}
        self.on_order_filled: Optional[Callable] = None
        self.on_position_opened: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
        self._total_orders = self._filled_orders = self._rejected_orders = 0
        logger.info(f"Initialized Execution Engine [{'DRY RUN' if dry_run else 'LIVE'}]")

    def _resolve_direction(self, signal_direction):
        """Resolve signal direction to order side and direction string."""
        if signal_direction == SignalDirection.BULLISH:
            return OrderSide.BUY, "long"
        elif signal_direction == SignalDirection.BEARISH:
            return OrderSide.SELL, "short"
        return None, None

    async def execute_signal(
        self, signal_direction: SignalDirection, signal_confidence: float,
        signal_score: float, current_price: Decimal,
        stop_loss: Optional[Decimal] = None, take_profit: Optional[Decimal] = None,
    ) -> Optional[Order]:
        """Execute trading signal. Returns Order if created, None if rejected."""
        logger.info(f"EXECUTING: {signal_direction.value} conf={signal_confidence:.2%} "
                    f"score={signal_score:.1f} price=${current_price:,.2f}")
        position_size = self.risk_engine.calculate_position_size(
            signal_confidence=signal_confidence, signal_score=signal_score,
            current_price=current_price)
        side, direction = self._resolve_direction(signal_direction)
        if side is None:
            logger.warning("Neutral signal - no trade"); return None
        is_valid, error = self.risk_engine.validate_new_position(
            size=position_size, direction=direction, current_price=current_price)
        if not is_valid:
            logger.error(f"Rejected by risk engine: {error}")
            self._rejected_orders += 1; return None
        order = await self.place_market_order(
            side=side, size=position_size, stop_loss=stop_loss, take_profit=take_profit,
            metadata={"signal_direction": signal_direction.value,
                      "signal_confidence": signal_confidence, "signal_score": signal_score})
        if order and self.dry_run:
            await self._simulate_fill(order, current_price)
        return order

    def _generate_order_id(self):
        """Generate unique order ID."""
        self._order_counter += 1
        return f"order_{self._order_counter}_{datetime.now().timestamp()}"

    async def _submit_to_polymarket(self, order, side, size, metadata):
        """Submit order to Polymarket via Nautilus. Returns True on success."""
        try:
            from execution.nautilus_polymarket_integration import get_polymarket_integration
            integration = get_polymarket_integration(simulation_mode=False)
            oid = await integration.place_market_order(
                side=side.value, size_usd=size, metadata=metadata)
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

    async def place_market_order(
        self, side: OrderSide, size: Decimal,
        stop_loss: Optional[Decimal] = None, take_profit: Optional[Decimal] = None,
        metadata: Dict[str, Any] = None,
    ) -> Optional[Order]:
        """Place market order. Returns Order object."""
        order = Order(
            order_id=self._generate_order_id(), timestamp=datetime.now(),
            order_type=OrderType.MARKET, side=side, size=size, price=None,
            status=OrderStatus.PENDING, stop_loss=stop_loss,
            take_profit=take_profit, metadata=metadata or {})
        self._orders[order.order_id] = order
        self._total_orders += 1
        logger.info(f"Market order: {order.order_id} {side.value.upper()} ${size:.2f}")
        if not self.dry_run:
            if not await self._submit_to_polymarket(order, side, size, metadata):
                return None
        else:
            order.status = OrderStatus.SUBMITTED
        return order

    async def _simulate_fill(self, order: Order, fill_price: Decimal) -> None:
        """Simulate order fill (dry run mode)."""
        logger.info(f"[SIM] Fill {order.order_id} @ ${fill_price:.2f}")
        order.status = OrderStatus.FILLED
        order.filled_size = order.size
        order.filled_price = fill_price
        order.fills.append({"timestamp": datetime.now(), "price": fill_price, "size": order.size})
        self._filled_orders += 1
        await self._create_position(order, fill_price)
        if self.on_order_filled:
            await self.on_order_filled(order)

    async def _create_position(self, order: Order, entry_price: Decimal) -> None:
        """Create position from filled order."""
        pid = f"pos_{datetime.now().timestamp()}"
        direction = "long" if order.side == OrderSide.BUY else "short"
        position = {
            "position_id": pid, "order_id": order.order_id, "direction": direction,
            "entry_price": entry_price, "size": order.filled_size,
            "entry_time": datetime.now(), "stop_loss": order.stop_loss,
            "take_profit": order.take_profit, "status": "open", "metadata": order.metadata}
        self._positions[pid] = position
        order.position_id = pid
        self.risk_engine.add_position(
            position_id=pid, size=order.filled_size, entry_price=entry_price,
            direction=direction, stop_loss=order.stop_loss, take_profit=order.take_profit)
        logger.info(f"Position opened: {pid} {direction.upper()} ${order.filled_size:.2f} @ ${entry_price:.2f}")
        if self.on_position_opened:
            await self.on_position_opened(position)

    def _should_exit(self, position, current_price, level_key, long_cmp, short_cmp):
        """Check if position should exit at given level."""
        level = position.get(level_key)
        if not level:
            return False
        if position["direction"] == "long":
            return long_cmp(current_price, level)
        return short_cmp(current_price, level)

    async def close_position(self, position_id: str, exit_price: Decimal,
                             reason: str = "manual") -> Optional[Decimal]:
        """Close a position. Returns realized P&L or None."""
        if position_id not in self._positions:
            logger.error(f"Position not found: {position_id}"); return None
        position = self._positions[position_id]
        side = OrderSide.SELL if position["direction"] == "long" else OrderSide.BUY
        close_order = await self.place_market_order(
            side=side, size=position["size"],
            metadata={"position_id": position_id, "close_reason": reason})
        if not close_order:
            return None
        if self.dry_run:
            close_order.status = OrderStatus.FILLED
            close_order.filled_size = position["size"]
            close_order.filled_price = exit_price
        pnl = self.risk_engine.remove_position(position_id, exit_price)
        position.update({"status": "closed", "exit_price": exit_price,
                        "exit_time": datetime.now(), "pnl": pnl, "close_reason": reason})
        logger.info(f"Position closed: {position_id} P&L: ${pnl:+.2f} ({reason})")
        if self.on_position_closed:
            await self.on_position_closed(position)
        return pnl

    async def update_positions(self, current_price: Decimal) -> None:
        """Update all open positions — check stop loss and take profit."""
        import operator
        for pid, pos in list(self._positions.items()):
            if pos["status"] != "open":
                continue
            self.risk_engine.update_position(pid, current_price)
            if self._should_exit(pos, current_price, "stop_loss", operator.le, operator.ge):
                logger.warning(f"Stop loss hit: {pid}")
                await self.close_position(pid, current_price, "stop_loss"); continue
            if self._should_exit(pos, current_price, "take_profit", operator.ge, operator.le):
                logger.info(f"Take profit hit: {pid}")
                await self.close_position(pid, current_price, "take_profit")

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get position by ID."""
        return self._positions.get(position_id)
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        return [
            pos for pos in self._positions.values()
            if pos["status"] == "open"
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "mode": "dry_run" if self.dry_run else "live",
            "orders": {
                "total": self._total_orders,
                "filled": self._filled_orders,
                "rejected": self._rejected_orders,
                "pending": len([o for o in self._orders.values() if o.status == OrderStatus.PENDING]),
            },
            "positions": {
                "open": len(self.get_open_positions()),
                "total": len(self._positions),
            },
            "risk": self.risk_engine.get_risk_summary(),
        }


# Singleton instance
_execution_engine_instance = None

def get_execution_engine() -> ExecutionEngine:
    """Get singleton execution engine."""
    global _execution_engine_instance
    if _execution_engine_instance is None:
        _execution_engine_instance = ExecutionEngine(dry_run=True)
    return _execution_engine_instance