"""
Position Manager — owns position lifecycle (creation, SL/TP monitoring, closure).

SRP: This class handles ONLY position tracking, P&L computation, and exit
     monitoring. It delegates order submission to OrderManager.

DIP: Depends on IRiskEngine protocol, not concrete RiskEngine.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional
from loguru import logger

from interfaces import IRiskEngine
from execution.execution_engine import Order, OrderSide, OrderStatus
from execution.order_manager import OrderManager


class ExecPositionManager:
    """Creates, monitors, and closes positions using OrderManager."""

    def __init__(self, order_mgr: OrderManager, risk_engine: IRiskEngine):
        self._orders = order_mgr
        self._risk = risk_engine
        self._positions: Dict[str, Dict[str, Any]] = {}
        self.on_position_opened: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None

    async def open_from_order(self, order: Order, entry_price: Decimal):
        """Create a position from a filled order."""
        pid = f"pos_{datetime.now().timestamp()}"
        direction = "long" if order.side == OrderSide.BUY else "short"
        pos = {
            "position_id": pid, "order_id": order.order_id, "direction": direction,
            "entry_price": entry_price, "size": order.filled_size,
            "entry_time": datetime.now(), "stop_loss": order.stop_loss,
            "take_profit": order.take_profit, "status": "open", "metadata": order.metadata}
        self._positions[pid] = pos
        order.position_id = pid
        self._risk.add_position(
            position_id=pid, size=order.filled_size, entry_price=entry_price,
            direction=direction, stop_loss=order.stop_loss, take_profit=order.take_profit)
        logger.info(f"Position opened: {pid} {direction.upper()} ${order.filled_size:.2f} @ ${entry_price:.2f}")
        if self.on_position_opened:
            await self.on_position_opened(pos)

    async def close(self, pid: str, exit_price: Decimal,
                    reason: str = "manual") -> Optional[Decimal]:
        """Close position. Returns P&L or None."""
        pos = self._positions.get(pid)
        if not pos:
            logger.error(f"Position not found: {pid}"); return None
        side = OrderSide.SELL if pos["direction"] == "long" else OrderSide.BUY
        close_order = await self._orders.place_market(
            side=side, size=pos["size"],
            metadata={"position_id": pid, "close_reason": reason})
        if not close_order:
            return None
        if self._orders.dry_run:
            close_order.status = OrderStatus.FILLED
            close_order.filled_size = pos["size"]
            close_order.filled_price = exit_price
        pnl = self._risk.remove_position(pid, exit_price)
        pos.update({"status": "closed", "exit_price": exit_price,
                    "exit_time": datetime.now(), "pnl": pnl, "close_reason": reason})
        logger.info(f"Position closed: {pid} P&L=${pnl:+.2f} ({reason})")
        if self.on_position_closed:
            await self.on_position_closed(pos)
        return pnl

    async def check_exits(self, current_price: Decimal):
        """Check all open positions for SL/TP exits."""
        import operator
        for pid, pos in list(self._positions.items()):
            if pos["status"] != "open":
                continue
            self._risk.update_position(pid, current_price)
            if self._should_exit(pos, current_price, "stop_loss", operator.le, operator.ge):
                await self.close(pid, current_price, "stop_loss"); continue
            if self._should_exit(pos, current_price, "take_profit", operator.ge, operator.le):
                await self.close(pid, current_price, "take_profit")

    def open_positions(self) -> List[Dict[str, Any]]:
        return [p for p in self._positions.values() if p["status"] == "open"]

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        return self._positions.get(pid)

    @staticmethod
    def _should_exit(pos, price, key, long_cmp, short_cmp) -> bool:
        lvl = pos.get(key)
        if not lvl:
            return False
        return long_cmp(price, lvl) if pos["direction"] == "long" else short_cmp(price, lvl)

