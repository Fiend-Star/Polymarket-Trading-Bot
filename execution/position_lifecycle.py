"""
PositionLifecycleMixin — position exit logic, queries, statistics.

SRP: Position monitoring and close logic (extracted from ExecutionEngine).
"""
import operator
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List, Optional
from loguru import logger


class PositionLifecycleMixin:
    """Mixin for position exit monitoring, closing, and statistics queries."""

    def _should_exit(self, position, current_price, level_key, long_cmp, short_cmp):
        level = position.get(level_key)
        if not level:
            return False
        if position["direction"] == "long":
            return long_cmp(current_price, level)
        return short_cmp(current_price, level)

    async def close_position(self, position_id: str, exit_price: Decimal,
                             reason: str = "manual") -> Optional[Decimal]:
        """Close a position. Returns realized P&L or None."""
        from execution.execution_engine import OrderSide, OrderStatus
        if position_id not in self._positions:
            logger.error(f"Position not found: {position_id}"); return None
        pos = self._positions[position_id]
        side = OrderSide.SELL if pos["direction"] == "long" else OrderSide.BUY
        co = await self.place_market_order(
            side=side, size=pos["size"],
            metadata={"position_id": position_id, "close_reason": reason})
        if not co:
            return None
        if self.dry_run:
            co.status = OrderStatus.FILLED
            co.filled_size = pos["size"]
            co.filled_price = exit_price
        pnl = self.risk_engine.remove_position(position_id, exit_price)
        pos.update({"status": "closed", "exit_price": exit_price,
                    "exit_time": datetime.now(), "pnl": pnl, "close_reason": reason})
        logger.info(f"Position closed: {position_id} P&L=${pnl:+.2f} ({reason})")
        if self.on_position_closed:
            await self.on_position_closed(pos)
        return pnl

    async def update_positions(self, current_price: Decimal) -> None:
        """Check stop loss and take profit on all open positions."""
        for pid, pos in list(self._positions.items()):
            if pos["status"] != "open":
                continue
            self.risk_engine.update_position(pid, current_price)
            if self._should_exit(pos, current_price, "stop_loss", operator.le, operator.ge):
                logger.warning(f"Stop loss: {pid}")
                await self.close_position(pid, current_price, "stop_loss")
                continue
            if self._should_exit(pos, current_price, "take_profit", operator.ge, operator.le):
                logger.info(f"Take profit: {pid}")
                await self.close_position(pid, current_price, "take_profit")

    def get_order(self, order_id: str):
        return self._orders.get(order_id)

    def get_position(self, position_id: str):
        return self._positions.get(position_id)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        return [p for p in self._positions.values() if p["status"] == "open"]

    def get_statistics(self) -> Dict[str, Any]:
        from execution.execution_engine import OrderStatus
        return {
            "mode": "dry_run" if self.dry_run else "live",
            "orders": {"total": self._total_orders, "filled": self._filled_orders,
                       "rejected": self._rejected_orders,
                       "pending": len([o for o in self._orders.values()
                                       if o.status == OrderStatus.PENDING])},
            "positions": {"open": len(self.get_open_positions()),
                          "total": len(self._positions)},
            "risk": self.risk_engine.get_risk_summary(),
        }

