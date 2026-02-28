"""
OrderDispatcher — owns order construction and submission.

SRP: This class's sole responsibility is building and dispatching orders
     (paper, limit, market) to the Nautilus order factory.  It knows nothing
     about *when* to trade — that decision lives in the strategy.

OCP: New order types are added by subclassing or adding a new dispatch method;
     existing callers are not modified.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import List
from loguru import logger

from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.objects import Quantity, Price


@dataclass
class PaperTrade:
    """Immutable record of a simulated trade."""
    timestamp: datetime
    direction: str
    size_usd: float
    price: float
    signal_score: float
    signal_confidence: float
    outcome: str = "PENDING"

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "size_usd": self.size_usd,
            "price": self.price,
            "signal_score": self.signal_score,
            "signal_confidence": self.signal_confidence,
            "outcome": self.outcome,
        }


class OrderDispatcher:
    """Builds and dispatches limit / market / paper orders."""

    def __init__(self, strategy_ref, position_tracker):
        """
        Args:
            strategy_ref: Nautilus Strategy (owns order_factory, cache, submit_order)
            position_tracker: PositionTracker for recording positions
        """
        self._strategy = strategy_ref
        self._positions = position_tracker
        self.paper_trades: List[PaperTrade] = []

    # ── Paper trade ──────────────────────────────────────────────────────

    def record_paper_trade(self, signal, size: Decimal, price: Decimal,
                           direction: str, market: dict):
        """Record a paper trade and open a tracked position."""
        ts_ms = int(time.time() * 1000)
        oid = f"paper_{ts_ms}"
        iid = (self._strategy._yes_iid if direction == "long"
               else self._strategy._no_iid or self._strategy._yes_iid)

        self._positions.add(
            market_slug=market["slug"], direction=direction,
            entry_price=float(price), size_usd=float(size),
            market_end_time=market["end_time"],
            instrument_id=iid, order_id=oid)

        self.paper_trades.append(PaperTrade(
            timestamp=datetime.now(timezone.utc), direction=direction.upper(),
            size_usd=float(size), price=float(price),
            signal_score=signal.score, signal_confidence=signal.confidence))

        token = "YES (UP)" if direction == "long" else "NO (DOWN)"
        logger.info(f"[SIM] Paper {direction.upper()} {token} ${float(size):.2f} "
                    f"@ ${float(price):.4f} | {market['slug']}")
        self._save()

    # ── Limit order ──────────────────────────────────────────────────────

    def place_limit(self, direction: str, size: Decimal, market: dict):
        """Place a LIMIT order at best bid (maker, 0 % fee)."""
        resolved = self._resolve_instrument(direction)
        if not resolved:
            return
        iid, label, inst = resolved
        bid, ask = self._get_book(iid, size)
        lp = max(Decimal("0.01"), min(Decimal("0.99"), bid))
        qty = self._qty(size, lp, inst)

        logger.info(f"LIMIT {label}: {qty:.2f} tok @ ${float(lp):.4f} "
                    f"(spread=${float(ask - bid):.4f})")

        order = self._strategy.order_factory.limit(
            instrument_id=iid, order_side=OrderSide.BUY,
            quantity=Quantity(qty, precision=inst.size_precision),
            price=Price(float(lp), precision=inst.price_precision),
            client_order_id=ClientOrderId(f"BTC-15MIN-V3-{int(time.time()*1000)}"),
            time_in_force=TimeInForce.GTC)
        self._strategy.submit_order(order)

        self._positions.add(
            market_slug=market["slug"], direction=direction,
            entry_price=float(lp), size_usd=float(size),
            market_end_time=market["end_time"],
            instrument_id=iid, order_id=str(order.client_order_id))

    # ── Market order ─────────────────────────────────────────────────────

    def place_market(self, direction: str, size: Decimal,
                     price: float, market: dict):
        """Place a MARKET order (taker, 10 % fee)."""
        resolved = self._resolve_instrument(direction)
        if not resolved:
            return
        iid, label, inst = resolved
        prec = inst.size_precision
        qty = round(max(float(getattr(inst, "min_quantity", None) or 5.0), 5.0), prec)
        uid = f"BTC-15MIN-MKT-{int(time.time()*1000)}"
        order = self._strategy.order_factory.market(
            instrument_id=iid, order_side=OrderSide.BUY,
            quantity=Quantity(qty, precision=prec),
            client_order_id=ClientOrderId(uid),
            quote_quantity=False, time_in_force=TimeInForce.IOC)
        self._strategy.submit_order(order)
        logger.info(f"MARKET {label}: {uid}")
        self._positions.add(
            market_slug=market["slug"], direction=direction,
            entry_price=price, size_usd=float(size),
            market_end_time=market["end_time"],
            instrument_id=iid, order_id=uid)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_instrument(self, direction):
        s = self._strategy
        if direction == "long":
            iid = getattr(s, "_yes_iid", s.instrument_id)
            label = "YES (UP)"
        else:
            iid = getattr(s, "_no_iid", None)
            if iid is None:
                logger.warning("NO token not found"); return None
            label = "NO (DOWN)"
        inst = s.cache.instrument(iid)
        if not inst:
            logger.error(f"Instrument not in cache: {iid}"); return None
        return iid, label, inst

    def _get_book(self, iid, fallback_price):
        try:
            q = self._strategy.cache.quote_tick(iid)
            if q:
                return q.bid_price.as_decimal(), q.ask_price.as_decimal()
        except Exception:
            pass
        return fallback_price, fallback_price + Decimal("0.02")

    def _qty(self, size, price, inst):
        raw = float(size / price)
        return max(round(raw, inst.size_precision), 5.0)

    def _save(self):
        try:
            with open("paper_trades.json", "w") as f:
                json.dump([t.to_dict() for t in self.paper_trades], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save paper trades: {e}")

    def update_paper_outcome(self, direction: str, entry_price: float, outcome: str):
        """Mark matching paper trade as resolved."""
        for pt in self.paper_trades:
            if (pt.outcome == "PENDING" and pt.direction == direction.upper()
                    and abs(pt.price - entry_price) < 0.0001):
                pt.outcome = outcome
                break
        self._save()


