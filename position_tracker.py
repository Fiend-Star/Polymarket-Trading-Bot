"""
PositionTracker — owns the lifecycle of open positions and P&L resolution.

SRP: This class's sole responsibility is tracking positions from entry to
     resolution at market end, computing P&L, and recording results.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional
from loguru import logger

from interfaces import IPerformanceTracker, IMetricsExporter


@dataclass
class OpenPosition:
    """Tracks a live position until market resolves."""
    market_slug: str
    direction: str  # "long" (bought YES) or "short" (bought NO)
    entry_price: float
    size_usd: float
    entry_time: datetime
    market_end_time: datetime
    instrument_id: object
    order_id: str
    actual_qty: float = 0.0  # Actual filled quantity
    resolved: bool = False
    exit_price: Optional[float] = None
    pnl: Optional[float] = None


class PositionTracker:
    """Tracks open positions and resolves them at market end."""

    def __init__(self, cache_ref, performance: IPerformanceTracker,
                 metrics: Optional[IMetricsExporter] = None):
        """
        Args:
            cache_ref: Nautilus cache (for reading final quote ticks)
            performance: performance tracker to record resolved trades
            metrics: optional Grafana exporter for trade counters
        """
        self._cache = cache_ref
        self._performance = performance
        self._metrics = metrics
        self.positions: List[OpenPosition] = []

    # ── Public API ───────────────────────────────────────────────────────

    def add(self, market_slug: str, direction: str, entry_price: float,
            size_usd: float, market_end_time: datetime,
            instrument_id, order_id: str):
        """Record a new open position."""
        self.positions.append(OpenPosition(
            market_slug=market_slug, direction=direction,
            entry_price=entry_price, size_usd=size_usd,
            entry_time=datetime.now(timezone.utc),
            market_end_time=market_end_time,
            instrument_id=instrument_id, order_id=order_id))

    def resolve_market(self, slug: str):
        """Resolve all open positions for a given market slug."""
        for pos in self.positions:
            if pos.market_slug == slug and not pos.resolved:
                self._resolve(pos)

    def update_entry_price(self, order_id: str, price: float):
        """Update a position's entry price (on fill)."""
        for pos in self.positions:
            if pos.order_id == order_id:
                pos.entry_price = price
                return pos
        return None

    def unresolved(self) -> List[OpenPosition]:
        """Return all unresolved positions."""
        return [p for p in self.positions if not p.resolved]

    # ── Private ──────────────────────────────────────────────────────────

    def _resolve(self, pos: OpenPosition):
        final = self._get_final_price(pos)
        if final is None:
            logger.warning(f"Cannot resolve {pos.market_slug} — no final price")
            return
        pnl = pos.size_usd * (final - pos.entry_price) / pos.entry_price
        pos.resolved, pos.exit_price, pos.pnl = True, final, pnl
        outcome = "WIN" if pnl > 0 else "LOSS"
        is_paper = pos.order_id.startswith("paper_")
        tag = "[PAPER] " if is_paper else ""
        token = "YES" if pos.direction == "long" else "NO"
        logger.info(f"📊 {tag}RESOLVED: {pos.market_slug} {token} "
                    f"${pos.entry_price:.4f}→${final:.4f} P&L=${pnl:+.4f} ({outcome})")
        self._record(pos, final, is_paper, outcome)
        if self._metrics:
            self._metrics.increment_trade_counter(won=(pnl > 0))

    def _get_final_price(self, pos) -> Optional[float]:
        try:
            q = self._cache.quote_tick(pos.instrument_id)
            if q:
                return float((q.bid_price.as_decimal() + q.ask_price.as_decimal()) / 2)
        except Exception:
            pass
        return None

    def _record(self, pos, final_price, is_paper, outcome):
        self._performance.record_trade(
            trade_id=pos.order_id, direction="long",
            entry_price=Decimal(str(pos.entry_price)),
            exit_price=Decimal(str(final_price)),
            size=Decimal(str(pos.size_usd)),
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc),
            signal_score=0, signal_confidence=0,
            metadata={
                "resolved": True, "market": pos.market_slug,
                "paper": is_paper, "original_direction": pos.direction,
                "token": "YES" if pos.direction == "long" else "NO",
                "pnl_computed": round(pos.pnl, 4),
            })


