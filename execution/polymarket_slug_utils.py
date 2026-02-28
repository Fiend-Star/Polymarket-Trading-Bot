"""
Polymarket slug utilities and read-only integration queries.

SRP: Market slug generation and account/position queries.
"""
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, Optional
from loguru import logger


def current_btc_15m_slug() -> str:
    """Get the current BTC 15-minute market slug."""
    now = datetime.now(timezone.utc)
    start = math.floor(int(now.timestamp()) / 900) * 900
    slug = f"btc-updown-15m-{start}"
    logger.info(f"Current BTC 15-min slug: {slug}")
    return slug


def get_next_btc_15m_markets(count: int = 3) -> list:
    """Get next N BTC 15-minute market slugs (current + future)."""
    now = datetime.now(timezone.utc)
    start = math.floor(int(now.timestamp()) / 900) * 900
    slugs = [f"btc-updown-15m-{start + i * 900}" for i in range(count)]
    logger.info(f"BTC 15-min slugs (next {count}): {slugs}")
    return slugs


class IntegrationQueryMixin:
    """Read-only query methods for PolymarketBTCIntegration."""

    def _calculate_token_qty(self, size_usd, price, precision):
        qty = float(size_usd) / float(price) if price > 0 else float(size_usd) * 2
        return round(qty, precision)

    def _get_current_price(self):
        quote = self.node.cache.quote(self.btc_instrument_id)
        if not quote:
            return Decimal("0.5")
        return (quote.bid_price + quote.ask_price) / 2

    def get_open_positions(self) -> list:
        if not self.node:
            return []
        return list(self.node.cache.positions_open())

    def get_balance(self) -> Dict[str, Any]:
        if not self.node:
            return {"USDC": 0.0}
        acct = self.node.cache.account(self.node.trader.id.get_tag())
        if not acct:
            return {"USDC": 0.0}
        return {
            "USDC": float(acct.balance_total().as_decimal()),
            "free": float(acct.balance_free().as_decimal()),
            "locked": float(acct.balance_locked().as_decimal()),
        }

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "simulation_mode": self.simulation_mode,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "instrument_id": str(self.btc_instrument_id) if self.btc_instrument_id else None,
            "node_running": self.node is not None,
        }

