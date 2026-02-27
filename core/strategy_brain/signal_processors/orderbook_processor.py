"""
Order Book Imbalance Signal Processor
Reads the Polymarket CLOB order book for the current YES token and
detects when buy-side or sell-side pressure is heavily skewed.

WHY THIS WORKS:
  The Polymarket CLOB (Central Limit Order Book) shows exactly how many
  dollars are queued to buy "Up" vs sell "Up" at various price levels.

  If $800 is sitting on the bid (buy side) and only $200 on the ask
  (sell side), someone large expects BTC to go UP → follow them BULLISH.

  This is real-time, forward-looking information that reflects what
  sophisticated market participants are actually doing RIGHT NOW —
  not a lagging indicator.

API USED:
  GET https://clob.polymarket.com/book?token_id=<YES_token_id>

  Returns:
    {
      "bids": [{"price": "0.52", "size": "150"}, ...],  ← buyers of YES
      "asks": [{"price": "0.54", "size": "80"},  ...],  ← sellers of YES
    }

SIGNAL LOGIC:
  imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
  Range: -1.0 (all sellers) to +1.0 (all buyers)

  imbalance > +0.30  → BULLISH  (heavy buy pressure)
  imbalance < -0.30  → BEARISH  (heavy sell pressure)
  |imbalance| < 0.30 → no signal (balanced book)

  We also check WALL detection: a single order > 20% of total book
  volume indicates a large player taking a strong position.
"""
import httpx
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any, List
from loguru import logger

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    TradingSignal,
    SignalType,
    SignalDirection,
    SignalStrength,
)

CLOB_BASE = "https://clob.polymarket.com"


class OrderBookImbalanceProcessor(BaseSignalProcessor):
    """
    Detects order book imbalance on the Polymarket CLOB.

    Wired into the strategy by passing the YES token_id via metadata:
      metadata['yes_token_id'] = <token id string>

    This is set once per market in _load_all_btc_instruments and stored
    on the strategy as self._yes_token_id.
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.30,   # 30% skew to signal
        wall_threshold: float = 0.20,         # single order > 20% of book = wall
        min_book_volume: float = 50.0,        # ignore books with < $50 total (illiquid)
        min_confidence: float = 0.55,
        top_levels: int = 10,                 # how many price levels to consider
    ):
        super().__init__("OrderBookImbalance")

        self.imbalance_threshold = imbalance_threshold
        self.wall_threshold = wall_threshold
        self.min_book_volume = min_book_volume
        self.min_confidence = min_confidence
        self.top_levels = top_levels

        # HTTP client created fresh per request (synchronous, safe in Nautilus event loop)
        self._cache: Optional[Dict] = None

        logger.info(
            f"Initialized Order Book Imbalance Processor: "
            f"imbalance_threshold={imbalance_threshold:.0%}, "
            f"wall_threshold={wall_threshold:.0%}, "
            f"min_book_volume=${min_book_volume:.0f}"
        )

    def _get_client(self) -> httpx.Client:
        """Return a synchronous httpx client (safe inside NautilusTrader's event loop)."""
        return httpx.Client(timeout=5.0)

    def fetch_order_book(self, token_id: str) -> Optional[Dict]:
        """Fetch order book from Polymarket CLOB synchronously."""
        try:
            with self._get_client() as client:
                resp = client.get(
                    f"{CLOB_BASE}/book",
                    params={"token_id": token_id},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(f"OrderBook fetch failed for {token_id[:16]}…: {e}")
            return None

    def _parse_levels(self, levels: List[Dict]) -> float:
        """Sum total volume across price levels (returns USD volume)."""
        total = 0.0
        for level in levels[:self.top_levels]:
            try:
                price = float(level.get("price", 0))
                size = float(level.get("size", 0))
                total += price * size   # USD value at each level
            except (ValueError, TypeError):
                continue
        return total

    def _detect_wall(self, levels: List[Dict], total_volume: float) -> Optional[float]:
        """Return the size of the largest single order if it's a wall, else None."""
        if total_volume <= 0:
            return None
        for level in levels[:self.top_levels]:
            try:
                price = float(level.get("price", 0))
                size = float(level.get("size", 0))
                order_usd = price * size
                if order_usd / total_volume >= self.wall_threshold:
                    return order_usd
            except (ValueError, TypeError):
                continue
        return None

    def _calculate_book_stats(self, token_id):
        """Fetch order book and calculate volumes/imbalance. Returns dict or None."""
        book = self.fetch_order_book(token_id)
        if not book:
            return None
        bids, asks = book.get("bids", []), book.get("asks", [])
        bid_vol = self._parse_levels(bids)
        ask_vol = self._parse_levels(asks)
        total = bid_vol + ask_vol
        if total < self.min_book_volume:
            logger.debug(f"OrderBook too thin: ${total:.1f} < ${self.min_book_volume:.1f}")
            return None
        return {"bids": bids, "asks": asks, "bid_vol": bid_vol,
                "ask_vol": ask_vol, "total": total,
                "imbalance": (bid_vol - ask_vol) / total}

    def _classify_imbalance(self, abs_imb):
        """Return strength for given absolute imbalance."""
        if abs_imb >= 0.70: return SignalStrength.VERY_STRONG
        if abs_imb >= 0.50: return SignalStrength.STRONG
        if abs_imb >= 0.35: return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def _build_ob_signal(self, stats, current_price):
        """Build orderbook signal from book stats. Returns signal or None."""
        imb = stats["imbalance"]
        if abs(imb) < self.imbalance_threshold:
            return None
        direction = SignalDirection.BULLISH if imb > 0 else SignalDirection.BEARISH
        strength = self._classify_imbalance(abs(imb))
        confidence = min(0.85, 0.55 + abs(imb) * 0.40)
        wall = self._detect_wall(stats["bids" if imb > 0 else "asks"], stats["total"])
        if wall:
            confidence = min(0.90, confidence + 0.05)
        if confidence < self.min_confidence:
            return None
        meta = {"bid_volume_usd": round(stats["bid_vol"], 2),
                "ask_volume_usd": round(stats["ask_vol"], 2),
                "total_volume_usd": round(stats["total"], 2),
                "imbalance": round(imb, 4),
                "bid_wall_usd": round(self._detect_wall(stats["bids"], stats["total"]) or 0, 2) or None,
                "ask_wall_usd": round(self._detect_wall(stats["asks"], stats["total"]) or 0, 2) or None}
        sig = self._build_and_record(SignalType.VOLUME_SURGE, direction, strength, confidence, current_price, meta)
        logger.info(f"{direction.value.upper()} OrderBook: imbalance={imb:+.3f}, conf={confidence:.2%}")
        return sig

    def process(self, current_price: Decimal, historical_prices: list,
                metadata: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Fetch order book synchronously and generate signal."""
        if not self.is_enabled or not metadata:
            return None
        token_id = metadata.get("yes_token_id")
        if not token_id:
            return None
        try:
            stats = self._calculate_book_stats(token_id)
            if not stats:
                return None
            logger.info(f"OrderBook: bids=${stats['bid_vol']:.1f}, asks=${stats['ask_vol']:.1f}, "
                       f"imbalance={stats['imbalance']:+.3f}")
            return self._build_ob_signal(stats, current_price)
        except Exception as e:
            logger.warning(f"OrderBookImbalance process error: {e}")
            return None