"""
MarketManager — owns instrument discovery, subscription, and market switching.

SRP: This class's sole responsibility is managing the lifecycle of BTC 15-min
     markets: loading, pairing YES/NO tokens, activating, and switching.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger


class MarketManager:
    """Discovers, subscribes, and rotates BTC 15-min Polymarket instruments."""

    def __init__(self, strategy_ref, market_interval_seconds: int = 900):
        """
        Args:
            strategy_ref: reference to the Nautilus Strategy (for cache/subscribe)
            market_interval_seconds: length of each market window
        """
        self._strategy = strategy_ref
        self._interval = market_interval_seconds
        self.all_instruments: List[Dict] = []
        self.current_index: int = -1
        self.next_switch_time: Optional[datetime] = None
        self.waiting_for_open: bool = False
        self.yes_instrument_id = None
        self.no_instrument_id = None
        self.yes_token_id: Optional[str] = None

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def current_market(self) -> Optional[Dict]:
        if 0 <= self.current_index < len(self.all_instruments):
            return self.all_instruments[self.current_index]
        return None

    @property
    def instrument_id(self):
        m = self.current_market
        return m["instrument"].id if m else None

    def load_all(self):
        """Load and activate BTC instruments from cache."""
        instruments = self._strategy.cache.instruments()
        now_ts = int(datetime.now(timezone.utc).timestamp())
        parsed = self._parse_all(instruments, now_ts)
        paired = self._dedup_and_pair(parsed)
        self.all_instruments = paired
        if paired:
            logger.info(f"Found {len(paired)} BTC 15-min markets")
        else:
            logger.warning("No BTC 15-min markets found!")
        self._activate_current(now_ts)

    def try_switch(self) -> bool:
        """Attempt to switch to next market. Returns True on success."""
        nxt = self.current_index + 1
        if nxt >= len(self.all_instruments):
            logger.warning("No more markets — restart needed")
            return False
        m = self.all_instruments[nxt]
        if datetime.now(timezone.utc) < m["start_time"]:
            return False
        self._subscribe(nxt, m)
        logger.info(f"SWITCHED → {m['slug']}")
        return True

    def handle_market_open(self):
        """Transition a waiting market to active."""
        m = self.current_market
        if m:
            self.next_switch_time = m["end_time"]
        self.waiting_for_open = False

    def subscribe_current(self):
        """Subscribe to quote ticks for the current market."""
        if not self.instrument_id:
            return
        self._strategy.subscribe_quote_ticks(self.instrument_id)
        if self.no_instrument_id:
            self._strategy.subscribe_quote_ticks(self.no_instrument_id)

    # ── Private helpers ──────────────────────────────────────────────────

    def _parse_all(self, instruments, now_ts) -> List[Dict]:
        result = []
        for inst in instruments:
            try:
                if not (hasattr(inst, "info") and inst.info):
                    continue
                q = inst.info.get("question", "").lower()
                slug = inst.info.get("market_slug", "").lower()
                if not (("btc" in q or "btc" in slug) and "15m" in slug):
                    continue
                parsed = self._parse_single(inst, slug, now_ts)
                if parsed:
                    result.append(parsed)
            except Exception:
                continue
        return result

    def _parse_single(self, inst, slug, now_ts):
        try:
            ts_part = slug.split("-")[-1]
            mkt_ts = int(ts_part)
            end_ts = mkt_ts + self._interval
            if end_ts <= now_ts:
                return None
            raw = str(inst.id)
            base = raw.split(".")[0] if "." in raw else raw
            tok = base.split("-")[-1] if "-" in base else base
            return {
                "instrument": inst, "slug": slug,
                "start_time": datetime.fromtimestamp(mkt_ts, tz=timezone.utc),
                "end_time": datetime.fromtimestamp(end_ts, tz=timezone.utc),
                "market_timestamp": mkt_ts, "end_timestamp": end_ts,
                "time_diff_minutes": (mkt_ts - now_ts) / 60,
                "yes_token_id": tok,
            }
        except (ValueError, IndexError):
            return None

    def _dedup_and_pair(self, items: List[Dict]) -> List[Dict]:
        seen, out = {}, []
        for it in items:
            s = it["slug"]
            if s not in seen:
                it["yes_instrument_id"] = it["instrument"].id
                it["no_instrument_id"] = None
                seen[s] = it
                out.append(it)
            else:
                seen[s]["no_instrument_id"] = it["instrument"].id
        out.sort(key=lambda x: x["market_timestamp"])
        return out

    def _activate_current(self, now_ts):
        for i, m in enumerate(self.all_instruments):
            if m["time_diff_minutes"] <= 0 and m["end_timestamp"] > now_ts:
                self._subscribe(i, m)
                return
        if not self.all_instruments:
            return
        future = [m for m in self.all_instruments if m["time_diff_minutes"] > 0]
        nearest = min(future, key=lambda x: x["time_diff_minutes"]) if future else self.all_instruments[-1]
        idx = self.all_instruments.index(nearest)
        self._subscribe(idx, nearest)
        self.next_switch_time = nearest["start_time"]
        self.waiting_for_open = True
        logger.info(f"Waiting for: {nearest['slug']}")

    def _subscribe(self, idx, m):
        self.current_index = idx
        self.next_switch_time = m["end_time"]
        self.yes_token_id = m.get("yes_token_id")
        self.yes_instrument_id = m.get("yes_instrument_id", m["instrument"].id)
        self.no_instrument_id = m.get("no_instrument_id")
        self._strategy.subscribe_quote_ticks(m["instrument"].id)
        if m.get("no_instrument_id"):
            self._strategy.subscribe_quote_ticks(m["no_instrument_id"])

