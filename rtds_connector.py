"""
Polymarket RTDS Connector — Chainlink + Binance BTC price feeds.

Subscribes to crypto_prices (Binance) and crypto_prices_chainlink (settlement oracle).
WebSocket transport is in rtds_websocket.py (RTDSWebSocketMixin).
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, Deque

from loguru import logger
from rtds_websocket import RTDSWebSocketMixin

try:
    import aiohttp
except ImportError:
    aiohttp = None

PRICE_HISTORY_MAXLEN = 500


@dataclass
class PriceTick:
    """A single price observation from RTDS."""
    source: str
    symbol: str
    price: float
    source_timestamp_ms: int
    received_timestamp_ms: int
    latency_ms: int


@dataclass
class LateWindowSignal:
    """Signal from late-window Chainlink strategy."""
    direction: str
    chainlink_price: float
    strike: float
    delta_bps: float
    confidence: float
    time_remaining_sec: float
    reason: str


@dataclass
class DivergenceSignal:
    """Binance-Chainlink price divergence signal."""
    binance_price: float
    chainlink_price: float
    divergence_bps: float
    direction: str
    is_significant: bool
    staleness_ms: int


class RTDSConnector(RTDSWebSocketMixin):
    """Real-time Chainlink + Binance BTC price feed via Polymarket RTDS."""

    def __init__(self, vol_estimator=None, on_chainlink_tick=None, on_binance_tick=None,
                 divergence_threshold_bps=5.0, late_window_min_bps=3.0,
                 late_window_max_sec=15.0, late_window_high_conf_bps=10.0):
        self.vol_estimator = vol_estimator
        self._on_chainlink_tick = on_chainlink_tick
        self._on_binance_tick = on_binance_tick
        self.divergence_threshold_bps = divergence_threshold_bps
        self.late_window_min_bps = late_window_min_bps
        self.late_window_max_sec = late_window_max_sec
        self.late_window_high_conf_bps = late_window_high_conf_bps
        self._init_price_state()
        self._init_connection_state()

    def _init_price_state(self):
        self._lock = threading.Lock()
        self._chainlink_price: float = 0.0
        self._chainlink_ts: int = 0
        self._binance_price: float = 0.0
        self._binance_ts: int = 0
        self._chainlink_ticks: Deque[PriceTick] = deque(maxlen=PRICE_HISTORY_MAXLEN)
        self._binance_ticks: Deque[PriceTick] = deque(maxlen=PRICE_HISTORY_MAXLEN)

    def _init_connection_state(self):
        self._ws = None
        self._session = None
        self._connected = False
        self._reconnect_count = 0
        self._total_messages = 0
        self._should_stop = False
        self._chainlink_msg_count = 0
        self._binance_msg_count = 0
        self._avg_latency_ms = 0.0
        self._last_divergence: Optional[DivergenceSignal] = None

    # ── Thread-safe price access ─────────────────────────────────────

    @property
    def chainlink_btc_price(self) -> float:
        with self._lock: return self._chainlink_price

    @property
    def binance_btc_price(self) -> float:
        with self._lock: return self._binance_price

    @property
    def chainlink_age_ms(self) -> int:
        with self._lock:
            return 999999 if self._chainlink_ts == 0 else int(time.time()*1000) - self._chainlink_ts

    @property
    def binance_age_ms(self) -> int:
        with self._lock:
            return 999999 if self._binance_ts == 0 else int(time.time()*1000) - self._binance_ts

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Late-window Chainlink strategy ───────────────────────────────

    def _no_signal(self, px, strike, bps, trs, reason):
        return LateWindowSignal(direction="NO_SIGNAL", chainlink_price=px,
                                strike=strike, delta_bps=bps, confidence=0,
                                time_remaining_sec=trs, reason=reason)

    def get_late_window_signal(self, strike: float, time_remaining_sec: float) -> LateWindowSignal:
        px = self.chainlink_btc_price
        no_sig = self._validate_late_window(px, strike, time_remaining_sec)
        if no_sig is not None:
            return no_sig
        delta_bps = (px - strike) / strike * 10000.0
        if delta_bps > self.late_window_min_bps:
            direction = "BUY_YES"
        elif delta_bps < -self.late_window_min_bps:
            direction = "BUY_NO"
        else:
            return self._no_signal(px, strike, delta_bps, time_remaining_sec,
                                   f"Too close: {abs(delta_bps):.1f}bps < {self.late_window_min_bps}bps")
        conf = self._compute_late_confidence(delta_bps, time_remaining_sec)
        side = 'above' if delta_bps > 0 else 'below'
        return LateWindowSignal(direction=direction, chainlink_price=px, strike=strike,
                                delta_bps=delta_bps, confidence=conf,
                                time_remaining_sec=time_remaining_sec,
                                reason=f"Chainlink {side} strike by {abs(delta_bps):.1f}bps")

    def _validate_late_window(self, px, strike, trs):
        if px <= 0 or strike <= 0:
            return self._no_signal(px, strike, 0, trs, "No Chainlink price")
        if trs > self.late_window_max_sec:
            return self._no_signal(px, strike, 0, trs,
                                   f"Too early: {trs:.0f}s > {self.late_window_max_sec}s")
        if self.chainlink_age_ms > 5000:
            return self._no_signal(px, strike, 0, trs,
                                   f"Stale: {self.chainlink_age_ms}ms")
        return None

    def _compute_late_confidence(self, delta_bps, trs):
        dist = min(1.0, abs(delta_bps) / self.late_window_high_conf_bps)
        tfac = min(1.0, max(0.0, (self.late_window_max_sec - trs) / self.late_window_max_sec))
        return min(0.95, 0.5 + 0.35 * dist + 0.15 * tfac)

    # ── Divergence ───────────────────────────────────────────────────

    def get_divergence(self) -> DivergenceSignal:
        with self._lock:
            bn, cl = self._binance_price, self._chainlink_price
            bn_ts, cl_ts = self._binance_ts, self._chainlink_ts
        if bn <= 0 or cl <= 0:
            return DivergenceSignal(bn, cl, 0.0, "NEUTRAL", False, 999999)
        now_ms = int(time.time() * 1000)
        stale = max(now_ms - bn_ts, now_ms - cl_ts) if bn_ts and cl_ts else 999999
        div = (bn - cl) / cl * 10000.0
        if div > self.divergence_threshold_bps:
            d = "BINANCE_LEADING"
        elif div < -self.divergence_threshold_bps:
            d = "CHAINLINK_LEADING"
        else:
            d = "NEUTRAL"
        sig = DivergenceSignal(bn, cl, div, d, abs(div) > self.divergence_threshold_bps, stale)
        self._last_divergence = sig
        return sig

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "connected": self._connected, "total_messages": self._total_messages,
            "chainlink_ticks": self._chainlink_msg_count, "binance_ticks": self._binance_msg_count,
            "chainlink_price": self.chainlink_btc_price, "binance_price": self.binance_btc_price,
            "chainlink_age_ms": self.chainlink_age_ms, "binance_age_ms": self.binance_age_ms,
            "avg_chainlink_latency_ms": round(self._avg_latency_ms, 1),
            "reconnect_count": self._reconnect_count,
            "last_divergence_bps": (round(self._last_divergence.divergence_bps, 1)
                                    if self._last_divergence else None),
        }

    def get_recent_chainlink_prices(self, n=60):
        return list(self._chainlink_ticks)[-n:]

    def get_recent_binance_prices(self, n=60):
        return list(self._binance_ticks)[-n:]

