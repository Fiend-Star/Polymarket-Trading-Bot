"""
Polymarket RTDS WebSocket Connector — V1

Connects to Polymarket's Real-Time Data Stream at:
  wss://ws-live-data.polymarket.com

Subscribes to TWO crypto price feeds simultaneously:
  1. crypto_prices (Binance) → btcusdt reference price
  2. crypto_prices_chainlink (Chainlink) → btc/usd SETTLEMENT ORACLE

This is the EXACT Chainlink feed that determines 15-min binary resolution.
No authentication required. Sub-second updates.

KEY FEATURES:
  - Real-time Chainlink settlement oracle monitoring (replaces Coinbase polling)
  - Late-window strategy: at T-10s, Chainlink price vs strike → 85% directional certainty
  - Binance-Chainlink divergence tracking (proxy for order flow)
  - Feeds VolEstimator with sub-second tick data (10x better than 10s Coinbase polls)
  - Thread-safe price access for the main trading bot

USAGE IN BOT:
  rtds = RTDSConnector(vol_estimator=vol_est)
  await rtds.connect()
  
  # Get latest prices
  chainlink_price = rtds.chainlink_btc_price
  binance_price = rtds.binance_btc_price
  
  # Late-window signal
  signal = rtds.get_late_window_signal(strike=66000.0, time_remaining_sec=8.0)
  
  # Divergence signal
  div = rtds.get_divergence()
"""

import asyncio
import json
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable, Deque

from loguru import logger

try:
    import aiohttp
except ImportError:
    aiohttp = None
    logger.warning("aiohttp not installed — RTDS connector requires it: pip install aiohttp")


# =============================================================================
# Constants
# =============================================================================
RTDS_WS_URL = "wss://ws-live-data.polymarket.com"
PING_INTERVAL_SEC = 30.0
RECONNECT_DELAY_SEC = 5.0
MAX_RECONNECT_ATTEMPTS = 50
PRICE_HISTORY_MAXLEN = 500  # Last N price observations per feed


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class PriceTick:
    """A single price observation from RTDS."""
    source: str             # "chainlink" or "binance"
    symbol: str             # "btc/usd" or "btcusdt"
    price: float
    source_timestamp_ms: int  # When the price was recorded (exchange/oracle)
    received_timestamp_ms: int  # When we received it via WebSocket
    latency_ms: int          # received - source


@dataclass
class LateWindowSignal:
    """Signal from late-window Chainlink strategy."""
    direction: str          # "BUY_YES", "BUY_NO", or "NO_SIGNAL"
    chainlink_price: float
    strike: float
    delta_bps: float        # Price distance in basis points
    confidence: float       # 0-1, based on distance + time remaining
    time_remaining_sec: float
    reason: str


@dataclass
class DivergenceSignal:
    """Binance-Chainlink price divergence signal."""
    binance_price: float
    chainlink_price: float
    divergence_bps: float   # (binance - chainlink) / chainlink × 10000
    direction: str          # "BINANCE_LEADING" or "CHAINLINK_LEADING"
    is_significant: bool    # > threshold
    staleness_ms: int       # Max age of either price


# =============================================================================
# RTDS Connector
# =============================================================================

class RTDSConnector:
    """
    Real-time connection to Polymarket RTDS for Chainlink + Binance BTC prices.

    Feeds the settlement oracle data directly to the trading strategy,
    enabling the late-window strategy and better vol estimation.
    """

    def __init__(
        self,
        vol_estimator=None,
        on_chainlink_tick: Optional[Callable] = None,
        on_binance_tick: Optional[Callable] = None,
        divergence_threshold_bps: float = 5.0,
        late_window_min_bps: float = 3.0,      # Min delta for late-window signal
        late_window_max_sec: float = 15.0,      # Activate within last N seconds
        late_window_high_conf_bps: float = 10.0,  # >10 bps = high confidence
    ):
        self.vol_estimator = vol_estimator
        self._on_chainlink_tick = on_chainlink_tick
        self._on_binance_tick = on_binance_tick
        self.divergence_threshold_bps = divergence_threshold_bps
        self.late_window_min_bps = late_window_min_bps
        self.late_window_max_sec = late_window_max_sec
        self.late_window_high_conf_bps = late_window_high_conf_bps

        # Thread-safe price state
        self._lock = threading.Lock()
        self._chainlink_price: float = 0.0
        self._chainlink_ts: int = 0
        self._binance_price: float = 0.0
        self._binance_ts: int = 0

        # Price histories
        self._chainlink_ticks: Deque[PriceTick] = deque(maxlen=PRICE_HISTORY_MAXLEN)
        self._binance_ticks: Deque[PriceTick] = deque(maxlen=PRICE_HISTORY_MAXLEN)

        # Connection state
        self._ws = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._reconnect_count = 0
        self._total_messages = 0
        self._should_stop = False

        # Stats
        self._chainlink_msg_count = 0
        self._binance_msg_count = 0
        self._avg_latency_ms = 0.0
        self._last_divergence: Optional[DivergenceSignal] = None

        logger.info(
            f"Initialized RTDS Connector: endpoint={RTDS_WS_URL}, "
            f"divergence_threshold={divergence_threshold_bps}bps, "
            f"late_window_max={late_window_max_sec}s"
        )

    # ==================================================================
    # Public: Price access (thread-safe)
    # ==================================================================

    @property
    def chainlink_btc_price(self) -> float:
        """Latest Chainlink BTC/USD price (settlement oracle)."""
        with self._lock:
            return self._chainlink_price

    @property
    def binance_btc_price(self) -> float:
        """Latest Binance BTCUSDT price."""
        with self._lock:
            return self._binance_price

    @property
    def chainlink_age_ms(self) -> int:
        """Milliseconds since last Chainlink update."""
        with self._lock:
            if self._chainlink_ts == 0:
                return 999999
            return int(time.time() * 1000) - self._chainlink_ts

    @property
    def binance_age_ms(self) -> int:
        """Milliseconds since last Binance update."""
        with self._lock:
            if self._binance_ts == 0:
                return 999999
            return int(time.time() * 1000) - self._binance_ts

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ==================================================================
    # Public: Late-window Chainlink strategy
    # ==================================================================

    def get_late_window_signal(
        self,
        strike: float,
        time_remaining_sec: float,
    ) -> LateWindowSignal:
        """
        Late-window strategy: at T-10s, compare Chainlink price vs strike.
        
        At T-10 seconds, BTC direction is ~85% deterministic.
        If Chainlink shows clear direction, signal postOnly limit order
        at 0.90-0.95 on the winning side.

        Research basis:
        - "Window delta is king" — BTC price vs strike is strongest signal
        - Direction becomes 85% knowable in final 10 seconds
        - postOnly maker orders = 0% fee + daily rebates
        """
        chainlink_px = self.chainlink_btc_price

        if chainlink_px <= 0 or strike <= 0:
            return LateWindowSignal(
                direction="NO_SIGNAL", chainlink_price=chainlink_px,
                strike=strike, delta_bps=0, confidence=0,
                time_remaining_sec=time_remaining_sec,
                reason="No Chainlink price available",
            )

        if time_remaining_sec > self.late_window_max_sec:
            return LateWindowSignal(
                direction="NO_SIGNAL", chainlink_price=chainlink_px,
                strike=strike, delta_bps=0, confidence=0,
                time_remaining_sec=time_remaining_sec,
                reason=f"Too early: {time_remaining_sec:.0f}s > {self.late_window_max_sec}s",
            )

        # Check Chainlink data freshness
        age = self.chainlink_age_ms
        if age > 5000:
            return LateWindowSignal(
                direction="NO_SIGNAL", chainlink_price=chainlink_px,
                strike=strike, delta_bps=0, confidence=0,
                time_remaining_sec=time_remaining_sec,
                reason=f"Chainlink data stale: {age}ms old",
            )

        # Compute distance from strike
        delta_pct = (chainlink_px - strike) / strike
        delta_bps = delta_pct * 10000.0

        # Direction
        if delta_bps > self.late_window_min_bps:
            direction = "BUY_YES"  # Chainlink above strike → Up wins
        elif delta_bps < -self.late_window_min_bps:
            direction = "BUY_NO"   # Chainlink below strike → Down wins
        else:
            return LateWindowSignal(
                direction="NO_SIGNAL", chainlink_price=chainlink_px,
                strike=strike, delta_bps=delta_bps, confidence=0,
                time_remaining_sec=time_remaining_sec,
                reason=f"Too close to strike: {abs(delta_bps):.1f}bps < {self.late_window_min_bps}bps",
            )

        # Confidence: scales with distance and inverse of time remaining
        # More distance → more confident, less time → more confident
        dist_factor = min(1.0, abs(delta_bps) / self.late_window_high_conf_bps)
        time_factor = min(1.0, max(0.0, (self.late_window_max_sec - time_remaining_sec)
                                   / self.late_window_max_sec))
        confidence = 0.5 + 0.35 * dist_factor + 0.15 * time_factor
        confidence = min(0.95, confidence)  # Cap — nothing is certain

        return LateWindowSignal(
            direction=direction,
            chainlink_price=chainlink_px,
            strike=strike,
            delta_bps=delta_bps,
            confidence=confidence,
            time_remaining_sec=time_remaining_sec,
            reason=f"Chainlink {'above' if delta_bps > 0 else 'below'} strike by {abs(delta_bps):.1f}bps",
        )

    # ==================================================================
    # Public: Binance-Chainlink divergence
    # ==================================================================

    def get_divergence(self) -> DivergenceSignal:
        """
        Compute Binance-Chainlink price divergence.
        
        When Binance leads Chainlink significantly, it suggests directional
        pressure that the settlement oracle hasn't reflected yet.
        This is a simplified proxy for order flow imbalance:
        - If Binance >> Chainlink → buying pressure → BTC likely going up
        - If Binance << Chainlink → selling pressure → BTC likely going down
        """
        with self._lock:
            bn_px = self._binance_price
            cl_px = self._chainlink_price
            bn_ts = self._binance_ts
            cl_ts = self._chainlink_ts

        if bn_px <= 0 or cl_px <= 0:
            return DivergenceSignal(
                binance_price=bn_px, chainlink_price=cl_px,
                divergence_bps=0.0, direction="NEUTRAL",
                is_significant=False, staleness_ms=999999,
            )

        now_ms = int(time.time() * 1000)
        staleness = max(now_ms - bn_ts, now_ms - cl_ts) if bn_ts and cl_ts else 999999

        divergence_bps = (bn_px - cl_px) / cl_px * 10000.0

        if divergence_bps > self.divergence_threshold_bps:
            direction = "BINANCE_LEADING"
        elif divergence_bps < -self.divergence_threshold_bps:
            direction = "CHAINLINK_LEADING"
        else:
            direction = "NEUTRAL"

        is_significant = abs(divergence_bps) > self.divergence_threshold_bps

        signal = DivergenceSignal(
            binance_price=bn_px, chainlink_price=cl_px,
            divergence_bps=divergence_bps, direction=direction,
            is_significant=is_significant, staleness_ms=staleness,
        )
        self._last_divergence = signal
        return signal

    # ==================================================================
    # Public: Connection management
    # ==================================================================

    async def connect(self):
        """Connect to RTDS WebSocket and start streaming."""
        if aiohttp is None:
            logger.error("aiohttp required for RTDS connector: pip install aiohttp")
            return

        self._should_stop = False
        while not self._should_stop and self._reconnect_count < MAX_RECONNECT_ATTEMPTS:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                logger.info("RTDS connector cancelled")
                break
            except Exception as e:
                self._reconnect_count += 1
                self._connected = False
                logger.warning(
                    f"RTDS connection lost ({e}), "
                    f"reconnecting in {RECONNECT_DELAY_SEC}s "
                    f"(attempt {self._reconnect_count}/{MAX_RECONNECT_ATTEMPTS})"
                )
                await asyncio.sleep(RECONNECT_DELAY_SEC)

        if self._reconnect_count >= MAX_RECONNECT_ATTEMPTS:
            logger.error(f"RTDS: max reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) reached")

    async def disconnect(self):
        """Gracefully disconnect."""
        self._should_stop = True
        self._connected = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("RTDS connector disconnected")

    def start_background(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start the RTDS connector in a background thread."""
        def _run():
            _loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_loop)
            try:
                _loop.run_until_complete(self.connect())
            except Exception as e:
                logger.error(f"RTDS background thread error: {e}")
            finally:
                _loop.close()

        t = threading.Thread(target=_run, daemon=True, name="rtds-connector")
        t.start()
        logger.info("RTDS connector started in background thread")
        return t

    # ==================================================================
    # Internal: WebSocket connection + message handling
    # ==================================================================

    async def _connect_and_stream(self):
        """Establish WS connection, subscribe, and process messages."""
        self._session = aiohttp.ClientSession()

        async with self._session.ws_connect(
            RTDS_WS_URL,
            heartbeat=PING_INTERVAL_SEC,
            timeout=aiohttp.ClientWSTimeout(ws_close=10),
        ) as ws:
            self._ws = ws
            self._connected = True
            self._reconnect_count = 0
            logger.info(f"✓ RTDS WebSocket connected: {RTDS_WS_URL}")

            # Subscribe to both feeds
            await self._subscribe(ws)

            # Process messages
            async for msg in ws:
                if self._should_stop:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"RTDS WS error: {ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("RTDS WS closed by server")
                    break

        self._connected = False
        if self._session and not self._session.closed:
            await self._session.close()

    async def _subscribe(self, ws):
        """Subscribe to Binance + Chainlink BTC price feeds."""
        # Binance feed
        binance_sub = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": "btcusdt",
                }
            ],
        }
        await ws.send_json(binance_sub)
        logger.info("RTDS: Subscribed to crypto_prices (Binance btcusdt)")

        # Chainlink feed — THE SETTLEMENT ORACLE
        chainlink_sub = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": json.dumps({"symbol": "btc/usd"}),
                }
            ],
        }
        await ws.send_json(chainlink_sub)
        logger.info("RTDS: Subscribed to crypto_prices_chainlink (btc/usd) — SETTLEMENT ORACLE")

    def _handle_message(self, raw: str):
        """Parse and route an incoming RTDS message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        topic = data.get("topic", "")
        payload = data.get("payload")
        msg_ts = data.get("timestamp", 0)
        self._total_messages += 1

        if not payload or "value" not in payload:
            return

        price = float(payload["value"])
        source_ts = int(payload.get("timestamp", msg_ts))
        received_ts = int(time.time() * 1000)
        latency = received_ts - source_ts if source_ts > 0 else 0

        if topic == "crypto_prices" and "btc" in payload.get("symbol", "").lower():
            self._handle_binance_tick(price, source_ts, received_ts, latency, payload)
        elif topic == "crypto_prices_chainlink" and "btc" in payload.get("symbol", "").lower():
            self._handle_chainlink_tick(price, source_ts, received_ts, latency, payload)

    def _handle_binance_tick(self, price, source_ts, received_ts, latency, payload):
        """Process a Binance BTC price tick."""
        tick = PriceTick(
            source="binance", symbol=payload.get("symbol", "btcusdt"),
            price=price, source_timestamp_ms=source_ts,
            received_timestamp_ms=received_ts, latency_ms=latency,
        )

        with self._lock:
            self._binance_price = price
            self._binance_ts = received_ts
        self._binance_ticks.append(tick)
        self._binance_msg_count += 1

        # Feed VolEstimator
        if self.vol_estimator:
            self.vol_estimator.add_price(price, received_ts / 1000.0)

        if self._on_binance_tick:
            try:
                self._on_binance_tick(tick)
            except Exception as e:
                logger.debug(f"Binance tick callback error: {e}")

        # Periodic stats
        if self._binance_msg_count % 100 == 0:
            logger.debug(
                f"RTDS Binance: {self._binance_msg_count} ticks, "
                f"price=${price:,.2f}, latency={latency}ms"
            )

    def _handle_chainlink_tick(self, price, source_ts, received_ts, latency, payload):
        """Process a Chainlink BTC price tick (SETTLEMENT ORACLE)."""
        tick = PriceTick(
            source="chainlink", symbol=payload.get("symbol", "btc/usd"),
            price=price, source_timestamp_ms=source_ts,
            received_timestamp_ms=received_ts, latency_ms=latency,
        )

        with self._lock:
            self._chainlink_price = price
            self._chainlink_ts = received_ts
        self._chainlink_ticks.append(tick)
        self._chainlink_msg_count += 1

        # Chainlink IS the settlement price — more important for the model
        if self.vol_estimator:
            self.vol_estimator.add_price(price, received_ts / 1000.0)

        if self._on_chainlink_tick:
            try:
                self._on_chainlink_tick(tick)
            except Exception as e:
                logger.debug(f"Chainlink tick callback error: {e}")

        # Update average latency
        if self._chainlink_msg_count > 1:
            alpha = 0.05
            self._avg_latency_ms = (1 - alpha) * self._avg_latency_ms + alpha * latency
        else:
            self._avg_latency_ms = float(latency)

        if self._chainlink_msg_count % 100 == 0:
            logger.debug(
                f"RTDS Chainlink: {self._chainlink_msg_count} ticks, "
                f"price=${price:,.2f}, latency={latency}ms, "
                f"avg_lat={self._avg_latency_ms:.0f}ms"
            )

    # ==================================================================
    # Stats
    # ==================================================================

    def get_stats(self) -> dict:
        """Get connector statistics."""
        return {
            "connected": self._connected,
            "total_messages": self._total_messages,
            "chainlink_ticks": self._chainlink_msg_count,
            "binance_ticks": self._binance_msg_count,
            "chainlink_price": self.chainlink_btc_price,
            "binance_price": self.binance_btc_price,
            "chainlink_age_ms": self.chainlink_age_ms,
            "binance_age_ms": self.binance_age_ms,
            "avg_chainlink_latency_ms": round(self._avg_latency_ms, 1),
            "reconnect_count": self._reconnect_count,
            "last_divergence_bps": (
                round(self._last_divergence.divergence_bps, 1)
                if self._last_divergence else None
            ),
        }

    def get_recent_chainlink_prices(self, n: int = 60) -> list:
        """Get last N Chainlink price ticks."""
        return list(self._chainlink_ticks)[-n:]

    def get_recent_binance_prices(self, n: int = 60) -> list:
        """Get last N Binance price ticks."""
        return list(self._binance_ticks)[-n:]
