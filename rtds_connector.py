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
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, Deque, Dict

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

# V3.2: Staleness detection — if a price feed's VALUE hasn't changed
# in this many ms, treat it as frozen/stale (even if ticks keep arriving
# with the same value).  Prevents phantom divergence signals.
STALE_PRICE_THRESHOLD_MS = 30_000  # 30 seconds


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class PriceTick:
    """A single price observation from RTDS."""
    source: str  # "chainlink" or "binance"
    symbol: str  # "btc/usd" or "btcusdt"
    price: float
    source_timestamp_ms: int  # When the price was recorded (exchange/oracle)
    received_timestamp_ms: int  # When we received it via WebSocket
    latency_ms: int  # received - source


@dataclass
class LateWindowSignal:
    """Signal from late-window Chainlink strategy."""
    direction: str  # "BUY_YES", "BUY_NO", or "NO_SIGNAL"
    chainlink_price: float
    strike: float
    delta_bps: float  # Price distance in basis points
    confidence: float  # 0-1, based on distance + time remaining
    time_remaining_sec: float
    reason: str


@dataclass
class DivergenceSignal:
    """Binance-Chainlink price divergence signal."""
    binance_price: float
    chainlink_price: float
    divergence_bps: float  # (binance - chainlink) / chainlink × 10000
    direction: str  # "BINANCE_LEADING" or "CHAINLINK_LEADING"
    is_significant: bool  # > threshold
    staleness_ms: int  # Max age of either price


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
            late_window_min_bps: float = 3.0,  # Min delta for late-window signal
            late_window_max_sec: float = 15.0,  # Activate within last N seconds
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

        # V3.2: Track when price VALUE last changed (not just when a tick arrived)
        self._binance_last_changed_price: float = 0.0
        self._binance_last_changed_ts: int = 0
        self._chainlink_last_changed_price: float = 0.0
        self._chainlink_last_changed_ts: int = 0

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

        # V3.3: Pre-scheduled boundary snapshots.
        # The bot registers upcoming 15-min boundary timestamps.  As Chainlink
        # ticks stream in, we capture the tick whose *source* timestamp is
        # the tick whose *source* timestamp is closest to each boundary, looking strictly backwards.
        self._boundary_snapshots: Dict[int, Optional[tuple]] = {}
        # How far back (in ms) to look for the last known price.
        # With the strict `source_ts <= boundary` rule, we need a wide window (2 minutes)
        # in case Chainlink didn't update right before the boundary.
        self._boundary_capture_window_ms: int = 120_000  # 120 seconds

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
    # Public: Boundary snapshot — pre-timed Chainlink capture
    # ==================================================================

    def register_boundary(self, boundary_timestamp_s: int) -> None:
        """
        Pre-register a 15-min boundary so we capture the Chainlink price
        at that exact moment as ticks stream in.  Call this BEFORE the
        boundary arrives (e.g. on startup or market switch).

        The tick handler will automatically snapshot the closest Chainlink
        tick within ±5 seconds of the boundary — zero extra latency.
        """
        with self._lock:
            if boundary_timestamp_s not in self._boundary_snapshots:
                self._boundary_snapshots[boundary_timestamp_s] = None
                logger.debug(f"Registered boundary snapshot for ts={boundary_timestamp_s}")

            # Cleanup: remove snapshots older than 30 minutes
            now_s = int(time.time())
            stale = [ts for ts in self._boundary_snapshots if now_s - ts > 1800]
            for ts in stale:
                del self._boundary_snapshots[ts]

    def get_boundary_price(self, boundary_timestamp_s: int) -> Optional[float]:
        """
        Get the Chainlink price captured at a pre-registered boundary.

        Returns the price if a tick was captured within the capture window,
        otherwise None.  This is near-zero latency — the price was stored
        inline during tick processing, not fetched retroactively.
        """
        with self._lock:
            snap = self._boundary_snapshots.get(boundary_timestamp_s)
            if snap is not None:
                return snap[0]  # (price, delta_ms)
            return None

    def get_boundary_detail(self, boundary_timestamp_s: int) -> Optional[tuple]:
        """Get (price, delta_ms) for a boundary snapshot, or None."""
        with self._lock:
            return self._boundary_snapshots.get(boundary_timestamp_s)

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
            direction = "BUY_NO"  # Chainlink below strike → Down wins
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

        V3.2: Staleness guard — if either feed's VALUE hasn't changed in
        STALE_PRICE_THRESHOLD_MS, the divergence is phantom (frozen feed)
        and we return NEUTRAL.
        """
        with self._lock:
            bn_px = self._binance_price
            cl_px = self._chainlink_price
            bn_ts = self._binance_ts
            cl_ts = self._chainlink_ts
            bn_changed_ts = self._binance_last_changed_ts
            cl_changed_ts = self._chainlink_last_changed_ts

        if bn_px <= 0 or cl_px <= 0:
            return DivergenceSignal(
                binance_price=bn_px, chainlink_price=cl_px,
                divergence_bps=0.0, direction="NEUTRAL",
                is_significant=False, staleness_ms=999999,
            )

        now_ms = int(time.time() * 1000)
        staleness = max(now_ms - bn_ts, now_ms - cl_ts) if bn_ts and cl_ts else 999999

        # V3.2: Staleness guard — detect frozen price feeds
        bn_value_age = (now_ms - bn_changed_ts) if bn_changed_ts > 0 else 999999
        cl_value_age = (now_ms - cl_changed_ts) if cl_changed_ts > 0 else 999999

        if bn_value_age > STALE_PRICE_THRESHOLD_MS or cl_value_age > STALE_PRICE_THRESHOLD_MS:
            stale_source = "Binance" if bn_value_age > cl_value_age else "Chainlink"
            stale_age_sec = max(bn_value_age, cl_value_age) / 1000.0
            logger.warning(
                f"⚠ Divergence STALE: {stale_source} price unchanged for "
                f"{stale_age_sec:.0f}s — treating as NEUTRAL"
            )
            signal = DivergenceSignal(
                binance_price=bn_px, chainlink_price=cl_px,
                divergence_bps=0.0, direction="NEUTRAL",
                is_significant=False, staleness_ms=max(bn_value_age, cl_value_age),
            )
            self._last_divergence = signal
            return signal

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
        try:
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

                # V3.3: Track if subscriptions need retry after rate-limit / validation error
                _resub_pending = False
                _resub_after = 0.0

                # Process messages
                while not self._should_stop:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                    except TimeoutError:
                        # Timed out waiting for message; evaluate if we need to retry subscription
                        if _resub_pending and time.time() >= _resub_after:
                            logger.info("RTDS: Retrying subscriptions after error cooldown (timeout)...")
                            await self._subscribe(ws)
                            _resub_pending = False
                        continue
                    except asyncio.TimeoutError:
                        # Depending on Python version, asyncio.TimeoutError vs TimeoutError
                        if _resub_pending and time.time() >= _resub_after:
                            logger.info("RTDS: Retrying subscriptions after error cooldown (timeout)...")
                            await self._subscribe(ws)
                            _resub_pending = False
                        continue

                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSING):
                        if msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"RTDS WS error: {ws.exception()}")
                        else:
                            logger.warning(f"RTDS WS closed by server ({msg.type})")
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # V3.3: Detect 429 or 400 errors on subscriptions
                        if not _resub_pending:
                            data_lower = msg.data.lower()
                            if '"message"' in data_lower and ('too many' in data_lower or 'rate limit' in data_lower):
                                logger.warning(
                                    f"RTDS: Subscription rate-limited (429) — will retry in 10s. Server replied: {msg.data}")
                                _resub_pending = True
                                _resub_after = time.time() + 10.0
                            elif 'failed validation' in data_lower or 'invalid' in data_lower:
                                logger.warning(
                                    f"RTDS: Subscription validation error (400) — will retry in 5s. Server replied: {msg.data}")
                                _resub_pending = True
                                _resub_after = time.time() + 5.0

                        # V3.3: Retry subscriptions after rate-limit cooldown
                        if _resub_pending and time.time() >= _resub_after:
                            logger.info("RTDS: Retrying subscriptions after error cooldown...")
                            await self._subscribe(ws)
                            _resub_pending = False

                        self._handle_message(msg.data)
        finally:
            # Always clean up the session — prevents "Unclosed client session" warnings
            self._connected = False
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

    async def _subscribe(self, ws):
        """Subscribe to Binance + Chainlink BTC price feeds.

        Chainlink subscription is unfiltered — server-side filters silently
        drop all ticks without error. Client-side filtering in _handle_message
        routes btc/usd ticks to the Chainlink handler.
        """
        # Small delay after connect to avoid immediate rate-limit
        await asyncio.sleep(0.5)

        combined_sub = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": json.dumps({"symbol": "btcusdt"}),
                },
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "update",
                },
            ],
        }
        await ws.send_json(combined_sub)
        logger.info("RTDS: Subscribed to crypto_prices (btcusdt) + crypto_prices_chainlink (unfiltered)")

    def _handle_message(self, raw: str):
        """Parse and route an incoming RTDS message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        topic = data.get("topic", "")
        payload = data.get("payload")
        self._total_messages += 1

        # V3.4: Log first 2 messages for format diagnostics
        if self._total_messages <= 2:
            raw_preview = raw[:500] if len(raw) > 500 else raw
            logger.info(
                f"RTDS msg #{self._total_messages}: topic='{topic}', "
                f"has_payload={payload is not None}, "
                f"raw={raw_preview}"
            )

        if not payload:
            if self._total_messages <= 20:
                logger.debug(f"RTDS msg dropped (no payload): topic='{topic}', keys={list(data.keys())}, raw={raw}")
            return

        # RTDS delivers two formats:
        #   1. Array: payload = {"data": [{timestamp, value}, ...], "symbol": "btcusdt"} (or inside the item)
        #      Used by: crypto_prices topic (both btcusdt AND btc/usd!)
        #   2. Flat:   payload = {"value": ..., "symbol": "btc/usd", "timestamp": ..., "full_accuracy_value": ...}
        #      Used by: crypto_prices_chainlink topic
        
        # Default symbols from outer envelope
        outer_symbol = payload.get("symbol", "").lower()

        tick_data = payload.get("data")

        if tick_data and isinstance(tick_data, list) and len(tick_data) > 0:
            # Array format — use the LATEST tick (last element = most recent)
            latest = tick_data[-1]
            price = float(latest.get("value", 0))
            source_ts = int(latest.get("timestamp", 0))
            received_ts = int(time.time() * 1000)
            latency = received_ts - source_ts if source_ts > 0 else 0

            # Some array formats bury the symbol inside the innermost item
            inner_symbol = latest.get("symbol", "").lower() or outer_symbol
            
            is_chainlink = (topic == "crypto_prices_chainlink" or "btc/usd" in inner_symbol)
            is_binance = ("btcusdt" in inner_symbol and not is_chainlink)
            has_btc = ("btc" in inner_symbol)

            tick_payload = {"symbol": inner_symbol, "timestamp": source_ts, "value": price}

            if price > 0 and has_btc:
                if is_chainlink:
                    self._handle_chainlink_tick(price, source_ts, received_ts, latency, tick_payload)
                elif is_binance:
                    self._handle_binance_tick(price, source_ts, received_ts, latency, tick_payload)
            return

        # Flat format — payload has "value" directly
        is_chainlink = (topic == "crypto_prices_chainlink" or "btc/usd" in outer_symbol)
        is_binance = ("btcusdt" in outer_symbol and not is_chainlink)
        has_btc = ("btc" in outer_symbol)
        
        raw_value = payload.get("value") or payload.get("price")
        if raw_value is not None:
            price = float(raw_value)
            source_ts = int(payload.get("timestamp", data.get("timestamp", 0)))
            received_ts = int(time.time() * 1000)
            latency = received_ts - source_ts if source_ts > 0 else 0

            if has_btc:
                if is_chainlink:
                    self._handle_chainlink_tick(price, source_ts, received_ts, latency, payload)
                elif is_binance:
                    self._handle_binance_tick(price, source_ts, received_ts, latency, payload)
            return

            if has_btc:
                if is_chainlink:
                    self._handle_chainlink_tick(price, source_ts, received_ts, latency, payload)
                elif is_binance:
                    self._handle_binance_tick(price, source_ts, received_ts, latency, payload)
            return

        if self._total_messages <= 20:
            logger.debug(
                f"RTDS msg dropped (unknown format): "
                f"topic='{topic}', payload_keys={list(payload.keys())}"
            )

    def _handle_binance_tick(self, price, source_ts, received_ts, latency, payload):
        """Process a Binance BTC price tick."""
        tick = PriceTick(
            source="binance", symbol=payload.get("symbol", "btcusdt"),
            price=price, source_timestamp_ms=source_ts,
            received_timestamp_ms=received_ts, latency_ms=latency,
        )

        with self._lock:
            # V3.2: Track when price VALUE actually changes (not just new tick)
            if price != self._binance_price:
                self._binance_last_changed_price = price
                self._binance_last_changed_ts = received_ts
            elif self._binance_last_changed_ts == 0:
                # First tick — initialize
                self._binance_last_changed_price = price
                self._binance_last_changed_ts = received_ts
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
            # V3.2: Track when price VALUE actually changes
            if price != self._chainlink_price:
                self._chainlink_last_changed_price = price
                self._chainlink_last_changed_ts = received_ts
            elif self._chainlink_last_changed_ts == 0:
                self._chainlink_last_changed_price = price
                self._chainlink_last_changed_ts = received_ts
            self._chainlink_price = price
            self._chainlink_ts = received_ts

            # V3.3: Boundary snapshot — capture price at pre-registered boundaries
            # in real-time as the tick arrives (zero additional latency).
            for boundary_ts, existing in list(self._boundary_snapshots.items()):
                boundary_ms = boundary_ts * 1000

                # V3.4: Only capture ticks that occurred AT OR BEFORE the boundary timestamp!
                # We define a range of [0, boundary_capture_window_ms] looking backward.
                delta_ms = boundary_ms - source_ts
                if 0 <= delta_ms <= self._boundary_capture_window_ms:
                    # Keep the tick with the SMALLEST delta (closest to boundary without going over)
                    # Store richer snapshot information: (price, delta_ms, source_ts, received_ts, latency)
                    # NOTE: Keep delta_ms as the second element so existing consumers (bot.py)
                    # that index [1] continue to work.
                    new_snap = (price, delta_ms, source_ts, received_ts, latency)
                    if existing is None or delta_ms < existing[1]:
                        self._boundary_snapshots[boundary_ts] = new_snap
                        if existing is None:
                            # Log richer debug info to help diagnose cases where delta>0
                            logger.info(
                                f"⚡ Boundary snapshot: ts={boundary_ts} → ${price:,.2f} "
                                f"(Δ={delta_ms}ms from boundary, source_ts={source_ts}, "
                                f"received_ts={received_ts}, latency={latency}ms)"
                            )

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

        if self._avg_latency_ms == 0.0:
            self._avg_latency_ms = float(latency)
        else:
            self._avg_latency_ms = (self._avg_latency_ms * 0.99) + (latency * 0.01)

        # Periodic stats
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

    def get_chainlink_price_at(
            self,
            target_timestamp_s: float,
            max_staleness_ms: int = 5000,
    ) -> Optional[float]:
        """
        Get the Chainlink BTC/USD price closest to a specific timestamp.

        V3.3: Checks pre-registered boundary snapshots first (captured in
        real-time by the tick handler with zero extra latency), then falls
        back to scanning the tick history deque.

        Args:
            target_timestamp_s: Unix epoch seconds of the desired moment.
            max_staleness_ms: Maximum allowed distance (ms) from the target.
                              Returns None if no tick is close enough.

        Returns:
            The Chainlink price at (or very near) the target time, or None.
        """
        target_int = int(target_timestamp_s)

        # V3.3: Check boundary snapshot first — captured in real-time
        snap = self.get_boundary_detail(target_int)
        if snap is not None:
            # snap may be (price, delta_ms) or our richer (price, delta_ms, source_ts, received_ts, latency)
            if len(snap) >= 2:
                price = snap[0]
                delta_ms = snap[1]
            else:
                price, delta_ms = snap
            if delta_ms <= max_staleness_ms:
                logger.debug(
                    f"Chainlink price at target {target_int} (snapshot): "
                    f"${price:,.2f} (Δ={delta_ms}ms from boundary)"
                )
                return price

        # Fallback: scan tick history
        target_ms = int(target_timestamp_s * 1000)
        ticks = list(self._chainlink_ticks)
        if not ticks:
            return None

        best_tick = None
        best_delta = float("inf")
        for tick in ticks:
            # V3.4: The price "at" a timestamp is the most recent update <= timestamp
            # We look strictly backward in a defining range [0, max_staleness_ms]
            delta = target_ms - tick.source_timestamp_ms
            if 0 <= delta <= max_staleness_ms:
                if delta < best_delta:
                    best_delta = delta
                    best_tick = tick

        if best_tick is None or best_delta > max_staleness_ms:
            return None

        logger.debug(
            f"Chainlink price at target {target_timestamp_s:.0f}: "
            f"${best_tick.price:,.2f} (Δ={best_delta}ms from boundary)"
        )
        return best_tick.price

