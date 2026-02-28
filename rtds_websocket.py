"""
RTDS WebSocket transport — connection management & message handling.

SRP: WebSocket lifecycle, subscription, and tick parsing.
"""
import asyncio
import json
import time
import threading
from typing import Optional
from loguru import logger

try:
    import aiohttp
except ImportError:
    aiohttp = None

# Constants
RTDS_WS_URL = "wss://ws-live-data.polymarket.com"
PING_INTERVAL_SEC = 30.0
MAX_RECONNECT_ATTEMPTS = 50
RECONNECT_DELAY_SEC = 5.0


def _make_tick(source, symbol, price, src_ts, rx_ts, lat):
    """Create a PriceTick without circular import."""
    from rtds_connector import PriceTick
    return PriceTick(source=source, symbol=symbol, price=price,
                     source_timestamp_ms=src_ts, received_timestamp_ms=rx_ts,
                     latency_ms=lat)


class RTDSWebSocketMixin:
    """WebSocket connection, subscription, and message handling for RTDSConnector."""

    async def connect(self):
        """Connect to RTDS WebSocket and stream with auto-reconnect."""
        if aiohttp is None:
            logger.error("aiohttp required: pip install aiohttp"); return
        self._should_stop = False
        while not self._should_stop and self._reconnect_count < MAX_RECONNECT_ATTEMPTS:
            try:
                await self._connect_and_stream()
            except asyncio.CancelledError:
                logger.info("RTDS cancelled"); break
            except Exception as e:
                self._reconnect_count += 1
                self._connected = False
                logger.warning(f"RTDS lost ({e}), reconnect in {RECONNECT_DELAY_SEC}s "
                               f"({self._reconnect_count}/{MAX_RECONNECT_ATTEMPTS})")
                await asyncio.sleep(RECONNECT_DELAY_SEC)
        if self._reconnect_count >= MAX_RECONNECT_ATTEMPTS:
            logger.error(f"RTDS: max reconnects ({MAX_RECONNECT_ATTEMPTS}) reached")

    async def disconnect(self):
        """Gracefully disconnect."""
        self._should_stop = True
        self._connected = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("RTDS disconnected")

    def start_background(self, loop=None):
        """Start RTDS in a background thread."""
        def _run():
            _l = asyncio.new_event_loop()
            asyncio.set_event_loop(_l)
            try:
                _l.run_until_complete(self.connect())
            except Exception as e:
                logger.error(f"RTDS background error: {e}")
            finally:
                _l.close()
        t = threading.Thread(target=_run, daemon=True, name="rtds-connector")
        t.start()
        logger.info("RTDS started (background thread)")
        return t

    async def _connect_and_stream(self):
        """Establish WS, subscribe, process messages."""
        self._session = aiohttp.ClientSession()
        try:
            async with self._session.ws_connect(
                RTDS_WS_URL, heartbeat=PING_INTERVAL_SEC,
                timeout=aiohttp.ClientWSTimeout(ws_close=10),
            ) as ws:
                self._ws = ws
                self._connected = True
                self._reconnect_count = 0
                logger.info(f"✓ RTDS connected: {RTDS_WS_URL}")
                await self._subscribe(ws)
                await self._process_messages(ws)
        finally:
            self._connected = False
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

    async def _subscribe(self, ws):
        """Subscribe to Binance + Chainlink feeds."""
        await ws.send_json({"action": "subscribe", "subscriptions": [
            {"topic": "crypto_prices", "type": "update", "filters": "btcusdt"}]})
        await ws.send_json({"action": "subscribe", "subscriptions": [
            {"topic": "crypto_prices_chainlink", "type": "*",
             "filters": json.dumps({"symbol": "btc/usd"})}]})
        logger.info("RTDS: subscribed (Binance + Chainlink)")

    async def _process_messages(self, ws):
        """Process WS messages until stop/error/close."""
        async for msg in ws:
            if self._should_stop:
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                self._handle_message(msg.data)
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                break

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
        src_ts = int(payload.get("timestamp", msg_ts))
        rx_ts = int(time.time() * 1000)
        lat = rx_ts - src_ts if src_ts > 0 else 0
        sym = payload.get("symbol", "").lower()
        if topic == "crypto_prices" and "btc" in sym:
            self._handle_binance_tick(price, src_ts, rx_ts, lat, payload)
        elif topic == "crypto_prices_chainlink" and "btc" in sym:
            self._handle_chainlink_tick(price, src_ts, rx_ts, lat, payload)

    def _handle_binance_tick(self, price, src_ts, rx_ts, lat, payload):
        tick = _make_tick("binance", payload.get("symbol", "btcusdt"),
                          price, src_ts, rx_ts, lat)
        with self._lock:
            self._binance_price = price
            self._binance_ts = rx_ts
        self._binance_ticks.append(tick)
        self._binance_msg_count += 1
        if self.vol_estimator:
            self.vol_estimator.add_price(price, rx_ts / 1000.0)
        if self._on_binance_tick:
            try: self._on_binance_tick(tick)
            except Exception: pass

    def _update_latency(self, latency):
        if self._chainlink_msg_count > 1:
            self._avg_latency_ms = 0.95 * self._avg_latency_ms + 0.05 * latency
        else:
            self._avg_latency_ms = float(latency)

    def _handle_chainlink_tick(self, price, src_ts, rx_ts, lat, payload):
        tick = _make_tick("chainlink", payload.get("symbol", "btc/usd"),
                          price, src_ts, rx_ts, lat)
        with self._lock:
            self._chainlink_price = price
            self._chainlink_ts = rx_ts
        self._chainlink_ticks.append(tick)
        self._chainlink_msg_count += 1
        if self.vol_estimator:
            self.vol_estimator.add_price(price, rx_ts / 1000.0)
        if self._on_chainlink_tick:
            try: self._on_chainlink_tick(tick)
            except Exception: pass
        self._update_latency(lat)




