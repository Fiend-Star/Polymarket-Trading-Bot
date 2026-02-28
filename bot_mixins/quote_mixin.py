"""
QuoteHandlerMixin — quote tick processing, timer loop, market stability.

SRP: Real-time tick ingestion and time-based market management.
"""
import asyncio
import math
import os
from datetime import datetime, timezone
from loguru import logger
from nautilus_trader.model.data import QuoteTick


class QuoteHandlerMixin:
    """Methods for processing quote ticks, timer loop, and stability gating."""

    def _extract_mid_price(self, tick):
        if self.instrument_id is None or tick.instrument_id != self.instrument_id:
            return None
        bid, ask = tick.bid_price, tick.ask_price
        if bid is None or ask is None:
            return None
        try:
            bd, ad = bid.as_decimal(), ask.as_decimal()
        except Exception:
            return None
        return (bd + ad) / 2, bd, ad

    def on_quote_tick(self, tick: QuoteTick):
        try:
            result = self._extract_mid_price(tick)
            if result is None:
                return
            mid, bid_d, ask_d = result
            self._update_price_tracking(mid, bid_d, ask_d)
            if not self._ensure_market_stable(bid_d, ask_d):
                return
            if self._waiting_for_market_open:
                return
            if not (0 <= self.current_instrument_index < len(self.all_btc_instruments)):
                return
            mkt = self.all_btc_instruments[self.current_instrument_index]
            elapsed, sub_sec = self._compute_market_timing(mkt)
            if elapsed < 0:
                return
            self._check_early_window_trade(mkt, elapsed, sub_sec, mid, bid_d, ask_d)
            self._check_late_window_trade(mkt, sub_sec, mid)
        except Exception as e:
            logger.error(f"Error processing quote tick: {e}")

    def _update_price_tracking(self, mid, bid_d, ask_d):
        now = datetime.now(timezone.utc)
        self.price_history.append(mid)
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        self._last_bid_ask = (bid_d, ask_d)
        self._tick_buffer.append({'ts': now, 'price': mid})

    def _ensure_market_stable(self, bid_d, ask_d):
        from bot_mixins._constants import QUOTE_STABILITY_REQUIRED
        if not self._market_stable:
            if self._is_quote_valid(bid_d, ask_d):
                self._stable_tick_count += 1
                if self._stable_tick_count >= QUOTE_STABILITY_REQUIRED:
                    self._market_stable = True
                    logger.info(f"✓ Market STABLE after {QUOTE_STABILITY_REQUIRED} valid ticks")
                    return True
                return False
            else:
                self._stable_tick_count = 0
                return False
        return True

    def _compute_market_timing(self, mkt):
        from bot_mixins._constants import MARKET_INTERVAL_SECONDS
        now_ts = datetime.now(timezone.utc).timestamp()
        elapsed = now_ts - mkt['market_timestamp']
        return elapsed, elapsed % MARKET_INTERVAL_SECONDS

    def _check_early_window_trade(self, mkt, elapsed, sub_sec, mid, bid_d, ask_d):
        from bot_mixins._constants import (
            TRADE_WINDOW_START_SEC, TRADE_WINDOW_END_SEC, MARKET_INTERVAL_SECONDS)
        sub_interval = int(elapsed // MARKET_INTERVAL_SECONDS)
        key = (mkt['market_timestamp'], sub_interval)
        if TRADE_WINDOW_START_SEC <= sub_sec < TRADE_WINDOW_END_SEC and key != self.last_trade_time:
            self.last_trade_time = key
            self._retry_count_this_window = 0
            now = datetime.now(timezone.utc)
            logger.info(f"🎯 EARLY-WINDOW: {now.strftime('%H:%M:%S')} | {mkt['slug']} | "
                        f"YES=${float(mid):,.4f}")
            self.run_in_executor(lambda: self._make_trading_decision_sync(float(mid)))

    def _check_late_window_trade(self, mkt, sub_sec, mid):
        from bot_mixins._constants import (
            LATE_WINDOW_ENABLED, RTDS_LATE_WINDOW_SEC, MARKET_INTERVAL_SECONDS)
        remaining = MARKET_INTERVAL_SECONDS - sub_sec
        if not (LATE_WINDOW_ENABLED and self.rtds and self._btc_strike_price
                and 0 < remaining <= RTDS_LATE_WINDOW_SEC
                and not self._late_window_traded):
            return
        sig = self.rtds.get_late_window_signal(
            strike=self._btc_strike_price, time_remaining_sec=remaining)
        if sig.direction != "NO_SIGNAL" and sig.confidence >= 0.70:
            self._late_window_traded = True
            logger.info(f"🎯 LATE-WINDOW: T-{remaining:.0f}s | {sig.direction} | "
                        f"conf={sig.confidence:.0%} | {sig.delta_bps:+.1f}bps")
            self.run_in_executor(
                lambda: self._execute_late_window_trade(sig, float(mid)))

    # ── Timer loop ───────────────────────────────────────────────────

    def _start_timer_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._timer_loop())
        finally:
            loop.close()

    async def _check_auto_restart(self):
        uptime = (datetime.now(timezone.utc) - self.bot_start_time).total_seconds() / 60
        if uptime >= self.restart_after_minutes:
            logger.warning("AUTO-RESTART — Loading fresh filters")
            await self._teardown_data_sources()
            import signal as _signal
            os.kill(os.getpid(), _signal.SIGTERM)
            return True
        return False

    def _record_strike_if_needed(self):
        spot = self._data_mgr.cached_spot
        if not self._strike_recorded and spot:
            self._btc_strike_price = spot
            self._strike_recorded = True
            logger.info(f"📌 Strike: ${self._btc_strike_price:,.2f}")

    async def _timer_loop(self):
        await self._init_data_sources()
        await self._preseed_vol_estimator()
        while True:
            if await self._check_auto_restart():
                return
            now = datetime.now(timezone.utc)
            if self.next_switch_time and now >= self.next_switch_time:
                if self._waiting_for_market_open:
                    self._handle_market_open(now)
                else:
                    self._switch_to_next_market()
            try:
                await self._refresh_cached_data()
                self._record_strike_if_needed()
            except Exception as e:
                logger.debug(f"Cache refresh error: {e}")
            await asyncio.sleep(10)

