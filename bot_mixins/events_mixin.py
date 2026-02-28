"""
EventsMixin — order events, lifecycle callbacks, shutdown.

SRP: Handles Nautilus order events and strategy lifecycle.
"""
import asyncio
import threading
from datetime import datetime, timezone
from loguru import logger


class EventsMixin:
    """Order event handlers, Grafana sync, stop/cleanup."""

    def _track_order_event(self, event_type: str):
        try:
            pt = self.performance_tracker
            if hasattr(pt, 'record_order_event'):
                pt.record_order_event(event_type)
            elif hasattr(pt, 'increment_counter'):
                pt.increment_counter(event_type)
        except Exception as e:
            logger.warning(f"Track event '{event_type}' failed: {e}")

    def on_order_filled(self, event):
        logger.info(f"ORDER FILLED: {event.client_order_id} @ ${float(event.last_px):.4f}")
        self._track_order_event("filled")
        oid = str(event.client_order_id)
        updated = self._pos_tracker.update_entry_price(oid, float(event.last_px))
        if updated:
            logger.info(f"  ✓ Entry updated: ${updated.entry_price:.4f}")
            if self._active_entry:
                self._active_entry["entry_price"] = updated.entry_price

    def on_order_denied(self, event):
        logger.error(f"ORDER DENIED: {event.client_order_id} — {event.reason}")
        self._track_order_event("rejected")

    def on_order_rejected(self, event):
        from bot_mixins._constants import MAX_RETRIES_PER_WINDOW
        reason = str(getattr(event, 'reason', ''))
        rl = reason.lower()
        if 'no orders found' in rl or 'fak' in rl or 'no match' in rl:
            self._retry_count_this_window += 1
            if self._retry_count_this_window <= MAX_RETRIES_PER_WINDOW:
                logger.warning(f"FAK rejected — retry {self._retry_count_this_window}/{MAX_RETRIES_PER_WINDOW}")
                self.last_trade_time = -1
            else:
                logger.warning(f"FAK rejected — max retries reached")
        else:
            logger.warning(f"Order rejected: {reason}")

    # ── Lifecycle ────────────────────────────────────────────────────

    def _subscribe_and_seed_price(self):
        if not self.instrument_id:
            return
        self.subscribe_quote_ticks(self.instrument_id)
        no_id = getattr(self, '_no_instrument_id', None)
        if no_id:
            self.subscribe_quote_ticks(no_id)
        try:
            q = self.cache.quote_tick(self.instrument_id)
            if q and q.bid_price and q.ask_price:
                p = (q.bid_price + q.ask_price) / 2
                self.price_history.append(p)
                logger.info(f"✓ Initial price: ${float(p):.4f}")
        except Exception as e:
            logger.debug(f"No initial price: {e}")

    def _init_services(self):
        self.run_in_executor(self._start_timer_loop)
        if self.rtds:
            self.rtds.start_background()
            logger.info("✓ RTDS started")
        if self.funding_filter:
            try:
                r = self.funding_filter.update_sync()
                if r:
                    logger.info(f"✓ Funding: {r.funding_rate_pct:+.4f}% → {r.classification}")
            except Exception:
                pass
        if self.grafana_exporter:
            threading.Thread(target=self._start_grafana_sync, daemon=True).start()

    def on_start(self):
        logger.info("Strategy starting — loading instruments...")
        self._load_all_btc_instruments()
        self._subscribe_and_seed_price()
        self._init_services()
        logger.info(f"V3.1 active — {len(self.price_history)} price points")

    def _start_grafana_sync(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.grafana_exporter.start())
        except Exception as e:
            logger.error(f"Grafana start failed: {e}")

    def on_stop(self):
        logger.info("Strategy V3.1 stopped")
        logger.info(f"Paper trades: {len(self.paper_trades)}")
        for pos in self._pos_tracker.unresolved():
            logger.info(f"Unresolved: {pos.market_slug} {pos.direction} @ ${pos.entry_price:.4f}")
        self._log_stop_stats()
        self._cleanup_connections()

    def _log_stop_stats(self):
        s = self.mispricing_detector.get_stats()
        logger.info(f"Mispricing: {s['tradeable_detections']}/{s['total_detections']} ({s['hit_rate']:.0%})")
        logger.info(f"Vol: {self.vol_estimator.get_stats()}")
        if self.rtds:
            r = self.rtds.get_stats()
            logger.info(f"RTDS: {r['chainlink_ticks']}CL, {r['binance_ticks']}BN")
        if self.funding_filter:
            logger.info(f"Funding: {self.funding_filter.get_stats()}")

    def _cleanup_connections(self):
        if self.rtds:
            try:
                l = asyncio.new_event_loop()
                l.run_until_complete(self.rtds.disconnect()); l.close()
            except Exception: pass
        if self.grafana_exporter:
            try:
                l = asyncio.new_event_loop()
                l.run_until_complete(self.grafana_exporter.stop())
            except Exception: pass

