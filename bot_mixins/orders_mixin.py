"""
OrdersMixin — order execution, paper trades, signal processing.

SRP: Constructs and submits orders (paper, limit, market).
"""
import json
import math
import time
from decimal import Decimal
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List

from loguru import logger
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity, Price


class OrdersMixin:
    """Paper trades, limit orders, market orders, signal processing."""

    # ── Signal processing ────────────────────────────────────────────

    def _process_signals(self, current_price, metadata=None):
        if metadata is None:
            metadata = {}
        pm = self._prepare_signal_metadata(metadata)
        return self._run_all_processors(current_price, pm)

    def _prepare_signal_metadata(self, metadata):
        return {k: Decimal(str(v)) if isinstance(v, float) else v
                for k, v in metadata.items()}

    def _run_all_processors(self, current_price, pm):
        signals = []
        args = {"current_price": current_price,
                "historical_prices": self.price_history, "metadata": pm}
        dispatch = [
            (self.spike_detector, None), (self.sentiment_processor, 'sentiment_score'),
            (self.divergence_processor, 'spot_price'),
            (self.orderbook_processor, 'yes_token_id'),
            (self.tick_velocity_processor, 'tick_buffer'),
            (self.deribit_pcr_processor, None)]
        for proc, key in dispatch:
            if key and not pm.get(key):
                continue
            s = proc.process(**args)
            if s:
                signals.append(s)
        return signals

    def _build_metadata(self, cached: dict, current_price: Decimal) -> dict:
        cpf = float(current_price)
        recent = [float(p) for p in self.price_history[-20:]]
        md = {"tick_buffer": list(self._tick_buffer), "yes_token_id": self._yes_token_id}
        if len(recent) >= 5:
            sma = sum(recent) / len(recent)
            md["deviation"] = (cpf - sma) / sma if sma else 0.0
            md["momentum"] = ((cpf - float(self.price_history[-5])) / float(self.price_history[-5])
                              if len(self.price_history) >= 5 else 0.0)
            md["volatility"] = math.sqrt(sum((p - sma)**2 for p in recent) / len(recent))
        if cached.get("spot_price"):
            md["spot_price"] = cached["spot_price"]
        if cached.get("sentiment_score") is not None:
            md["sentiment_score"] = cached["sentiment_score"]
            md["sentiment_classification"] = cached.get("sentiment_classification", "")
        return md

    def _fuse_heuristic_signals(self, signals, current_price):
        from bot_mixins._constants import (
            MIN_FUSION_SIGNALS, MIN_FUSION_SCORE, TREND_UP_THRESHOLD, TREND_DOWN_THRESHOLD)
        fused = self.fusion_engine.fuse_signals(
            signals, min_signals=MIN_FUSION_SIGNALS, min_score=MIN_FUSION_SCORE)
        if not fused:
            logger.info(f"Fusion: no signal (need {MIN_FUSION_SIGNALS}+)"); return None, None
        pf = float(current_price)
        if TREND_DOWN_THRESHOLD <= pf <= TREND_UP_THRESHOLD and fused.confidence < 0.70:
            logger.info("⏭ Coin-flip zone, weak conf"); return None, None
        d = "long" if "BULLISH" in str(fused.direction).upper() else "short"
        return fused, d

    async def _dispatch_order(self, signal, direction, current_price, is_sim):
        from bot_mixins._constants import POSITION_SIZE_USD, USE_LIMIT_ORDERS
        mock = SimpleNamespace(
            direction="BULLISH" if direction == "long" else "BEARISH",
            score=signal.confidence * 100, confidence=signal.confidence)
        if is_sim:
            await self._record_paper_trade(mock, POSITION_SIZE_USD, current_price, direction)
        elif USE_LIMIT_ORDERS:
            await self._place_limit_order(mock, POSITION_SIZE_USD, current_price, direction)
        else:
            await self._place_real_order(mock, POSITION_SIZE_USD, current_price, direction)

    # ── Paper trades ─────────────────────────────────────────────────

    async def _record_paper_trade(self, signal, size, price, direction):
        from order_dispatcher import PaperTrade
        if not (0 <= self.current_instrument_index < len(self.all_btc_instruments)):
            logger.warning("No active market"); return
        mkt = self.all_btc_instruments[self.current_instrument_index]
        oid = f"paper_{int(time.time()*1000)}"
        iid = (getattr(self, '_yes_instrument_id', self.instrument_id) if direction == "long"
               else getattr(self, '_no_instrument_id', None) or self.instrument_id)
        tok = "YES (UP)" if direction == "long" else "NO (DOWN)"
        self._pos_tracker.add(market_slug=mkt['slug'], direction=direction,
                              entry_price=float(price), size_usd=float(size),
                              market_end_time=mkt['end_time'], instrument_id=iid, order_id=oid)
        self.paper_trades.append(PaperTrade(
            timestamp=datetime.now(timezone.utc), direction=direction.upper(),
            size_usd=float(size), price=float(price),
            signal_score=signal.score, signal_confidence=signal.confidence))
        logger.info(f"[SIM] {direction.upper()} {tok} ${float(size):.2f} @ ${float(price):,.4f}")
        self._save_paper_trades()

    def _update_paper_trade_outcome(self, pos, outcome):
        for pt in self.paper_trades:
            if (pt.outcome == "PENDING" and pt.direction == pos.direction.upper()
                    and abs(pt.price - pos.entry_price) < 0.0001):
                pt.outcome = outcome; break
        self._save_paper_trades()

    def _save_paper_trades(self):
        try:
            with open('paper_trades.json', 'w') as f:
                json.dump([t.to_dict() for t in self.paper_trades], f, indent=2)
        except Exception as e:
            logger.error(f"Save paper trades failed: {e}")

    # ── Limit orders ─────────────────────────────────────────────────

    async def _place_limit_order(self, signal, size, price, direction):
        if not self.instrument_id:
            logger.error("No instrument"); return
        try:
            r = self._resolve_trade_instrument(direction)
            if not r: return
            iid, label, inst = r
            bid, ask = self._get_token_book(iid, price)
            lp = max(Decimal("0.01"), min(Decimal("0.99"), bid))
            qty = self._compute_token_qty(size, lp, inst)
            logger.info(f"LIMIT {label}: {qty:.2f} tok @ ${float(lp):.4f}")
            self._submit_limit_order(iid, inst, qty, lp, label, size, direction)
        except Exception as e:
            logger.error(f"Limit order error: {e}")
            self._track_order_event("rejected")

    def _resolve_trade_instrument(self, direction):
        if direction == "long":
            iid = getattr(self, '_yes_instrument_id', self.instrument_id)
            label = "YES (UP)"
        else:
            iid = getattr(self, '_no_instrument_id', None)
            if not iid:
                logger.warning("NO token not found"); return None
            label = "NO (DOWN)"
        inst = self.cache.instrument(iid)
        if not inst:
            logger.error(f"Not in cache: {iid}"); return None
        return iid, label, inst

    def _get_token_book(self, iid, fallback):
        try:
            q = self.cache.quote_tick(iid)
            if q: return q.bid_price.as_decimal(), q.ask_price.as_decimal()
        except Exception: pass
        return fallback, fallback + Decimal("0.02")

    def _compute_token_qty(self, size, price, inst):
        return max(round(float(size / price), inst.size_precision), 5.0)

    def _submit_limit_order(self, iid, inst, qty, lp, label, size, direction):
        q = Quantity(qty, precision=inst.size_precision)
        p = Price(float(lp), precision=inst.price_precision)
        uid = f"BTC-15MIN-V3-{int(time.time()*1000)}"
        order = self.order_factory.limit(
            instrument_id=iid, order_side=OrderSide.BUY, quantity=q, price=p,
            client_order_id=ClientOrderId(uid), time_in_force=TimeInForce.GTC)
        self.submit_order(order)
        logger.info(f"LIMIT SUBMITTED: {uid} | {label}")
        self._track_open_position(direction, float(lp), float(size), iid, uid)

    # ── Market orders ────────────────────────────────────────────────

    async def _place_real_order(self, signal, size, price, direction):
        if not self.instrument_id:
            logger.error("No instrument"); return
        try:
            r = self._resolve_trade_instrument(direction)
            if not r: return
            iid, label, inst = r
            self._submit_market_order(iid, label, inst, float(price), float(size), direction)
        except Exception as e:
            logger.error(f"Market order error: {e}")
            self._track_order_event("rejected")

    def _submit_market_order(self, iid, label, inst, price, size, direction):
        prec = inst.size_precision
        qty = round(max(float(getattr(inst, 'min_quantity', None) or 5.0), 5.0), prec)
        q = Quantity(qty, precision=prec)
        uid = f"BTC-15MIN-MKT-{int(time.time()*1000)}"
        order = self.order_factory.market(
            instrument_id=iid, order_side=OrderSide.BUY, quantity=q,
            client_order_id=ClientOrderId(uid), quote_quantity=False,
            time_in_force=TimeInForce.IOC)
        self.submit_order(order)
        logger.info(f"MARKET SUBMITTED: {uid} | {label}")
        self._track_open_position(direction, price, size, iid, uid)

    def _track_open_position(self, direction, entry, size, iid, oid):
        mkt = self.all_btc_instruments[self.current_instrument_index]
        self._pos_tracker.add(market_slug=mkt['slug'], direction=direction,
                              entry_price=entry, size_usd=size,
                              market_end_time=mkt['end_time'], instrument_id=iid, order_id=oid)
        self._track_order_event("placed")

