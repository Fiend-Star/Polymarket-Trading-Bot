"""
TradingDecisionMixin — quant model + heuristic trading decisions.

SRP: Decides WHEN and WHAT to trade. Does not execute orders.
"""
import asyncio
import math
from decimal import Decimal
from datetime import datetime, timezone
from types import SimpleNamespace
from loguru import logger


class TradingDecisionMixin:
    """Binary-option quant model + heuristic fallback decisions."""

    def _make_trading_decision_sync(self, current_price):
        price_decimal = Decimal(str(current_price))
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._make_trading_decision(price_decimal))
        finally:
            loop.close()

    def _execute_late_window_trade(self, late_signal, current_yes_price):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self._execute_late_window_trade_async(late_signal, current_yes_price))
        finally:
            loop.close()

    async def _execute_late_window_trade_async(self, late_signal, current_yes_price):
        from bot_mixins._constants import POSITION_SIZE_USD
        is_sim = await self.check_simulation_mode()
        direction = "long" if late_signal.direction == "BUY_YES" else "short"
        lp = Decimal(str(min(0.93, 0.88 + 0.05 * late_signal.confidence)))
        is_valid, error = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD, direction=direction,
            current_price=Decimal(str(current_yes_price)))
        if not is_valid:
            logger.warning(f"Late-window risk fail: {error}"); return
        mock = SimpleNamespace(
            direction="BULLISH" if direction == "long" else "BEARISH",
            score=late_signal.confidence * 100, confidence=late_signal.confidence)
        if is_sim:
            await self._record_paper_trade(mock, POSITION_SIZE_USD, lp, direction)
        else:
            await self._place_limit_order(mock, POSITION_SIZE_USD, lp, direction)

    async def _make_trading_decision(self, current_price: Decimal):
        from bot_mixins._constants import POSITION_SIZE_USD, VOL_METHOD, USE_LIMIT_ORDERS
        from binary_pricer import BinaryOptionPricer
        is_sim = await self.check_simulation_mode()
        logger.info(f"Mode: {'SIM' if is_sim else 'LIVE'}")
        if len(self.price_history) < 5:
            logger.warning(f"Not enough history ({len(self.price_history)}/5)"); return
        cached = self._get_cached_data()
        btc_spot = cached.get("spot_price")
        fb = self._log_cached_data(cached, btc_spot)
        if btc_spot is None or self._btc_strike_price is None:
            logger.warning("No spot/strike — heuristic fallback")
            await self._make_trading_decision_heuristic(current_price, cached, is_sim)
            return
        mkt = self.all_btc_instruments[self.current_instrument_index]
        trm = max(0, (mkt['end_timestamp'] - datetime.now(timezone.utc).timestamp()) / 60.0)
        logger.info(f"BTC: spot=${btc_spot:,.2f}, strike=${self._btc_strike_price:,.2f}, T={trm:.1f}min")
        yes_p, no_p = self._get_market_prices(current_price)
        signal = self._run_quant_signal(btc_spot, trm, yes_p, no_p, fb)
        if not signal.is_tradeable:
            logger.info(f"NO TRADE: edge=${signal.edge:+.4f}"); return
        if self._check_confirmation_veto(signal, cached, current_price):
            return
        direction = "long" if signal.direction == "BUY_YES" else "short"
        await self._execute_quant_trade(signal, direction, current_price, no_p, is_sim)

    def _log_cached_data(self, cached, btc_spot):
        if btc_spot:
            src = "Chainlink" if (self.rtds and self.rtds.chainlink_btc_price > 0) else "Coinbase"
            logger.info(f"BTC spot ({src}): ${btc_spot:,.2f}")
        if cached.get("sentiment_score") is not None:
            logger.info(f"F&G: {cached['sentiment_score']:.0f} ({cached.get('sentiment_classification','')})")
        fb = 0.0
        if self.funding_filter:
            r = self.funding_filter.get_regime()
            fb = r.mean_reversion_bias
            if r.classification != "NEUTRAL":
                logger.info(f"Funding: {r.classification} ({r.funding_rate_pct:+.4f}%), bias={fb:+.3f}")
        return fb

    def _get_market_prices(self, current_price):
        yes_p = float(current_price)
        no_p = 1.0 - yes_p
        if self._no_instrument_id:
            try:
                q = self.cache.quote_tick(self._no_instrument_id)
                if q:
                    no_p = float((q.bid_price.as_decimal() + q.ask_price.as_decimal()) / 2)
            except Exception:
                pass
        return yes_p, no_p

    def _run_quant_signal(self, btc_spot, trm, yes_p, no_p, fb):
        from bot_mixins._constants import VOL_METHOD, POSITION_SIZE_USD, USE_LIMIT_ORDERS
        from binary_pricer import BinaryOptionPricer
        skew = BinaryOptionPricer.estimate_btc_vol_skew(
            btc_spot, self._btc_strike_price,
            self.vol_estimator.get_vol(VOL_METHOD).annualized_vol, trm)
        return self.mispricing_detector.detect(
            yes_market_price=yes_p, no_market_price=no_p,
            btc_spot=btc_spot, btc_strike=self._btc_strike_price,
            time_remaining_min=trm, position_size_usd=float(POSITION_SIZE_USD),
            use_maker=USE_LIMIT_ORDERS, vol_skew=skew, funding_bias=fb)

    def _check_confirmation_veto(self, signal, cached, current_price):
        md = self._build_metadata(cached, current_price)
        sigs = self._process_signals(current_price, md)
        conf, contra = self._count_confirmations(signal, sigs)
        if contra > conf and signal.confidence < 0.6:
            logger.info(f"⏭ SKIP: {contra}/{len(sigs)} disagree"); return True
        return False

    def _count_confirmations(self, signal, old_signals):
        bull = signal.direction == "BUY_YES"
        c = d = 0
        for s in old_signals:
            if ("BULLISH" in str(s.direction).upper()) == bull:
                c += 1
            else:
                d += 1
        logger.info(f"Confirmation: {c} agree, {d} disagree (of {len(old_signals)})")
        return c, d

    async def _execute_quant_trade(self, signal, direction, current_price, no_p, is_sim):
        from bot_mixins._constants import POSITION_SIZE_USD
        label = "BUY YES" if direction == "long" else "BUY NO"
        logger.info(f"{'📈' if direction=='long' else '📉'} {label} edge=${signal.edge:+.4f}")
        is_valid, err = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD, direction=direction, current_price=current_price)
        if not is_valid:
            logger.warning(f"Risk blocked: {err}"); return
        if not self._check_liquidity(direction):
            return
        await self._dispatch_order(signal, direction, current_price, is_sim)
        self._active_entry = {
            "direction": signal.direction,
            "entry_price": float(current_price) if direction == "long" else no_p,
            "entry_time": datetime.now(timezone.utc),
            "btc_strike": self._btc_strike_price}

    def _check_liquidity(self, direction):
        last = getattr(self, '_last_bid_ask', None)
        if not last:
            return True
        bid, ask = last
        MIN_LIQ = Decimal("0.02")
        if direction == "long" and ask <= MIN_LIQ:
            logger.warning(f"No liquidity BUY YES: ask=${float(ask):.4f}")
            self.last_trade_time = -1; return False
        if direction == "short" and bid <= MIN_LIQ:
            logger.warning(f"No liquidity BUY NO: bid=${float(bid):.4f}")
            self.last_trade_time = -1; return False
        return True

    async def _make_trading_decision_heuristic(self, current_price, cached, is_sim):
        from bot_mixins._constants import POSITION_SIZE_USD, USE_LIMIT_ORDERS
        md = self._build_metadata(cached, current_price)
        sigs = self._process_signals(current_price, md)
        if not sigs:
            logger.info("No signals — no trade"); return
        fused, direction = self._fuse_heuristic_signals(sigs, current_price)
        if fused is None:
            return
        ok, err = self.risk_engine.validate_new_position(
            size=POSITION_SIZE_USD, direction=direction, current_price=current_price)
        if not ok:
            logger.warning(f"Risk blocked: {err}"); return
        if is_sim:
            await self._record_paper_trade(fused, POSITION_SIZE_USD, current_price, direction)
        elif USE_LIMIT_ORDERS:
            await self._place_limit_order(fused, POSITION_SIZE_USD, current_price, direction)
        else:
            await self._place_real_order(fused, POSITION_SIZE_USD, current_price, direction)

