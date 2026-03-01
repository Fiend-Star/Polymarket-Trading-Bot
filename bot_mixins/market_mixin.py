"""
MarketLifecycleMixin — instrument loading, pairing, switching, resolution.

SRP: Market discovery and rotation lifecycle.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger


class MarketLifecycleMixin:
    """Methods for loading, pairing, subscribing, and switching BTC 15-min markets."""

    def _load_all_btc_instruments(self):
        instruments = self.cache.instruments()
        logger.info(f"Loading ALL BTC instruments from {len(instruments)} total...")
        now = datetime.now(timezone.utc)
        current_timestamp = int(now.timestamp())
        btc_instruments = self._parse_btc_instruments(instruments, current_timestamp)
        btc_instruments = self._dedup_and_pair_instruments(btc_instruments)
        if btc_instruments:
            first_t = btc_instruments[0]['start_time'].strftime('%H:%M')
            last_t = btc_instruments[-1]['end_time'].strftime('%H:%M')
            logger.info(f"Found {len(btc_instruments)} BTC 15-min markets ({first_t}–{last_t})")
        else:
            logger.warning("No BTC 15-min markets found!")
        self.all_btc_instruments = btc_instruments
        self._activate_current_instrument(btc_instruments, current_timestamp)

    def _parse_btc_instruments(self, instruments, current_timestamp):
        btc_instruments = []
        for instrument in instruments:
            try:
                if not (hasattr(instrument, 'info') and instrument.info):
                    continue
                q = instrument.info.get('question', '').lower()
                slug = instrument.info.get('market_slug', '').lower()
                if not (('btc' in q or 'btc' in slug) and '15m' in slug):
                    continue
                parsed = self._parse_single_instrument(instrument, slug, current_timestamp)
                if parsed:
                    btc_instruments.append(parsed)
            except Exception:
                continue
        return btc_instruments

    def _parse_single_instrument(self, instrument, slug, current_timestamp):
        try:
            ts_part = slug.split('-')[-1]
            mkt_ts = int(ts_part)
            end_ts = mkt_ts + 900
            if end_ts <= current_timestamp:
                return None
            raw_id = str(instrument.id)
            base = raw_id.split('.')[0] if '.' in raw_id else raw_id
            tok = base.split('-')[-1] if '-' in base else base
            return {
                'instrument': instrument, 'slug': slug,
                'start_time': datetime.fromtimestamp(mkt_ts, tz=timezone.utc),
                'end_time': datetime.fromtimestamp(end_ts, tz=timezone.utc),
                'market_timestamp': mkt_ts, 'end_timestamp': end_ts,
                'time_diff_minutes': (mkt_ts - current_timestamp) / 60,
                'yes_token_id': tok,
            }
        except (ValueError, IndexError):
            return None

    def _dedup_and_pair_instruments(self, btc_instruments):
        """
        Deduplicate and pair YES/NO tokens by slug using case-insensitive outcome parsing.
        V3.4 FIX: Use outcome.lower() for case-insensitive comparison.
        """
        seen_slugs = {}
        for inst in btc_instruments:
            slug = inst['slug']
            instrument = inst['instrument']
            outcome = getattr(instrument, 'outcome', None)
            outcome_str = outcome.lower() if outcome else ""

            if slug not in seen_slugs:
                base_inst = inst.copy()
                base_inst['yes_instrument_id'] = None
                base_inst['no_instrument_id'] = None
                base_inst['yes_token_id'] = inst.get('yes_token_id')
                seen_slugs[slug] = base_inst

            if outcome_str in ('yes', 'up'):
                seen_slugs[slug]['yes_instrument_id'] = instrument.id
                if 'yes_token_id' not in seen_slugs[slug] or seen_slugs[slug]['yes_token_id'] is None:
                    seen_slugs[slug]['yes_token_id'] = inst.get('yes_token_id')
            elif outcome_str in ('no', 'down'):
                seen_slugs[slug]['no_instrument_id'] = instrument.id

        deduped = list(seen_slugs.values())
        deduped.sort(key=lambda x: x['market_timestamp'])
        return deduped

    def _activate_current_instrument(self, btc_instruments, current_timestamp):
        for i, inst in enumerate(btc_instruments):
            if inst['time_diff_minutes'] <= 0 and inst['end_timestamp'] > current_timestamp:
                self._subscribe_to_market(i, inst)
                logger.info(f"✓ Active: {inst['slug']} → switch at {self.next_switch_time.strftime('%H:%M:%S')}")
                return
        if not btc_instruments:
            return
        future = [x for x in btc_instruments if x['time_diff_minutes'] > 0]
        nearest = min(future, key=lambda x: x['time_diff_minutes']) if future else btc_instruments[-1]
        idx = btc_instruments.index(nearest)
        self._subscribe_to_market(idx, nearest)
        self.next_switch_time = nearest['start_time']
        logger.info(f"⚠ WAITING FOR: {nearest['slug']}")
        self._waiting_for_market_open = True

    def _subscribe_to_market(self, index, inst):
        self.current_instrument_index = index
        self.instrument_id = inst['instrument'].id
        self.next_switch_time = inst['end_time']
        self._yes_token_id = inst.get('yes_token_id')
        self._yes_instrument_id = inst.get('yes_instrument_id', inst['instrument'].id)
        self._no_instrument_id = inst.get('no_instrument_id')
        self.subscribe_quote_ticks(self.instrument_id)
        if inst.get('no_instrument_id'):
            self.subscribe_quote_ticks(inst['no_instrument_id'])

    def _switch_to_next_market(self):
        from bot_mixins import _constants as C
        if not self.all_btc_instruments:
            logger.error("No instruments loaded!"); return False
        nxt = self.current_instrument_index + 1
        if nxt >= len(self.all_btc_instruments):
            logger.warning("No more markets — will restart bot"); return False
        m = self.all_btc_instruments[nxt]
        now = datetime.now(timezone.utc)
        if now < m['start_time']:
            logger.info(f"Waiting for next market at {m['start_time'].strftime('%H:%M:%S')}"); return False
        self._resolve_positions_for_market(self.current_instrument_index)
        self._subscribe_to_market(nxt, m)
        logger.info(f"SWITCHING: {m['slug']} ends {self.next_switch_time.strftime('%H:%M:%S')}")
        self._reset_market_state()
        return True

    def _reset_market_state(self):
        self._stable_tick_count = 0
        self._market_stable = False
        self._waiting_for_market_open = False
        self.last_trade_time = -1
        self._retry_count_this_window = 0
        self._btc_strike_price = None
        self._strike_recorded = False
        self._active_entry = None
        self._late_window_traded = False

    def _handle_market_open(self, now):
        logger.info(f"⏰ MARKET OPEN: {now.strftime('%H:%M:%S')} UTC")
        if 0 <= self.current_instrument_index < len(self.all_btc_instruments):
            self.next_switch_time = self.all_btc_instruments[self.current_instrument_index]['end_time']
        self._reset_market_state()
        self._waiting_for_market_open = False

    def _resolve_positions_for_market(self, market_index: int):
        if market_index < 0 or market_index >= len(self.all_btc_instruments):
            return
        slug = self.all_btc_instruments[market_index]['slug']
        self._pos_tracker.resolve_market(slug)
        for pos in self._pos_tracker.positions:
            if pos.market_slug == slug and pos.resolved and pos.order_id.startswith("paper_"):
                outcome = "WIN" if pos.pnl and pos.pnl > 0 else "LOSS"
                self._update_paper_trade_outcome(pos, outcome)

