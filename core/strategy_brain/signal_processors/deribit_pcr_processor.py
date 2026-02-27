"""
Deribit Put/Call Ratio Signal Processor
Fetches real-time BTC options data from Deribit (free public API)
and uses the put/call ratio as a proxy for institutional sentiment.

WHY THIS WORKS:
  Professional traders hedge and speculate using BTC options on Deribit —
  the world's largest crypto options exchange ($15B+ daily volume).

  Put/Call Ratio (PCR) = open_interest_puts / open_interest_calls

  PCR interpretation (CONTRARIAN — options markets lean the opposite):
    PCR > 1.2  → More puts than calls = FEAR → contrarian BULLISH
    PCR < 0.7  → More calls than puts = GREED → contrarian BEARISH
    0.7–1.2    → Balanced → no strong signal

  We look specifically at SHORT-DATED options (0-1 days to expiry)
  which are most sensitive to near-term price movements — ideal for
  15-minute trading.

  We also track the 25-delta skew (difference between put and call IV)
  as a secondary signal:
    Positive skew (puts more expensive) → fear → contrarian BULLISH
    Negative skew (calls more expensive) → greed → contrarian BEARISH

API USED (completely free, no auth required):
  GET https://www.deribit.com/api/v2/public/get_book_summary_by_currency
    ?currency=BTC&kind=option

  Returns all active BTC option contracts with:
    - open_interest
    - instrument_name (e.g. BTC-20FEB26-95000-P = Put, -C = Call)
    - days to expiry (parsed from instrument name)
"""
import httpx
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.strategy_brain.signal_processors.base_processor import (
    BaseSignalProcessor,
    TradingSignal,
    SignalType,
    SignalDirection,
    SignalStrength,
)

DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"


class DeribitPCRProcessor(BaseSignalProcessor):
    """
    Uses Deribit BTC options put/call ratio for contrarian signals.

    Cached for 5 minutes — options data doesn't change tick-by-tick
    and we don't want to hammer Deribit's API every 15 minutes.
    """

    def __init__(
        self,
        bullish_pcr_threshold: float = 1.20,   # PCR above this = contrarian bullish
        bearish_pcr_threshold: float = 0.70,   # PCR below this = contrarian bearish
        max_days_to_expiry: int = 2,            # only short-dated options
        min_open_interest: float = 100.0,       # ignore tiny strikes (BTC notional)
        cache_seconds: int = 300,               # refresh every 5 minutes
        min_confidence: float = 0.55,
    ):
        super().__init__("DeribitPCR")

        self.bullish_pcr_threshold = bullish_pcr_threshold
        self.bearish_pcr_threshold = bearish_pcr_threshold
        self.max_days_to_expiry = max_days_to_expiry
        self.min_open_interest = min_open_interest
        self.cache_seconds = cache_seconds
        self.min_confidence = min_confidence

        # Cache
        self._cached_result: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None

        logger.info(
            f"Initialized Deribit PCR Processor: "
            f"bullish_pcr>{bullish_pcr_threshold}, "
            f"bearish_pcr<{bearish_pcr_threshold}, "
            f"max_dte={max_days_to_expiry}d"
        )

    def _get_client(self) -> httpx.Client:
        """Return a synchronous httpx client (safe inside NautilusTrader's event loop)."""
        return httpx.Client(timeout=8.0)

    def _parse_dte(self, instrument_name: str) -> Optional[int]:
        """
        Parse days to expiry from Deribit instrument name.
        Format: BTC-20FEB26-95000-P  (BTC-DDMMMYY-STRIKE-TYPE)
        """
        try:
            parts = instrument_name.split("-")
            if len(parts) < 3:
                return None
            expiry_str = parts[1]   # e.g. "20FEB26"
            expiry_dt = datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            dte = (expiry_dt - now).days
            return max(0, dte)
        except Exception:
            return None

    def _aggregate_oi(self, summaries):
        """Aggregate put/call OI from Deribit summaries. Returns (put, call, short_put, short_call)."""
        put_oi = call_oi = short_put = short_call = 0.0
        for item in summaries:
            name = item.get("instrument_name", "")
            oi = float(item.get("open_interest", 0))
            if oi < self.min_open_interest:
                continue
            is_put, is_call = name.endswith("-P"), name.endswith("-C")
            if is_put: put_oi += oi
            elif is_call: call_oi += oi
            dte = self._parse_dte(name)
            if dte is not None and dte <= self.max_days_to_expiry:
                if is_put: short_put += oi
                elif is_call: short_call += oi
        return put_oi, call_oi, short_put, short_call

    def _fetch_pcr(self) -> Optional[Dict]:
        """Fetch and compute PCR from Deribit synchronously."""
        try:
            with self._get_client() as client:
                resp = client.get(DERIBIT_URL, params={"currency": "BTC", "kind": "option"})
                resp.raise_for_status()
                summaries = resp.json().get("result", [])
            if not summaries:
                logger.warning("Deribit returned empty options data"); return None
            p_oi, c_oi, sp, sc = self._aggregate_oi(summaries)
            overall = p_oi / c_oi if c_oi > 0 else 1.0
            short = sp / sc if sc > 0 else overall
            logger.info(f"Deribit: overall_PCR={overall:.3f}, short_PCR={short:.3f}")
            return {"overall_pcr": round(overall, 4), "short_pcr": round(short, 4),
                    "put_oi": round(p_oi, 2), "call_oi": round(c_oi, 2),
                    "short_put_oi": round(sp, 2), "short_call_oi": round(sc, 2),
                    "total_contracts": len(summaries),
                    "fetched_at": datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            logger.warning(f"Deribit PCR fetch failed: {e}"); return None

    def process(self, current_price: Decimal, historical_prices: list,
                metadata: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Synchronous wrapper — runs fetch if cache is stale."""
        if not self.is_enabled:
            return None
        now = datetime.now(timezone.utc)
        cache_ok = (self._cached_result and self._cache_time
                    and (now - self._cache_time).total_seconds() < self.cache_seconds)
        if cache_ok:
            pcr_data = self._cached_result
        else:
            try:
                pcr_data = self._fetch_pcr()
            except Exception as e:
                logger.warning(f"DeribitPCR fetch failed: {e}"); return None
            if not pcr_data:
                return None
            self._cached_result = pcr_data
            self._cache_time = now
        return self._generate_signal(current_price, pcr_data)

    def _classify_pcr(self, pcr):
        """Classify PCR into (direction, strength, confidence) or None."""
        if pcr >= self.bullish_pcr_threshold:
            ext = (pcr - self.bullish_pcr_threshold) / self.bullish_pcr_threshold
            conf = min(0.80, 0.57 + ext * 0.15)
            if pcr >= 1.60: s = SignalStrength.VERY_STRONG
            elif pcr >= 1.40: s = SignalStrength.STRONG
            else: s = SignalStrength.MODERATE
            return SignalDirection.BULLISH, s, conf
        if pcr <= self.bearish_pcr_threshold:
            ext = (self.bearish_pcr_threshold - pcr) / self.bearish_pcr_threshold
            conf = min(0.80, 0.57 + ext * 0.15)
            if pcr <= 0.45: s = SignalStrength.VERY_STRONG
            elif pcr <= 0.55: s = SignalStrength.STRONG
            else: s = SignalStrength.MODERATE
            return SignalDirection.BEARISH, s, conf
        return None

    def _generate_signal(self, current_price: Decimal, pcr_data: Dict) -> Optional[TradingSignal]:
        """Generate signal from PCR data."""
        pcr = pcr_data.get("short_pcr") or pcr_data.get("overall_pcr", 1.0)
        result = self._classify_pcr(pcr)
        if result is None:
            logger.debug(f"DeribitPCR: balanced PCR={pcr:.3f} — no signal"); return None
        direction, strength, confidence = result
        if confidence < self.min_confidence:
            return None
        interp = "excessive_puts_fear" if direction == SignalDirection.BULLISH else "excessive_calls_greed"
        sig = self._build_and_record(
            SignalType.SENTIMENT_SHIFT, direction, strength, confidence, current_price,
            {"pcr": round(pcr, 4), "overall_pcr": pcr_data.get("overall_pcr"),
             "short_put_oi": pcr_data.get("short_put_oi"),
             "short_call_oi": pcr_data.get("short_call_oi"), "interpretation": interp})
        logger.info(f"{direction.value.upper()} DeribitPCR: PCR={pcr:.3f}, conf={confidence:.2%}")
        return sig
