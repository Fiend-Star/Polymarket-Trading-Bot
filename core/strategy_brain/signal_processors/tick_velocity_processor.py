"""
Tick Velocity Signal Processor
Measures how fast the Polymarket UP probability is moving in the
last 60 seconds before the trade window opens.

WHY THIS WORKS:
  If the "Up" probability moves from 0.50 → 0.57 in the last 45 seconds,
  BEFORE your bot even looks at it, that reflects real order flow from
  other traders reacting to BTC spot movement. The probability is being
  pushed by real money, and it often continues in the same direction
  for at least another 30–60 seconds.

  This is "price action" on the Polymarket itself — not a lagging
  external indicator. It's the most direct signal available.

HOW IT WORKS:
  The strategy stores a rolling tick buffer:
    self._tick_buffer = deque of {'ts': datetime, 'price': Decimal}

  This processor receives that buffer via metadata['tick_buffer'].

  It computes:
    1. 60s velocity  = (now_price - price_60s_ago) / price_60s_ago
    2. 30s velocity  = (now_price - price_30s_ago) / price_30s_ago
    3. Acceleration  = 30s_velocity - (60s_velocity - 30s_velocity)
                       (is it speeding up or slowing down?)

  Signal thresholds (for 0–1 probability prices):
    velocity > +1.5%  in 60s → BULLISH
    velocity < -1.5%  in 60s → BEARISH
    acceleration bonus: if move is accelerating → higher confidence

INTEGRATION:
  In bot.py on_quote_tick(), add:
    self._tick_buffer.append({'ts': now, 'price': mid_price})

  In _fetch_market_context(), add:
    metadata['tick_buffer'] = list(self._tick_buffer)
"""
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Optional, Dict, Any, List
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


class TickVelocityProcessor(BaseSignalProcessor):
    """
    Measures Polymarket probability velocity over the last 60 seconds.

    Fast moves in probability = real order flow = tradeable signal.
    """

    def __init__(
        self,
        velocity_threshold_60s: float = 0.015,   # 1.5% move in 60s
        velocity_threshold_30s: float = 0.010,   # 1.0% move in 30s
        min_ticks: int = 5,                       # need at least 5 ticks in window
        min_confidence: float = 0.55,
    ):
        super().__init__("TickVelocity")

        self.velocity_threshold_60s = velocity_threshold_60s
        self.velocity_threshold_30s = velocity_threshold_30s
        self.min_ticks = min_ticks
        self.min_confidence = min_confidence

        logger.info(
            f"Initialized Tick Velocity Processor: "
            f"60s_threshold={velocity_threshold_60s:.1%}, "
            f"30s_threshold={velocity_threshold_30s:.1%}"
        )

    def _get_price_at(
        self,
        tick_buffer: List[Dict],
        seconds_ago: float,
        now: datetime,
    ) -> Optional[float]:
        """Find the tick price closest to `seconds_ago` seconds before now."""
        target = now - timedelta(seconds=seconds_ago)
        best = None
        best_diff = float('inf')

        for tick in tick_buffer:
            ts = tick['ts']
            # Normalise timezone
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            diff = abs((ts - target).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best = float(tick['price'])

        # Only return if within ±15s of target
        if best_diff <= 15:
            return best
        return None

    def _compute_velocities(self, tick_buffer, current_price):
        """Compute 30s and 60s velocities plus acceleration. Returns dict or None."""
        now = datetime.now(timezone.utc)
        curr = float(current_price)
        p60 = self._get_price_at(tick_buffer, 60, now)
        p30 = self._get_price_at(tick_buffer, 30, now)
        if p60 is None and p30 is None:
            return None
        v60 = ((curr - p60) / p60) if p60 else None
        v30 = ((curr - p30) / p30) if p30 else None
        accel = 0.0
        if v60 is not None and v30 is not None:
            accel = v30 - (v60 - v30)
        return {"v60": v60, "v30": v30, "accel": accel, "p60": p60, "p30": p30, "ticks": len(tick_buffer)}

    def _classify_velocity(self, abs_vel):
        """Classify velocity magnitude into signal strength."""
        if abs_vel >= 0.04: return SignalStrength.VERY_STRONG
        if abs_vel >= 0.025: return SignalStrength.STRONG
        if abs_vel >= 0.015: return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def _compute_confidence(self, pv, threshold, accel, v60, v30):
        """Compute confidence with acceleration/reversal adjustments."""
        conf = min(0.82, 0.55 + (abs(pv) / threshold - 1) * 0.12)
        same_dir = (accel > 0 and pv > 0) or (accel < 0 and pv < 0)
        if same_dir and abs(accel) > 0.005:
            conf = min(0.88, conf + 0.06)
        if v60 is not None and v30 is not None and (v60 > 0) != (v30 > 0):
            conf *= 0.80
        return conf

    def process(self, current_price: Decimal, historical_prices: list,
                metadata: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Generate momentum signal from tick velocity analysis."""
        if not self.is_enabled or not metadata:
            return None
        tick_buffer = metadata.get("tick_buffer")
        if not tick_buffer or len(tick_buffer) < self.min_ticks:
            return None
        v = self._compute_velocities(tick_buffer, current_price)
        if v is None:
            return None
        pv = v["v30"] if v["v30"] is not None else v["v60"]
        thresh = self.velocity_threshold_30s if v["v30"] is not None else self.velocity_threshold_60s
        if abs(pv) < thresh:
            return None
        direction = SignalDirection.BULLISH if pv > 0 else SignalDirection.BEARISH
        strength = self._classify_velocity(abs(pv))
        conf = self._compute_confidence(pv, thresh, v["accel"], v["v60"], v["v30"])
        if conf < self.min_confidence:
            return None
        meta = {"velocity_60s": round(v["v60"], 6) if v["v60"] else None,
                "velocity_30s": round(v["v30"], 6) if v["v30"] else None,
                "acceleration": round(v["accel"], 6),
                "price_60s_ago": round(v["p60"], 6) if v["p60"] else None,
                "price_30s_ago": round(v["p30"], 6) if v["p30"] else None,
                "ticks_in_buffer": v["ticks"]}
        sig = self._build_and_record(SignalType.MOMENTUM, direction, strength, conf, current_price, meta)
        logger.info(f"{direction.value.upper()} TickVelocity: vel={pv*100:+.3f}%, conf={conf:.2%}")
        return sig
