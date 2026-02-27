"""
Spike Detection Signal Processor
Detects sudden price movements in Polymarket UP probability and generates signals.

FIX: Original threshold was 15% deviation, designed for dollar prices.
     Polymarket prices are probabilities (0.0-1.0), so:
       - "Normal" range at market open: 0.40-0.60
       - A 15% deviation from MA of 0.50 = price must reach 0.575 or 0.425
       - This rarely fires, making the detector useless at market open

     New approach:
       - Spike threshold: 5% deviation (not 15%) — meaningful for probabilities
       - Also detect VELOCITY (fast moves in the last 3 ticks)
       - Mean reversion logic is still correct: spike up → BEARISH, spike down → BULLISH
"""
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any
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


class SpikeDetectionProcessor(BaseSignalProcessor):
    """
    Detects price spikes in Polymarket UP probability.

    Two detection modes:
    1. MA DEVIATION: Price deviates >5% from 20-period MA → mean reversion
    2. VELOCITY SPIKE: Price moves >3% in last 3 ticks → momentum continuation
       (short bursts often continue briefly before reverting)

    Direction logic:
    - Deviation spike UP → expect reversion → BEARISH
    - Deviation spike DOWN → expect reversion → BULLISH
    - Velocity spike UP → momentum continuation → BULLISH (for first ~30s)
    - Velocity spike DOWN → momentum continuation → BEARISH
    """

    def __init__(
        self,
        spike_threshold: float = 0.05,    # FIXED: was 0.15, now 0.05 for probability prices
        lookback_periods: int = 20,
        min_confidence: float = 0.55,     # FIXED: was 0.60, slightly lower for more signals
        velocity_threshold: float = 0.03, # 3% move in 3 ticks = velocity spike
    ):
        super().__init__("SpikeDetection")

        self.spike_threshold = spike_threshold
        self.lookback_periods = lookback_periods
        self.min_confidence = min_confidence
        self.velocity_threshold = velocity_threshold

        logger.info(
            f"Initialized Spike Detector (FIXED): "
            f"deviation_threshold={spike_threshold:.1%}, "
            f"velocity_threshold={velocity_threshold:.1%}, "
            f"lookback={lookback_periods}"
        )

    def _compute_stats(self, current_price, historical_prices):
        """Compute MA deviation and velocity from price history."""
        recent = historical_prices[-self.lookback_periods:]
        ma = sum(float(p) for p in recent) / len(recent)
        curr = float(current_price)
        deviation = (curr - ma) / ma if ma > 0 else 0.0
        velocity = 0.0
        if len(historical_prices) >= 3:
            prev3 = float(historical_prices[-3])
            velocity = (curr - prev3) / prev3 if prev3 > 0 else 0.0
        return curr, ma, deviation, velocity

    def _classify_deviation_strength(self, dev_abs):
        """Classify strength from deviation magnitude."""
        if dev_abs >= 0.12: return SignalStrength.VERY_STRONG
        if dev_abs >= 0.08: return SignalStrength.STRONG
        if dev_abs >= 0.05: return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def _check_ma_deviation_spike(self, curr, ma, deviation, velocity, current_price):
        """Check for MA deviation spike (mean reversion signal)."""
        dev_abs = abs(deviation)
        if dev_abs < self.spike_threshold:
            return None
        direction = SignalDirection.BEARISH if deviation > 0 else SignalDirection.BULLISH
        strength = self._classify_deviation_strength(dev_abs)
        confidence = min(0.90, 0.50 + (dev_abs - self.spike_threshold) * 3.0)
        if confidence < self.min_confidence:
            return None
        target = Decimal(str(ma))
        stop_dist = abs(Decimal(str(curr)) - Decimal(str(ma))) * Decimal("1.5")
        stop_loss = Decimal(str(curr)) + stop_dist if direction == SignalDirection.BEARISH else Decimal(str(curr)) - stop_dist
        signal = TradingSignal(
            timestamp=datetime.now(), source=self.name,
            signal_type=SignalType.SPIKE_DETECTED, direction=direction,
            strength=strength, confidence=confidence, current_price=current_price,
            target_price=target, stop_loss=stop_loss,
            metadata={"detection_mode": "ma_deviation", "deviation_pct": deviation,
                      "moving_average": ma, "velocity": velocity,
                      "spike_direction": "up" if deviation > 0 else "down"}
        )
        self._record_signal(signal)
        logger.info(f"{direction.value.upper()} MA deviation: {deviation:+.3%}, conf={confidence:.2%}")
        return signal

    def _check_velocity_spike(self, deviation, velocity, current_price, ma):
        """Check for velocity spike (short-term momentum signal)."""
        if abs(velocity) < self.velocity_threshold:
            return None
        if abs(deviation) >= self.spike_threshold * 0.6:
            return None
        direction = SignalDirection.BULLISH if velocity > 0 else SignalDirection.BEARISH
        vs = abs(velocity) / self.velocity_threshold
        if vs >= 3: strength, confidence = SignalStrength.MODERATE, 0.65
        elif vs >= 2: strength, confidence = SignalStrength.WEAK, 0.60
        else: strength, confidence = SignalStrength.WEAK, 0.57
        if confidence < self.min_confidence:
            return None
        signal = self._build_and_record(
            SignalType.MOMENTUM, direction, strength, confidence, current_price,
            {"detection_mode": "velocity", "velocity_pct": velocity,
             "moving_average": ma, "deviation_pct": deviation}
        )
        logger.info(f"{direction.value.upper()} velocity: {velocity:+.3%}, conf={confidence:.2%}")
        return signal

    def process(self, current_price: Decimal, historical_prices: list,
                metadata: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Detect probability spikes and generate mean-reversion or momentum signals."""
        if not self.is_enabled or len(historical_prices) < self.lookback_periods:
            return None
        curr, ma, deviation, velocity = self._compute_stats(current_price, historical_prices)
        logger.debug(f"SpikeDetector: price={curr:.4f}, MA={ma:.4f}, "
                     f"deviation={deviation:+.3%}, velocity={velocity:+.3%}")
        sig = self._check_ma_deviation_spike(curr, ma, deviation, velocity, current_price)
        if sig:
            return sig
        return self._check_velocity_spike(deviation, velocity, current_price, ma)
