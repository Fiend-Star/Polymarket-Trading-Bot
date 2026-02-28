"""
Price Divergence Signal Processor
Detects when Polymarket UP probability misprices BTC spot momentum

KEY INSIGHT:
  Polymarket "Up" price = probability that BTC will be HIGHER at market close
  Coinbase price       = actual BTC spot price in USD

  These are INCOMPARABLE. Never subtract them.

  Instead, the signal is:
    - Use SPOT MOMENTUM (recent BTC price direction) to predict whether
      "Up" is more or less likely than the current Polymarket probability implies.
    - Use POLYMARKET MISPRICING: if the market heavily favors Up (>0.65) but
      BTC momentum is bearish, bet DOWN. Vice versa.

  Two sub-signals:
  1. MOMENTUM SIGNAL: Is BTC trending up or down over the last ~15 min?
     → metadata['momentum'] is already computed as 5-period ROC of poly prices,
       but we use spot_price vs spot_price_prev if available, else fall back
       to polymarket momentum.

  2. MISPRICING SIGNAL: Is the Polymarket UP probability too extreme vs
     what momentum/sentiment suggest?
     → If poly_price > 0.65 and momentum is bearish → BEARISH (market over-priced Up)
     → If poly_price < 0.35 and momentum is bullish → BULLISH (market over-priced Down)
     → If poly_price near 0.50, no strong edge → skip
"""
from decimal import Decimal
from datetime import datetime
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


class PriceDivergenceProcessor(BaseSignalProcessor):
    """
    Detects mispricings between Polymarket UP probability and BTC spot momentum.

    Replaces the broken "compare probability to dollar price" approach with two
    meaningful signals:

    1. MOMENTUM MISPRICING:
       If BTC spot is trending UP strongly but Polymarket "Up" probability is
       below 0.50, the market is underpricing the move → BUY UP.
       Vice versa for downtrend.

    2. EXTREME PROBABILITY FADE:
       If Polymarket "Up" is priced above 0.70, it's unlikely to resolve even
       higher — fade toward DOWN. If below 0.30, fade toward UP.
       (Markets rarely sustain >70% probability at interval open.)
    """

    def __init__(
        self,
        divergence_threshold: float = 0.05,   # kept for API compatibility (unused)
        min_confidence: float = 0.55,
        momentum_threshold: float = 0.003,     # 0.3% spot move = meaningful momentum
        extreme_prob_threshold: float = 0.68,  # above this → fade to Down
        low_prob_threshold: float = 0.32,      # below this → fade to Up
    ):
        super().__init__("PriceDivergence")

        self.min_confidence = min_confidence
        self.momentum_threshold = momentum_threshold
        self.extreme_prob_threshold = extreme_prob_threshold
        self.low_prob_threshold = low_prob_threshold

        # Rolling spot price history for momentum calculation
        self._spot_history: List[float] = []
        self._max_spot_history = 10  # last 10 readings (~2.5 min of data)

        logger.info(
            f"Initialized Price Divergence Processor (FIXED): "
            f"momentum_thresh={momentum_threshold:.1%}, "
            f"extreme_fade={extreme_prob_threshold:.0%}/{low_prob_threshold:.0%}"
        )

    def _make_signal(self, direction, strength, confidence, current_price, meta):
        """Build, record, and return a TradingSignal."""
        signal = TradingSignal(
            timestamp=datetime.now(), source=self.name,
            signal_type=SignalType.PRICE_DIVERGENCE, direction=direction,
            strength=strength, confidence=confidence,
            current_price=current_price, metadata=meta)
        self._record_signal(signal)
        return signal

    def _update_spot_history(self, metadata):
        """Update spot price history and compute spot momentum."""
        spot_price = metadata.get('spot_price')
        poly_momentum = float(metadata.get('momentum', 0.0))
        if spot_price is not None:
            self._spot_history.append(float(spot_price))
            if len(self._spot_history) > self._max_spot_history:
                self._spot_history.pop(0)
        spot_momentum = 0.0
        if spot_price is not None and len(self._spot_history) >= 3:
            oldest = self._spot_history[-min(3, len(self._spot_history))]
            spot_momentum = (float(spot_price) - oldest) / oldest
        elif spot_price is None:
            spot_momentum = poly_momentum
        return spot_price, spot_momentum

    def _check_extreme_fade(self, poly_prob, spot_momentum, current_price):
        """Check for extreme probability fade signal."""
        if poly_prob >= self.extreme_prob_threshold and spot_momentum <= 0.001:
            ext = (poly_prob - self.extreme_prob_threshold) / (1.0 - self.extreme_prob_threshold)
            conf = min(0.80, self.min_confidence + ext * 0.25)
            strength = SignalStrength.STRONG if ext > 0.5 else SignalStrength.MODERATE
            logger.info(f"BEARISH fade: poly Up {poly_prob:.0%} with weak momentum, conf={conf:.2%}")
            return self._make_signal(SignalDirection.BEARISH, strength, conf, current_price,
                                     {"signal_type": "extreme_prob_fade_down", "poly_prob": poly_prob,
                                      "spot_momentum": spot_momentum, "extremeness": ext})
        if poly_prob <= self.low_prob_threshold and spot_momentum >= -0.001:
            ext = (self.low_prob_threshold - poly_prob) / self.low_prob_threshold
            conf = min(0.80, self.min_confidence + ext * 0.25)
            strength = SignalStrength.STRONG if ext > 0.5 else SignalStrength.MODERATE
            logger.info(f"BULLISH fade: poly Down {1-poly_prob:.0%} with weak neg momentum, conf={conf:.2%}")
            return self._make_signal(SignalDirection.BULLISH, strength, conf, current_price,
                                     {"signal_type": "extreme_prob_fade_up", "poly_prob": poly_prob,
                                      "spot_momentum": spot_momentum, "extremeness": ext})
        return None

    def _check_momentum_mispricing(self, poly_prob, spot_momentum, current_price):
        """Check for momentum mispricing signal."""
        if not (0.35 <= poly_prob <= 0.65 and abs(spot_momentum) >= self.momentum_threshold):
            return None
        ms = abs(spot_momentum) / self.momentum_threshold
        conf = min(0.78, 0.55 + min(ms - 1, 2) * 0.08)
        if conf < self.min_confidence:
            return None
        if ms >= 3: strength = SignalStrength.STRONG
        elif ms >= 2: strength = SignalStrength.MODERATE
        else: strength = SignalStrength.WEAK
        direction = SignalDirection.BULLISH if spot_momentum > 0 else SignalDirection.BEARISH
        logger.info(f"{direction.value} momentum: spot {spot_momentum:+.3%}, poly {poly_prob:.0%}, conf={conf:.2%}")
        return self._make_signal(direction, strength, conf, current_price,
                                 {"signal_type": "momentum_mispricing", "poly_prob": poly_prob,
                                  "spot_momentum": spot_momentum, "momentum_strength": ms})

    def process(self, current_price: Decimal, historical_prices: list,
                metadata: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Generate signal from spot momentum vs Polymarket probability."""
        if not self.is_enabled or not metadata:
            return None
        poly_prob = float(current_price)
        spot_price, spot_momentum = self._update_spot_history(metadata)
        logger.info(f"PriceDivergence: poly_prob={poly_prob:.3f}, spot_momentum={spot_momentum:+.4f}, "
                    f"spot={'${:,.2f}'.format(spot_price) if spot_price else 'N/A'}")

        sig = self._check_extreme_fade(poly_prob, spot_momentum, current_price)
        if sig:
            return sig
        sig = self._check_momentum_mispricing(poly_prob, spot_momentum, current_price)
        if sig:
            return sig
        logger.debug(f"PriceDivergence: no signal — prob={poly_prob:.2f}, momentum={spot_momentum:+.4f}")
        return None
