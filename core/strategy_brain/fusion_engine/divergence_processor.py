"""
Price Divergence Signal Processor
Detects when Polymarket price diverges from spot exchanges
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


class PriceDivergenceProcessor(BaseSignalProcessor):
    """
    Detects price divergence between Polymarket and spot exchanges.
    
    Logic:
    - Compare Polymarket prediction price vs actual BTC spot price
    - If divergence > threshold, signal arbitrage opportunity
    - Direction: Trade toward convergence
    """
    
    def __init__(
        self,
        divergence_threshold: float = 0.05,  # 5% divergence
        min_confidence: float = 0.65,
    ):
        """
        Initialize divergence processor.
        
        Args:
            divergence_threshold: Minimum divergence to signal (0.05 = 5%)
            min_confidence: Minimum confidence threshold
        """
        super().__init__("PriceDivergence")
        
        self.divergence_threshold = divergence_threshold
        self.min_confidence = min_confidence
        
        logger.info(
            f"Initialized Price Divergence Processor: "
            f"threshold={divergence_threshold:.1%}"
        )
    
    @staticmethod
    def _classify_divergence_strength(div_pct):
        """Classify divergence magnitude into signal strength."""
        if div_pct >= 0.15: return SignalStrength.VERY_STRONG
        if div_pct >= 0.10: return SignalStrength.STRONG
        if div_pct >= 0.07: return SignalStrength.MODERATE
        return SignalStrength.WEAK

    def process(self, current_price: Decimal, historical_prices: list,
                metadata: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Detect price divergence between Polymarket and spot."""
        if not self.is_enabled or not metadata or 'spot_price' not in metadata:
            return None
        spot = Decimal(str(metadata['spot_price']))
        div = (current_price - spot) / spot
        div_pct = float(abs(div))
        if div_pct < self.divergence_threshold:
            return None
        direction = SignalDirection.BEARISH if div > 0 else SignalDirection.BULLISH
        strength = self._classify_divergence_strength(div_pct)
        confidence = min(0.90, 0.60 + div_pct)
        if confidence < self.min_confidence:
            return None
        signal = TradingSignal(
            timestamp=datetime.now(), source=self.name,
            signal_type=SignalType.PRICE_DIVERGENCE, direction=direction,
            strength=strength, confidence=confidence,
            current_price=current_price, target_price=spot,
            metadata={"divergence_pct": div_pct, "spot_price": float(spot),
                      "polymarket_price": float(current_price)})
        self._record_signal(signal)
        logger.info(f"Divergence {direction.value}: {div_pct:.2%}, conf={confidence:.2%}")
        return signal