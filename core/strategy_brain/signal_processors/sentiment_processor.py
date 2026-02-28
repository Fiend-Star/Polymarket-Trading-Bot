"""
Sentiment Signal Processor
Generates signals based on market sentiment (Fear & Greed Index)
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


class SentimentProcessor(BaseSignalProcessor):
    """
    Generates signals from sentiment data.
    
    Logic:
    - Extreme Fear (0-25) → Contrarian bullish (buy the fear)
    - Fear (25-45) → Mild bullish
    - Neutral (45-55) → No signal
    - Greed (55-75) → Mild bearish
    - Extreme Greed (75-100) → Contrarian bearish (fade the greed)
    """
    
    def __init__(
        self,
        extreme_fear_threshold: float = 25,
        extreme_greed_threshold: float = 75,
        min_confidence: float = 0.50,
    ):
        """
        Initialize sentiment processor.
        
        Args:
            extreme_fear_threshold: Score below this = extreme fear
            extreme_greed_threshold: Score above this = extreme greed
            min_confidence: Minimum confidence to generate signal
        """
        super().__init__("SentimentAnalysis")
        
        self.extreme_fear = extreme_fear_threshold
        self.extreme_greed = extreme_greed_threshold
        self.min_confidence = min_confidence
        
        logger.info(
            f"Initialized Sentiment Processor: "
            f"fear<{extreme_fear_threshold}, greed>{extreme_greed_threshold}"
        )
    
    def _classify_sentiment(self, score):
        """Classify sentiment score into direction, strength, confidence. Returns tuple or None."""
        if score <= self.extreme_fear:
            ext = (self.extreme_fear - score) / self.extreme_fear
            return self._classify_extreme(ext, SignalDirection.BULLISH)
        elif score >= self.extreme_greed:
            ext = (score - self.extreme_greed) / (100 - self.extreme_greed)
            return self._classify_extreme(ext, SignalDirection.BEARISH)
        elif score < 45:
            return SignalDirection.BULLISH, SignalStrength.WEAK, 0.55
        elif score > 55:
            return SignalDirection.BEARISH, SignalStrength.WEAK, 0.55
        return None

    @staticmethod
    def _classify_extreme(extremeness, direction):
        """Classify extreme sentiment into strength/confidence."""
        if extremeness >= 0.8:
            return direction, SignalStrength.VERY_STRONG, 0.85
        elif extremeness >= 0.5:
            return direction, SignalStrength.STRONG, 0.75
        return direction, SignalStrength.MODERATE, 0.65

    def process(self, current_price: Decimal, historical_prices: list,
                metadata: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """Generate signal from sentiment data (Fear & Greed Index)."""
        if not self.is_enabled or not metadata or 'sentiment_score' not in metadata:
            return None
        score = float(metadata['sentiment_score'])
        result = self._classify_sentiment(score)
        if result is None:
            return None
        direction, strength, confidence = result
        if confidence < self.min_confidence:
            return None
        signal = self._build_and_record(
            SignalType.SENTIMENT_SHIFT, direction, strength, confidence, current_price,
            {"sentiment_score": score,
             "sentiment_classification": metadata.get('sentiment_classification', 'unknown')})
        logger.info(f"Sentiment signal: {direction.value}, score={score:.1f}, conf={confidence:.2%}")
        return signal