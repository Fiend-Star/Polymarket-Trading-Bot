"""
Signal Fusion Engine
Combines multiple signals with weighted voting
"""
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.strategy_brain.signal_processors.base_processor import (
    TradingSignal,
    SignalDirection,
    SignalStrength,
)


@dataclass
class FusedSignal:
    timestamp: datetime
    direction: SignalDirection
    confidence: float
    score: float
    
    signals: List[TradingSignal]
    weights: Dict[str, float]
    metadata: Dict[str, Any]
    
    @property
    def num_signals(self) -> int:
        return len(self.signals)
    
    @property
    def is_strong(self) -> bool:
        return self.score >= 70.0
    
    @property
    def is_actionable(self) -> bool:
        return self.score >= 60.0 and self.confidence >= 0.6


class SignalFusionEngine:
    def __init__(self):
        self.weights = {
            "SpikeDetection":    0.40,
            "PriceDivergence":   0.30,
            "SentimentAnalysis": 0.20,
            "default":           0.10,
        }
        
        self._signal_history: List[FusedSignal] = []
        self._max_history = 100
        self._fusions_performed = 0
        
        logger.info("Initialized Signal Fusion Engine")

    def set_weight(self, processor_name: str, weight: float) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        self.weights[processor_name] = weight
        logger.debug(f"Set weight for {processor_name}: {weight:.2f}")

    def _filter_recent(self, signals, min_signals):
        """Filter to recent signals. Returns list or None."""
        now = datetime.now()
        recent = [s for s in signals if (now - s.timestamp) < timedelta(minutes=5)]
        if len(recent) < min_signals:
            return None
        return recent

    def _calculate_contributions(self, signals):
        """Calculate bullish/bearish weighted contributions."""
        bull = bear = 0.0
        for s in signals:
            w = self.weights.get(s.source, self.weights["default"])
            sf = (s.strength.value if s.strength else 2) / 4.0
            c = min(1.0, max(0.0, s.confidence))
            contrib = w * c * sf
            d = str(s.direction).upper()
            if "BULLISH" in d: bull += contrib
            elif "BEARISH" in d: bear += contrib
        return bull, bear

    def _build_fused(self, recent, bull, bear, min_score):
        """Build FusedSignal from contributions. Returns signal or None."""
        total = bull + bear
        if total < 0.0001:
            return None
        if bull >= bear:
            direction = SignalDirection.BULLISH
            dominant = bull
        else:
            direction = SignalDirection.BEARISH
            dominant = bear
        score = (dominant / total) * 100
        if score < min_score:
            return None
        avg_conf = sum(s.confidence for s in recent) / len(recent)
        return FusedSignal(
            timestamp=datetime.now(), direction=direction, confidence=avg_conf,
            score=score, signals=recent, weights=self.weights.copy(),
            metadata={"bullish_contrib": round(bull, 4), "bearish_contrib": round(bear, 4),
                      "total_contrib": round(total, 4),
                      "num_bullish": sum(1 for s in recent if "BULLISH" in str(s.direction).upper()),
                      "num_bearish": sum(1 for s in recent if "BEARISH" in str(s.direction).upper())})

    def fuse_signals(self, signals: List[TradingSignal], min_signals: int = 1,
                     min_score: float = 50.0) -> Optional[FusedSignal]:
        """Fuse multiple signals into a single consensus signal."""
        if not signals:
            return None
        recent = self._filter_recent(signals, min_signals)
        if not recent:
            return None
        bull, bear = self._calculate_contributions(recent)
        fused = self._build_fused(recent, bull, bear, min_score)
        if not fused:
            return None
        self._fusions_performed += 1
        self._signal_history.append(fused)
        if len(self._signal_history) > self._max_history:
            self._signal_history.pop(0)
        logger.info(f"Fused {len(recent)} signals → {fused.direction} "
                    f"(score={fused.score:.1f}, conf={fused.confidence:.1%})")
        return fused
    
    def get_recent_fusions(self, limit: int = 10) -> List[FusedSignal]:
        return self._signal_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self._signal_history:
            return {
                "total_fusions": self._fusions_performed,
                "recent_fusions": 0,
                "avg_score": 0.0,
                "avg_confidence": 0.0,
            }
        
        recent = self._signal_history[-20:]
        return {
            "total_fusions": self._fusions_performed,
            "recent_fusions": len(recent),
            "avg_score": sum(f.score for f in recent) / len(recent) if recent else 0.0,
            "avg_confidence": sum(f.confidence for f in recent) / len(recent) if recent else 0.0,
            "weights": self.weights.copy(),
        }


_fusion_engine_instance = None

def get_fusion_engine() -> SignalFusionEngine:
    global _fusion_engine_instance
    if _fusion_engine_instance is None:
        _fusion_engine_instance = SignalFusionEngine()
    return _fusion_engine_instance