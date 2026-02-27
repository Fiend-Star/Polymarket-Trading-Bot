"""
Learning Engine
Learns from trading performance to optimize strategy weights
"""
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from monitoring.performance_tracker import get_performance_tracker, Trade
from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine


@dataclass
class SignalPerformance:
    """Performance metrics for a signal source."""
    source_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl: Decimal
    total_pnl: Decimal
    avg_confidence: float
    avg_score: float
    last_updated: datetime


class LearningEngine:
    """
    Learning engine that optimizes strategy based on performance.
    
    Features:
    - Analyzes signal source performance
    - Adjusts fusion weights
    - Identifies winning patterns
    - Improves over time
    """
    
    def __init__(self, learning_rate: float = 0.1, min_trades_for_learning: int = 10,
                 performance_tracker=None, fusion_engine=None):
        """Initialize learning engine with optional injected dependencies (DIP)."""
        self.learning_rate = learning_rate
        self.min_trades = min_trades_for_learning
        self.performance = performance_tracker or get_performance_tracker()
        self.fusion = fusion_engine or get_fusion_engine()
        self._signal_performance: Dict[str, SignalPerformance] = {}
        self._weight_adjustments: List[Dict[str, Any]] = []
        logger.info(f"Learning Engine (lr={learning_rate}, min_trades={min_trades_for_learning})")

    def _group_trades_by_source(self, lookback_days):
        """Group recent trades by signal source."""
        cutoff = datetime.now() - timedelta(days=lookback_days)
        trades = self.performance.get_trade_history(limit=1000, start_date=cutoff)
        groups: Dict[str, list] = {}
        for t in trades:
            for src in t.metadata.get("signal_sources", []):
                groups.setdefault(src, []).append(t)
        return groups

    def _compute_source_perf(self, source, trades):
        """Compute SignalPerformance for a single source."""
        n = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl < 0)
        total_pnl = sum(t.pnl for t in trades)
        return SignalPerformance(
            source_name=source, total_trades=n, winning_trades=wins,
            losing_trades=losses, win_rate=wins / n if n > 0 else 0.0,
            avg_pnl=total_pnl / n if n > 0 else Decimal("0"), total_pnl=total_pnl,
            avg_confidence=sum(t.signal_confidence for t in trades) / n if n > 0 else 0.0,
            avg_score=sum(t.signal_score for t in trades) / n if n > 0 else 0.0,
            last_updated=datetime.now())

    def analyze_signal_performance(self, lookback_days: int = 7) -> Dict[str, SignalPerformance]:
        """Analyze performance of each signal source."""
        groups = self._group_trades_by_source(lookback_days)
        perfs = {}
        for src, trades in groups.items():
            perf = self._compute_source_perf(src, trades)
            perfs[src] = perf
            self._signal_performance[src] = perf
        logger.info(f"Analyzed {len(perfs)} signal sources")
        return perfs

    def calculate_optimal_weights(self, performances: Dict[str, SignalPerformance]) -> Dict[str, float]:
        """Calculate optimal weights based on performance."""
        weights = {}
        for src, perf in performances.items():
            if perf.total_trades < self.min_trades:
                weights[src] = self.fusion.weights.get(src, 0.1); continue
            wr_score = perf.win_rate
            pnl_score = min(1.0, max(0.0, float(perf.total_pnl / Decimal("100"))))
            perf_score = wr_score * 0.6 + pnl_score * 0.4
            cur = self.fusion.weights.get(src, 0.1)
            weights[src] = max(0.05, min(0.50, cur + (perf_score - cur) * self.learning_rate))
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()} if total > 0 else weights

    def _record_adjustment(self, new_weights, performances):
        """Record weight adjustment for audit trail."""
        self._weight_adjustments.append({
            "timestamp": datetime.now(),
            "old_weights": self.fusion.weights.copy(),
            "new_weights": new_weights.copy(),
            "performances": {s: {"win_rate": p.win_rate, "total_pnl": float(p.total_pnl),
                                 "trades": p.total_trades} for s, p in performances.items()}})

    async def optimize_weights(self) -> Dict[str, float]:
        """Optimize signal fusion weights based on recent performance."""
        logger.info("=" * 60 + " OPTIMIZING WEIGHTS " + "=" * 60)
        perfs = self.analyze_signal_performance(lookback_days=7)
        if not perfs:
            logger.warning("No performance data"); return self.fusion.weights.copy()
        new_w = self.calculate_optimal_weights(perfs)
        for src, w in new_w.items():
            old = self.fusion.weights.get(src, 0.0)
            logger.info(f"  {src}: {old:.3f} → {w:.3f} ({w - old:+.3f})")
            self.fusion.set_weight(src, w)
        self._record_adjustment(new_w, perfs)
        logger.info("✓ Weights optimized")
        return new_w

    def get_signal_rankings(self) -> List[Dict[str, Any]]:
        """
        Get signals ranked by performance.
        
        Returns:
            List of signals sorted by performance
        """
        rankings = []
        
        for source, perf in self._signal_performance.items():
            rankings.append({
                "source": source,
                "win_rate": perf.win_rate,
                "total_pnl": float(perf.total_pnl),
                "avg_pnl": float(perf.avg_pnl),
                "total_trades": perf.total_trades,
                "current_weight": self.fusion.weights.get(source, 0.0),
            })
        
        # Sort by total P&L
        rankings.sort(key=lambda x: x["total_pnl"], reverse=True)
        
        return rankings
    
    def get_learning_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of weight adjustments.
        
        Args:
            limit: Max adjustments to return
            
        Returns:
            List of weight adjustments
        """
        return self._weight_adjustments[-limit:]
    
    def export_insights(self) -> Dict[str, Any]:
        """
        Export learning insights.
        
        Returns:
            Insights dict
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "signal_performance": {
                source: {
                    "win_rate": perf.win_rate,
                    "total_pnl": float(perf.total_pnl),
                    "total_trades": perf.total_trades,
                    "current_weight": self.fusion.weights.get(source, 0.0),
                }
                for source, perf in self._signal_performance.items()
            },
            "signal_rankings": self.get_signal_rankings(),
            "recent_adjustments": self.get_learning_history(5),
        }


# Singleton instance
_learning_engine_instance = None

def get_learning_engine() -> LearningEngine:
    """Get singleton learning engine."""
    global _learning_engine_instance
    if _learning_engine_instance is None:
        _learning_engine_instance = LearningEngine()
    return _learning_engine_instance