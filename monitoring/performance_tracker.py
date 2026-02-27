"""
Performance Tracker
Tracks and analyzes trading performance metrics
"""
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


@dataclass
class Trade:
    """Individual trade record."""
    trade_id: str
    timestamp: datetime
    direction: str  # "long" or "short"
    entry_price: Decimal
    exit_price: Decimal
    size: Decimal
    pnl: Decimal
    pnl_pct: float
    duration_seconds: float
    signal_score: float
    signal_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    
    # P&L metrics
    total_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Return metrics
    roi: float  # Return on investment
    sharpe_ratio: float
    max_drawdown: float
    
    # Position metrics
    open_positions: int
    avg_position_size: Decimal
    avg_hold_time: float  # seconds
    
    # Risk metrics
    total_exposure: Decimal
    risk_utilization: float  # % of max risk used
    
    # Signal performance
    avg_signal_score: float
    avg_signal_confidence: float


class PerformanceTracker:
    """
    Tracks and analyzes trading performance.
    
    Features:
    - Trade history
    - Performance metrics
    - Risk analytics
    - Signal effectiveness
    """
    
    def __init__(
        self,
        initial_capital: Decimal = Decimal("1000.0"),
    ):
        """
        Initialize performance tracker.
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Trade history
        self._trades: List[Trade] = []
        self._max_trades_history = 1000
        
        # Metrics history (for Grafana)
        self._metrics_history: deque = deque(maxlen=10000)
        
        # Performance cache
        self._last_metrics: Optional[PerformanceMetrics] = None
        self._metrics_dirty = True
        
        # Peak tracking for drawdown
        self._peak_capital = initial_capital
        
        logger.info(f"Initialized Performance Tracker (capital=${initial_capital})")
    
    def _compute_pnl(self, direction, entry_price, exit_price, size):
        """Compute trade P&L."""
        pnl_pct = ((exit_price - entry_price) / entry_price if direction == "long"
                   else (entry_price - exit_price) / entry_price)
        return size * pnl_pct, pnl_pct

    def record_trade(self, trade_id: str, direction: str, entry_price: Decimal,
                     exit_price: Decimal, size: Decimal, entry_time: datetime,
                     exit_time: datetime, signal_score: float = 0.0,
                     signal_confidence: float = 0.0, metadata: Dict[str, Any] = None) -> Trade:
        """Record a completed trade."""
        pnl, pnl_pct = self._compute_pnl(direction, entry_price, exit_price, size)
        trade = Trade(
            trade_id=trade_id, timestamp=exit_time, direction=direction,
            entry_price=entry_price, exit_price=exit_price, size=size,
            pnl=pnl, pnl_pct=float(pnl_pct),
            duration_seconds=(exit_time - entry_time).total_seconds(),
            signal_score=signal_score, signal_confidence=signal_confidence,
            metadata=metadata or {})
        self._trades.append(trade)
        if len(self._trades) > self._max_trades_history:
            self._trades.pop(0)
        self.current_capital += pnl
        if self.current_capital > self._peak_capital:
            self._peak_capital = self.current_capital
        self._metrics_dirty = True
        logger.info(f"Trade: {trade_id} {direction.upper()} P&L=${pnl:+.2f} ({pnl_pct:+.2%})")
        return trade
    
    def _compute_trade_averages(self):
        """Compute average trade metrics."""
        n = len(self._trades)
        if n == 0:
            return Decimal("0"), 0.0, 0.0, 0.0
        return (sum(t.size for t in self._trades) / n,
                sum(t.duration_seconds for t in self._trades) / n,
                sum(t.signal_score for t in self._trades) / n,
                sum(t.signal_confidence for t in self._trades) / n)

    def calculate_metrics(self, force: bool = False) -> PerformanceMetrics:
        """Calculate current performance metrics."""
        if not force and not self._metrics_dirty and self._last_metrics:
            return self._last_metrics
        total_pnl = self.current_capital - self.initial_capital
        n = len(self._trades)
        wins = len([t for t in self._trades if t.pnl > 0])
        losses = len([t for t in self._trades if t.pnl < 0])
        roi = float(total_pnl / self.initial_capital)
        max_dd = float((self._peak_capital - self.current_capital) / self._peak_capital) if self._peak_capital > 0 else 0.0
        avg_size, avg_hold, avg_score, avg_conf = self._compute_trade_averages()
        metrics = PerformanceMetrics(
            timestamp=datetime.now(), total_pnl=total_pnl,
            realized_pnl=total_pnl, unrealized_pnl=Decimal("0"),
            total_trades=n, winning_trades=wins, losing_trades=losses,
            win_rate=wins / n if n > 0 else 0.0, roi=roi,
            sharpe_ratio=self._calculate_sharpe_ratio(), max_drawdown=max_dd,
            open_positions=0, avg_position_size=avg_size, avg_hold_time=avg_hold,
            total_exposure=Decimal("0"), risk_utilization=0.0,
            avg_signal_score=avg_score, avg_signal_confidence=avg_conf)
        self._last_metrics = metrics
        self._metrics_dirty = False
        self._metrics_history.append(metrics)
        return metrics
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio from trade returns."""
        if len(self._trades) < 2:
            return 0.0
        returns = [float(t.pnl / t.size) for t in self._trades if t.size > 0]
        if not returns:
            return 0.0
        mean_r = sum(returns) / len(returns)
        std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
        if std_r == 0:
            return 0.0
        return (mean_r - risk_free_rate / 252) / std_r * (252 ** 0.5)

    def get_trade_history(
        self,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Trade]:
        """
        Get trade history.
        
        Args:
            limit: Maximum trades to return
            start_date: Filter trades after this date
            end_date: Filter trades before this date
            
        Returns:
            List of trades
        """
        trades = self._trades
        
        # Apply date filters
        if start_date:
            trades = [t for t in trades if t.timestamp >= start_date]
        
        if end_date:
            trades = [t for t in trades if t.timestamp <= end_date]
        
        # Return most recent trades
        return trades[-limit:]
    
    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """
        Get equity curve over time.
        
        Returns:
            List of {timestamp, equity} points
        """
        curve = [
            {
                "timestamp": self._trades[0].timestamp if self._trades else datetime.now(),
                "equity": float(self.initial_capital),
            }
        ]
        
        running_capital = self.initial_capital
        
        for trade in self._trades:
            running_capital += trade.pnl
            curve.append({
                "timestamp": trade.timestamp,
                "equity": float(running_capital),
            })
        
        return curve
    
    def get_daily_pnl(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily P&L summary for last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        daily: Dict[str, Decimal] = {}
        for t in self._trades:
            if t.timestamp >= cutoff:
                key = t.timestamp.strftime("%Y-%m-%d")
                daily[key] = daily.get(key, Decimal("0")) + t.pnl
        return [{"date": d, "pnl": float(p)} for d, p in sorted(daily.items())]

    def get_win_loss_distribution(self) -> Dict[str, Any]:
        """
        Get win/loss distribution statistics.
        
        Returns:
            Distribution statistics
        """
        wins = [t.pnl for t in self._trades if t.pnl > 0]
        losses = [t.pnl for t in self._trades if t.pnl < 0]
        
        return {
            "total_trades": len(self._trades),
            "wins": {
                "count": len(wins),
                "total": float(sum(wins)),
                "avg": float(sum(wins) / len(wins)) if wins else 0.0,
                "max": float(max(wins)) if wins else 0.0,
            },
            "losses": {
                "count": len(losses),
                "total": float(sum(losses)),
                "avg": float(sum(losses) / len(losses)) if losses else 0.0,
                "max": float(min(losses)) if losses else 0.0,
            },
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0.0,
        }
    
    def export_for_grafana(self) -> Dict[str, Any]:
        """
        Export data in Grafana-friendly format.
        
        Returns:
            Dict with time-series data
        """
        metrics = self.calculate_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_pnl": float(metrics.total_pnl),
                "roi": metrics.roi * 100,  # As percentage
                "win_rate": metrics.win_rate * 100,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown * 100,
                "total_trades": metrics.total_trades,
                "current_capital": float(self.current_capital),
            },
            "equity_curve": self.get_equity_curve(),
            "daily_pnl": self.get_daily_pnl(30),
        }


# Singleton instance
_performance_tracker_instance = None

def get_performance_tracker() -> PerformanceTracker:
    """Get singleton performance tracker."""
    global _performance_tracker_instance
    if _performance_tracker_instance is None:
        _performance_tracker_instance = PerformanceTracker()
    return _performance_tracker_instance