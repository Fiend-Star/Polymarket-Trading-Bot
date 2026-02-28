"""
Performance reporting — query and export methods for PerformanceTracker.

SRP: Read-only reporting/export. PerformanceTracker owns write-path.
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional


class PerformanceReportMixin:
    """Mixin adding reporting methods to PerformanceTracker."""

    def get_trade_history(self, limit: int = 100,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> List:
        """Get filtered trade history."""
        trades = self._trades
        if start_date:
            trades = [t for t in trades if t.timestamp >= start_date]
        if end_date:
            trades = [t for t in trades if t.timestamp <= end_date]
        return trades[-limit:]

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Get equity curve over time."""
        curve = [{
            "timestamp": self._trades[0].timestamp if self._trades else datetime.now(),
            "equity": float(self.initial_capital),
        }]
        running = self.initial_capital
        for t in self._trades:
            running += t.pnl
            curve.append({"timestamp": t.timestamp, "equity": float(running)})
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
        """Get win/loss distribution statistics."""
        wins = [t.pnl for t in self._trades if t.pnl > 0]
        losses = [t.pnl for t in self._trades if t.pnl < 0]
        return {
            "total_trades": len(self._trades),
            "wins": {"count": len(wins), "total": float(sum(wins)),
                     "avg": float(sum(wins)/len(wins)) if wins else 0.0,
                     "max": float(max(wins)) if wins else 0.0},
            "losses": {"count": len(losses), "total": float(sum(losses)),
                       "avg": float(sum(losses)/len(losses)) if losses else 0.0,
                       "max": float(min(losses)) if losses else 0.0},
            "profit_factor": abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else 0.0,
        }

    def export_for_grafana(self) -> Dict[str, Any]:
        """Export data in Grafana-friendly format."""
        m = self.calculate_metrics()
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_pnl": float(m.total_pnl), "roi": m.roi * 100,
                "win_rate": m.win_rate * 100, "sharpe_ratio": m.sharpe_ratio,
                "max_drawdown": m.max_drawdown * 100, "total_trades": m.total_trades,
                "current_capital": float(self.current_capital)},
            "equity_curve": self.get_equity_curve(),
            "daily_pnl": self.get_daily_pnl(30),
        }

