"""
Risk Engine
Manages position sizing, risk limits, and portfolio constraints
"""
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: Decimal  # Max USD per position
    max_total_exposure: Decimal  # Max total USD exposure
    max_positions: int  # Max concurrent positions
    max_drawdown_pct: float  # Max drawdown % before stop
    max_loss_per_day: Decimal  # Max daily loss
    max_leverage: float = 1.0  # Max leverage (1.0 = no leverage)


@dataclass
class PositionRisk:
    """Risk assessment for a position."""
    position_id: str
    current_size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    risk_level: RiskLevel
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    time_held: float  # seconds
    metadata: Dict[str, Any]


class RiskEngine:
    """
    Risk management engine.
    
    Enforces:
    - Position size limits (max $1 per trade)
    - Portfolio exposure limits
    - Drawdown controls
    - Loss limits
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """Initialize risk engine with limits."""
        self.limits = limits or RiskLimits(
            max_position_size=Decimal("1.0"), max_total_exposure=Decimal("10.0"),
            max_positions=5, max_drawdown_pct=0.15,
            max_loss_per_day=Decimal("5.0"), max_leverage=1.0)
        self._positions: Dict[str, PositionRisk] = {}
        self._daily_pnl = Decimal("0")
        self._daily_trades = 0
        self._peak_balance = self._current_balance = Decimal("1000.0")
        self._alerts: List[Dict[str, Any]] = []
        logger.info(f"Risk Engine: max_pos=${self.limits.max_position_size}, "
                    f"max_exp=${self.limits.max_total_exposure}")

    def validate_new_position(self, size: Decimal, direction: str,
                              current_price: Decimal) -> tuple[bool, Optional[str]]:
        """Validate if new position is within risk limits. Returns (is_valid, error)."""
        if size > self.limits.max_position_size:
            return False, f"Size ${size} exceeds max ${self.limits.max_position_size}"
        if len(self._positions) >= self.limits.max_positions:
            return False, f"Max positions reached ({self.limits.max_positions})"
        new_exp = self.get_total_exposure() + size
        if new_exp > self.limits.max_total_exposure:
            return False, f"Exposure ${new_exp} exceeds max ${self.limits.max_total_exposure}"
        if self._daily_pnl < -self.limits.max_loss_per_day:
            return False, f"Daily loss limit reached (${abs(self._daily_pnl)})"
        dd = self.get_current_drawdown()
        if dd > self.limits.max_drawdown_pct:
            return False, f"Drawdown {dd:.1%} exceeds max {self.limits.max_drawdown_pct:.1%}"
        return True, None
    
    def calculate_position_size(self, signal_confidence: float, signal_score: float,
                                current_price: Decimal, risk_percent: float = 0.02) -> Decimal:
        """Calculate position size capped at $1.00."""
        risk_amt = self._current_balance * Decimal(str(risk_percent))
        strength = Decimal(str(signal_confidence)) * Decimal(str(signal_score / 100))
        size = risk_amt * strength
        size = max(min(size, Decimal("1.0")), Decimal("1.0"))
        logger.info(f"Position size: ${size:.2f} (conf={signal_confidence:.2%}, score={signal_score:.1f})")
        return size

    def add_position(self, position_id: str, size: Decimal, entry_price: Decimal,
                     direction: str, stop_loss: Optional[Decimal] = None,
                     take_profit: Optional[Decimal] = None) -> None:
        """Add a new position to track."""
        self._positions[position_id] = PositionRisk(
            position_id=position_id, current_size=size,
            entry_price=entry_price, current_price=entry_price,
            unrealized_pnl=Decimal("0"), risk_level=RiskLevel.LOW,
            stop_loss=stop_loss, take_profit=take_profit, time_held=0.0,
            metadata={"direction": direction, "entry_time": datetime.now()})
        self._daily_trades += 1
        logger.info(f"Added position: {position_id} (${size:.2f} @ ${entry_price:.2f})")

    def _compute_pnl_pct(self, position, price):
        """Compute PnL percentage for a position at given price."""
        d = position.metadata.get("direction", "long")
        if d == "long":
            return (price - position.entry_price) / position.entry_price
        return (position.entry_price - price) / position.entry_price

    def update_position(self, position_id: str, current_price: Decimal) -> Optional[PositionRisk]:
        """Update position with current price. Returns updated position or None."""
        if position_id not in self._positions:
            return None
        pos = self._positions[position_id]
        pos.current_price = current_price
        pos.unrealized_pnl = pos.current_size * self._compute_pnl_pct(pos, current_price)
        entry_time = pos.metadata.get("entry_time", datetime.now())
        pos.time_held = (datetime.now() - entry_time).total_seconds()
        pos.risk_level = self._assess_risk_level(pos)
        if pos.stop_loss and self._check_stop_loss(pos, current_price):
            self._create_alert("STOP_LOSS", f"Stop loss hit: {position_id}", RiskLevel.HIGH)
        if pos.take_profit and self._check_take_profit(pos, current_price):
            self._create_alert("TAKE_PROFIT", f"Take profit hit: {position_id}", RiskLevel.LOW)
        return pos

    def remove_position(self, position_id: str, exit_price: Decimal) -> Optional[Decimal]:
        """Remove position, record P&L. Returns realized P&L or None."""
        if position_id not in self._positions:
            return None
        pos = self._positions[position_id]
        pnl_pct = self._compute_pnl_pct(pos, exit_price)
        realized = pos.current_size * pnl_pct
        self._current_balance += realized
        self._daily_pnl += realized
        if self._current_balance > self._peak_balance:
            self._peak_balance = self._current_balance
        del self._positions[position_id]
        logger.info(f"Closed: {position_id} P&L=${realized:+.2f} ({pnl_pct:+.2%})")
        return realized

    def _assess_risk_level(self, position: PositionRisk) -> RiskLevel:
        """Assess risk level of a position."""
        pnl_pct = position.unrealized_pnl / position.current_size if position.current_size > 0 else 0
        
        if pnl_pct < -0.10:  # -10% or worse
            return RiskLevel.CRITICAL
        elif pnl_pct < -0.05:  # -5% to -10%
            return RiskLevel.HIGH
        elif pnl_pct < -0.02:  # -2% to -5%
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _check_stop_loss(self, position: PositionRisk, current_price: Decimal) -> bool:
        """Check if stop loss is hit."""
        if not position.stop_loss:
            return False
        
        direction = position.metadata.get("direction", "long")
        
        if direction == "long":
            return current_price <= position.stop_loss
        else:  # short
            return current_price >= position.stop_loss
    
    def _check_take_profit(self, position: PositionRisk, current_price: Decimal) -> bool:
        """Check if take profit is hit."""
        if not position.take_profit:
            return False
        
        direction = position.metadata.get("direction", "long")
        
        if direction == "long":
            return current_price >= position.take_profit
        else:  # short
            return current_price <= position.take_profit
    
    def _create_alert(self, alert_type: str, message: str, risk_level: RiskLevel) -> None:
        """Create a risk alert."""
        alert = {
            "timestamp": datetime.now(),
            "type": alert_type,
            "message": message,
            "risk_level": risk_level.value,
        }
        
        self._alerts.append(alert)
        
        logger.warning(f"[{risk_level.value.upper()}] {alert_type}: {message}")
    
    def get_total_exposure(self) -> Decimal:
        """Get total current exposure across all positions."""
        return sum(pos.current_size for pos in self._positions.values())
    
    def get_total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self._positions.values())
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        if self._peak_balance == 0:
            return 0.0
        
        drawdown = (self._peak_balance - self._current_balance) / self._peak_balance
        return float(drawdown)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        return {
            "timestamp": datetime.now(),
            "positions": {
                "count": len(self._positions),
                "max_allowed": self.limits.max_positions,
            },
            "exposure": {
                "current": float(self.get_total_exposure()),
                "max_allowed": float(self.limits.max_total_exposure),
                "utilization_pct": float(self.get_total_exposure() / self.limits.max_total_exposure * 100) if self.limits.max_total_exposure > 0 else 0,
            },
            "pnl": {
                "daily": float(self._daily_pnl),
                "unrealized": float(self.get_total_unrealized_pnl()),
                "daily_limit": float(self.limits.max_loss_per_day),
            },
            "balance": {
                "current": float(self._current_balance),
                "peak": float(self._peak_balance),
                "drawdown_pct": self.get_current_drawdown() * 100,
                "max_drawdown_pct": self.limits.max_drawdown_pct * 100,
            },
            "daily_stats": {
                "trades": self._daily_trades,
                "pnl": float(self._daily_pnl),
            },
            "alerts": len([a for a in self._alerts if (datetime.now() - a["timestamp"]).seconds < 3600]),
        }
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of each day)."""
        self._daily_pnl = Decimal("0")
        self._daily_trades = 0
        logger.info("Reset daily statistics")


# Singleton instance
_risk_engine_instance = None

def get_risk_engine() -> RiskEngine:
    """Get singleton risk engine."""
    global _risk_engine_instance
    if _risk_engine_instance is None:
        _risk_engine_instance = RiskEngine()
    return _risk_engine_instance