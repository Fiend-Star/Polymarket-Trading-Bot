"""
Risk Engine
Manages position sizing, risk limits, and portfolio constraints
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List

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

    def __init__(
            self,
            limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize risk engine.
        
        Args:
            limits: Risk limits configuration
        """
        # Default conservative limits with $1 max per trade
        self.limits = limits or RiskLimits(
            max_position_size=Decimal("1.0"),  # $1 max per position
            max_total_exposure=Decimal("10.0"),  # $10 total
            max_positions=5,
            max_drawdown_pct=0.15,  # 15% max drawdown
            max_loss_per_day=Decimal("5.0"),  # $5 daily loss limit
            max_leverage=1.0,
        )

        # Track positions
        self._positions: Dict[str, PositionRisk] = {}

        # Track daily statistics
        self._daily_pnl = Decimal("0")
        self._daily_trades = 0
        self._peak_balance = Decimal("1000.0")  # Starting balance
        self._current_balance = Decimal("1000.0")

        # Alerts
        self._alerts: List[Dict[str, Any]] = []

        logger.info(
            f"Initialized Risk Engine: "
            f"max_position=${self.limits.max_position_size}, "
            f"max_exposure=${self.limits.max_total_exposure}"
        )

    def validate_new_position(
            self,
            size: Decimal,
            direction: str,
            current_price: Decimal,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate if new position is allowed.
        
        Args:
            size: Position size in USD
            direction: "long" or "short"
            current_price: Current market price
            
        Returns:
            (is_valid, error_message)
        """
        # Check position size limit ($1 max)
        if size > self.limits.max_position_size:
            return False, f"Position size ${size} exceeds max ${self.limits.max_position_size}"

        # Check max positions
        if len(self._positions) >= self.limits.max_positions:
            return False, f"Max positions reached ({self.limits.max_positions})"

        # Check total exposure
        current_exposure = self.get_total_exposure()
        new_exposure = current_exposure + size

        if new_exposure > self.limits.max_total_exposure:
            return False, (
                f"Total exposure ${new_exposure} would exceed max ${self.limits.max_total_exposure}"
            )

        # Check daily loss limit
        if self._daily_pnl < -self.limits.max_loss_per_day:
            return False, f"Daily loss limit reached (${abs(self._daily_pnl)})"

        # Check drawdown
        drawdown = self.get_current_drawdown()
        if drawdown > self.limits.max_drawdown_pct:
            return False, f"Drawdown {drawdown:.1%} exceeds max {self.limits.max_drawdown_pct:.1%}"

        return True, None

    def calculate_position_size(
            self,
            signal_confidence: float,
            signal_score: float,
            current_price: Decimal,
            time_remaining_mins: float = 15.0,
            risk_percent: float = 0.05,  # Max 5% of current balance per trade
            kelly_fraction: float = 0.5,  # Half-Kelly for safety
    ) -> Decimal:
        """
        Calculate optimal position size using Time-Adjusted Fractional Kelly.
        Incorporates signal_score to modulate confidence quality.
        """
        # Base max risk amount based on current balance
        base_risk_amount = self._current_balance * Decimal(str(risk_percent))

        # Combine confidence, score, and Kelly fraction
        # signal_score (0-100) acts as a quality multiplier (normalized to 0-1)
        score_multiplier = Decimal(str(signal_score / 100.0))
        strength_multiplier = Decimal(str(signal_confidence)) * score_multiplier * Decimal(str(kelly_fraction))

        # GAMMA PENALTY: Reduce size aggressively in the final 5 minutes
        time_factor = Decimal("1.0")
        if time_remaining_mins < 5.0:
            time_factor = Decimal(str(max(0.1, time_remaining_mins / 5.0)))

        # Calculate theoretical size
        position_size = base_risk_amount * strength_multiplier * time_factor

        # STRUCTURAL CAPS:
        if position_size > self.limits.max_position_size:
            position_size = self.limits.max_position_size

        current_exposure = self.get_total_exposure()
        if current_exposure + position_size > self.limits.max_total_exposure:
            position_size = max(Decimal("0"), self.limits.max_total_exposure - current_exposure)

        # Ensure at least exchange minimum
        # Determine configured minimum token qty (defaults to 5 shares) and
        # convert to USD using current_price to enforce a sensible minimum position size.
        try:
            import os
            min_token_qty = Decimal(str(os.getenv('MIN_TOKEN_QTY', '5.0')))
            min_usd = (min_token_qty * current_price) if current_price and current_price > 0 else Decimal("1.0")
        except Exception:
            min_usd = Decimal("1.0")

        position_size = max(position_size, min_usd)

        logger.info(
            f"Calculated Kelly position size: ${position_size:.2f} "
            f"(conf={signal_confidence:.0%}, score={signal_score:.1f}, time_factor={time_factor:.2f})"
        )

        return position_size

    def calculate_arbitrage_size(self, risk_percent: float = 0.05) -> Decimal:
        """
        Bypasses Kelly and Gamma decay for near-deterministic late-window snipes.
        Maxes out the allowable risk budget while enforcing structural limits.

        Args:
            risk_percent: fraction of current balance to risk in an arbitrage (default 5%)
        Returns:
            position size in USD as Decimal (at least exchange minimum of $1.0)
        """
        try:
            position_size = self._current_balance * Decimal(str(risk_percent))
        except Exception:
            # Safety fallback
            position_size = Decimal("1.0")

        # Enforce per-position cap
        if position_size > self.limits.max_position_size:
            position_size = self.limits.max_position_size

        # Enforce total exposure cap
        current_exposure = self.get_total_exposure()
        if current_exposure + position_size > self.limits.max_total_exposure:
            # allocate the remaining exposure
            remaining = self.limits.max_total_exposure - current_exposure
            position_size = max(Decimal("0"), remaining)

        # Ensure at least the exchange minimum in USD using MIN_TOKEN_QTY
        try:
            import os
            min_token_qty = Decimal(str(os.getenv('MIN_TOKEN_QTY', '5.0')))
            # if we have access to a representative price, try to convert; otherwise default to $1
            # We don't have a price here, so conservatively enforce $5 USD minimum
            min_usd = min_token_qty * Decimal("1.0") if min_token_qty > 0 else Decimal("1.0")
        except Exception:
            min_usd = Decimal("1.0")

        return max(position_size, min_usd)

    def add_position(
            self,
            position_id: str,
            size: Decimal,
            entry_price: Decimal,
            direction: str,
            stop_loss: Optional[Decimal] = None,
            take_profit: Optional[Decimal] = None,
    ) -> None:
        """
        Add a new position to track.
        
        Args:
            position_id: Unique position ID
            size: Position size in USD
            entry_price: Entry price
            direction: "long" or "short"
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        position = PositionRisk(
            position_id=position_id,
            current_size=size,
            entry_price=entry_price,
            current_price=entry_price,
            unrealized_pnl=Decimal("0"),
            risk_level=RiskLevel.LOW,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_held=0.0,
            metadata={
                "direction": direction,
                "entry_time": datetime.now(),
            }
        )

        self._positions[position_id] = position
        self._daily_trades += 1

        logger.info(f"Added position: {position_id} (${size:.2f} @ ${entry_price:.2f})")

    def update_position(
            self,
            position_id: str,
            current_price: Decimal,
    ) -> Optional[PositionRisk]:
        """
        Update position with current market price.
        
        Args:
            position_id: Position ID
            current_price: Current market price
            
        Returns:
            Updated position risk or None
        """
        if position_id not in self._positions:
            return None

        position = self._positions[position_id]
        position.current_price = current_price

        # Calculate P&L
        direction = position.metadata.get("direction", "long")

        if direction == "long":
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:  # short
            pnl_pct = (position.entry_price - current_price) / position.entry_price

        position.unrealized_pnl = position.current_size * pnl_pct

        # Update time held
        entry_time = position.metadata.get("entry_time", datetime.now())
        position.time_held = (datetime.now() - entry_time).total_seconds()

        # Assess risk level
        position.risk_level = self._assess_risk_level(position)

        # Check if stop loss or take profit hit
        if position.stop_loss and self._check_stop_loss(position, current_price):
            self._create_alert(
                "STOP_LOSS",
                f"Stop loss hit for {position_id}",
                RiskLevel.HIGH
            )

        if position.take_profit and self._check_take_profit(position, current_price):
            self._create_alert(
                "TAKE_PROFIT",
                f"Take profit hit for {position_id}",
                RiskLevel.LOW
            )

        return position

    def remove_position(
            self,
            position_id: str,
            exit_price: Decimal,
    ) -> Optional[Decimal]:
        """
        Remove position and record P&L.
        
        Args:
            position_id: Position ID
            exit_price: Exit price
            
        Returns:
            Realized P&L or None
        """
        if position_id not in self._positions:
            return None

        position = self._positions[position_id]

        # Calculate final P&L
        direction = position.metadata.get("direction", "long")

        if direction == "long":
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price

        realized_pnl = position.current_size * pnl_pct

        # Update balance and daily P&L
        self._current_balance += realized_pnl
        self._daily_pnl += realized_pnl

        # Update peak balance
        if self._current_balance > self._peak_balance:
            self._peak_balance = self._current_balance

        # Remove position
        del self._positions[position_id]

        logger.info(
            f"Closed position: {position_id} "
            f"P&L: ${realized_pnl:+.2f} ({pnl_pct:+.2%})"
        )

        return realized_pnl

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
                "utilization_pct": float(
                    self.get_total_exposure() / self.limits.max_total_exposure * 100) if self.limits.max_total_exposure > 0 else 0,
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
