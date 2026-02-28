"""
Protocol interfaces for dependency injection (DIP — Dependency Inversion Principle).

High-level modules (strategy, execution) depend on these abstractions,
not on concrete implementations.  This allows swapping live ↔ backtest ↔ mock
without touching business logic.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


# ── Volatility Estimation ────────────────────────────────────────────────────

@runtime_checkable
class IVolEstimator(Protocol):
    """Provides real-time volatility estimates."""

    def add_price(self, price: float, timestamp: float | None = None) -> None: ...

    def get_vol(self, method: str = "ewma") -> Any: ...

    def get_stats(self) -> Dict[str, Any]: ...

    def set_simulated_time(self, ts: float) -> None: ...


# ── Binary Option Pricing ────────────────────────────────────────────────────

@runtime_checkable
class IPricer(Protocol):
    """Prices binary options."""

    def price(self, spot: float, strike: float, vol: float,
              time_remaining_min: float, **kwargs) -> Any: ...


# ── Mispricing Detection ────────────────────────────────────────────────────

@runtime_checkable
class IMispricingDetector(Protocol):
    """Detects mispricing between model fair-value and market price."""

    def detect(self, *, yes_market_price: float, no_market_price: float,
               btc_spot: float, btc_strike: float,
               time_remaining_min: float, position_size_usd: float,
               use_maker: bool, **kwargs) -> Any: ...

    def get_stats(self) -> Dict[str, Any]: ...


# ── Risk Management ──────────────────────────────────────────────────────────

@runtime_checkable
class IRiskEngine(Protocol):
    """Validates position sizing and risk limits."""

    def validate_new_position(self, size: Decimal, direction: str,
                              current_price: Decimal) -> Tuple[bool, Optional[str]]: ...

    def calculate_position_size(self, signal_confidence: float,
                                signal_score: float,
                                current_price: Decimal) -> Decimal: ...

    def add_position(self, position_id: str, size: Decimal,
                     entry_price: Decimal, direction: str,
                     stop_loss: Optional[Decimal] = None,
                     take_profit: Optional[Decimal] = None) -> None: ...

    def update_position(self, position_id: str,
                        current_price: Decimal) -> Any: ...

    def remove_position(self, position_id: str,
                        exit_price: Decimal) -> Decimal: ...

    def get_risk_summary(self) -> Dict[str, Any]: ...


# ── Performance Tracking ─────────────────────────────────────────────────────

@runtime_checkable
class IPerformanceTracker(Protocol):
    """Records and queries trade history."""

    def record_trade(self, **kwargs) -> None: ...

    def get_trade_history(self, **kwargs) -> List[Any]: ...


# ── Signal Fusion ────────────────────────────────────────────────────────────

@runtime_checkable
class IFusionEngine(Protocol):
    """Fuses multiple trading signals into a single decision."""

    def fuse_signals(self, signals: List[Any], **kwargs) -> Any: ...

    def set_weight(self, source: str, weight: float) -> None: ...

    def get_statistics(self) -> Dict[str, Any]: ...


# ── Price Data Source ────────────────────────────────────────────────────────

@runtime_checkable
class IPriceSource(Protocol):
    """Abstract data source that provides BTC prices."""

    async def connect(self) -> bool: ...

    async def disconnect(self) -> None: ...

    async def get_current_price(self) -> Optional[float]: ...


# ── Sentiment Data Source ────────────────────────────────────────────────────

@runtime_checkable
class ISentimentSource(Protocol):
    """Abstract data source that provides sentiment data."""

    async def connect(self) -> bool: ...

    async def disconnect(self) -> None: ...

    async def get_fear_greed_index(self) -> Optional[Dict[str, Any]]: ...


# ── Metrics Exporter ─────────────────────────────────────────────────────────

@runtime_checkable
class IMetricsExporter(Protocol):
    """Exports metrics (e.g. to Prometheus/Grafana)."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def increment_trade_counter(self, won: bool) -> None: ...

