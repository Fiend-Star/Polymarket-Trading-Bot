"""
Service Container — wires and owns all shared service instances.

SRP:  This module's only job is construction + lifecycle of shared services.
DIP:  All consumers receive interfaces, not concrete classes.
OCP:  Adding a new service = one new property; existing code untouched.

Usage:
    container = ServiceContainer(cfg)       # build once at startup
    strategy  = IntegratedBTCStrategy(container)

    # Or use the backward-compatible module-level getters:
    from container import get_container
    container = get_container()
"""
from __future__ import annotations

from typing import Optional
from loguru import logger

from config import BotConfig, get_config
from interfaces import (
    IVolEstimator, IPricer, IMispricingDetector,
    IRiskEngine, IPerformanceTracker, IFusionEngine,
    IMetricsExporter,
)


class ServiceContainer:
    """
    Owns and lazily constructs all shared service instances.

    Every property returns a Protocol-typed reference so consumers
    never depend on concrete implementations.
    """

    def __init__(self, cfg: Optional[BotConfig] = None):
        self.cfg = cfg or get_config()
        self._vol_estimator: Optional[IVolEstimator] = None
        self._pricer: Optional[IPricer] = None
        self._mispricing_detector: Optional[IMispricingDetector] = None
        self._risk_engine: Optional[IRiskEngine] = None
        self._performance_tracker: Optional[IPerformanceTracker] = None
        self._fusion_engine: Optional[IFusionEngine] = None
        self._grafana_exporter: Optional[IMetricsExporter] = None
        self._learning_engine = None
        logger.info("ServiceContainer initialised")

    # ── Lazy constructors ────────────────────────────────────────────────

    @property
    def vol_estimator(self) -> IVolEstimator:
        if self._vol_estimator is None:
            from vol_estimator import get_vol_estimator
            self._vol_estimator = get_vol_estimator()
        return self._vol_estimator

    @property
    def pricer(self) -> IPricer:
        if self._pricer is None:
            from binary_pricer import get_binary_pricer
            self._pricer = get_binary_pricer()
        return self._pricer

    @property
    def mispricing_detector(self) -> IMispricingDetector:
        if self._mispricing_detector is None:
            from mispricing_detector import get_mispricing_detector
            q = self.cfg.quant
            self._mispricing_detector = get_mispricing_detector(
                maker_fee=0.00, taker_fee=0.10,
                min_edge_cents=q.min_edge_cents,
                min_edge_after_fees=0.005,
                take_profit_pct=q.take_profit_pct,
                cut_loss_pct=q.cut_loss_pct,
                vol_method=q.vol_method,
            )
        return self._mispricing_detector

    @property
    def risk_engine(self) -> IRiskEngine:
        if self._risk_engine is None:
            from execution.risk_engine import get_risk_engine
            self._risk_engine = get_risk_engine()
        return self._risk_engine

    @property
    def performance_tracker(self) -> IPerformanceTracker:
        if self._performance_tracker is None:
            from monitoring.performance_tracker import get_performance_tracker
            self._performance_tracker = get_performance_tracker()
        return self._performance_tracker

    @property
    def fusion_engine(self) -> IFusionEngine:
        if self._fusion_engine is None:
            from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine
            self._fusion_engine = get_fusion_engine()
        return self._fusion_engine

    @property
    def grafana_exporter(self) -> Optional[IMetricsExporter]:
        return self._grafana_exporter

    @grafana_exporter.setter
    def grafana_exporter(self, value: Optional[IMetricsExporter]):
        self._grafana_exporter = value

    @property
    def learning_engine(self):
        if self._learning_engine is None:
            from feedback.learning_engine import get_learning_engine
            self._learning_engine = get_learning_engine()
        return self._learning_engine

    # ── Inject overrides (for testing / backtesting) ─────────────────────

    def override(self, **kwargs):
        """
        Override any service with a mock/stub.

        Example:
            container.override(vol_estimator=FakeVolEstimator())
        """
        for key, value in kwargs.items():
            attr = f"_{key}"
            if hasattr(self, attr):
                setattr(self, attr, value)
                logger.debug(f"ServiceContainer: overrode {key}")
            else:
                raise KeyError(f"Unknown service: {key}")


# ── Module-level singleton ───────────────────────────────────────────────────

_container: Optional[ServiceContainer] = None


def get_container(cfg: Optional[BotConfig] = None) -> ServiceContainer:
    """Get or create the global service container."""
    global _container
    if _container is None:
        _container = ServiceContainer(cfg)
    return _container

