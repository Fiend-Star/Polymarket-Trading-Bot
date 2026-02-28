"""
Grafana Metrics Exporter
Exports trading metrics in Prometheus format for Grafana
"""
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server, REGISTRY, generate_latest,
    CONTENT_TYPE_LATEST, CollectorRegistry, multiprocess,
)
from http.server import HTTPServer
import threading
from loguru import logger

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from monitoring.performance_tracker import get_performance_tracker
from execution.risk_engine import get_risk_engine
from execution.execution_engine import get_execution_engine
from monitoring.metrics_handler import MetricsHandler


class GrafanaMetricsExporter:
    """
    Exports metrics to Prometheus/Grafana.

    Exposes metrics on HTTP endpoint for Grafana to scrape.
    Now handles Grafana's API probes correctly.
    """

    def __init__(self, port: int = 8000, update_interval: int = 5,
                 performance_tracker=None, risk_engine=None, execution_engine=None):
        """Initialize exporter with optional injected dependencies (DIP)."""
        self.port = port
        self.update_interval = update_interval
        self.performance = performance_tracker or get_performance_tracker()
        self.risk = risk_engine or get_risk_engine()
        self.execution = execution_engine or get_execution_engine()
        self._setup_metrics()
        self._is_running = False
        self._server = None
        self._thread = None
        logger.info(f"Grafana Metrics Exporter (port {port})")

    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics."""
        self._setup_gauges()
        self._setup_counters()
        self.trade_duration = Histogram(
            'trading_trade_duration_seconds', 'Trade duration in seconds',
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800])

    def _setup_gauges(self):
        """Setup all Gauge metrics."""
        gauge_defs = [
            ('total_pnl', 'trading_total_pnl', 'Total profit/loss in USD'),
            ('roi', 'trading_roi', 'Return on investment as percentage'),
            ('win_rate', 'trading_win_rate', 'Percentage of winning trades'),
            ('sharpe_ratio', 'trading_sharpe_ratio', 'Sharpe ratio'),
            ('max_drawdown', 'trading_max_drawdown', 'Maximum drawdown as percentage'),
            ('open_positions', 'trading_open_positions', 'Number of open positions'),
            ('total_exposure', 'trading_total_exposure', 'Total exposure in USD'),
            ('risk_utilization', 'trading_risk_utilization', 'Risk limits utilized pct'),
            ('current_capital', 'trading_current_capital', 'Current capital in USD'),
            ('avg_signal_score', 'trading_avg_signal_score', 'Average signal score'),
            ('avg_signal_confidence', 'trading_avg_signal_confidence', 'Average confidence'),
        ]
        for attr, name, desc in gauge_defs:
            setattr(self, attr, Gauge(name, desc))

    def _setup_counters(self):
        """Setup all Counter metrics."""
        counter_defs = [
            ('total_trades', 'trades_total', 'Total trades executed'),
            ('winning_trades', 'trading_winning_trades', 'Winning trades'),
            ('losing_trades', 'trading_losing_trades', 'Losing trades'),
            ('orders_placed', 'trading_orders_placed', 'Orders placed'),
            ('orders_filled', 'trading_orders_filled', 'Orders filled'),
            ('orders_rejected', 'trading_orders_rejected', 'Orders rejected'),
        ]
        for attr, name, desc in counter_defs:
            setattr(self, attr, Counter(name, desc))

    def update_metrics(self) -> None:
        """Update all metrics with current values."""
        try:
            m = self.performance.calculate_metrics()
            self.total_pnl.set(float(m.total_pnl))
            self.roi.set(m.roi * 100)
            self.win_rate.set(m.win_rate * 100)
            self.sharpe_ratio.set(m.sharpe_ratio)
            self.max_drawdown.set(m.max_drawdown * 100)
            self.open_positions.set(m.open_positions)
            self.total_exposure.set(float(m.total_exposure))
            self.avg_signal_score.set(m.avg_signal_score)
            self.avg_signal_confidence.set(m.avg_signal_confidence)
            self.current_capital.set(float(self.performance.current_capital))
            risk = self.risk.get_risk_summary()
            if risk:
                self.risk_utilization.set(risk['exposure']['utilization_pct'])
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def start(self) -> None:
        """Start metrics server and update loop."""
        if self._is_running:
            logger.warning("Metrics exporter already running")
            return
        
        try:
            # Set the exporter reference in the handler
            MetricsHandler.exporter = self
            
            # Create and start custom HTTP server
            self._server = HTTPServer(('0.0.0.0', self.port), MetricsHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            
            logger.info(f"✓ Metrics server started on http://localhost:{self.port}/metrics")

            self._is_running = True
            
            # Start update loop
            asyncio.create_task(self._update_loop())
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    async def _update_loop(self) -> None:
        """Periodically update metrics."""
        while self._is_running:
            try:
                self.update_metrics()
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def stop(self) -> None:
        """Stop metrics server."""
        self._is_running = False
        
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            
        logger.info("Metrics exporter stopped")
    
    def increment_trade_counter(self, won: bool) -> None:
        """
        Increment trade counter.
        
        Args:
            won: True if trade was profitable
        """
        self.total_trades.inc()
        
        if won:
            self.winning_trades.inc()
        else:
            self.losing_trades.inc()
    
    def record_trade_duration(self, duration_seconds: float) -> None:
        """
        Record trade duration.
        
        Args:
            duration_seconds: Duration in seconds
        """
        self.trade_duration.observe(duration_seconds)
    
    def increment_order_counter(self, status: str) -> None:
        """
        Increment order counter.
        
        Args:
            status: "placed", "filled", or "rejected"
        """
        if status == "placed":
            self.orders_placed.inc()
        elif status == "filled":
            self.orders_filled.inc()
        elif status == "rejected":
            self.orders_rejected.inc()


# Singleton instance
_grafana_exporter_instance = None

def get_grafana_exporter() -> GrafanaMetricsExporter:
    """Get singleton Grafana exporter."""
    global _grafana_exporter_instance
    if _grafana_exporter_instance is None:
        _grafana_exporter_instance = GrafanaMetricsExporter()
    return _grafana_exporter_instance