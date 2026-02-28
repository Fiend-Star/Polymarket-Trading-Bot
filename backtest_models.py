"""
Backtest data models and configuration.

SRP: Data structures (Candle, BacktestTrade, BacktestResult) and env config.
"""
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# V3 quant parameters (from .env)
MIN_EDGE_CENTS      = float(os.getenv("MIN_EDGE_CENTS", "0.02"))
TAKE_PROFIT_PCT     = float(os.getenv("TAKE_PROFIT_PCT", "0.30"))
CUT_LOSS_PCT        = float(os.getenv("CUT_LOSS_PCT", "-0.50"))
VOL_METHOD          = os.getenv("VOL_METHOD", "ewma")
DEFAULT_VOL         = float(os.getenv("DEFAULT_VOL", "0.65"))
BANKROLL_USD        = float(os.getenv("BANKROLL_USD", "20.0"))
MARKET_BUY_USD      = float(os.getenv("MARKET_BUY_USD", "1.00"))
USE_LIMIT_ORDERS    = os.getenv("USE_LIMIT_ORDERS", "true").lower() == "true"

# Old fusion params
MIN_FUSION_SIGNALS  = int(os.getenv("MIN_FUSION_SIGNALS", "2"))
MIN_FUSION_SCORE    = float(os.getenv("MIN_FUSION_SCORE", "55.0"))
TREND_UP_THRESHOLD  = float(os.getenv("TREND_UP_THRESHOLD", "0.60"))
TREND_DOWN_THRESHOLD = float(os.getenv("TREND_DOWN_THRESHOLD", "0.40"))

WINDOW_MINUTES = 15


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestTrade:
    window_start: datetime
    direction: str
    entry_price: float
    position_size: float
    actual_btc_open: float
    actual_btc_close: float
    btc_moved_up: bool
    outcome: str
    pnl: float
    pnl_before_fees: float
    fee_paid: float
    model_yes_price: float
    model_no_price: float
    edge: float
    edge_pct: float
    kelly_fraction: float
    realized_vol: float
    implied_vol: float
    vrp: float
    pricing_method: str
    confirming_signals: int
    contradicting_signals: int
    vol_skew: float
    funding_bias: float
    funding_regime: str
    time_remaining_min: float


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    strategy_mode: str
    total_windows: int
    trades_taken: int
    trades_skipped: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    total_fees: float
    max_drawdown: float
    profit_factor: float
    avg_pnl_per_trade: float
    avg_edge_per_trade: float
    avg_kelly: float
    best_trade: float
    worst_trade: float
    sharpe_ratio: float
    bankroll_final: float
    bankroll_growth: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

