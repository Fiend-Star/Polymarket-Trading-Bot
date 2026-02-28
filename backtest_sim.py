"""
Backtest market simulation — synthetic probability, RSI, realistic config.

SRP: Synthetic market simulation for backtesting.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional

from backtest_models import Candle


@dataclass
class RealisticConfig:
    """Calibrated simulation parameters from live Polymarket observations."""
    market_noise_std: float = 0.025
    market_efficiency: float = 0.35
    spread_cents: float = 0.03
    spread_widen_late: float = 0.02
    fill_rate: float = 0.72
    fill_rate_boost_edge: float = 0.10
    min_edge_realistic: float = 0.05
    min_vol_confidence: float = 0.40
    max_consecutive_losses: int = 3
    max_trade_rate: float = 0.60
    enabled: bool = False


def _intrawindow_vol(window_candles, decision_minute):
    if len(window_candles) <= 1: return 0.0
    returns = []
    for i in range(1, min(decision_minute + 1, len(window_candles))):
        prev = window_candles[i - 1].close
        if prev > 0:
            returns.append((window_candles[i].close - prev) / prev)
    if not returns: return 0.0
    return (sum(r**2 for r in returns) / len(returns))**0.5


def _realistic_probability(pct_change, rc, model_fv, wc, dm):
    iwv = _intrawindow_vol(wc, dm)
    vol_factor = max(0.5, min(1.2, 1.2 - iwv * 80))
    base = 0.50 + pct_change * 11.0 * vol_factor
    if model_fv is not None:
        base = rc.market_efficiency * model_fv + (1.0 - rc.market_efficiency) * base
    return base + random.gauss(0, rc.market_noise_std)


def btc_to_probability(window_candles: List[Candle], decision_minute: int = 2,
                       realistic: Optional[RealisticConfig] = None,
                       model_fair_value: Optional[float] = None) -> float:
    """Convert BTC price movement into synthetic Polymarket 'Up' probability."""
    if len(window_candles) < decision_minute + 1: return 0.50
    wo = window_candles[0].open
    cur = window_candles[decision_minute].close
    if wo == 0: return 0.50
    pct = (cur - wo) / wo
    if realistic and realistic.enabled:
        prob = _realistic_probability(pct, realistic, model_fair_value, window_candles, decision_minute)
    else:
        prob = 0.50 + pct * 15.0
    return max(0.05, min(0.95, prob))


def compute_rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1: return 50.0
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    recent = deltas[-period:]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    ag = sum(gains) / period if gains else 0.0001
    al = sum(losses) / period if losses else 0.0001
    return 100 - (100 / (1 + ag / al))


def apply_realistic_entry(rc, entry_price, time_remaining_min):
    if not rc: return entry_price
    hs = rc.spread_cents / 2.0
    if time_remaining_min <= 3.0: hs += rc.spread_widen_late / 2.0
    return min(0.95, entry_price + hs)


def check_realistic_fill(rc, signal):
    if not rc: return True
    eb = min(0.15, abs(signal.edge) * rc.fill_rate_boost_edge / 0.10)
    return random.random() <= min(0.95, rc.fill_rate + eb)

