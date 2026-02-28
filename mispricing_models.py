"""
Mispricing data models, fee calculation, and Kelly sizing.

SRP: Shared types and pure functions used by MispricingDetector.
"""
import math
from dataclasses import dataclass


# ── Polymarket fee calculation ───────────────────────────────────────

def polymarket_taker_fee(price: float) -> float:
    """Nonlinear taker fee: p × (1-p) × 0.0624. Max ~1.56% at p=0.50."""
    p = max(0.01, min(0.99, price))
    return 0.0624 * p * (1 - p)


def polymarket_taker_fee_usd(price: float, num_shares: float) -> float:
    """Total taker fee in USD."""
    return polymarket_taker_fee(price) * num_shares * price


# ── Kelly criterion ──────────────────────────────────────────────────

def kelly_fraction(true_prob: float, market_price: float,
                   use_half_kelly: bool = True, max_fraction: float = 0.05) -> float:
    """Kelly-optimal bankroll fraction for a binary option. Half-Kelly by default."""
    p = max(0.01, min(0.99, true_prob))
    q = max(0.01, min(0.99, market_price))
    if p <= q:
        return 0.0
    f = (p - q) / (1 - q)
    if use_half_kelly:
        f *= 0.5
    return max(0.0, min(max_fraction, f))


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class MispricingSignal:
    """A detected mispricing with trade recommendation."""
    edge: float
    edge_pct: float
    direction: str
    yes_market: float
    no_market: float
    yes_model: float
    no_model: float
    spot: float
    strike: float
    vol: float
    time_remaining_min: float
    delta: float
    gamma: float
    theta_per_min: float
    confidence: float
    vol_confidence: float
    implied_vol: float
    realized_vol: float
    vol_spread: float
    vrp: float
    vrp_signal: str
    expected_pnl: float
    fee_cost: float
    net_expected_pnl: float
    is_tradeable: bool
    kelly_fraction: float
    kelly_bet_usd: float
    pricing_method: str

    def __repr__(self):
        return (f"Mispricing({self.direction}: edge={self.edge:+.4f}, "
                f"net_EV=${self.net_expected_pnl:+.4f}, "
                f"kelly={self.kelly_fraction:.1%}, "
                f"{'✓ TRADE' if self.is_tradeable else '✗ SKIP'})")


@dataclass
class ExitSignal:
    """Signal to exit an existing position mid-market."""
    action: str
    current_value: float
    entry_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    reason: str

