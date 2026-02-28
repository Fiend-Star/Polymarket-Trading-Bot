"""
Mispricing Detector for Polymarket Binary Options — V2

RESEARCH-DRIVEN UPGRADES over V1:

  ██ CRITICAL BUG FIX ██
  1. Fee calculation: V1 used FLAT 10% taker fee — WRONG
     Actual Polymarket fee is NONLINEAR:
       fee_per_share = p × (1-p) × base_rate
     Maximum ~1.56% at p=0.50, declining to ~0.56% at p=0.10/0.90
     Maker fee: 0% (unchanged)
     THIS ALONE changes edge detection — V1 required 10% edge, real threshold is ~2%

  2. Kelly criterion position sizing
     f* = p - q(1-p)/(1-q) for binary payoffs
     Use half-Kelly in practice (75% of optimal growth, 50% less drawdown)
     Cap at 5% of bankroll per trade

  3. Jump-diffusion integration
     Pricer now uses Merton JD with jump params from VolEstimator
     Detector passes jump parameters through to pricer automatically

  4. Variance Risk Premium (VRP) edge
     BTC IV overprices RV ~70% of time by ~15 vol points
     When VRP is wide + market prices extreme → fade the extreme
     Structural edge: ~14% annualized VRP for BTC

  5. Return statistics integration
     Mean reversion signal from VolEstimator feeds into pricer overlays
     Recent return + σ-units passed through automatically

  6. Improved tradeability logic
     - Min edge scaled to fee (not hardcoded)
     - Time-decay awareness (don't trade when theta > edge)
     - Vol confidence weighting
     - VRP confirmation
"""

import time
import math
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from binary_pricer import BinaryOptionPricer, BinaryOptionPrice, get_binary_pricer
from vol_estimator import VolEstimator, VolEstimate, get_vol_estimator


# ---------------------------------------------------------------------------
# Polymarket fee calculation (CRITICAL FIX)
# ---------------------------------------------------------------------------

def polymarket_taker_fee(price: float) -> float:
    """
    Polymarket nonlinear taker fee for gamma/crypto 15-min markets.

    Formula: fee_per_share = p × (1-p) × base_rate
    Where base_rate ≈ 0.0624 (calibrated so max fee = 1.56% at p=0.50)

    V1 BUG: Used flat 10% — was 6.4× too high, killed all thin edges.

    Returns: effective fee rate (0 to 0.0156)
    """
    p = max(0.01, min(0.99, price))
    # base_rate calibrated: 0.0624 × 0.25 = 0.0156 = 1.56% at p=0.50
    return 0.0624 * p * (1 - p)


def polymarket_taker_fee_usd(price: float, num_shares: float) -> float:
    """Total taker fee in USD for a trade."""
    rate = polymarket_taker_fee(price)
    return rate * num_shares * price


# ---------------------------------------------------------------------------
# Kelly criterion for binary payoffs
# ---------------------------------------------------------------------------

def kelly_fraction(
    true_prob: float,
    market_price: float,
    use_half_kelly: bool = True,
    max_fraction: float = 0.05,
) -> float:
    """
    Kelly-optimal fraction of bankroll to bet on a binary option.

    For binary: buy at price q, win (1-q) if correct, lose q if wrong.
      Full Kelly: f* = (p(1-q) - (1-p)q) / (1-q)
                     = (p - q) / (1 - q)

    Half Kelly captures ~75% of optimal growth with ~50% less drawdown.
    Full Kelly has 1/3 chance of halving bankroll before doubling.

    Args:
        true_prob: Model's estimated probability of winning (p)
        market_price: Price paid for the token (q)
        use_half_kelly: Use half Kelly (recommended)
        max_fraction: Maximum bet as fraction of bankroll

    Returns:
        Optimal fraction of bankroll to bet (0 if no edge)
    """
    p = max(0.01, min(0.99, true_prob))
    q = max(0.01, min(0.99, market_price))

    if p <= q:
        return 0.0  # No edge

    # Kelly for binary: f* = (p - q) / (1 - q)
    f_star = (p - q) / (1 - q)

    if use_half_kelly:
        f_star *= 0.5

    # Cap at max_fraction (never risk more than 5% of bankroll on one trade)
    return max(0.0, min(max_fraction, f_star))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MispricingSignal:
    """A detected mispricing with trade recommendation."""
    # Edge
    edge: float
    edge_pct: float
    direction: str              # "BUY_YES", "BUY_NO", or "NO_TRADE"

    # Prices
    yes_market: float
    no_market: float
    yes_model: float
    no_model: float

    # Model inputs
    spot: float
    strike: float
    vol: float
    time_remaining_min: float

    # Greeks
    delta: float
    gamma: float
    theta_per_min: float

    # Confidence
    confidence: float
    vol_confidence: float

    # Vol analysis
    implied_vol: float
    realized_vol: float
    vol_spread: float           # IV - RV (positive = market overpricing vol)

    # V2: Variance Risk Premium
    vrp: float                  # Current VRP (IV - RV annualized)
    vrp_signal: str             # "FADE_EXTREME", "NEUTRAL", "VOL_CHEAP"

    # Expected PnL
    expected_pnl: float
    fee_cost: float             # V2: Uses correct nonlinear formula
    net_expected_pnl: float
    is_tradeable: bool

    # V2: Kelly sizing
    kelly_fraction: float       # Optimal bankroll fraction (half-Kelly)
    kelly_bet_usd: float        # Dollar amount at optimal sizing

    # V2: Pricing method
    pricing_method: str         # "adjusted", "merton_jd", "bsm"

    def __repr__(self):
        return (
            f"Mispricing({self.direction}: edge={self.edge:+.4f} ({self.edge_pct:+.1%}), "
            f"YES mkt=${self.yes_market:.3f} vs model=${self.yes_model:.3f}, "
            f"IV={self.implied_vol:.0%} vs RV={self.realized_vol:.0%}, "
            f"VRP={self.vrp:+.0%}, "
            f"net_EV=${self.net_expected_pnl:+.4f}, "
            f"kelly={self.kelly_fraction:.1%}, "
            f"{'✓ TRADE' if self.is_tradeable else '✗ SKIP'})"
        )


@dataclass
class ExitSignal:
    """Signal to exit an existing position mid-market."""
    action: str                 # "TAKE_PROFIT", "CUT_LOSS", "HOLD"
    current_value: float
    entry_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    reason: str


# ---------------------------------------------------------------------------
# Mispricing Detector V2
# ---------------------------------------------------------------------------

class MispricingDetector:
    """
    Quantitative edge detection engine — V2.

    Key V2 changes:
      - Correct Polymarket fee (nonlinear, ~1.56% max, not 10%)
      - Kelly criterion position sizing
      - Jump-diffusion parameters from VolEstimator
      - Mean reversion + seasonality overlays via pricer
      - VRP-based edge detection
    """

    def __init__(
        self,
        pricer: Optional[BinaryOptionPricer] = None,
        vol_estimator: Optional[VolEstimator] = None,
        # V2: maker_fee stays 0%, taker_fee is now computed dynamically
        maker_fee: float = 0.00,
        taker_fee: float = 0.02,    # V2: DEFAULT ~2% (was 10%!!) — overridden by nonlinear formula
        min_edge_cents: float = 0.02,
        min_edge_after_fees: float = 0.005,
        take_profit_pct: float = 0.30,
        cut_loss_pct: float = -0.50,
        vol_method: str = "ewma",
        # V2: Kelly parameters
        bankroll: float = 50.0,     # Total bankroll in USD
        use_half_kelly: bool = True,
        max_kelly_fraction: float = 0.05,
    ):
        self.pricer = pricer or get_binary_pricer()
        self.vol_est = vol_estimator or get_vol_estimator()
        self.maker_fee = maker_fee
        self.taker_fee_default = taker_fee  # Fallback only
        self.min_edge_cents = min_edge_cents
        self.min_edge_after_fees = min_edge_after_fees
        self.take_profit_pct = take_profit_pct
        self.cut_loss_pct = cut_loss_pct
        self.vol_method = vol_method
        self.bankroll = bankroll
        self.use_half_kelly = use_half_kelly
        self.max_kelly_fraction = max_kelly_fraction

        self._total_detections = 0
        self._tradeable_detections = 0

        logger.info(
            f"Initialized MispricingDetector V2: "
            f"maker_fee={maker_fee:.0%}, "
            f"min_edge=${min_edge_cents:.2f}, "
            f"bankroll=${bankroll:.0f}, "
            f"kelly={'half' if use_half_kelly else 'full'}, "
            f"vol_method={vol_method}"
        )

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(
        self,
        yes_market_price: float,
        no_market_price: float,
        btc_spot: float,
        btc_strike: float,
        time_remaining_min: float,
        position_size_usd: float = 1.0,
        use_maker: bool = True,
        # V3: Additional overlays
        vol_skew: Optional[float] = None,
        funding_bias: float = 0.0,
    ) -> MispricingSignal:
        """
        Detect mispricing between model and market.
        V2: Uses jump-diffusion pricer, correct fees, Kelly sizing.
        """
        self._total_detections += 1

        # ---- Step 1: Get vol estimate + jump params + return stats ----
        vol_estimate = self.vol_est.get_vol(method=self.vol_method)
        realized_vol = vol_estimate.annualized_vol
        vol_confidence = vol_estimate.confidence

        jump_params = self.vol_est.get_jump_params()
        return_stats = self.vol_est.get_return_stats(reference_price=btc_strike)

        # ---- Step 2: Price with jump-diffusion + overlays ----
        model = self.pricer.price(
            spot=btc_spot,
            strike=btc_strike,
            vol=realized_vol,
            time_remaining_min=time_remaining_min,
            # V2: Jump parameters from estimator
            jump_intensity=jump_params.intensity,
            jump_mean=jump_params.mean,
            jump_vol=jump_params.vol,
            # V2: Mean reversion inputs
            recent_return=return_stats.recent_return,
            recent_return_sigma=return_stats.recent_return_sigma,
            apply_overlays=True,
            # V3: Skew + funding overlays
            vol_skew=vol_skew,
            funding_bias=funding_bias,
        )

        # ---- Step 2b: Strike validation gate ----
        # If model says ~50% but market says ~20%, our strike is likely wrong.
        # This catches the cold-start bug where current Chainlink ≠ settlement reference.
        model_market_divergence = abs(model.yes_fair_value - yes_market_price)
        if model_market_divergence > 0.25:
            logger.warning(
                f"⚠ STRIKE VALIDATION FAILED: model YES={model.yes_fair_value:.3f} vs "
                f"market YES={yes_market_price:.3f} (Δ={model_market_divergence:.3f}) — "
                f"possible strike mismatch, skipping trade"
            )
            return MispricingSignal(
                edge=0.0, edge_pct=0.0, direction="NO_TRADE",
                yes_market=yes_market_price, no_market=no_market_price,
                yes_model=model.yes_fair_value, no_model=model.no_fair_value,
                spot=btc_spot, strike=btc_strike, vol=realized_vol,
                time_remaining_min=time_remaining_min,
                delta=model.delta, gamma=model.gamma, theta_per_min=model.theta,
                confidence=0.0, vol_confidence=vol_confidence,
                implied_vol=realized_vol, realized_vol=realized_vol,
                vol_spread=0.0, vrp=0.0, vrp_signal="STRIKE_MISMATCH",
                expected_pnl=0.0, fee_cost=0.0, net_expected_pnl=0.0,
                is_tradeable=False, kelly_fraction=0.0, kelly_bet_usd=0.0,
                pricing_method=model.method,
            )

        # ---- Step 3: Implied vol (BSM-based, standard) ----
        try:
            iv = self.pricer.implied_vol(
                market_price=yes_market_price,
                spot=btc_spot,
                strike=btc_strike,
                time_remaining_min=time_remaining_min,
                is_call=True,
            )
        except Exception:
            iv = None

        # IV convergence guard: if bisection couldn't find any vol that explains
        # the market price, the spot/strike inputs are wrong. This is the smoke
        # signal — treat it as a hard trading blocker.
        if iv is None:
            logger.warning(
                f"⚠ IV EXTRACTION FAILED: no vol in [0.01, 5.0] explains "
                f"market_price={yes_market_price:.4f} at spot={btc_spot:.2f}, "
                f"strike={btc_strike:.2f}, T={time_remaining_min:.1f}min — "
                f"marking untradeable"
            )
            return MispricingSignal(
                edge=0.0, edge_pct=0.0, direction="NO_TRADE",
                yes_market=yes_market_price, no_market=no_market_price,
                yes_model=model.yes_fair_value, no_model=model.no_fair_value,
                spot=btc_spot, strike=btc_strike, vol=realized_vol,
                time_remaining_min=time_remaining_min,
                delta=model.delta, gamma=model.gamma, theta_per_min=model.theta,
                confidence=0.0, vol_confidence=vol_confidence,
                implied_vol=0.0, realized_vol=realized_vol,
                vol_spread=0.0, vrp=0.0, vrp_signal="IV_FAILED",
                expected_pnl=0.0, fee_cost=0.0, net_expected_pnl=0.0,
                is_tradeable=False, kelly_fraction=0.0, kelly_bet_usd=0.0,
                pricing_method=model.method,
            )

        vol_spread = iv - realized_vol

        # V2: Track VRP
        self.vol_est.record_iv(iv)
        self.vol_est.record_rv(realized_vol)
        vrp = self.vol_est.get_vrp()

        # VRP signal interpretation
        if vrp > 0.20:  # IV exceeds RV by 20%+
            vrp_signal = "FADE_EXTREME"  # Market overpricing vol → fade extreme prices
        elif vrp < -0.05:
            vrp_signal = "VOL_CHEAP"     # Rare: market underpricing vol
        else:
            vrp_signal = "NEUTRAL"

        # ---- Step 4: Compute edge on both sides ----
        yes_edge = model.yes_fair_value - yes_market_price
        no_edge = model.no_fair_value - no_market_price

        # Pick the side with more edge
        if abs(yes_edge) >= abs(no_edge):
            primary_edge = yes_edge
            if yes_edge > 0:
                direction = "BUY_YES"
                trade_price = yes_market_price
                model_prob = model.yes_fair_value
            else:
                direction = "BUY_NO"
                trade_price = no_market_price
                primary_edge = no_edge
                model_prob = model.no_fair_value
        else:
            primary_edge = no_edge
            if no_edge > 0:
                direction = "BUY_NO"
                trade_price = no_market_price
                model_prob = model.no_fair_value
            else:
                direction = "BUY_YES"
                trade_price = yes_market_price
                primary_edge = yes_edge
                model_prob = model.yes_fair_value

        abs_edge = abs(primary_edge)
        edge_pct = abs_edge / trade_price if trade_price > 0 else 0.0

        # ---- Step 5: Fee calculation — V2 CRITICAL FIX ----
        if use_maker:
            fee_rate = self.maker_fee  # 0%
        else:
            # V2: Nonlinear Polymarket taker fee
            fee_rate = polymarket_taker_fee(trade_price)

        num_tokens = position_size_usd / trade_price if trade_price > 0 else 0
        expected_pnl = abs_edge * num_tokens
        fee_cost = fee_rate * num_tokens * trade_price  # Correct nonlinear fee
        net_expected_pnl = expected_pnl - fee_cost

        # ---- Step 6: Kelly criterion sizing ----
        kf = kelly_fraction(
            true_prob=model_prob,
            market_price=trade_price,
            use_half_kelly=self.use_half_kelly,
            max_fraction=self.max_kelly_fraction,
        )
        kelly_bet = kf * self.bankroll

        # ---- Step 7: Tradeability decision ----
        # V2: More nuanced than V1's simple threshold check
        min_edge = max(
            self.min_edge_cents,
            fee_rate * 2.0,  # Edge must be at least 2× the fee
        )

        is_tradeable = (
            abs_edge >= min_edge
            and net_expected_pnl >= self.min_edge_after_fees
            and time_remaining_min >= 1.0
            and vol_confidence >= 0.2    # V2: Lowered from 0.3 (JD more robust)
            and kf > 0.001              # V2: Kelly says there's an edge
        )

        # V2: VRP confirmation — if VRP is wide and we're fading an extreme, boost confidence
        vrp_boost = 0.0
        if vrp_signal == "FADE_EXTREME":
            # We should be fading extreme market prices
            if (direction == "BUY_NO" and yes_market_price > 0.70) or \
               (direction == "BUY_YES" and no_market_price > 0.70):
                vrp_boost = 0.10  # Extra confidence for VRP-confirmed trades

        # V2: Theta check — don't trade if time decay exceeds edge
        if time_remaining_min < 3.0 and abs(model.theta) * time_remaining_min > abs_edge * 0.5:
            is_tradeable = False  # Theta bleed would eat most of our edge

        if not is_tradeable:
            direction = "NO_TRADE"
            kf = 0.0
            kelly_bet = 0.0

        # Confidence score
        edge_conf = min(1.0, abs_edge / 0.10)
        time_conf = min(1.0, time_remaining_min / 5.0)
        confidence = (
            vol_confidence * 0.3
            + edge_conf * 0.3
            + time_conf * 0.2
            + (0.1 if kf > 0.01 else 0.0)  # Kelly confirms edge
            + vrp_boost
        )

        if is_tradeable:
            self._tradeable_detections += 1

        signal = MispricingSignal(
            edge=primary_edge,
            edge_pct=edge_pct,
            direction=direction,
            yes_market=yes_market_price,
            no_market=no_market_price,
            yes_model=model.yes_fair_value,
            no_model=model.no_fair_value,
            spot=btc_spot,
            strike=btc_strike,
            vol=realized_vol,
            time_remaining_min=time_remaining_min,
            delta=model.delta,
            gamma=model.gamma,
            theta_per_min=model.theta,
            confidence=confidence,
            vol_confidence=vol_confidence,
            implied_vol=iv,
            realized_vol=realized_vol,
            vol_spread=vol_spread,
            vrp=vrp,
            vrp_signal=vrp_signal,
            expected_pnl=expected_pnl,
            fee_cost=fee_cost,
            net_expected_pnl=net_expected_pnl,
            is_tradeable=is_tradeable,
            kelly_fraction=kf,
            kelly_bet_usd=kelly_bet,
            pricing_method=model.method,
        )

        # ---- Logging ----
        logger.info("=" * 70)
        logger.info("BINARY OPTION MISPRICING ANALYSIS — V2")
        logger.info(f"  BTC: spot=${btc_spot:,.2f}, strike=${btc_strike:,.2f}, T={time_remaining_min:.1f}min")
        logger.info(f"  Vol: RV={realized_vol:.1%} ({vol_estimate.method}), IV={iv:.1%}, spread={vol_spread:+.1%}")
        logger.info(f"  VRP: {vrp:+.1%} → {vrp_signal}")
        logger.info(f"  Jumps: λ={jump_params.intensity:.0f}/yr, recent={jump_params.recent_jump}")
        logger.info(f"  Returns: ret={return_stats.recent_return:+.3%}, σ={return_stats.recent_return_sigma:+.2f}")
        logger.info(f"  Model: YES=${model.yes_fair_value:.4f}, NO=${model.no_fair_value:.4f} [{model.method}]")
        logger.info(f"    BSM base={model.bsm_base:.4f}, JD adj={model.jump_adjustment:+.4f}")
        logger.info(f"    MR adj={model.mean_reversion_adj:+.4f}, candle={model.candle_effect_adj:+.4f}, season={model.seasonality_adj:+.4f}")
        logger.info(f"  Market: YES=${yes_market_price:.4f}, NO=${no_market_price:.4f}")
        logger.info(f"  Edge: {primary_edge:+.4f} ({edge_pct:+.1%})")
        logger.info(f"  Fee: {fee_rate:.2%} (V2 nonlinear) → cost=${fee_cost:.4f}")
        logger.info(f"  Net EV: ${net_expected_pnl:+.4f}")
        logger.info(f"  Kelly: f*={kf:.2%}, bet=${kelly_bet:.2f} of ${self.bankroll:.0f} bankroll")
        logger.info(f"  Greeks: Δ={model.delta:.6f}, Γ={model.gamma:.6f}, Θ={model.theta:.6f}/min")
        logger.info(f"  → {direction} {'✓ TRADEABLE' if is_tradeable else '✗ NO TRADE'} (conf={confidence:.0%})")
        logger.info("=" * 70)

        return signal

    # ------------------------------------------------------------------
    # Exit monitoring
    # ------------------------------------------------------------------

    def check_exit(
        self,
        entry_price: float,
        direction: str,
        btc_spot: float,
        btc_strike: float,
        time_remaining_min: float,
        yes_market_price: float,
        no_market_price: float,
    ) -> ExitSignal:
        """Check if an open position should be exited mid-market."""
        vol_estimate = self.vol_est.get_vol(method=self.vol_method)

        model = self.pricer.price(
            spot=btc_spot,
            strike=btc_strike,
            vol=vol_estimate.annualized_vol,
            time_remaining_min=time_remaining_min,
        )

        if direction == "BUY_YES":
            current_model_value = model.yes_fair_value
            current_market_value = yes_market_price
        else:
            current_model_value = model.no_fair_value
            current_market_value = no_market_price

        current_value = max(current_model_value, current_market_value)
        unrealized_pnl = current_value - entry_price
        unrealized_pnl_pct = unrealized_pnl / entry_price if entry_price > 0 else 0.0

        if unrealized_pnl_pct >= self.take_profit_pct:
            return ExitSignal(
                action="TAKE_PROFIT",
                current_value=current_value,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                reason=f"Profit target hit: {unrealized_pnl_pct:.0%} >= {self.take_profit_pct:.0%}"
            )

        if unrealized_pnl_pct <= self.cut_loss_pct and time_remaining_min > 2.0:
            return ExitSignal(
                action="CUT_LOSS",
                current_value=current_value,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                reason=f"Stop loss hit: {unrealized_pnl_pct:.0%} <= {self.cut_loss_pct:.0%}"
            )

        if time_remaining_min < 2.0 and unrealized_pnl_pct < -0.20:
            return ExitSignal(
                action="HOLD",
                current_value=current_value,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                reason=f"Near expiry ({time_remaining_min:.1f}min) — hold to resolution"
            )

        return ExitSignal(
            action="HOLD",
            current_value=current_value,
            entry_price=entry_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            reason=f"No exit trigger (PnL={unrealized_pnl_pct:+.0%}, T={time_remaining_min:.1f}min)"
        )

    # ------------------------------------------------------------------
    # Greeks-based timing advice
    # ------------------------------------------------------------------

    def optimal_entry_window(
        self, btc_spot: float, btc_strike: float, total_market_minutes: float = 15.0,
    ) -> dict:
        """Analyze Greeks across market lifetime for optimal entry timing."""
        vol_est = self.vol_est.get_vol(method=self.vol_method)
        vol = vol_est.annualized_vol

        analysis = []
        for minutes_left in [14, 12, 10, 8, 6, 4, 2, 1, 0.5]:
            model = self.pricer.price(
                spot=btc_spot, strike=btc_strike,
                vol=vol, time_remaining_min=minutes_left,
            )
            analysis.append({
                "T_min": minutes_left,
                "yes_fv": model.yes_fair_value,
                "delta": model.delta,
                "gamma": model.gamma,
                "theta_per_min": model.theta,
            })

        best_window = max(
            [a for a in analysis if a["T_min"] > 1.0],
            key=lambda x: abs(x["gamma"]) / (abs(x["theta_per_min"]) + 1e-10),
            default=analysis[0],
        )

        return {
            "recommended_entry_T": best_window["T_min"],
            "profile": analysis,
            "vol_used": vol,
        }

    # ------------------------------------------------------------------
    # V2: Update bankroll (for Kelly tracking)
    # ------------------------------------------------------------------

    def update_bankroll(self, new_bankroll: float):
        """Update bankroll for Kelly sizing (call after each trade settles)."""
        old = self.bankroll
        self.bankroll = max(1.0, new_bankroll)
        logger.info(f"Bankroll updated: ${old:.2f} → ${self.bankroll:.2f}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "total_detections": self._total_detections,
            "tradeable_detections": self._tradeable_detections,
            "hit_rate": self._tradeable_detections / max(self._total_detections, 1),
            "bankroll": self.bankroll,
            "vol_estimator": self.vol_est.get_stats(),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_detector_instance = None

def get_mispricing_detector(**kwargs) -> MispricingDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MispricingDetector(**kwargs)
    return _detector_instance
