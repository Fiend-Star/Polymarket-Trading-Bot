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
    true_prob: float, market_price: float,
    use_half_kelly: bool = True, max_fraction: float = 0.05,
) -> float:
    """Kelly-optimal fraction of bankroll to bet on a binary option.
    f* = (p - q) / (1 - q).  Half Kelly captures ~75% of growth with ~50% less drawdown."""
    p = max(0.01, min(0.99, true_prob))
    q = max(0.01, min(0.99, market_price))
    if p <= q:
        return 0.0
    f_star = (p - q) / (1 - q)
    if use_half_kelly:
        f_star *= 0.5
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
        self, pricer=None, vol_estimator=None,
        maker_fee=0.00, taker_fee=0.02, min_edge_cents=0.02,
        min_edge_after_fees=0.005, take_profit_pct=0.30, cut_loss_pct=-0.50,
        vol_method="ewma", bankroll=50.0, use_half_kelly=True,
        max_kelly_fraction=0.05,
    ):
        self.pricer = pricer or get_binary_pricer()
        self.vol_est = vol_estimator or get_vol_estimator()
        self.maker_fee = maker_fee
        self.taker_fee_default = taker_fee
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
        logger.info(f"MispricingDetector V2: fee={maker_fee:.0%}, min_edge=${min_edge_cents:.2f}, "
                    f"bankroll=${bankroll:.0f}, kelly={'half' if use_half_kelly else 'full'}")

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def _compute_metrics(self, yes_mp, no_mp, btc_spot, btc_strike, trm, ps_usd, use_maker, vol_skew, fb):
        """Compute all intermediate metrics for detection."""
        ve, rv, vc, jp, rs = self._get_vol_and_jumps()
        model = self._compute_model_price(btc_spot, btc_strike, rv, trm, jp, rs, vol_skew, fb)
        iv, vs, vrp, vrp_sig = self._compute_iv_and_vrp(yes_mp, btc_spot, btc_strike, trm, rv)
        d, pe, tp, mp = self._determine_direction(model, yes_mp, no_mp)
        ae = abs(pe)
        fr, fc, ev, nev = self._compute_fees_and_ev(use_maker, tp, ae, ps_usd)
        kf, kb = self._compute_kelly(mp, tp)
        tradeable = self._evaluate_tradeability(ae, fr, nev, trm, vc, kf, model, pe)
        vrp_b = self._compute_vrp_boost(vrp_sig, d, yes_mp, no_mp)
        conf = self._compute_confidence(vc, ae, trm, kf, vrp_b)
        return (ve, rv, vc, jp, rs, model, iv, vs, vrp, vrp_sig,
                d, pe, ae, tp, mp, fc, ev, nev, tradeable, kf, kb, conf)

    def detect(
        self, yes_market_price: float, no_market_price: float,
        btc_spot: float, btc_strike: float, time_remaining_min: float,
        position_size_usd: float = 1.0, use_maker: bool = True,
        vol_skew: Optional[float] = None, funding_bias: float = 0.0,
    ) -> MispricingSignal:
        """Detect mispricing between model and market."""
        self._total_detections += 1
        (ve, rv, vc, jp, rs, model, iv, vs, vrp, vrp_sig,
         direction, pe, ae, tp, mp, fc, ev, nev, tradeable, kf, kb, conf
        ) = self._compute_metrics(yes_market_price, no_market_price, btc_spot,
                                   btc_strike, time_remaining_min, position_size_usd,
                                   use_maker, vol_skew, funding_bias)
        if not tradeable:
            direction, kf, kb = "NO_TRADE", 0.0, 0.0
        else:
            self._tradeable_detections += 1
        ep = ae / tp if tp > 0 else 0.0
        signal = self._build_signal(pe, ep, direction, yes_market_price, no_market_price,
                                    model, btc_spot, btc_strike, rv, time_remaining_min,
                                    conf, vc, iv, vs, vrp, vrp_sig, ev, fc, nev, tradeable, kf, kb)
        self._log_detection(signal, ve, jp, rs, model)
        return signal

    def _build_signal(self, edge, edge_pct, direction, yes_mkt, no_mkt,
                      model, spot, strike, vol, trm, conf, vol_conf,
                      iv, vs, vrp, vrp_sig, ev, fee, net_ev, tradeable, kf, kb):
        """Construct MispricingSignal from computed values."""
        return MispricingSignal(
            edge=edge, edge_pct=edge_pct, direction=direction,
            yes_market=yes_mkt, no_market=no_mkt,
            yes_model=model.yes_fair_value, no_model=model.no_fair_value,
            spot=spot, strike=strike, vol=vol, time_remaining_min=trm,
            delta=model.delta, gamma=model.gamma, theta_per_min=model.theta,
            confidence=conf, vol_confidence=vol_conf,
            implied_vol=iv, realized_vol=vol, vol_spread=vs,
            vrp=vrp, vrp_signal=vrp_sig,
            expected_pnl=ev, fee_cost=fee,
            net_expected_pnl=net_ev, is_tradeable=tradeable,
            kelly_fraction=kf, kelly_bet_usd=kb,
            pricing_method=model.method)

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _get_vol_and_jumps(self):
        """Get volatility estimate and jump parameters from vol estimator."""
        vol_estimate = self.vol_est.get_vol(method=self.vol_method)
        realized_vol = vol_estimate.annualized_vol
        vol_confidence = vol_estimate.confidence
        jump_params = self.vol_est.get_jump_params()
        return_stats = self.vol_est.get_return_stats()
        return vol_estimate, realized_vol, vol_confidence, jump_params, return_stats

    def _compute_model_price(self, btc_spot, btc_strike, realized_vol,
                             time_remaining_min, jump_params, return_stats,
                             vol_skew, funding_bias):
        """Price with jump-diffusion + overlays."""
        return self.pricer.price(
            spot=btc_spot, strike=btc_strike, vol=realized_vol,
            time_remaining_min=time_remaining_min,
            jump_intensity=jump_params.intensity,
            jump_mean=jump_params.mean, jump_vol=jump_params.vol,
            recent_return=return_stats.recent_return,
            recent_return_sigma=return_stats.recent_return_sigma,
            apply_overlays=True,
            vol_skew=vol_skew, funding_bias=funding_bias,
        )

    def _compute_iv_and_vrp(self, yes_market_price, btc_spot, btc_strike,
                            time_remaining_min, realized_vol):
        """Compute implied vol, vol spread, and VRP."""
        try:
            iv = self.pricer.implied_vol(
                market_price=yes_market_price, spot=btc_spot,
                strike=btc_strike, time_remaining_min=time_remaining_min,
                is_call=True,
            )
        except Exception:
            iv = realized_vol

        vol_spread = iv - realized_vol
        self.vol_est.record_iv(iv)
        self.vol_est.record_rv(realized_vol)
        vrp = self.vol_est.get_vrp()

        if vrp > 0.20:
            vrp_signal = "FADE_EXTREME"
        elif vrp < -0.05:
            vrp_signal = "VOL_CHEAP"
        else:
            vrp_signal = "NEUTRAL"

        return iv, vol_spread, vrp, vrp_signal

    def _determine_direction(self, model, yes_market_price, no_market_price):
        """Pick the side with more edge and determine trade direction."""
        yes_edge = model.yes_fair_value - yes_market_price
        no_edge = model.no_fair_value - no_market_price

        if abs(yes_edge) >= abs(no_edge):
            if yes_edge > 0:
                return "BUY_YES", yes_edge, yes_market_price, model.yes_fair_value
            else:
                return "BUY_NO", no_edge, no_market_price, model.no_fair_value
        else:
            if no_edge > 0:
                return "BUY_NO", no_edge, no_market_price, model.no_fair_value
            else:
                return "BUY_YES", yes_edge, yes_market_price, model.yes_fair_value

    def _compute_fees_and_ev(self, use_maker, trade_price, abs_edge, position_size_usd):
        """Compute fee rate, fee cost, and expected PnL."""
        fee_rate = self.maker_fee if use_maker else polymarket_taker_fee(trade_price)
        num_tokens = position_size_usd / trade_price if trade_price > 0 else 0
        expected_pnl = abs_edge * num_tokens
        fee_cost = fee_rate * num_tokens * trade_price
        net_expected_pnl = expected_pnl - fee_cost
        return fee_rate, fee_cost, expected_pnl, net_expected_pnl

    def _compute_kelly(self, model_prob, trade_price):
        """Compute Kelly fraction and bet size."""
        kf = kelly_fraction(
            true_prob=model_prob, market_price=trade_price,
            use_half_kelly=self.use_half_kelly,
            max_fraction=self.max_kelly_fraction,
        )
        return kf, kf * self.bankroll

    def _evaluate_tradeability(self, abs_edge, fee_rate, net_expected_pnl,
                               time_remaining_min, vol_confidence, kf, model,
                               primary_edge):
        """Determine if the detected edge is tradeable."""
        min_edge = max(self.min_edge_cents, fee_rate * 2.0)

        is_tradeable = (
            abs_edge >= min_edge
            and net_expected_pnl >= self.min_edge_after_fees
            and time_remaining_min >= 1.0
            and vol_confidence >= 0.2
            and kf > 0.001
        )

        # Theta check
        if time_remaining_min < 3.0 and abs(model.theta) * time_remaining_min > abs_edge * 0.5:
            is_tradeable = False

        return is_tradeable

    def _compute_vrp_boost(self, vrp_signal, direction, yes_market_price, no_market_price):
        """Compute VRP-based confidence boost."""
        if vrp_signal == "FADE_EXTREME":
            if (direction == "BUY_NO" and yes_market_price > 0.70) or \
               (direction == "BUY_YES" and no_market_price > 0.70):
                return 0.10
        return 0.0

    def _compute_confidence(self, vol_confidence, abs_edge, time_remaining_min,
                            kf, vrp_boost):
        """Compute overall confidence score."""
        edge_conf = min(1.0, abs_edge / 0.10)
        time_conf = min(1.0, time_remaining_min / 5.0)
        return (
            vol_confidence * 0.3
            + edge_conf * 0.3
            + time_conf * 0.2
            + (0.1 if kf > 0.01 else 0.0)
            + vrp_boost
        )

    def _log_detection(self, signal, vol_estimate, jump_params, return_stats, model):
        """Log the detection analysis."""
        logger.info("=" * 70)
        logger.info("BINARY OPTION MISPRICING ANALYSIS — V2")
        logger.info(f"  BTC: spot=${signal.spot:,.2f}, strike=${signal.strike:,.2f}, T={signal.time_remaining_min:.1f}min")
        logger.info(f"  Vol: RV={signal.realized_vol:.1%} ({vol_estimate.method}), IV={signal.implied_vol:.1%}, spread={signal.vol_spread:+.1%}")
        logger.info(f"  VRP: {signal.vrp:+.1%} → {signal.vrp_signal}")
        logger.info(f"  Jumps: λ={jump_params.intensity:.0f}/yr, recent={jump_params.recent_jump}")
        logger.info(f"  Returns: ret={return_stats.recent_return:+.3%}, σ={return_stats.recent_return_sigma:+.2f}")
        logger.info(f"  Model: YES=${model.yes_fair_value:.4f}, NO=${model.no_fair_value:.4f} [{model.method}]")
        logger.info(f"    BSM base={model.bsm_base:.4f}, JD adj={model.jump_adjustment:+.4f}")
        logger.info(f"    MR adj={model.mean_reversion_adj:+.4f}, candle={model.candle_effect_adj:+.4f}, season={model.seasonality_adj:+.4f}")
        logger.info(f"  Market: YES=${signal.yes_market:.4f}, NO=${signal.no_market:.4f}")
        logger.info(f"  Edge: {signal.edge:+.4f} ({signal.edge_pct:+.1%})")
        logger.info(f"  Fee: cost=${signal.fee_cost:.4f}, Net EV: ${signal.net_expected_pnl:+.4f}")
        logger.info(f"  Kelly: f*={signal.kelly_fraction:.2%}, bet=${signal.kelly_bet_usd:.2f} of ${self.bankroll:.0f} bankroll")
        logger.info(f"  Greeks: Δ={model.delta:.6f}, Γ={model.gamma:.6f}, Θ={model.theta:.6f}/min")
        logger.info(f"  → {signal.direction} {'✓ TRADEABLE' if signal.is_tradeable else '✗ NO TRADE'} (conf={signal.confidence:.0%})")
        logger.info("=" * 70)


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
        current_value, unrealized_pnl, unrealized_pnl_pct = self._compute_exit_value(
            entry_price, direction, btc_spot, btc_strike,
            time_remaining_min, yes_market_price, no_market_price,
        )
        return self._build_exit_signal(
            entry_price, current_value, unrealized_pnl,
            unrealized_pnl_pct, time_remaining_min,
        )

    def _compute_exit_value(self, entry_price, direction, btc_spot, btc_strike,
                            time_remaining_min, yes_market_price, no_market_price):
        """Compute current value and unrealized PnL for an open position."""
        vol_estimate = self.vol_est.get_vol(method=self.vol_method)
        model = self.pricer.price(
            spot=btc_spot, strike=btc_strike,
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
        return current_value, unrealized_pnl, unrealized_pnl_pct

    def _build_exit_signal(self, entry_price, current_value, unrealized_pnl,
                           unrealized_pnl_pct, time_remaining_min):
        """Build an ExitSignal based on PnL thresholds."""
        if unrealized_pnl_pct >= self.take_profit_pct:
            return ExitSignal(
                action="TAKE_PROFIT", current_value=current_value,
                entry_price=entry_price, unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                reason=f"Profit target hit: {unrealized_pnl_pct:.0%} >= {self.take_profit_pct:.0%}",
            )
        if unrealized_pnl_pct <= self.cut_loss_pct and time_remaining_min > 2.0:
            return ExitSignal(
                action="CUT_LOSS", current_value=current_value,
                entry_price=entry_price, unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                reason=f"Stop loss hit: {unrealized_pnl_pct:.0%} <= {self.cut_loss_pct:.0%}",
            )
        return ExitSignal(
            action="HOLD", current_value=current_value,
            entry_price=entry_price, unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            reason=f"No exit trigger (PnL={unrealized_pnl_pct:+.0%}, T={time_remaining_min:.1f}min)",
        )

    # ------------------------------------------------------------------
    # Greeks-based timing advice
    # ------------------------------------------------------------------

    def optimal_entry_window(self, btc_spot: float, btc_strike: float,
                             total_market_minutes: float = 15.0) -> dict:
        """Analyze Greeks across market lifetime for optimal entry timing."""
        vol = self.vol_est.get_vol(method=self.vol_method).annualized_vol
        analysis = []
        for ml in [14, 12, 10, 8, 6, 4, 2, 1, 0.5]:
            m = self.pricer.price(spot=btc_spot, strike=btc_strike, vol=vol, time_remaining_min=ml)
            analysis.append({"T_min": ml, "yes_fv": m.yes_fair_value,
                           "delta": m.delta, "gamma": m.gamma, "theta_per_min": m.theta})
        best = max([a for a in analysis if a["T_min"] > 1.0],
                   key=lambda x: abs(x["gamma"]) / (abs(x["theta_per_min"]) + 1e-10),
                   default=analysis[0])
        return {"recommended_entry_T": best["T_min"], "profile": analysis, "vol_used": vol}

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
