"""
Binary Option Pricer for Polymarket 15-Minute BTC Markets — V2

RESEARCH-DRIVEN UPGRADES over V1 (pure Black-Scholes):
  1. Merton jump-diffusion pricing (20-40% error reduction over BSM)
     - BTC 15-min returns have kurtosis 15-100; BSM assumes 3
     - Liquidation cascades cause discontinuous jumps BSM can't model
     - Compound Poisson jump process captures fat tails

  2. Mean reversion overlay
     - BTC has negative first-order autocorrelation at 15-min
     - After >1.5σ move, reversal probability ~55-60%
     - Asymmetric: negative returns revert faster (3.66% asymmetry)

  3. Turn-of-candle effect
     - Positive returns of +0.58 bps/min at minutes 0, 15, 30, 45
     - t-statistics above 9 across all exchanges (Shanaev & Vasenin 2023)
     - Other minutes have slightly negative average returns

  4. Intraday seasonality
     - 22:00-23:00 UTC strongest (post-NYSE close)
     - 03:00-04:00 UTC weakest
     - Friday strongest day

Maps Polymarket YES/NO tokens to cash-or-nothing binary options:
  - YES token = Binary Call (pays $1 if BTC finishes above reference price)
  - NO token  = Binary Put  (pays $1 if BTC finishes below reference price)
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Normal distribution helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using Abramowitz & Stegun approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BinaryOptionPrice:
    """Complete pricing output for a binary option."""
    # Fair values
    yes_fair_value: float       # Binary call price (0 to 1)
    no_fair_value: float        # Binary put price (0 to 1)

    # Greeks (for the YES/call side)
    delta: float                # dP/dS — sensitivity to spot
    gamma: float                # d²P/dS² — convexity
    theta: float                # dP/dT — time decay (per minute)
    vega: float                 # dP/dσ — vol sensitivity

    # Inputs (for logging/debugging)
    spot: float
    strike: float
    vol: float                  # Annualized
    time_remaining_min: float
    d2: float

    # Derived
    moneyness: float            # S/K ratio
    implied_prob: float         # Model-implied probability of YES outcome

    # V2 fields: pricing method + adjustment components
    method: str = "bsm"        # "bsm", "merton_jd", "adjusted"
    bsm_base: float = 0.0
    jump_adjustment: float = 0.0
    mean_reversion_adj: float = 0.0
    candle_effect_adj: float = 0.0
    seasonality_adj: float = 0.0
    skew_adj: float = 0.0

    def __repr__(self):
        return (
            f"BinaryPrice(YES=${self.yes_fair_value:.4f}, NO=${self.no_fair_value:.4f}, "
            f"Δ={self.delta:.4f}, Γ={self.gamma:.4f}, Θ={self.theta:.6f}/min, "
            f"spot={self.spot:.2f}, strike={self.strike:.2f}, "
            f"vol={self.vol:.1%}, T={self.time_remaining_min:.1f}min, "
            f"method={self.method})"
        )


# ---------------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------------

class BinaryOptionPricer:
    """
    Prices Polymarket YES/NO tokens using jump-diffusion + statistical overlays.

    Pricing hierarchy (V2):
      1. Merton jump-diffusion → base probability (handles fat tails + jumps)
      2. Mean reversion overlay → shift prob based on recent price action
      3. Turn-of-candle effect → shift based on minute-of-hour
      4. Intraday seasonality → shift based on hour-of-day

    Usage:
        pricer = BinaryOptionPricer()
        result = pricer.price(
            spot=66200.0,
            strike=66000.0,
            vol=0.65,
            time_remaining_min=12.0,
            recent_return=0.003,
            jump_intensity=1500.0,
            jump_mean=-0.001,
            jump_vol=0.004,
        )
    """

    MINUTES_PER_YEAR = 365.25 * 24 * 60  # 525,960

    # --- Turn-of-candle effect (Shanaev & Vasenin 2023) ---
    CANDLE_MINUTES = {0, 15, 30, 45}

    # --- Intraday seasonality (QuantPedia / SSRN 4581124) ---
    # Hour-of-day bullish/bearish bias in BTC returns (UTC)
    HOURLY_BIAS = {
        0: 0.10,  1: 0.05,  2: 0.02,  3: -0.15,
        4: -0.12, 5: -0.05, 6: 0.00,  7: 0.02,
        8: 0.03,  9: 0.05, 10: 0.04, 11: 0.02,
        12: 0.00, 13: -0.02, 14: 0.03, 15: 0.05,
        16: 0.04, 17: 0.02, 18: -0.03, 19: -0.05,
        20: 0.00, 21: 0.08, 22: 0.15, 23: 0.12,
    }
    SEASONALITY_SCALE = 0.001  # Convert bias units → probability shift

    # --- Mean reversion ---
    MEAN_REVERSION_THRESHOLD_SIGMA = 1.5
    MEAN_REVERSION_STRENGTH = 0.06       # Max ~6% probability shift
    NEGATIVE_REVERSION_MULT = 1.25       # Negative returns revert 25% faster

    def __init__(self, risk_free_rate: float = 0.0):
        self.r = risk_free_rate
        logger.info(f"Initialized BinaryOptionPricer V2 (jump-diffusion + overlays)")

    # ==================================================================
    # Main entry point
    # ==================================================================

    def price(
        self,
        spot: float,
        strike: float,
        vol: float,
        time_remaining_min: float,
        # V2: Jump-diffusion parameters (from VolEstimator)
        jump_intensity: float = 1500.0,
        jump_mean: float = -0.0008,
        jump_vol: float = 0.003,
        # V2: Mean reversion inputs
        recent_return: Optional[float] = None,
        recent_return_sigma: Optional[float] = None,
        # V2: Overlay toggle
        apply_overlays: bool = True,
        # V3: Skew-adjusted binary correction (dσ/dK)
        vol_skew: Optional[float] = None,
        # V3: Funding rate regime bias
        funding_bias: float = 0.0,
        # V3.4: Exact market end time
        market_end_time: Optional[datetime] = None,
    ) -> BinaryOptionPrice:
        """
        Price a binary call (YES) and put (NO) using jump-diffusion + overlays.
        """
        if spot <= 0 or strike <= 0:
            return self._intrinsic_price(spot, strike)

        vol = max(vol, 0.0)
        T = max(time_remaining_min / self.MINUTES_PER_YEAR, 0.0)

        if T < 1e-12 or time_remaining_min < 0.05:
            return self._intrinsic_price(spot, strike)
        if vol < 1e-8:
            return self._intrinsic_price(spot, strike)

        # --- Step 1: Merton Jump-Diffusion base probability ---
        yes_jd, d2_jd = self._merton_jump_diffusion(
            spot, strike, vol, T, jump_intensity, jump_mean, jump_vol
        )

        # --- Step 2: BSM for Greeks + comparison ---
        yes_bsm, d2_bsm, greeks = self._bsm_with_greeks(spot, strike, vol, T)

        yes_base = yes_jd
        jump_adj = yes_jd - yes_bsm
        method = "merton_jd"
        mr_adj = 0.0
        candle_adj = 0.0
        season_adj = 0.0
        skew_adj_val = 0.0

        # --- Step 3: Statistical overlays ---
        if recent_return is not None and recent_return_sigma is not None:
            mr_adj = self._mean_reversion_adj(
                recent_return, recent_return_sigma, time_remaining_min
            )
        candle_adj = self._candle_effect_adj(time_remaining_min, market_end_time)
        season_adj = self._seasonality_adj()

        # V3: Skew-adjusted binary correction
        # Binary_adjusted = N(d2) - Vega × dσ/dK
        # BTC has persistent negative vol skew (OTM puts > OTM calls)
        if vol_skew is not None and greeks["vega"] != 0:
            skew_adj_val = -greeks["vega"] * vol_skew
            # Clamp to reasonable range
            skew_adj_val = max(-0.03, min(0.03, skew_adj_val))

        method = "adjusted"

        # --- Step 4: Funding rate regime bias ---
        # From FundingRateFilter: crowded longs → negative bias, crowded shorts → positive
        funding_adj = max(-0.02, min(0.02, funding_bias))

        # --- Combine ---
        yes_fv = max(0.01, min(0.99,
            yes_base + mr_adj + candle_adj + season_adj + skew_adj_val + funding_adj
        ))
        no_fv = max(0.01, min(0.99, 1.0 - yes_fv))

        return BinaryOptionPrice(
            yes_fair_value=yes_fv,
            no_fair_value=no_fv,
            delta=greeks["delta"],
            gamma=greeks["gamma"],
            theta=greeks["theta_per_min"],
            vega=greeks["vega"],
            spot=spot,
            strike=strike,
            vol=vol,
            time_remaining_min=time_remaining_min,
            d2=d2_jd,
            moneyness=spot / strike,
            implied_prob=yes_fv,
            method=method,
            bsm_base=yes_bsm,
            jump_adjustment=jump_adj,
            mean_reversion_adj=mr_adj,
            candle_effect_adj=candle_adj,
            seasonality_adj=season_adj,
            skew_adj=skew_adj_val,
        )

    # ==================================================================
    # Merton Jump-Diffusion binary pricing
    # ==================================================================

    def _merton_jump_diffusion(
        self,
        spot: float,
        strike: float,
        vol: float,
        T: float,
        jump_intensity: float,
        jump_mean: float,
        jump_vol: float,
        num_terms: int = 25,
    ) -> tuple:
        """
        Merton jump-diffusion binary option pricing.

        Physical measure:
          P(S_T > K) = Σ_{n=0}^{N} Poisson(λT, n) × Φ(d_n)

        where:
          d_n = [ln(S/K) + (-σ²/2 - λm̄)T + nμ_J] / sqrt(σ²T + nσ_J²)
          m̄ = E[e^J - 1] = exp(μ_J + σ_J²/2) - 1

        Returns (yes_probability, d2_weighted)
        """
        lambda_T = jump_intensity * T
        m_bar = math.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0

        prob_up = 0.0
        poisson_cumul = 0.0
        d2_weighted = 0.0

        # Pre-compute log(S/K) once
        log_moneyness = math.log(spot / strike)
        base_drift_T = (-0.5 * vol ** 2 - jump_intensity * m_bar) * T
        base_var_T = vol ** 2 * T

        for n in range(num_terms):
            # Poisson weight P(N_T = n)
            if lambda_T < 1e-15:
                pw = 1.0 if n == 0 else 0.0
            else:
                log_pw = -lambda_T + n * math.log(lambda_T) - math.lgamma(n + 1)
                pw = math.exp(log_pw)

            if pw < 1e-15 and poisson_cumul > 0.9999:
                break

            poisson_cumul += pw

            # Conditional variance and drift given n jumps
            total_var = base_var_T + n * jump_vol ** 2
            drift = base_drift_T + n * jump_mean

            if total_var < 1e-20:
                d_n = 1e6 if spot > strike else (-1e6 if spot < strike else 0.0)
            else:
                d_n = (log_moneyness + drift) / math.sqrt(total_var)

            cond_prob = _norm_cdf(d_n)
            prob_up += pw * cond_prob
            d2_weighted += pw * d_n

        # Normalize if truncated early
        if 0 < poisson_cumul < 0.999:
            prob_up /= poisson_cumul
            d2_weighted /= poisson_cumul

        return prob_up, d2_weighted

    # ==================================================================
    # BSM with Greeks
    # ==================================================================

    def _bsm_with_greeks(
        self, spot: float, strike: float, vol: float, T: float
    ) -> tuple:
        """Standard BSM binary pricing + Greeks. Returns (yes_prob, d2, greeks)."""
        sqrt_T = math.sqrt(T)
        vol_sqrt_T = vol * sqrt_T

        d2 = (math.log(spot / strike) + (self.r - 0.5 * vol * vol) * T) / vol_sqrt_T
        d1 = d2 + vol_sqrt_T
        discount = math.exp(-self.r * T)
        yes_fv = discount * _norm_cdf(d2)
        n_d2 = _norm_pdf(d2)

        delta = discount * n_d2 / (spot * vol_sqrt_T) if vol_sqrt_T > 0 else 0.0

        denom = spot * spot * vol * vol * T
        gamma = -discount * n_d2 * d1 / denom if denom > 0 else 0.0

        theta_yr = discount * n_d2 * d1 / (2.0 * T) if T > 0 else 0.0
        theta_min = theta_yr / self.MINUTES_PER_YEAR

        vega = -discount * n_d2 * d1 * sqrt_T / vol if vol > 0 else 0.0

        return yes_fv, d2, {
            "delta": delta, "gamma": gamma,
            "theta_per_min": theta_min, "vega": vega,
        }

    # ==================================================================
    # Statistical overlays
    # ==================================================================

    def _mean_reversion_adj(
        self, recent_return: float, recent_sigma: float, t_min: float
    ) -> float:
        """
        Mean reversion overlay. BTC has negative autocorrelation at 15-min.
        After >1.5σ move, probability of reversal modestly elevated (~55-60%).
        Asymmetric: negative returns revert faster (3.66% asymmetry).
        """
        abs_sig = abs(recent_sigma)
        if abs_sig < self.MEAN_REVERSION_THRESHOLD_SIGMA:
            return 0.0

        excess = abs_sig - self.MEAN_REVERSION_THRESHOLD_SIGMA
        raw = min(excess / 2.0, 1.0) * self.MEAN_REVERSION_STRENGTH
        time_factor = min(t_min / 10.0, 1.0)
        strength = raw * time_factor

        if recent_return > 0:
            # Up move → expect reversion DOWN → lower YES prob
            return -strength
        else:
            # Down move → expect reversion UP → raise YES prob
            return strength * self.NEGATIVE_REVERSION_MULT

    def _candle_effect_adj(self, time_remaining_min: float, market_end_time: Optional[datetime] = None) -> float:
        """
        Turn-of-candle effect (Shanaev & Vasenin 2023).
        +0.58 bps/min concentrated at minutes 0, 15, 30, 45 of each hour.
        """
        # V3.4 FIX: Use exact market_end_time instead of volatile float math
        if market_end_time is not None:
            resolve_minute = market_end_time.minute
        else:
            now = datetime.now(timezone.utc)
            resolve_minute = (now.minute + round(time_remaining_min)) % 60

        if resolve_minute in self.CANDLE_MINUTES:
            return 0.002   # +0.2% bullish bias at candle boundaries
        elif resolve_minute % 15 <= 1 or resolve_minute % 15 >= 14:
            return 0.001   # Partial effect near boundaries
        else:
            return -0.0005  # Slight negative at non-boundary minutes

    def _seasonality_adj(self) -> float:
        """
        Intraday seasonality (QuantPedia / SSRN 4581124).
        22:00-23:00 UTC strongest positive, 03:00-04:00 weakest.
        """
        now = datetime.now(timezone.utc)
        bias = self.HOURLY_BIAS.get(now.hour, 0.0)

        # Friday amplification
        if now.weekday() == 4:
            bias *= 1.2

        return bias * self.SEASONALITY_SCALE

    # ==================================================================
    # V3: Vol skew estimation
    # ==================================================================

    @staticmethod
    def estimate_btc_vol_skew(
        spot: float,
        strike: float,
        vol: float,
        time_remaining_min: float,
    ) -> float:
        """
        Estimate dσ/dK (vol skew slope) for BTC at short horizons.

        BTC has a persistent negative vol skew:
        - OTM puts are more expensive than OTM calls (crash risk premium)
        - The skew steepens at shorter horizons (0DTE effect)
        - Research: BTC skew ~-0.1 to -0.3 per 1% moneyness at daily horizons

        For 15-min binary options, the relevant metric is how vol changes
        as strike moves relative to spot. A higher strike (OTM call) should
        have lower IV; a lower strike (OTM put) should have higher IV.

        Returns dσ/dK (change in vol per unit change in strike price).
        Negative = typical BTC skew (OTM puts more expensive).
        """
        if spot <= 0 or strike <= 0 or vol <= 0:
            return 0.0

        moneyness = spot / strike  # >1 means ITM call, <1 means OTM call

        # Base skew slope: -0.15 per 1% moneyness at daily horizon
        # Steeper at shorter horizons (scale by sqrt(T_daily / T_actual))
        T_daily = 1.0 / (365.25 * 24 * 60)  # 1 day in years
        T_actual = max(time_remaining_min, 0.1) / (365.25 * 24 * 60)
        time_scale = min(3.0, math.sqrt(T_daily / T_actual))

        # BTC base skew: approximately -2.0 per unit moneyness at daily
        # This means for a 1% OTM call (moneyness 0.99), IV drops ~2%
        base_skew_per_strike = -2.0 * vol / spot  # Convert to per-dollar

        # Scale by time horizon
        skew = base_skew_per_strike * time_scale

        # Clamp to prevent extreme adjustments
        return max(-0.001, min(0.001, skew))

    # ==================================================================
    # Intrinsic price (edge case)
    # ==================================================================

    def _intrinsic_price(self, spot: float, strike: float) -> BinaryOptionPrice:
        if spot > strike:
            yes_fv, no_fv = 0.99, 0.01
        elif spot < strike:
            yes_fv, no_fv = 0.01, 0.99
        else:
            yes_fv, no_fv = 0.50, 0.50

        return BinaryOptionPrice(
            yes_fair_value=yes_fv, no_fair_value=no_fv,
            delta=0.0, gamma=0.0, theta=0.0, vega=0.0,
            spot=spot, strike=strike, vol=0.0,
            time_remaining_min=0.0, d2=0.0,
            moneyness=spot / strike if strike > 0 else 1.0,
            implied_prob=yes_fv, method="intrinsic",
        )

    # ==================================================================
    # Implied vol (bisection on BSM component — standard practice)
    # ==================================================================

    def implied_vol(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_remaining_min: float,
        is_call: bool = True,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> Optional[float]:
        """
        Back out implied volatility from market price using bisection.
        Uses BSM component for IV — the gap between BSM-IV and realized vol
        IS the Variance Risk Premium signal (~14% annualized for BTC).

        Returns None if bisection fails to converge — this is a strong signal
        that spot/strike inputs are wrong (e.g. ATM where N(d2)≈0.5 for all vol).
        """
        if time_remaining_min < 0.1:
            return 0.0

        T = max(time_remaining_min / self.MINUTES_PER_YEAR, 1e-12)
        # INCREASE BOUNDS: 15-minute intraday vol can easily exceed 1000% annualized
        # during liquidation wicks. Capping at 2.0 (200%) causes false failures.
        vol_low, vol_high = 0.01, 15.0  # Raised from 2.0 to 15.0 (1500% vol)
        sqrt_T = math.sqrt(T)

        def _bsm_price(vol: float) -> float:
            vst = vol * sqrt_T
            if vst < 1e-12:
                # At near-zero vol: price is 1 if ITM, 0 if OTM, 0.5 if ATM
                if spot > strike:
                    return 1.0 if is_call else 0.0
                elif spot < strike:
                    return 0.0 if is_call else 1.0
                else:
                    return 0.5
            d2 = (math.log(spot / strike) + (self.r - 0.5 * vol ** 2) * T) / vst
            return _norm_cdf(d2) if is_call else _norm_cdf(-d2)

        # Determine monotonicity: binary option price is NOT always monotonic
        # in vol. For ITM calls (spot>strike), higher vol → lower price.
        # For OTM calls (spot<strike), higher vol → higher price (up to a point).
        p_low = _bsm_price(vol_low)
        p_high = _bsm_price(vol_high)

        # Check if the target is achievable within [vol_low, vol_high]
        p_min = min(p_low, p_high)
        p_max = max(p_low, p_high)
        if market_price < p_min - 0.05 or market_price > p_max + 0.05:
            logger.warning(
                f"IV bisection: target={market_price:.4f} outside achievable range "
                f"[{p_min:.4f}, {p_max:.4f}] — likely strike mismatch"
            )
            return None

        # price_decreasing: True if higher vol → lower price (ITM case)
        price_decreasing = (p_low > p_high)

        for _ in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2.0
            p = _bsm_price(vol_mid)

            if abs(p - market_price) < tolerance:
                # V3.2: Boundary guard — if bisection "converged" to the upper
                # bound, it's a numerical failure (no finite vol explains the
                # price), not a genuine IV.
                if vol_mid > vol_high * 0.99:
                    logger.warning(
                        f"IV bisection hit upper bound ({vol_high:.0%}): "
                        f"vol={vol_mid:.4f}, spot={spot:.2f}, strike={strike:.2f}, "
                        f"T={time_remaining_min:.1f}min — marking untradeable"
                    )
                    return None
                return vol_mid

            if price_decreasing:
                # Higher vol → lower price: if p > target, need more vol
                if p > market_price:
                    vol_low = vol_mid
                else:
                    vol_high = vol_mid
            else:
                # Higher vol → higher price: if p > target, need less vol
                if p > market_price:
                    vol_high = vol_mid
                else:
                    vol_low = vol_mid

        # Convergence check: verify the final midpoint actually prices close
        # to the target. For ATM binaries (spot≈strike), N(d2)≈0.5 for ALL
        # vol values, so bisection converges to vol_low=vol_high but the
        # price residual is huge. This is a smoke signal: if no vol can
        # explain the market price at these inputs, the inputs are wrong.
        final_vol = (vol_low + vol_high) / 2.0
        final_p = _bsm_price(final_vol)
        if abs(final_p - market_price) > 0.05:
            logger.warning(
                f"IV bisection did NOT converge: target={market_price:.4f}, "
                f"best={final_p:.4f}, vol={final_vol:.4f}, "
                f"spot={spot:.2f}, strike={strike:.2f}, T={time_remaining_min:.1f}min "
                f"— likely strike mismatch"
            )
            return None

        # V3.2: Final boundary guard (for the case where loop exhausted iterations)
        if final_vol > vol_high * 0.99:
            logger.warning(
                f"IV bisection converged to upper bound ({vol_high:.0%}): "
                f"vol={final_vol:.4f} — marking untradeable"
            )
            return None
        return final_vol

    # ==================================================================
    # BSM-only pricing (V1 backward compatible)
    # ==================================================================

    def price_bsm(
        self, spot: float, strike: float, vol: float, time_remaining_min: float,
    ) -> BinaryOptionPrice:
        """Pure BSM pricing for comparison/debugging."""
        if spot <= 0 or strike <= 0:
            return self._intrinsic_price(spot, strike)

        vol = max(vol, 0.0)
        T = max(time_remaining_min / self.MINUTES_PER_YEAR, 0.0)

        if T < 1e-12 or time_remaining_min < 0.05 or vol < 1e-8:
            return self._intrinsic_price(spot, strike)

        yes_fv, d2, greeks = self._bsm_with_greeks(spot, strike, vol, T)
        yes_fv = max(0.01, min(0.99, yes_fv))
        no_fv = max(0.01, min(0.99, 1.0 - yes_fv))

        return BinaryOptionPrice(
            yes_fair_value=yes_fv, no_fair_value=no_fv,
            delta=greeks["delta"], gamma=greeks["gamma"],
            theta=greeks["theta_per_min"], vega=greeks["vega"],
            spot=spot, strike=strike, vol=vol,
            time_remaining_min=time_remaining_min, d2=d2,
            moneyness=spot / strike, implied_prob=yes_fv, method="bsm",
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_pricer_instance = None

def get_binary_pricer() -> BinaryOptionPricer:
    global _pricer_instance
    if _pricer_instance is None:
        _pricer_instance = BinaryOptionPricer()
    return _pricer_instance