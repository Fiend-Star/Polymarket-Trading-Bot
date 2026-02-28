"""
Binary Option Pricer V2 — Merton jump-diffusion + statistical overlays.

Maps Polymarket YES/NO tokens to cash-or-nothing binary options:
  YES = Binary Call (pays $1 if BTC > strike), NO = Binary Put.

Upgrades over V1 (pure BSM):
  1. Merton JD pricing (fat tails + jumps)
  2. Mean reversion overlay (negative autocorrelation at 15-min)
  3. Turn-of-candle effect (Shanaev & Vasenin 2023)
  4. Intraday seasonality (QuantPedia / SSRN 4581124)
"""

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from loguru import logger
from binary_overlays import BinaryOverlaysMixin


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
        return (f"BinaryPrice(YES=${self.yes_fair_value:.4f}, NO=${self.no_fair_value:.4f}, "
                f"Δ={self.delta:.4f}, vol={self.vol:.1%}, T={self.time_remaining_min:.1f}min, "
                f"method={self.method})")


# ---------------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------------

class BinaryOptionPricer(BinaryOverlaysMixin):
    """
    Prices Polymarket YES/NO tokens using Merton JD + statistical overlays.

    Pricing hierarchy: JD base → mean reversion → candle effect → seasonality.
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

    def _validate_and_prepare(self, spot, strike, vol, time_remaining_min):
        """Validate inputs and compute T. Returns (T,) or None if intrinsic."""
        if spot <= 0 or strike <= 0:
            return None
        vol = max(vol, 0.0)
        T = max(time_remaining_min / self.MINUTES_PER_YEAR, 0.0)
        if T < 1e-12 or time_remaining_min < 0.05 or vol < 1e-8:
            return None
        return T, vol

    def price(self, spot: float, strike: float, vol: float, time_remaining_min: float,
              jump_intensity: float = 1500.0, jump_mean: float = -0.0008,
              jump_vol: float = 0.003, recent_return: Optional[float] = None,
              recent_return_sigma: Optional[float] = None, apply_overlays: bool = True,
              vol_skew: Optional[float] = None, funding_bias: float = 0.0) -> BinaryOptionPrice:
        """Price a binary call (YES) and put (NO) using jump-diffusion + overlays."""
        result = self._validate_and_prepare(spot, strike, vol, time_remaining_min)
        if result is None:
            return self._intrinsic_price(spot, strike)
        T, vol = result
        yes_jd, d2_jd = self._merton_jump_diffusion(spot, strike, vol, T, jump_intensity, jump_mean, jump_vol)
        yes_bsm, d2_bsm, greeks = self._bsm_with_greeks(spot, strike, vol, T)
        jump_adj = yes_jd - yes_bsm
        method = "merton_jd"
        mr_adj, candle_adj, season_adj, skew_adj_val, funding_adj = self._compute_overlays(
            apply_overlays, recent_return, recent_return_sigma, time_remaining_min, vol_skew, greeks, funding_bias)
        if apply_overlays: method = "adjusted"
        adjs = {"jump": jump_adj, "mr": mr_adj, "candle": candle_adj, "season": season_adj, "skew": skew_adj_val}
        total_adj = mr_adj + candle_adj + season_adj + skew_adj_val + funding_adj
        yes_fv = max(0.01, min(0.99, yes_jd + total_adj))
        return self._build_price_result(
            yes_fv, max(0.01, min(0.99, 1.0 - yes_fv)), greeks, spot, strike, vol,
            time_remaining_min, d2_jd, method, yes_bsm, adjs)

    def _compute_overlays(
        self, apply_overlays, recent_return, recent_return_sigma,
        time_remaining_min, vol_skew, greeks, funding_bias,
    ):
        """Compute all statistical overlay adjustments."""
        mr_adj = candle_adj = season_adj = skew_adj_val = 0.0

        if apply_overlays:
            if recent_return is not None and recent_return_sigma is not None:
                mr_adj = self._mean_reversion_adj(
                    recent_return, recent_return_sigma, time_remaining_min
                )
            candle_adj = self._candle_effect_adj(time_remaining_min)
            season_adj = self._seasonality_adj()

            if vol_skew is not None and greeks["vega"] != 0:
                skew_adj_val = -greeks["vega"] * vol_skew
                skew_adj_val = max(-0.03, min(0.03, skew_adj_val))

        funding_adj = max(-0.02, min(0.02, funding_bias))
        return mr_adj, candle_adj, season_adj, skew_adj_val, funding_adj

    def _build_price_result(
        self, yes_fv, no_fv, greeks, spot, strike, vol,
        time_remaining_min, d2, method, bsm_base, adjs,
    ):
        """Construct the BinaryOptionPrice result."""
        return BinaryOptionPrice(
            yes_fair_value=yes_fv, no_fair_value=no_fv,
            delta=greeks["delta"], gamma=greeks["gamma"],
            theta=greeks["theta_per_min"], vega=greeks["vega"],
            spot=spot, strike=strike, vol=vol,
            time_remaining_min=time_remaining_min, d2=d2,
            moneyness=spot / strike,
            implied_prob=yes_fv, method=method,
            bsm_base=bsm_base, jump_adjustment=adjs["jump"],
            mean_reversion_adj=adjs["mr"],
            candle_effect_adj=adjs["candle"],
            seasonality_adj=adjs["season"],
            skew_adj=adjs["skew"],
        )

    # ==================================================================
    # Merton Jump-Diffusion binary pricing
    # ==================================================================

    def _merton_jump_diffusion(
        self, spot: float, strike: float, vol: float, T: float,
        jump_intensity: float, jump_mean: float, jump_vol: float,
        num_terms: int = 25,
    ) -> tuple:
        """Merton jump-diffusion binary option pricing. Returns (yes_probability, d2_weighted)."""
        lambda_t = jump_intensity * T
        m_bar = math.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0
        log_m = math.log(spot / strike)
        base_drift = (-0.5 * vol ** 2 - jump_intensity * m_bar) * T
        base_var = vol ** 2 * T
        prob_up = poisson_cumul = d2_w = 0.0

        for n in range(num_terms):
            pw = self._poisson_weight(lambda_t, n)
            if pw < 1e-15 and poisson_cumul > 0.9999:
                break
            poisson_cumul += pw
            d_n = self._jump_diffusion_d(log_m, base_drift, base_var, n, jump_mean, jump_vol, spot, strike)
            prob_up += pw * _norm_cdf(d_n)
            d2_w += pw * d_n

        if 0 < poisson_cumul < 0.999:
            prob_up /= poisson_cumul
            d2_w /= poisson_cumul
        return prob_up, d2_w

    @staticmethod
    def _poisson_weight(lambda_t: float, n: int) -> float:
        """Compute Poisson probability P(N_T = n)."""
        if lambda_t < 1e-15:
            return 1.0 if n == 0 else 0.0
        log_pw = -lambda_t + n * math.log(lambda_t) - math.lgamma(n + 1)
        return math.exp(log_pw)

    @staticmethod
    def _jump_diffusion_d(log_moneyness, base_drift, base_var,
                          n, jump_mean, jump_vol, spot, strike) -> float:
        """Compute d_n for n jumps in Merton model."""
        total_var = base_var + n * jump_vol ** 2
        drift = base_drift + n * jump_mean
        if total_var < 1e-20:
            if spot > strike:
                return 1e6
            elif spot < strike:
                return -1e6
            return 0.0
        return (log_moneyness + drift) / math.sqrt(total_var)

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

    # Overlay/secondary methods inherited from BinaryOverlaysMixin:
    # _mean_reversion_adj, _candle_effect_adj, _seasonality_adj,
    # estimate_btc_vol_skew, _intrinsic_price, implied_vol,
    # _bsm_price_at_vol, price_bsm


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_pricer_instance = None
def get_binary_pricer() -> BinaryOptionPricer:
    global _pricer_instance
    if _pricer_instance is None:
        _pricer_instance = BinaryOptionPricer()
    return _pricer_instance