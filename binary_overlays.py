"""
Binary pricer overlays — mean reversion, candle effect, seasonality,
vol skew estimation, implied vol solver, BSM-only pricing.

SRP: Statistical adjustments and secondary pricing methods.
"""
import math
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional


class BinaryOverlaysMixin:
    """Mixin adding statistical overlays and secondary methods to BinaryOptionPricer."""

    # ── Mean reversion ───────────────────────────────────────────────

    def _mean_reversion_adj(self, recent_return: float, recent_sigma: float,
                            t_min: float) -> float:
        """Mean reversion overlay. BTC has negative autocorrelation at 15-min."""
        abs_sig = abs(recent_sigma)
        if abs_sig < self.MEAN_REVERSION_THRESHOLD_SIGMA:
            return 0.0
        excess = abs_sig - self.MEAN_REVERSION_THRESHOLD_SIGMA
        raw = min(excess / 2.0, 1.0) * self.MEAN_REVERSION_STRENGTH
        time_factor = min(t_min / 10.0, 1.0)
        strength = raw * time_factor
        if recent_return > 0:
            return -strength
        else:
            return strength * self.NEGATIVE_REVERSION_MULT

    # ── Candle effect ────────────────────────────────────────────────

    def _candle_effect_adj(self, time_remaining_min: float) -> float:
        """Turn-of-candle effect (Shanaev & Vasenin 2023)."""
        now = datetime.now(timezone.utc)
        resolve_minute = (now.minute + int(time_remaining_min)) % 60
        if resolve_minute in self.CANDLE_MINUTES:
            return 0.002
        elif resolve_minute % 15 <= 1 or resolve_minute % 15 >= 14:
            return 0.001
        else:
            return -0.0005

    # ── Seasonality ──────────────────────────────────────────────────

    def _seasonality_adj(self) -> float:
        """Intraday seasonality (QuantPedia / SSRN 4581124)."""
        now = datetime.now(timezone.utc)
        bias = self.HOURLY_BIAS.get(now.hour, 0.0)
        if now.weekday() == 4:
            bias *= 1.2
        return bias * self.SEASONALITY_SCALE

    # ── Vol skew estimation ──────────────────────────────────────────

    @staticmethod
    def estimate_btc_vol_skew(spot: float, strike: float, vol: float,
                              time_remaining_min: float) -> float:
        """Estimate dσ/dK for BTC at short horizons."""
        if spot <= 0 or strike <= 0 or vol <= 0:
            return 0.0
        T_daily = 1.0 / (365.25 * 24 * 60)
        T_actual = max(time_remaining_min, 0.1) / (365.25 * 24 * 60)
        time_scale = min(3.0, math.sqrt(T_daily / T_actual))
        base_skew = -2.0 * vol / spot
        return max(-0.001, min(0.001, base_skew * time_scale))

    # ── Intrinsic price ──────────────────────────────────────────────

    def _intrinsic_price(self, spot: float, strike: float):
        """Edge-case pricing when no vol or time."""
        from binary_pricer import BinaryOptionPrice
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
            implied_prob=yes_fv, method="intrinsic")

    # ── Implied vol ──────────────────────────────────────────────────

    def _bsm_price_at_vol(self, vol_mid, T, spot, strike, is_call):
        from binary_pricer import _norm_cdf
        vst = vol_mid * math.sqrt(T)
        if vst < 1e-12:
            return None
        d2 = (math.log(spot / strike) + (self.r - 0.5 * vol_mid**2) * T) / vst
        return _norm_cdf(d2) if is_call else _norm_cdf(-d2)

    def implied_vol(self, market_price: float, spot: float, strike: float,
                    time_remaining_min: float, is_call: bool = True,
                    max_iterations: int = 50, tolerance: float = 1e-6) -> float:
        """Back out implied vol from market price using bisection."""
        if time_remaining_min < 0.1:
            return 0.0
        T = max(time_remaining_min / self.MINUTES_PER_YEAR, 1e-12)
        lo, hi = 0.01, 5.0
        for _ in range(max_iterations):
            mid = (lo + hi) / 2.0
            p = self._bsm_price_at_vol(mid, T, spot, strike, is_call)
            if p is None:
                lo = mid; continue
            if abs(p - market_price) < tolerance:
                return mid
            if p > market_price:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    # ── BSM-only pricing (V1 backward compat) ────────────────────────

    def price_bsm(self, spot: float, strike: float, vol: float,
                  time_remaining_min: float):
        """Pure BSM pricing for comparison/debugging."""
        from binary_pricer import BinaryOptionPrice
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
            moneyness=spot / strike, implied_prob=yes_fv, method="bsm")

