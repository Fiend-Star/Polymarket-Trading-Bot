"""
VolAnalyticsMixin — jump estimation, return stats, VRP tracking, utility.

SRP: Statistical analytics layered on top of core vol estimation.
"""
import math
import time
from typing import Optional
from loguru import logger

MINUTES_PER_YEAR = 365.25 * 24 * 60


class VolAnalyticsMixin:
    """Mixin adding jump estimation, return stats, VRP, and utility to VolEstimator."""

    def get_jump_params(self):
        """Export jump parameters for Merton JD pricer."""
        from vol_estimator import JumpEstimate
        now = self._now()
        while self._detected_jumps and self._detected_jumps[0][0] < now - self.window_minutes * 60:
            self._detected_jumps.popleft()
        n = len(self._detected_jumps)
        wy = self.window_minutes / MINUTES_PER_YEAR
        recent = any(ts > now - self._recent_jump_window_sec for ts, _ in self._detected_jumps)
        if n < 2:
            return JumpEstimate(intensity=1500.0, mean=-0.0008, vol=0.003,
                                recent_jump=recent, num_jumps_detected=n, confidence=0.1)
        jrs = [lr for _, lr in self._detected_jumps]
        return self._estimate_jump_distribution(jrs, wy, n, recent)

    def _estimate_jump_distribution(self, jrs, wy, n, recent):
        from vol_estimator import JumpEstimate
        intensity = n / wy if wy > 0 else 1500.0
        mean_j = sum(jrs) / len(jrs)
        var_j = (sum((j - mean_j)**2 for j in jrs) / (len(jrs)-1)
                 if len(jrs) > 1 else 0.003**2)
        vol_j = math.sqrt(max(var_j, 1e-10))
        intensity = max(100.0, min(50000.0, intensity))
        vol_j = max(0.001, min(0.05, vol_j))
        conf = min(1.0, n / 10.0)
        if recent:
            intensity *= 2.0
        return JumpEstimate(intensity=intensity, mean=mean_j, vol=vol_j,
                            recent_jump=recent, num_jumps_detected=n, confidence=conf)

    # ── Return statistics ────────────────────────────────────────────

    def get_return_stats(self, reference_price=None):
        from vol_estimator import ReturnStats
        bars = list(self._bars)
        if len(bars) < 2:
            return ReturnStats(0.0, 0.0, 0.0, False, 0)
        cur = bars[-1].price
        ref = reference_price or bars[0].price
        rr = (cur - ref) / ref if ref > 0 and cur > 0 else 0.0
        rs = self._compute_recent_sigma(bars, rr)
        lrs = [math.log(bars[i].price / bars[i-1].price)
               for i in range(1, len(bars)) if bars[i-1].price > 0 and bars[i].price > 0]
        ac = self._compute_autocorr(lrs) if len(lrs) > 5 else 0.0
        return ReturnStats(rr, rs, ac, abs(rs) > 1.5, len(lrs))

    def _compute_recent_sigma(self, bars, recent_return):
        ve = self.get_vol(method="ewma")
        elapsed = (bars[-1].timestamp - bars[0].timestamp) / 60.0
        if ve.vol_per_minute > 0 and elapsed > 0:
            em = ve.vol_per_minute * math.sqrt(max(elapsed, 0.1))
            return recent_return / em if em > 0 else 0.0
        return 0.0

    def _compute_autocorr(self, returns):
        if len(returns) < 3:
            return 0.0
        n = len(returns)
        m = sum(returns) / n
        v = sum((r - m)**2 for r in returns) / n
        if v < 1e-20:
            return 0.0
        cov = sum((returns[i]-m) * (returns[i-1]-m) for i in range(1, n)) / (n-1)
        return cov / v

    # ── VRP tracking ─────────────────────────────────────────────────

    def record_iv(self, iv, timestamp=None):
        self._iv_observations.append((timestamp or time.time(), iv))

    def record_rv(self, rv, timestamp=None):
        self._rv_observations.append((timestamp or time.time(), rv))

    def get_vrp(self) -> float:
        """Get Variance Risk Premium (IV - RV). Default ~15%."""
        if not self._iv_observations or not self._rv_observations:
            return 0.15
        return self._iv_observations[-1][1] - self._rv_observations[-1][1]

    # ── Utility ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "raw_ticks": len(self._raw_ticks), "bars": len(self._bars),
            "window_minutes": self.window_minutes,
            "last_bar_time": self._last_bar_time,
            "cached_vol": self._cached_vol.annualized_vol if self._cached_vol else None,
            "jumps_detected": len(self._detected_jumps),
            "iv_observations": len(self._iv_observations),
            "rv_observations": len(self._rv_observations),
        }

    def reset(self):
        """Clear data but keep jump history and VRP."""
        self._raw_ticks.clear()
        self._bars.clear()
        self._last_bar_time = 0.0
        self._cached_vol = None
        self._cache_time = 0.0

