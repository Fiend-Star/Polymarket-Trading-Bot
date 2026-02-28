"""
VolMethodsMixin — alternative vol estimation methods (close-to-close, Parkinson).

SRP: Additional vol computation algorithms layered on VolEstimator core.
"""
import math


class VolMethodsMixin:
    """Mixin adding close-to-close and Parkinson vol methods to VolEstimator."""

    def _close_to_close_vol(self):
        """Standard close-to-close realized vol from log returns."""
        now = self._now()
        bars = list(self._bars)
        log_returns = [math.log(bars[i].price / bars[i-1].price)
                       for i in range(1, len(bars))
                       if bars[i-1].price > 0 and bars[i].price > 0]
        if len(log_returns) < self.min_samples:
            return self._default_vol_estimate(now, len(log_returns))
        mean_lr = sum(log_returns) / len(log_returns)
        variance = sum((lr - mean_lr)**2 for lr in log_returns) / (len(log_returns) - 1)
        return self._build_vol_estimate(variance, bars, log_returns, "close_to_close", now)

    def _parkinson_vol(self):
        """Parkinson high-low vol estimator (5x more efficient than close-to-close)."""
        now = self._now()
        bars = [b for b in self._bars if b.high is not None and b.low is not None]
        if len(bars) < self.min_samples:
            return self._close_to_close_vol()
        sum_sq, valid = self._parkinson_sum_squares(bars)
        if valid < self.min_samples:
            return self._close_to_close_vol()
        var = sum_sq / (4.0 * valid * math.log(2))
        pseudo = [0.0] * valid
        return self._build_vol_estimate(var, bars, pseudo, "parkinson", now)

    def _parkinson_sum_squares(self, bars: list) -> tuple:
        sum_sq, valid = 0.0, 0
        for b in bars:
            if b.high > 0 and b.low > 0 and b.high >= b.low:
                r = b.high / b.low
                if r > 0:
                    sum_sq += math.log(r)**2
                    valid += 1
        return sum_sq, valid

