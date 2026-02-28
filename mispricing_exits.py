"""
MispricingExitsMixin — exit monitoring, Greeks timing, bankroll, stats.

SRP: Position exit logic and secondary analytics for MispricingDetector.
"""
from loguru import logger
from mispricing_models import ExitSignal


class MispricingExitsMixin:
    """Mixin adding exit monitoring and utility methods to MispricingDetector."""

    def check_exit(self, entry_price, direction, btc_spot, btc_strike,
                   time_remaining_min, yes_market_price, no_market_price):
        """Check if an open position should be exited mid-market."""
        cv, upnl, upct = self._compute_exit_value(
            entry_price, direction, btc_spot, btc_strike,
            time_remaining_min, yes_market_price, no_market_price)
        return self._build_exit_signal(entry_price, cv, upnl, upct, time_remaining_min)

    def _compute_exit_value(self, entry_price, direction, btc_spot, btc_strike,
                            trm, yes_mp, no_mp):
        vol = self.vol_est.get_vol(method=self.vol_method).annualized_vol
        model = self.pricer.price(spot=btc_spot, strike=btc_strike, vol=vol,
                                  time_remaining_min=trm)
        if direction == "BUY_YES":
            cmv, cmk = model.yes_fair_value, yes_mp
        else:
            cmv, cmk = model.no_fair_value, no_mp
        cv = max(cmv, cmk)
        upnl = cv - entry_price
        return cv, upnl, upnl / entry_price if entry_price > 0 else 0.0

    def _build_exit_signal(self, entry_price, cv, upnl, upct, trm):
        if upct >= self.take_profit_pct:
            return ExitSignal(action="TAKE_PROFIT", current_value=cv,
                              entry_price=entry_price, unrealized_pnl=upnl,
                              unrealized_pnl_pct=upct,
                              reason=f"TP hit: {upct:.0%} >= {self.take_profit_pct:.0%}")
        if upct <= self.cut_loss_pct and trm > 2.0:
            return ExitSignal(action="CUT_LOSS", current_value=cv,
                              entry_price=entry_price, unrealized_pnl=upnl,
                              unrealized_pnl_pct=upct,
                              reason=f"SL hit: {upct:.0%} <= {self.cut_loss_pct:.0%}")
        return ExitSignal(action="HOLD", current_value=cv,
                          entry_price=entry_price, unrealized_pnl=upnl,
                          unrealized_pnl_pct=upct,
                          reason=f"Hold (PnL={upct:+.0%}, T={trm:.1f}min)")

    def optimal_entry_window(self, btc_spot, btc_strike, total_market_minutes=15.0):
        """Analyze Greeks across market lifetime for optimal entry timing."""
        vol = self.vol_est.get_vol(method=self.vol_method).annualized_vol
        analysis = []
        for ml in [14, 12, 10, 8, 6, 4, 2, 1, 0.5]:
            m = self.pricer.price(spot=btc_spot, strike=btc_strike, vol=vol,
                                  time_remaining_min=ml)
            analysis.append({"T_min": ml, "yes_fv": m.yes_fair_value,
                             "delta": m.delta, "gamma": m.gamma,
                             "theta_per_min": m.theta})
        best = max([a for a in analysis if a["T_min"] > 1.0],
                   key=lambda x: abs(x["gamma"]) / (abs(x["theta_per_min"]) + 1e-10),
                   default=analysis[0])
        return {"recommended_entry_T": best["T_min"], "profile": analysis, "vol_used": vol}

    def update_bankroll(self, new_bankroll: float):
        old = self.bankroll
        self.bankroll = max(1.0, new_bankroll)
        logger.info(f"Bankroll: ${old:.2f} → ${self.bankroll:.2f}")

    def get_stats(self) -> dict:
        return {
            "total_detections": self._total_detections,
            "tradeable_detections": self._tradeable_detections,
            "hit_rate": self._tradeable_detections / max(self._total_detections, 1),
            "bankroll": self.bankroll,
            "vol_estimator": self.vol_est.get_stats(),
        }
