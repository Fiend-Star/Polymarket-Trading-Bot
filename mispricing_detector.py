"""
Mispricing Detector V2 — Quantitative edge detection for Polymarket binary options.

Key V2 upgrades: nonlinear fee, Kelly sizing, JD integration, VRP edge,
return stats overlays, improved tradeability logic.
"""
from typing import Optional
from loguru import logger

from binary_pricer import get_binary_pricer
from vol_estimator import get_vol_estimator
from mispricing_exits import MispricingExitsMixin
from mispricing_models import (
    polymarket_taker_fee, kelly_fraction,
    MispricingSignal, ExitSignal,
)


class MispricingDetector(MispricingExitsMixin):
    """Quantitative edge detection engine — V2."""

    def __init__(self, pricer=None, vol_estimator=None,
                 maker_fee=0.00, taker_fee=0.02, min_edge_cents=0.02,
                 min_edge_after_fees=0.005, take_profit_pct=0.30, cut_loss_pct=-0.50,
                 vol_method="ewma", bankroll=50.0, use_half_kelly=True,
                 max_kelly_fraction=0.05):
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

    def _compute_metrics(self, yes_mp, no_mp, btc_spot, btc_strike, trm, ps_usd, use_maker, vol_skew, fb):
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

    def detect(self, yes_market_price: float, no_market_price: float,
               btc_spot: float, btc_strike: float, time_remaining_min: float,
               position_size_usd: float = 1.0, use_maker: bool = True,
               vol_skew: Optional[float] = None, funding_bias: float = 0.0) -> MispricingSignal:
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
        return MispricingSignal(
            edge=edge, edge_pct=edge_pct, direction=direction,
            yes_market=yes_mkt, no_market=no_mkt,
            yes_model=model.yes_fair_value, no_model=model.no_fair_value,
            spot=spot, strike=strike, vol=vol, time_remaining_min=trm,
            delta=model.delta, gamma=model.gamma, theta_per_min=model.theta,
            confidence=conf, vol_confidence=vol_conf,
            implied_vol=iv, realized_vol=vol, vol_spread=vs,
            vrp=vrp, vrp_signal=vrp_sig, expected_pnl=ev, fee_cost=fee,
            net_expected_pnl=net_ev, is_tradeable=tradeable,
            kelly_fraction=kf, kelly_bet_usd=kb, pricing_method=model.method)

    def _get_vol_and_jumps(self):
        ve = self.vol_est.get_vol(method=self.vol_method)
        return ve, ve.annualized_vol, ve.confidence, self.vol_est.get_jump_params(), self.vol_est.get_return_stats()

    def _compute_model_price(self, spot, strike, rv, trm, jp, rs, vol_skew, fb):
        return self.pricer.price(
            spot=spot, strike=strike, vol=rv, time_remaining_min=trm,
            jump_intensity=jp.intensity, jump_mean=jp.mean, jump_vol=jp.vol,
            recent_return=rs.recent_return, recent_return_sigma=rs.recent_return_sigma,
            apply_overlays=True, vol_skew=vol_skew, funding_bias=fb)

    def _compute_iv_and_vrp(self, yes_mp, spot, strike, trm, rv):
        try:
            iv = self.pricer.implied_vol(market_price=yes_mp, spot=spot,
                                          strike=strike, time_remaining_min=trm, is_call=True)
        except Exception:
            iv = rv
        vs = iv - rv
        self.vol_est.record_iv(iv)
        self.vol_est.record_rv(rv)
        vrp = self.vol_est.get_vrp()
        vrp_sig = "FADE_EXTREME" if vrp > 0.20 else ("VOL_CHEAP" if vrp < -0.05 else "NEUTRAL")
        return iv, vs, vrp, vrp_sig

    def _determine_direction(self, model, yes_mp, no_mp):
        ye = model.yes_fair_value - yes_mp
        ne = model.no_fair_value - no_mp
        if abs(ye) >= abs(ne):
            return ("BUY_YES", ye, yes_mp, model.yes_fair_value) if ye > 0 else ("BUY_NO", ne, no_mp, model.no_fair_value)
        return ("BUY_NO", ne, no_mp, model.no_fair_value) if ne > 0 else ("BUY_YES", ye, yes_mp, model.yes_fair_value)

    def _compute_fees_and_ev(self, use_maker, tp, ae, ps_usd):
        fr = self.maker_fee if use_maker else polymarket_taker_fee(tp)
        nt = ps_usd / tp if tp > 0 else 0
        ev = ae * nt
        fc = fr * nt * tp
        return fr, fc, ev, ev - fc

    def _compute_kelly(self, model_prob, trade_price):
        kf = kelly_fraction(model_prob, trade_price, self.use_half_kelly, self.max_kelly_fraction)
        return kf, kf * self.bankroll

    def _evaluate_tradeability(self, ae, fr, nev, trm, vc, kf, model, pe):
        min_edge = max(self.min_edge_cents, fr * 2.0)
        ok = (ae >= min_edge and nev >= self.min_edge_after_fees
              and trm >= 1.0 and vc >= 0.2 and kf > 0.001)
        if trm < 3.0 and abs(model.theta) * trm > ae * 0.5:
            ok = False
        return ok

    def _compute_vrp_boost(self, vrp_sig, direction, yes_mp, no_mp):
        if vrp_sig == "FADE_EXTREME":
            if (direction == "BUY_NO" and yes_mp > 0.70) or (direction == "BUY_YES" and no_mp > 0.70):
                return 0.10
        return 0.0

    def _compute_confidence(self, vc, ae, trm, kf, vrp_b):
        return vc * 0.3 + min(1.0, ae / 0.10) * 0.3 + min(1.0, trm / 5.0) * 0.2 + (0.1 if kf > 0.01 else 0.0) + vrp_b

    def _log_detection(self, signal, ve, jp, rs, model):
        s = signal
        logger.info(f"═ MISPRICING ═ ${s.spot:,.0f}→${s.strike:,.0f} T={s.time_remaining_min:.1f}m | "
                    f"edge={s.edge:+.4f} net=${s.net_expected_pnl:+.4f} kelly={s.kelly_fraction:.1%} | "
                    f"{'✓' if s.is_tradeable else '✗'} {s.direction}")

    # Exit/utility methods inherited from MispricingExitsMixin


# Singleton

_detector_instance = None

def get_mispricing_detector(**kwargs) -> MispricingDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MispricingDetector(**kwargs)
    return _detector_instance
