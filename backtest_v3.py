"""
Backtest Engine V3.1 — BTC 15-Minute Polymarket Trading Strategy.
Core engine only. Split modules: backtest_models, backtest_data,
backtest_sim, backtest_confirmation, backtest_reporting.
"""
import math
import sys
from decimal import Decimal
from typing import List, Optional
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from binary_pricer import BinaryOptionPricer
from vol_estimator import VolEstimator
from mispricing_detector import MispricingDetector
from mispricing_models import polymarket_taker_fee

from backtest_models import (
    Candle, BacktestTrade, BacktestResult,
    MIN_EDGE_CENTS, TAKE_PROFIT_PCT, CUT_LOSS_PCT, VOL_METHOD,
    DEFAULT_VOL, BANKROLL_USD, MARKET_BUY_USD, WINDOW_MINUTES,
)
from backtest_data import FundingSnapshot, get_funding_bias_at
from backtest_sim import (
    RealisticConfig, btc_to_probability,
    apply_realistic_entry, check_realistic_fill,
)
from backtest_confirmation import (
    create_fusion_processors, run_confirmation, build_confirmation_metadata,
)
from backtest_reporting import (
    print_results, export_results, parse_args, build_realistic_config,
    print_config_banner, load_candles, load_funding,
)


def _init_backtest_modules():
    pricer = BinaryOptionPricer()
    vol_est = VolEstimator(
        window_minutes=60.0, resample_interval_sec=60.0,
        ewma_halflife_samples=20, min_samples=5, default_vol=DEFAULT_VOL)
    detector = MispricingDetector(
        pricer=pricer, vol_estimator=vol_est, maker_fee=0.00,
        min_edge_cents=MIN_EDGE_CENTS, min_edge_after_fees=0.005,
        take_profit_pct=TAKE_PROFIT_PCT, cut_loss_pct=CUT_LOSS_PCT,
        vol_method=VOL_METHOD, bankroll=BANKROLL_USD,
        use_half_kelly=True, max_kelly_fraction=0.05)
    return vol_est, detector


def _chunk_candles_to_windows(candles: List[Candle]) -> List[List[Candle]]:
    windows, current, ws = [], [], None
    for c in candles:
        ms = c.timestamp.minute // WINDOW_MINUTES
        ss = c.timestamp.replace(minute=ms * WINDOW_MINUTES, second=0, microsecond=0)
        if ws is None: ws = ss
        if ss != ws:
            if len(current) >= WINDOW_MINUTES: windows.append(current)
            current, ws = [c], ss
        else:
            current.append(c)
    if len(current) >= WINDOW_MINUTES: windows.append(current)
    return windows


def _get_funding_at_window(use_funding, fs, ts):
    if use_funding and fs:
        bias = get_funding_bias_at(fs, ts)
        regime = "NEUTRAL"
        for s in fs:
            if s.timestamp <= ts: regime = s.classification
            else: break
        return bias, regime
    return 0.0, "DISABLED"


def _compute_position_size(rc, signal, bankroll):
    if rc:
        ae = abs(signal.edge)
        scale = 0.6 if ae < 0.10 else (1.0 if ae < 0.20 else 1.5)
        ps = min(signal.kelly_bet_usd, bankroll * 0.05, MARKET_BUY_USD * scale)
    else:
        ps = min(signal.kelly_bet_usd, bankroll * 0.05, MARKET_BUY_USD)
    return max(ps, 0.10)


def _compute_trade_pnl(direction, entry_price, btc_moved_up, position_size, use_maker):
    won = btc_moved_up if direction == "long" else not btc_moved_up
    nt = position_size / entry_price if entry_price > 0 else 0
    fr = 0.0 if use_maker else polymarket_taker_fee(entry_price)
    fee = fr * nt * entry_price
    pnl_bf = nt * (1.0 - entry_price) if won else -nt * entry_price
    return won, "WIN" if won else "LOSS", pnl_bf - fee, pnl_bf, fee


def _check_window_skip(hl, wl, dm, rc, cl, tts, wp, sr):
    if hl < 30: sr["warmup"] += 1; return "warmup"
    if wl <= dm: sr["no_data"] += 1; return "no_data"
    if rc and cl >= rc.max_consecutive_losses: sr["streak_throttle"] += 1; return "streak_throttle"
    if rc and wp > 20 and tts / wp > rc.max_trade_rate: sr["trade_rate_cap"] += 1; return "trade_rate_cap"
    return None


def _get_signal_for_window(vol_est, detector, window, dm, rc, lmy, use_funding, fs, use_maker, ph):
    bo = window[0].open
    bs = window[dm].close
    trm = float(WINDOW_MINUTES - dm)
    pp = btc_to_probability(window, dm, realistic=rc, model_fair_value=lmy)
    ph.append(Decimal(str(round(pp, 4))))
    if len(ph) > 100: ph.pop(0)
    ve = vol_est.get_vol(VOL_METHOD)
    vs = BinaryOptionPricer.estimate_btc_vol_skew(spot=bs, strike=bo, vol=ve.annualized_vol, time_remaining_min=trm)
    fb, fr = _get_funding_at_window(use_funding, fs, window[dm].timestamp)
    sig = detector.detect(
        yes_market_price=pp, no_market_price=1.0 - pp,
        btc_spot=bs, btc_strike=bo, time_remaining_min=trm,
        position_size_usd=MARKET_BUY_USD, use_maker=use_maker, vol_skew=vs, funding_bias=fb)
    return pp, vs, fb, fr, sig, trm


def _run_bt_confirmation(use_conf, fproc, ph, window, dm, hbc, pp, mb):
    if not (use_conf and fproc and len(ph) >= 5): return 0, 0
    md = build_confirmation_metadata(window, dm, hbc[-100:])
    if not md: return 0, 0
    return run_confirmation(fproc, Decimal(str(round(pp, 4))), list(ph), md, mb)


def _should_skip_signal(sig, rc, sr, use_conf, fproc, ph, window, dm, hbc, pp):
    if rc and sig.vol_confidence < rc.min_vol_confidence: sr["vol_confidence"] += 1; return True
    if rc and abs(sig.edge) < rc.min_edge_realistic: sr["edge_too_small"] += 1; return True
    mb = sig.direction == "BUY_YES"
    conf, contra = _run_bt_confirmation(use_conf, fproc, ph, window, dm, hbc, pp, mb)
    if contra > conf and sig.confidence < 0.6: sr["confirmation_veto"] += 1; return True
    return False


def _compute_max_drawdown(eq):
    peak = mdd = 0.0
    for v in eq:
        if v > peak: peak = v
        dd = peak - v
        if dd > mdd: mdd = dd
    return mdd


def _compute_sharpe(trades):
    if len(trades) <= 1: return 0.0
    pnls = [t.pnl for t in trades]
    mu = sum(pnls) / len(pnls)
    sd = math.sqrt(sum((p - mu)**2 for p in pnls) / (len(pnls) - 1))
    return (mu / sd * math.sqrt(96 * 365)) if sd > 0 else 0.0


def _build_backtest_trade(window, d, ep, ps, bo, bc, bmu, outcome, pnl, pnl_bf, fee,
                          sig, conf, contra, vs, fb, fr, trm):
    return BacktestTrade(
        window_start=window[0].timestamp, direction=d,
        entry_price=round(ep, 4), position_size=round(ps, 2),
        actual_btc_open=bo, actual_btc_close=bc, btc_moved_up=bmu, outcome=outcome,
        pnl=round(pnl, 4), pnl_before_fees=round(pnl_bf, 4), fee_paid=round(fee, 4),
        model_yes_price=round(sig.yes_model, 4), model_no_price=round(sig.no_model, 4),
        edge=round(sig.edge, 4), edge_pct=round(sig.edge_pct, 4),
        kelly_fraction=round(sig.kelly_fraction, 4),
        realized_vol=round(sig.realized_vol, 4), implied_vol=round(sig.implied_vol, 4),
        vrp=round(sig.vrp, 4), pricing_method=sig.pricing_method,
        confirming_signals=conf, contradicting_signals=contra,
        vol_skew=round(vs, 8), funding_bias=round(fb, 6),
        funding_regime=fr, time_remaining_min=trm)


def _compute_backtest_stats(trades, eq, tw, skipped, candles, bankroll, tf, rc):
    n, wins = len(trades), sum(1 for t in trades if t.outcome == "WIN")
    tp = sum(t.pnl for t in trades)
    gp, gl = sum(t.pnl for t in trades if t.pnl > 0), abs(sum(t.pnl for t in trades if t.pnl < 0))
    return BacktestResult(
        start_date=candles[0].timestamp.strftime("%Y-%m-%d") if candles else "",
        end_date=candles[-1].timestamp.strftime("%Y-%m-%d") if candles else "",
        strategy_mode="quant_v3.1_realistic" if (rc and rc.enabled) else "quant_v3.1",
        total_windows=tw, trades_taken=n, trades_skipped=skipped, wins=wins, losses=n-wins,
        win_rate=round((wins/n*100) if n else 0, 2), total_pnl=round(tp, 4), total_fees=round(tf, 4),
        max_drawdown=round(_compute_max_drawdown(eq), 4),
        profit_factor=round((gp/gl) if gl > 0 else float("inf"), 2),
        avg_pnl_per_trade=round((tp/n) if n else 0, 4),
        avg_edge_per_trade=round((sum(abs(t.edge) for t in trades)/n) if n else 0, 4),
        avg_kelly=round((sum(t.kelly_fraction for t in trades)/n) if n else 0, 4),
        best_trade=round(max((t.pnl for t in trades), default=0), 4),
        worst_trade=round(min((t.pnl for t in trades), default=0), 4),
        sharpe_ratio=round(_compute_sharpe(trades), 2), bankroll_final=round(bankroll, 2),
        bankroll_growth=round(((bankroll-BANKROLL_USD)/BANKROLL_USD*100) if BANKROLL_USD > 0 else 0, 2),
        trades=trades, equity_curve=eq)


def _init_state(realistic):
    return {
        "ph": [], "hbc": [], "trades": [], "eq": [0.0],
        "bankroll": BANKROLL_USD, "tf": 0.0, "skipped": 0,
        "sr": {"warmup": 0, "no_edge": 0, "confirmation_veto": 0, "theta_decay": 0,
               "no_data": 0, "funding_veto": 0, "vol_confidence": 0,
               "streak_throttle": 0, "no_fill": 0, "edge_too_small": 0, "trade_rate_cap": 0},
        "rc": realistic if (realistic and realistic.enabled) else None,
        "closs": 0, "tts": 0, "wp": 0, "lmy": None,
    }


def _process_window(wi, window, st, vol_est, detector, fproc, dm,
                    use_conf, use_maker, use_funding, fs, tw, verbose):
    for c in window:
        vol_est.add_price(c.close, c.timestamp.timestamp())
        st["hbc"].append(c.close)
    vol_est.set_simulated_time(window[min(dm, len(window)-1)].timestamp.timestamp())

    skip = _check_window_skip(len(st["hbc"]), len(window), dm, st["rc"],
                              st["closs"], st["tts"], st["wp"], st["sr"])
    if skip:
        st["skipped"] += 1
        if skip == "streak_throttle": st["closs"] = max(0, st["closs"] - 1)
        return
    st["wp"] += 1

    pp, vs, fb, fr, sig, trm = _get_signal_for_window(
        vol_est, detector, window, dm, st["rc"], st["lmy"], use_funding, fs, use_maker, st["ph"])
    if not sig.is_tradeable:
        st["sr"]["no_edge"] += 1; st["skipped"] += 1; st["lmy"] = sig.yes_model; return
    st["lmy"] = sig.yes_model

    if _should_skip_signal(sig, st["rc"], st["sr"], use_conf, fproc,
                           st["ph"], window, dm, st["hbc"], pp):
        st["skipped"] += 1; return

    d = "long" if sig.direction == "BUY_YES" else "short"
    ep = pp if d == "long" else 1.0 - pp
    ep = apply_realistic_entry(st["rc"], ep, trm)
    if not check_realistic_fill(st["rc"], sig):
        st["sr"]["no_fill"] += 1; st["skipped"] += 1; return
    ps = _compute_position_size(st["rc"], sig, st["bankroll"])

    bo, bc = window[0].open, window[-1].close
    won, outcome, pnl, pnl_bf, fee = _compute_trade_pnl(d, ep, bc > bo, ps, use_maker)
    st["closs"] = 0 if won else st["closs"] + 1
    st["tts"] += 1; st["tf"] += fee
    st["bankroll"] = max(1.0, st["bankroll"] + pnl)
    cum = st["eq"][-1] + pnl; st["eq"].append(cum)

    conf, contra = _run_bt_confirmation(
        use_conf, fproc, st["ph"], window, dm, st["hbc"], pp, sig.direction == "BUY_YES")
    st["trades"].append(_build_backtest_trade(
        window, d, ep, ps, bo, bc, bc > bo, outcome, pnl, pnl_bf, fee,
        sig, conf, contra, vs, fb, fr, trm))
    detector.update_bankroll(st["bankroll"])


def run_backtest_v3(
    candles: List[Candle], decision_minute: int = 2, verbose: bool = False,
    use_confirmation: bool = True, use_maker: bool = True, use_funding: bool = True,
    funding_snapshots=None, realistic=None,
) -> BacktestResult:
    """Backtest the V3.1 quant strategy."""
    vol_est, detector = _init_backtest_modules()
    fproc = create_fusion_processors() if use_confirmation else None
    windows = _chunk_candles_to_windows(candles)
    st = _init_state(realistic)
    tw = len(windows)
    for wi, window in enumerate(windows):
        _process_window(wi, window, st, vol_est, detector, fproc, decision_minute,
                        use_confirmation, use_maker, use_funding, funding_snapshots, tw, verbose)
    if verbose: logger.info(f"Skip reasons: {st['sr']}")
    return _compute_backtest_stats(
        st["trades"], st["eq"], tw, st["skipped"], candles, st["bankroll"], st["tf"], st.get("rc"))


def main():
    args = parse_args()
    if args.verbose:
        logger.remove(); logger.add(sys.stderr, level="DEBUG")
    use_maker = not args.taker
    use_funding = not args.no_funding
    rc = build_realistic_config(args)
    print_config_banner(args, rc, use_maker, use_funding)
    candles = load_candles(args)
    if not candles:
        print("ERROR: No candle data"); sys.exit(1)
    funding_snapshots, use_funding = load_funding(candles, use_funding)
    result = run_backtest_v3(
        candles=candles, decision_minute=args.decision_minute,
        verbose=args.verbose, use_confirmation=not args.no_confirmation,
        use_maker=use_maker, use_funding=use_funding,
        funding_snapshots=funding_snapshots, realistic=rc)
    print_results(result)
    if args.output: export_results(result, args.output)


if __name__ == "__main__":
    main()
