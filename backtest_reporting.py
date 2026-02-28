"""
Backtest reporting — pretty printing, JSON export, CLI.

SRP: Output formatting and CLI argument handling for backtest.
"""
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict

from loguru import logger
from backtest_models import (
    BacktestResult, BacktestTrade, BANKROLL_USD, MIN_EDGE_CENTS, VOL_METHOD,
)
from backtest_data import fetch_binance_klines, load_csv_candles, fetch_funding_rates
from backtest_sim import RealisticConfig


# ── Pretty printer ───────────────────────────────────────────────────

def _print_summary_header(r: BacktestResult):
    print()
    print("=" * 90)
    print(f"          BACKTEST RESULTS — {r.strategy_mode.upper()}")
    print("=" * 90)
    print(f"  Period:              {r.start_date} → {r.end_date}")
    print(f"  Total 15m windows:   {r.total_windows}")
    print(f"  Trades taken:        {r.trades_taken}")
    print(f"  Trades skipped:      {r.trades_skipped}")
    print(f"  Trade rate:          {r.trades_taken / max(r.total_windows, 1) * 100:.1f}%")
    print("-" * 90)
    print(f"  Wins / Losses:       {r.wins} / {r.losses}")
    print(f"  Win Rate:            {r.win_rate:.1f}%")
    print("-" * 90)
    print(f"  Total P&L:           ${r.total_pnl:+.4f}")
    print(f"  Total Fees Paid:     ${r.total_fees:.4f}")
    print(f"  Avg P&L / Trade:     ${r.avg_pnl_per_trade:+.4f}")
    print(f"  Avg Edge / Trade:    ${r.avg_edge_per_trade:.4f}")
    print(f"  Avg Kelly Fraction:  {r.avg_kelly:.1%}")
    print(f"  Best / Worst Trade:  ${r.best_trade:+.4f} / ${r.worst_trade:+.4f}")
    print(f"  Max Drawdown:        ${r.max_drawdown:.4f}")
    print(f"  Profit Factor:       {r.profit_factor:.2f}")
    print(f"  Sharpe Ratio (ann):  {r.sharpe_ratio:.2f}")
    print("-" * 90)
    print(f"  Bankroll:            ${BANKROLL_USD:.2f} → ${r.bankroll_final:.2f} ({r.bankroll_growth:+.1f}%)")
    print("=" * 90)


def _print_direction_and_vol(r: BacktestResult):
    longs = [t for t in r.trades if t.direction == "long"]
    shorts = [t for t in r.trades if t.direction == "short"]
    lw = sum(1 for t in longs if t.outcome == "WIN")
    sw = sum(1 for t in shorts if t.outcome == "WIN")
    print(f"\n  Long:  {len(longs):>4} ({(lw/len(longs)*100) if longs else 0:.1f}% WR) | "
          f"Short: {len(shorts):>4} ({(sw/len(shorts)*100) if shorts else 0:.1f}% WR)")
    n = len(r.trades)
    arv = sum(t.realized_vol for t in r.trades) / n
    aiv = sum(t.implied_vol for t in r.trades) / n
    avrp = sum(t.vrp for t in r.trades) / n
    print(f"  Avg RV: {arv:.0%} | IV: {aiv:.0%} | VRP: {avrp:+.0%}")
    methods: Dict[str, int] = {}
    for t in r.trades:
        methods[t.pricing_method] = methods.get(t.pricing_method, 0) + 1
    print(f"\n  Pricing methods:")
    for m, c in sorted(methods.items(), key=lambda x: -x[1]):
        print(f"    {m:<20} {c:>4} ({c/n*100:.0f}%)")


def _print_funding_and_edge(r: BacktestResult):
    regimes: Dict[str, list] = {}
    for t in r.trades:
        regimes.setdefault(t.funding_regime, []).append(t)
    if any(t.funding_regime != "DISABLED" for t in r.trades):
        print(f"\n  Funding regime breakdown:")
        for regime, rt in sorted(regimes.items()):
            rw = sum(1 for t in rt if t.outcome == "WIN")
            rp = sum(t.pnl for t in rt)
            print(f"    {regime:<20} {len(rt):>4} trades, "
                  f"{(rw/len(rt)*100) if rt else 0:.1f}% WR, PnL ${rp:+.4f}")
    edges = sorted([abs(t.edge) for t in r.trades])
    n = len(edges)
    print(f"\n  Edge dist: min=${edges[0]:.4f} | 25th=${edges[n//4]:.4f} | "
          f"med=${edges[n//2]:.4f} | 75th=${edges[3*n//4]:.4f} | max=${edges[-1]:.4f}")
    print(f"\n  Win rate by edge size:")
    for lo, hi in [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.00)]:
        bt = [t for t in r.trades if lo <= abs(t.edge) < hi]
        if bt:
            bw = sum(1 for t in bt if t.outcome == "WIN")
            print(f"    ${lo:.2f}-${hi:.2f}: {len(bt):>4}, {bw/len(bt)*100:.1f}% WR, "
                  f"avg PnL ${sum(t.pnl for t in bt)/len(bt):+.4f}")


def _print_recent_trades(r: BacktestResult):
    print(f"\n  Last 15 trades:")
    print(f"  {'Time':<13} {'Dir':<6} {'Entry':<6} {'Model':<6} "
          f"{'Edge':<7} {'Kelly':<6} {'Size':<6} {'Result':<5} {'P&L':<9} {'Fee':<6} "
          f"{'RV':<5} {'IV':<5} {'Method'}")
    print(f"  {'-'*105}")
    for t in r.trades[-15:]:
        mv = t.model_yes_price if t.direction == "long" else t.model_no_price
        print(f"  {t.window_start.strftime('%m-%d %H:%M'):<13} "
              f"{t.direction:<6} {t.entry_price:<6.2f} {mv:<6.2f} "
              f"{t.edge:+6.3f} {t.kelly_fraction:5.1%} "
              f"${t.position_size:<5.2f} "
              f"{'W' if t.outcome == 'WIN' else 'L':<5} "
              f"${t.pnl:+8.4f} ${t.fee_paid:<5.4f} "
              f"{t.realized_vol:4.0%} {t.implied_vol:4.0%} {t.pricing_method}")


def print_results(result: BacktestResult):
    _print_summary_header(result)
    if not result.trades:
        print("\n  No trades taken."); return
    _print_direction_and_vol(result)
    _print_funding_and_edge(result)
    _print_recent_trades(result)
    print("\n  NOTES:")
    print("  • Market prices are synthetic | Fees: correct nonlinear formula")
    print("  • Sizing: half-Kelly, 5% cap | V3.1: vol skew + funding bias\n")


# ── Export ───────────────────────────────────────────────────────────

def export_results(result: BacktestResult, output_path: str):
    data = {
        "summary": {
            k: getattr(result, k) for k in [
                "strategy_mode", "start_date", "end_date", "total_windows",
                "trades_taken", "trades_skipped", "wins", "losses", "win_rate",
                "total_pnl", "total_fees", "max_drawdown", "profit_factor",
                "avg_pnl_per_trade", "avg_edge_per_trade", "avg_kelly",
                "sharpe_ratio", "bankroll_final", "bankroll_growth"]},
        "trades": [
            {k: (getattr(t, k).isoformat() if k == "window_start" else getattr(t, k))
             for k in BacktestTrade.__dataclass_fields__}
            for t in result.trades],
        "equity_curve": result.equity_curve,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results exported to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Backtest V3.1 — BTC 15-Min Quant Strategy")
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--csv", type=str)
    p.add_argument("--decision-minute", type=int, default=2)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--output", "-o", type=str)
    p.add_argument("--no-confirmation", action="store_true")
    p.add_argument("--taker", action="store_true")
    p.add_argument("--no-funding", action="store_true")
    p.add_argument("--realistic", action="store_true")
    p.add_argument("--spread", type=float, default=None)
    p.add_argument("--fill-rate", type=float, default=None)
    return p.parse_args()


def build_realistic_config(args):
    import random
    rc = RealisticConfig(enabled=args.realistic)
    if args.spread is not None: rc.spread_cents = args.spread
    if args.fill_rate is not None: rc.fill_rate = args.fill_rate
    if rc.enabled: random.seed(42)
    return rc


def print_config_banner(args, rc, use_maker, use_funding):
    label = "V3.1 REALISTIC" if rc.enabled else "V3.1"
    print("=" * 90)
    print(f"  POLYMARKET BTC 15-MIN — {label} QUANT BACKTESTER")
    print("=" * 90)
    print(f"  Fees: {'Maker 0%' if use_maker else 'Taker (max 1.56%)'} | "
          f"Min edge: ${rc.min_edge_realistic if rc.enabled else MIN_EDGE_CENTS:.2f} | "
          f"Vol: {VOL_METHOD}")
    print(f"  Confirmation: {'ON' if not args.no_confirmation else 'OFF'} | "
          f"Funding: {'ON' if use_funding else 'OFF'} | "
          f"Decision min: {args.decision_minute}")
    if rc.enabled:
        print(f"  Realistic: spread={rc.spread_cents*100:.0f}¢ | "
              f"fill={rc.fill_rate:.0%} | noise=±{rc.market_noise_std*100:.1f}¢ | "
              f"eff={rc.market_efficiency:.0%}")
    print()


def load_candles(args):
    if args.csv: return load_csv_candles(args.csv)
    if args.start and args.end:
        s = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        e = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        e = datetime.now(timezone.utc)
        s = e - timedelta(days=args.days)
    return fetch_binance_klines(s, e)


def load_funding(candles, use_funding):
    if not use_funding: return None, use_funding
    try:
        snaps = fetch_funding_rates(candles[0].timestamp, candles[-1].timestamp)
        if snaps:
            rc = {}
            for s in snaps: rc[s.classification] = rc.get(s.classification, 0) + 1
            print(f"  Funding: {len(snaps)} snapshots — " +
                  ", ".join(f"{k}: {v}" for k, v in sorted(rc.items())))
            print()
        return snaps, use_funding
    except Exception as ex:
        print(f"  WARNING: Funding rates unavailable: {ex}")
        return None, False

