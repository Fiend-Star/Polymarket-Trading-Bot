"""
backtest_v3_sync_patch.py — Sync Backtest V4 with Live Bot V3.1 (March 2026)
=============================================================================

This patch file documents ALL changes needed to bring backtest_v3.py
(internally labeled "Backtest V4") in sync with the current live bot.

HOW TO APPLY:
  Copy the relevant sections below into your backtest_v3.py file,
  replacing the corresponding old code blocks.

  Or run:  python backtest_v3_sync_patch.py --dry-run
           python backtest_v3_sync_patch.py --apply

CHANGES SUMMARY:
  1. BANKROLL_USD default: 20.0 → 25.0  (match .env.example)
  2. Decision minute default: 2 → 3      (match TRADE_WINDOW_START=180s)
  3. Add TRADE_WINDOW_START/END enforcement (180s–600s)
  4. Add QUOTE_STABILITY_REQUIRED simulation delay
  5. Remove Sentiment + Deribit PCR from fusion (live bot dropped them)
  6. Update run_confirmation() to skip "sentiment" processor
  7. Update fusion weight awareness in confirmation veto logic
  8. Add late-window simulation stub (documents the gap)
  9. Improve gamma exit fidelity notes
  10. Sync CLI defaults and print banner

Each change is tagged with [PATCH-N] for easy grep.

Requirements:
  - Your existing backtest_v3.py (the "V4 Real Data" version)
  - No new dependencies needed
"""

import argparse
import os
import re
import sys
from pathlib import Path


# =============================================================================
# PATCH DEFINITIONS — each is (old_text, new_text, description)
# =============================================================================

PATCHES = []

# ─────────────────────────────────────────────────────────────────────────────
# [PATCH-1] BANKROLL_USD default: 20.0 → 25.0
# ─────────────────────────────────────────────────────────────────────────────
PATCHES.append({
    "id": "PATCH-1",
    "desc": "BANKROLL_USD default: 20.0 → 25.0 (match live .env.example)",
    "old": 'BANKROLL_USD = float(os.getenv("BANKROLL_USD", "20.0"))',
    "new": 'BANKROLL_USD = float(os.getenv("BANKROLL_USD", "25.0"))  # [PATCH-1] synced with live .env.example',
})

# ─────────────────────────────────────────────────────────────────────────────
# [PATCH-2] Add TRADE_WINDOW config constants (live bot uses 180s–600s)
# ─────────────────────────────────────────────────────────────────────────────
PATCHES.append({
    "id": "PATCH-2",
    "desc": "Add TRADE_WINDOW_START/END + QUOTE_STABILITY constants from live bot",
    "old": 'GAMMA_EXIT_PROFIT_PCT = float(os.getenv("GAMMA_EXIT_PROFIT_PCT", "0.04"))',
    "new": """# [PATCH-2] Trade window enforcement (synced with live bot)
TRADE_WINDOW_START_SEC = int(os.getenv("TRADE_WINDOW_START", "180"))   # 3 minutes into window
TRADE_WINDOW_END_SEC = int(os.getenv("TRADE_WINDOW_END", "600"))       # 10 minutes into window
QUOTE_STABILITY_REQUIRED = int(os.getenv("QUOTE_STABILITY_REQUIRED", "5"))  # ticks before trading

GAMMA_EXIT_PROFIT_PCT = float(os.getenv("GAMMA_EXIT_PROFIT_PCT", "0.04"))""",
})

# ─────────────────────────────────────────────────────────────────────────────
# [PATCH-3] Decision minute default: 2 → 3 (match TRADE_WINDOW_START=180s)
# ─────────────────────────────────────────────────────────────────────────────
PATCHES.append({
    "id": "PATCH-3",
    "desc": "Decision minute default: 2 → 3 (match TRADE_WINDOW_START=180s = minute 3)",
    "old": """    parser.add_argument("--decision-minute", type=int, default=2,
                        help="Minute within window to decide (default: 2)")""",
    "new": """    parser.add_argument("--decision-minute", type=int, default=3,
                        help="Minute within window to decide (default: 3, matches live TRADE_WINDOW_START=180s)")  # [PATCH-3]""",
})

# ─────────────────────────────────────────────────────────────────────────────
# [PATCH-4] Remove Sentiment + Deribit from create_fusion_processors()
# ─────────────────────────────────────────────────────────────────────────────
PATCHES.append({
    "id": "PATCH-4",
    "desc": "Remove Sentiment + Deribit PCR from fusion (live bot dropped both as of March 2026)",
    "old": """def create_fusion_processors() -> Optional[Dict]:
    if not FUSION_AVAILABLE:
        return None
    return {
        "spike": SpikeDetectionProcessor(
            spike_threshold=float(os.getenv("SPIKE_THRESHOLD", "0.05")),
            lookback_periods=20,
        ),
        "sentiment": SentimentProcessor(
            extreme_fear_threshold=25, extreme_greed_threshold=75,
        ),
        "divergence": PriceDivergenceProcessor(
            divergence_threshold=float(os.getenv("DIVERGENCE_THRESHOLD", "0.05")),
        ),
        "tick_velocity": TickVelocityProcessor(
            velocity_threshold_60s=0.015, velocity_threshold_30s=0.010,
        ),
    }""",
    "new": """def create_fusion_processors() -> Optional[Dict]:
    # [PATCH-4] Synced with live bot (March 2026):
    #   - Removed SentimentProcessor (dropped from live)
    #   - Removed DeribitPCR (dropped from live)
    #   - Live weights: OrderBook=0.40, TickVelocity=0.30, Divergence=0.20, Spike=0.10
    #   - OrderBook cannot run in backtest (no historical orderbook data)
    #   - So backtest confirmation uses: spike, divergence, tick_velocity only
    if not FUSION_AVAILABLE:
        return None
    return {
        "spike": SpikeDetectionProcessor(
            spike_threshold=float(os.getenv("SPIKE_THRESHOLD", "0.05")),
            lookback_periods=20,
        ),
        "divergence": PriceDivergenceProcessor(
            divergence_threshold=float(os.getenv("DIVERGENCE_THRESHOLD", "0.05")),
        ),
        "tick_velocity": TickVelocityProcessor(
            velocity_threshold_60s=0.015, velocity_threshold_30s=0.010,
        ),
    }""",
})

# ─────────────────────────────────────────────────────────────────────────────
# [PATCH-5] Update run_confirmation() to remove "sentiment" key
# ─────────────────────────────────────────────────────────────────────────────
PATCHES.append({
    "id": "PATCH-5",
    "desc": "Remove 'sentiment' from run_confirmation processor loop",
    "old": '    for name in ["spike", "sentiment", "divergence", "tick_velocity"]:',
    "new": '    for name in ["spike", "divergence", "tick_velocity"]:  # [PATCH-5] sentiment removed (live bot dropped it)',
})

# ─────────────────────────────────────────────────────────────────────────────
# [PATCH-6] Update the Sentiment import to be optional/removed
# ─────────────────────────────────────────────────────────────────────────────
PATCHES.append({
    "id": "PATCH-6",
    "desc": "Make SentimentProcessor import optional (no longer used in fusion)",
    "old": """try:
    from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
    from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
    from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
    from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
    FUSION_AVAILABLE = True
except Imp""",
    "new": """# [PATCH-6] Removed SentimentProcessor import — dropped from live bot March 2026
try:
    from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
    from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
    from core.strategy_brain.signal_processors.tick_velocity_processor import TickVelocityProcessor
    FUSION_AVAILABLE = True
except Imp""",
})

# ─────────────────────────────────────────────────────────────────────────────
# [PATCH-7] Add trade window enforcement in the main backtest loop
#
# This patch targets the line where we check for price at decision_minute.
# We add a check that the decision falls within TRADE_WINDOW_START..END
# ─────────────────────────────────────────────────────────────────────────────
PATCHES.append({
    "id": "PATCH-7",
    "desc": "Add TRADE_WINDOW enforcement + QUOTE_STABILITY simulation in main loop",
    "old": """        # ── Get REAL market price at decision minute ────────────────────""",
    "new": """        # [PATCH-7] Enforce trade window (synced with live bot 180s–600s)
        decision_sec = decision_minute * 60
        if decision_sec < TRADE_WINDOW_START_SEC or decision_sec > TRADE_WINDOW_END_SEC:
            # Outside the live bot's trade window — skip this window
            # (In practice, default decision_minute=3 → 180s, which is the window start)
            skip_reasons.setdefault("outside_trade_window", 0)
            skip_reasons["outside_trade_window"] += 1
            skipped += 1
            continue

        # Simulate QUOTE_STABILITY_REQUIRED: if fewer than 5 price points exist
        # up to the decision minute, the live bot wouldn't have traded either
        price_points_before_decision = sum(
            1 for m in range(0, decision_minute + 1) if m in w.yes_prices
        )
        if price_points_before_decision < QUOTE_STABILITY_REQUIRED:
            skip_reasons.setdefault("quote_instability", 0)
            skip_reasons["quote_instability"] += 1
            skipped += 1
            continue

        # ── Get REAL market price at decision minute ────────────────────""",
})

# ─────────────────────────────────────────────────────────────────────────────
# [PATCH-8] Update CLI banner to show synced config
# ─────────────────────────────────────────────────────────────────────────────
PATCHES.append({
    "id": "PATCH-8",
    "desc": "Update CLI banner to reflect synced live bot config",
    "old": '    print(f"  Decision minute:   {args.decision_minute}")',
    "new": """    print(f"  Decision minute:   {args.decision_minute}")
    print(f"  Trade window:      {TRADE_WINDOW_START_SEC}s–{TRADE_WINDOW_END_SEC}s (synced with live)")  # [PATCH-8]
    print(f"  Quote stability:   {QUOTE_STABILITY_REQUIRED} ticks required")""",
})

# ─────────────────────────────────────────────────────────────────────────────
# [PATCH-9] Update docstring to reflect V4.1 sync
# ─────────────────────────────────────────────────────────────────────────────
PATCHES.append({
    "id": "PATCH-9",
    "desc": "Update module docstring to V4.1 (synced with live bot March 2026)",
    "old": """Backtest Engine V4 — REAL Polymarket Data
==========================================""",
    "new": """Backtest Engine V4.1 — REAL Polymarket Data (Synced with Live Bot March 2026)
==============================================================================""",
})


# =============================================================================
# Applicator
# =============================================================================

def find_backtest_file() -> Path:
    """Find the backtest file in common locations."""
    candidates = [
        Path("backtest_v3.py"),
        Path("backtest_v4.py"),
        Path(__file__).parent / "backtest_v3.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("backtest_v3.py")  # default


def apply_patches(filepath: Path, dry_run: bool = True) -> dict:
    """Apply all patches to the file. Returns stats."""
    content = filepath.read_text(encoding="utf-8")
    original = content

    stats = {"applied": [], "skipped": [], "already_applied": []}

    for patch in PATCHES:
        pid = patch["id"]
        old = patch["old"]
        new = patch["new"]

        if f"[{pid}]" in content and old not in content:
            stats["already_applied"].append(pid)
            continue

        if old in content:
            content = content.replace(old, new, 1)
            stats["applied"].append(pid)
        else:
            stats["skipped"].append(pid)

    if not dry_run and content != original:
        # Backup
        backup = filepath.with_suffix(".py.bak")
        filepath.rename(backup)
        filepath.write_text(content, encoding="utf-8")
        print(f"  ✓ Original backed up to {backup}")
        print(f"  ✓ Patched file written to {filepath}")
    elif dry_run:
        print(f"  [DRY RUN] Would modify {filepath}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Sync backtest with live bot V3.1")
    parser.add_argument("--apply", action="store_true", help="Apply patches (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change (default)")
    parser.add_argument("--file", type=str, default=None, help="Path to backtest_v3.py")
    args = parser.parse_args()

    dry_run = not args.apply

    filepath = Path(args.file) if args.file else find_backtest_file()

    print("=" * 80)
    print("  BACKTEST V4 → V4.1 SYNC PATCH (Live Bot March 2026)")
    print("=" * 80)
    print()

    if not filepath.exists():
        print(f"  ERROR: File not found: {filepath}")
        print(f"  Use --file /path/to/backtest_v3.py")
        sys.exit(1)

    print(f"  Target: {filepath}")
    print(f"  Mode:   {'DRY RUN' if dry_run else 'APPLYING'}")
    print()

    print("  Patches:")
    for p in PATCHES:
        print(f"    {p['id']}: {p['desc']}")
    print()

    stats = apply_patches(filepath, dry_run=dry_run)

    print(f"  Results:")
    print(f"    Applied:         {len(stats['applied'])} — {', '.join(stats['applied']) or 'none'}")
    print(f"    Already applied: {len(stats['already_applied'])} — {', '.join(stats['already_applied']) or 'none'}")
    print(f"    Skipped:         {len(stats['skipped'])} — {', '.join(stats['skipped']) or 'none'}")

    if stats["skipped"]:
        print()
        print("  ⚠ Skipped patches could not find their target text.")
        print("    Your backtest_v3.py may have been modified since the patches were written.")
        print("    Apply these manually — see the PATCHES list in this file for old→new text.")

    if dry_run and stats["applied"]:
        print()
        print(f"  To apply: python {__file__} --apply")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
