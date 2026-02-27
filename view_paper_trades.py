"""
Paper Trading Viewer
View and analyze simulation trades
"""
import json
from datetime import datetime
from pathlib import Path


def load_paper_trades():
    """Load paper trades from file."""
    try:
        with open('paper_trades.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("No paper trades file found.")
        return []
    except Exception as e:
        print(f"Error loading paper trades: {e}")
        return []


def _print_trade_summary(trades):
    """Print summary statistics for paper trades."""
    total = len(trades)
    wins = sum(1 for t in trades if t.get('outcome') == 'WIN')
    losses = sum(1 for t in trades if t.get('outcome') == 'LOSS')
    pending = sum(1 for t in trades if t.get('outcome') == 'PENDING')
    print(f"Total Trades: {total} | Wins: {wins} | Losses: {losses} | Pending: {pending}")
    if wins + losses > 0:
        print(f"Win Rate: {wins / (wins + losses) * 100:.1f}%")

def _print_trade_table(trades):
    """Print trade detail table."""
    print("-" * 100)
    print(f"{'#':<4} {'Time':<20} {'Direction':<10} {'Size':<12} {'Price':<12} {'Score':<8} {'Confidence':<12} {'Outcome':<10}")
    print("-" * 100)
    for i, t in enumerate(trades, 1):
        ts = datetime.fromisoformat(t['timestamp']).strftime('%Y-%m-%d %H:%M')
        print(f"{i:<4} {ts:<20} {t['direction']:<10} ${t['size_usd']:<11.2f} ${t['price']:<11,.2f} "
              f"{t['signal_score']:<7.1f} {t['signal_confidence']:<11.1%} {t.get('outcome', 'PENDING'):<10}")
    print("-" * 100)

def display_paper_trades(trades):
    """Display paper trades in a nice format."""
    if not trades:
        print("\nNo paper trades recorded yet."); return
    print("\n" + "=" * 100)
    print("PAPER TRADING RESULTS (SIMULATION)")
    print("=" * 100 + "\n")
    _print_trade_summary(trades)
    print()
    _print_trade_table(trades)
    print()


def main():
    """Main entry point."""
    trades = load_paper_trades()
    display_paper_trades(trades)
    
    if trades:
        print("\nNOTE: These are SIMULATION trades only - no real money involved!")
        print("To update outcomes, edit paper_trades.json manually")
        print()


if __name__ == "__main__":
    main()