#!/usr/bin/env python3
"""
Test Script for Phase 4: Strategy Brain

Tests:
1. Signal Processors (Spike, Sentiment, Divergence)
2. Signal Fusion Engine
3. 15-Minute BTC Strategy
4. End-to-End Signal Flow

Run this after Phase 3 tests pass.
"""
import asyncio
from decimal import Decimal
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger


import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.strategy_brain.signal_processors.spike_detector import SpikeDetectionProcessor
from core.strategy_brain.signal_processors.sentiment_processor import SentimentProcessor
from core.strategy_brain.signal_processors.divergence_processor import PriceDivergenceProcessor
from core.strategy_brain.fusion_engine.signal_fusion import get_fusion_engine
from core.strategy_brain.strategies.btc_15min_strategy import get_btc_strategy

app = typer.Typer()
console = Console()


def _test_processor(name, proc, current, history, metadata=None):
    """Test a single signal processor and print results."""
    console.print(f"\nTesting {name}...", end="")
    sig = proc.process(current, history, metadata) if metadata else proc.process(current, history)
    if sig:
        console.print(f" [green]✓ {sig.direction.value} conf={sig.confidence:.2%} score={sig.score:.1f}[/green]")
    else:
        console.print(" [yellow]⚠ No signal[/yellow]")
    return sig


async def test_signal_processors():
    """Test all signal processors."""
    console.print("\n[cyan]═══ Testing Signal Processors ═══[/cyan]")
    try:
        spike = SpikeDetectionProcessor()
        sentiment = SentimentProcessor()
        divergence = PriceDivergenceProcessor()
        console.print("✓ All processors initialized")
        history = [Decimal(str(65000 + i*10)) for i in range(-20, 0)]
        _test_processor("Spike Detector", spike, Decimal("70000"), history)
        _test_processor("Sentiment", sentiment, Decimal("70000"), history, {"sentiment_score": 15.0})
        _test_processor("Divergence", divergence, Decimal("70000"), history, {"spot_price": 67000.0})
        console.print("\n[green]✓ Signal Processors passed[/green]"); return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False


def _make_test_signals():
    """Create test trading signals for fusion."""
    from strategy_brain.signal_processors.base_processor import (
        TradingSignal, SignalType, SignalDirection, SignalStrength)
    return [
        TradingSignal(timestamp=datetime.now(), source="SpikeDetection",
                      signal_type=SignalType.SPIKE_DETECTED, direction=SignalDirection.BULLISH,
                      strength=SignalStrength.STRONG, confidence=0.80, current_price=Decimal("65000")),
        TradingSignal(timestamp=datetime.now(), source="SentimentAnalysis",
                      signal_type=SignalType.SENTIMENT_SHIFT, direction=SignalDirection.BULLISH,
                      strength=SignalStrength.MODERATE, confidence=0.70, current_price=Decimal("65000"))]


async def test_fusion_engine():
    """Test signal fusion engine."""
    console.print("\n[cyan]═══ Testing Fusion Engine ═══[/cyan]")
    try:
        fusion = get_fusion_engine()
        signals = _make_test_signals()
        console.print(f"Fusing {len(signals)} signals...", end="")
        fused = fusion.fuse_signals(signals)
        if not fused: console.print(" [red]✗[/red]"); return False
        console.print(f" [green]✓ {fused.direction.value} score={fused.score:.1f} "
                      f"conf={fused.confidence:.2%} actionable={fused.is_actionable}[/green]")
        console.print(f"  Total fusions: {fusion.get_statistics()['total_fusions']}")
        console.print("[green]✓ Fusion Engine passed[/green]"); return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False


async def test_btc_strategy():
    """Test 15-minute BTC strategy."""
    console.print("\n[cyan]═══ Testing BTC Strategy ═══[/cyan]")
    try:
        strat = get_btc_strategy()
        console.print("Simulating data...", end="")
        for i in range(30):
            strat.update_market_data(Decimal(str(65000+i*50)), Decimal(str(65000+i*50)), 20.0)
        strat.update_market_data(Decimal("75000"), Decimal("67000"), 85.0)
        console.print(" [green]✓[/green]")
        sigs = strat._process_signals()
        console.print(f"  Signals: {len(sigs)}")
        for s in sigs: console.print(f"  • {s.source}: {s.direction.value} ({s.score:.1f})")
        st = strat.get_statistics()
        console.print(f"  processed={st['signals_processed']} trades={st['trades_executed']}")
        console.print("[green]✓ BTC Strategy passed[/green]"); return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False


def _print_summary(results, next_phase="5"):
    """Print test summary table."""
    console.print("\n" + "=" * 60 + "\n[bold]SUMMARY[/bold]\n" + "=" * 60)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Status", width=15)
    for n, ok in results.items():
        table.add_row(n, "[green]✓[/green]" if ok else "[red]✗[/red]")
    console.print(table)
    if all(results.values()):
        console.print(f"[bold green]✓ ALL PASSED → Phase {next_phase}[/bold green]")
    else:
        console.print("[bold red]✗ SOME FAILED[/bold red]")


async def run_all_tests():
    """Run all Phase 4 tests."""
    console.print(Panel.fit("[bold cyan]PHASE 4: STRATEGY BRAIN TESTS[/bold cyan]", border_style="cyan"))
    results = {"Processors": await test_signal_processors(),
               "Fusion": await test_fusion_engine(), "BTC Strategy": await test_btc_strategy()}
    _print_summary(results, "5")
    return 0 if all(results.values()) else 1


@app.command()
def test(
    component: str = typer.Option(
        "all",
        "--component",
        "-c",
        help="Test specific component: all, processors, fusion, strategy"
    )
):
    """
    Test Strategy Brain components.
    
    Example:
        python scripts/test_strategy.py test
        python scripts/test_strategy.py test --component processors
    """
    async def run_specific_test():
        if component == "all":
            return await run_all_tests()
        elif component == "processors":
            return 0 if await test_signal_processors() else 1
        elif component == "fusion":
            return 0 if await test_fusion_engine() else 1
        elif component == "strategy":
            return 0 if await test_btc_strategy() else 1
        else:
            console.print(f"[red]Unknown component: {component}[/red]")
            return 1
    
    exit_code = asyncio.run(run_specific_test())
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()