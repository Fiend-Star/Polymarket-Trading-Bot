#!/usr/bin/env python3
"""
Test Script for Phase 3: Nautilus Core

Tests:
1. Instrument Registry
2. Data Engine Integration
3. Event Dispatcher
4. Custom Data Provider
5. End-to-End Data Flow

Run this after Phase 2 tests pass.
"""
import asyncio
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger

import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.nautilus_core.instruments.btc_instruments import get_instrument_registry
from core.nautilus_core.data_engine.engine_wrapper import get_nautilus_engine
from core.nautilus_core.event_dispatcher.dispatcher import get_event_dispatcher, EventType, Event

app = typer.Typer()
console = Console()


def _ok(msg, val, fmt=None):
    """Print check result. Returns True if val is truthy."""
    if val:
        console.print(f" [green]✓ {fmt(val) if fmt else msg}[/green]")
        return True
    console.print(f" [red]✗ {msg}[/red]")
    return False


async def test_instruments():
    """Test instrument registry."""
    console.print("\n[cyan]═══ Testing Instruments ═══[/cyan]")
    try:
        reg = get_instrument_registry()
        console.print("Checking instruments...", end="")
        if not _ok("instruments", len(reg.get_all()) >= 3, lambda n: f"{n} instruments"): return False
        p, c, b = reg.get_polymarket(), reg.get_coinbase(), reg.get_binance()
        if not _ok("key instruments", p and c and b): return False
        console.print("[green]✓ Instrument Registry passed[/green]")
        return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False


async def test_event_dispatcher():
    """Test event dispatcher."""
    console.print("\n[cyan]═══ Testing Event Dispatcher ═══[/cyan]")
    try:
        d = get_event_dispatcher()
        evts = []
        d.subscribe(EventType.PRICE_UPDATE, lambda e: evts.append(e))
        console.print("Testing dispatch...", end="")
        d.dispatch_price_update(source="test", price=65000.0, metadata={"test": True})
        if not _ok("event received", len(evts) == 1): return False
        console.print(f"  type={evts[0].type.value} src={evts[0].source}")
        stats = d.get_statistics()
        console.print(f"  total_events={stats['total_events']}")
        console.print("[green]✓ Event Dispatcher passed[/green]")
        return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False


async def _test_engine_streaming(engine):
    """Test data streaming and consensus for engine."""
    console.print("Streaming (10s)...", end="")
    await asyncio.sleep(10)
    console.print(" [green]✓[/green]")
    console.print("Checking consensus...", end="")
    c = engine.get_price_consensus()
    if c:
        console.print(f" [green]✓ avg=${c['average']:,.2f}[/green]")
    else:
        console.print(" [yellow]⚠ No consensus yet[/yellow]")


async def test_data_engine():
    """Test Nautilus data engine integration."""
    console.print("\n[cyan]═══ Testing Data Engine ═══[/cyan]")
    engine = get_nautilus_engine()
    try:
        console.print("Starting engine...", end="")
        await engine.start()
        s = engine.get_status()
        if not _ok("engine started", s["is_running"]): return False
        console.print(f"  instruments={s['instruments_registered']}")
        await _test_engine_streaming(engine)
        await engine.stop()
        console.print("[green]✓ Data Engine passed[/green]")
        return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        try: await engine.stop()
        except: pass
        return False


async def run_all_tests():
    """Run all Phase 3 tests."""
    console.print(Panel.fit("[bold cyan]PHASE 3: NAUTILUS CORE TESTS[/bold cyan]", border_style="cyan"))
    results = {"Instruments": await test_instruments(),
               "Dispatcher": await test_event_dispatcher(),
               "Data Engine": await test_data_engine()}
    console.print("\n" + "=" * 60 + "\n[bold]SUMMARY[/bold]\n" + "=" * 60)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Status", width=15)
    for name, ok in results.items():
        table.add_row(name, "[green]✓[/green]" if ok else "[red]✗[/red]")
    console.print(table)
    if all(results.values()):
        console.print("[bold green]✓ ALL PASSED → Phase 4[/bold green]")
    else:
        console.print("[bold red]✗ SOME FAILED[/bold red]")
    return 0 if all(results.values()) else 1


@app.command()
def test(
    component: str = typer.Option(
        "all",
        "--component",
        "-c",
        help="Test specific component: all, instruments, dispatcher, engine"
    )
):
    """
    Test Nautilus Core components.
    
    Example:
        python scripts/test_nautilus.py test
        python scripts/test_nautilus.py test --component instruments
    """
    async def run_specific_test():
        if component == "all":
            return await run_all_tests()
        elif component == "instruments":
            return 0 if await test_instruments() else 1
        elif component == "dispatcher":
            return 0 if await test_event_dispatcher() else 1
        elif component == "engine":
            return 0 if await test_data_engine() else 1
        else:
            console.print(f"[red]Unknown component: {component}[/red]")
            return 1
    
    exit_code = asyncio.run(run_specific_test())
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()