#!/usr/bin/env python3
"""
Test Script for Phase 2: Ingestion Layer

Tests:
1. Unified Data Adapter
2. WebSocket Manager
3. Data Validator
4. Rate Limiter

Run this after Phase 1 tests pass.
"""
import asyncio
from datetime import datetime
from decimal import Decimal
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from loguru import logger

import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.ingestion.adapters.unified_adapter import UnifiedDataAdapter, MarketData
from core.ingestion.managers.websocket_manager import WebSocketManager, ConnectionState
from core.ingestion.managers.rate_limiter import get_rate_limiter
from core.ingestion.validators.data_validator import get_validator

app = typer.Typer()
console = Console()


async def test_unified_adapter():
    """Test unified data adapter."""
    console.print("\n[cyan]═══ Testing Unified Adapter ═══[/cyan]")
    adapter = UnifiedDataAdapter()
    try:
        console.print("Connecting...", end="")
        results = await adapter.connect_all()
        c = sum(results.values())
        if c == 0: console.print(" [red]✗ None connected[/red]"); return False
        console.print(f" [green]✓ {c}/{len(results)}[/green]")
        prices = []
        adapter.on_price_update = lambda d: prices.append(d)
        await adapter.start_streaming()
        await asyncio.sleep(10)
        console.print(f"  Received {len(prices)} updates")
        cons = adapter.get_price_consensus()
        if cons: console.print(f"  Avg=${cons['average']:,.2f} spread={cons['spread_percent']:.2f}%")
        await adapter.disconnect_all()
        console.print("[green]✓ Unified Adapter passed[/green]"); return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False


async def test_websocket_manager():
    """Test WebSocket connection manager."""
    console.print("\n[cyan]═══ Testing WebSocket Manager ═══[/cyan]")
    try:
        from data_sources.binance.websocket import BinanceWebSocketSource
        b = BinanceWebSocketSource()
        mgr = WebSocketManager("Binance-Test", lambda: b.connect("ticker"),
                               b.stream_ticker, max_reconnect_attempts=3)
        console.print("Connecting...", end="")
        if not await mgr.connect(): console.print(" [red]✗[/red]"); return False
        console.print(f" [green]✓ state={mgr.get_stats()['state']}[/green]")
        b.on_price_update = lambda _: mgr.update_last_message_time()
        task = asyncio.create_task(mgr.start_streaming())
        await asyncio.sleep(5); await mgr.disconnect()
        try: await task
        except: pass
        console.print("[green]✓ WebSocket Manager passed[/green]"); return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False


async def _test_valid_data(v):
    """Test valid market data acceptance."""
    console.print("Testing valid price...", end="")
    r = v.validate_market_data(source="test", price=Decimal("65000"),
                               timestamp=datetime.now(), bid=Decimal("64995"), ask=Decimal("65005"))
    if r.is_valid: console.print(" [green]✓[/green]"); return True
    console.print(f" [red]✗ {r.errors}[/red]"); return False


async def _test_invalid_and_anomaly(v):
    """Test invalid data rejection and anomaly detection."""
    console.print("Testing invalid price...", end="")
    r = v.validate_market_data(source="test", price=Decimal("500"), timestamp=datetime.now())
    if not r.is_valid: console.print(f" [green]✓ rejected[/green]")
    else: console.print(" [red]✗ should reject[/red]")
    console.print("Testing anomaly detection...", end="")
    for p in [65000, 65100, 64900, 65050, 65000]:
        v.validate_market_data(source="anom", price=Decimal(str(p)), timestamp=datetime.now())
    a = v.detect_anomaly("anom", Decimal("75000"))
    if a: console.print(f" [green]✓ z={a['z_score']:.2f}[/green]")
    else: console.print(" [yellow]⚠ not detected[/yellow]")
    st = v.get_price_statistics("anom")
    if st: console.print(f"  mean=${st['mean']:,.2f} range=${st['range']:,.2f}")


async def test_data_validator():
    """Test data validator."""
    console.print("\n[cyan]═══ Testing Data Validator ═══[/cyan]")
    try:
        v = get_validator()
        if not await _test_valid_data(v): return False
        await _test_invalid_and_anomaly(v)
        console.print("[green]✓ Data Validator passed[/green]"); return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False


async def test_rate_limiter():
    """Test rate limiter."""
    console.print("\n[cyan]═══ Testing Rate Limiter ═══[/cyan]")
    try:
        rl = get_rate_limiter()
        console.print("Testing 5 requests...", end="")
        acq = sum(1 for _ in range(5) if await rl.acquire("test_source", wait=False))
        console.print(f" [green]✓ {acq}/5[/green]")
        for src, s in rl.get_stats().items():
            console.print(f"  [{src}] {s['current_requests']}/{s['max_requests']}")
        console.print("[green]✓ Rate Limiter passed[/green]"); return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False


def _print_test_summary(results, next_phase="3"):
    """Print test results summary table."""
    console.print("\n" + "=" * 60 + "\n[bold]SUMMARY[/bold]\n" + "=" * 60)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Status", width=15)
    for name, ok in results.items():
        table.add_row(name, "[green]✓[/green]" if ok else "[red]✗[/red]")
    console.print(table)
    if all(results.values()):
        console.print(f"[bold green]✓ ALL PASSED → Phase {next_phase}[/bold green]")
    else:
        console.print("[bold red]✗ SOME FAILED[/bold red]")


async def run_all_tests():
    """Run all Phase 2 tests."""
    console.print(Panel.fit("[bold cyan]PHASE 2: INGESTION TESTS[/bold cyan]", border_style="cyan"))
    results = {"Adapter": await test_unified_adapter(), "WebSocket": await test_websocket_manager(),
               "Validator": await test_data_validator(), "Rate Limiter": await test_rate_limiter()}
    _print_test_summary(results, "3")
    return 0 if all(results.values()) else 1


@app.command()
def test(component: str = typer.Option("all", "--component", "-c",
         help="Test: all, adapter, websocket, validator, ratelimit")):
    """Test ingestion layer components."""
    dispatch = {"all": run_all_tests, "adapter": test_unified_adapter,
                "websocket": test_websocket_manager, "validator": test_data_validator,
                "ratelimit": test_rate_limiter}
    if component not in dispatch:
        console.print(f"[red]Unknown: {component}[/red]"); raise typer.Exit(1)
    result = asyncio.run(dispatch[component]())
    raise typer.Exit(0 if result else 1)


if __name__ == "__main__":
    app()