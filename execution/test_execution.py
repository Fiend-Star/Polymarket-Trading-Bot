#!/usr/bin/env python3
"""
Test Script for Phase 5: Execution Layer

Tests:
1. Risk Engine
2. Execution Engine
3. Order Placement and Fills
4. Position Management
5. Stop Loss / Take Profit

Run this after Phase 4 tests pass.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from execution.risk_engine import get_risk_engine, RiskLimits
from execution.execution_engine import get_execution_engine
from execution.polymarket_client import get_polymarket_client
from core.strategy_brain.signal_processors.base_processor import SignalDirection

app = typer.Typer()
console = Console()


async def _test_risk_validation(risk):
    """Test risk position validation."""
    console.print("Testing validation...", end="")
    ok, _ = risk.validate_new_position(Decimal("50.0"), "long", Decimal("65000"))
    if not ok: console.print(" [red]✗[/red]"); return False
    ok2, err = risk.validate_new_position(Decimal("500.0"), "long", Decimal("65000"))
    if ok2: console.print(" [red]✗ should reject oversized[/red]"); return False
    console.print(f" [green]✓ valid accepted, oversized rejected[/green]")
    return True

async def _test_risk_position(risk):
    """Test position tracking and summary."""
    console.print("Testing position tracking...", end="")
    risk.add_position("test_pos_1", Decimal("50.0"), Decimal("65000"), "long",
                      Decimal("58500"), Decimal("71500"))
    rp = risk.update_position("test_pos_1", Decimal("67000"))
    if rp: console.print(f" [green]✓ PnL=${rp.unrealized_pnl:+.2f}[/green]")
    s = risk.get_risk_summary()
    console.print(f"  positions={s['positions']['count']} exp=${s['exposure']['current']:.2f}")
    return True

async def test_risk_engine():
    """Test risk management engine."""
    console.print("\n[cyan]═══ Testing Risk Engine ═══[/cyan]")
    try:
        risk = get_risk_engine()
        if not await _test_risk_validation(risk): return False
        console.print("Testing size calc...", end="")
        sz = risk.calculate_position_size(0.80, 75.0, Decimal("65000"))
        console.print(f" [green]✓ ${sz:.2f}[/green]")
        if not await _test_risk_position(risk): return False
        console.print("[green]✓ Risk Engine passed[/green]"); return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False

async def _test_exec_signal(execution):
    """Execute a test signal."""
    console.print("Executing bullish signal...", end="")
    order = await execution.execute_signal(
        signal_direction=SignalDirection.BULLISH, signal_confidence=0.75,
        signal_score=80.0, current_price=Decimal("65000"),
        stop_loss=Decimal("58500"), take_profit=Decimal("71500"))
    if not order: console.print(" [red]✗[/red]"); return None
    console.print(f" [green]✓ {order.order_id} ${order.size:.2f}[/green]")
    return order

async def _test_exec_positions(execution):
    """Test position management."""
    await asyncio.sleep(1)
    positions = execution.get_open_positions()
    console.print(f"  Positions: {len(positions)}")
    await execution.update_positions(Decimal("67000"))
    if positions:
        pnl = await execution.close_position(positions[0]["position_id"], Decimal("67000"), "test")
        console.print(f"  Closed PnL=${pnl:+.2f}" if pnl is not None else "  Close failed")

async def test_execution_engine():
    """Test execution engine."""
    console.print("\n[cyan]═══ Testing Execution Engine ═══[/cyan]")
    try:
        ex = get_execution_engine()
        if not await _test_exec_signal(ex): return False
        await _test_exec_positions(ex)
        stats = ex.get_statistics()
        console.print(f"  orders={stats['orders']['total']} filled={stats['orders']['filled']}")
        console.print("[green]✓ Execution Engine passed[/green]"); return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False

async def test_polymarket_client():
    """Test Polymarket client placeholder."""
    console.print("\n[cyan]═══ Testing Polymarket Client ═══[/cyan]")
    try:
        c = get_polymarket_client()
        console.print("Connecting...", end="")
        if not await c.connect(): console.print(" [red]✗[/red]"); return False
        console.print(" [green]✓[/green]")
        p = await c.get_market_price("btc_market_test")
        if p: console.print(f"  Price: {p:.4f}")
        b = await c.get_balance()
        if b: console.print(f"  USDC: ${b.get('USDC', 0):,.2f}")
        await c.disconnect()
        console.print("[green]✓ Polymarket Client passed (placeholder)[/green]"); return True
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False

async def run_all_tests():
    """Run all Phase 5 tests."""
    console.print(Panel.fit("[bold cyan]PHASE 5: EXECUTION TESTS[/bold cyan]", border_style="cyan"))
    results = {"Risk": await test_risk_engine(), "Execution": await test_execution_engine(),
               "Polymarket": await test_polymarket_client()}
    console.print("\n" + "=" * 60 + "\n[bold]SUMMARY[/bold]\n" + "=" * 60)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Status", width=15)
    for n, ok in results.items():
        table.add_row(n, "[green]✓[/green]" if ok else "[red]✗[/red]")
    console.print(table)
    if all(results.values()):
        console.print("[bold green]✓ ALL PASSED → Phase 6[/bold green]")
    else:
        console.print("[bold red]✗ SOME FAILED[/bold red]")
    return 0 if all(results.values()) else 1


@app.command()
def test(
    component: str = typer.Option(
        "all",
        "--component",
        "-c",
        help="Test specific component: all, risk, execution, polymarket"
    )
):
    """
    Test Execution Layer components.
    
    Example:
        python scripts/test_execution.py test
        python scripts/test_execution.py test --component risk
    """
    async def run_specific_test():
        if component == "all":
            return await run_all_tests()
        elif component == "risk":
            return 0 if await test_risk_engine() else 1
        elif component == "execution":
            return 0 if await test_execution_engine() else 1
        elif component == "polymarket":
            return 0 if await test_polymarket_client() else 1
        else:
            console.print(f"[red]Unknown component: {component}[/red]")
            return 1
    
    exit_code = asyncio.run(run_specific_test())
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()