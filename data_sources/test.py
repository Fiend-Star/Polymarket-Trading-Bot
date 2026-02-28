import asyncio
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from loguru import logger
import os
# Import data sources
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_sources.coinbase.adapter import CoinbaseDataSource
from data_sources.binance.websocket import BinanceWebSocketSource
from data_sources.news_social.adapter import NewsSocialDataSource
from data_sources.solana.rpc import SolanaRPCDataSource

app = typer.Typer()
console = Console()


def _check(label, value, fmt=None):
    """Print test check result."""
    if value:
        display = fmt(value) if fmt else "OK"
        console.print(f" [green]✓ {display}[/green]")
        return True
    console.print(f" [red]✗ FAILED[/red]")
    return False


async def _test_source(name, source, tests):
    """Generic test runner for a data source. tests = list of (label, coro, fmt)."""
    console.print(f"\n[cyan]═══ Testing {name} ═══[/cyan]")
    try:
        console.print("Connecting...", end="")
        if not await source.connect():
            console.print(" [red]✗ FAILED[/red]"); return False
        console.print(" [green]✓ Connected[/green]")
        for label, coro, fmt in tests:
            console.print(f"{label}...", end="")
            if not _check(label, await coro, fmt):
                return False
        await source.disconnect()
        console.print(f"[green]✓ {name} - All tests passed![/green]")
        return True
    except Exception as e:
        console.print(f"\n[red]Error testing {name}: {e}[/red]"); return False


async def test_coinbase():
    """Test Coinbase API data source."""
    s = CoinbaseDataSource()
    return await _test_source("Coinbase API", s, [
        ("Fetching BTC price", s.get_current_price(), lambda p: f"${p:,.2f}"),
        ("Fetching 24h stats", s.get_24h_stats(), lambda st: f"Vol: ${st['volume']:,.2f}"),
        ("Fetching order book", s.get_order_book(level=1),
         lambda b: f"Bid/Ask: ${b['bids'][0]['price']:,.2f}/{b['asks'][0]['price']:,.2f}" if b.get('bids') else None),
        ("Fetching trades", s.get_recent_trades(limit=5), lambda t: f"Got {len(t)} trades")])


async def test_binance():
    """Test Binance WebSocket data source."""
    console.print("\n[cyan]═══ Testing Binance WebSocket ═══[/cyan]")
    source = BinanceWebSocketSource()
    try:
        console.print("Starting ticker stream (5s)...", end="")
        prices = []
        source.on_price_update = lambda t: prices.append(t["price"])
        task = asyncio.create_task(source.stream_ticker())
        await asyncio.sleep(5)
        await source.disconnect()
        try: await task
        except: pass
        if prices:
            console.print(f"\n[green]✓ {len(prices)} updates, latest ${prices[-1]:,.2f}[/green]")
            return True
        console.print("\n[red]✗ No updates[/red]"); return False
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]"); return False


async def test_news_social():
    """Test News/Social data source."""
    s = NewsSocialDataSource()
    return await _test_source("News/Social APIs", s, [
        ("Fetching F&G Index", s.get_fear_greed_index(),
         lambda d: f"{d['value']} - {d['classification']}"),
        ("Calculating sentiment", s.get_sentiment_score(),
         lambda sc: f"Score: {sc:.1f}/100" if sc is not None else None)])


async def test_solana():
    """Test Solana RPC data source."""
    s = SolanaRPCDataSource()
    return await _test_source("Solana RPC", s, [
        ("Fetching slot", s.get_slot(), lambda sl: f"Slot: {sl:,}"),
        ("Fetching network stats", s.get_network_stats(),
         lambda st: f"TPS: {st['tps']:.1f}")])


async def run_all_tests():
    """Run all data source tests."""
    console.print(Panel.fit("[bold cyan]PHASE 1: DATA SOURCES TEST[/bold cyan]", border_style="cyan"))
    results = {"Coinbase": await test_coinbase(), "Binance": await test_binance(),
               "News/Social": await test_news_social(), "Solana": await test_solana()}
    _print_summary(results)
    return 0 if all(results.values()) else 1


def _print_summary(results):
    """Print test summary table."""
    console.print("\n" + "=" * 60 + "\n[bold]TEST SUMMARY[/bold]\n" + "=" * 60)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Source", style="cyan", width=20)
    table.add_column("Status", width=15)
    for src, ok in results.items():
        table.add_row(src, "[green]✓ PASSED[/green]" if ok else "[red]✗ FAILED[/red]")
    console.print(table)
    if all(results.values()):
        console.print("[bold green]✓ ALL PASSED — Ready for Phase 2[/bold green]")
    else:
        console.print("[bold red]✗ SOME FAILED[/bold red]")


@app.command()
def test(source: str = typer.Option("all", "--source", "-s",
         help="Test: all, coinbase, binance, news, solana")):
    """Test external data sources."""
    dispatch = {"all": run_all_tests, "coinbase": test_coinbase,
                "binance": test_binance, "news": test_news_social, "solana": test_solana}
    if source not in dispatch:
        console.print(f"[red]Unknown source: {source}[/red]"); raise typer.Exit(1)
    result = asyncio.run(dispatch[source]())
    raise typer.Exit(0 if result else 1)


if __name__ == "__main__":
    app()