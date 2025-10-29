#!/usr/bin/env python3
"""
AI Trading Bot CLI Interface - Version Fix Approach

Unified command line interface for live trading, paper trading, and backtesting.
This version relies on PROPER DEPENDENCY VERSIONS instead of warning suppression.

IMPORTANT: Use requirements_fixed.txt to install compatible TensorFlow/Protobuf versions
instead of suppressing warnings.
"""

import asyncio
import logging
import sys
from pathlib import Path
import compat
compat.apply()

# NOTE: No warning suppression import here!
# Instead, use compatible library versions via requirements_fixed.txt

from infra.settings import load_profile, load_overrides, apply_settings_to_config
# --- POLYFILL: привести PositionManager / OrderManager к ожидаемому интерфейсу ---
import inspect, time
import compat
compat.apply()

try:
    from exchange.positions import PositionManager
except Exception:
    PositionManager = None

try:
    from exchange.orders import OrderManager
except Exception:
    OrderManager = None

def apply_live_fixes_after_import():
    """Apply fixes after all imports are complete."""
    try:
        from simple_live_fixes import apply_fixes
        apply_fixes()
        print("✅ Applied simple live fixes successfully!")
    except ImportError:
        print(⚠️ simple_live_fixes.py not found")
    except Exception as e:
        print(f"⚠️ Error applying live fixes: {e}")

def check_dependency_versions():
    """Check if compatible TensorFlow/Protobuf versions are installed."""
    try:
        import tensorflow as tf
        import google.protobuf
        
        tf_version = tf.__version__
        pb_version = google.protobuf.__version__
        
        print(f"🔍 Dependency versions:")
        print(f"   TensorFlow: {tf_version}")
        print(f"   Protobuf: {pb_version}")
        
        # Check compatibility
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Import a TensorFlow component that would trigger the warning
            import tensorflow.python.framework.dtypes
            
            protobuf_warnings = [warning for warning in w 
                               if 'protobuf' in str(warning.message).lower()]
            
            if protobuf_warnings:
                print("⚠️ PROTOBUF VERSION WARNINGS DETECTED!")
                print("   Consider running: python fix_protobuf_versions.py")
                print("   Or: pip install -r requirements_fixed.txt")
                return False
            else:
                print("✅ Compatible TensorFlow/Protobuf versions detected")
                return True
                
    except ImportError:
        print("⚠️ TensorFlow not installed")
        return True  # Not a problem if TensorFlow isn't needed
    except Exception as e:
        print(f"⚠️ Version check failed: {e}")
        return True  # Don't block startup on version check failures

# Rest of the CLI code remains the same...
# (The full cli.py implementation would continue here)

try:
    import typer
    from rich.console import Console
    from rich.progress import track
    from rich.table import Table
except ImportError:
    print("Missing dependencies. Install with: pip install typer rich")
    print("Or use: pip install -r requirements_fixed.txt")
    sys.exit(1)

from core.config import get_config, load_config
from core.constants import TradingMode
from core.types import BacktestResult

# Setup rich console
console = Console()
app = typer.Typer(
    name="trading-bot",
    help="AI Trading Bot with LSTM prediction and advanced risk management",
    add_completion=False,
)

def print_version_fix_info():
    """Display information about the version fix approach."""
    console.print("🎯 [bold cyan]Using Version Fix Approach[/bold cyan]")
    console.print("   This version relies on compatible library versions")
    console.print("   instead of suppressing warnings.")
    console.print()
    console.print("📋 [bold]If you see protobuf warnings:[/bold]")
    console.print("   1. Run: [green]python fix_protobuf_versions.py[/green]")
    console.print("   2. Or: [green]pip install -r requirements_fixed.txt[/green]")
    console.print()

@app.command()
def live(
    symbols: list[str] | None = typer.Option(
        None, "--symbols", "-s", help="Trading symbols (e.g. BTCUSDT)"
    ),
    symbol: str | None = typer.Option(
        None, "--symbol", help="Single trading symbol (alternative to --symbols)"
    ),
    timeframe: str | None = typer.Option(
        None, "--timeframe", "-t", help="Timeframe (e.g. 1m, 5m, 1h)"
    ),
    testnet: bool = typer.Option(
        False, "--testnet", help="Use testnet instead of mainnet"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Simulate trades without real orders"
    ),
    use_imba: bool = typer.Option(
        False, "--use-imba", help="Use IMBA research signals (9 advanced signals + regime detection)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    config_file: str | None = typer.Option(
        None, "--config", help="Config file path"
    ),
    check_versions: bool = typer.Option(
        True, "--check-versions", help="Check dependency versions for compatibility"
    ),
) -> None:
    """
    Run live trading mode with proper dependency management.

    This version uses compatible TensorFlow/Protobuf versions instead of
    suppressing warnings.
    """
    if check_versions:
        print_version_fix_info()
        if not check_dependency_versions():
            console.print("[yellow]Consider fixing versions for cleaner output[/yellow]")
        console.print()

    # Rest of live trading implementation...
    console.print("[green]Starting Live Trading Mode (Version Fix Approach)[/green]")
    console.print("🎯 Using proper dependency versions instead of warning suppression")

if __name__ == "__main__":
    print("🔧 CLI with Version Fix Approach")
    print("Use compatible TensorFlow/Protobuf versions for clean output!")
    print()
    app()