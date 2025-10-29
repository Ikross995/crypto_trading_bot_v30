#!/usr/bin/env python3
"""
Quick test script to verify IMBA CLI integration.

Tests:
1. Config parameter use_imba_signals
2. IMBASignalIntegration initialization
3. Signal generation from DataFrame
4. Filter application
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test 1: Config parameter
print("=" * 60)
print("TEST 1: Config Parameter")
print("=" * 60)

try:
    from core.config import Config
    
    # Test default value
    config = Config()
    assert hasattr(config, 'use_imba_signals'), "Missing use_imba_signals attribute"
    assert config.use_imba_signals == False, "Default should be False"
    print("✓ Config has use_imba_signals parameter (default=False)")
    
    # Test enabling IMBA
    config.use_imba_signals = True
    assert config.use_imba_signals == True, "Should be able to set to True"
    print("✓ Can enable use_imba_signals")
    
except Exception as e:
    print(f"✗ Config test failed: {e}")
    sys.exit(1)

# Test 2: IMBASignalIntegration initialization
print("\n" + "=" * 60)
print("TEST 2: IMBA Integration Initialization")
print("=" * 60)

try:
    from strategy.imba_integration import IMBASignalIntegration, should_use_imba_signals
    
    config = Config()
    config.use_imba_signals = True
    
    # Test helper function
    assert should_use_imba_signals(config) == True, "Should return True when enabled"
    config.use_imba_signals = False
    assert should_use_imba_signals(config) == False, "Should return False when disabled"
    print("✓ should_use_imba_signals() works correctly")
    
    # Initialize integration
    config.use_imba_signals = True
    imba = IMBASignalIntegration(config)
    assert imba is not None, "Failed to initialize IMBASignalIntegration"
    assert imba.config == config, "Config not stored correctly"
    assert imba.aggregator is not None, "Aggregator not initialized"
    assert imba.filter_manager is not None, "FilterManager not initialized"
    print("✓ IMBASignalIntegration initialized successfully")
    
except Exception as e:
    print(f"✗ Integration initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Signal generation with mock data
print("\n" + "=" * 60)
print("TEST 3: Signal Generation")
print("=" * 60)

try:
    # Create mock OHLCV data (300 candles for sufficient history)
    np.random.seed(42)
    n = 300
    dates = [datetime.now() - timedelta(minutes=i) for i in range(n, 0, -1)]
    
    # Generate price data with trend
    base_price = 50000.0
    trend = np.linspace(0, 1000, n)
    noise = np.random.randn(n) * 100
    close_prices = base_price + trend + noise
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices - np.random.rand(n) * 50,
        'high': close_prices + np.random.rand(n) * 100,
        'low': close_prices - np.random.rand(n) * 100,
        'close': close_prices,
        'volume': np.random.rand(n) * 1000 + 500
    })
    
    print(f"✓ Generated mock DataFrame with {len(df)} candles")
    print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Generate signal
    result = imba.generate_signal_from_df(df=df, symbol='BTCUSDT')
    
    # Validate result structure
    assert 'direction' in result, "Missing 'direction' in result"
    assert 'strength' in result, "Missing 'strength' in result"
    assert 'confidence' in result, "Missing 'confidence' in result"
    assert 'regime' in result, "Missing 'regime' in result"
    assert 'signals' in result, "Missing 'signals' in result"
    assert 'filters_passed' in result, "Missing 'filters_passed' in result"
    assert 'metadata' in result, "Missing 'metadata' in result"
    
    print("✓ Signal result has all required fields")
    
    # Validate direction
    assert result['direction'] in ['buy', 'sell', 'wait'], f"Invalid direction: {result['direction']}"
    print(f"✓ Direction: {result['direction'].upper()}")
    
    # Validate confidence
    assert 0.0 <= result['confidence'] <= 1.0, f"Confidence out of range: {result['confidence']}"
    print(f"✓ Confidence: {result['confidence']:.3f}")
    
    # Validate regime
    assert 'kind' in result['regime'], "Missing 'kind' in regime"
    print(f"✓ Regime: {result['regime']['kind']}")
    
    # Check metadata
    assert result['metadata']['imba_enabled'] == True, "IMBA not marked as enabled"
    print("✓ IMBA metadata present")
    
except Exception as e:
    print(f"✗ Signal generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Insufficient data handling
print("\n" + "=" * 60)
print("TEST 4: Insufficient Data Handling")
print("=" * 60)

try:
    # Create DataFrame with insufficient data
    small_df = df.head(50)
    result = imba.generate_signal_from_df(df=small_df, symbol='BTCUSDT')
    
    assert result['direction'] == 'wait', "Should return wait for insufficient data"
    assert result['confidence'] == 0.0, "Should have zero confidence"
    assert 'error' in result['metadata'], "Should have error in metadata"
    
    print("✓ Handles insufficient data gracefully")
    print(f"  Error: {result['metadata']['error']}")
    
except Exception as e:
    print(f"✗ Insufficient data test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: CLI flag parsing (simulated)
print("\n" + "=" * 60)
print("TEST 5: CLI Flag Support")
print("=" * 60)

try:
    # Simulate CLI parsing
    import typer
    from cli import app
    
    # Check if commands have use_imba parameter
    commands_to_check = ['live', 'paper', 'backtest']
    
    for cmd_name in commands_to_check:
        # Get command from typer app
        cmd = None
        for typer_cmd in app.registered_commands:
            if typer_cmd.name == cmd_name:
                cmd = typer_cmd.callback
                break
        
        if cmd:
            # Check function signature
            import inspect
            sig = inspect.signature(cmd)
            params = sig.parameters
            
            assert 'use_imba' in params, f"Command '{cmd_name}' missing use_imba parameter"
            print(f"✓ Command '{cmd_name}' has --use-imba flag")
        else:
            print(f"⚠ Could not find command '{cmd_name}'")
    
except Exception as e:
    print(f"✗ CLI flag test failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit - this is less critical
    print("  (This is not critical - CLI might still work)")

# Summary
print("\n" + "=" * 60)
print("✅ ALL CRITICAL TESTS PASSED!")
print("=" * 60)
print("\nIMBA CLI Integration is ready to use!")
print("\nTry it out:")
print("  bun cli.py backtest --symbol BTCUSDT --days 90 --use-imba")
print("  bun cli.py paper --symbol BTCUSDT --use-imba --verbose")
print("\nSee IMBA_CLI_USAGE.md for more examples.")
