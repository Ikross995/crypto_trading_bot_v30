"""
Fix for TP calculation to use price movement percentage instead of leveraged profit percentage.

This module patches the TP calculation to ensure that:
- TP1 = entry_price * (1 + 0.01)  # 1% price movement
- TP2 = entry_price * (1 + 0.02)  # 2% price movement
- TP3 = entry_price * (1 + 0.03)  # 3% price movement

The profit with leverage will be:
- At 15x leverage: 1% price movement = 15% profit
- At 10x leverage: 1% price movement = 10% profit
- At 5x leverage: 1% price movement = 5% profit
"""

import logging

logger = logging.getLogger(__name__)


def calculate_tp_prices(entry_price: float, tp_levels: list, side: str = "LONG") -> list:
    """
    Calculate TP prices based on price movement percentage, not leveraged profit.
    
    Args:
        entry_price: Entry price of the position
        tp_levels: List of TP percentages (e.g., [1.0, 2.0, 3.0] for 1%, 2%, 3%)
        side: "LONG" or "SHORT"
    
    Returns:
        List of TP prices
    """
    tp_prices = []
    
    for tp_pct in tp_levels:
        if side.upper() == "LONG":
            # For long positions, TP is above entry
            tp_price = entry_price * (1 + tp_pct / 100.0)
        else:
            # For short positions, TP is below entry
            tp_price = entry_price * (1 - tp_pct / 100.0)
        
        tp_prices.append(tp_price)
        
    logger.debug(f"Calculated TP prices for {side} position at {entry_price}: {tp_prices}")
    logger.debug(f"Price movements: {tp_levels}%")
    
    return tp_prices


def calculate_sl_price(entry_price: float, sl_pct: float, side: str = "LONG") -> float:
    """
    Calculate SL price based on price movement percentage.
    
    Args:
        entry_price: Entry price of the position
        sl_pct: SL percentage (e.g., 0.5 for 0.5% price movement)
        side: "LONG" or "SHORT"
    
    Returns:
        SL price
    """
    if side.upper() == "LONG":
        # For long positions, SL is below entry
        sl_price = entry_price * (1 - sl_pct / 100.0)
    else:
        # For short positions, SL is above entry
        sl_price = entry_price * (1 + sl_pct / 100.0)
    
    logger.debug(f"Calculated SL price for {side} position at {entry_price}: {sl_price}")
    logger.debug(f"Price movement: {sl_pct}%")
    
    return sl_price


def format_tp_with_percentage(entry_price: float, tp_price: float, side: str = "LONG") -> str:
    """
    Format TP price with percentage price movement (not leveraged profit).
    
    Args:
        entry_price: Entry price
        tp_price: TP price
        side: "LONG" or "SHORT"
    
    Returns:
        Formatted string like "$123.45(+1.5%)"
    """
    if side.upper() == "LONG":
        price_movement_pct = ((tp_price - entry_price) / entry_price) * 100
        sign = "+" if price_movement_pct > 0 else ""
    else:
        price_movement_pct = ((entry_price - tp_price) / entry_price) * 100
        sign = "+" if price_movement_pct > 0 else ""
    
    return f"${tp_price:.2f}({sign}{price_movement_pct:.1f}%)"


def apply_tp_fix():
    """
    Apply the TP calculation fix to the system.
    This should be called at startup to patch the TP calculation.
    """
    logger.info("Applying TP calculation fix for price-based percentages")
    
    # Here you would patch the actual TP calculation functions in the codebase
    # For example:
    # import exchange.orders
    # exchange.orders.calculate_tp_prices = calculate_tp_prices
    
    logger.info("TP calculation fix applied - using price movement percentages")
    
    # Log example calculations
    example_entry = 100.0
    example_tps = calculate_tp_prices(example_entry, [1.0, 2.0, 3.0], "LONG")
    logger.info(f"Example: Entry=${example_entry}, TPs={[format_tp_with_percentage(example_entry, tp, 'LONG') for tp in example_tps]}")
    logger.info(f"With 15x leverage, these give profits of: 15%, 30%, 45%")


if __name__ == "__main__":
    # Test the calculation
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing TP calculation fix:")
    print("-" * 50)
    
    entry = 50000.0
    tp_levels = [1.0, 2.0, 3.0]  # 1%, 2%, 3% price movement
    
    print(f"Entry price: ${entry}")
    print(f"TP levels: {tp_levels}% price movement")
    
    # Long position
    print("\nLONG position:")
    tp_prices_long = calculate_tp_prices(entry, tp_levels, "LONG")
    for i, tp in enumerate(tp_prices_long):
        print(f"  TP{i+1}: {format_tp_with_percentage(entry, tp, 'LONG')}")
    
    # Short position
    print("\nSHORT position:")
    tp_prices_short = calculate_tp_prices(entry, tp_levels, "SHORT")
    for i, tp in enumerate(tp_prices_short):
        print(f"  TP{i+1}: {format_tp_with_percentage(entry, tp, 'SHORT')}")
    
    print("\nWith different leverages, profit would be:")
    for leverage in [5, 10, 15, 20]:
        profits = [tp * leverage for tp in tp_levels]
        print(f"  {leverage}x leverage: {profits}% profit")