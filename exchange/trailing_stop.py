"""
Trailing Stop Loss Manager with Take Profit Integration.

Automatically adjusts stop loss after each take profit fill:
- After 1st TP: Move SL to 50% between entry and current price (partial protection)
- After 2nd TP: Move SL to break-even (entry price)
- After 3rd TP: Keep trailing or close remaining position

Features:
- Monitors TP order fills in real-time
- Cancels old SL before placing new one
- Prevents spam with cooldown logic
- Supports both LONG and SHORT positions
"""

import logging
import time
from typing import Dict, Optional, List
from dataclasses import dataclass

from core.types import Position
from .client import BinanceClient
from .orders import OrderManager

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop behavior."""
    # Move SL after nth TP fill (1-indexed)
    move_sl_after_tp: List[int] = None  # [1, 2] means move after 1st and 2nd TP
    
    # SL adjustment rules per TP level
    # 1st TP: Move to 50% between entry and current price
    # 2nd TP: Move to break-even (entry price)
    # 3rd TP: Keep current or close
    sl_adjustments: Dict[int, str] = None  # {1: "50%", 2: "breakeven", 3: "close"}
    
    # Cooldown between SL updates (seconds)
    update_cooldown: float = 5.0
    
    # Safety buffer (percentage) to avoid immediate SL trigger
    safety_buffer_pct: float = 0.1  # 0.1% buffer
    
    def __post_init__(self):
        if self.move_sl_after_tp is None:
            self.move_sl_after_tp = [1, 2]  # Default: move after 1st and 2nd TP
        
        if self.sl_adjustments is None:
            self.sl_adjustments = {
                1: "50%",       # After 1st TP: move to 50% protection
                2: "breakeven",  # After 2nd TP: move to entry (break-even)
                3: "keep"        # After 3rd TP: keep current SL
            }


class TrailingStopManager:
    """
    Manages trailing stop loss based on take profit fills.
    
    Workflow:
    1. Position opened with entry price and 3 TP levels
    2. Monitor TP order fills via exchange API
    3. When TP fills detected → calculate new SL level
    4. Cancel old SL order
    5. Place new SL order at adjusted level
    6. Update internal state
    """
    
    def __init__(self, client: BinanceClient, order_manager: OrderManager, 
                 config: Optional[TrailingStopConfig] = None):
        """
        Initialize trailing stop manager.
        
        Args:
            client: Binance client for API calls
            order_manager: Order manager for placing/canceling orders
            config: Configuration for trailing behavior
        """
        self.client = client
        self.order_manager = order_manager
        self.config = config or TrailingStopConfig()
        
        # Track state per symbol
        # Format: {symbol: {"entry": float, "side": str, "tp_filled": int, "last_sl": float, "last_update": float}}
        self._positions: Dict[str, dict] = {}
        
        # Cache filled TP orders to avoid re-processing
        self._filled_tps: Dict[str, set] = {}  # {symbol: {order_id1, order_id2, ...}}
        
        logger.info("TrailingStopManager initialized")
        logger.info(f"  Move SL after TP levels: {self.config.move_sl_after_tp}")
        logger.info(f"  SL adjustments: {self.config.sl_adjustments}")
    
    def register_position(self, symbol: str, entry_price: float, side: str, 
                          tp_levels: List[float], initial_sl: float) -> None:
        """
        Register a new position for trailing stop management.
        
        Args:
            symbol: Trading pair
            entry_price: Entry price of position
            side: Position side ("BUY" or "SELL")
            tp_levels: List of take profit prices
            initial_sl: Initial stop loss price
        """
        self._positions[symbol] = {
            "entry": entry_price,
            "side": side.upper(),
            "tp_levels": tp_levels,
            "tp_filled": 0,  # Number of TPs filled
            "last_sl": initial_sl,
            "last_update": time.time(),
            "total_tps": len(tp_levels)
        }
        
        # Reset filled TP cache
        self._filled_tps[symbol] = set()
        
        logger.info(f"[TRAIL_SL] Registered position for {symbol}: "
                   f"entry={entry_price:.4f}, side={side}, "
                   f"TPs={len(tp_levels)}, initial_sl={initial_sl:.4f}")
    
    def unregister_position(self, symbol: str) -> None:
        """Remove position from tracking (when closed)."""
        if symbol in self._positions:
            del self._positions[symbol]
        if symbol in self._filled_tps:
            del self._filled_tps[symbol]
        logger.info(f"[TRAIL_SL] Unregistered position for {symbol}")
    
    async def check_and_update(self, symbol: str) -> bool:
        """
        Check if TP orders filled and update SL if needed.
        
        Args:
            symbol: Trading pair to check
        
        Returns:
            True if SL was updated, False otherwise
        """
        if symbol not in self._positions:
            return False
        
        pos_data = self._positions[symbol]
        
        # Check cooldown
        now = time.time()
        if now - pos_data["last_update"] < self.config.update_cooldown:
            return False
        
        # Fetch open and filled orders from exchange
        try:
            filled_count = await self._count_filled_tps(symbol)
            
            if filled_count > pos_data["tp_filled"]:
                logger.info(f"[TRAIL_SL] {symbol}: Detected {filled_count} TP fills "
                           f"(was {pos_data['tp_filled']})")
                
                # Update SL based on filled count
                updated = await self._update_stop_loss(symbol, filled_count)
                
                if updated:
                    pos_data["tp_filled"] = filled_count
                    pos_data["last_update"] = now
                    return True
            
        except Exception as e:
            logger.error(f"[TRAIL_SL] Failed to check/update {symbol}: {e}")
        
        return False
    
    async def _count_filled_tps(self, symbol: str) -> int:
        """
        Count how many TP orders have been filled.
        
        Returns:
            Number of filled TP orders
        """
        # Get TP orders from order manager's internal state
        exit_orders = self.order_manager._exit_orders.get(symbol, {})
        tp_orders = exit_orders.get("take_profits", [])
        
        if not tp_orders:
            return 0
        
        filled_count = 0
        
        # Check each TP order status
        for tp in tp_orders:
            order_id = tp["order_id"]
            
            # Skip if already counted
            if order_id in self._filled_tps.get(symbol, set()):
                filled_count += 1
                continue
            
            # Fetch order status from exchange
            try:
                order_status = await self.client.get_order(symbol, order_id)
                
                if order_status and order_status.get("status") == "FILLED":
                    # Mark as filled
                    self._filled_tps.setdefault(symbol, set()).add(order_id)
                    filled_count += 1
                    logger.info(f"[TRAIL_SL] {symbol}: TP level {tp['level']} filled "
                               f"@ {tp['price']:.4f} (order {order_id})")
            
            except Exception as e:
                logger.warning(f"[TRAIL_SL] Failed to fetch order {order_id} for {symbol}: {e}")
        
        return filled_count
    
    async def _update_stop_loss(self, symbol: str, tp_filled_count: int) -> bool:
        """
        Update stop loss based on TP fill count.
        
        Args:
            symbol: Trading pair
            tp_filled_count: Number of TPs that have filled
        
        Returns:
            True if SL was updated successfully
        """
        if tp_filled_count not in self.config.move_sl_after_tp:
            logger.debug(f"[TRAIL_SL] {symbol}: No SL adjustment needed for TP count {tp_filled_count}")
            return False
        
        pos_data = self._positions[symbol]
        entry = pos_data["entry"]
        side = pos_data["side"]
        old_sl = pos_data["last_sl"]
        
        # Get current market price
        try:
            ticker = await self.client.get_ticker(symbol)
            current_price = float(ticker.get("lastPrice", entry))
        except Exception as e:
            logger.warning(f"[TRAIL_SL] Failed to get current price for {symbol}, using entry: {e}")
            current_price = entry
        
        # Calculate new SL based on adjustment rule
        adjustment_rule = self.config.sl_adjustments.get(tp_filled_count, "keep")
        
        if adjustment_rule == "keep":
            logger.info(f"[TRAIL_SL] {symbol}: Keeping current SL (rule: keep)")
            return False
        
        new_sl = self._calculate_new_sl(entry, current_price, side, adjustment_rule)
        
        if new_sl is None:
            logger.warning(f"[TRAIL_SL] {symbol}: Failed to calculate new SL")
            return False
        
        # Add safety buffer to avoid immediate trigger
        new_sl = self._apply_safety_buffer(new_sl, side)
        
        # Validate new SL is better than old SL
        if not self._is_better_sl(old_sl, new_sl, side):
            logger.warning(f"[TRAIL_SL] {symbol}: New SL {new_sl:.4f} is not better than old {old_sl:.4f}")
            return False
        
        # Cancel old SL and place new one
        try:
            logger.info(f"[TRAIL_SL] {symbol}: Updating SL: {old_sl:.4f} → {new_sl:.4f} "
                       f"(after {tp_filled_count} TP fills, rule: {adjustment_rule})")
            
            # Cancel existing SL
            self.order_manager._cancel_existing_exit_order(symbol, "stop_loss")
            
            # Place new SL
            close_side = "SELL" if side == "BUY" else "BUY"
            self.order_manager._setup_stop_loss(symbol, close_side, new_sl)
            
            # Update internal state
            pos_data["last_sl"] = new_sl
            
            logger.info(f"[TRAIL_SL] {symbol}: SL updated successfully to {new_sl:.4f}")
            return True
        
        except Exception as e:
            logger.error(f"[TRAIL_SL] {symbol}: Failed to update SL: {e}")
            return False
    
    def _calculate_new_sl(self, entry: float, current: float, side: str, rule: str) -> Optional[float]:
        """
        Calculate new SL price based on adjustment rule.
        
        Args:
            entry: Entry price
            current: Current market price
            side: Position side ("BUY" or "SELL")
            rule: Adjustment rule ("50%", "breakeven", "close")
        
        Returns:
            New SL price or None if invalid
        """
        if rule == "breakeven":
            # Move to entry price (break-even)
            return entry
        
        elif rule == "50%":
            # Move to 50% between entry and current price
            if side == "BUY":
                # Long: SL should be below entry
                # Move from below entry to halfway between entry and current
                return entry + (current - entry) * 0.5 * 0.5  # 50% of gain protection
            else:
                # Short: SL should be above entry
                return entry - (entry - current) * 0.5 * 0.5
        
        elif rule == "close":
            # Close position (return None to indicate close)
            return None
        
        else:
            logger.warning(f"Unknown SL adjustment rule: {rule}")
            return None
    
    def _apply_safety_buffer(self, price: float, side: str) -> float:
        """
        Apply safety buffer to SL price to avoid immediate trigger.
        
        Args:
            price: Original SL price
            side: Position side
        
        Returns:
            Adjusted SL price with buffer
        """
        buffer = self.config.safety_buffer_pct / 100.0
        
        if side == "BUY":
            # Long: reduce SL slightly (move down)
            return price * (1 - buffer)
        else:
            # Short: increase SL slightly (move up)
            return price * (1 + buffer)
    
    def _is_better_sl(self, old_sl: float, new_sl: float, side: str) -> bool:
        """
        Check if new SL is better (tighter) than old SL.
        
        Args:
            old_sl: Current stop loss price
            new_sl: Proposed new stop loss price
            side: Position side
        
        Returns:
            True if new SL is better (tighter protection)
        """
        if side == "BUY":
            # Long: better SL is higher (closer to current price)
            return new_sl > old_sl
        else:
            # Short: better SL is lower
            return new_sl < old_sl
    
    async def monitor_all_positions(self) -> None:
        """Monitor all registered positions and update SL if needed."""
        for symbol in list(self._positions.keys()):
            try:
                await self.check_and_update(symbol)
            except Exception as e:
                logger.error(f"[TRAIL_SL] Error monitoring {symbol}: {e}")
    
    def get_status(self, symbol: str) -> Optional[dict]:
        """Get current trailing stop status for a symbol."""
        return self._positions.get(symbol)
