#!/usr/bin/env python3
"""
Exit Tracker System

Real-time monitoring of TP/SL order executions with AI integration.
Tracks when positions are closed and updates the adaptive learning system with accurate data.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from loguru import logger


@dataclass
class TrackedExit:
    """Represents a tracked exit order."""
    
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    order_type: str  # "STOP_MARKET", "LIMIT", etc.
    quantity: float
    price: float
    exit_type: str  # "sl", "tp1", "tp2", "tp3"
    trade_id: str  # Unique trade ID for AI system
    created_time: datetime
    status: str = "NEW"  # NEW, FILLED, CANCELLED, EXPIRED


@dataclass
class ExitEvent:
    """Represents an exit execution event."""
    
    trade_id: str
    symbol: str
    exit_price: float
    exit_reason: str  # "sl", "tp1", "tp2", "tp3", "manual"
    quantity_filled: float
    fees_paid: float
    fill_time: datetime
    order_id: str


class ExitTracker:
    """
    Real-time exit tracker that monitors TP/SL order executions.
    
    Features:
    - Tracks all exit orders (TP/SL) for open positions
    - Monitors order status changes in real-time
    - Calculates actual PnL with real fill prices
    - Updates AI system with accurate trade results
    - Handles partial fills and order modifications
    """
    
    def __init__(self, binance_client, adaptive_learning_system=None):
        self.client = binance_client
        self.adaptive_learning = adaptive_learning_system
        self.logger = logging.getLogger("exchange.exit_tracker")
        
        # Tracked exits: {order_id: TrackedExit}
        self.tracked_exits: Dict[str, TrackedExit] = {}
        
        # Position tracking: {symbol: trade_id}
        self.position_trade_map: Dict[str, str] = {}
        
        # Callbacks for exit events
        self.exit_callbacks: List[Callable[[ExitEvent], None]] = []
        
        # Monitoring settings
        self.monitoring_enabled = True
        self.check_interval = 2.0  # seconds
        self.max_check_attempts = 1800  # 1 hour max tracking
        
        logger.info("🎯 [EXIT_TRACKER] Exit tracking system initialized")
    
    def register_position_exits(self, symbol: str, trade_id: str, sl_order_id: str = None, 
                               tp_order_ids: List[str] = None, sl_price: float = None, 
                               tp_prices: List[float] = None) -> None:
        """Register exit orders for a position to track."""
        try:
            self.position_trade_map[symbol] = trade_id
            
            # Register stop loss
            if sl_order_id and sl_price:
                tracked_sl = TrackedExit(
                    order_id=sl_order_id,
                    symbol=symbol,
                    side="SELL",  # Will be determined from position
                    order_type="STOP_MARKET",
                    quantity=0.0,  # Will be updated from order details
                    price=sl_price,
                    exit_type="sl",
                    trade_id=trade_id,
                    created_time=datetime.now(timezone.utc)
                )
                self.tracked_exits[sl_order_id] = tracked_sl
                logger.info(f"🎯 [EXIT_TRACKER] Registered SL: {sl_order_id} @ ${sl_price:.2f}")
            
            # Register take profits
            if tp_order_ids and tp_prices:
                for i, (tp_order_id, tp_price) in enumerate(zip(tp_order_ids, tp_prices)):
                    if tp_order_id:
                        tracked_tp = TrackedExit(
                            order_id=tp_order_id,
                            symbol=symbol,
                            side="SELL",  # Will be determined from position
                            order_type="LIMIT",
                            quantity=0.0,  # Will be updated from order details
                            price=tp_price,
                            exit_type=f"tp{i+1}",
                            trade_id=trade_id,
                            created_time=datetime.now(timezone.utc)
                        )
                        self.tracked_exits[tp_order_id] = tracked_tp
                        logger.info(f"🎯 [EXIT_TRACKER] Registered TP{i+1}: {tp_order_id} @ ${tp_price:.2f}")
            
            logger.info(f"🎯 [EXIT_TRACKER] Position {symbol} exits registered for trade {trade_id}")
            
        except Exception as e:
            logger.error(f"❌ [EXIT_TRACKER] Failed to register exits: {e}")
    
    def add_exit_callback(self, callback: Callable[[ExitEvent], None]) -> None:
        """Add a callback function to be called when exits are executed."""
        self.exit_callbacks.append(callback)
        logger.debug(f"🎯 [EXIT_TRACKER] Added exit callback: {callback.__name__}")
    
    async def start_monitoring(self) -> None:
        """Start the exit monitoring loop."""
        if not self.monitoring_enabled:
            logger.warning("🎯 [EXIT_TRACKER] Monitoring disabled")
            return
        
        logger.info("🎯 [EXIT_TRACKER] Starting exit monitoring loop")
        
        attempt_count = 0
        while self.monitoring_enabled and attempt_count < self.max_check_attempts:
            try:
                if self.tracked_exits:
                    await self._check_all_exits()
                
                await asyncio.sleep(self.check_interval)
                attempt_count += 1
                
                # Log status every 5 minutes
                if attempt_count % 150 == 0:  # 150 * 2s = 5min
                    active_exits = len([e for e in self.tracked_exits.values() if e.status == "NEW"])
                    logger.info(f"🎯 [EXIT_TRACKER] Active exits: {active_exits}, Total tracked: {len(self.tracked_exits)}")
                
            except Exception as e:
                logger.error(f"❌ [EXIT_TRACKER] Monitoring error: {e}")
                await asyncio.sleep(5.0)  # Back off on errors
        
        logger.info("🎯 [EXIT_TRACKER] Monitoring loop ended")
    
    async def _check_all_exits(self) -> None:
        """Check status of all tracked exits."""
        try:
            # Get all open orders from exchange
            if not self.client:
                return
            
            # Group exits by symbol for efficient API calls
            symbols_to_check = set(exit.symbol for exit in self.tracked_exits.values() if exit.status == "NEW")
            
            for symbol in symbols_to_check:
                try:
                    open_orders = self.client.get_open_orders(symbol=symbol)
                    await self._process_symbol_exits(symbol, open_orders)
                    
                except Exception as e:
                    logger.debug(f"🎯 [EXIT_TRACKER] Error checking {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"❌ [EXIT_TRACKER] Error in check_all_exits: {e}")
    
    async def _process_symbol_exits(self, symbol: str, open_orders: List[Dict]) -> None:
        """Process exits for a specific symbol."""
        try:
            # Create map of open order IDs
            open_order_ids = {str(order.get('orderId', '')): order for order in open_orders}
            
            # Check each tracked exit for this symbol
            symbol_exits = [exit for exit in self.tracked_exits.values() 
                          if exit.symbol == symbol and exit.status == "NEW"]
            
            for tracked_exit in symbol_exits:
                order_id = tracked_exit.order_id
                
                if order_id not in open_order_ids:
                    # Order is no longer open - likely filled or cancelled
                    await self._handle_missing_order(tracked_exit)
                else:
                    # Order still open - update status if needed
                    open_order = open_order_ids[order_id]
                    await self._update_exit_status(tracked_exit, open_order)
            
        except Exception as e:
            logger.error(f"❌ [EXIT_TRACKER] Error processing {symbol} exits: {e}")
    
    async def _handle_missing_order(self, tracked_exit: TrackedExit) -> None:
        """Handle an order that's no longer in open orders (likely filled)."""
        try:
            # Try to get order history to determine what happened
            try:
                # Get account trade history for this symbol to find the execution
                trades = self.client.get_account_trades(symbol=tracked_exit.symbol, limit=50)
                
                # Look for trades matching this order ID
                matching_trade = None
                for trade in trades:
                    if str(trade.get('orderId', '')) == tracked_exit.order_id:
                        matching_trade = trade
                        break
                
                if matching_trade:
                    # Order was filled
                    fill_price = float(matching_trade.get('price', tracked_exit.price))
                    fill_qty = float(matching_trade.get('qty', 0))
                    fees = float(matching_trade.get('commission', 0))
                    fill_time_ms = int(matching_trade.get('time', 0))
                    fill_time = datetime.fromtimestamp(fill_time_ms / 1000, tz=timezone.utc)
                    
                    # Create exit event
                    exit_event = ExitEvent(
                        trade_id=tracked_exit.trade_id,
                        symbol=tracked_exit.symbol,
                        exit_price=fill_price,
                        exit_reason=tracked_exit.exit_type,
                        quantity_filled=fill_qty,
                        fees_paid=fees,
                        fill_time=fill_time,
                        order_id=tracked_exit.order_id
                    )
                    
                    # Process the exit
                    await self._process_exit_event(exit_event)
                    
                    # Mark as filled
                    tracked_exit.status = "FILLED"
                    
                    logger.info(f"🎯 [EXIT_FILLED] {tracked_exit.exit_type.upper()}: {tracked_exit.symbol} "
                              f"@ ${fill_price:.2f} | PnL will be calculated by AI system")
                
                else:
                    # Order likely cancelled
                    tracked_exit.status = "CANCELLED"
                    logger.info(f"🎯 [EXIT_CANCELLED] {tracked_exit.exit_type.upper()}: {tracked_exit.order_id}")
                
            except Exception as history_e:
                logger.debug(f"🎯 [EXIT_TRACKER] Could not get trade history: {history_e}")
                # Assume filled and use estimated price
                await self._process_estimated_exit(tracked_exit)
                
        except Exception as e:
            logger.error(f"❌ [EXIT_TRACKER] Error handling missing order {tracked_exit.order_id}: {e}")
    
    async def _update_exit_status(self, tracked_exit: TrackedExit, open_order: Dict) -> None:
        """Update exit status from open order information."""
        try:
            order_status = open_order.get('status', 'NEW')
            executed_qty = float(open_order.get('executedQty', 0))
            
            if order_status == 'PARTIALLY_FILLED' and executed_qty > 0:
                logger.info(f"🎯 [EXIT_PARTIAL] {tracked_exit.exit_type.upper()}: {tracked_exit.symbol} "
                          f"partially filled {executed_qty}/{tracked_exit.quantity}")
            
            # Note: We mainly detect fills by absence from open orders
            # This function is for logging partial fills and other status updates
            
        except Exception as e:
            logger.debug(f"🎯 [EXIT_TRACKER] Error updating exit status: {e}")
    
    async def _process_estimated_exit(self, tracked_exit: TrackedExit) -> None:
        """Process exit with estimated data when exact fill data unavailable."""
        try:
            # Use order price as estimate
            exit_event = ExitEvent(
                trade_id=tracked_exit.trade_id,
                symbol=tracked_exit.symbol,
                exit_price=tracked_exit.price,
                exit_reason=tracked_exit.exit_type,
                quantity_filled=tracked_exit.quantity,
                fees_paid=0.0,  # Estimate
                fill_time=datetime.now(timezone.utc),
                order_id=tracked_exit.order_id
            )
            
            await self._process_exit_event(exit_event)
            tracked_exit.status = "FILLED"
            
            logger.warning(f"🎯 [EXIT_ESTIMATED] {tracked_exit.exit_type.upper()}: {tracked_exit.symbol} "
                         f"@ ${tracked_exit.price:.2f} (estimated)")
            
        except Exception as e:
            logger.error(f"❌ [EXIT_TRACKER] Error processing estimated exit: {e}")
    
    async def _process_exit_event(self, exit_event: ExitEvent) -> None:
        """Process an exit event and notify callbacks."""
        try:
            # Update AI system if available
            if self.adaptive_learning:
                try:
                    success = await self.adaptive_learning.update_trade_exit(
                        trade_id=exit_event.trade_id,
                        exit_price=exit_event.exit_price,
                        exit_reason=exit_event.exit_reason,
                        tp_level_hit=exit_event.exit_reason if exit_event.exit_reason.startswith('tp') else None,
                        fees_paid=exit_event.fees_paid
                    )
                    
                    if success:
                        logger.info(f"🤖 [AI_UPDATE] ✅ Trade {exit_event.trade_id} exit recorded: "
                                  f"{exit_event.exit_reason} @ ${exit_event.exit_price:.2f}")
                    else:
                        logger.error(f"🤖 [AI_UPDATE] ❌ Failed to update trade {exit_event.trade_id}")
                        
                except Exception as ai_e:
                    logger.error(f"🤖 [AI_UPDATE] Exception: {ai_e}")
            
            # Call registered callbacks
            for callback in self.exit_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(exit_event)
                    else:
                        callback(exit_event)
                except Exception as cb_e:
                    logger.error(f"❌ [EXIT_CALLBACK] Callback error: {cb_e}")
            
            # Enhanced logging
            logger.info(f"🏁 [EXIT_EVENT] {exit_event.symbol} {exit_event.exit_reason.upper()}: "
                      f"${exit_event.exit_price:.2f} | Qty: {exit_event.quantity_filled:.6f} | "
                      f"Fees: ${exit_event.fees_paid:.4f}")
            
        except Exception as e:
            logger.error(f"❌ [EXIT_TRACKER] Error processing exit event: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop the exit monitoring."""
        self.monitoring_enabled = False
        logger.info("🎯 [EXIT_TRACKER] Monitoring stopped")
    
    def cleanup_position(self, symbol: str) -> None:
        """Clean up tracking data for a closed position."""
        try:
            # Remove position mapping
            if symbol in self.position_trade_map:
                trade_id = self.position_trade_map[symbol]
                del self.position_trade_map[symbol]
                
                # Remove related exit tracking
                exits_to_remove = []
                for order_id, tracked_exit in self.tracked_exits.items():
                    if tracked_exit.symbol == symbol and tracked_exit.trade_id == trade_id:
                        exits_to_remove.append(order_id)
                
                for order_id in exits_to_remove:
                    del self.tracked_exits[order_id]
                
                logger.info(f"🎯 [EXIT_TRACKER] Cleaned up {symbol} tracking ({len(exits_to_remove)} exits removed)")
                
        except Exception as e:
            logger.error(f"❌ [EXIT_TRACKER] Error cleaning up {symbol}: {e}")
    
    def get_tracking_status(self) -> Dict[str, Any]:
        """Get current tracking status."""
        try:
            status = {
                'monitoring_enabled': self.monitoring_enabled,
                'total_tracked_exits': len(self.tracked_exits),
                'active_exits': len([e for e in self.tracked_exits.values() if e.status == "NEW"]),
                'filled_exits': len([e for e in self.tracked_exits.values() if e.status == "FILLED"]),
                'tracked_positions': len(self.position_trade_map),
                'callbacks_registered': len(self.exit_callbacks)
            }
            
            # Position breakdown
            position_breakdown = {}
            for symbol, trade_id in self.position_trade_map.items():
                symbol_exits = [e for e in self.tracked_exits.values() 
                              if e.symbol == symbol and e.trade_id == trade_id]
                position_breakdown[symbol] = {
                    'trade_id': trade_id,
                    'total_exits': len(symbol_exits),
                    'active_exits': len([e for e in symbol_exits if e.status == "NEW"]),
                    'exit_types': [e.exit_type for e in symbol_exits]
                }
            
            status['positions'] = position_breakdown
            return status
            
        except Exception as e:
            logger.error(f"❌ [EXIT_TRACKER] Error getting status: {e}")
            return {'error': str(e)}
    
    async def manual_exit_check(self, symbol: str = None) -> Dict[str, Any]:
        """Manually trigger exit check for debugging."""
        try:
            logger.info(f"🎯 [EXIT_TRACKER] Manual exit check triggered{f' for {symbol}' if symbol else ''}")
            
            if symbol:
                # Check specific symbol
                open_orders = self.client.get_open_orders(symbol=symbol)
                await self._process_symbol_exits(symbol, open_orders)
            else:
                # Check all symbols
                await self._check_all_exits()
            
            return self.get_tracking_status()
            
        except Exception as e:
            logger.error(f"❌ [EXIT_TRACKER] Manual check error: {e}")
            return {'error': str(e)}


# Utility functions for integration

async def create_exit_tracker(binance_client, adaptive_learning_system=None) -> ExitTracker:
    """Factory function to create and start an exit tracker."""
    try:
        tracker = ExitTracker(binance_client, adaptive_learning_system)
        
        # Start monitoring in background
        asyncio.create_task(tracker.start_monitoring())
        
        logger.info("🎯 [EXIT_TRACKER] Created and started exit tracker")
        return tracker
        
    except Exception as e:
        logger.error(f"❌ [EXIT_TRACKER] Failed to create exit tracker: {e}")
        raise


def integrate_exit_tracker_callbacks(exit_tracker: ExitTracker, trading_engine) -> None:
    """Integrate exit tracker with trading engine callbacks."""
    try:
        def on_exit_event(exit_event: ExitEvent):
            """Callback for exit events."""
            try:
                # Log to trading engine
                trading_engine.logger.info(
                    f"🏁 [TRADE_EXIT] {exit_event.symbol} closed via {exit_event.exit_reason} "
                    f"@ ${exit_event.exit_price:.2f}"
                )
                
                # 🧠 Update AI Learning System
                if hasattr(trading_engine, 'adaptive_learning') and trading_engine.adaptive_learning:
                    try:
                        import asyncio
                        
                        # Try to find trade_id from pending_trades
                        trade_id = None
                        if hasattr(trading_engine, 'pending_trades') and exit_event.symbol in trading_engine.pending_trades:
                            trade_id = trading_engine.pending_trades[exit_event.symbol].get('trade_id')
                        
                        # Create async task to update trade exit in learning system
                        asyncio.create_task(
                            trading_engine.adaptive_learning.update_trade_exit(
                                symbol=exit_event.symbol,
                                exit_price=exit_event.exit_price,
                                exit_reason=exit_event.exit_reason,
                                trade_id=trade_id
                            )
                        )
                        
                        if trade_id:
                            trading_engine.logger.info(f"🧠 [AI_LEARNING] Updated trade exit for {exit_event.symbol} (trade_id: {trade_id})")
                        else:
                            trading_engine.logger.info(f"🧠 [AI_LEARNING] Updated trade exit for {exit_event.symbol} (no trade_id)")
                    except Exception as ai_e:
                        trading_engine.logger.error(f"🧠 [AI_LEARNING] Failed to update trade exit: {ai_e}")
                
                # Update position tracking
                if hasattr(trading_engine, 'active_positions') and exit_event.symbol in trading_engine.active_positions:
                    # Position closed, remove from tracking
                    del trading_engine.active_positions[exit_event.symbol]
                    trading_engine.logger.info(f"🎯 [POSITION_CLOSED] Removed {exit_event.symbol} from active positions")
                
                # Send Telegram notification if available
                if hasattr(trading_engine, 'telegram') and trading_engine.telegram:
                    try:
                        pnl_text = f"Fees: ${exit_event.fees_paid:.2f}" if exit_event.fees_paid > 0 else "Fees: minimal"
                        message = (
                            f"🏁 Trade Closed\n"
                            f"Symbol: {exit_event.symbol}\n"
                            f"Exit: {exit_event.exit_reason.upper()}\n"
                            f"Price: ${exit_event.exit_price:.2f}\n"
                            f"Qty: {exit_event.quantity_filled:.6f}\n"
                            f"{pnl_text}"
                        )
                        asyncio.create_task(trading_engine.telegram.send_message(message))
                    except Exception as tg_e:
                        trading_engine.logger.debug(f"Telegram notification failed: {tg_e}")
                
            except Exception as e:
                trading_engine.logger.error(f"❌ [EXIT_CALLBACK] Error in exit callback: {e}")
        
        # Register the callback
        exit_tracker.add_exit_callback(on_exit_event)
        
        logger.info("🎯 [EXIT_TRACKER] Integrated with trading engine")
        
    except Exception as e:
        logger.error(f"❌ [EXIT_TRACKER] Failed to integrate callbacks: {e}")