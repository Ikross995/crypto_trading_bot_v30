# exchange/exits.py
"""
Adapter layer: ensure_* exits API that delegates to OrderManager logic.

This keeps a single source of truth for SL/TP behaviour in exchange.orders
and preserves compatibility for modules expecting ensure_* helpers.

FIXED: Reuses shared BinanceClient instead of creating new ones.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from core.config import get_config
from .client import BinanceClient
from .orders import OrderManager
from core.constants import OrderSide
from core.types import Position

logger = logging.getLogger(__name__)

# Shared client instance to avoid creating new connections
_shared_client: Optional[BinanceClient] = None
_shared_order_manager: Optional[OrderManager] = None


def _get_shared_client() -> BinanceClient:
    """Get or create shared BinanceClient instance."""
    global _shared_client
    if _shared_client is None:
        cfg = get_config()
        _shared_client = BinanceClient(testnet=cfg.testnet)
        logger.debug("[MONOLITH] Created shared BinanceClient")
    return _shared_client


def _get_shared_order_manager() -> OrderManager:
    """Get or create shared OrderManager instance."""
    global _shared_order_manager
    if _shared_order_manager is None:
        _shared_order_manager = OrderManager(_get_shared_client())
        logger.debug("[MONOLITH] Created shared OrderManager")
    return _shared_order_manager


def ensure_sl_on_exchange(symbol: str, pos_sign_or_side, stop_price: float) -> None:
    """Setup stop loss using shared OrderManager."""
    try:
        om = _get_shared_order_manager()
        # Normalize sign to OrderSide for closePosition SL
        sign = _pos_sign(pos_sign_or_side)
        side = "SELL" if sign > 0 else "BUY"
        # Use internal stop loss setup (closePosition)
        om._setup_stop_loss(symbol, side, float(stop_price))
        logger.info(f"[MONOLITH] Stop loss setup successful for {symbol}")
    except Exception as e:
        logger.error(f"[MONOLITH] Stop loss setup failed for {symbol}: {e}")
        raise


def ensure_tp_on_exchange(symbol: str, pos_sign_or_side, qty: float, entry: float, tps: List[float], tp_shares: List[float]) -> None:
    """Setup take profit orders using shared OrderManager."""
    try:
        cfg = get_config()
        om = _get_shared_order_manager()
        sign = _pos_sign(pos_sign_or_side)
        side = "SELL" if sign > 0 else "BUY"
        
        logger.debug(f"[MONOLITH] Setting up TP for {symbol}: qty={qty}, entry={entry}, tps={tps}")
        
        # Build pseudo-position to reuse OrderManager logic  
        pos = Position(symbol=symbol, side=sign, size=(abs(float(qty)) if sign > 0 else -abs(float(qty))), entry_price=float(entry))
        
        # Quantities per TP share
        shares = tp_shares or cfg.tp_shares()
        sm = sum(shares) if shares else 0.0
        if sm <= 0:
            shares = [1.0]
            sm = 1.0
        shares = [x / sm for x in shares]
        tp_qty = [abs(float(pos.size)) * s for s in shares[: len(tps)]]
        
        logger.debug(f"[MONOLITH] TP quantities calculated: {tp_qty}")
        logger.debug(f"[MONOLITH] TP prices received: {tps}")
        logger.debug(f"[MONOLITH] Position side: {side}, Entry: {entry}")
        
        # CRITICAL FIX: Validate TP prices make sense for position direction
        if side == "BUY":  # Closing LONG position
            # TP prices should be ABOVE entry for profit
            valid_tps = [tp for tp in tps if float(tp) > float(entry)]
            if len(valid_tps) != len(tps):
                logger.warning(f"[MONOLITH] Invalid TP prices for LONG: entry={entry}, tps={tps}")
        else:  # Closing SHORT position  
            # TP prices should be BELOW entry for profit
            valid_tps = [tp for tp in tps if float(tp) < float(entry)]
            if len(valid_tps) != len(tps):
                logger.warning(f"[MONOLITH] Invalid TP prices for SHORT: entry={entry}, tps={tps}")
                # Don't place invalid TP orders
                logger.error(f"[MONOLITH] Rejecting invalid TP setup for {symbol}")
                return
        
        om._setup_take_profits(symbol, side, [float(x) for x in tps[: len(tp_qty)]], tp_qty)
        logger.info(f"[MONOLITH] Take profits setup successful for {symbol}")
    except Exception as e:
        logger.error(f"[MONOLITH] Take profits setup failed for {symbol}: {e}")
        raise


def ensure_exits_on_exchange(symbol: str, pos_sign_or_side, qty: float, sl: float, tps: List[float], tp_shares: List[float]) -> None:
    """Setup combined SL + TP orders using shared OrderManager."""
    try:
        cfg = get_config()
        om = _get_shared_order_manager()
        sign = _pos_sign(pos_sign_or_side)
        
        logger.info(f"[MONOLITH] Setting up exits for {symbol}: qty={qty}, sl={sl}, tps={len(tps)} levels")
        
        # CRITICAL FIX: Get actual entry price from current position
        current_entry = 0.0
        try:
            positions = om.client.get_positions()
            for pos in positions:
                if pos.get('symbol') == symbol:
                    pos_amt = float(pos.get('positionAmt', 0))
                    if abs(pos_amt) > 0:
                        current_entry = float(pos.get('entryPrice', 0))
                        logger.debug(f"[MONOLITH] Found entry price: {current_entry} for {symbol}")
                        break
        except Exception as e:
            logger.warning(f"[MONOLITH] Could not get entry price for {symbol}: {e}")
        
        # Build position with actual entry price
        pos = Position(symbol=symbol, side=sign, size=(abs(float(qty)) if sign > 0 else -abs(float(qty))), entry_price=current_entry)
        
        # CRITICAL FIX: Calculate proper TP quantities instead of using shares
        tp_shares_list = tp_shares or cfg.tp_shares() or [0.4, 0.35, 0.25]
        total_qty = abs(float(qty))
        tp_quantities = [total_qty * share for share in tp_shares_list[:len(tps)]]
        
        logger.debug(f"[MONOLITH] Calculated TP quantities: {tp_quantities} from shares {tp_shares_list}")
        
        # Use OrderManager's unified exit setup with proper quantities
        om.setup_exit_orders(symbol, pos, float(sl), [float(x) for x in tps], tp_quantities)
        
        logger.info(f"[MONOLITH] Combined exits setup successful for {symbol}")
    except Exception as e:
        logger.error(f"[MONOLITH] Combined exits setup failed for {symbol}: {e}")
        raise


def _pos_sign(s) -> int:
    if isinstance(s, (int, float)):
        v = int(s)
        return 1 if v > 0 else (-1 if v < 0 else 0)
    s = str(s).lower()
    if s.startswith(("b", "l")):
        return 1
    if s.startswith("s"):
        return -1
    return 0
