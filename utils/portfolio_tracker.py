"""
Portfolio Tracker - Real-time Balance, P&L, and Performance Metrics

Features:
- Real-time balance from exchange
- Unrealized P&L per position
- Total portfolio value
- Daily/Weekly/Monthly returns
- Annual Percentage Rate (APR)
- Sharpe Ratio
- Max Drawdown
- Win Rate

Usage:
    tracker = PortfolioTracker(client)
    
    # Get current stats
    stats = tracker.get_portfolio_stats()
    
    # Display beautiful summary
    tracker.log_portfolio_summary()
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Information about an open position."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    current_price: float
    quantity: float
    notional_value: float  # Current value in USDT
    unrealized_pnl: float
    unrealized_pnl_pct: float
    leverage: int
    liquidation_price: Optional[float] = None


@dataclass
class PortfolioStats:
    """Portfolio statistics and performance metrics."""
    # Balance
    total_balance: float  # Total wallet balance
    available_balance: float  # Available for trading
    margin_balance: float  # Used as margin
    
    # Positions
    open_positions: List[PositionInfo]
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    
    # Performance
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    weekly_pnl_pct: float
    monthly_pnl: float
    monthly_pnl_pct: float
    
    # Annual metrics
    apr: float  # Annual Percentage Rate
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    
    # Trading stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Timestamp
    timestamp: datetime


class PortfolioTracker:
    """
    Track portfolio balance, positions, and performance metrics.
    
    Uses Binance Futures API to fetch real-time data.
    Calculates P&L, APR, Sharpe Ratio, and other metrics.
    """
    
    def __init__(self, client, config=None):
        """
        Initialize portfolio tracker.
        
        Args:
            client: Binance client instance
            config: Trading config (optional)
        """
        self.client = client
        self.config = config
        
        # Historical data for performance calculations
        self.history_file = Path("data/portfolio_history.json")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load historical data
        self.history = self._load_history()
        
        logger.info("Portfolio Tracker initialized")
    
    def _load_history(self) -> List[Dict]:
        """Load historical portfolio snapshots."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load portfolio history: {e}")
        return []
    
    def _save_history(self):
        """Save historical portfolio snapshots."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save portfolio history: {e}")
    
    def _save_snapshot(self, balance: float, unrealized_pnl: float):
        """Save current portfolio snapshot."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'balance': balance,
            'unrealized_pnl': unrealized_pnl,
            'total_value': balance + unrealized_pnl
        }
        
        self.history.append(snapshot)
        
        # Keep only last 365 days
        cutoff = datetime.now() - timedelta(days=365)
        self.history = [
            s for s in self.history
            if datetime.fromisoformat(s['timestamp']) > cutoff
        ]
        
        self._save_history()
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance from Binance Futures.
        
        Returns:
            Dict with balance info:
            {
                'total_balance': float,  # Total wallet balance
                'available_balance': float,  # Available for trading
                'margin_balance': float  # Used as margin
            }
        """
        try:
            # Get futures account info
            account = self.client.futures_account()
            
            total_balance = float(account.get('totalWalletBalance', 0))
            available_balance = float(account.get('availableBalance', 0))
            margin_balance = total_balance - available_balance
            
            return {
                'total_balance': total_balance,
                'available_balance': available_balance,
                'margin_balance': margin_balance
            }
            
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return {
                'total_balance': 0.0,
                'available_balance': 0.0,
                'margin_balance': 0.0
            }
    
    def get_open_positions(self) -> List[PositionInfo]:
        """
        Get all open positions from Binance Futures.
        
        Returns:
            List of PositionInfo objects
        """
        positions = []
        
        try:
            # Get position information
            position_info = self.client.futures_position_information()
            
            for pos in position_info:
                quantity = float(pos.get('positionAmt', 0))
                
                # Skip if no position
                if quantity == 0:
                    continue
                
                symbol = pos.get('symbol', '')
                entry_price = float(pos.get('entryPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                leverage = int(pos.get('leverage', 1))
                liquidation_price = float(pos.get('liquidationPrice', 0))
                
                # Determine side
                side = 'LONG' if quantity > 0 else 'SHORT'
                quantity = abs(quantity)
                
                # Calculate notional value
                notional_value = quantity * mark_price
                
                # Calculate P&L percentage
                if entry_price > 0:
                    pnl_pct = (unrealized_pnl / (quantity * entry_price)) * 100
                else:
                    pnl_pct = 0.0
                
                position = PositionInfo(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    current_price=mark_price,
                    quantity=quantity,
                    notional_value=notional_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=pnl_pct,
                    leverage=leverage,
                    liquidation_price=liquidation_price if liquidation_price > 0 else None
                )
                
                positions.append(position)
        
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
        
        return positions
    
    def calculate_performance_metrics(
        self,
        current_balance: float,
        unrealized_pnl: float
    ) -> Dict[str, float]:
        """
        Calculate performance metrics based on historical data.
        
        Args:
            current_balance: Current total balance
            unrealized_pnl: Current unrealized P&L
            
        Returns:
            Dict with performance metrics
        """
        if not self.history:
            return {
                'daily_pnl': 0.0,
                'daily_pnl_pct': 0.0,
                'weekly_pnl': 0.0,
                'weekly_pnl_pct': 0.0,
                'monthly_pnl': 0.0,
                'monthly_pnl_pct': 0.0,
                'apr': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0
            }
        
        current_value = current_balance + unrealized_pnl
        now = datetime.now()
        
        # Helper to find snapshot at time
        def find_snapshot(days_ago: int) -> Optional[Dict]:
            target = now - timedelta(days=days_ago)
            for snapshot in reversed(self.history):
                ts = datetime.fromisoformat(snapshot['timestamp'])
                if ts <= target:
                    return snapshot
            return self.history[0] if self.history else None
        
        # Daily P&L
        daily_snapshot = find_snapshot(1)
        if daily_snapshot:
            daily_start = daily_snapshot['total_value']
            daily_pnl = current_value - daily_start
            daily_pnl_pct = (daily_pnl / daily_start * 100) if daily_start > 0 else 0
        else:
            daily_pnl = 0.0
            daily_pnl_pct = 0.0
        
        # Weekly P&L
        weekly_snapshot = find_snapshot(7)
        if weekly_snapshot:
            weekly_start = weekly_snapshot['total_value']
            weekly_pnl = current_value - weekly_start
            weekly_pnl_pct = (weekly_pnl / weekly_start * 100) if weekly_start > 0 else 0
        else:
            weekly_pnl = 0.0
            weekly_pnl_pct = 0.0
        
        # Monthly P&L
        monthly_snapshot = find_snapshot(30)
        if monthly_snapshot:
            monthly_start = monthly_snapshot['total_value']
            monthly_pnl = current_value - monthly_start
            monthly_pnl_pct = (monthly_pnl / monthly_start * 100) if monthly_start > 0 else 0
        else:
            monthly_pnl = 0.0
            monthly_pnl_pct = 0.0
        
        # Annual Percentage Rate (APR)
        # Based on available historical data
        if len(self.history) > 1:
            first_snapshot = self.history[0]
            first_value = first_snapshot['total_value']
            first_time = datetime.fromisoformat(first_snapshot['timestamp'])
            
            days_elapsed = (now - first_time).days
            if days_elapsed > 0 and first_value > 0:
                total_return = (current_value - first_value) / first_value
                apr = (total_return / days_elapsed) * 365 * 100
            else:
                apr = 0.0
        else:
            apr = 0.0
        
        # Sharpe Ratio (simplified - using daily returns)
        if len(self.history) >= 7:
            daily_returns = []
            for i in range(1, len(self.history)):
                prev_val = self.history[i-1]['total_value']
                curr_val = self.history[i]['total_value']
                if prev_val > 0:
                    daily_ret = (curr_val - prev_val) / prev_val
                    daily_returns.append(daily_ret)
            
            if daily_returns:
                import statistics
                avg_return = statistics.mean(daily_returns)
                std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
                
                # Annualize
                if std_return > 0:
                    sharpe_ratio = (avg_return * 252) / (std_return * (252 ** 0.5))
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Max Drawdown
        if len(self.history) >= 2:
            peak = self.history[0]['total_value']
            max_dd = 0.0
            max_dd_pct = 0.0
            
            for snapshot in self.history:
                value = snapshot['total_value']
                if value > peak:
                    peak = value
                
                dd = peak - value
                dd_pct = (dd / peak * 100) if peak > 0 else 0
                
                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct
        else:
            max_dd = 0.0
            max_dd_pct = 0.0
        
        return {
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'weekly_pnl': weekly_pnl,
            'weekly_pnl_pct': weekly_pnl_pct,
            'monthly_pnl': monthly_pnl,
            'monthly_pnl_pct': monthly_pnl_pct,
            'apr': apr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct
        }
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """
        Get trading statistics (win rate, etc.).
        
        Returns:
            Dict with trading stats
        """
        # TODO: Implement based on trade history
        # For now, return placeholders
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0
        }
    
    def get_portfolio_stats(self) -> PortfolioStats:
        """
        Get complete portfolio statistics.
        
        Returns:
            PortfolioStats object with all metrics
        """
        # Get balance
        balance_info = self.get_account_balance()
        
        # Get positions
        positions = self.get_open_positions()
        
        # Calculate total unrealized P&L
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        
        # Calculate unrealized P&L percentage
        total_balance = balance_info['total_balance']
        if total_balance > 0:
            unrealized_pnl_pct = (total_unrealized_pnl / total_balance) * 100
        else:
            unrealized_pnl_pct = 0.0
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(
            current_balance=total_balance,
            unrealized_pnl=total_unrealized_pnl
        )
        
        # Get trading stats
        trading_stats = self.get_trading_stats()
        
        # Save snapshot
        self._save_snapshot(total_balance, total_unrealized_pnl)
        
        # Build stats object
        stats = PortfolioStats(
            # Balance
            total_balance=total_balance,
            available_balance=balance_info['available_balance'],
            margin_balance=balance_info['margin_balance'],
            
            # Positions
            open_positions=positions,
            total_unrealized_pnl=total_unrealized_pnl,
            total_unrealized_pnl_pct=unrealized_pnl_pct,
            
            # Performance
            daily_pnl=performance['daily_pnl'],
            daily_pnl_pct=performance['daily_pnl_pct'],
            weekly_pnl=performance['weekly_pnl'],
            weekly_pnl_pct=performance['weekly_pnl_pct'],
            monthly_pnl=performance['monthly_pnl'],
            monthly_pnl_pct=performance['monthly_pnl_pct'],
            
            # Annual metrics
            apr=performance['apr'],
            sharpe_ratio=performance['sharpe_ratio'],
            max_drawdown=performance['max_drawdown'],
            max_drawdown_pct=performance['max_drawdown_pct'],
            
            # Trading stats
            total_trades=trading_stats['total_trades'],
            winning_trades=trading_stats['winning_trades'],
            losing_trades=trading_stats['losing_trades'],
            win_rate=trading_stats['win_rate'],
            
            # Timestamp
            timestamp=datetime.now()
        )
        
        return stats
    
    def log_portfolio_summary(self):
        """Log beautiful portfolio summary with all metrics."""
        try:
            stats = self.get_portfolio_stats()
            
            # Format currency
            def fmt_usd(val: float) -> str:
                if abs(val) < 0.01:
                    return f"${val:.4f}"
                return f"${val:,.2f}"
            
            # Format percentage with color
            def fmt_pct(val: float) -> str:
                sign = "+" if val >= 0 else ""
                emoji = "🟢" if val >= 0 else "🔴"
                return f"{emoji} {sign}{val:.2f}%"
            
            # Build summary
            summary = f"\n{'='*80}\n"
            summary += f"💼 PORTFOLIO SUMMARY - {stats.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            summary += f"{'='*80}\n"
            
            # Balance section
            total_value = stats.total_balance + stats.total_unrealized_pnl
            summary += f"\n💰 BALANCE:\n"
            summary += f"  ├─ Total Balance: {fmt_usd(stats.total_balance)}\n"
            summary += f"  ├─ Available: {fmt_usd(stats.available_balance)}\n"
            summary += f"  ├─ Margin Used: {fmt_usd(stats.margin_balance)}\n"
            summary += f"  ├─ Unrealized P&L: {fmt_usd(stats.total_unrealized_pnl)} ({fmt_pct(stats.total_unrealized_pnl_pct)})\n"
            summary += f"  └─ Total Value: {fmt_usd(total_value)}\n"
            
            # Open positions
            if stats.open_positions:
                summary += f"\n📊 OPEN POSITIONS ({len(stats.open_positions)}):\n"
                for i, pos in enumerate(stats.open_positions, 1):
                    entry_emoji = "🟢" if pos.side == "LONG" else "🔴"
                    pnl_emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
                    
                    summary += f"  {i}. {entry_emoji} {pos.symbol} {pos.side} {pos.leverage}x\n"
                    summary += f"     ├─ Entry: {fmt_usd(pos.entry_price)} → Current: {fmt_usd(pos.current_price)}\n"
                    summary += f"     ├─ Quantity: {pos.quantity:.4f} (Notional: {fmt_usd(pos.notional_value)})\n"
                    summary += f"     ├─ P&L: {pnl_emoji} {fmt_usd(pos.unrealized_pnl)} ({pos.unrealized_pnl_pct:+.2f}%)\n"
                    if pos.liquidation_price:
                        summary += f"     └─ Liquidation: {fmt_usd(pos.liquidation_price)}\n"
            else:
                summary += f"\n📊 OPEN POSITIONS: None\n"
            
            # Performance section
            summary += f"\n📈 PERFORMANCE:\n"
            summary += f"  ├─ Daily: {fmt_usd(stats.daily_pnl)} {fmt_pct(stats.daily_pnl_pct)}\n"
            summary += f"  ├─ Weekly: {fmt_usd(stats.weekly_pnl)} {fmt_pct(stats.weekly_pnl_pct)}\n"
            summary += f"  ├─ Monthly: {fmt_usd(stats.monthly_pnl)} {fmt_pct(stats.monthly_pnl_pct)}\n"
            summary += f"  └─ APR: {stats.apr:+.2f}% 🎯\n"
            
            # Risk metrics
            summary += f"\n⚠️  RISK METRICS:\n"
            summary += f"  ├─ Max Drawdown: {fmt_usd(stats.max_drawdown)} ({stats.max_drawdown_pct:.2f}%)\n"
            summary += f"  └─ Sharpe Ratio: {stats.sharpe_ratio:.2f}\n"
            
            # Trading stats
            if stats.total_trades > 0:
                summary += f"\n🎲 TRADING STATS:\n"
                summary += f"  ├─ Total Trades: {stats.total_trades}\n"
                summary += f"  ├─ Winning: {stats.winning_trades} | Losing: {stats.losing_trades}\n"
                summary += f"  └─ Win Rate: {stats.win_rate:.1f}%\n"
            
            summary += f"\n{'='*80}\n"
            
            logger.info(summary)
            
        except Exception as e:
            logger.error(f"Failed to log portfolio summary: {e}", exc_info=True)


if __name__ == "__main__":
    # Test the portfolio tracker
    print("Portfolio Tracker module - use in live trading engine")
