"""
Trade Journal - Self-Learning System

Records every trade with full context for analysis and learning.
Helps the bot understand what works and what doesn't.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Complete record of a trade with all context."""
    
    # Trade basics
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    
    # P&L
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    commission: float
    
    # Market context at entry
    market_regime: str  # 'trend', 'flat', 'volatile'
    volatility: float
    adx: float
    rsi: float
    funding_rate: Optional[float]
    
    # Signal information
    signal_type: str  # Which IMBA signal triggered
    signal_confidence: float
    signals_agreed: int  # How many signals voted for this direction
    signals_total: int
    
    # Position management
    stop_loss: Optional[float]
    take_profit: Optional[float]
    exit_reason: Optional[str]  # 'stop_loss', 'take_profit', 'signal_reversal', 'timeout'
    
    # Additional metadata
    leverage: int
    risk_reward_ratio: Optional[float]
    hold_time_minutes: Optional[int]
    max_drawdown: Optional[float]  # During the trade
    max_profit: Optional[float]  # Peak profit during the trade
    
    # Learning tags
    is_winning: Optional[bool]
    quality_score: Optional[float]  # 0-1, how "good" was this trade
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert datetime to ISO string
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Create from dictionary."""
        # Convert ISO strings back to datetime
        if 'entry_time' in data and isinstance(data['entry_time'], str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if 'exit_time' in data and data['exit_time'] and isinstance(data['exit_time'], str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        return cls(**data)


class TradeJournal:
    """
    Records and manages trade history for self-learning.
    
    Features:
    - Records every trade with full context
    - Persistent storage (JSON and CSV)
    - Query and analysis capabilities
    - Pattern recognition data
    """
    
    def __init__(self, journal_dir: str = "data/trade_journal"):
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session file
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_file = self.journal_dir / f"session_{session_id}.jsonl"
        
        # In-memory cache
        self.trades: List[TradeRecord] = []
        
        # Load previous trades for analysis
        self._load_historical_trades()
        
        logger.info(f"Trade Journal initialized: {self.journal_dir}")
        logger.info(f"Loaded {len(self.trades)} historical trades")
    
    def record_trade_entry(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        signal_info: Dict[str, Any],
        market_context: Dict[str, Any],
        position_params: Dict[str, Any]
    ) -> TradeRecord:
        """Record a new trade entry."""
        
        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_time=datetime.now(),
            exit_time=None,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            pnl=None,
            pnl_percentage=None,
            commission=position_params.get('commission', 0.0),
            
            # Market context
            market_regime=market_context.get('regime', 'unknown'),
            volatility=market_context.get('volatility', 0.0),
            adx=market_context.get('adx', 0.0),
            rsi=market_context.get('rsi', 50.0),
            funding_rate=market_context.get('funding_rate'),
            
            # Signal info
            signal_type=signal_info.get('type', 'unknown'),
            signal_confidence=signal_info.get('confidence', 0.0),
            signals_agreed=signal_info.get('signals_agreed', 0),
            signals_total=signal_info.get('signals_total', 9),
            
            # Position management
            stop_loss=position_params.get('stop_loss'),
            take_profit=position_params.get('take_profit'),
            exit_reason=None,
            
            # Additional
            leverage=position_params.get('leverage', 1),
            risk_reward_ratio=position_params.get('risk_reward_ratio'),
            hold_time_minutes=None,
            max_drawdown=None,
            max_profit=None,
            
            # Learning
            is_winning=None,
            quality_score=None
        )
        
        self.trades.append(record)
        self._save_trade(record)
        
        logger.info(f"Trade entry recorded: {trade_id} - {side} {symbol} @ {entry_price}")
        return record
    
    def record_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        max_drawdown: Optional[float] = None,
        max_profit: Optional[float] = None
    ):
        """Record trade exit and calculate metrics."""
        
        # Find the trade
        trade = self._find_trade(trade_id)
        if not trade:
            logger.warning(f"Trade not found for exit: {trade_id}")
            return
        
        # Update trade record
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.pnl = pnl
        trade.pnl_percentage = (pnl / (trade.entry_price * trade.quantity)) * 100
        trade.is_winning = pnl > 0
        trade.max_drawdown = max_drawdown
        trade.max_profit = max_profit
        
        # Calculate hold time
        if trade.entry_time:
            hold_time = (trade.exit_time - trade.entry_time).total_seconds() / 60
            trade.hold_time_minutes = int(hold_time)
        
        # Calculate quality score
        trade.quality_score = self._calculate_quality_score(trade)
        
        # Save updated trade
        self._save_trade(trade)
        
        win_emoji = "✅" if trade.is_winning else "❌"
        logger.info(
            f"{win_emoji} Trade exit recorded: {trade_id} - "
            f"P&L: ${pnl:.2f} ({trade.pnl_percentage:.2f}%), "
            f"Quality: {trade.quality_score:.2f}"
        )
    
    def _find_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Find trade by ID."""
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None
    
    def _calculate_quality_score(self, trade: TradeRecord) -> float:
        """
        Calculate quality score (0-1) for a trade.
        
        Considers:
        - P&L (most important)
        - Risk/reward ratio
        - Exit reason (take_profit better than stop_loss)
        - Market regime match
        - Signal confidence
        """
        score = 0.5  # Base score
        
        # P&L contribution (40% weight)
        if trade.pnl_percentage:
            if trade.pnl_percentage > 0:
                score += min(0.4, trade.pnl_percentage / 10 * 0.4)  # Cap at 10% profit
            else:
                score += max(-0.4, trade.pnl_percentage / 10 * 0.4)  # Cap at -10% loss
        
        # Exit reason (20% weight)
        if trade.exit_reason == 'take_profit':
            score += 0.2
        elif trade.exit_reason == 'stop_loss':
            score -= 0.1
        elif trade.exit_reason == 'signal_reversal':
            score += 0.1  # Good if we exit on signal
        
        # Signal confidence (20% weight)
        if trade.signal_confidence:
            score += (trade.signal_confidence - 0.5) * 0.4  # -0.2 to +0.2
        
        # Signal agreement (10% weight)
        if trade.signals_total > 0:
            agreement_ratio = trade.signals_agreed / trade.signals_total
            score += (agreement_ratio - 0.5) * 0.2  # -0.1 to +0.1
        
        # Max profit capture (10% weight)
        if trade.max_profit and trade.pnl and trade.max_profit > 0:
            capture_ratio = trade.pnl / trade.max_profit
            score += capture_ratio * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _save_trade(self, trade: TradeRecord):
        """Save trade to persistent storage."""
        try:
            # Append to JSONL file (one JSON per line)
            with open(self.current_session_file, 'a', encoding='utf-8') as f:
                json.dump(trade.to_dict(), f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
    
    def _load_historical_trades(self):
        """Load all historical trades from journal directory."""
        jsonl_files = list(self.journal_dir.glob("session_*.jsonl"))
        
        for file_path in jsonl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            trade_dict = json.loads(line)
                            trade = TradeRecord.from_dict(trade_dict)
                            self.trades.append(trade)
            except Exception as e:
                logger.warning(f"Failed to load trades from {file_path}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall trading statistics."""
        completed_trades = [t for t in self.trades if t.exit_time is not None]
        
        if not completed_trades:
            return {"total_trades": 0}
        
        winning_trades = [t for t in completed_trades if t.is_winning]
        losing_trades = [t for t in completed_trades if not t.is_winning]
        
        total_pnl = sum(t.pnl for t in completed_trades if t.pnl)
        avg_quality = sum(t.quality_score for t in completed_trades if t.quality_score) / len(completed_trades)
        
        return {
            "total_trades": len(completed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(completed_trades) * 100,
            "total_pnl": total_pnl,
            "avg_win": sum(t.pnl for t in winning_trades if t.pnl) / max(1, len(winning_trades)),
            "avg_loss": sum(t.pnl for t in losing_trades if t.pnl) / max(1, len(losing_trades)),
            "avg_quality_score": avg_quality,
            "best_trade": max((t.pnl for t in completed_trades if t.pnl), default=0),
            "worst_trade": min((t.pnl for t in completed_trades if t.pnl), default=0),
        }
    
    def get_trades_by_signal(self, signal_type: str) -> List[TradeRecord]:
        """Get all trades for a specific signal type."""
        return [t for t in self.trades if t.signal_type == signal_type and t.exit_time]
    
    def get_trades_by_regime(self, regime: str) -> List[TradeRecord]:
        """Get all trades for a specific market regime."""
        return [t for t in self.trades if t.market_regime == regime and t.exit_time]
    
    def export_to_csv(self, output_path: str):
        """Export all trades to CSV for analysis."""
        if not self.trades:
            logger.warning("No trades to export")
            return
        
        df = pd.DataFrame([t.to_dict() for t in self.trades])
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(self.trades)} trades to {output_path}")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Analyze trades and extract learning insights.
        
        Returns what's working and what's not.
        """
        completed = [t for t in self.trades if t.exit_time is not None]
        
        if len(completed) < 10:
            return {"status": "insufficient_data", "trades_needed": 10 - len(completed)}
        
        insights = {}
        
        # Best performing signals
        signal_performance = {}
        for trade in completed:
            signal = trade.signal_type
            if signal not in signal_performance:
                signal_performance[signal] = {"wins": 0, "losses": 0, "pnl": 0}
            
            if trade.is_winning:
                signal_performance[signal]["wins"] += 1
            else:
                signal_performance[signal]["losses"] += 1
            signal_performance[signal]["pnl"] += trade.pnl or 0
        
        # Calculate win rates
        for signal, stats in signal_performance.items():
            total = stats["wins"] + stats["losses"]
            stats["win_rate"] = stats["wins"] / total * 100 if total > 0 else 0
        
        # Sort by win rate
        best_signals = sorted(
            signal_performance.items(),
            key=lambda x: x[1]["win_rate"],
            reverse=True
        )[:3]
        
        insights["best_signals"] = [
            {"signal": s, "win_rate": stats["win_rate"], "pnl": stats["pnl"]}
            for s, stats in best_signals
        ]
        
        # Best regime
        regime_performance = {}
        for trade in completed:
            regime = trade.market_regime
            if regime not in regime_performance:
                regime_performance[regime] = {"wins": 0, "losses": 0}
            
            if trade.is_winning:
                regime_performance[regime]["wins"] += 1
            else:
                regime_performance[regime]["losses"] += 1
        
        insights["best_regime"] = max(
            regime_performance.items(),
            key=lambda x: x[1]["wins"] / (x[1]["wins"] + x[1]["losses"])
        )[0] if regime_performance else "unknown"
        
        # Optimal confidence threshold
        high_conf_trades = [t for t in completed if t.signal_confidence >= 0.6]
        if high_conf_trades:
            high_conf_win_rate = sum(1 for t in high_conf_trades if t.is_winning) / len(high_conf_trades) * 100
            insights["high_confidence_win_rate"] = high_conf_win_rate
        
        # Average quality by exit reason
        exit_quality = {}
        for trade in completed:
            reason = trade.exit_reason
            if reason not in exit_quality:
                exit_quality[reason] = []
            exit_quality[reason].append(trade.quality_score)
        
        insights["exit_quality"] = {
            reason: sum(scores) / len(scores)
            for reason, scores in exit_quality.items()
            if scores
        }
        
        return insights
