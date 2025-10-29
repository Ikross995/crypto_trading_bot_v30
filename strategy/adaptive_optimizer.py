"""
Adaptive Optimizer - Self-Learning Parameter Tuning

Automatically adjusts trading parameters based on performance.
The bot learns what works and adapts in real-time.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from strategy.trade_journal import TradeJournal

logger = logging.getLogger(__name__)


class AdaptiveOptimizer:
    """
    Automatically optimizes trading parameters based on performance.
    
    Features:
    - Learns from trade history
    - Adjusts signal weights dynamically
    - Adapts confidence thresholds
    - Optimizes position sizing
    - Self-corrects based on win/loss patterns
    """
    
    def __init__(
        self,
        journal: TradeJournal,
        config,
        optimization_interval_hours: int = 24,
        min_trades_for_optimization: int = 20
    ):
        self.journal = journal
        self.config = config
        self.optimization_interval = timedelta(hours=optimization_interval_hours)
        self.min_trades = min_trades_for_optimization
        
        # Last optimization time
        self.last_optimization = datetime.now()
        
        # Learning state
        self.state_file = Path("data/optimizer_state.json")
        self.state = self._load_state()
        
        # Parameter history
        self.parameter_history: List[Dict] = []
        
        logger.info(f"Adaptive Optimizer initialized (interval: {optimization_interval_hours}h)")
    
    def should_optimize(self) -> bool:
        """Check if it's time to run optimization."""
        time_since_last = datetime.now() - self.last_optimization
        
        completed_trades = len([t for t in self.journal.trades if t.exit_time])
        
        return (
            time_since_last >= self.optimization_interval and
            completed_trades >= self.min_trades
        )
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization based on recent performance.
        
        Returns:
            Dictionary with optimized parameters
        """
        logger.info("Starting adaptive optimization...")
        
        insights = self.journal.get_learning_insights()
        
        if insights.get("status") == "insufficient_data":
            logger.info(f"Insufficient data for optimization. Need {insights.get('trades_needed')} more trades.")
            return {}
        
        optimizations = {}
        
        # 1. Optimize confidence threshold
        new_confidence = self._optimize_confidence_threshold()
        if new_confidence != self.config.bt_conf_min:
            optimizations["bt_conf_min"] = new_confidence
            logger.info(f"Adjusted confidence threshold: {self.config.bt_conf_min:.2f} -> {new_confidence:.2f}")
        
        # 2. Optimize signal weights
        new_weights = self._optimize_signal_weights(insights)
        if new_weights:
            optimizations["signal_weights"] = new_weights
            logger.info(f"Adjusted signal weights based on performance")
        
        # 3. Optimize regime thresholds
        new_adx = self._optimize_regime_thresholds(insights)
        if new_adx:
            optimizations.update(new_adx)
            logger.info(f"Adjusted regime detection thresholds")
        
        # 4. Optimize risk parameters
        new_risk = self._optimize_risk_parameters()
        if new_risk:
            optimizations.update(new_risk)
            logger.info(f"Adjusted risk parameters")
        
        # Apply optimizations
        if optimizations:
            self._apply_optimizations(optimizations)
            self._save_state()
        
        self.last_optimization = datetime.now()
        
        logger.info(f"Optimization complete. Applied {len(optimizations)} adjustments.")
        return optimizations
    
    def _optimize_confidence_threshold(self) -> float:
        """
        Optimize minimum confidence threshold based on results.
        
        Strategy:
        - If win rate is high (>60%), slightly decrease threshold to get more trades
        - If win rate is low (<45%), increase threshold to be more selective
        - Keep within safe bounds (0.2 - 0.7)
        """
        stats = self.journal.get_statistics()
        
        if stats["total_trades"] < 10:
            return self.config.bt_conf_min
        
        win_rate = stats["win_rate"]
        current_threshold = self.config.bt_conf_min
        
        if win_rate > 60:
            # High win rate - be slightly more aggressive
            new_threshold = max(0.2, current_threshold - 0.05)
        elif win_rate < 45:
            # Low win rate - be more selective
            new_threshold = min(0.7, current_threshold + 0.05)
        else:
            # Good range - minor adjustments
            if win_rate > 55:
                new_threshold = current_threshold - 0.02
            elif win_rate < 50:
                new_threshold = current_threshold + 0.02
            else:
                new_threshold = current_threshold
        
        return round(new_threshold, 2)
    
    def _optimize_signal_weights(self, insights: Dict) -> Optional[Dict[str, float]]:
        """
        Optimize weights for each IMBA signal based on performance.
        
        Signals that perform well get higher weights.
        """
        if "best_signals" not in insights:
            return None
        
        best_signals = insights["best_signals"]
        
        # Create weight adjustments
        weights = {}
        for signal_info in best_signals:
            signal = signal_info["signal"]
            win_rate = signal_info["win_rate"]
            
            # Calculate weight based on performance
            # 50% win rate = 1.0 weight (neutral)
            # 70% win rate = 1.4 weight (boost)
            # 30% win rate = 0.6 weight (reduce)
            weight = 0.6 + (win_rate / 100) * 0.8
            weights[signal] = round(weight, 2)
        
        return weights if len(weights) > 0 else None
    
    def _optimize_regime_thresholds(self, insights: Dict) -> Optional[Dict]:
        """
        Optimize ADX thresholds based on which regimes perform best.
        """
        if "best_regime" not in insights:
            return None
        
        best_regime = insights["best_regime"]
        adjustments = {}
        
        if best_regime == "trend":
            # Trend works well - lower trend threshold to catch more trends
            new_trend = max(20.0, self.config.trend_adx_threshold - 2.0)
            if new_trend != self.config.trend_adx_threshold:
                adjustments["trend_adx_threshold"] = new_trend
        
        elif best_regime == "flat":
            # Flat works well - raise flat threshold
            new_flat = min(25.0, self.config.flat_adx_threshold + 2.0)
            if new_flat != self.config.flat_adx_threshold:
                adjustments["flat_adx_threshold"] = new_flat
        
        return adjustments if adjustments else None
    
    def _optimize_risk_parameters(self) -> Optional[Dict]:
        """
        Optimize risk parameters based on performance.
        """
        stats = self.journal.get_statistics()
        
        if stats["total_trades"] < 20:
            return None
        
        adjustments = {}
        win_rate = stats["win_rate"]
        avg_quality = stats.get("avg_quality_score", 0.5)
        
        # Adjust risk per trade based on quality
        current_risk = getattr(self.config, 'risk_per_trade_pct', 1.0)
        
        if avg_quality > 0.7 and win_rate > 55:
            # High quality trades - slightly increase risk
            new_risk = min(2.0, current_risk + 0.1)
            if new_risk != current_risk:
                adjustments["risk_per_trade_pct"] = new_risk
        
        elif avg_quality < 0.4 or win_rate < 45:
            # Low quality trades - reduce risk
            new_risk = max(0.3, current_risk - 0.1)
            if new_risk != current_risk:
                adjustments["risk_per_trade_pct"] = new_risk
        
        return adjustments if adjustments else None
    
    def _apply_optimizations(self, optimizations: Dict):
        """Apply optimized parameters to config."""
        for param, value in optimizations.items():
            if hasattr(self.config, param):
                old_value = getattr(self.config, param)
                setattr(self.config, param, value)
                
                logger.info(f"Parameter updated: {param} = {old_value} -> {value}")
                
                # Record in history
                self.parameter_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "parameter": param,
                    "old_value": old_value,
                    "new_value": value,
                    "reason": "adaptive_optimization"
                })
    
    def _save_state(self):
        """Save optimizer state to file."""
        try:
            state = {
                "last_optimization": self.last_optimization.isoformat(),
                "parameter_history": self.parameter_history[-50:],  # Keep last 50
                "total_optimizations": len(self.parameter_history)
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save optimizer state: {e}")
    
    def _load_state(self) -> Dict:
        """Load optimizer state from file."""
        if not self.state_file.exists():
            return {}
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                
            if "last_optimization" in state:
                self.last_optimization = datetime.fromisoformat(state["last_optimization"])
            
            if "parameter_history" in state:
                self.parameter_history = state["parameter_history"]
            
            logger.info(f"Loaded optimizer state: {state.get('total_optimizations', 0)} optimizations")
            return state
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
            return {}
    
    def get_optimization_report(self) -> str:
        """Generate a report of recent optimizations."""
        if not self.parameter_history:
            return "No optimizations yet"
        
        report = ["=== Adaptive Optimization Report ===\n"]
        report.append(f"Total Optimizations: {len(self.parameter_history)}\n")
        report.append(f"Last Optimization: {self.last_optimization.strftime('%Y-%m-%d %H:%M')}\n\n")
        
        report.append("Recent Changes:\n")
        for entry in self.parameter_history[-10:]:
            report.append(
                f"  {entry['timestamp'][:19]} - {entry['parameter']}: "
                f"{entry['old_value']} -> {entry['new_value']}\n"
            )
        
        return "".join(report)
    
    def auto_optimize_if_needed(self) -> bool:
        """
        Automatically run optimization if conditions are met.
        
        Returns:
            True if optimization was performed
        """
        if self.should_optimize():
            try:
                optimizations = self.optimize()
                return len(optimizations) > 0
            except Exception as e:
                logger.error(f"Auto-optimization failed: {e}")
                return False
        return False


class RealTimeAdaptation:
    """
    Real-time adaptation based on immediate feedback.
    
    Makes micro-adjustments during trading without waiting for
    full optimization cycle.
    """
    
    def __init__(self, journal: TradeJournal):
        self.journal = journal
        self.recent_trades_window = 10  # Look at last N trades
    
    def get_real_time_adjustments(self) -> Dict[str, Any]:
        """
        Get real-time adjustments based on very recent performance.
        
        Used for quick reactions to changing conditions.
        """
        completed = [t for t in self.journal.trades if t.exit_time]
        
        if len(completed) < self.recent_trades_window:
            return {}
        
        recent = completed[-self.recent_trades_window:]
        
        adjustments = {}
        
        # Check recent win rate
        recent_wins = sum(1 for t in recent if t.is_winning)
        recent_win_rate = recent_wins / len(recent) * 100
        
        if recent_win_rate < 30:
            # Poor recent performance - pause or reduce risk
            adjustments["confidence_multiplier"] = 1.2  # Require 20% higher confidence
            adjustments["risk_multiplier"] = 0.5  # Half the risk
            adjustments["alert"] = "Poor recent performance - reducing activity"
        
        elif recent_win_rate > 70:
            # Excellent recent performance - slightly more aggressive
            adjustments["confidence_multiplier"] = 0.9  # Allow 10% lower confidence
            adjustments["risk_multiplier"] = 1.1  # Slightly increase risk
            adjustments["alert"] = "Excellent recent performance - slightly more aggressive"
        
        return adjustments
    
    def should_pause_trading(self) -> bool:
        """
        Determine if trading should be paused due to poor performance.
        """
        completed = [t for t in self.journal.trades if t.exit_time]
        
        if len(completed) < 5:
            return False
        
        recent = completed[-5:]
        recent_losses = sum(1 for t in recent if not t.is_winning)
        
        # Pause if last 5 trades are all losses
        return recent_losses >= 5
