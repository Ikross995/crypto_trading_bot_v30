"""
🧠 AI STATUS MONITOR
===================

Real-time monitoring and display of AI system status with clear indicators.
Shows exactly what the AI is doing, learning progress, and performance.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

class AIStatusMonitor:
    """Monitors and displays AI system status in real-time"""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.predictions_made = 0
        self.trades_learned = 0
        self.ai_blocks = 0
        self.ai_approvals = 0
        self.ml_adjustments = 0
        self.last_prediction = None
        self.last_learning = None
        self.prediction_accuracy_history = []
        self.feature_importance_cache = {}
        
        # Performance tracking
        self.prediction_times = []
        self.learning_times = []
        
        logger.info("🧠 [AI_MONITOR] AI Status Monitor initialized")
    
    def log_prediction(self, symbol: str, predictions: Dict[str, Any], 
                      decision: Dict[str, Any], processing_time: float):
        """Log ML prediction with clear indicators"""
        
        self.predictions_made += 1
        self.last_prediction = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc),
            'predictions': predictions,
            'decision': decision,
            'processing_time': processing_time
        }
        self.prediction_times.append(processing_time)
        
        # Extract key metrics
        expected_pnl = predictions.get('expected_pnl_pct', 0)
        win_prob = predictions.get('win_probability', 0.5) * 100
        confidence = predictions.get('prediction_confidence', 0)
        risk_level = decision.get('risk_level', 'unknown')
        should_trade = decision.get('should_trade', False)
        
        # Create visual indicators
        pnl_emoji = "📈" if expected_pnl > 0 else "📉"
        prob_emoji = "🟢" if win_prob > 60 else "🟡" if win_prob > 40 else "🔴"
        conf_emoji = "⭐" if confidence > 0.7 else "✨" if confidence > 0.4 else "💫"
        trade_emoji = "✅" if should_trade else "🚫"
        
        logger.info("=" * 80)
        logger.info("🧠 [AI_PREDICTION] #%d - %s", self.predictions_made, symbol)
        logger.info("=" * 80)
        
        # 🧠 Cold start information
        if self.predictions_made < 50:
            logger.info("🎯 [COLD_START] Exploration mode: %d/50 samples - IMBA signals only", self.predictions_made)
        elif self.predictions_made < 200:
            progress = (self.predictions_made - 50) / 150
            logger.info("🎓 [LEARNING] Learning mode: %d/200 samples - Gradual ML integration (%.1f%%)", 
                       self.predictions_made, progress * 100)
        logger.info("%s Expected PnL: %+.2f%%", pnl_emoji, expected_pnl)
        logger.info("%s Win Probability: %.0f%%", prob_emoji, win_prob)
        logger.info("%s ML Confidence: %.2f", conf_emoji, confidence)
        logger.info("🛡️ Risk Level: %s", risk_level.upper())
        logger.info("%s Decision: %s", trade_emoji, "TRADE" if should_trade else "SKIP")
        logger.info("⚡ Processing: %.2fms", processing_time * 1000)
        logger.info("=" * 80)
        
        # Track approvals vs blocks
        if should_trade:
            self.ai_approvals += 1
            logger.info("🎯 [AI_DECISION] APPROVED for trading (Total: %d)", self.ai_approvals)
        else:
            self.ai_blocks += 1
            logger.info("🚫 [AI_DECISION] BLOCKED trade (Total: %d blocks)", self.ai_blocks)
        
        # Show approval rate
        total_decisions = self.ai_approvals + self.ai_blocks
        if total_decisions > 0:
            approval_rate = (self.ai_approvals / total_decisions) * 100
            logger.info("📊 [AI_STATS] Approval Rate: %.1f%% (%d/%d)", 
                       approval_rate, self.ai_approvals, total_decisions)
    
    def log_position_adjustment(self, symbol: str, original_strength: float, 
                              ml_multiplier: float, new_strength: float):
        """Log ML position size adjustment"""
        
        self.ml_adjustments += 1
        
        if ml_multiplier > 1.0:
            adj_emoji = "📈"
            adj_text = "INCREASED"
        elif ml_multiplier < 1.0:
            adj_emoji = "📉" 
            adj_text = "DECREASED"
        else:
            adj_emoji = "➡️"
            adj_text = "UNCHANGED"
        
        logger.info("🎛️ [ML_SIZING] %s %s position size:", adj_emoji, adj_text)
        logger.info("    Signal: %.2f → %.2f (ML × %.2f)", 
                   original_strength, new_strength, ml_multiplier)
        logger.info("    Total adjustments: %d", self.ml_adjustments)
    
    def log_learning_event(self, symbol: str, trade_outcome: Dict[str, Any], 
                          models_updated: List[str], learning_time: float):
        """Log ML learning from completed trade"""
        
        self.trades_learned += 1
        self.last_learning = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc),
            'outcome': trade_outcome,
            'models_updated': models_updated,
            'learning_time': learning_time
        }
        self.learning_times.append(learning_time)
        
        pnl = trade_outcome.get('pnl_pct', 0)
        hold_time = trade_outcome.get('hold_time_minutes', 0)
        exit_reason = trade_outcome.get('exit_reason', 'unknown')
        
        # Visual indicators
        pnl_emoji = "💚" if pnl > 0 else "❤️" if pnl < 0 else "💙"
        learn_emoji = "🧠"
        
        logger.info("🔄" + "=" * 79)
        logger.info("%s [AI_LEARNING] #%d - %s", learn_emoji, self.trades_learned, symbol)
        logger.info("🔄" + "=" * 79)
        logger.info("%s Trade Result: %+.2f%% PnL", pnl_emoji, pnl)
        logger.info("⏱️ Hold Time: %.1f minutes", hold_time)
        logger.info("🚪 Exit: %s", exit_reason)
        logger.info("🔧 Models Updated: %s", ", ".join(models_updated))
        logger.info("⚡ Learning Time: %.2fms", learning_time * 1000)
        logger.info("📈 Total Trades Learned: %d", self.trades_learned)
        logger.info("🔄" + "=" * 79)
    
    def log_prediction_accuracy(self, predicted_pnl: float, actual_pnl: float, 
                               predicted_win_prob: float, actual_win: bool):
        """Track and log prediction accuracy"""
        
        pnl_error = abs(predicted_pnl - actual_pnl)
        win_correct = (predicted_win_prob > 0.5) == actual_win
        
        self.prediction_accuracy_history.append({
            'pnl_error': pnl_error,
            'win_correct': win_correct,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Keep only last 50 predictions for accuracy calculation
        if len(self.prediction_accuracy_history) > 50:
            self.prediction_accuracy_history = self.prediction_accuracy_history[-50:]
        
        # Calculate accuracy metrics
        if len(self.prediction_accuracy_history) >= 5:
            recent_errors = [p['pnl_error'] for p in self.prediction_accuracy_history[-20:]]
            recent_wins = [p['win_correct'] for p in self.prediction_accuracy_history[-20:]]
            
            avg_pnl_error = np.mean(recent_errors)
            win_accuracy = np.mean(recent_wins) * 100
            
            # Visual indicators for accuracy
            acc_emoji = "🎯" if win_accuracy > 60 else "🎪" if win_accuracy > 50 else "🎲"
            error_emoji = "✅" if avg_pnl_error < 1.0 else "⚠️" if avg_pnl_error < 2.0 else "❌"
            
            logger.info("📊 [ACCURACY_UPDATE] Recent ML Performance:")
            logger.info("    %s Win Prediction: %.1f%% accurate", acc_emoji, win_accuracy)
            logger.info("    %s PnL Error: %.2f%% average", error_emoji, avg_pnl_error)
            logger.info("    📈 Sample Size: %d predictions", len(recent_wins))
    
    def update_feature_importance(self, importance_dict: Dict[str, float]):
        """Update and display feature importance"""
        
        self.feature_importance_cache = importance_dict.copy()
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("🔍 [FEATURE_IMPORTANCE] Top ML Features:")
        for i, (feature, importance) in enumerate(sorted_features[:5]):
            bar_length = int(importance * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            logger.info("    %d. %s │%s│ %.3f", i+1, feature.ljust(15), bar, importance)
    
    def get_ai_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive AI status summary"""
        
        uptime = datetime.now(timezone.utc) - self.start_time
        total_decisions = self.ai_approvals + self.ai_blocks
        
        return {
            'uptime_minutes': uptime.total_seconds() / 60,
            'predictions_made': self.predictions_made,
            'trades_learned': self.trades_learned,
            'ai_approvals': self.ai_approvals,
            'ai_blocks': self.ai_blocks,
            'approval_rate': (self.ai_approvals / max(1, total_decisions)) * 100,
            'ml_adjustments': self.ml_adjustments,
            'avg_prediction_time_ms': np.mean(self.prediction_times[-100:]) * 1000 if self.prediction_times else 0,
            'avg_learning_time_ms': np.mean(self.learning_times[-50:]) * 1000 if self.learning_times else 0,
            'prediction_accuracy': self._get_recent_accuracy(),
            'last_prediction': self.last_prediction,
            'last_learning': self.last_learning,
            'feature_importance': self.feature_importance_cache,
            'status': 'ACTIVE' if self.predictions_made > 0 else 'WAITING'
        }
    
    def _get_recent_accuracy(self) -> Dict[str, float]:
        """Calculate recent prediction accuracy metrics"""
        
        if len(self.prediction_accuracy_history) < 5:
            return {'win_accuracy': 0.0, 'pnl_error': 0.0, 'sample_size': 0}
        
        recent = self.prediction_accuracy_history[-20:]
        win_accuracy = np.mean([p['win_correct'] for p in recent]) * 100
        pnl_error = np.mean([p['pnl_error'] for p in recent])
        
        return {
            'win_accuracy': win_accuracy,
            'pnl_error': pnl_error,
            'sample_size': len(recent)
        }
    
    def log_periodic_status(self):
        """Log periodic AI status summary"""
        
        status = self.get_ai_status_summary()
        
        logger.info("🤖" + "=" * 79)
        logger.info("🤖 [AI_STATUS] Periodic Status Report")
        logger.info("🤖" + "=" * 79)
        logger.info("⏰ Uptime: %.1f minutes", status['uptime_minutes'])
        logger.info("🧠 Predictions Made: %d", status['predictions_made'])
        logger.info("📚 Trades Learned: %d", status['trades_learned'])
        logger.info("✅ Approvals: %d | 🚫 Blocks: %d (%.1f%% approval)", 
                   status['ai_approvals'], status['ai_blocks'], status['approval_rate'])
        logger.info("🎛️ Position Adjustments: %d", status['ml_adjustments'])
        logger.info("⚡ Avg Processing: %.1fms predictions, %.1fms learning", 
                   status['avg_prediction_time_ms'], status['avg_learning_time_ms'])
        
        accuracy = status['prediction_accuracy']
        if accuracy['sample_size'] > 0:
            logger.info("🎯 Accuracy: %.1f%% wins, %.2f%% PnL error (%d samples)", 
                       accuracy['win_accuracy'], accuracy['pnl_error'], accuracy['sample_size'])
        else:
            logger.info("🎯 Accuracy: Collecting data...")
        
        logger.info("📊 Status: %s", status['status'])
        logger.info("🤖" + "=" * 79)
    
    def create_visual_dashboard_data(self) -> Dict[str, Any]:
        """Create data for visual dashboard display"""
        
        status = self.get_ai_status_summary()
        
        # Create visual indicators
        dashboard_data = {
            'ai_health': {
                'status': status['status'],
                'color': 'green' if status['status'] == 'ACTIVE' else 'orange',
                'icon': '🧠' if status['status'] == 'ACTIVE' else '💤'
            },
            'predictions': {
                'total': status['predictions_made'],
                'rate_per_minute': status['predictions_made'] / max(1, status['uptime_minutes']),
                'avg_time_ms': status['avg_prediction_time_ms']
            },
            'learning': {
                'total': status['trades_learned'],
                'rate_per_hour': status['trades_learned'] / max(1, status['uptime_minutes'] / 60),
                'avg_time_ms': status['avg_learning_time_ms']
            },
            'decisions': {
                'approvals': status['ai_approvals'],
                'blocks': status['ai_blocks'],
                'approval_rate': status['approval_rate'],
                'adjustments': status['ml_adjustments']
            },
            'accuracy': status['prediction_accuracy'],
            'features': status['feature_importance'],
            'last_activity': {
                'prediction': status['last_prediction'],
                'learning': status['last_learning']
            }
        }
        
        return dashboard_data

# Global instance for easy access
ai_monitor = AIStatusMonitor()