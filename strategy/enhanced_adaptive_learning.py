"""
🧠 ENHANCED ADAPTIVE LEARNING SYSTEM
====================================

Продвинутая замена для adaptive_learning.py с:
- Реальным машинным обучением
- Богатым feature engineering
- Онлайн обучением в реальном времени
- Multi-objective optimization
- Intelligent recommendations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path

from strategy.ml_learning_system import AdvancedMLLearningSystem, MarketContext, TradeOutcome
from strategy.market_context_collector import MarketContextCollector
from strategy.adaptive_learning import TradeRecord, PerformanceMetrics

logger = logging.getLogger(__name__)

class EnhancedAdaptiveLearningSystem:
    """Продвинутая система адаптивного обучения с ML"""
    
    def __init__(self, config):
        self.config = config
        
        # Инициализируем компоненты ML системы
        self.ml_system = AdvancedMLLearningSystem(config)
        self.context_collector = MarketContextCollector()
        
        # Кеш для текущего контекста рынка
        self.current_market_context = None
        self.recent_performance_cache = {}
        
        # Улучшенные метрики
        self.enhanced_metrics = {
            'prediction_accuracy': 0.5,
            'ml_confidence': 0.0,
            'feature_importance': {},
            'model_performance': {},
            'recommendation_success_rate': 0.5
        }
        
        # Статистика предсказаний
        self.prediction_history = []
        self.recommendation_history = []
        
        logger.info("🧠 [ENHANCED_ML] Advanced adaptive learning system initialized")
        
    async def analyze_signal_context(self, 
                                   symbol: str, 
                                   candles_data: pd.DataFrame,
                                   current_price: float,
                                   signal_strength: float,
                                   additional_context: Dict = None) -> Dict[str, Any]:
        """Анализирует контекст сигнала с ML предсказаниями"""
        
        try:
            # Собираем полный контекст рынка
            self.current_market_context = await self.context_collector.collect_market_context(
                symbol=symbol,
                candles_data=candles_data,
                current_price=current_price,
                fear_greed_index=additional_context.get('fear_greed_index', 50) if additional_context else 50,
                btc_dominance=additional_context.get('btc_dominance', 50.0) if additional_context else 50.0
            )
            
            # Получаем недавнюю производительность
            recent_performance = self._get_recent_performance_metrics()
            
            # ML предсказания результата сделки
            ml_predictions = await self.ml_system.predict_trade_outcome(
                market_context=self.current_market_context,
                signal_strength=signal_strength,
                recent_performance=recent_performance
            )
            
            # AI рекомендации
            ai_recommendations = await self.ml_system.get_intelligent_recommendations(
                current_market=self.current_market_context,
                recent_performance=recent_performance
            )
            
            # Создаем улучшенный анализ
            enhanced_analysis = {
                'signal_strength': signal_strength,
                'market_context': {
                    'regime': self.current_market_context.market_regime,
                    'volatility_percentile': self.current_market_context.volatility_percentile,
                    'trend_strength': self.current_market_context.trend_strength,
                    'session': self.current_market_context.session,
                    'support_distance': self.current_market_context.support_distance,
                    'resistance_distance': self.current_market_context.resistance_distance
                },
                'ml_predictions': ml_predictions,
                'ai_recommendations': ai_recommendations,
                'trading_decision': self._make_enhanced_trading_decision(
                    signal_strength, ml_predictions, ai_recommendations
                ),
                'risk_assessment': self._assess_comprehensive_risk(
                    self.current_market_context, ml_predictions
                )
            }
            
            logger.info(f"🎯 [ENHANCED_ANALYSIS] {symbol}: "
                       f"Expected {ml_predictions['expected_pnl_pct']:+.2f}% PnL, "
                       f"{ml_predictions['win_probability']:.0%} win prob, "
                       f"Confidence: {ml_predictions['prediction_confidence']:.2f}")
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"❌ [ENHANCED_ANALYSIS] Error analyzing signal context: {e}")
            return self._get_fallback_analysis(signal_strength)
    
    async def record_trade_with_ml(self, trade_record: TradeRecord, candles_data: pd.DataFrame = None):
        """Записывает сделку и обучает ML модели"""
        
        try:
            # Если нет контекста, собираем базовый
            if not self.current_market_context and candles_data is not None:
                self.current_market_context = await self.context_collector.collect_market_context(
                    symbol=trade_record.symbol,
                    candles_data=candles_data,
                    current_price=trade_record.entry_price
                )
            
            # Сохраняем предсказание для последующей оценки
            if hasattr(trade_record, 'ml_prediction'):
                self.prediction_history.append({
                    'trade_id': trade_record.trade_id,
                    'predicted_pnl': trade_record.ml_prediction.get('expected_pnl_pct', 0),
                    'predicted_win_prob': trade_record.ml_prediction.get('win_probability', 0.5),
                    'actual_pnl': None,  # Будет заполнено при закрытии
                    'timestamp': datetime.now(timezone.utc)
                })
            
            logger.info(f"📝 [ENHANCED_RECORD] {trade_record.symbol} {trade_record.side}: "
                       f"Entry @ ${trade_record.entry_price:.2f}, Qty: {trade_record.quantity}")
                       
        except Exception as e:
            logger.error(f"❌ [ENHANCED_RECORD] Error recording trade: {e}")
    
    async def update_trade_exit_with_ml(self, 
                                      trade_record: TradeRecord,
                                      exit_price: float,
                                      exit_reason: str,
                                      candles_data: pd.DataFrame = None):
        """Обновляет завершенную сделку и обучает ML модели"""
        
        try:
            # Создаем TradeOutcome для ML системы
            if self.current_market_context:
                trade_outcome = self.context_collector.create_trade_outcome(
                    trade_record=trade_record,
                    market_context=self.current_market_context,
                    exit_price=exit_price,
                    exit_reason=exit_reason
                )
                
                # Обучаем ML модели на завершенной сделке
                recent_performance = self._get_recent_performance_metrics()
                signal_strength = getattr(trade_record, 'signal_strength', 1.0)
                
                await self.ml_system.learn_from_trade(
                    market_context=self.current_market_context,
                    trade_outcome=trade_outcome,
                    signal_strength=signal_strength,
                    recent_performance=recent_performance
                )
                
                # Обновляем предсказания в истории
                self._update_prediction_accuracy(trade_record.trade_id, trade_record.pnl_pct)
                
                logger.info(f"🧠 [ML_LEARNING] Learned from {trade_record.symbol}: "
                           f"{trade_record.pnl_pct:+.2f}% PnL in {trade_record.hold_time_seconds/60:.1f} min")
                
                # Периодически оцениваем и обновляем метрики
                if len(self.prediction_history) % 10 == 0:
                    await self._update_enhanced_metrics()
                    
        except Exception as e:
            logger.error(f"❌ [ML_EXIT_UPDATE] Error updating trade exit: {e}")
    
    def _make_enhanced_trading_decision(self, 
                                      signal_strength: float,
                                      ml_predictions: Dict,
                                      ai_recommendations: Dict) -> Dict[str, Any]:
        """Принимает улучшенное торговое решение на основе ML"""
        
        try:
            # Базовые факторы
            base_confidence = signal_strength
            ml_confidence = ml_predictions.get('prediction_confidence', 0.0)
            expected_pnl = ml_predictions.get('expected_pnl_pct', 0.0)
            win_probability = ml_predictions.get('win_probability', 0.5)
            risk_score = ml_predictions.get('risk_score', 0.5)
            
            # 🧠 COLD START STRATEGY - Решаем проблему "курицы и яйца"
            should_trade = self._cold_start_decision_logic(
                signal_strength, ml_confidence, expected_pnl, 
                win_probability, risk_score
            )
            
            # Комбинированная уверенность (для логирования)
            combined_confidence = (base_confidence * 0.4 + 
                                 ml_confidence * 0.3 + 
                                 win_probability * 0.3)
            
            # Размер позиции на основе ML
            if should_trade:
                base_size = 1.0
                
                # ML adjustments
                if expected_pnl > 1.0 and win_probability > 0.7:
                    size_multiplier = 1.3  # Увеличиваем при высокой уверенности
                elif risk_score > 0.6:
                    size_multiplier = 0.7  # Уменьшаем при высоком риске
                else:
                    size_multiplier = 1.0
                
                # AI recommendations adjustments
                for rec in ai_recommendations.get('recommendations', []):
                    if rec['action'] == 'increase_position_size' and rec['confidence'] > 0.7:
                        size_multiplier *= 1.2
                    elif rec['action'] == 'reduce_exposure' and rec['confidence'] > 0.6:
                        size_multiplier *= 0.8
                
                position_size = base_size * size_multiplier
            else:
                position_size = 0.0
            
            decision = {
                'should_trade': should_trade,
                'position_size_multiplier': position_size,
                'combined_confidence': combined_confidence,
                'reasoning': self._generate_decision_reasoning(
                    signal_strength, ml_predictions, ai_recommendations, should_trade
                ),
                'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low'
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ [TRADING_DECISION] Error making decision: {e}")
            return {
                'should_trade': True,
                'position_size_multiplier': 1.0,
                'combined_confidence': signal_strength,
                'reasoning': 'Fallback decision due to error',
                'risk_level': 'medium'
            }
    
    def _assess_comprehensive_risk(self, market_context: MarketContext, ml_predictions: Dict) -> Dict[str, Any]:
        """Комплексная оценка риска"""
        
        try:
            risk_factors = {}
            
            # Рыночные риски
            if market_context.volatility_percentile > 80:
                risk_factors['high_volatility'] = market_context.volatility_percentile / 100
            
            if market_context.market_regime == 'volatile':
                risk_factors['volatile_regime'] = 0.8
            
            # Технические риски
            if market_context.support_distance < 1.0:
                risk_factors['near_support'] = 1.0 - market_context.support_distance / 5.0
                
            if market_context.resistance_distance < 1.0:
                risk_factors['near_resistance'] = 1.0 - market_context.resistance_distance / 5.0
            
            # ML риски
            ml_risk_score = ml_predictions.get('risk_score', 0.5)
            if ml_risk_score > 0.6:
                risk_factors['ml_high_risk'] = ml_risk_score
            
            prediction_confidence = ml_predictions.get('prediction_confidence', 0.5)
            if prediction_confidence < 0.3:
                risk_factors['low_ml_confidence'] = 1.0 - prediction_confidence
            
            # Временные риски
            if market_context.session == 'asian':
                risk_factors['low_liquidity_session'] = 0.3
            
            # Общий риск
            overall_risk = np.mean(list(risk_factors.values())) if risk_factors else 0.2
            
            return {
                'overall_risk_score': overall_risk,
                'risk_factors': risk_factors,
                'risk_level': 'high' if overall_risk > 0.7 else 'medium' if overall_risk > 0.4 else 'low',
                'recommendations': self._generate_risk_recommendations(risk_factors)
            }
            
        except Exception as e:
            logger.error(f"❌ [RISK_ASSESSMENT] Error: {e}")
            return {
                'overall_risk_score': 0.5,
                'risk_factors': {},
                'risk_level': 'medium',
                'recommendations': []
            }
    
    def _get_recent_performance_metrics(self) -> Dict[str, Any]:
        """Получает метрики недавней производительности"""
        
        try:
            # В реальной реализации здесь были бы данные из базы/файлов
            return {
                'today_pnl_pct': np.random.uniform(-2, 2),
                'recent_accuracy': self.enhanced_metrics.get('prediction_accuracy', 0.5),
                'recent_trades': len(self.prediction_history),
                'avg_hold_time': 45.0,  # минуты
                'win_rate': 0.6,
                'profit_factor': 1.3
            }
            
        except Exception as e:
            logger.error(f"❌ [PERFORMANCE_METRICS] Error: {e}")
            return {}
    
    def _update_prediction_accuracy(self, trade_id: str, actual_pnl: float):
        """Обновляет точность предсказаний"""
        
        try:
            for prediction in self.prediction_history:
                if prediction['trade_id'] == trade_id and prediction['actual_pnl'] is None:
                    prediction['actual_pnl'] = actual_pnl
                    
                    # Рассчитываем ошибку предсказания
                    predicted_pnl = prediction['predicted_pnl']
                    error = abs(actual_pnl - predicted_pnl)
                    prediction['error'] = error
                    
                    break
                    
        except Exception as e:
            logger.error(f"❌ [PREDICTION_UPDATE] Error: {e}")
    
    async def _update_enhanced_metrics(self):
        """Обновляет улучшенные метрики системы"""
        
        try:
            # Точность предсказаний
            completed_predictions = [p for p in self.prediction_history if p['actual_pnl'] is not None]
            
            if len(completed_predictions) > 5:
                errors = [p['error'] for p in completed_predictions[-20:]]  # Последние 20
                accuracy = 1.0 - (np.mean(errors) / 5.0)  # Нормализуем к 5% ошибке
                self.enhanced_metrics['prediction_accuracy'] = max(0.0, min(1.0, accuracy))
            
            # ML уверенность
            if hasattr(self.ml_system, 'models'):
                total_samples = sum(model.samples_seen for model in self.ml_system.models.values())
                self.enhanced_metrics['ml_confidence'] = min(1.0, total_samples / 1000)
            
            # Feature importance
            self.enhanced_metrics['feature_importance'] = self.ml_system._get_feature_importance()
            
            logger.debug(f"📊 [ENHANCED_METRICS] Updated: "
                        f"Accuracy: {self.enhanced_metrics['prediction_accuracy']:.2f}, "
                        f"ML Confidence: {self.enhanced_metrics['ml_confidence']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ [METRICS_UPDATE] Error: {e}")
    
    def _generate_decision_reasoning(self, signal_strength: float, ml_predictions: Dict, 
                                   ai_recommendations: Dict, should_trade: bool) -> str:
        """Генерирует объяснение торгового решения"""
        
        try:
            if should_trade:
                reasons = [
                    f"Signal strength: {signal_strength:.2f}",
                    f"Expected PnL: {ml_predictions.get('expected_pnl_pct', 0):+.2f}%",
                    f"Win probability: {ml_predictions.get('win_probability', 0.5):.0%}",
                    f"ML confidence: {ml_predictions.get('prediction_confidence', 0):.2f}"
                ]
                
                if ai_recommendations.get('recommendations'):
                    rec_count = len(ai_recommendations['recommendations'])
                    reasons.append(f"AI suggestions: {rec_count} recommendations")
                
                return f"✅ TRADE: {' | '.join(reasons)}"
            else:
                return f"❌ SKIP: Insufficient confidence or high risk"
                
        except Exception as e:
            return "Decision made with limited information due to error"
    
    def _generate_risk_recommendations(self, risk_factors: Dict) -> List[str]:
        """Генерирует рекомендации по управлению рисками"""
        
        recommendations = []
        
        try:
            if 'high_volatility' in risk_factors:
                recommendations.append("Consider reducing position size due to high volatility")
            
            if 'near_support' in risk_factors:
                recommendations.append("Watch for support break - consider tighter stop loss")
                
            if 'near_resistance' in risk_factors:
                recommendations.append("Close to resistance - consider taking profits early")
            
            if 'ml_high_risk' in risk_factors:
                recommendations.append("ML models indicate high risk environment")
            
            if 'low_ml_confidence' in risk_factors:
                recommendations.append("Low ML prediction confidence - proceed with caution")
                
            if 'low_liquidity_session' in risk_factors:
                recommendations.append("Asian session has lower liquidity - use limit orders")
                
        except Exception as e:
            logger.error(f"❌ [RISK_RECOMMENDATIONS] Error: {e}")
        
        return recommendations
    
    def _get_fallback_analysis(self, signal_strength: float) -> Dict[str, Any]:
        """Возвращает базовый анализ в случае ошибки"""
        
        return {
            'signal_strength': signal_strength,
            'market_context': {
                'regime': 'unknown',
                'volatility_percentile': 50.0,
                'trend_strength': 0.5,
                'session': 'unknown'
            },
            'ml_predictions': {
                'expected_pnl_pct': 0.0,
                'win_probability': 0.5,
                'expected_hold_time': 30.0,
                'risk_score': 0.5,
                'prediction_confidence': 0.0
            },
            'ai_recommendations': {
                'confidence': 0.0,
                'recommendations': []
            },
            'trading_decision': {
                'should_trade': True,
                'position_size_multiplier': 1.0,
                'combined_confidence': signal_strength,
                'reasoning': 'Fallback decision - ML system unavailable',
                'risk_level': 'medium'
            },
            'risk_assessment': {
                'overall_risk_score': 0.5,
                'risk_factors': {},
                'risk_level': 'medium',
                'recommendations': []
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Возвращает статус системы обучения"""
        
        try:
            return {
                'ml_system_status': 'active' if hasattr(self.ml_system, 'models') else 'inactive',
                'models_trained': len([m for m in self.ml_system.models.values() if m.is_fitted]),
                'total_predictions': len(self.prediction_history),
                'completed_predictions': len([p for p in self.prediction_history if p['actual_pnl'] is not None]),
                'enhanced_metrics': self.enhanced_metrics,
                'current_market_context': bool(self.current_market_context),
                'last_update': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ [SYSTEM_STATUS] Error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def save_all_data(self):
        """Сохраняет все данные системы"""
        
        try:
            # Сохраняем ML данные
            self.ml_system.save_data()
            
            # Сохраняем улучшенные метрики
            enhanced_data_dir = Path("enhanced_learning_data")
            enhanced_data_dir.mkdir(exist_ok=True)
            
            import json
            
            # Prediction history
            with open(enhanced_data_dir / "prediction_history.json", 'w') as f:
                json.dump(self.prediction_history, f, indent=2, default=str)
            
            # Enhanced metrics
            with open(enhanced_data_dir / "enhanced_metrics.json", 'w') as f:
                json.dump(self.enhanced_metrics, f, indent=2, default=str)
            
            logger.info("💾 [ENHANCED_SAVE] All enhanced learning data saved")
            
        except Exception as e:
            logger.error(f"❌ [ENHANCED_SAVE] Error saving data: {e}")
    
    def get_current_adaptive_params(self):
        """Backward compatibility method for existing code"""
        try:
            return {
                'confidence_threshold': 0.5,  # Default values
                'position_size_multiplier': 1.0,
                'is_ab_testing': False,
                'ab_variant': 'control'
            }
        except Exception as e:
            logger.warning(f"🔧 [COMPAT] get_current_adaptive_params error: {e}")
            return {}
    
    @property
    def advanced_ai(self):
        """Backward compatibility property for advanced_ai access"""
        return self.ml_system
    
    @property  
    def learning_visualizer(self):
        """Backward compatibility property for learning_visualizer access"""
        return None  # Placeholder for now
    
    async def get_advanced_ai_recommendations(self, market_data: Dict = None):
        """Backward compatibility method for advanced AI recommendations"""
        try:
            if market_data:
                return await self.ml_system.get_intelligent_recommendations(
                    current_market=self.current_market_context or {},
                    recent_performance=self._get_recent_performance_metrics()
                )
            return {
                'confidence': 0.5,
                'recommendations': [
                    {'action': 'maintain_current_strategy', 'confidence': 0.7}
                ]
            }
        except Exception as e:
            logger.warning(f"🔧 [COMPAT] get_advanced_ai_recommendations error: {e}")
            return {'confidence': 0.0, 'recommendations': []}
    
    def _cold_start_decision_logic(self, signal_strength: float, ml_confidence: float, 
                                 expected_pnl: float, win_probability: float, 
                                 risk_score: float) -> bool:
        """🧠 COLD START STRATEGY - Решает проблему 'курицы и яйца'"""
        
        try:
            # Считаем количество обученных образцов
            total_samples = 0
            if hasattr(self.ml_system, 'models'):
                total_samples = sum(getattr(model, 'samples_seen', 0) 
                                  for model in self.ml_system.models.values())
            
            # 🎯 EXPLORATION PHASE - Первые 50 сделок
            if total_samples < 50:
                logger.info(f"🧠 [COLD_START] Exploration mode: {total_samples}/50 samples")
                
                # В начале торгуем на основе ТОЛЬКО сигналов IMBA (не ML)
                # Но с повышенным порогом для безопасности
                exploration_threshold = 1.4  # Выше обычного 1.2
                
                if signal_strength >= exploration_threshold:
                    logger.info(f"🚀 [EXPLORATION] TRADE: Signal {signal_strength:.2f} >= {exploration_threshold}")
                    return True
                else:
                    logger.info(f"🚫 [EXPLORATION] SKIP: Signal {signal_strength:.2f} < {exploration_threshold}")
                    return False
            
            # 🧠 LEARNING PHASE - 50-200 сделок (постепенно добавляем ML)
            elif total_samples < 200:
                learning_progress = (total_samples - 50) / 150  # 0.0 to 1.0
                logger.info(f"🎓 [LEARNING] Learning mode: {total_samples}/200 samples, progress: {learning_progress:.1%}")
                
                # Постепенно снижаем порог сигнала и добавляем ML
                adaptive_threshold = 1.4 - (learning_progress * 0.3)  # 1.4 → 1.1
                ml_weight = learning_progress * 0.3  # 0 → 0.3
                
                # Комбинированная логика
                signal_ok = signal_strength >= adaptive_threshold
                ml_suggests = (ml_confidence * ml_weight + 
                             win_probability * ml_weight) > (ml_weight * 0.6)
                
                should_trade = signal_ok and (ml_weight == 0 or ml_suggests)
                
                logger.info(f"🎓 [LEARNING] Signal: {signal_strength:.2f}>={adaptive_threshold:.2f}? {signal_ok}, "
                           f"ML weight: {ml_weight:.2f}, Decision: {'TRADE' if should_trade else 'SKIP'}")
                
                return should_trade
            
            # 🎯 FULL ML PHASE - После 200 сделок (полный ML)
            else:
                logger.debug(f"🧠 [FULL_ML] Full ML mode: {total_samples} samples")
                
                # Полная ML логика с learned thresholds
                return (
                    signal_strength >= 1.2 and  # Базовый порог IMBA
                    ml_confidence > 0.4 and     # ML должна быть уверена
                    expected_pnl > 0.1 and      # Ожидаемая прибыль
                    win_probability > 0.55 and  # Вероятность выигрыша
                    risk_score < 0.7             # Приемлемый риск
                )
                
        except Exception as e:
            logger.error(f"❌ [COLD_START] Error in decision logic: {e}")
            # Fallback - торгуем только на сильных сигналах
            return signal_strength >= 1.5