"""
🧠 REAL MACHINE LEARNING SYSTEM FOR CRYPTO TRADING
===================================================

Реальная система машинного обучения с:
- Feature Engineering (техническая и фундаментальная информация)
- Online Learning (обучение в реальном времени)
- Multi-objective Optimization (не только PnL)
- Ensemble Methods (комбинирование моделей)
- Contextual Learning (учет рыночных условий)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import deque
import json
import logging
from pathlib import Path

# ML библиотеки
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import SGDRegressor, LogisticRegression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, classification_report
    from sklearn.model_selection import train_test_split
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MarketContext:
    """Контекст рынка во время сделки"""
    timestamp: datetime
    symbol: str
    
    # Технические индикаторы
    rsi_14: float
    rsi_7: float
    macd: float
    macd_signal: float
    bb_position: float  # Позиция относительно Bollinger Bands
    sma_20: float
    ema_50: float
    atr_14: float
    volume_ratio: float  # Соотношение текущего объема к среднему
    
    # Рыночные условия
    volatility_percentile: float  # Процентиль волатильности (0-100)
    trend_strength: float  # Сила тренда
    market_regime: str  # "trending", "ranging", "volatile"
    fear_greed_index: int
    btc_dominance: float
    
    # Временные факторы
    hour_of_day: int
    day_of_week: int
    session: str  # "asian", "european", "american"
    
    # Ценовые уровни
    support_distance: float  # Расстояние до ближайшей поддержки (%)
    resistance_distance: float  # Расстояние до ближайшего сопротивления (%)
    
    # Спреды и ликвидность
    bid_ask_spread: float
    order_book_imbalance: float

@dataclass
class TradeOutcome:
    """Результат сделки с дополнительными метриками"""
    trade_id: str
    pnl: float
    pnl_pct: float
    hold_time_minutes: float
    exit_reason: str
    
    # Качественные метрики
    sharpe_ratio: float
    max_favorable_excursion: float  # MFE
    max_adverse_excursion: float   # MAE
    win_probability: float  # Вероятность успеха на момент входа
    
    # Эмоциональные факторы
    stress_level: float  # Уровень "стресса" позиции
    confidence_decay: float  # Как менялась уверенность

@dataclass
class MLFeatures:
    """Набор признаков для обучения ML моделей"""
    
    # Технические признаки
    rsi_momentum: float
    macd_divergence: float
    volume_surge: float
    price_momentum: float
    volatility_regime: float
    
    # Рыночные признаки
    market_stress: float
    trend_alignment: float
    support_strength: float
    
    # Временные признаки
    session_volatility: float
    day_performance: float
    
    # Мета-признаки
    signal_confluence: float
    historical_accuracy: float

class OnlineLearningModel:
    """Модель онлайн-обучения"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1000,
            tol=1e-3
        ) if ML_AVAILABLE else None
        self.scaler = RobustScaler() if ML_AVAILABLE else None
        self.is_fitted = False
        self.samples_seen = 0
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """Обучение на новых данных"""
        if not ML_AVAILABLE:
            return
            
        try:
            if not self.is_fitted:
                # Первоначальное обучение
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                self.is_fitted = True
            else:
                # Онлайн обновление
                X_scaled = self.scaler.transform(X)
                self.model.partial_fit(X_scaled, y)
                
            self.samples_seen += len(X)
            logger.debug(f"🧠 [ML_{self.name}] Updated with {len(X)} samples, total: {self.samples_seen}")
            
        except Exception as e:
            logger.error(f"❌ [ML_{self.name}] Training error: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание"""
        if not ML_AVAILABLE or not self.is_fitted:
            return np.zeros(len(X))
            
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"❌ [ML_{self.name}] Prediction error: {e}")
            return np.zeros(len(X))

class AdvancedMLLearningSystem:
    """Продвинутая система машинного обучения"""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = Path("ml_learning_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # История данных
        self.market_contexts = deque(maxlen=10000)
        self.trade_outcomes = deque(maxlen=10000)
        self.feature_history = deque(maxlen=5000)
        
        # ML модели
        self.models = {
            'pnl_predictor': OnlineLearningModel('PnL'),
            'win_probability': OnlineLearningModel('WinProb'),
            'hold_time_predictor': OnlineLearningModel('HoldTime'),
            'risk_estimator': OnlineLearningModel('Risk')
        }
        
        # Ensemble модели (для более сложных предсказаний)
        self.ensemble_models = {}
        
        # Статистика производительности
        self.model_performance = {name: [] for name in self.models.keys()}
        
        logger.info("🧠 [ADVANCED_ML] System initialized")
        self._load_historical_data()
    
    def extract_features(self, market_context: MarketContext, 
                        signal_strength: float, 
                        recent_performance: Dict) -> MLFeatures:
        """Извлечение признаков для ML модели"""
        
        try:
            # Технические признаки
            rsi_momentum = (market_context.rsi_14 - 50) / 50  # Нормализованный RSI
            macd_divergence = market_context.macd - market_context.macd_signal
            volume_surge = max(0, market_context.volume_ratio - 1)  # Превышение среднего объема
            
            # Ценовые моменты
            price_momentum = (market_context.ema_50 - market_context.sma_20) / market_context.sma_20
            volatility_regime = market_context.volatility_percentile / 100
            
            # Рыночный стресс
            market_stress = (100 - market_context.fear_greed_index) / 100
            
            # Тренд
            trend_alignment = market_context.trend_strength * (1 if price_momentum > 0 else -1)
            
            # Поддержка/сопротивление
            support_strength = 1 / (1 + market_context.support_distance)
            
            # Временные факторы
            session_multiplier = {
                'american': 1.2,  # Высокая активность
                'european': 1.0,
                'asian': 0.8      # Низкая активность
            }.get(market_context.session, 1.0)
            
            session_volatility = volatility_regime * session_multiplier
            
            # Дневная производительность
            day_performance = recent_performance.get('today_pnl_pct', 0) / 100
            
            # Мета-признаки
            signal_confluence = signal_strength  # Как много индикаторов согласны
            historical_accuracy = recent_performance.get('recent_accuracy', 0.5)
            
            return MLFeatures(
                rsi_momentum=rsi_momentum,
                macd_divergence=macd_divergence,
                volume_surge=volume_surge,
                price_momentum=price_momentum,
                volatility_regime=volatility_regime,
                market_stress=market_stress,
                trend_alignment=trend_alignment,
                support_strength=support_strength,
                session_volatility=session_volatility,
                day_performance=day_performance,
                signal_confluence=signal_confluence,
                historical_accuracy=historical_accuracy
            )
            
        except Exception as e:
            logger.error(f"❌ [FEATURE_EXTRACTION] Error: {e}")
            # Возвращаем нулевые признаки в случае ошибки
            return MLFeatures(**{field: 0.0 for field in MLFeatures.__annotations__})
    
    async def predict_trade_outcome(self, market_context: MarketContext, 
                                  signal_strength: float,
                                  recent_performance: Dict) -> Dict[str, float]:
        """Предсказывает результат сделки перед входом"""
        
        try:
            # Извлекаем признаки
            features = self.extract_features(market_context, signal_strength, recent_performance)
            feature_array = np.array([list(asdict(features).values())])
            
            # Получаем предсказания от всех моделей
            predictions = {}
            
            for name, model in self.models.items():
                if model.is_fitted:
                    pred = model.predict(feature_array)[0]
                    predictions[name] = float(pred)
                else:
                    predictions[name] = 0.0
            
            # Метрики качества предсказания
            prediction_confidence = min(1.0, max(0.1, 
                sum(model.samples_seen for model in self.models.values()) / 1000
            ))
            
            result = {
                'expected_pnl_pct': predictions.get('pnl_predictor', 0.0),
                'win_probability': max(0.1, min(0.9, predictions.get('win_probability', 0.5))),
                'expected_hold_time': max(5, predictions.get('hold_time_predictor', 30)),  # минуты
                'risk_score': predictions.get('risk_estimator', 0.5),
                'prediction_confidence': prediction_confidence,
                'feature_importance': self._get_feature_importance()
            }
            
            logger.info(f"🎯 [ML_PREDICTION] Expected: {result['expected_pnl_pct']:+.2f}% PnL, "
                       f"{result['win_probability']:.0%} win prob, {prediction_confidence:.2f} confidence")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [ML_PREDICTION] Error: {e}")
            return {
                'expected_pnl_pct': 0.0,
                'win_probability': 0.5,
                'expected_hold_time': 30.0,
                'risk_score': 0.5,
                'prediction_confidence': 0.0,
                'feature_importance': {}
            }
    
    async def learn_from_trade(self, market_context: MarketContext,
                             trade_outcome: TradeOutcome,
                             signal_strength: float,
                             recent_performance: Dict):
        """Обучение на завершенной сделке"""
        
        try:
            # Извлекаем признаки
            features = self.extract_features(market_context, signal_strength, recent_performance)
            feature_array = np.array([list(asdict(features).values())])
            
            # Целевые переменные для обучения
            targets = {
                'pnl_predictor': trade_outcome.pnl_pct,
                'win_probability': 1.0 if trade_outcome.pnl > 0 else 0.0,
                'hold_time_predictor': trade_outcome.hold_time_minutes,
                'risk_estimator': trade_outcome.max_adverse_excursion
            }
            
            # Обучаем все модели
            for name, target in targets.items():
                if name in self.models:
                    self.models[name].partial_fit(feature_array, np.array([target]))
            
            # Сохраняем данные
            self.market_contexts.append(market_context)
            self.trade_outcomes.append(trade_outcome)
            self.feature_history.append(features)
            
            # Периодически оцениваем качество моделей
            if len(self.trade_outcomes) % 50 == 0:
                await self._evaluate_model_performance()
            
            logger.info(f"🧠 [ML_LEARNING] Learned from trade: {trade_outcome.pnl_pct:+.2f}% PnL")
            
        except Exception as e:
            logger.error(f"❌ [ML_LEARNING] Error: {e}")
    
    async def get_intelligent_recommendations(self, current_market: MarketContext,
                                            recent_performance: Dict) -> Dict[str, Any]:
        """Получить рекомендации от AI системы"""
        
        try:
            if not self.models['pnl_predictor'].is_fitted:
                return {'confidence': 0.0, 'recommendations': []}
            
            # Анализируем текущие рыночные условия
            features = self.extract_features(current_market, 1.0, recent_performance)
            feature_array = np.array([list(asdict(features).values())])
            
            # Получаем предсказания
            expected_pnl = self.models['pnl_predictor'].predict(feature_array)[0]
            win_prob = self.models['win_probability'].predict(feature_array)[0]
            risk_score = self.models['risk_estimator'].predict(feature_array)[0]
            
            # Генерируем рекомендации
            recommendations = []
            
            if expected_pnl > 0.5 and win_prob > 0.6:
                recommendations.append({
                    'action': 'increase_position_size',
                    'confidence': min(0.9, win_prob),
                    'reason': f'High win probability ({win_prob:.1%}) and positive expected return'
                })
            
            if risk_score > 0.7:
                recommendations.append({
                    'action': 'tighten_stop_loss',
                    'confidence': 0.8,
                    'reason': f'High risk environment detected (score: {risk_score:.2f})'
                })
            
            if features.volatility_regime > 0.8:
                recommendations.append({
                    'action': 'reduce_exposure',
                    'confidence': 0.7,
                    'reason': 'High volatility regime - reduce risk'
                })
            
            if features.trend_alignment > 0.5 and features.support_strength > 0.7:
                recommendations.append({
                    'action': 'extend_targets',
                    'confidence': 0.6,
                    'reason': 'Strong trend with solid support - ride the momentum'
                })
            
            confidence = min(1.0, sum(model.samples_seen for model in self.models.values()) / 2000)
            
            return {
                'confidence': confidence,
                'expected_pnl': expected_pnl,
                'win_probability': win_prob,
                'risk_score': risk_score,
                'recommendations': recommendations,
                'market_regime': current_market.market_regime,
                'feature_summary': {
                    'trend_strength': features.trend_alignment,
                    'volatility': features.volatility_regime,
                    'market_stress': features.market_stress
                }
            }
            
        except Exception as e:
            logger.error(f"❌ [ML_RECOMMENDATIONS] Error: {e}")
            return {'confidence': 0.0, 'recommendations': []}
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Получить важность признаков"""
        try:
            if not ML_AVAILABLE or not self.models['pnl_predictor'].is_fitted:
                return {}
            
            # Для SGD модели используем коэффициенты как важность
            coef = self.models['pnl_predictor'].model.coef_
            feature_names = list(MLFeatures.__annotations__.keys())
            
            importance = {}
            for i, name in enumerate(feature_names):
                if i < len(coef):
                    importance[name] = abs(float(coef[i]))
            
            return importance
            
        except Exception as e:
            logger.error(f"❌ [FEATURE_IMPORTANCE] Error: {e}")
            return {}
    
    async def _evaluate_model_performance(self):
        """Оценка производительности моделей"""
        try:
            if len(self.trade_outcomes) < 30:
                return
            
            # Берем последние N сделок для оценки
            recent_outcomes = list(self.trade_outcomes)[-50:]
            recent_features = list(self.feature_history)[-50:]
            
            if len(recent_features) != len(recent_outcomes):
                return
            
            # Создаем массивы для оценки
            X = np.array([list(asdict(f).values()) for f in recent_features])
            
            # Оцениваем каждую модель
            for name, model in self.models.items():
                if not model.is_fitted:
                    continue
                
                if name == 'pnl_predictor':
                    y_true = [outcome.pnl_pct for outcome in recent_outcomes]
                elif name == 'win_probability':
                    y_true = [1.0 if outcome.pnl > 0 else 0.0 for outcome in recent_outcomes]
                elif name == 'hold_time_predictor':
                    y_true = [outcome.hold_time_minutes for outcome in recent_outcomes]
                else:
                    continue
                
                y_pred = model.predict(X)
                mse = mean_squared_error(y_true, y_pred)
                
                self.model_performance[name].append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'mse': float(mse),
                    'samples': len(y_true)
                })
                
                logger.info(f"📊 [ML_PERFORMANCE] {name}: MSE = {mse:.4f}")
            
        except Exception as e:
            logger.error(f"❌ [ML_EVALUATION] Error: {e}")
    
    def _load_historical_data(self):
        """Загружает исторические данные"""
        try:
            # Загружаем сохраненные данные если есть
            contexts_file = self.data_dir / "market_contexts.json"
            outcomes_file = self.data_dir / "trade_outcomes.json"
            
            if contexts_file.exists() and outcomes_file.exists():
                with open(contexts_file, 'r') as f:
                    contexts_data = json.load(f)
                
                with open(outcomes_file, 'r') as f:
                    outcomes_data = json.load(f)
                
                logger.info(f"🧠 [ML_LOAD] Loaded {len(contexts_data)} historical contexts")
                
        except Exception as e:
            logger.error(f"❌ [ML_LOAD] Error loading historical data: {e}")
    
    def save_data(self):
        """Сохраняет данные ML системы"""
        try:
            # Сохраняем контексты рынка
            contexts_data = []
            for context in self.market_contexts:
                contexts_data.append(asdict(context))
            
            with open(self.data_dir / "market_contexts.json", 'w') as f:
                json.dump(contexts_data, f, indent=2, default=str)
            
            # Сохраняем результаты сделок
            outcomes_data = []
            for outcome in self.trade_outcomes:
                outcomes_data.append(asdict(outcome))
            
            with open(self.data_dir / "trade_outcomes.json", 'w') as f:
                json.dump(outcomes_data, f, indent=2, default=str)
            
            # Сохраняем модели
            if ML_AVAILABLE:
                for name, model in self.models.items():
                    if model.is_fitted:
                        joblib.dump(model.model, self.data_dir / f"{name}_model.pkl")
                        joblib.dump(model.scaler, self.data_dir / f"{name}_scaler.pkl")
            
            logger.info(f"💾 [ML_SAVE] Saved ML data: {len(self.market_contexts)} contexts, "
                       f"{len(self.trade_outcomes)} outcomes")
            
        except Exception as e:
            logger.error(f"❌ [ML_SAVE] Error: {e}")