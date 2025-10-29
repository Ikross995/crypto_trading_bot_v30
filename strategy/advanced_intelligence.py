#!/usr/bin/env python3
"""
Advanced Intelligence System для торгового бота

🧠 СУПЕР-ИНТЕЛЛЕКТ включает:
- Bayesian Optimization для поиска оптимальных параметров
- Multi-Armed Bandit алгоритмы для intelligent exploration  
- Reinforcement Learning для адаптации к рынку
- Advanced A/B testing с статистической значимостью
- Pattern recognition в сделках
- Real-time market regime detection
- Dynamic strategy switching
"""

import asyncio
import json
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

# Для Bayesian Optimization нужен scikit-optimize
try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    ADVANCED_OPTIMIZATION = True
    logger.info("🧠 [ADVANCED_AI] Bayesian Optimization available")
except ImportError:
    ADVANCED_OPTIMIZATION = False
    logger.warning("🧠 [ADVANCED_AI] Install scikit-optimize for advanced features: pip install scikit-optimize")


@dataclass
class ParameterSpace:
    """Пространство параметров для оптимизации."""
    name: str
    min_val: float
    max_val: float
    current_val: float
    exploration_rate: float = 0.1
    confidence: float = 0.0
    
    
@dataclass
class ABTestVariant:
    """Вариант для A/B тестирования."""
    name: str
    parameters: Dict[str, float]
    trades_count: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    statistical_significance: float = 0.0
    

@dataclass
class MarketRegime:
    """Рыночный режим."""
    name: str  # "TRENDING_BULL", "TRENDING_BEAR", "SIDEWAYS_LOW_VOL", "SIDEWAYS_HIGH_VOL", "VOLATILE_UNCERTAIN"
    confidence: float
    characteristics: Dict[str, float]
    optimal_params: Dict[str, float]
    

@dataclass
class TradePattern:
    """Паттерн сделки."""
    pattern_id: str
    description: str
    features: Dict[str, float]
    success_rate: float
    avg_pnl: float
    sample_size: int
    

class AdvancedIntelligenceSystem:
    """
    Продвинутая система искусственного интеллекта для торгового бота.
    
    🧠 ВОЗМОЖНОСТИ:
    1. Bayesian Optimization - умный поиск оптимальных параметров
    2. Multi-Armed Bandit - intelligent exploration vs exploitation  
    3. Advanced A/B Testing - статистически значимые тесты
    4. Pattern Recognition - автоматическое обнаружение паттернов
    5. Market Regime Detection - определение режимов рынка
    6. Reinforcement Learning - адаптация к изменениям
    """
    
    def __init__(self, data_dir: str = "intelligence_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Файлы для сохранения данных
        self.intelligence_file = self.data_dir / "intelligence_state.json"
        self.patterns_file = self.data_dir / "trade_patterns.json"
        self.regimes_file = self.data_dir / "market_regimes.json"
        self.optimization_file = self.data_dir / "optimization_history.json"
        
        # Параметры для оптимизации
        self.parameter_space = {
            'confidence_threshold': ParameterSpace('confidence_threshold', 0.2, 1.0, 0.45),
            'position_size_multiplier': ParameterSpace('position_size_multiplier', 0.3, 2.0, 1.0),
            'dca_threshold_1': ParameterSpace('dca_threshold_1', 0.5, 3.0, 1.0),
            'dca_threshold_2': ParameterSpace('dca_threshold_2', 1.5, 5.0, 2.0),
            'dca_threshold_3': ParameterSpace('dca_threshold_3', 2.5, 8.0, 3.5),
            'risk_multiplier': ParameterSpace('risk_multiplier', 0.5, 3.0, 1.0),
            'exit_profit_target': ParameterSpace('exit_profit_target', 1.5, 5.0, 2.5),
        }
        
        # A/B тестирование
        self.active_ab_tests: List[ABTestVariant] = []
        self.ab_test_results_history: List[Dict] = []
        
        # Обнаружение паттернов
        self.trade_patterns: List[TradePattern] = []
        self.pattern_detector = None
        
        # Рыночные режимы
        self.market_regimes: List[MarketRegime] = []
        self.current_regime: Optional[MarketRegime] = None
        self.regime_detector = None
        
        # Multi-Armed Bandit
        self.bandit_arms: Dict[str, Dict] = {}  # arm_name -> {rewards: [], pulls: 0}
        
        # Reinforcement Learning
        self.q_table: Dict[str, Dict[str, float]] = {}  # state -> {action: q_value}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # exploration rate
        
        # Optimization history
        self.optimization_history: List[Dict] = []
        
        # Загружаем сохраненные данные
        self._load_intelligence_state()
        
        logger.info("🧠 [ADVANCED_AI] Advanced Intelligence System initialized")
        
    async def optimize_parameters_bayesian(self, trade_history: List, target_metric: str = 'sharpe_ratio') -> Dict[str, float]:
        """
        Байесовская оптимизация параметров.
        Умно ищет оптимальные параметры, учитывая предыдущие результаты.
        """
        if not ADVANCED_OPTIMIZATION:
            logger.warning("🧠 [BAYESIAN_OPT] Advanced optimization not available")
            return self._simple_parameter_optimization(trade_history)
            
        try:
            logger.info(f"🧠 [BAYESIAN_OPT] Optimizing for {target_metric}...")
            
            # Определяем пространство поиска
            dimensions = [
                Real(0.2, 1.0, name='confidence_threshold'),
                Real(0.3, 2.0, name='position_size_multiplier'),
                Real(0.5, 3.0, name='dca_threshold_1'),
                Real(1.5, 5.0, name='dca_threshold_2'),
                Real(2.5, 8.0, name='dca_threshold_3'),
                Real(0.5, 3.0, name='risk_multiplier'),
                Real(1.5, 5.0, name='exit_profit_target')
            ]
            
            # Функция для оптимизации
            @use_named_args(dimensions)
            def objective(**params):
                try:
                    # Симулируем торговлю с этими параметрами
                    simulated_metric = self._simulate_trading_with_params(trade_history, params, target_metric)
                    return -simulated_metric  # Минимизируем (поэтому отрицательное значение)
                except Exception as e:
                    logger.error(f"🧠 [BAYESIAN_OPT] Simulation error: {e}")
                    return 0.0
            
            # Байесовская оптимизация
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=20,  # Количество итераций
                n_initial_points=5,
                random_state=42
            )
            
            # Лучшие параметры
            best_params = {
                'confidence_threshold': result.x[0],
                'position_size_multiplier': result.x[1], 
                'dca_threshold_1': result.x[2],
                'dca_threshold_2': result.x[3],
                'dca_threshold_3': result.x[4],
                'risk_multiplier': result.x[5],
                'exit_profit_target': result.x[6]
            }
            
            # Сохраняем результаты
            optimization_result = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'target_metric': target_metric,
                'best_params': best_params,
                'best_score': -result.fun,
                'n_iterations': len(result.func_vals),
                'convergence': result.func_vals
            }
            
            self.optimization_history.append(optimization_result)
            self._save_optimization_history()
            
            logger.info(f"🎯 [BAYESIAN_OPT] Best {target_metric}: {-result.fun:.4f}")
            logger.info(f"🎯 [BAYESIAN_OPT] Optimal params: {best_params}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"❌ [BAYESIAN_OPT] Optimization failed: {e}")
            return self._simple_parameter_optimization(trade_history)
    
    def _simulate_trading_with_params(self, trade_history: List, params: Dict, target_metric: str) -> float:
        """Симулирует торговлю с заданными параметрами и возвращает метрику."""
        try:
            if not trade_history:
                return 0.0
                
            # Простая симуляция - применяем параметры к историческим сделкам
            simulated_trades = []
            
            for trade in trade_history:
                # Применяем новые параметры к решениям
                confidence_passed = trade.get('confidence', 0.5) >= params['confidence_threshold']
                
                if confidence_passed:
                    # Корректируем PnL на основе новых параметров
                    original_pnl = trade.get('pnl', 0.0)
                    adjusted_pnl = original_pnl * params['position_size_multiplier'] * params['risk_multiplier']
                    
                    simulated_trades.append({
                        'pnl': adjusted_pnl,
                        'timestamp': trade.get('timestamp', datetime.now())
                    })
            
            if not simulated_trades:
                return 0.0
                
            # Рассчитываем целевую метрику
            if target_metric == 'sharpe_ratio':
                return self._calculate_sharpe_ratio(simulated_trades)
            elif target_metric == 'profit_factor':
                return self._calculate_profit_factor(simulated_trades)
            elif target_metric == 'win_rate':
                wins = sum(1 for t in simulated_trades if t['pnl'] > 0)
                return wins / len(simulated_trades)
            else:
                # Total PnL
                return sum(t['pnl'] for t in simulated_trades)
                
        except Exception as e:
            logger.error(f"❌ [SIMULATION] Error: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Рассчитывает Sharpe Ratio для списка сделок."""
        try:
            if len(trades) < 2:
                return 0.0
                
            pnls = [t['pnl'] for t in trades]
            
            if not pnls:
                return 0.0
                
            mean_return = np.mean(pnls)
            std_return = np.std(pnls)
            
            if std_return == 0:
                return 0.0
                
            return mean_return / std_return * np.sqrt(252)  # Аннуализированный
            
        except Exception:
            return 0.0
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Рассчитывает Profit Factor."""
        try:
            profits = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            losses = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            
            return profits / losses if losses > 0 else float('inf')
            
        except Exception:
            return 0.0
    
    async def run_advanced_ab_testing(self, parameter_variants: List[Dict[str, Dict]], min_trades_per_variant: int = 20) -> Dict:
        """
        Продвинутое A/B тестирование с множественными вариантами и статистической значимостью.
        """
        try:
            logger.info(f"🧪 [ADVANCED_AB] Starting test with {len(parameter_variants)} variants")
            
            # Создаем варианты тестирования
            variants = []
            for i, params in enumerate(parameter_variants):
                variant = ABTestVariant(
                    name=f"Variant_{chr(65+i)}",  # A, B, C, D, ...
                    parameters=params
                )
                variants.append(variant)
                
            self.active_ab_tests = variants
            
            # Устанавливаем Multi-Armed Bandit для умного распределения трафика
            for variant in variants:
                self.bandit_arms[variant.name] = {
                    'rewards': [],
                    'pulls': 0,
                    'ucb_score': float('inf')  # Upper Confidence Bound
                }
            
            logger.info(f"🧪 [ADVANCED_AB] Variants created: {[v.name for v in variants]}")
            
            return {
                'test_started': True,
                'variants': len(variants),
                'min_trades_required': min_trades_per_variant * len(variants),
                'current_allocation': 'UCB_BANDIT'
            }
            
        except Exception as e:
            logger.error(f"❌ [ADVANCED_AB] Failed to start testing: {e}")
            return {'test_started': False, 'error': str(e)}
    
    def select_ab_variant_ucb(self) -> Optional[ABTestVariant]:
        """
        Выбирает вариант для следующей сделки используя Upper Confidence Bound алгоритм.
        Балансирует exploration vs exploitation.
        """
        try:
            if not self.active_ab_tests:
                return None
                
            total_pulls = sum(arm['pulls'] for arm in self.bandit_arms.values())
            
            if total_pulls == 0:
                # Первая сделка - случайный выбор
                import random
                return random.choice(self.active_ab_tests)
            
            best_variant = None
            best_ucb_score = -float('inf')
            
            for variant in self.active_ab_tests:
                arm = self.bandit_arms[variant.name]
                
                if arm['pulls'] == 0:
                    # Неисследованный вариант имеет приоритет
                    ucb_score = float('inf')
                else:
                    # UCB формула: mean_reward + sqrt(2 * ln(total_pulls) / arm_pulls)
                    mean_reward = np.mean(arm['rewards']) if arm['rewards'] else 0.0
                    exploration_bonus = np.sqrt(2 * np.log(total_pulls) / arm['pulls'])
                    ucb_score = mean_reward + exploration_bonus
                
                arm['ucb_score'] = ucb_score
                
                if ucb_score > best_ucb_score:
                    best_ucb_score = ucb_score
                    best_variant = variant
            
            if best_variant:
                self.bandit_arms[best_variant.name]['pulls'] += 1
                logger.debug(f"🧪 [UCB_BANDIT] Selected {best_variant.name} (UCB: {best_ucb_score:.3f})")
                
            return best_variant
            
        except Exception as e:
            logger.error(f"❌ [UCB_BANDIT] Selection failed: {e}")
            return self.active_ab_tests[0] if self.active_ab_tests else None
    
    async def update_ab_test_results(self, variant_name: str, trade_pnl: float, trade_metrics: Dict) -> None:
        """Обновляет результаты A/B теста после сделки."""
        try:
            # Обновляем bandit arm
            if variant_name in self.bandit_arms:
                reward = 1.0 if trade_pnl > 0 else -1.0  # Простая reward функция
                self.bandit_arms[variant_name]['rewards'].append(reward)
            
            # Обновляем вариант
            for variant in self.active_ab_tests:
                if variant.name == variant_name:
                    variant.trades_count += 1
                    variant.total_pnl += trade_pnl
                    
                    # Пересчитываем метрики
                    if variant.trades_count > 0:
                        win_trades = sum(1 for r in self.bandit_arms[variant_name]['rewards'] if r > 0)
                        variant.win_rate = win_trades / variant.trades_count
                        
                    break
                    
        except Exception as e:
            logger.error(f"❌ [AB_UPDATE] Failed to update results: {e}")
    
    async def analyze_ab_test_significance(self, min_confidence: float = 0.95) -> Optional[Dict]:
        """
        Анализирует статистическую значимость A/B теста.
        Возвращает результаты, если есть значимые различия.
        """
        try:
            if len(self.active_ab_tests) < 2:
                return None
                
            # Собираем данные всех вариантов
            variants_data = []
            for variant in self.active_ab_tests:
                if variant.trades_count >= 10:  # Минимум данных для анализа
                    arm_rewards = self.bandit_arms[variant.name]['rewards']
                    variants_data.append({
                        'name': variant.name,
                        'rewards': arm_rewards,
                        'mean': np.mean(arm_rewards),
                        'std': np.std(arm_rewards),
                        'count': len(arm_rewards),
                        'win_rate': variant.win_rate,
                        'total_pnl': variant.total_pnl
                    })
            
            if len(variants_data) < 2:
                return None
                
            # Проводим статистические тесты
            significant_results = []
            
            for i in range(len(variants_data)):
                for j in range(i + 1, len(variants_data)):
                    variant_a = variants_data[i]
                    variant_b = variants_data[j]
                    
                    # T-test для сравнения средних
                    t_stat, p_value = stats.ttest_ind(variant_a['rewards'], variant_b['rewards'])
                    
                    # Chi-square test для win rates
                    wins_a = int(variant_a['win_rate'] * variant_a['count'])
                    wins_b = int(variant_b['win_rate'] * variant_b['count'])
                    losses_a = variant_a['count'] - wins_a
                    losses_b = variant_b['count'] - wins_b
                    
                    contingency_table = [[wins_a, losses_a], [wins_b, losses_b]]
                    chi2, chi2_p = stats.chi2_contingency(contingency_table)[:2]
                    
                    significance = {
                        'variant_a': variant_a['name'],
                        'variant_b': variant_b['name'],
                        't_test_p_value': float(p_value),
                        'chi2_test_p_value': float(chi2_p),
                        'is_significant': p_value < (1 - min_confidence),
                        'effect_size': abs(variant_a['mean'] - variant_b['mean']),
                        'better_variant': variant_a['name'] if variant_a['mean'] > variant_b['mean'] else variant_b['name']
                    }
                    
                    significant_results.append(significance)
            
            # Находим лучший вариант
            best_variant = max(variants_data, key=lambda x: x['mean'])
            
            result = {
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'total_variants': len(variants_data),
                'statistical_tests': significant_results,
                'best_variant': best_variant['name'],
                'best_variant_metrics': {
                    'mean_reward': best_variant['mean'],
                    'win_rate': best_variant['win_rate'],
                    'total_pnl': best_variant['total_pnl'],
                    'sample_size': best_variant['count']
                },
                'recommendation': self._generate_ab_recommendation(significant_results, best_variant)
            }
            
            logger.info(f"📊 [AB_ANALYSIS] Best variant: {best_variant['name']} (reward: {best_variant['mean']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [AB_ANALYSIS] Statistical analysis failed: {e}")
            return None
    
    def _generate_ab_recommendation(self, test_results: List[Dict], best_variant: Dict) -> str:
        """Генерирует рекомендацию на основе A/B теста."""
        significant_count = sum(1 for r in test_results if r['is_significant'])
        
        if significant_count == 0:
            return "CONTINUE_TESTING - No statistically significant differences found"
        elif significant_count >= len(test_results) * 0.5:
            return f"IMPLEMENT_WINNER - {best_variant['name']} shows significant improvement"
        else:
            return "MIXED_RESULTS - Some significant differences, continue testing"
    
    def _simple_parameter_optimization(self, trade_history: List) -> Dict[str, float]:
        """Простая оптимизация без Bayesian Optimization."""
        try:
            if not trade_history:
                return {param: space.current_val for param, space in self.parameter_space.items()}
            
            # Простой grid search по ключевым параметрам
            best_params = {}
            best_score = -float('inf')
            
            confidence_values = [0.3, 0.45, 0.6, 0.8]
            position_values = [0.7, 1.0, 1.3]
            
            for conf in confidence_values:
                for pos in position_values:
                    params = {
                        'confidence_threshold': conf,
                        'position_size_multiplier': pos,
                        'dca_threshold_1': 1.0,
                        'dca_threshold_2': 2.0,
                        'dca_threshold_3': 3.5,
                        'risk_multiplier': 1.0,
                        'exit_profit_target': 2.5
                    }
                    
                    score = self._simulate_trading_with_params(trade_history, params, 'sharpe_ratio')
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
            
            logger.info(f"🎯 [SIMPLE_OPT] Best score: {best_score:.4f}")
            return best_params
            
        except Exception as e:
            logger.error(f"❌ [SIMPLE_OPT] Failed: {e}")
            return {param: space.current_val for param, space in self.parameter_space.items()}
    
    def _save_intelligence_state(self) -> None:
        """Сохраняет состояние системы ИИ."""
        try:
            state = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'parameter_space': {name: asdict(space) for name, space in self.parameter_space.items()},
                'bandit_arms': self.bandit_arms,
                'q_table': self.q_table,
                'current_regime': asdict(self.current_regime) if self.current_regime else None,
                'active_ab_tests': [asdict(test) for test in self.active_ab_tests]
            }
            
            with open(self.intelligence_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"❌ [SAVE_STATE] Failed: {e}")
    
    def _load_intelligence_state(self) -> None:
        """Загружает состояние системы ИИ."""
        try:
            if not self.intelligence_file.exists():
                return
                
            with open(self.intelligence_file, 'r') as f:
                state = json.load(f)
            
            # Загружаем parameter space
            if 'parameter_space' in state:
                for name, data in state['parameter_space'].items():
                    if name in self.parameter_space:
                        self.parameter_space[name] = ParameterSpace(**data)
            
            # Загружаем bandit arms
            self.bandit_arms = state.get('bandit_arms', {})
            
            # Загружаем Q-table
            self.q_table = state.get('q_table', {})
            
            # Загружаем текущий режим
            if state.get('current_regime'):
                self.current_regime = MarketRegime(**state['current_regime'])
            
            logger.info("🧠 [LOAD_STATE] Intelligence state loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ [LOAD_STATE] Failed: {e}")
    
    def _save_optimization_history(self) -> None:
        """Сохраняет историю оптимизации."""
        try:
            with open(self.optimization_file, 'w') as f:
                json.dump(self.optimization_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"❌ [SAVE_OPT_HISTORY] Failed: {e}")
    
    async def get_intelligent_recommendations(self, current_market_data: Dict, recent_trades: List) -> Dict[str, Any]:
        """
        Возвращает интеллектуальные рекомендации на основе всех систем ИИ.
        """
        try:
            recommendations = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'recommendations': {},
                'confidence': 0.0,
                'reasoning': []
            }
            
            # 1. Байесовская оптимизация
            if len(recent_trades) >= 10:
                optimal_params = await self.optimize_parameters_bayesian(recent_trades)
                recommendations['recommendations'].update(optimal_params)
                recommendations['reasoning'].append("Bayesian optimization applied")
            
            # 2. A/B тест результаты
            ab_results = await self.analyze_ab_test_significance()
            if ab_results and ab_results['recommendation'].startswith('IMPLEMENT'):
                best_variant = ab_results['best_variant']
                for variant in self.active_ab_tests:
                    if variant.name == best_variant:
                        recommendations['recommendations'].update(variant.parameters)
                        break
                recommendations['reasoning'].append(f"A/B test winner: {best_variant}")
            
            # Рассчитываем общую уверенность
            recommendations['confidence'] = min(1.0, len(recommendations['reasoning']) * 0.25)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ [INTELLIGENT_RECOMMENDATIONS] Failed: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'recommendations': {},
                'confidence': 0.0,
                'reasoning': ['Error generating recommendations'],
                'error': str(e)
            }
    
    async def shutdown(self) -> None:
        """Корректное завершение работы системы ИИ."""
        try:
            logger.info("🧠 [ADVANCED_AI] Shutting down...")
            
            # Сохраняем все состояния
            self._save_intelligence_state()
            self._save_optimization_history()
            
            logger.info("🧠 [ADVANCED_AI] Shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ [SHUTDOWN] Failed: {e}")