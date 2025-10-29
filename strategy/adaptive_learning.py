#!/usr/bin/env python3
"""
Adaptive Learning System

Система непрерывного обучения и адаптации торгового бота на основе реальной производительности.
Автоматически анализирует ошибки, корректирует параметры и переобучает модели.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import logging

from loguru import logger

# Импортируем продвинутую систему ИИ
try:
    from strategy.advanced_intelligence import AdvancedIntelligenceSystem
    ADVANCED_AI_AVAILABLE = True
    logger.info("🧠 [ADAPTIVE_LEARNING] Advanced Intelligence System available")
except ImportError as e:
    ADVANCED_AI_AVAILABLE = False
    logger.warning(f"🧠 [ADAPTIVE_LEARNING] Advanced AI not available: {e}")

from core.config import Config
from core.constants import OrderSide


@dataclass
class PerformanceMetrics:
    """Метрики производительности торговли."""
    
    # Основные метрики
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L метрики
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Риск метрики
    max_drawdown: float = 0.0
    max_consecutive_losses: int = 0
    current_streak: int = 0
    sharpe_ratio: float = 0.0
    
    # Временные метрики
    avg_hold_time: float = 0.0
    last_updated: datetime = None
    
    # Адаптивные параметры
    confidence_threshold: float = 0.45
    position_size_multiplier: float = 1.0
    dca_enabled: bool = True


@dataclass
class TradeRecord:
    """Запись о торговой сделке."""
    
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    hold_time_seconds: float
    signal_strength: float
    market_conditions: Dict[str, Any]
    was_dca: bool = False
    exit_reason: str = "unknown"  # 'tp', 'sl', 'manual', 'timeout'


class AdaptiveLearningSystem:
    """
    Система адаптивного обучения для торгового бота.
    
    Функции:
    - Отслеживание производительности в реальном времени
    - Анализ паттернов прибыльных/убыточных сделок
    - Автоматическая корректировка параметров
    - A/B тестирование стратегий
    - Переобучение ML моделей
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path("adaptive_learning_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Файлы для хранения данных
        self.trades_file = self.data_dir / "trades_history.json"
        self.metrics_file = self.data_dir / "performance_metrics.json"
        self.adaptations_file = self.data_dir / "adaptations_log.json"
        
        # Текущие метрики и история
        self.current_metrics = PerformanceMetrics()
        self.trades_history: deque = deque(maxlen=1000)  # Последние 1000 сделок
        self.adaptations_log: List[Dict] = []
        
        # Параметры для адаптации
        self.learning_rate = 0.1
        self.min_trades_for_adaptation = 10
        self.adaptation_interval_hours = 6
        self.last_adaptation_time = datetime.now(timezone.utc)
        
        # A/B тестирование
        self.ab_test_active = False
        self.ab_test_variant = "A"  # A или B
        self.ab_test_start_time = None
        self.ab_test_duration_hours = 24
        
        # Продвинутая система ИИ
        self.advanced_ai = None
        if ADVANCED_AI_AVAILABLE:
            try:
                self.advanced_ai = AdvancedIntelligenceSystem()
                logger.info("🧠 [ADVANCED_AI] Integrated into adaptive learning")
            except Exception as e:
                logger.error(f"❌ [ADVANCED_AI] Failed to initialize: {e}")
        
        # Загружаем исторические данные
        self._load_historical_data()
        
        logger.info("🤖 [ADAPTIVE_LEARNING] System initialized")
        logger.info(f"📊 [METRICS] Loaded {len(self.trades_history)} historical trades")
        
    def _load_historical_data(self) -> None:
        """Загружает исторические данные о сделках и метриках."""
        try:
            # Загружаем сделки
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    trades_data = json.load(f)
                    for trade_data in trades_data:
                        trade_data['timestamp'] = datetime.fromisoformat(trade_data['timestamp'])
                        self.trades_history.append(TradeRecord(**trade_data))
                        
            # Загружаем метрики
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    if 'last_updated' in metrics_data:
                        metrics_data['last_updated'] = datetime.fromisoformat(metrics_data['last_updated'])
                    self.current_metrics = PerformanceMetrics(**metrics_data)
                    
            # Загружаем лог адаптаций
            if self.adaptations_file.exists():
                with open(self.adaptations_file, 'r') as f:
                    self.adaptations_log = json.load(f)
                    
        except Exception as e:
            logger.error(f"❌ [ADAPTIVE_LEARNING] Failed to load historical data: {e}")
            
    def _save_data(self) -> None:
        """Сохраняет данные на диск."""
        try:
            # Сохраняем сделки
            trades_data = []
            for trade in self.trades_history:
                trade_dict = asdict(trade)
                trade_dict['timestamp'] = trade.timestamp.isoformat()
                trades_data.append(trade_dict)
                
            with open(self.trades_file, 'w') as f:
                json.dump(trades_data, f, indent=2, default=str)
                
            # Сохраняем метрики
            metrics_dict = asdict(self.current_metrics)
            if self.current_metrics.last_updated:
                metrics_dict['last_updated'] = self.current_metrics.last_updated.isoformat()
                
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2, default=str)
                
            # Сохраняем адаптации
            with open(self.adaptations_file, 'w') as f:
                json.dump(self.adaptations_log, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"❌ [ADAPTIVE_LEARNING] Failed to save data: {e}")
            
    async def record_trade(self, trade: TradeRecord) -> None:
        """Записывает новую сделку и обновляет метрики."""
        try:
            logger.info(f"📝 [TRADE_RECORD] {trade.symbol} {trade.side}: PnL {trade.pnl:+.2f} ({trade.pnl_pct:+.2f}%)")
            
            # Добавляем сделку в историю
            self.trades_history.append(trade)
            
            # Обновляем метрики
            await self._update_metrics()
            
            # Проверяем, нужна ли адаптация
            await self._check_adaptation_needed()
            
            # Сохраняем данные
            self._save_data()
            
        except Exception as e:
            logger.error(f"❌ [TRADE_RECORD] Failed to record trade: {e}")
            
    async def _update_metrics(self) -> None:
        """Обновляет метрики производительности."""
        try:
            if not self.trades_history:
                return
                
            # Базовые подсчеты
            total_trades = len(self.trades_history)
            winning_trades = sum(1 for t in self.trades_history if t.pnl > 0)
            losing_trades = sum(1 for t in self.trades_history if t.pnl < 0)
            
            # Расчет метрик
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_pnl = sum(t.pnl for t in self.trades_history)
            
            # P&L статистика
            wins = [t.pnl for t in self.trades_history if t.pnl > 0]
            losses = [t.pnl for t in self.trades_history if t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Максимальная просадка
            max_drawdown = self._calculate_max_drawdown()
            
            # Sharpe ratio (упрощенный)
            returns = [t.pnl_pct for t in self.trades_history]
            sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
            
            # Среднее время удержания
            avg_hold_time = np.mean([t.hold_time_seconds for t in self.trades_history])
            
            # Обновляем метрики
            self.current_metrics = PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                avg_hold_time=avg_hold_time,
                last_updated=datetime.now(timezone.utc),
                confidence_threshold=self.current_metrics.confidence_threshold,
                position_size_multiplier=self.current_metrics.position_size_multiplier,
                dca_enabled=self.current_metrics.dca_enabled
            )
            
            # Логируем обновленные метрики
            if total_trades % 10 == 0:  # Каждые 10 сделок
                await self._log_performance_summary()
                
        except Exception as e:
            logger.error(f"❌ [METRICS_UPDATE] Failed to update metrics: {e}")
            
    def _calculate_max_drawdown(self) -> float:
        """Рассчитывает максимальную просадку."""
        if not self.trades_history:
            return 0.0
            
        # Кумулятивная кривая P&L
        cumulative_pnl = []
        running_total = 0
        
        for trade in self.trades_history:
            running_total += trade.pnl
            cumulative_pnl.append(running_total)
            
        if not cumulative_pnl:
            return 0.0
            
        # Найти максимальную просадку
        peak = cumulative_pnl[0]
        max_dd = 0.0
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            else:
                drawdown = peak - value
                if drawdown > max_dd:
                    max_dd = drawdown
                    
        return max_dd
        
    async def _check_adaptation_needed(self) -> None:
        """Проверяет, нужна ли адаптация параметров."""
        try:
            # Проверяем условия для адаптации
            time_since_last = (datetime.now(timezone.utc) - self.last_adaptation_time).total_seconds() / 3600
            
            should_adapt = (
                len(self.trades_history) >= self.min_trades_for_adaptation and
                time_since_last >= self.adaptation_interval_hours and
                len(self.trades_history) % 20 == 0  # Каждые 20 сделок
            )
            
            if should_adapt:
                logger.info("🧠 [ADAPTATION] Triggering parameter adaptation...")
                await self._adapt_parameters()
                self.last_adaptation_time = datetime.now(timezone.utc)
                
        except Exception as e:
            logger.error(f"❌ [ADAPTATION_CHECK] Failed to check adaptation: {e}")
            
    async def _adapt_parameters(self) -> None:
        """Адаптирует параметры торговли на основе производительности."""
        try:
            old_params = {
                'confidence_threshold': self.current_metrics.confidence_threshold,
                'position_size_multiplier': self.current_metrics.position_size_multiplier,
                'dca_enabled': self.current_metrics.dca_enabled
            }
            
            # Анализируем последние N сделок для адаптации
            recent_trades = list(self.trades_history)[-50:]  # Последние 50 сделок
            
            if len(recent_trades) < 10:
                return
                
            # 1. Адаптация confidence threshold
            await self._adapt_confidence_threshold(recent_trades)
            
            # 2. Адаптация размера позиций
            await self._adapt_position_sizing(recent_trades)
            
            # 3. Адаптация DCA настроек
            await self._adapt_dca_settings(recent_trades)
            
            # Логируем изменения
            new_params = {
                'confidence_threshold': self.current_metrics.confidence_threshold,
                'position_size_multiplier': self.current_metrics.position_size_multiplier,
                'dca_enabled': self.current_metrics.dca_enabled
            }
            
            adaptation_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'trigger': 'performance_analysis',
                'old_params': old_params,
                'new_params': new_params,
                'performance_metrics': {
                    'win_rate': self.current_metrics.win_rate,
                    'profit_factor': self.current_metrics.profit_factor,
                    'max_drawdown': self.current_metrics.max_drawdown,
                    'total_trades': self.current_metrics.total_trades
                }
            }
            
            self.adaptations_log.append(adaptation_record)
            
            logger.info("🔧 [ADAPTATION] Parameters adapted:")
            for param, old_val in old_params.items():
                new_val = new_params[param]
                if old_val != new_val:
                    logger.info(f"   📊 {param}: {old_val} → {new_val}")
                    
        except Exception as e:
            logger.error(f"❌ [ADAPTATION] Failed to adapt parameters: {e}")
            
    async def _adapt_confidence_threshold(self, recent_trades: List[TradeRecord]) -> None:
        """Адаптирует порог confidence на основе результатов."""
        try:
            # Анализ по уровням confidence
            high_conf_trades = [t for t in recent_trades if t.signal_strength > 1.5]
            low_conf_trades = [t for t in recent_trades if 0.8 <= t.signal_strength <= 1.2]
            
            if high_conf_trades and low_conf_trades:
                high_conf_win_rate = sum(1 for t in high_conf_trades if t.pnl > 0) / len(high_conf_trades)
                low_conf_win_rate = sum(1 for t in low_conf_trades if t.pnl > 0) / len(low_conf_trades)
                
                # Если высокий confidence действительно лучше
                if high_conf_win_rate > low_conf_win_rate + 0.1:  # 10% разница
                    # Повышаем threshold для более селективной торговли
                    new_threshold = min(1.0, self.current_metrics.confidence_threshold + 0.05)
                    self.current_metrics.confidence_threshold = new_threshold
                    logger.info(f"🎯 [CONFIDENCE] Raised threshold: high_conf_wr={high_conf_win_rate:.2%}, low_conf_wr={low_conf_win_rate:.2%}")
                    
                elif low_conf_win_rate > high_conf_win_rate and self.current_metrics.win_rate > 0.55:
                    # Понижаем threshold для большего количества сделок
                    new_threshold = max(0.3, self.current_metrics.confidence_threshold - 0.03)
                    self.current_metrics.confidence_threshold = new_threshold
                    logger.info(f"🎯 [CONFIDENCE] Lowered threshold: more opportunities needed")
                    
        except Exception as e:
            logger.error(f"❌ [CONFIDENCE_ADAPT] Failed to adapt confidence: {e}")
            
    async def _adapt_position_sizing(self, recent_trades: List[TradeRecord]) -> None:
        """Адаптирует размер позиций на основе результатов."""
        try:
            recent_win_rate = sum(1 for t in recent_trades if t.pnl > 0) / len(recent_trades)
            recent_profit_factor = self._calculate_profit_factor(recent_trades)
            recent_max_dd = self._calculate_recent_max_drawdown(recent_trades)
            
            # Корректировка размера позиций
            if recent_win_rate > 0.65 and recent_profit_factor > 1.5:
                # Хорошие результаты - можем увеличить размер
                new_multiplier = min(1.5, self.current_metrics.position_size_multiplier * 1.1)
                self.current_metrics.position_size_multiplier = new_multiplier
                logger.info(f"📈 [POSITION_SIZE] Increased: win_rate={recent_win_rate:.2%}, pf={recent_profit_factor:.2f}")
                
            elif recent_win_rate < 0.45 or recent_max_dd > 100:  # Просадка >$100
                # Плохие результаты - уменьшаем размер
                new_multiplier = max(0.5, self.current_metrics.position_size_multiplier * 0.9)
                self.current_metrics.position_size_multiplier = new_multiplier
                logger.info(f"📉 [POSITION_SIZE] Decreased: win_rate={recent_win_rate:.2%}, max_dd=${recent_max_dd:.0f}")
                
        except Exception as e:
            logger.error(f"❌ [POSITION_SIZE_ADAPT] Failed to adapt position sizing: {e}")
            
    async def _adapt_dca_settings(self, recent_trades: List[TradeRecord]) -> None:
        """Адаптирует настройки DCA на основе результатов."""
        try:
            dca_trades = [t for t in recent_trades if t.was_dca]
            non_dca_trades = [t for t in recent_trades if not t.was_dca]
            
            if dca_trades and non_dca_trades:
                dca_win_rate = sum(1 for t in dca_trades if t.pnl > 0) / len(dca_trades)
                non_dca_win_rate = sum(1 for t in non_dca_trades if t.pnl > 0) / len(non_dca_trades)
                
                # Если DCA показывает плохие результаты
                if dca_win_rate < non_dca_win_rate - 0.15:  # На 15% хуже
                    if self.current_metrics.dca_enabled:
                        logger.warning(f"🚫 [DCA] Disabling DCA: dca_wr={dca_win_rate:.2%} vs non_dca_wr={non_dca_win_rate:.2%}")
                        self.current_metrics.dca_enabled = False
                        
                # Если DCA показывает хорошие результаты, а он выключен
                elif dca_win_rate > non_dca_win_rate + 0.1 and not self.current_metrics.dca_enabled:
                    logger.info(f"✅ [DCA] Enabling DCA: dca_wr={dca_win_rate:.2%} vs non_dca_wr={non_dca_win_rate:.2%}")
                    self.current_metrics.dca_enabled = True
                    
        except Exception as e:
            logger.error(f"❌ [DCA_ADAPT] Failed to adapt DCA settings: {e}")
            
    def _calculate_profit_factor(self, trades: List[TradeRecord]) -> float:
        """Рассчитывает profit factor для списка сделок."""
        if not trades:
            return 0.0
            
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
    def _calculate_recent_max_drawdown(self, trades: List[TradeRecord]) -> float:
        """Рассчитывает максимальную просадку для списка сделок."""
        if not trades:
            return 0.0
            
        cumulative_pnl = []
        running_total = 0
        
        for trade in trades:
            running_total += trade.pnl
            cumulative_pnl.append(running_total)
            
        if not cumulative_pnl:
            return 0.0
            
        peak = cumulative_pnl[0]
        max_dd = 0.0
        
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            else:
                drawdown = peak - value
                if drawdown > max_dd:
                    max_dd = drawdown
                    
        return max_dd
        
    async def _log_performance_summary(self) -> None:
        """Выводит сводку производительности."""
        try:
            logger.info("=" * 60)
            logger.info("📊 ADAPTIVE LEARNING PERFORMANCE SUMMARY")
            logger.info("=" * 60)
            logger.info(f"📈 Total Trades: {self.current_metrics.total_trades}")
            logger.info(f"🎯 Win Rate: {self.current_metrics.win_rate:.1%}")
            logger.info(f"💰 Total P&L: ${self.current_metrics.total_pnl:+.2f}")
            logger.info(f"📊 Profit Factor: {self.current_metrics.profit_factor:.2f}")
            logger.info(f"📉 Max Drawdown: ${self.current_metrics.max_drawdown:.2f}")
            logger.info(f"⚡ Sharpe Ratio: {self.current_metrics.sharpe_ratio:.2f}")
            logger.info("")
            logger.info("🎛️ CURRENT ADAPTIVE PARAMETERS:")
            logger.info(f"   🎯 Confidence Threshold: {self.current_metrics.confidence_threshold:.3f}")
            logger.info(f"   📊 Position Size Multiplier: {self.current_metrics.position_size_multiplier:.2f}x")
            logger.info(f"   🛡️ DCA Enabled: {self.current_metrics.dca_enabled}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ [PERFORMANCE_LOG] Failed to log summary: {e}")
            
    async def start_ab_test(self, variant_a_params: Dict, variant_b_params: Dict) -> None:
        """Запускает A/B тестирование параметров."""
        try:
            import random
            
            self.ab_test_active = True
            self.ab_test_start_time = datetime.now(timezone.utc)
            self.ab_test_variant = "A" if random.random() < 0.5 else "B"
            
            # Применяем параметры выбранного варианта
            params = variant_a_params if self.ab_test_variant == "A" else variant_b_params
            
            for param, value in params.items():
                setattr(self.current_metrics, param, value)
                
            logger.info(f"🧪 [A/B_TEST] Started variant {self.ab_test_variant} for {self.ab_test_duration_hours}h")
            logger.info(f"🧪 [A/B_TEST] Params: {params}")
            
        except Exception as e:
            logger.error(f"❌ [A/B_TEST] Failed to start A/B test: {e}")
            
    async def check_ab_test_completion(self) -> Optional[Dict]:
        """Проверяет завершение A/B теста и возвращает результаты."""
        if not self.ab_test_active or not self.ab_test_start_time:
            return None
            
        time_elapsed = (datetime.now(timezone.utc) - self.ab_test_start_time).total_seconds() / 3600
        
        if time_elapsed >= self.ab_test_duration_hours:
            # A/B тест завершен
            self.ab_test_active = False
            
            results = {
                'variant': self.ab_test_variant,
                'duration_hours': time_elapsed,
                'final_metrics': asdict(self.current_metrics),
                'trades_during_test': len([t for t in self.trades_history 
                                         if t.timestamp >= self.ab_test_start_time])
            }
            
            logger.info(f"🧪 [A/B_TEST] Completed variant {self.ab_test_variant}")
            logger.info(f"🧪 [A/B_TEST] Results: {results['trades_during_test']} trades, "
                       f"WR: {self.current_metrics.win_rate:.1%}")
            
            return results
            
        return None
        
    def get_current_adaptive_params(self) -> Dict[str, Any]:
        """Возвращает текущие адаптивные параметры для применения в боте."""
        return {
            'confidence_threshold': self.current_metrics.confidence_threshold,
            'position_size_multiplier': self.current_metrics.position_size_multiplier,
            'dca_enabled': self.current_metrics.dca_enabled,
            'is_ab_testing': self.ab_test_active,
            'ab_variant': self.ab_test_variant if self.ab_test_active else None
        }
        
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Возвращает текущие метрики производительности."""
        return self.current_metrics
        
    async def emergency_stop_learning(self) -> None:
        """Экстренная остановка обучения при критических потерях."""
        try:
            # Возвращаем консервативные параметры
            self.current_metrics.confidence_threshold = 0.8  # Высокий порог
            self.current_metrics.position_size_multiplier = 0.5  # Маленькие позиции
            self.current_metrics.dca_enabled = False  # Отключаем DCA
            
            # Останавливаем A/B тесты
            self.ab_test_active = False
            
            logger.critical("🚨 [EMERGENCY_STOP] Learning system switched to conservative mode!")
            logger.critical("🚨 [EMERGENCY_STOP] All adaptive features temporarily disabled")
            
            # Логируем экстренную остановку
            emergency_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'trigger': 'emergency_stop',
                'reason': 'critical_losses',
                'conservative_params': {
                    'confidence_threshold': 0.8,
                    'position_size_multiplier': 0.5,
                    'dca_enabled': False
                }
            }
            
            self.adaptations_log.append(emergency_record)
            self._save_data()
            
        except Exception as e:
            logger.error(f"❌ [EMERGENCY_STOP] Failed to execute emergency stop: {e}")
    
    async def get_advanced_ai_recommendations(self, market_data: Dict = None) -> Dict[str, Any]:
        """Получает рекомендации от продвинутой системы ИИ."""
        try:
            if not self.advanced_ai:
                logger.warning("🧠 [ADVANCED_AI] Not available for recommendations")
                return {}
            
            # Подготавливаем данные для анализа
            trade_history_for_ai = []
            for trade in self.trades_history:
                trade_data = {
                    'pnl': trade.pnl,
                    'confidence': trade.signal_strength,
                    'timestamp': trade.timestamp,
                    'was_dca': trade.was_dca,
                    'side': trade.side,
                    'symbol': trade.symbol,
                    'hold_duration_minutes': trade.hold_time_seconds / 60,
                    'market_volatility': trade.market_conditions.get('volatility', 0.02),
                    'volume_ratio': trade.market_conditions.get('volume_ratio', 1.0)
                }
                trade_history_for_ai.append(trade_data)
            
            # Получаем рекомендации
            recommendations = await self.advanced_ai.get_intelligent_recommendations(
                current_market_data=market_data or {},
                recent_trades=trade_history_for_ai
            )
            
            logger.info(f"🧠 [ADVANCED_AI] Generated recommendations with confidence: {recommendations.get('confidence', 0):.2f}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ [ADVANCED_AI] Failed to get recommendations: {e}")
            return {}
    
    async def start_advanced_ab_testing(self, parameter_variants: List[Dict]) -> Dict:
        """Запускает продвинутое A/B тестирование."""
        try:
            if not self.advanced_ai:
                logger.warning("🧠 [ADVANCED_AI] Not available for advanced A/B testing")
                return {}
            
            result = await self.advanced_ai.run_advanced_ab_testing(
                parameter_variants=parameter_variants,
                min_trades_per_variant=15
            )
            
            if result.get('test_started'):
                logger.info(f"🧪 [ADVANCED_AB] Started testing with {result['variants']} variants")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [ADVANCED_AB] Failed to start advanced A/B testing: {e}")
            return {}
    
    async def update_advanced_ab_test(self, trade: TradeRecord) -> None:
        """Обновляет результаты продвинутого A/B тестирования."""
        try:
            if not self.advanced_ai or not self.advanced_ai.active_ab_tests:
                return
            
            # Выбираем вариант для сделки
            selected_variant = self.advanced_ai.select_ab_variant_ucb()
            if selected_variant:
                # Применяем параметры варианта
                for param, value in selected_variant.parameters.items():
                    if hasattr(self.current_metrics, param):
                        setattr(self.current_metrics, param, value)
                
                # Обновляем результаты после сделки
                await self.advanced_ai.update_ab_test_results(
                    variant_name=selected_variant.name,
                    trade_pnl=trade.pnl,
                    trade_metrics={
                        'win_rate': self.current_metrics.win_rate,
                        'profit_factor': self.current_metrics.profit_factor
                    }
                )
                
                logger.debug(f"🧪 [ADVANCED_AB] Updated {selected_variant.name} with PnL: {trade.pnl:+.2f}")
            
        except Exception as e:
            logger.error(f"❌ [ADVANCED_AB] Failed to update A/B test: {e}")
    
    async def check_advanced_ab_significance(self) -> Optional[Dict]:
        """Проверяет статистическую значимость продвинутого A/B теста."""
        try:
            if not self.advanced_ai:
                return None
            
            significance_results = await self.advanced_ai.analyze_ab_test_significance()
            
            if significance_results:
                logger.info(f"📊 [ADVANCED_AB] Analysis: {significance_results['recommendation']}")
                
                # Если найден победитель, применяем его параметры
                if significance_results['recommendation'].startswith('IMPLEMENT'):
                    best_variant = significance_results['best_variant']
                    logger.info(f"🏆 [ADVANCED_AB] Implementing winner: {best_variant}")
                    
                    # Находим параметры победителя
                    for variant in self.advanced_ai.active_ab_tests:
                        if variant.name == best_variant:
                            for param, value in variant.parameters.items():
                                if hasattr(self.current_metrics, param):
                                    setattr(self.current_metrics, param, value)
                            break
                    
                    # Останавливаем тест
                    self.advanced_ai.active_ab_tests = []
                    self.advanced_ai.bandit_arms = {}
            
            return significance_results
            
        except Exception as e:
            logger.error(f"❌ [ADVANCED_AB] Failed to check significance: {e}")
            return None
    
    async def optimize_parameters_with_ai(self, target_metric: str = 'sharpe_ratio') -> Dict[str, float]:
        """Оптимизирует параметры с помощью продвинутого ИИ."""
        try:
            if not self.advanced_ai:
                logger.warning("🧠 [ADVANCED_AI] Not available for optimization")
                return {}
            
            # Подготавливаем данные для оптимизации
            trade_history_for_ai = []
            for trade in self.trades_history:
                trade_data = {
                    'pnl': trade.pnl,
                    'confidence': trade.signal_strength,
                    'timestamp': trade.timestamp,
                    'was_dca': trade.was_dca,
                    'side': trade.side
                }
                trade_history_for_ai.append(trade_data)
            
            # Запускаем Байесовскую оптимизацию
            optimal_params = await self.advanced_ai.optimize_parameters_bayesian(
                trade_history=trade_history_for_ai,
                target_metric=target_metric
            )
            
            if optimal_params:
                logger.info(f"🎯 [AI_OPTIMIZATION] Optimal parameters found for {target_metric}")
                
                # Применяем оптимальные параметры
                for param, value in optimal_params.items():
                    if hasattr(self.current_metrics, param):
                        setattr(self.current_metrics, param, value)
                        logger.info(f"   📊 {param}: {value:.3f}")
            
            return optimal_params
            
        except Exception as e:
            logger.error(f"❌ [AI_OPTIMIZATION] Failed: {e}")
            return {}
    
    async def shutdown_advanced_ai(self) -> None:
        """Корректно завершает работу продвинутой системы ИИ."""
        try:
            if self.advanced_ai:
                await self.advanced_ai.shutdown()
                logger.info("🧠 [ADVANCED_AI] Shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ [ADVANCED_AI] Shutdown error: {e}")